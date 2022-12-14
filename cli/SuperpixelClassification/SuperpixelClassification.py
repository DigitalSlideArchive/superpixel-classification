import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import pickle
import pprint
import random
import re
import tempfile
import time

import girder_client
import h5py
import numpy as np
import tensorflow as tf
from histomicstk.cli.utils import CLIArgumentParser


def getItemsAndAnnotations(gc, folderId, annotationName, missing=False):
    results = []
    for item in gc.listItem(folderId):
        if not item.get('largeImage'):
            continue
        found = False
        for annotrec in gc.get(
                'annotation', parameters=dict(itemId=item['_id'], sort='updated', sortdir=-1)):
            if (annotationName not in annotrec['annotation']['name'] or
                    'Predictions' in annotrec['annotation']['name']):
                continue
            annot = gc.get(f'annotation/{annotrec["_id"]}')
            if ('annotation' not in annot or 'elements' not in annot['annotation'] or
                    not len(annot['annotation']['elements'])):
                continue
            elem = annot['annotation']['elements'][0]
            if elem['type'] != 'pixelmap' or not elem.get('user', {}).get('bbox'):
                continue
            if not missing:
                results.append((item, annotrec, elem))
            found = True
            break
        if not found and missing:
            results.append(item)
    return results


def getCurrentEpoch(itemsAndAnnot):
    epoch = 0
    for _, annot, _ in itemsAndAnnot:
        matches = re.search(r' epoch (\d+)', annot['annotation']['name'], re.IGNORECASE)
        if matches:
            epoch = max(epoch, int(matches.groups()[0]))
    return epoch


def createSuperpixelsForItem(gc, annotationName, item, radius, magnification,
                             annotationFolderId, userId):
    from histomicstk.cli.SuperpixelSegmentation import SuperpixelSegmentation

    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tempdir:
        print('Create superpixels for %s' % item['name'])
        imagePath = os.path.join(tempdir, item['name'])
        gc.downloadFile(item['largeImage']['fileId'], imagePath)
        outImagePath = os.path.join(tempdir, 'superpixel.tiff')
        outAnnotationPath = os.path.join(tempdir, '%s.anot' % annotationName)

        if True:
            import large_image
            ts = large_image.open(imagePath)
            pprint.pprint(ts.getMetadata())

        spopts = argparse.Namespace(
            inputImageFile=imagePath,
            outputImageFile=outImagePath,
            outputAnnotationFile=outAnnotationPath,
            roi=[-1, -1, -1, -1],
            tileSize=4096,
            superpixelSize=radius,
            magnification=magnification,
            overlap=True,
            boundaries=True,
            bounding='Internal',
            slic_zero=True,
            compactness=0.1,
            sigma=1,
            default_category_label='default',
            default_fillColor='rgba(0, 0, 0, 0)',
            default_strokeColor='rgba(0, 0, 0, 1)',
        )
        print(spopts)

        SuperpixelSegmentation.createSuperPixels(spopts)
        outImageFile = gc.uploadFileToFolder(annotationFolderId, outImagePath)
        outImageId = outImageFile['itemId']
        annot = json.loads(open(outAnnotationPath).read())
        annot['name'] = '%s Epoch 0' % annotationName
        annot['elements'][0]['girderId'] = outImageId
        print('Bounding boxes span to',
              max(annot['elements'][0]['user']['bbox'][2::4]),
              max(annot['elements'][0]['user']['bbox'][3::4]))
        with open(outAnnotationPath, 'w') as annotation_file:
            json.dump(annot, annotation_file, indent=2, sort_keys=False)
        count = len(gc.get('annotation', parameters=dict(itemId=item['_id'])))
        gc.uploadFileToItem(
            item['_id'], outAnnotationPath,
            reference=json.dumps({
                'identifier': 'LargeImageAnnotationUpload',
                'itemId': item['_id'],
                'fileId': item['largeImage']['fileId'],
                'userId': userId,
            }))
        # Wait for the upload to complete
        waittime = time.time()
        while time.time() - waittime < 120:
            if len(gc.get('annotation', parameters=dict(itemId=item['_id']))) > count:
                break
            time.sleep(0.1)
        print('Created superpixels')


def createSuperpixels(gc, folderId, annotationName, radius, magnification,
                      annotationFolderId, numWorkers):
    items = getItemsAndAnnotations(gc, folderId, annotationName, True)
    if not len(items):
        return
    print('Create superpixels as needed for %d item(s)' % len(items))
    folder = gc.getFolder(folderId)
    results = {}
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=numWorkers) as executor:
        for item in items:
            futures.append((item, executor.submit(
                createSuperpixelsForItem,
                gc, annotationName, item, radius, magnification,
                annotationFolderId, folder['creatorId'])))
    for item, future in futures:
        results[item['_id']] = future.result()
    return results


def createFeaturesForItem(gc, item, elem, featureFolderId, fileName,
                          patchSize):
    print('Create feature', fileName)
    lastlog = starttime = time.time()
    ds = None
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tempdir:
        filePath = os.path.join(tempdir, fileName)
        with h5py.File(filePath, 'w') as fptr:
            for idx, _ in enumerate(elem['values']):
                bbox = elem['user']['bbox'][idx * 4:idx * 4 + 4]
                # use masked superpixel
                patch = pickle.loads(gc.get(f'item/{item["_id"]}/tiles/region', parameters=dict(
                    left=int(bbox[0]), top=int(bbox[1]),
                    right=int(bbox[2]), bottom=int(bbox[3]),
                    width=patchSize, height=patchSize, fill='#000',
                    encoding='pickle:' + str(pickle.HIGHEST_PROTOCOL)
                ), jsonResp=False).content)
                if patch.shape[2] in (2, 4):
                    patch = patch[:, :, :-1]
                scale = 1
                try:
                    scale = elem['transform']['matrix'][0][0]
                except Exception:
                    pass
                mask = pickle.loads(gc.get(
                    f'item/{elem["girderId"]}/tiles/region', parameters=dict(
                        left=int(bbox[0] / scale), top=int(bbox[1] / scale),
                        right=int(bbox[2] / scale), bottom=int(bbox[3] / scale),
                        width=patchSize, height=patchSize, fill='#000',
                        encoding='pickle:' + str(pickle.HIGHEST_PROTOCOL)
                    ), jsonResp=False).content)
                if mask.shape[2] == 4:
                    mask = mask[:, :, :-1]
                maskvals = [
                    [val % 256, val // 256 % 256, val // 65536 % 256]
                    for val in [idx * 2, idx * 2 + 1]]
                patch = patch.copy()
                patch[(mask != maskvals[0]).any(axis=-1) &
                      (mask != maskvals[1]).any(axis=-1)] = [0, 0, 0]
                # TODO: ensure this is uint8
                if not ds:
                    ds = fptr.create_dataset(
                        'images', (1, ) + patch.shape,
                        maxshape=(None, ) + patch.shape, dtype=patch.dtype, chunks=True)
                else:
                    ds.resize((ds.shape[0] + 1,) + patch.shape)
                ds[ds.shape[0] - 1] = patch
                if time.time() - lastlog > 5:
                    lastlog = time.time()
                    print(ds.shape, len(elem['values']),
                          '%5.3f' % (time.time() - starttime),
                          '%5.3f' % ((len(elem['values']) - idx - 1) / (idx + 1) *
                                     (time.time() - starttime)),
                          item['name'])
            print(ds.shape, len(elem['values']),
                  '%5.3f' % (time.time() - starttime), item['name'])
        return gc.uploadFileToFolder(featureFolderId, filePath)


def createFeatures(gc, folderId, annotationName, featureFolderId, patchSize,
                   numWorkers):
    itemsAndAnnot = getItemsAndAnnotations(gc, folderId, annotationName)
    results = {}
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=numWorkers) as executor:
        for item, _, elem in itemsAndAnnot:
            bbox = elem['user']['bbox']
            hashval = repr(dict(
                itemId=item['_id'], bbox=[int(v) for v in bbox], patchSize=patchSize))
            hashval = hashlib.new('sha256', hashval.encode()).hexdigest()
            fileName = 'feature-%s.h5' % (hashval)
            found = False
            for existing in gc.listItem(featureFolderId, name=fileName):
                results[item['_id']] = next(gc.listFile(existing['_id'], limit=1))
                found = True
                break
            if not found:
                futures.append((item, executor.submit(
                    createFeaturesForItem,
                    gc, item, elem, featureFolderId, fileName, patchSize)))
    for item, future in futures:
        file = future.result()
        try:
            if file and '_id' in file:
                results[item['_id']] = file
        except Exception:
            pass
    print('Found %d item(s) with features' % len(results))
    return results


def trainModelAddItem(gc, record, item, annotrec, elem, feature, randomInput,
                      labelList):
    if not randomInput and not any(v for v in elem['values']):
        return
    print('Adding %s, %s (%s:%r) for training' % (
        item['name'], annotrec['annotation']['name'], annotrec['_id'], annotrec['_version']))
    featurePath = os.path.join(record['tempdir'], feature['name'])
    gc.downloadFile(feature['_id'], featurePath)
    with h5py.File(featurePath, 'r') as ffptr:
        fds = ffptr['images']
        for idx, labelnum in enumerate(elem['values']):
            if labelnum and labelnum < len(elem['categories']):
                labelname = elem['categories'][labelnum]['label']
                if labelname not in record['groups']:
                    record['groups'][labelname] = elem['categories'][labelnum]
            elif randomInput:
                labelnum = random.randint(1, len(labelList))
                labelname = labelList[labelnum - 1]
            else:
                continue
            patch = fds[idx]
            if not record['ds']:
                record['ds'] = record['fptr'].create_dataset(
                    'images', (1, ) + patch.shape,
                    maxshape=(None, ) + patch.shape, dtype=patch.dtype, chunks=True)
            else:
                record['ds'].resize((record['ds'].shape[0] + 1,) + patch.shape)
            record['ds'][record['ds'].shape[0] - 1] = patch
            if labelname not in record['labels']:
                record['labels'].append(labelname)
                record['counts'][labelname] = 0
            labelidx = record['labels'].index(labelname)
            # print(idx, labelnum, labelidx, labelname)
            record['labelvals'].append(labelidx)
            record['counts'][labelname] += 1
            if time.time() - record['lastlog'] > 5:
                record['lastlog'] = time.time()
                print(record['ds'].shape, record['counts'],
                      '%5.3f' % (time.time() - record['starttime']))


def trainModel(gc, folderId, annotationName, features, modelFolderId,
               batchSize, epochs, trainingSplit, randomInput, labelList):
    itemsAndAnnot = getItemsAndAnnotations(gc, folderId, annotationName)
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tempdir:
        trainingPath = os.path.join(tempdir, 'training.h5')
        with h5py.File(trainingPath, 'w') as fptr:
            # collect data
            record = {
                'tempdir': tempdir,
                'ds': None,
                'fptr': fptr,
                'labelvals': [],
                'labels': [],
                'counts': {},
                'labelds': None,
                'groups': {},
                'lastlog': time.time(),
                'starttime': time.time(),
            }
            for item, annotrec, elem in itemsAndAnnot:
                if item['_id'] not in features:
                    continue
                trainModelAddItem(
                    gc, record, item, annotrec, elem,
                    features.get(item['_id']), randomInput, labelList)
            if not record['ds']:
                print('No labelled data')
                return
            record['labelds'] = fptr.create_dataset(
                'labels', (len(record['labelvals']), ), dtype=int)
            record['labelds'] = np.array(record['labelvals'], dtype=int)
            print(record['ds'].shape, record['counts'],
                  '%5.3f' % (time.time() - record['starttime']))
            # generate split
            full_ds = tf.data.Dataset.from_tensor_slices((record['ds'], record['labelds']))
            full_ds = full_ds.shuffle(1000)  # add seed=123 ?
            count = len(full_ds)
            train_size = int(count * trainingSplit)
            train_ds = full_ds.take(train_size).batch(batchSize)
            val_ds = full_ds.skip(train_size).batch(batchSize)
            print(batchSize, train_ds, val_ds)
            # make model
            num_classes = len(record['labels'])
            model = tf.keras.Sequential([
                tf.keras.layers.Rescaling(1./255),
                tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                # tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(num_classes)
            ])
            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs
            )
            modelPath = os.path.join(tempdir, '%s Model Epoch %d.h5' % (
                annotationName, getCurrentEpoch(itemsAndAnnot)))
            model.save(modelPath)
            print(repr(record['labels']), repr(record['groups']))
            with h5py.File(modelPath, 'r+') as mptr:
                mptr.create_dataset('labels', data=np.void(
                    pickle.dumps(record['labels'])))
                mptr.create_dataset('groups', data=np.void(
                    pickle.dumps(record['groups'])))
                mptr.create_dataset('history', data=np.void(
                    pickle.dumps(history)))
        modelFile = gc.uploadFileToFolder(modelFolderId, modelPath)
        print('Saved model')
        return modelFile


def predictLabelsForItem(gc, annotationName, annotationFolderId, tempdir,
                         model, item, annotrec, elem, feature, curEpoch,
                         userId, labels, groups, makeHeatmaps, radius,
                         magnification):
    print('Predicting %s' % (item['name']))
    featurePath = os.path.join(tempdir, feature['name'])
    gc.downloadFile(feature['_id'], featurePath)
    annotrec = annotrec['annotation']
    annotrec['elements'] = [elem]
    with h5py.File(featurePath, 'r') as ffptr:
        # Create predicted annotation
        annot = copy.deepcopy(annotrec)
        annot['elements'][0].pop('id', None)
        annot['name'] = '%s Epoch %d Predictions' % (annotationName, curEpoch)
        annot['elements'][0]['categories'] = [groups[key] for key in labels]
        values = annot['elements'][0]['values']
        conf = annot['elements'][0]['user']['confidence'] = []
        catConf = annot['elements'][0]['user']['categoryConfidence'] = []
        ds = ffptr['images']
        predictions = model.predict(ds)
        # scale to units
        cats = [np.argmax(r) for r in predictions]
        # softmax to scale to 0 to 1
        catWeights = tf.nn.softmax(predictions)
        for eidx, entry in enumerate(predictions):
            values[len(conf)] = int(cats[eidx])
            catConf.append([float(v) for v in entry])
            # This is probably a poor confidence metric, as it is just the
            # probability of the most likely class
            conf.append(float(catWeights[eidx][cats[eidx]]))
        outAnnotationPath = os.path.join(tempdir, '%s.anot' % annot['name'])
        with open(outAnnotationPath, 'w') as annotation_file:
            json.dump(annot, annotation_file, indent=2, sort_keys=False)
        gc.uploadFileToItem(
            item['_id'], outAnnotationPath,
            reference=json.dumps({
                'identifier': 'LargeImageAnnotationUpload',
                'itemId': item['_id'],
                'fileId': item['largeImage']['fileId'],
                'userId': userId,
            }))
        # Upload new user annotation
        newAnnot = annotrec.copy()
        newAnnot['elements'][0].pop('id', None)
        newAnnot['name'] = '%s Epoch %d' % (annotationName, curEpoch + 1)
        outAnnotationPath = os.path.join(tempdir, '%s.anot' % newAnnot['name'])
        with open(outAnnotationPath, 'w') as annotation_file:
            json.dump(newAnnot, annotation_file, indent=2, sort_keys=False)
        gc.uploadFileToItem(
            item['_id'], outAnnotationPath,
            reference=json.dumps({
                'identifier': 'LargeImageAnnotationUpload',
                'itemId': item['_id'],
                'fileId': item['largeImage']['fileId'],
                'userId': userId,
            }))
        if makeHeatmaps:
            makeHeatmapsForItem(
                gc, annotationName, userId, tempdir, radius, item, elem,
                labels, groups, curEpoch, conf, catWeights, catConf)


def makeHeatmapsForItem(gc, annotationName, userId, tempdir, radius, item,
                        elem, labels, groups, curEpoch, conf, catWeights,
                        catConf):
    scale = 1
    try:
        scale = elem['transform']['matrix'][0][0]
    except Exception:
        pass
    heatmaps = []
    bbox = elem['user']['bbox']
    for idx, key in enumerate(labels):
        fillColor = re.search(
            r'(rgba\(\s*[0-9.]+\s*,\s*[0-9.]+\s*,\s*[0-9.]+)(\s*,\s*[0-9.]+\s*\))',
            groups[key]['fillColor'])
        fillColor = (
            fillColor.groups()[0] + ',1)') if fillColor else groups[key]['fillColor']
        heatmaps.append({
            'name': 'Attention Logit Epoch %d' % (curEpoch),
            'description': 'Attention Logit for %s - %s Epoch %d' % (
                annotationName, key, curEpoch),
            'elements': [{
                'type': 'heatmap',
                'group': key,
                'label': {'value': 'Attention Logit %s' % key},
                'radius': radius * scale * 2,
                'scaleWithZoom': True,
                'points': [[
                    (bbox[ci * 4] + bbox[ci * 4 + 2]) * 0.5,
                    (bbox[ci * 4 + 1] + bbox[ci * 4 + 3]) * 0.5,
                    0,
                    float(catConf[ci][idx])] for ci in range(len(catConf))],
                'colorRange': ['rgba(0,0,0,0)', fillColor],
                'rangeValues': [0, 1],
                'normalizeRange': True,
            }]})
        heatmaps.append({
            'name': 'Attention Epoch %d' % (curEpoch),
            'description': 'Attention for %s - %s Epoch %d' % (
                annotationName, key, curEpoch),
            'elements': [{
                'type': 'heatmap',
                'group': key,
                'label': {'value': 'Attention %s' % key},
                'radius': radius * scale * 2,
                'scaleWithZoom': True,
                'points': [[
                    (bbox[ci * 4] + bbox[ci * 4 + 2]) * 0.5,
                    (bbox[ci * 4 + 1] + bbox[ci * 4 + 3]) * 0.5,
                    0,
                    float(catWeights[ci][idx])] for ci in range(len(catWeights))],
                'colorRange': ['rgba(0,0,0,0)', fillColor],
                'rangeValues': [0, 1],
                'normalizeRange': False,
            }]})
    heatmaps.append({
        'name': 'Uncertainty Epoch %d' % (curEpoch),
        'description': 'Uncertainty for %s Epoch %d' % (annotationName, curEpoch),
        'elements': [{
            'type': 'heatmap',
            'group': 'default',
            'label': {'value': 'Uncertainty'},
            'radius': radius * scale * 2,
            'scaleWithZoom': True,
            'points': [[
                (bbox[ci * 4] + bbox[ci * 4 + 2]) * 0.5,
                (bbox[ci * 4 + 1] + bbox[ci * 4 + 3]) * 0.5,
                0,
                1 - conf[ci]] for ci in range(len(conf))],
            'colorRange': [
                'rgba(0,0,0,0)', 'rgba(0,0,255,0.75)',
                'rgba(255,255,0,0.9)', 'rgba(255,0,0,1)'],
            'rangeValues': [0, 0.5, 0.75, 1],
            'normalizeRange': False,
        }]})
    outAnnotationPath = os.path.join(tempdir, '%s.anot' % heatmaps[-1]['name'])
    with open(outAnnotationPath, 'w') as annotation_file:
        json.dump(heatmaps, annotation_file, indent=2, sort_keys=False)
    gc.uploadFileToItem(
        item['_id'], outAnnotationPath,
        reference=json.dumps({
            'identifier': 'LargeImageAnnotationUpload',
            'itemId': item['_id'],
            'fileId': item['largeImage']['fileId'],
            'userId': userId,
        }))


def predictLabels(gc, folderId, annotationName, features, modelFolderId,
                  annotationFolderId, saliencyMaps, radius, magnification):
    itemsAndAnnot = getItemsAndAnnotations(gc, folderId, annotationName)
    curEpoch = getCurrentEpoch(itemsAndAnnot)
    folder = gc.getFolder(folderId)
    userId = folder['creatorId']
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tempdir:
        modelFile = None
        for item in gc.listResource(
                'item', {'folderId': modelFolderId, 'sort': 'updated', 'sortdir': -1}):
            if annotationName in item['name'] and 'model' in item['name'].lower():
                modelFile = next(gc.listFile(item['_id'], limit=1))
                break
        if not modelFile:
            print('No model file found')
            return
        print(modelFile['name'], item)
        modelPath = os.path.join(tempdir, modelFile['name'])
        gc.downloadFile(modelFile['_id'], modelPath)
        model = tf.keras.models.load_model(modelPath)
        with h5py.File(modelPath, 'r+') as mptr:
            labels = pickle.loads(mptr['labels'][()].tobytes())
            groups = pickle.loads(mptr['groups'][()].tobytes())
        for label in labels:
            if label not in groups:
                ll = len(groups)
                rgbtbl = [[255, 0, 0], [0, 191, 0], [0, 0, 255],
                          [255, 255, 0], [255, 0, 255], [0, 255, 255]]
                fac = 1 - 1 / (ll // 6) if ll // 6 else 1
                groups[label] = {
                    'label': label,
                    'strokeColor': 'rgb(%d, %d, %d)' % (
                        int(rgbtbl[ll % 6][0] * fac),
                        int(rgbtbl[ll % 6][1] * fac),
                        int(rgbtbl[ll % 6][2] * fac),
                    ),
                    'fillColor': 'rgba(%d, %d, %d, 0.25)' % (
                        int(rgbtbl[ll % 6][0] * fac),
                        int(rgbtbl[ll % 6][1] * fac),
                        int(rgbtbl[ll % 6][2] * fac),
                    ),
                }
        for item, annotrec, elem in itemsAndAnnot:
            if item['_id'] not in features:
                continue
            predictLabelsForItem(
                gc, annotationName, annotationFolderId, tempdir,
                model, item, annotrec, elem, features.get(item['_id']),
                curEpoch, userId, labels, groups, saliencyMaps, radius,
                magnification)


def main(args):
    print('\n>> CLI Parameters ...\n')
    pprint.pprint(vars(args))

    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.token = args.girderToken

    if args.gensuperpixels:
        createSuperpixels(
            gc, args.images, args.annotationName, args.radius,
            args.magnification, args.annotationDir, args.numWorkers)

    features = createFeatures(
        gc, args.images, args.annotationName, args.features, args.patchSize,
        args.numWorkers)

    if args.train:
        trainModel(
            gc, args.images, args.annotationName, features, args.modeldir,
            args.batchSize, args.epochs, args.split, args.randominput,
            args.labels)

    predictLabels(
        gc, args.images, args.annotationName, features, args.modeldir,
        args.annotationDir, args.heatmaps, args.radius, args.magnification)


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
