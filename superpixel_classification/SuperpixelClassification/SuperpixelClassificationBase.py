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
import tenacity
from numpy.typing import NDArray
from progress_helper import ProgressHelper


def summary_repr(contents, collapseSequences=False):
    """
    Like Python `repr`, returns a string representing the contents.  However, numpy
    arrays are summarized as their shape and unknown types are summarized by their type.

    Parameters
    ----------
    contents :
        Python object
    collapseSequences :
        Set to True to summarize only the first of any list, tuple, or set that has
        length longer than one.  In contrast, a dict will be presented in full.

    Returns
    -------
    A string representation of a summary of the object

    """
    if isinstance(contents, (bool, int, float, str, np.int32, np.int64, np.float32, np.float64)):
        return repr(contents)
    if isinstance(contents, (list, tuple, dict, set)) and len(contents) == 0:
        return repr(type(contents)())
    if isinstance(contents, list):
        if collapseSequences and len(contents) > 1:
            return (
                '[' +
                summary_repr(contents[0], collapseSequences) +
                f", 'and {len(contents) - 1} more'" +
                ']'
            )
        return (
            '[' +
            ', '.join([summary_repr(elem, collapseSequences) for elem in contents]) +
            ']'
        )
    if isinstance(contents, tuple):
        if collapseSequences and len(contents) > 1:
            return (
                '(' +
                summary_repr(contents[0], collapseSequences) +
                f", 'and {len(contents) - 1} more'" +
                ',)'
            )
        return (
            '(' +
            ', '.join([summary_repr(elem, collapseSequences) for elem in contents]) +
            ',)'
        )
    if isinstance(contents, dict):
        return (
            '{' +
            ', '.join(
                [
                    summary_repr(key, collapseSequences) +
                    ': ' +
                    summary_repr(value, collapseSequences)
                    for key, value in contents.items()
                ],
            ) +
            '}'
        )
    if isinstance(contents, set):
        if collapseSequences and len(contents) > 1:
            return (
                '{' +
                summary_repr(next(iter(contents)), collapseSequences) +
                f", 'and {len(contents) - 1} more'" +
                '}'
            )
        return '{' + ', '.join([summary_repr(elem, collapseSequences) for elem in contents]) + '}'
    if isinstance(contents, np.ndarray):
        return (
            repr(type(contents)) +
            '(shape=' +
            repr(contents.shape) +
            ', dtype=np.' +
            repr(contents.dtype) +
            ')'
        )
    return repr(type(contents))


def print_fully(name, contents):
    pass
    # saved_threshold = np.get_printoptions()['threshold']
    # np.set_printoptions(threshold=9223372036854775807)
    # print(f'{name} = {summary_repr(contents, True)}')
    # print(f'{name} = ')
    # print(repr(contents))
    # np.set_printoptions(threshold=saved_threshold)


def find_first_numpy_type(contents):
    response = {
        np.int32: 'numpy.int32',
        np.int64: 'numpy.int64',
        np.float32: 'numpy.float32',
        np.float64: 'numpy.float64',
    }
    if isinstance(contents, (int, float, bool, str)):
        return ''
    if isinstance(contents, (np.int32, np.int64, np.float32, np.float64)):
        return ' is a ' + response[type(contents)]
    if isinstance(contents, (tuple, list, np.ndarray)):
        for i, e in enumerate(contents):
            r = find_first_numpy_type(e)
            if r != '':
                return f'[{i}]' + r
        return ''
    if isinstance(contents, dict):
        for i, (k, v) in enumerate(contents.items()):
            r = find_first_numpy_type(k)
            if r != '':
                return f'.keys()[{i}]' + r
            r = find_first_numpy_type(v)
            if r != '':
                return f'[{k!r}]' + r
        return ''
    if isinstance(contents, set):
        for i, k in enumerate(contents):
            r = find_first_numpy_type(k)
            if r != '':
                return f'.keys()[{i}]' + r
        return ''
    return ' is not a recognized type'


def check_for_numpy(name, contents):
    found_string = find_first_numpy_type(contents)
    if found_string != '':
        print(f'{name}{found_string}')


class SuperpixelClassificationBase:
    uploadRetries = 3

    def getItemsAndAnnotations(self, gc, folderId, annotationName, missing=False):
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

    def getCurrentEpoch(self, itemsAndAnnot):
        epoch = 0
        for _, annot, _ in itemsAndAnnot:
            matches = re.search(r' epoch (\d+)', annot['annotation']['name'], re.IGNORECASE)
            if matches:
                epoch = max(epoch, int(matches.groups()[0]))
        return epoch

    def createSuperpixelsForItem(self, gc, annotationName, item, radius, magnification,
                                 annotationFolderId, userId, prog):
        from histomicstk.cli.SuperpixelSegmentation import \
            SuperpixelSegmentation

        def progCallback(step, count, total):
            if step == 'tiles':
                prog.item_progress(item, 0.05 + 0.8 * (count / total))
            else:
                prog.item_progress(item, 0.85 + 0.05 * (count / total))

        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tempdir:
            print('Create superpixels for %s' % item['name'])
            imagePath = os.path.join(tempdir, item['name'])
            gc.downloadFile(item['largeImage']['fileId'], imagePath)
            outImagePath = os.path.join(tempdir, '%s.pixelmap.tiff' % item['name'])
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
                callback=progCallback)
            print(spopts)

            prog.item_progress(item, 0.05)
            # TODO: add a progress callback to the createSuperPixels method so
            # we get more granular progress (requires a change in HistomicsTK).
            SuperpixelSegmentation.createSuperPixels(spopts)
            del spopts.callback
            prog.item_progress(item, 0.9)
            for attempt in tenacity.Retrying(stop=tenacity.stop_after_attempt(self.uploadRetries)):
                with attempt:
                    outImageFile = gc.uploadFileToFolder(annotationFolderId, outImagePath)
            outImageId = outImageFile['itemId']
            annot = json.loads(open(outAnnotationPath).read())
            annot['name'] = '%s Epoch 0' % annotationName
            annot['elements'][0]['girderId'] = outImageId
            print('Bounding boxes span to',
                  max(annot['elements'][0]['user']['bbox'][2::4]),
                  max(annot['elements'][0]['user']['bbox'][3::4]))
            check_for_numpy('annot', annot)
            print_fully('annot', annot)
            with open(outAnnotationPath, 'w') as annotation_file:
                json.dump(annot, annotation_file, indent=2, sort_keys=False)
            count = len(gc.get('annotation', parameters=dict(itemId=item['_id'])))
            for attempt in tenacity.Retrying(stop=tenacity.stop_after_attempt(self.uploadRetries)):
                with attempt:
                    gc.uploadFileToItem(
                        item['_id'], outAnnotationPath,
                        reference=json.dumps({
                            'identifier': 'LargeImageAnnotationUpload',
                            'itemId': item['_id'],
                            'fileId': item['largeImage']['fileId'],
                            'userId': userId}))
            # Wait for the upload to complete
            waittime = time.time()
            while time.time() - waittime < 120:
                if len(gc.get('annotation', parameters=dict(itemId=item['_id']))) > count:
                    break
                time.sleep(0.1)
            prog.item_progress(item, 1)
            print('Created superpixels')

    def createSuperpixels(self, gc, folderId, annotationName, radius, magnification,
                          annotationFolderId, numWorkers, prog):
        items = self.getItemsAndAnnotations(gc, folderId, annotationName, True)
        if not len(items):
            return
        prog.message('Creating superpixels')
        prog.progress(0)
        prog.items(items)
        print('Create superpixels as needed for %d item(s)' % len(items))
        folder = gc.getFolder(folderId)
        results = {}
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=numWorkers) as executor:
            for item in items:
                futures.append((item, executor.submit(
                    self.createSuperpixelsForItem, gc, annotationName, item, radius, magnification,
                    annotationFolderId, folder['creatorId'], prog)))
        prog.progress(1)
        for item, future in futures:
            results[item['_id']] = future.result()
        return results

    def initializeCreateFeatureFromPatchAndMaskSimple(self):
        # There is nothing to initialize
        pass

    def initializeCreateFeatureFromPatchAndMask(self):
        # This SuperpixelClassificationBase implementation allows only the "Simple"
        # approach.
        # assert self.feature_is_image
        self.initializeCreateFeatureFromPatchAndMaskSimple()

    def createFeatureFromPatchAndMaskSimple(self, patch, mask, maskvals):
        feature = np.array(patch.copy()).astype(np.uint8)
        feature[(mask != maskvals[0]).any(axis=-1) & (mask != maskvals[1]).any(axis=-1)] = [0, 0, 0]
        return feature

    def createFeatureListFromPatchAndMaskListSimple(self, patch_list, mask_list, maskvals_list):
        feature_list = [
            self.createFeatureFromPatchAndMaskSimple(patch, mask, maskvals)
            for patch, mask, maskvals in zip(patch_list, mask_list, maskvals_list)
        ]
        return feature_list

    def createFeatureFromPatchAndMask(self, patch, mask, maskvals):
        # This SuperpixelClassificationBase implementation allows only the "Simple"
        # approach.
        # assert self.feature_is_image
        feature = self.createFeatureFromPatchAndMaskSimple(patch, mask, maskvals)
        return feature

    def createFeatureListFromPatchAndMaskList(self, patch_list, mask_list, maskvals_list):
        # This SuperpixelClassificationBase implementation allows only the "Simple"
        # approach.
        # assert self.feature_is_image
        feature_list = self.createFeatureListFromPatchAndMaskListSimple(
            patch_list, mask_list, maskvals_list,
        )
        return feature_list

    def createFeaturesForItem(self, gc, item, elem, featureFolderId, fileName, patchSize, prog, cutoff):
        import large_image

        print('Create feature', fileName)
        lastlog = starttime = time.time()
        ds = None
        self.initializeCreateFeatureFromPatchAndMask()
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tempdir:
            filePath = os.path.join(tempdir, fileName)
            imagePath = os.path.join(tempdir, item['name'])
            gc.downloadFile(item['largeImage']['fileId'], imagePath)
            ts = large_image.open(imagePath)
            maskItem = gc.getItem(elem['girderId'])
            maskPath = os.path.join(tempdir, maskItem['name'] + '.tiff')
            gc.downloadFile(maskItem['largeImage']['fileId'], maskPath)
            tsMask = large_image.open(maskPath)

            num_values = len(elem['values'])
            labeled_samples = set([i for i, x in enumerate(elem['values']) if x > 0])
            unlabeled_samples = [i for i, x in enumerate(elem['values']) if x == 0]
            if num_values - len(labeled_samples) > cutoff:
                # only select a subset of unlabeled samples, i.e., prune the feature list
                random.shuffle(unlabeled_samples)
                unlabeled_samples = unlabeled_samples[:cutoff]
            indices = list(sorted(list(labeled_samples) + unlabeled_samples))

            with h5py.File(filePath, 'w') as fptr:
                batch_size = 1024  # TODO: Is this the best value?
                total_size = len(indices)
                for batch_start in range(0, total_size, batch_size):
                    #batch_list = elem['values'][batch_start: batch_start + batch_size]
                    batch_list = indices[batch_start: batch_start + batch_size]
                    patch_list = []
                    mask_list = []
                    maskvals_list = []

                    for idx, i in enumerate(batch_list, start=batch_start):
                        prog.item_progress(item, 0.9 * idx / total_size)
                        bbox = elem['user']['bbox'][i * 4: i * 4 + 4]
                        # use masked superpixel
                        if len(bbox) < 4:
                            pass
                        patch = ts.getRegion(
                            region=dict(
                                left=int(bbox[0]), top=int(bbox[1]),
                                right=int(bbox[2]), bottom=int(bbox[3])),
                            output=dict(maxWidth=patchSize, maxHeight=patchSize),
                            fill='#000',
                            format=large_image.constants.TILE_FORMAT_NUMPY)[0]
                        if patch.shape[2] in (2, 4):
                            patch = patch[:, :, :-1]
                        scale = 1
                        try:
                            scale = elem['transform']['matrix'][0][0]
                        except Exception:
                            pass
                        mask = tsMask.getRegion(
                            region=dict(
                                left=int(bbox[0] / scale), top=int(bbox[1] / scale),
                                right=int(bbox[2] / scale), bottom=int(bbox[3] / scale)),
                            output=dict(maxWidth=patchSize, maxHeight=patchSize),
                            fill='#000',
                            format=large_image.constants.TILE_FORMAT_NUMPY)[0]
                        if mask.shape[2] == 4:
                            mask = mask[:, :, :-1]
                        maskvals = [[val % 256, val // 256 % 256, val // 65536 % 256]
                                    for val in [(i + 1) * 2, (i + 1) * 2 + 1]]
                        patch_list.append(patch)
                        mask_list.append(mask)
                        maskvals_list.append(maskvals)
                        # Make sure only the *_list forms are used subsequently
                        del patch, mask, maskvals
                    feature_list = self.createFeatureListFromPatchAndMaskList(
                        patch_list, mask_list, maskvals_list,
                    )
                    for idx, feature in enumerate(feature_list, start=batch_start):
                        if not ds:
                            ds = fptr.create_dataset(
                                'images', (1,) + feature.shape, maxshape=(None,) + feature.shape,
                                dtype=np.float32, chunks=True)
                        else:
                            ds.resize((ds.shape[0] + 1,) + feature.shape)
                        ds[ds.shape[0] - 1] = feature
                        if time.time() - lastlog > 5:
                            lastlog = time.time()
                            print(ds.shape, len(elem['values']),
                                  '%5.3f' % (time.time() - starttime),
                                  '%5.3f' % ((len(elem['values']) - idx - 1) / (idx + 1) *
                                             (time.time() - starttime)),
                                  item['name'])
                    del batch_list, patch_list, mask_list, maskvals_list, feature_list
                used_indices_ds = fptr.create_dataset(
                    'used_indices', data=np.array(indices), dtype='i')
                print(ds.shape, len(elem['values']), '%5.3f' % (time.time() - starttime),
                      item['name'])
            prog.item_progress(item, 0.9)
            for attempt in tenacity.Retrying(stop=tenacity.stop_after_attempt(self.uploadRetries)):
                with attempt:
                    file = gc.uploadFileToFolder(featureFolderId, filePath)
            prog.item_progress(item, 1)
            return file

    def createFeatures(self, gc, folderId, annotationName, itemsAndAnnot, featureFolderId, patchSize, numWorkers,
                       prog, cutoff):
        # itemsAndAnnot = self.getItemsAndAnnotations(gc, folderId, annotationName)
        prog.message('Creating features')
        prog.progress(0)
        prog.items([item for item, _, _ in itemsAndAnnot])
        results = {}
        futures = []
        featureFiles = [
            f for item in gc.listItem(featureFolderId) for f in gc.listFile(item['_id'])
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=numWorkers) as executor:
            for item, _, elem in itemsAndAnnot:
                match = [
                    f for f in featureFiles if
                    re.match('^%s.*[.]feature.h5$' % re.escape(item['name']), f['name'])
                ]
                if len(match):
                    results[item['_id']] = match[0]
                else:  # fallback to hash-based naming - generate features if necessary
                    bbox = elem['user']['bbox']
                    hashval = repr(dict(
                        itemId=item['_id'], bbox=[int(v) for v in bbox], patchSize=patchSize))
                    hashval = hashlib.new('sha256', hashval.encode()).hexdigest()
                    fileName = 'feature-%s.h5' % (hashval)
                    match = [f for f in featureFiles if f['name'] == fileName]
                    if len(match):
                        results[item['_id']] = match[0]
                    else:
                        futures.append((item, executor.submit(
                            self.createFeaturesForItem, gc, item, elem, featureFolderId,
                            '%s.feature.h5' % (item['name']), patchSize, prog, cutoff)))
        for item, future in futures:
            file = future.result()
            try:
                if file and '_id' in file:
                    results[item['_id']] = file
            except Exception:
                pass
        prog.progress(1)
        print('Found %d item(s) with features' % len(results))
        return results

    def trainModelAddItem(self, gc, record, item, annotrec, elem, feature,
                          randomInput, labelList, excludeLabelList):
        if not randomInput and not any(v for v in elem['values']):
            return
        print('Adding %s, %s (%s:%r) for training' % (
            item['name'], annotrec['annotation']['name'], annotrec['_id'], annotrec['_version']))
        featurePath = os.path.join(record['tempdir'], feature['name'])
        gc.downloadFile(feature['_id'], featurePath)
        with h5py.File(featurePath, 'r') as ffptr:
            fds = ffptr['images']
            if 'used_indices' in ffptr:
                indices = ffptr['used_indices']
            else:
                indices = range(len(elem['values']))
            for i,idx in enumerate(indices):
                labelnum = elem['values'][idx]
                if 0 < labelnum < len(elem['categories']):
                    labelname = elem['categories'][labelnum]['label']
                    if labelname in excludeLabelList:
                        continue
                    if labelname not in record['groups']:
                        record['groups'][labelname] = elem['categories'][labelnum]
                elif randomInput:
                    labelnum = random.randint(1, len(labelList))
                    labelname = labelList[labelnum - 1]
                else:
                    continue
                patch = fds[i]
                if not record['ds']:
                    record['ds'] = record['fptr'].create_dataset(
                        'images', (1,) + patch.shape, maxshape=(None,) + patch.shape,
                        dtype=patch.dtype, chunks=True)
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

    def trainModel(self, gc, annotationName, itemsAndAnnot, features, modelFolderId,
                   batchSize, epochs, trainingSplit, randomInput, labelList,
                   excludeLabelList, prog):
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
                    'starttime': time.time()}
                prog.message('Collecting items for training')
                for idx, (item, annotrec, elem) in enumerate(itemsAndAnnot):
                    prog.progress(idx / len(itemsAndAnnot))
                    if item['_id'] not in features:
                        continue
                    self.trainModelAddItem(
                        gc, record, item, annotrec, elem,
                        features.get(item['_id']), randomInput, labelList,
                        set(excludeLabelList))
                prog.progress(1)
                if not record['ds']:
                    print('No labeled data')
                    return None, None
                record['labelds'] = fptr.create_dataset(
                    'labels', (len(record['labelvals']),), dtype=int)
                record['labelds'] = np.array(record['labelvals'], dtype=int)
                print(record['ds'].shape, record['counts'],
                      '%5.3f' % (time.time() - record['starttime']))
                prog.message('Creating model')
                prog.progress(0)
                history, modelPath = self.trainModelDetails(
                    record, annotationName, batchSize, epochs, itemsAndAnnot, prog, tempdir,
                    trainingSplit)

                modTrainingPath = os.path.join(tempdir, '%s ModTraining Epoch %d.h5' % (
                    annotationName, self.getCurrentEpoch(itemsAndAnnot)))
                with h5py.File(modTrainingPath, 'w') as mtptr:
                    mtptr.create_dataset('labels', data=np.void(pickle.dumps(record['labels'])))
                    mtptr.create_dataset('groups', data=np.void(pickle.dumps(record['groups'])))
                    try:
                        mtptr.create_dataset('history', data=np.void(pickle.dumps(history)))
                    except AttributeError as exc:
                        print(f'Cannot pickle history; skipping.  {exc}')
                prog.progress(1)
            for attempt in tenacity.Retrying(stop=tenacity.stop_after_attempt(self.uploadRetries)):
                with attempt:
                    modelFile = gc.uploadFileToFolder(modelFolderId, modelPath)
            print('Saved model')
            for attempt in tenacity.Retrying(stop=tenacity.stop_after_attempt(self.uploadRetries)):
                with attempt:
                    modTrainingFile = gc.uploadFileToFolder(modelFolderId, modTrainingPath)
            print('Saved modTraining')
            return modelFile, modTrainingFile

    def predictLabelsForItem(self, gc, annotationName, tempdir, model, item,
                             annotrec, elem, feature, curEpoch, userId, labels, groups,
                             makeHeatmaps, radius, magnification, certainty, batchSize, prog):
        import al_bench.factory

        print('Predicting %s' % (item['name']))
        featurePath = os.path.join(tempdir, feature['name'])
        gc.downloadFile(feature['_id'], featurePath)
        annotrec = annotrec['annotation']
        annotrec['elements'] = [elem]

        # Figure out which samples are already labeled
        labeled_samples: NDArray[np.int_] = np.nonzero(np.array(elem['values']))
        number_annotations = len(elem['values'])
        tiny = np.finfo(np.float32).tiny

        print(f'{labeled_samples = }')
        print(f'certainty_type = {certainty!r}')
        compCertainty = al_bench.factory.ComputeCertainty(
            certainty_type=certainty,
            percentiles=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50),
            cutoffs=(0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.9975, 0.999))
        # In case we are computing batchbald
        compCertainty.set_batchbald_num_samples(16)
        compCertainty.set_batchbald_batch_size(100)
        #compCertainty.set_batchbald_excluded_samples(labeled_samples)

        with h5py.File(featurePath, 'r') as ffptr:
            if 'used_indices' in ffptr:
                used_indices = set(list(ffptr['used_indices']))
            else:
                used_indices = set(range(number_annotations))
            all_indices = set(range(number_annotations))
            unused_indices = list(sorted(all_indices.difference(used_indices)))
            compCertainty.set_batchbald_excluded_samples(np.array(unused_indices))

            prog.item_progress(item, 0)
            # Create predicted annotation
            annot = copy.deepcopy(annotrec)
            annot['elements'][0].pop('id', None)
            annot['name'] = '%s Epoch %d Predictions' % (annotationName, curEpoch)
            annot['elements'][0]['categories'] = [groups[key] for key in labels]
            ds = ffptr['images']
            prog.item_progress(item, 0.05)
            _catWeights, _predictions, indices = self.predictLabelsForItemDetails(
                batchSize, ds, np.array(list(used_indices), dtype=np.int64), item, model, use_cuda, prog)
            # expand catWeights and predictions to be length of elem['values'] instead of just `cutoff` samples
            # then copy in results from predictions
            catWeights = np.zeros((number_annotations,) + _catWeights.shape[1:], dtype=np.float32 if str(_catWeights.dtype).endswith("32") else np.float64)
            predictions = np.zeros((number_annotations,) + _predictions.shape[1:], dtype=np.float32 if str(_predictions.dtype).endswith("32") else np.float64)
            for cw,p,idx in zip(_catWeights, _predictions, indices):
                catWeights[idx] = cw
                predictions[idx] = p
                
            print_fully('predictions', predictions)
            prog.item_progress(item, 0.7)
            # compCertainty needs catWeights to have shape (num_superpixels,
            # bayesian_samples, num_classes) if 'batchbald' is selected, otherwise the
            # shape should be (num_superpixels, num_classes).
            # Ask compCertainty to compute certainties
            cert = compCertainty.from_numpy_array(catWeights + tiny)
            print_fully('catWeights', catWeights)

            # After the call to compCertainty, those numbers that end up as values for
            # annot's keys 'values', 'confidence', 'categoryConfidence', and 'certainty'
            # should have shape (num_superpixels, num_classes).

            print_fully('cert', cert)
            scores = cert[certainty]['scores']
            print_fully('scores', scores)
            if len(catWeights.shape) == 3:
                # Average over the Bayesian samples
                scores = scores.mean(axis=1)
                catWeights = catWeights.mean(axis=1)
                epsilon = 1e-50
                predictions = np.log(catWeights + epsilon)
            cats = np.argmax(catWeights, axis=-1)
            # 0 means we didn't make a prediction, so increment by one
            #cats[indices] += 1
            conf = catWeights[list(all_indices), cats[np.arange(cats.shape[0])]]
            print_fully('cats', cats)
            print_fully('conf', conf)

            # give unused_indices the highest possible confidence so that they show up last in the active learning UI
            # (because it sorts by confidence in descending order)
            scores[unused_indices] = np.finfo(scores.dtype).max
            # additionally, ensure that labels that are already labeled also end up last or late in the recommendations
            # for the DSA UI, this prevents labeled samples from being shown again to the user
            scores[labeled_samples] = np.finfo(scores.dtype).max

            # additionally, ensure that labels that are already labeled also end up last or late in the recommendations
            # for the DSA UI, this prevents labeled samples from being shown again to the user
            scores[labeled_samples] = np.finfo(scores.dtype).max

            cats = cats.tolist()
            conf = conf.tolist()

            # Should this be from predictions or from catWeights?!!!
            predictions[np.isneginf(predictions)] = np.finfo(predictions.dtype).min
            catConf = predictions.tolist()
            scores = scores.tolist()
            annot['elements'][0]['values'] = cats
            annot['elements'][0]['user']['confidence'] = conf
            annot['elements'][0]['user']['categoryConfidence'] = catConf
            annot['elements'][0]['user']['certainty'] = scores
            annot['elements'][0]['user']['certainty_info'] = {
                'type': certainty,
                'percentiles': cert[certainty]['percentiles'],
                'cdf': cert[certainty]['cdf']}
            outAnnotationPath = os.path.join(tempdir, '%s.anot' % annot['name'])
            prog.item_progress(item, 0.75)
            check_for_numpy('annot', annot)
            print_fully('annot', annot)
            with open(outAnnotationPath, 'w') as annotation_file:
                json.dump(annot, annotation_file, indent=2, sort_keys=False)
            for attempt in tenacity.Retrying(stop=tenacity.stop_after_attempt(self.uploadRetries)):
                with attempt:
                    gc.uploadFileToItem(
                        item['_id'], outAnnotationPath, reference=json.dumps({
                            'identifier': 'LargeImageAnnotationUpload',
                            'itemId': item['_id'],
                            'fileId': item['largeImage']['fileId'],
                            'userId': userId}))
            prog.item_progress(item, 0.8)
            # Upload new user annotation
            newAnnot = annotrec.copy()
            newAnnot['elements'][0].pop('id', None)
            newAnnot['name'] = '%s Epoch %d' % (annotationName, curEpoch + 1)
            outAnnotationPath = os.path.join(tempdir, '%s.anot' % newAnnot['name'])
            check_for_numpy('newAnnot', newAnnot)
            print_fully('newAnnot', newAnnot)
            with open(outAnnotationPath, 'w') as annotation_file:
                json.dump(newAnnot, annotation_file, indent=2, sort_keys=False)
            for attempt in tenacity.Retrying(stop=tenacity.stop_after_attempt(self.uploadRetries)):
                with attempt:
                    gc.uploadFileToItem(
                        item['_id'], outAnnotationPath, reference=json.dumps({
                            'identifier': 'LargeImageAnnotationUpload',
                            'itemId': item['_id'],
                            'fileId': item['largeImage']['fileId'],
                            'userId': userId}))
            prog.item_progress(item, 0.85)
            if makeHeatmaps:
                self.makeHeatmapsForItem(
                    gc, annotationName, userId, tempdir, radius, item, elem, labels, groups,
                    curEpoch, annot['elements'][0]['user']['certainty'], catWeights, catConf)
            prog.item_progress(item, 1)

    def makeHeatmapsForItem(self, gc, annotationName, userId, tempdir, radius, item, elem, labels,
                            groups, curEpoch, conf, catWeights, catConf):
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
            fillColor = (fillColor.groups()[0] + ',1)') if fillColor else groups[key]['fillColor']
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
                    'normalizeRange': True}]})
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
                    'normalizeRange': False}]})

        uncert = np.array(conf)
        uncert = 1 - (uncert - np.amin(uncert)) / ((np.amax(uncert) - np.amin(uncert)) or 1)
        uncert = uncert.tolist()
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
                    uncert[ci]] for ci in range(len(uncert))],
                'colorRange': ['rgba(0,0,0,0)', 'rgba(0,0,255,0.75)',
                               'rgba(255,255,0,0.9)', 'rgba(255,0,0,1)'],
                'rangeValues': [0, 0.5, 0.75, 1],
                'normalizeRange': False}]})
        outAnnotationPath = os.path.join(tempdir, '%s.anot' % heatmaps[-1]['name'])
        check_for_numpy('heatmaps', heatmaps)
        print_fully('heatmaps', heatmaps)
        with open(outAnnotationPath, 'w') as annotation_file:
            json.dump(heatmaps, annotation_file, indent=2, sort_keys=False)
        for attempt in tenacity.Retrying(stop=tenacity.stop_after_attempt(self.uploadRetries)):
            with attempt:
                gc.uploadFileToItem(
                    item['_id'],
                    outAnnotationPath,
                    reference=json.dumps({'identifier': 'LargeImageAnnotationUpload',
                                          'itemId': item['_id'],
                                          'fileId': item['largeImage']['fileId'],
                                          'userId': userId}))

    def predictLabels(self, gc, folderId, annotationName, itemsAndAnnot, features, modelFolderId,
                      annotationFolderId, saliencyMaps, radius, magnification,
                      certainty, batchSize, prog):
        #itemsAndAnnot = self.getItemsAndAnnotations(gc, folderId, annotationName)
        curEpoch = self.getCurrentEpoch(itemsAndAnnot)
        folder = gc.getFolder(folderId)
        userId = folder['creatorId']
        prog.message('Predicting')
        prog.progress(0)
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tempdir:
            modelFile = None
            for item in gc.listResource(
                'item', {'folderId': modelFolderId, 'sort': 'updated', 'sortdir': -1},
            ):
                if annotationName in item['name'] and 'model' in item['name'].lower():
                    modelFile = next(gc.listFile(item['_id'], limit=1))
                    break
            if not modelFile:
                print('No model file found')
                return
            print(modelFile['name'], item)
            modelPath = os.path.join(tempdir, modelFile['name'])
            gc.downloadFile(modelFile['_id'], modelPath)

            modTrainingFile = None
            for item in gc.listResource(
                    'item', {'folderId': modelFolderId, 'sort': 'updated', 'sortdir': -1}):
                if annotationName in item['name'] and 'modtraining' in item['name'].lower():
                    modTrainingFile = next(gc.listFile(item['_id'], limit=1))
                    break
            if not modTrainingFile:
                print('No modTraining file found')
                return
            print(modTrainingFile['name'], item)
            modTrainingPath = os.path.join(tempdir, modTrainingFile['name'])
            gc.downloadFile(modTrainingFile['_id'], modTrainingPath)

            model = self.loadModel(modelPath)
            with h5py.File(modTrainingPath, 'r+') as mtptr:
                labels = pickle.loads(mtptr['labels'][()].tobytes())
                groups = pickle.loads(mtptr['groups'][()].tobytes())
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
                            int(rgbtbl[ll % 6][2] * fac)),
                        'fillColor': 'rgba(%d, %d, %d, 0.25)' % (
                            int(rgbtbl[ll % 6][0] * fac),
                            int(rgbtbl[ll % 6][1] * fac),
                            int(rgbtbl[ll % 6][2] * fac))}
            prog.items([item for item, _, _ in itemsAndAnnot])
            for item, annotrec, elem in itemsAndAnnot:
                if item['_id'] not in features:
                    continue
                self.predictLabelsForItem(
                    gc, annotationName, annotationFolderId, tempdir, model, item, annotrec, elem,
                    features.get(item['_id']), curEpoch, userId, labels, groups, saliencyMaps,
                    radius, magnification, certainty, batchSize, prog)
            prog.progress(1)

    def main(self, args, gc = None):
        self.feature_is_image = args.feature != 'vector'
        self.certainty = args.certainty

        print('\n>> CLI Parameters ...\n')
        pprint.pprint(vars(args))

        if gc is None:
            gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
            gc.token = args.girderToken
            gc.authenticate('admin', 'password')

            # check to make sure we have access to server
            if not [x for x in list(gc.listCollection()) if x['name'] == 'Active Learning']:
                raise Exception("Unable to authenticate with girder")

        with ProgressHelper(
                'Superpixel Classification', 'Superpixel classification', args.progress) as prog:
            if args.gensuperpixels:
                self.createSuperpixels(
                    gc, args.images, args.annotationName, args.radius, args.magnification,
                    args.annotationDir, args.numWorkers, prog)

            itemsAndAnnot = self.getItemsAndAnnotations(gc, args.images, args.annotationName)
            features = self.createFeatures(
                gc, args.images, args.annotationName, itemsAndAnnot, args.features, args.patchSize,
                args.numWorkers, prog, args.cutoff)

            if args.train:
                print("Training...")
                self.trainModel(
                    gc, args.annotationName, itemsAndAnnot, features, args.modeldir, args.batchSize,
                    args.epochs, args.split, args.randominput, args.labels, args.exclude, prog)

            print("Predicting labels...")
            self.predictLabels(
                gc, args.images, args.annotationName, itemsAndAnnot, features, args.modeldir, args.annotationDir,
                args.heatmaps, args.radius, args.magnification, args.certainty, args.batchSize,
                prog)
