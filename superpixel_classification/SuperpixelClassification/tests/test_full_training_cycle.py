'''
This file contains tests for a full training cycle: extracting superpixels, training and evaluation.
The "cycle" is:
    1. generate NUM_WSIS different whole slide images using numbers from MNIST.
    2. extract features from said images.
    3. train a model on the features.
    4. evaluate the model on the features.
We expect an accuracy of at least 90%.

This test is to verify that the training cycle works as expected.
Since there is batching involved, we want to use a larger number of samples instead of just a quick mini-test, as found in the other files.
'''
import argparse
import glob
import json
import os
import re
import shutil
import sys
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
from xdg_base_dirs import (xdg_cache_home, )

# make pythonpath work out of the box - although your editor may complain
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from SuperpixelClassificationBase import SuperpixelClassificationBase
from SuperpixelClassificationTensorflow import SuperpixelClassificationTensorflow
from SuperpixelClassificationTorch import SuperpixelClassificationTorch
from tests.generate_MNIST_image import create_mnist_image

NUM_WSIS = 2
MNIST_IMAGE_SIZE = 28
NUM_IMAGES_PER_WSI = 10 ** 2
COLOR_DIM = 3
PATCH_SIZE = 100 # only size compatible with pytorch model for the time being (since there are hardcoded sizes in the definition of the model)
NUM_EPOCHS = 5

@pytest.fixture(scope="function")
def create_sample_data(request):
    global NUM_WSIS, NUM_IMAGES_PER_WSI
    wsi_paths, pm_paths, list_labels = [], [], []
    with tempfile.TemporaryDirectory() as tmpdirname:
        for i in range(NUM_WSIS):
            tiff_path    = os.path.join(tmpdirname, f"test_mnist_{i}.tiff")

            tiff_path, tiff_path_pm, labels = create_mnist_image(
                root_dataset_path=xdg_cache_home(),
                num_images=NUM_IMAGES_PER_WSI,
                output_path=tiff_path,
                test=False,
                start_value = request.param
            )
            # where labels['value'] == 0, put 10 instead, since 0 will be reserved for unlabeled
            labels.loc[labels['value'] == 0, 'value'] = 10

            wsi_paths.append(tiff_path)
            pm_paths.append(tiff_path_pm)
            list_labels.append(labels)

        # we use yield so that the temporarydirectory is still open in the tests
        yield wsi_paths, pm_paths, NUM_WSIS, list_labels

@pytest.mark.skipif("RUNALL" not in os.environ, reason="this is a slow test (~5-10 min), run only if you want to")
@pytest.mark.parametrize('create_sample_data', [0], indirect=True)
def test_main_pytorch(create_sample_data):
    global NUM_WSIS, PATCH_SIZE, MNIST_IMAGE_SIZE, COLOR_DIM, NUM_EPOCHS
    tiff_paths, tiff_path_pms, num_images, labels = create_sample_data
    base: SuperpixelClassificationBase = SuperpixelClassificationTorch()

    annotation_name = 'torchMNISTtest'
    config = dict(
        annotationDir = 'annotationdir',
        annotationName = annotation_name,
        batchSize = int(np.sqrt(NUM_IMAGES_PER_WSI)), # one row of the wsi at a time
        certainty = 'batchbald',
        cutoff = 600000, # plenty of space to allow all training samples
        epochs = NUM_EPOCHS,
        exclude = [],
        feature = 'patch',
        features = 'featuredir',
        gensuperpixels = False,
        girderApiUrl = 'http://localhost:8080/api/v1',
        girderToken = '<PASSWORD>',
        heatmaps = False,
        images = 'imagedir',
        labels = '',
        magnification = 40.0,
        modeldir = '',
        numWorkers = 1,
        patchSize = PATCH_SIZE,
        radius    = MNIST_IMAGE_SIZE,
        randominput = False,
        split = 0.7,
        train = True,
        useCuda = True,
        progress = True,
    )
    args = argparse.Namespace(**config)

    mnist_labels = ['default', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

    items = []
    for i in range(NUM_WSIS):
        bboxes = [[x, y, w + x, y + h] for _, (x, y, w, h) in labels[i][['x', 'y', 'w', 'h']].iterrows()]
        elem = {
            'girderId': f'test_girder_id{i}',
            'categories': [
                {"label": c} for c in mnist_labels
                ],
            'values': labels[i]['value'].tolist(),
            'user': {
                'bbox':  [item for sublist in bboxes for item in sublist]
            },
            'transform': {'matrix': [[1.0]]}
        }
        item = {
            '_id': f'test_file{i}',
            'name': os.path.basename(tiff_paths[i]),
            'largeImage': {'fileId': f'test_image_id{i}'},
        }
        mask_item = {
            '_id': f'test_file{i}',
            'name': '.tiff'.join(os.path.basename(tiff_path_pms[i]).split('.tiff')[:-1]),
            'largeImage': {'fileId': f'test_mask_id{i}'},
        }
        annotrec = {
            '_id': f'test_file{i}',
            '_version': 0,
            'annotation': {'name': 'TorchTest'},
        }
        items.append((item, annotrec, elem))


    gc = MagicMock()
    base.getItemsAndAnnotations = MagicMock(return_value=items)

    with tempfile.TemporaryDirectory() as tmpdirname:
        def mv_to_dst(req_pth : str, dst : str):
            if req_pth.startswith("test_"):
                for f in tiff_paths + tiff_path_pms:
                    dpath = os.path.join(dst, os.path.basename(f))
                    if not os.path.exists(dpath) and os.path.basename(f) == os.path.basename(dst):
                        shutil.copy(f, dst)
                        print(f"Copied {f} to {dst}")
            elif req_pth.startswith("feature"):
                feature_files = glob.glob(os.path.join(tmpdirname, "*feature.h5"))
                for f in feature_files:
                    dpath = os.path.join(dst, os.path.basename(f))
                    if not os.path.exists(dpath) and os.path.basename(f) == os.path.basename(dst):
                        shutil.copy(f, dst)
                        print(f"Copied {f} to {dst}")
            elif req_pth.endswith("model"):
                model_file = glob.glob(os.path.join(tmpdirname, f"*Model *{0}.pth"))[0]
                shutil.copy(model_file, dst)
            elif "modtraining" in req_pth:
                model_file = glob.glob(os.path.join(tmpdirname, f"*ModTraining *{0}.h5"))[0]
                shutil.copy(model_file, dst)
            else:
                print(f"Received unknown request path '{req_pth}'")
            return {}

        gc.downloadFile = MagicMock(side_effect=mv_to_dst)
        def mv_to_src(req, src, reference=None):
            shutil.copy(src, tmpdirname)
            print(f"Copied {src} to {tmpdirname}")
            # each WSI gets two separate .anot files. The below if statement gives them unique filenames so we can reference later
            if src.endswith(".anot"):
                # extract the number at the end of req, which can look like "testfile1" or "testfile1000"
                m = re.search(r'(\d+)$', req)
                num = int(m.group(1))
                s = os.path.basename(src).replace(".anot", f"_{num}.myanot")
                shutil.copy(src, os.path.join(tmpdirname, s))
                print(f"Also copied {s} to {tmpdirname}")
            return {'_id': 'feature', 'name': os.path.basename(src)}
        gc.uploadFileToFolder = MagicMock(side_effect=mv_to_src, return_value=True)

        gc.getItem = MagicMock(return_value=mask_item)

        gc.listResource = MagicMock(return_value=[dict(name=f"{annotation_name}model", _id = 'model'), dict(name=f"{annotation_name}modtraining", _id = 'modtraining')])
        gc.uploadFileToItem = MagicMock(side_effect=mv_to_src, return_value=True)
        gc.getFolder = MagicMock(return_value=dict(name='test_folder', creatorId='creatorId', _id='test_folder_id'))

        def list_file(req: str, limit: int = 0) -> iter:
            if "modtraining" in req:
                return iter([dict(name=req, _id = 'modtraining')])
            else:
                return iter([dict(name=req, _id='model')])
        gc.listFile = MagicMock(side_effect=list_file)

        base.main(args, gc)

        for file in sorted(glob.glob(os.path.join(tmpdirname, f"*Predictions*.myanot"))):
            assert os.path.exists(file)
            with open(file, 'r') as f:
                pred_json = json.load(f)
                e = pred_json['elements'][0]
                assert len(e['values']) == NUM_IMAGES_PER_WSI

                assert len(e['user']['bbox']) == NUM_IMAGES_PER_WSI * 4 # 4 is for x,y,w,h

                assert len(e['categories']) == len(mnist_labels) - 1 # -1 because we don't have a default category
                assert len(e['user']['confidence']) == NUM_IMAGES_PER_WSI

                # compare e['values'] to labels['values'], to make sure we've trained a valid model
                # the order of the values is shuffled in the annotation file, the ordering is in e['categories']
                file_num = int(file.split('Predictions_')[-1].split('.myanot')[0])
                predicted_labels = np.array([e['categories'][c]['label'] for c in e['values']])
                matches = (predicted_labels == np.array(list(map(str, labels[file_num]['value']))))
                similarity = matches.sum() / len(matches)
                expected_min_accuracy = 0.75
                assert similarity > expected_min_accuracy, f"File {file}: Similarity between predicted values and GT is {similarity}, expected > {expected_min_accuracy}"
                print(f"Similarity between predicted values and GT is {similarity}")

@pytest.mark.skipif("RUNALL" not in os.environ, reason="this is a slow test (~1-10 min), run only if you want to")
@pytest.mark.parametrize('create_sample_data', [0], indirect=True)
def test_main_tf(create_sample_data):
    global NUM_WSIS, PATCH_SIZE, MNIST_IMAGE_SIZE, COLOR_DIM, NUM_EPOCHS
    tiff_paths, tiff_path_pms, num_images, labels = create_sample_data
    base: SuperpixelClassificationBase = SuperpixelClassificationTensorflow()

    annotation_name = 'tensorflowMNISTtest'
    config = dict(
        annotationDir = 'annotationdir',
        annotationName = annotation_name,
        batchSize = int(np.sqrt(NUM_IMAGES_PER_WSI)), # one row of the wsi at a time
        certainty = 'confidence',
        cutoff = 600000, # plenty of space to allow all training samples
        epochs = NUM_EPOCHS,
        exclude = [],
        feature = 'patch',
        features = 'featuredir',
        gensuperpixels = False,
        girderApiUrl = 'http://localhost:8080/api/v1',
        girderToken = '<PASSWORD>',
        heatmaps = False,
        images = 'imagedir',
        labels = '',
        magnification = 40.0,
        modeldir = 'modeldir',
        numWorkers = 1,
        patchSize = PATCH_SIZE,
        radius    = MNIST_IMAGE_SIZE,
        randominput = False,
        split = 0.7,
        train = True,
        useCuda = False,
        progress = True,
    )
    args = argparse.Namespace(**config)

    mnist_labels = ['default', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

    items = []
    for i in range(NUM_WSIS):
        bboxes = [[x, y, w + x, y + h] for _, (x, y, w, h) in labels[i][['x', 'y', 'w', 'h']].iterrows()]
        elem = {
            'girderId': f'test_girder_id{i}',
            'categories': [
                {"label": c} for c in mnist_labels
            ],
            'values': labels[i]['value'].tolist(),
            'user': {
                'bbox':  [item for sublist in bboxes for item in sublist]
            },
            'transform': {'matrix': [[1.0]]}
        }
        item = {
            '_id': f'test_file{i}',
            'name': os.path.basename(tiff_paths[i]),
            'largeImage': {'fileId': f'test_image_id{i}'},
        }
        mask_item = {
            '_id': f'test_file{i}',
            'name': '.tiff'.join(os.path.basename(tiff_path_pms[i]).split('.tiff')[:-1]),
            'largeImage': {'fileId': f'test_mask_id{i}'},
        }
        annotrec = {
            '_id': f'test_file{i}',
            '_version': 0,
            'annotation': {'name': 'TorchTest'},
        }
        items.append((item, annotrec, elem))


    gc = MagicMock()
    base.getItemsAndAnnotations = MagicMock(return_value=items)

    with tempfile.TemporaryDirectory() as tmpdirname:
        def mv_to_dst(req_pth : str, dst : str):
            if req_pth.startswith("test_"):
                for f in tiff_paths + tiff_path_pms:
                    dpath = os.path.join(dst, os.path.basename(f))
                    if not os.path.exists(dpath) and os.path.basename(f) == os.path.basename(dst):
                        shutil.copy(f, dst)
                        print(f"MockDownload: Copied {f} to {dst}")
            elif req_pth.startswith("feature"):
                feature_files = glob.glob(os.path.join(tmpdirname, "*feature.h5"))
                for f in feature_files:
                    dpath = os.path.join(dst, os.path.basename(f))
                    if not os.path.exists(dpath) and os.path.basename(f) == os.path.basename(dst):
                        shutil.copy(f, dst)
                        print(f"MockDownload: Copied {f} to {dst}")
            elif req_pth.endswith("model"):
                model_file = glob.glob(os.path.join(tmpdirname, f"*Model *{0}.h5"))[0]
                shutil.copy(model_file, dst)
            elif "modtraining" in req_pth:
                model_file = glob.glob(os.path.join(tmpdirname, f"*ModTraining *{0}.h5"))[0]
                shutil.copy(model_file, dst)
            else:
                raise RuntimeError(f"Received unknown request path '{req_pth}'")
            return {}

        gc.downloadFile = MagicMock(side_effect=mv_to_dst)
        def mv_to_src(req, src, reference=None):
            shutil.copy(src, tmpdirname)
            print(f"MockUpload: Copied {src} to {tmpdirname}")
            # each WSI gets two separate .anot files. The below if statement gives them unique filenames so we can reference later
            if src.endswith(".anot"):
                # extract the number at the end of req, which can look like "testfile1" or "testfile1000"
                m = re.search(r'(\d+)$', req)
                num = int(m.group(1))
                s = os.path.basename(src).replace(".anot", f"_{num}.myanot")
                shutil.copy(src, os.path.join(tmpdirname, s))
                print(f"Also copied {s} to {tmpdirname}")
            return {'_id': 'feature', 'name': os.path.basename(src)}
        gc.uploadFileToFolder = MagicMock(side_effect=mv_to_src, return_value=True)

        gc.getItem = MagicMock(return_value=mask_item)

        modelName = f"{annotation_name} Model Epoch 0.h5"
        modTrainingName = f"{annotation_name} ModTraining Epoch 0.h5"
        gc.listResource = MagicMock(return_value=[dict(name=modelName, _id = 'model'), dict(name=modTrainingName, _id = 'modtraining')])
        gc.uploadFileToItem = MagicMock(side_effect=mv_to_src, return_value=True)
        gc.getFolder = MagicMock(return_value=dict(name='test_folder', creatorId='creatorId', _id='test_folder_id'))

        def list_file(req: str, limit: int = 0) -> iter:
            if "modtraining" in req:
                return iter([dict(name=modTrainingName, _id = 'modtraining')])
            else:
                return iter([dict(name=modelName, _id='model')])
        gc.listFile = MagicMock(side_effect=list_file)

        base.main(args, gc)

        for file in sorted(glob.glob(os.path.join(tmpdirname, f"*Predictions*.myanot"))):
            assert os.path.exists(file)
            with open(file, 'r') as f:
                pred_json = json.load(f)
                e = pred_json['elements'][0]
                assert len(e['values']) == NUM_IMAGES_PER_WSI

                assert len(e['user']['bbox']) == NUM_IMAGES_PER_WSI * 4 # 4 is for x,y,w,h

                assert len(e['categories']) == len(mnist_labels) - 1 # exclude the default category
                assert len(e['user']['confidence']) == NUM_IMAGES_PER_WSI

                # compare e['values'] to labels['values'], to make sure we've trained a valid model
                # the order of the values is shuffled in the annotation file, the ordering is in e['categories']
                file_num = int(file.split('Predictions_')[-1].split('.myanot')[0])
                predicted_labels = np.array([e['categories'][c]['label'] for c in e['values']])
                matches = (predicted_labels == np.array(list(map(str, labels[file_num]['value']))))
                similarity = matches.sum() / len(matches)
                expected_min_accuracy = 0.75
                assert similarity > expected_min_accuracy, f"File {file}: Similarity between predicted values and GT is {similarity}, expected > {expected_min_accuracy}"
                print(f"Similarity between predicted values and GT is {similarity}")

@pytest.mark.skipif("RUNALL" not in os.environ, reason="this is a slow test (~1-10 min), run only if you want to")
@pytest.mark.parametrize('create_sample_data', [2], indirect=True)
def test_main_tf_with_background(create_sample_data):
    global NUM_WSIS, PATCH_SIZE, MNIST_IMAGE_SIZE, COLOR_DIM, NUM_EPOCHS
    tiff_paths, tiff_path_pms, num_images, labels = create_sample_data
    base: SuperpixelClassificationBase = SuperpixelClassificationTensorflow()

    annotation_name = 'tensorflowMNISTtest'
    config = dict(
        annotationDir = 'annotationdir',
        annotationName = annotation_name,
        batchSize = int(np.sqrt(NUM_IMAGES_PER_WSI)), # one row of the wsi at a time
        certainty = 'confidence',
        cutoff = 600000, # plenty of space to allow all training samples
        epochs = NUM_EPOCHS,
        exclude = [],
        feature = 'patch',
        features = 'featuredir',
        gensuperpixels = False,
        girderApiUrl = 'http://localhost:8080/api/v1',
        girderToken = '<PASSWORD>',
        heatmaps = False,
        images = 'imagedir',
        labels = '',
        magnification = 40.0,
        modeldir = 'modeldir',
        numWorkers = 1,
        patchSize = PATCH_SIZE,
        radius    = MNIST_IMAGE_SIZE,
        randominput = False,
        split = 0.7,
        train = True,
        useCuda = False,
        progress = True,
    )
    args = argparse.Namespace(**config)

    mnist_labels = ['default', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

    items = []
    for i in range(NUM_WSIS):
        bboxes = [[x, y, w + x, y + h] for _, (x, y, w, h) in labels[i][['x', 'y', 'w', 'h']].iterrows()]
        elem = {
            'girderId': f'test_girder_id{i}',
            'categories': [
                {"label": c} for c in mnist_labels
            ],
            'values': [0] + labels[i]['value'].tolist(),
            'user': {
                'bbox':  [0,0,1,1] + [item for sublist in bboxes for item in sublist]
            },
            'transform': {'matrix': [[1.0]]}
        }
        item = {
            '_id': f'test_file{i}',
            'name': os.path.basename(tiff_paths[i]),
            'largeImage': {'fileId': f'test_image_id{i}'},
        }
        mask_item = {
            '_id': f'test_file{i}',
            'name': '.tiff'.join(os.path.basename(tiff_path_pms[i]).split('.tiff')[:-1]),
            'largeImage': {'fileId': f'test_mask_id{i}'},
        }
        annotrec = {
            '_id': f'test_file{i}',
            '_version': 0,
            'annotation': {'name': 'TorchTest'},
        }
        items.append((item, annotrec, elem))


    gc = MagicMock()
    base.getItemsAndAnnotations = MagicMock(return_value=items)

    with tempfile.TemporaryDirectory() as tmpdirname:
        def mv_to_dst(req_pth : str, dst : str):
            if req_pth.startswith("test_"):
                for f in tiff_paths + tiff_path_pms:
                    dpath = os.path.join(dst, os.path.basename(f))
                    if not os.path.exists(dpath) and os.path.basename(f) == os.path.basename(dst):
                        shutil.copy(f, dst)
                        print(f"MockDownload: Copied {f} to {dst}")
            elif req_pth.startswith("feature"):
                feature_files = glob.glob(os.path.join(tmpdirname, "*feature.h5"))
                for f in feature_files:
                    dpath = os.path.join(dst, os.path.basename(f))
                    if not os.path.exists(dpath) and os.path.basename(f) == os.path.basename(dst):
                        shutil.copy(f, dst)
                        print(f"MockDownload: Copied {f} to {dst}")
            elif req_pth.endswith("model"):
                model_file = glob.glob(os.path.join(tmpdirname, f"*Model *{0}.h5"))[0]
                shutil.copy(model_file, dst)
            elif "modtraining" in req_pth:
                model_file = glob.glob(os.path.join(tmpdirname, f"*ModTraining *{0}.h5"))[0]
                shutil.copy(model_file, dst)
            else:
                raise RuntimeError(f"Received unknown request path '{req_pth}'")
            return {}

        gc.downloadFile = MagicMock(side_effect=mv_to_dst)
        def mv_to_src(req, src, reference=None):
            shutil.copy(src, tmpdirname)
            print(f"MockUpload: Copied {src} to {tmpdirname}")
            # each WSI gets two separate .anot files. The below if statement gives them unique filenames so we can reference later
            if src.endswith(".anot"):
                # extract the number at the end of req, which can look like "testfile1" or "testfile1000"
                m = re.search(r'(\d+)$', req)
                num = int(m.group(1))
                s = os.path.basename(src).replace(".anot", f"_{num}.myanot")
                shutil.copy(src, os.path.join(tmpdirname, s))
                print(f"Also copied {s} to {tmpdirname}")
            return {'_id': 'feature', 'name': os.path.basename(src)}
        gc.uploadFileToFolder = MagicMock(side_effect=mv_to_src, return_value=True)

        gc.getItem = MagicMock(return_value=mask_item)

        modelName = f"{annotation_name} Model Epoch 0.h5"
        modTrainingName = f"{annotation_name} ModTraining Epoch 0.h5"
        gc.listResource = MagicMock(return_value=[dict(name=modelName, _id = 'model'), dict(name=modTrainingName, _id = 'modtraining')])
        gc.uploadFileToItem = MagicMock(side_effect=mv_to_src, return_value=True)
        gc.getFolder = MagicMock(return_value=dict(name='test_folder', creatorId='creatorId', _id='test_folder_id'))

        def list_file(req: str, limit: int = 0) -> iter:
            if "modtraining" in req:
                return iter([dict(name=modTrainingName, _id = 'modtraining')])
            else:
                return iter([dict(name=modelName, _id='model')])
        gc.listFile = MagicMock(side_effect=list_file)

        base.main(args, gc)

        for file in sorted(glob.glob(os.path.join(tmpdirname, f"*Predictions*.myanot"))):
            assert os.path.exists(file)
            with open(file, 'r') as f:
                pred_json = json.load(f)
                e = pred_json['elements'][0]
                assert len(e['values']) == NUM_IMAGES_PER_WSI + 1

                assert len(e['user']['bbox']) == (NUM_IMAGES_PER_WSI + 1) * 4 # 4 is for x,y,w,h

                assert len(e['categories']) == len(mnist_labels) - 1 # exclude the default category
                assert len(e['user']['confidence']) == (NUM_IMAGES_PER_WSI + 1)

                # compare e['values'] to labels['values'], to make sure we've trained a valid model
                # the order of the values is shuffled in the annotation file, the ordering is in e['categories']
                file_num = int(file.split('Predictions_')[-1].split('.myanot')[0])
                predicted_labels = np.array([e['categories'][c]['label'] for c in e['values']])
                assert e['values'][0] == 0, "Background should have prediction 0"
                matches = (predicted_labels == np.array([e['values'][0]] + list(map(str, labels[file_num]['value']))))
                similarity = matches.sum() / len(matches)
                expected_min_accuracy = 0.75
                assert similarity > expected_min_accuracy, f"File {file}: Similarity between predicted values and GT is {similarity}, expected > {expected_min_accuracy}"
                print(f"Similarity between predicted values and GT is {similarity}")
