import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
import torch

# make pythonpath work out of the box - although your editor may complain
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from SuperpixelClassificationBase import SuperpixelClassificationBase
from SuperpixelClassificationTorch import SuperpixelClassificationTorch, _BayesianPatchTorchModel
from progress_helper import ProgressHelper
from tests.validate_json_annotation import validate_json_file

# currently, torch model only supports 100x100
MNIST_IMAGE_SIZE=100
COLOR_DIM = 3
NUM_IMAGES = 64
CUTOFF_IMAGES = 2

@pytest.fixture(scope="session")
def create_sample_data():
    global NUM_IMAGES, CUTOFF_IMAGES
    with tempfile.TemporaryDirectory() as tmpdirname:
        h5_path = os.path.join(tmpdirname, "test_data.h5")

        images = np.random.randint(0, 255, size=(NUM_IMAGES - CUTOFF_IMAGES, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, COLOR_DIM), dtype=np.uint8)
        indices = np.arange(NUM_IMAGES - CUTOFF_IMAGES)
        assert images.shape[0] == indices.shape[0]

        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('images', data=images)
            f.create_dataset('used_indices', data=indices, dtype='i')

        # we use yield so that the temporarydirectory is still open in the tests
        yield h5_path

'''
This test checks to predictions on a dataset that is only labeled with two values of out ten categories.
'''
def test_subset_labels(create_sample_data):
    global NUM_IMAGES, CUTOFF_IMAGES
    h5_path = create_sample_data
    base: SuperpixelClassificationBase = SuperpixelClassificationTorch()
    base.certainty = 'batchbald'
    base.feature_is_image = True
    # Mock girder client
    gc = MagicMock()
    def mv_to_dst(_, dst):
        return shutil.copy(h5_path, dst)
    gc.downloadFile = MagicMock(side_effect=mv_to_dst)
    gc.uploadFileToItem = MagicMock()

    feature = {
        '_id': '0',
        'name': 'my_test_feature'
    }
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    annotrec = {
        'annotation': {
            'attributes': {},
            'name': 'TorchTest',
        },
    }

    # make a list with values 1 and 3 in a random order, and NUM_IMAGES entries
    value_list = [1, 3] * (NUM_IMAGES // 2)

    elem = {
        "type": "pixelmap",
        "girderId": "6838aab654f0ca783ff03871",
        "transform": {"matrix": [[1.0, 0], [0, 1.0]]},
        'values': value_list,
        'categories' : [{"label": k, "fillColor": "rgba(0,0,0,0)"} for k in labels],
        "boundaries": True,
        "id": "myid",
        'user': { },
    }

    groups = { k: {"label": k, "fillColor": "rgba(0,0,0,0)", "strokeColor": "rgba(0,0,0,0)" } for k in labels }

    device = torch.device("cpu")
    model = _BayesianPatchTorchModel(len(labels), device)
    model.device = device

    items = [(feature, annotrec, elem)]
    item = {'_id': 0, 'name': 'my_item', 'largeImage': {'fileId': 'test_image_id'}}
    with ProgressHelper( 'Superpixel Classification',
                         'Test feature', False) as prog:
        prog.progress(0)
        prog.items(items)

        annotation_name = 'testannotation'
        with tempfile.TemporaryDirectory() as tmpdirname:
            base.predictLabelsForItem(
                gc=gc,
                annotationName=annotation_name,
                tempdir=tmpdirname,
                model=model,
                item=item,
                annotrec=annotrec,
                elem=elem,
                feature=feature,
                curEpoch=0,
                userId='user_id',
                labels=labels,
                groups=groups,
                makeHeatmaps=False,
                radius=-1,
                magnification=40.0,
                certainty='batchbald',
                batchSize=NUM_IMAGES,
                use_cuda = False,
                prog=prog,
            )
            out_pth = os.path.join(tmpdirname, '%s Epoch 0 Predictions.anot' % annotation_name)
            assert os.path.exists(out_pth), "Output file %s does not exist" % out_pth
            with open(out_pth, 'r') as f:
                pred_json = json.load(f)
                e = pred_json['elements'][0]
                assert len(e['values']) == NUM_IMAGES
                for i in range(1, CUTOFF_IMAGES):
                    assert e['values'][-i] == 0, "Expected unknown/none label for cutoff images"
                assert len(e['categories']) == len(labels)
                assert len(e['user']['confidence']) == NUM_IMAGES
                assert len(e['user']['categoryConfidence']) == NUM_IMAGES
                assert len(e['user']['categoryConfidence'][0]) == len(labels)
                assert len(e['user']['certainty']) == NUM_IMAGES
                for i in range(1, CUTOFF_IMAGES):
                    assert e['user']['certainty'][-i] > 10000, "Expected certainty to be very high for unlabeled samples to ensure they occur last in the AL filmstrip (DSA)"
                assert 'percentiles' in e['user']['certainty_info']
                assert 'cdf' in e['user']['certainty_info']

            validate_json_file(out_pth)

            out_pth = os.path.join(tmpdirname, '%s Epoch 1.anot' % annotation_name)
            assert os.path.exists(out_pth), "Output file %s does not exist" % out_pth
            with open(out_pth, 'r') as f:
                annotation_file = json.load(f)
                e = annotation_file['elements'][0]
                assert len(e['values']) == NUM_IMAGES
                assert len(e['categories']) == len(labels)

            validate_json_file(out_pth)

def test_predict_unlabeled_with_cutoff(create_sample_data):
    global NUM_IMAGES, CUTOFF_IMAGES
    h5_path = create_sample_data
    base: SuperpixelClassificationBase = SuperpixelClassificationTorch()
    base.certainty = 'batchbald'
    base.feature_is_image = True
    # Mock girder client
    gc = MagicMock()
    def mv_to_dst(_, dst):
        return shutil.copy(h5_path, dst)
    gc.downloadFile = MagicMock(side_effect=mv_to_dst)
    gc.uploadFileToItem = MagicMock()

    feature = {
       '_id': '0',
       'name': 'my_test_feature'
    }
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    annotrec = {
        'annotation': {
            'attributes': {},
            'name': 'TorchTest',
        },
    }

    elem = {
        "type": "pixelmap",
        "girderId": "6838aab654f0ca783ff03871",
        "transform": {"matrix": [[1.0, 0], [0, 1.0]]},
        'values': [0] * NUM_IMAGES,
        'categories' : [{"label": k, "fillColor": "rgba(0,0,0,0)"} for k in labels],
        "boundaries": True,
        "id": "myid",
        'user': { },
    }

    groups = { k: {"label": k, "fillColor": "rgba(0,0,0,0)", "strokeColor": "rgba(0,0,0,0)" } for k in labels }

    device = torch.device("cpu")
    model = _BayesianPatchTorchModel(len(labels), device)
    model.device = device

    items = [(feature, annotrec, elem)]
    item = {'_id': 0, 'name': 'my_item', 'largeImage': {'fileId': 'test_image_id'}}
    with ProgressHelper( 'Superpixel Classification',
                         'Test feature', False) as prog:
        prog.progress(0)
        prog.items(items)

        annotation_name = 'testannotation'
        with tempfile.TemporaryDirectory() as tmpdirname:
            base.predictLabelsForItem(
                gc=gc,
                annotationName=annotation_name,
                tempdir=tmpdirname,
                model=model,
                item=item,
                annotrec=annotrec,
                elem=elem,
                feature=feature,
                curEpoch=0,
                userId='user_id',
                labels=labels,
                groups=groups,
                makeHeatmaps=False,
                radius=-1,
                magnification=40.0,
                certainty='batchbald',
                batchSize=NUM_IMAGES,
                use_cuda = False,
                prog=prog,
            )
            out_pth = os.path.join(tmpdirname, '%s Epoch 0 Predictions.anot' % annotation_name)
            assert os.path.exists(out_pth), "Output file %s does not exist" % out_pth
            with open(out_pth, 'r') as f:
                pred_json = json.load(f)
                e = pred_json['elements'][0]
                assert len(e['values']) == NUM_IMAGES
                for i in range(1, CUTOFF_IMAGES):
                    assert e['values'][-i] == 0, "Expected unknown/none label for cutoff images"
                assert len(e['categories']) == len(labels)
                assert len(e['user']['confidence']) == NUM_IMAGES
                assert len(e['user']['categoryConfidence']) == NUM_IMAGES
                assert len(e['user']['categoryConfidence'][0]) == len(labels)
                assert len(e['user']['certainty']) == NUM_IMAGES
                for i in range(1, CUTOFF_IMAGES):
                    assert e['user']['certainty'][-i] > 10000, "Expected certainty to be very high for unlabeled samples to ensure they occur last in the AL filmstrip (DSA)"
                assert 'percentiles' in e['user']['certainty_info']
                assert 'cdf' in e['user']['certainty_info']

            validate_json_file(out_pth)

            out_pth = os.path.join(tmpdirname, '%s Epoch 1.anot' % annotation_name)
            assert os.path.exists(out_pth), "Output file %s does not exist" % out_pth
            with open(out_pth, 'r') as f:
                annotation_file = json.load(f)
                e = annotation_file['elements'][0]
                assert len(e['values']) == NUM_IMAGES
                assert len(e['categories']) == len(labels)

            validate_json_file(out_pth)
