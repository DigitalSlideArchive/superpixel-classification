import os
import shutil
import tempfile
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest

# make pythonpath work out of the box - although your editor may complain
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from SuperpixelClassificationBase import SuperpixelClassificationBase
from SuperpixelClassificationTensorflow import SuperpixelClassificationTensorflow
from progress_helper import ProgressHelper

MNIST_IMAGE_SIZE=28
COLOR_DIM = 3
NUM_IMAGES = 64

@pytest.fixture(scope="session")
def create_sample_data():
    global NUM_IMAGES
    with tempfile.TemporaryDirectory() as tmpdirname:
        h5_path = os.path.join(tmpdirname, "test_data.h5")
        images = np.random.randint(0, 255, size=(NUM_IMAGES, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, COLOR_DIM), dtype=np.uint8)

        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('images', data=images)
            f.create_dataset('used_indices', data=np.arange(NUM_IMAGES - 2))

        # we use yield so that that the temporarydirectory is still open in the tests
        yield h5_path

def test_train_model(create_sample_data):
    global NUM_IMAGES
    h5_path = create_sample_data
    base: SuperpixelClassificationBase
    base = SuperpixelClassificationTensorflow()
    base.feature_is_image = True
    base.certainty = 'not batchbald' # same as using tensorflow

    # Mock girder client
    gc = MagicMock()
    def mv_to_dst(_, dst):
        return shutil.copy(h5_path, dst)
    gc.downloadFile = MagicMock(side_effect=mv_to_dst)
    def mv_to_src(_, src):
        dst = os.path.dirname(os.path.dirname(h5_path))
        return shutil.copy(src, dst)
    gc.uploadFileToFolder = MagicMock(side_effect=mv_to_src, return_value=True)

    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    elem = {
        'girderId': 'test_girder_id',
        'categories': [
            {"label": c} for c in labels
            ],
        'values':
            [] \
            + np.random.randint(1, len(labels) - 1, size=(NUM_IMAGES - 2), dtype=np.uint8).tolist()
            + [0, 0],  # last two images unlabeled
        'transform': {'matrix': [[1.0]]}
    }

    item = {'_id': 'test_h5_file', 'name': 'test'}
    annotrec = {'_id': '1', '_version': 0, 'annotation': {'name': 'TorchTest'}}
    items = [(item, annotrec, elem)]
    with ProgressHelper( 'Superpixel Classification',
                         'Test feature', False) as prog:
        prog.progress(0)
        prog.items(items)
        modelFile, modelTrainingFile = base.trainModel(
            annotationName="TorchTest",
            batchSize = 4,
            epochs = 1,
            excludeLabelList = [],
            features={'test_h5_file': {'_id': 'feature_id', 'name': 'test_h5_file'}},
            gc=gc,
            itemsAndAnnot=items,
            labelList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            modelFolderId="test_folder_id",
            prog=prog,
            randomInput = False,
            trainingSplit = 0.5,
            use_cuda = False,
        )

    assert os.path.exists(modelFile)
    assert os.path.exists(modelTrainingFile)