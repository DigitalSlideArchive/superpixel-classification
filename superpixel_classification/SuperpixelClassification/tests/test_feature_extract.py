import os
import shutil
import sys
import tempfile
from unittest.mock import MagicMock

import h5py
import large_image
import numpy as np
import pytest

# make pythonpath work out of the box - although your editor may complain
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from SuperpixelClassificationBase import SuperpixelClassificationBase
from progress_helper import ProgressHelper
from tests.generate_MNIST_image import create_mnist_image

from xdg_base_dirs import ( xdg_cache_home, )

NUM_IMAGES = 64

@pytest.fixture(scope="session")
def create_sample_data():
    global NUM_IMAGES
    with tempfile.TemporaryDirectory() as tmpdirname:
        tiff_path = os.path.join(tmpdirname, "test_mnist.tiff")
        #tiff_path_pm = os.path.join(tmpdirname, "test_mnist.tiff.pixelmap.tiff")

        tiff_path, tiff_path_pm, labels = create_mnist_image(
            root_dataset_path=xdg_cache_home(),
            num_images=NUM_IMAGES,
            output_path=tiff_path,
            test=False,
        )
        # 0 is background
        labels['value'] = labels['value'] + 1

        # we use yield so that the temporarydirectory is still open in the tests
        yield tiff_path, tiff_path_pm, NUM_IMAGES, labels

MNIST_IMAGE_SIZE=28
COLOR_DIM = 3

def test_cutoff(create_sample_data):
    global MNIST_IMAGE_SIZE, COLOR_DIM
    test_image_pth, test_image_pth_pm, num_images, labels = create_sample_data
    base = SuperpixelClassificationBase()

    # Create test data
    item = {
        'name': test_image_pth,
        'largeImage': {'fileId': 'test_image_id'}
    }

    # Mock girder client
    gc = MagicMock()
    def mv_to_dst(_, dst):
        if "pixelmap" in dst:
            if not os.path.exists(dst):
                return shutil.copy(test_image_pth_pm, dst)
        else:
            if not os.path.exists(dst):
                return shutil.copy(test_image_pth, dst)
        return None
    gc.downloadFile = MagicMock(side_effect=mv_to_dst)
    gc.getItem = MagicMock(return_value={'name': test_image_pth_pm, 'largeImage': {'fileId': 'foobar'}})
    def mv_to_src(_, src):
        dst = os.path.dirname(test_image_pth)
        return shutil.copy(src, dst)
    gc.uploadFileToFolder = MagicMock(side_effect=mv_to_src, return_value={'_id': 'test_file_id'})
    #gc.uploadFileToFolder = MagicMock(return_value={'_id': 'test_file_id'})

    bboxes = [[x, y, w + x, y + h] for _, (x, y, w, h) in labels[['x', 'y', 'w', 'h']].iterrows()]

    elem = {
        'girderId': 'test_girder_id',
        'values':
            [] \
            + list(labels['value'])[:-2]
            + [0, 0],  # last two images unlabeled
        'user': {
            'bbox':  [item for sublist in bboxes for item in sublist]
        },
        'transform': {'matrix': [[1.0]]}
    }

    filename = 'test_features.h5'
    h5_file = os.path.join(os.path.dirname(test_image_pth), filename)
    if os.path.exists(h5_file):
        os.remove(h5_file)

    assert not os.path.exists(h5_file)

    cutoff = 1
    with ProgressHelper( 'Superpixel Classification',
                         'Test feature', False) as prog:
        prog.progress(0)
        prog.items([item])
        result = base.createFeaturesForItem(
            gc=gc,
            item=item,
            elem=elem,
            featureFolderId='test_folder_id',
            fileName=filename,
            patchSize=MNIST_IMAGE_SIZE,
            prog=prog,
            cutoff=cutoff,
        )

    assert os.path.exists(h5_file), f"Output file {h5_file} does not exist"
    with h5py.File(h5_file, 'r') as ffptr:
        assert 'images' in ffptr
        assert ffptr['images'].shape == (NUM_IMAGES - cutoff, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, COLOR_DIM)
        assert len(ffptr['used_indices']) == NUM_IMAGES - cutoff # number of labeled - cutoff

def test_create_features_for_item(create_sample_data):
    global MNIST_IMAGE_SIZE, COLOR_DIM
    test_image_pth, test_image_pth_pm, num_images, labels = create_sample_data
    base = SuperpixelClassificationBase()

    # Create test data
    item = {
        'name': test_image_pth,
        'largeImage': {'fileId': 'test_image_id'}
    }

    # Mock girder client
    gc = MagicMock()
    def mv_to_dst(_, dst):
        if "pixelmap" in dst:
            if not os.path.exists(dst):
                return shutil.copy(test_image_pth_pm, dst)
        else:
            if not os.path.exists(dst):
                return shutil.copy(test_image_pth, dst)
        return None
    gc.downloadFile = MagicMock(side_effect=mv_to_dst)
    gc.getItem = MagicMock(return_value={'name': test_image_pth_pm, 'largeImage': {'fileId': 'foobar'}})
    def mv_to_src(_, src):
        dst = os.path.dirname(test_image_pth)
        return shutil.copy(src, dst)
    gc.uploadFileToFolder = MagicMock(side_effect=mv_to_src, return_value={'_id': 'test_file_id'})
    #gc.uploadFileToFolder = MagicMock(return_value={'_id': 'test_file_id'})

    bboxes = [[x, y, w + x, y + h] for _, (x, y, w, h) in labels[['x', 'y', 'w', 'h']].iterrows()]

    elem = {
        'girderId': 'test_girder_id',
        'values':
            [] \
            + list(labels['value'])[:-2]
            + [0, 0],  # last two images unlabeled
        'user': {
            'bbox':  [item for sublist in bboxes for item in sublist]
        },
        'transform': {'matrix': [[1.0]]}
    }

    filename = 'test_features.h5'
    h5_file = os.path.join(os.path.dirname(test_image_pth), filename)
    if os.path.exists(h5_file):
        os.remove(h5_file)

    assert not os.path.exists(h5_file)

    with ProgressHelper( 'Superpixel Classification',
                         'Test feature', False) as prog:
        prog.progress(0)
        prog.items([item])
        result = base.createFeaturesForItem(
            gc=gc,
            item=item,
            elem=elem,
            featureFolderId='test_folder_id',
            fileName=filename,
            patchSize=MNIST_IMAGE_SIZE,
            prog=prog,
            cutoff=9999
        )

    assert os.path.exists(h5_file), f"Output file {h5_file} does not exist"
    with h5py.File(h5_file, 'r') as ffptr:
        assert 'images' in ffptr
        assert ffptr['images'].shape == (num_images, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, COLOR_DIM)
        feature_img = ffptr['images'][0]
        # open test_image_pth using coordinates [x,y,w,h] from elem['user']['bbox'][:4] and make sure it's pixel-equal with first_img
        x, y, x2, y2 = elem['user']['bbox'][:4]
        ts = large_image.getTileSource(test_image_pth)
        orig_image = ts.getRegion(
            region=dict(left=x, top=y, right=x2, bottom=y2),
            format=large_image.tilesource.TILE_FORMAT_NUMPY
        )[0]
        orig_image = orig_image.astype(feature_img.dtype)
        print(orig_image.dtype)
        np.testing.assert_array_equal(orig_image, feature_img)

        # also check that the last image matches
        feature_img = ffptr['images'][-1]
        x, y, x2, y2 = elem['user']['bbox'][-4:]
        ts = large_image.getTileSource(test_image_pth)
        orig_image = ts.getRegion(
            region=dict(left=x, top=y, right=x2, bottom=y2),
            format=large_image.tilesource.TILE_FORMAT_NUMPY
        )[0]
        orig_image = orig_image.astype(feature_img.dtype)
        print(orig_image.dtype)
        np.testing.assert_array_equal(orig_image, feature_img)

        assert 'used_indices' in ffptr
        assert len(ffptr['used_indices']) == num_images

    # Assertions
    assert result == h5_file
    assert gc.downloadFile.call_count == 2  # Called for both image and mask
    assert gc.getItem.call_count == 1
    assert gc.uploadFileToFolder.call_count == 1
