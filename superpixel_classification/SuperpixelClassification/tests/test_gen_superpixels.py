import os
import shutil
import sys
import tempfile
from unittest.mock import MagicMock

import h5py
import large_image
import numpy as np
import pytest
from PIL.Image import Image
from tifffile import tifffile

# make pythonpath work out of the box - although your editor may complain
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from SuperpixelClassificationBase import SuperpixelClassificationBase
from progress_helper import ProgressHelper
from tests.generate_MNIST_image import create_mnist_image

from xdg_base_dirs import ( xdg_cache_home, )

NUM_IMAGES : int = 9
IMAGE_SIZE : int = 16 # 16 is the smallest tile size for .TIFFs, although we could operate within a single tile, too.
COLOR_DIM = 3


def d_to_rgb(d):
    r = d & 0xFF
    g = (d >> 8) & 0xFF
    b = (d >> 16) & 0xFF
    return [r, g, b]

@pytest.fixture(scope="session")
def create_sample_data():
    '''
    Create a sample WSI for testing.
    '''
    global NUM_IMAGES, IMAGE_SIZE
    num_images = NUM_IMAGES
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_filename = os.path.join(tmpdirname, "test.tiff")

        if os.path.dirname(output_filename):
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        if os.path.exists(output_filename):
            os.remove(output_filename)

        # Arrange the images in a grid (so num_images should be a number with an integer root)
        tile_rows, tile_cols = int(np.sqrt(num_images)), int(np.sqrt(num_images))
        tile_h, tile_w = 16, 16
        grid_h, grid_w = tile_rows * tile_h, tile_cols * tile_w
        base_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        vals = np.array([0, 127, 255], dtype=np.uint8)
        colors = np.stack(np.meshgrid(vals, vals, vals), axis=-1).reshape(-1, 3)[:NUM_IMAGES]
        images = np.tile(colors[:, None, None, :], (1, IMAGE_SIZE, IMAGE_SIZE, 1))

        for idx, img in enumerate(images):
            r = idx // tile_cols
            c = idx % tile_cols
            base_image[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w, :] = img

        pyramid = [base_image]
        current = base_image.copy()
        while min(current.shape) >= 64:
            # Use Pillow to resize (ANTIALIAS gives good quality downsampling)
            im = Image.fromarray(current)
            new_w, new_h = current.shape[1] // 2, current.shape[0] // 2
            if new_w < 1 or new_h < 1:
                break
            im_resized = im.resize((new_w, new_h))
            current = np.array(im_resized)
            pyramid.append(current)

        # Save the image as a pyramidal TIFF.
        # The base image is the main image and the pyramid list (excluding the base) is saved as subIFDs.
        if os.path.dirname(output_filename):
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        if os.path.exists(output_filename):
            os.remove(output_filename)

        with tifffile.TiffWriter(output_filename, bigtiff=False) as tif:
            tif.write(pyramid[0],
                      tile=(tile_w * 4, tile_h * 4),
                      photometric='RGB',
                      description='Whole-slide MNIST image at 40x magnification',
                      subifds=pyramid[1:])
        print(f"Pyramidal TIFF saved as {output_filename}")

        # we use yield so that the temporarydirectory is still open in the tests
        yield output_filename, images

def test_gen_superpixel(create_sample_data):
    global IMAGE_SIZE, COLOR_DIM
    test_image_pth, test_images = create_sample_data
    base = SuperpixelClassificationBase()

    # Create test data
    item = {
        "_id": "test_item_id",
        'largeImage': {'fileId': 'test_image_id'},
        'name': test_image_pth,
    }

    # Mock girder client
    gc = MagicMock()
    def mv_to_dst(_, dst):
        if not os.path.exists(os.path.join(dst, test_image_pth)):
            shutil.copy(test_image_pth, dst)
            print(">>> Copied file from", test_image_pth, "to", dst)
        return None
    gc.downloadFile = MagicMock(side_effect=mv_to_dst)
    gc.getItem = MagicMock(return_value={'name': test_image_pth, 'largeImage': {'fileId': 'foobar'}})
    def mv_to_src(_, src):
        dst = os.path.dirname(test_image_pth)
        if not os.path.exists(os.path.join(dst, src)):
            shutil.copy(src, dst)
            print(">>> Copied file from", src, "to", dst)
        return {'itemId': 'uploaded_item_id'}
    gc.uploadFileToFolder = MagicMock(side_effect=mv_to_src, return_value={'_id': 'test_file_id'})
    #gc.uploadFileToFolder = MagicMock(return_value={'_id': 'test_file_id'})

    #bboxes = [[x, y, w + x, y + h] for _, (x, y, w, h) in labels[['x', 'y', 'w', 'h']].iterrows()]
    bboxes = [[x, x, x + IMAGE_SIZE, x + IMAGE_SIZE] for x in range(0, NUM_IMAGES, IMAGE_SIZE)]

    with ProgressHelper( 'Superpixel Classification',
                         'Test feature', False) as prog:
        prog.progress(0)
        prog.items([item])
        result = base.createSuperpixelsForItem(
            gc=gc,
            annotationName="TorchTest",
            item=item,
            radius=IMAGE_SIZE,
            magnification=40,
            annotationFolderId='annotation_folder_id',
            userId="user_id",
            prog=prog,
        )

    out_pixelmap_file = os.path.join(os.path.dirname(test_image_pth), '%s.pixelmap.tiff' % item['name'])
    assert os.path.exists(out_pixelmap_file), f"Output file {out_pixelmap_file} does not exist"
    x, y, x2, y2 = 0, 0, IMAGE_SIZE, IMAGE_SIZE
    ts = large_image.getTileSource(test_image_pth)
    orig_image = ts.getRegion(
        region=dict(left=x, top=y, right=x2, bottom=y2),
        format=large_image.tilesource.TILE_FORMAT_NUMPY
    )[0]
    # test that all values in orig_image is equal to 1
    # TODO: waiting for another PR: want this to be 1
    assert np.all(orig_image == 0)

    feature_img = test_images[-1]
    x, y, x2, y2 = IMAGE_SIZE * (IMAGE_SIZE - 1), IMAGE_SIZE * (IMAGE_SIZE - 1), IMAGE_SIZE * IMAGE_SIZE, IMAGE_SIZE * IMAGE_SIZE
    ts = large_image.getTileSource(test_image_pth)
    orig_image = ts.getRegion(
        region=dict(left=x, top=y, right=x2, bottom=y2),
        format=large_image.tilesource.TILE_FORMAT_NUMPY
    )[0]
    orig_image = orig_image.astype(feature_img.dtype)
    # TODO: same as TODO above
    assert np.all(orig_image == NUM_IMAGES - 1)