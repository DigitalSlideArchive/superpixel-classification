#!/usr/bin/env python
'''
Generate a .tiff with numbers from MNIST
'''

import os
import argparse
import random

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from torchvision.datasets import MNIST

def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate a pyramidal MNIST image.")
    parser.add_argument('--root_dataset_path', type=str, default="/data/aza4423_anders/mnist", help='Path to download and store MNIST dataset')
    #parser.add_argument('--num_images', type=int, default=244 * 244, help='Number of random MNIST images to use')
    parser.add_argument('--num_images', type=int, default=4, help='Number of random MNIST images to use')
    parser.add_argument('--output_path', type=str, default="/data/aza4423_anders/aml-dsa/mnist_pyramid.tif", help='Output path for the pyramidal TIF file')
    parser.add_argument('--test', default=False, type=bool, action=argparse.BooleanOptionalAction,
                        metavar='T',
                        help='whether to use test MNIST or train'
                        )

    args = parser.parse_args()

    return args

def d_to_rgb(d):
    r = d & 0xFF
    g = (d >> 8) & 0xFF
    b = (d >> 16) & 0xFF
    return [r, g, b]


def create_mnist_image(root_dataset_path=".", num_images=100, output_path="./out", test=False, start_value=0):
    # verify that num_images has a square root; otherwise we'd have to insert blank tiles for the uneven grid
    assert num_images % np.sqrt(num_images) == 0

    # Download MNIST (if not already downloaded)
    dataset = MNIST(root=root_dataset_path, train=not test, download=True)

    # Select N random MNIST images (each image is PIL.Image in mode "L")
    # (Make the number square-rootable)
    num_images = num_images  # Number of images from argument
    # oversample if we want more images than the length of MNIST
    if num_images > len(dataset):
        indices = random.choices(range(len(dataset)), k=num_images)
    else:
        indices = list(range(num_images))
        random.shuffle(indices)

    #indices = random.sample(range(len(dataset)), num_images)
    mnist_images = [np.array(dataset[i][0]) for i in indices]  # each is 28x28, uint8
    mnist_labels = [np.array(dataset[i][1]) for i in indices]

    # Arrange the images in a grid (so num_images should be a number with an integer root)
    tile_rows, tile_cols = int(np.sqrt(num_images)), int(np.sqrt(num_images))
    tile_h, tile_w = mnist_images[0].shape  # typically 28x28
    grid_h, grid_w = tile_rows * tile_h, tile_cols * tile_w
    base_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    pm_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for idx, img in enumerate(mnist_images):
        r = idx // tile_cols
        c = idx % tile_cols
        # convert img to RGB
        rgb_img = np.stack([img, img, img], axis=-1)
        base_image[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w, :] = rgb_img

        value_img = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        i = (idx + 1) * 2
        rgb = d_to_rgb(i + start_value)
        value_img[1:-1, 1:-1] = rgb
        rgb = d_to_rgb(i + start_value + 1)
        value_img[0, :] = rgb
        value_img[-1, :] = rgb
        value_img[:, 0] = rgb
        value_img[:, -1] = rgb

        pm_image[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w, :] = value_img


    # Note: We assume that the base level corresponds to 40x magnification.
    # Now, build a pyramid (list of downsampled images).
    pyramid_pm = [pm_image]
    pm_current = pm_image.copy()

    pyramid = [base_image]
    current = base_image.copy()
    # Continue downsampling by a factor of 2 until one dimension becomes very small.
    while min(current.shape) >= 64:
        # Use Pillow to resize (ANTIALIAS gives good quality downsampling)
        im = Image.fromarray(current)
        new_w, new_h = current.shape[1] // 2, current.shape[0] // 2
        if new_w < 1 or new_h < 1:
            break
        im_resized = im.resize((new_w, new_h))
        current = np.array(im_resized)
        pyramid.append(current)

        im = Image.fromarray(pm_image)
        new_w, new_h = pm_current.shape[1] // 2, pm_current.shape[0] // 2
        if new_w < 1 or new_h < 1:
            break
        im_resized = im.resize((new_w, new_h))
        pm_current = np.array(im_resized)
        pyramid_pm.append(current)

    # Save the image as a pyramidal TIFF.
    # The base image is the main image and the pyramid list (excluding the base) is saved as subIFDs.
    output_filename = output_path  # Use the output path from argument
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

    output_filename_pm = output_filename + ".pixelmap.tiff"  # Use the output path from argument
    if os.path.dirname(output_filename_pm):
        os.makedirs(os.path.dirname(output_filename_pm), exist_ok=True)
    if os.path.exists(output_filename_pm):
        os.remove(output_filename_pm)
    with tifffile.TiffWriter(output_filename_pm, bigtiff=False) as tif:
        tif.write(pyramid_pm[0],
                  tile=(tile_w * 4, tile_h * 4),
                  photometric='RGB',
                  description='Pixelmap for Whole-slide MNIST image at 40x magnification',
                  subifds=pyramid_pm[1:])
    print(f"Pyramidal TIFF saved as {output_filename_pm}")

    # generate a corresponding CSV "cells" file
    # with headers "x,y,w,h" for each image
    csv_filename = output_filename + "_cells.csv"
    with open(csv_filename, 'w') as f:
        f.write("x,y,w,h,value\n")
        i = 0
        for r in range(tile_rows):
            for c in range(tile_cols):
                x, y = c * tile_w, r * tile_h
                f.write(f"{x},{y},{tile_w},{tile_h},{mnist_labels[i]}\n")
                i += 1
    df = pd.read_csv(csv_filename, header=0)
    print(f"Annotation file saved as {csv_filename}")
    return output_filename, output_filename_pm, df

if __name__ == "__main__":
    _args = parse_args()
    create_mnist_image(_args.root_dataset_path, _args.num_images, _args.output_path, _args.test)
