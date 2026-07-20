'''
This script will open a feature file (.h5) and show a 3x3 grid of images.
This tool is useful if you suspect that features are not extracted properly, for example due to erroneous mask values/indexing.
'''

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) > 0:
    feature_file = sys.argv[1]
else:
    feature_file = "features.h5"

# open the file
with h5py.File(feature_file, "r") as f:
    # get the images dataset
    images = f["images"]
    # get the first 9 images
    images = images[:9]
    # reshape the images to 3x3
    #images = np.reshape(images, (3,3,100,100,3))
    # transpose the images to 3x3
    #images = np.transpose(images, (0,2,1,3,4))
    # flatten the images to 9x100x100x3
    #images = np.reshape(images, (9,100,100,3))

    # hide axis from pyplot
    plt.axis('off')

    # plot the images
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i])
    plt.show()
    print(f"Image {i+1} is {images[i].shape}")
