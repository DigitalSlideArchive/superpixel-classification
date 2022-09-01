Superpixel Classification
=========================

This can be built like so::

    docker build --force-rm -t dsarchive/superpixel . 

To run it from the command line, you'll need to specify some parameters, notably for girder client and for various folders::

    docker run --gpus all --rm -it dsarchive/superpixel SuperpixelClassification \
    --api-url http://<girder server:port>/api/v1 \
    --girder-token <64-hexadecimal digit token> \
    62fd07a6e370d0bad9dd241b \
    --features 630d01243be9b4f10182cfb1 \
    --modeldir 630d012a3be9b4f10182cfb2 \
    --annotationdir 630d14b7384b2fd595b4cf54 \
    --workers 8

What It Does
------------

Internally, place a collection of large_images in a folder and use that folder Id as the primary.  Create folders to store annotations, models, and features.

(1) uses the HistomicsTK superpixel algorithm (which is just SLIC) to create superpixels.  These are stored in a large_image_annotation annotation element with ``values`` for the classes and ``user.bbox`` with the bounding box of each superpixel (this is 4 times the length of values and has left, top, right, bottom for each superpixel).

(2) Gets the region of each bound boxing at a specific scale and uses the raw pixels as the feature vector.

The user is expected to label some of the images on the "Superpixel Epoch <most recent>" annotation.  Creating a .histomicsui_config.yaml in the same directory as the images is useful to define your labels.  This file will contain something like::

    ---
    annotationGroups:
      replaceGroups: true
      defaultGroup: default
      groups:
        -
          id: Tissue
          fillColor: rgba(255, 255, 0, 0.25)
          lineColor: rgb(255, 255, 0)
          lineWidth: 2
        -
          id: Background
          fillColor: rgba(255, 0, 0, 0.25)
          lineColor: rgb(255, 0, 0)
          lineWidth: 2
        -
          id: Marker
          fillColor: rgba(0, 0, 255, 0.25)
          lineColor: rgb(0, 0, 255)
          lineWidth: 2

(3) Will train a model based off of a keras example for image classification.

(4) Emits a pair of new annotations for each prepared image.  One for the next epoch of labels and is a direct copy of the previous epoch.  The other has the predictions which labels the superpixels accordingly.  The predicted class is in the annotation element ``values`` array, ``user.bbox`` is based on the original annotation, ``user.confidence`` is an array with one entry per value containing a number from [0-1] which is the maximum softmax value from the predicted classes, ``user.categoryConfidence`` is an array with one entry per value, containing an array the same length as the ``categories`` (labels) with the raw logit value from the prediction.

Future
------

This should be refactored to use classes to make it easy to abstract the base process.  Specifically, this (1) selects image regions, (2) creates feature vectors per region, (3) trains a model using those features and some labels, (4) predicts labels based on that model.




