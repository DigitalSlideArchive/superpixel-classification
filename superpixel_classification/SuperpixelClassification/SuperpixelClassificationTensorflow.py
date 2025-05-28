import os
import time
from typing import Optional

import h5py
import numpy as np
import tensorflow as tf
from SuperpixelClassificationBase import SuperpixelClassificationBase


class _LogTensorflowProgress(tf.keras.callbacks.Callback):
    def __init__(self, prog, total, start=0, width=1, item=None):
        """Pass a progress class and the total number of total"""
        self.prog = prog
        self.total = total
        self.start = start
        self.width = width
        self.item = item

    def on_epoch_end(self, epoch, logs=None):
        val = ((epoch + 1) / self.total) * self.width + self.start
        if self.item is None:
            self.prog.progress(val)
        else:
            self.prog.item_progress(self.item, val)

    def on_predict_batch_end(self, batch, logs=None):
        val = ((batch + 1) / self.total) * self.width + self.start
        if self.item is None:
            self.prog.progress(val)
        else:
            self.prog.item_progress(self.item, val)


class SuperpixelClassificationTensorflow(SuperpixelClassificationBase):
    def __init__(self):
        self.training_optimal_batchsize: Optional[int] = None
        self.prediction_optimal_batchsize: Optional[int] = None
        self.use_cuda = False

    def trainModelDetails(self, record, annotationName, batchSize, epochs, itemsAndAnnot, prog,
                          tempdir, trainingSplit, use_cuda):
        self.use_cuda = use_cuda

        # Enable GPU memory growth globally to avoid precondition errors
        gpus = tf.config.list_physical_devices('GPU')
        if gpus and self.use_cuda:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Could not set memory growth: {e}")
        if not self.use_cuda:
            tf.config.set_visible_devices([], 'GPU')
        device = "gpu" if use_cuda else "cpu"
        print(f"Using device: {device}")

        # Dataset preparation (outside strategy scope)
        ds_h5 = record['ds']
        labelds_h5 = record['labelds']
        # Fully load to memory and break h5py reference
        ds_numpy = np.array(ds_h5[:])
        labelds_numpy = np.array(labelds_h5[:])

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            num_classes = len(record['labels'])
            model = tf.keras.Sequential([
                tf.keras.layers.Rescaling(1.0 / 255),
                tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(num_classes)])
            prog.progress(0.2)
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])

        prog.progress(0.7)
        # generate split using numpy arrays
        full_ds = tf.data.Dataset.from_tensor_slices((ds_numpy, labelds_numpy))
        full_ds = full_ds.shuffle(1000)
        count = len(ds_numpy)
        train_size = int(count * trainingSplit)
        if batchSize < 1:
            batchSize = self.findOptimalBatchSize(model, full_ds, training=True)
            print(f'Optimal batch size for training = {batchSize}')
        train_ds = full_ds.take(train_size).batch(batchSize)
        val_ds = full_ds.skip(train_size).batch(batchSize)
        print(batchSize, train_ds, val_ds)
        prog.progress(0.9)
        prog.progress(1)
        prog.message('Training model')
        prog.progress(0)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[_LogTensorflowProgress(prog, epochs)])
        prog.message('Saving model')
        prog.progress(0)
        modelPath = os.path.join(tempdir, '%s Model Epoch %d.h5' % (
            annotationName, self.getCurrentEpoch(itemsAndAnnot)))
        self.saveModel(model, modelPath)
        return history, modelPath

    def _get_device(self, use_cuda):
        if tf.config.list_physical_devices('GPU') and use_cuda:
            return '/GPU:0'
        return '/CPU:0'

    def predictLabelsForItemDetails(
            self, batchSize, ds: h5py._hl.dataset.Dataset, indices, item, model, use_cuda, prog,
    ):
        if batchSize < 1:
            batchSize = self.findOptimalBatchSize(
                model, tf.data.Dataset.from_tensor_slices(ds), training=False,
            )
            print(f'Optimal batch size for prediction = {batchSize}')

        device = self._get_device(use_cuda)
        with tf.device(device):
            # Create a dataset that pairs the data with their indices
            dataset = tf.data.Dataset.from_tensor_slices((ds, indices))
            dataset = dataset.batch(batchSize)
        
            # Initialize arrays to store results
            all_predictions = []
            all_cat_weights = []
            all_indices = []
        
            # Iterate through batches manually to keep track of indices
            for data, batch_indices in dataset:
                batch_predictions = model.predict(
                    data,
                    batch_size=batchSize,
                    verbose=0)  # Set verbose=0 to avoid multiple progress bars
            
                # Apply softmax to scale to 0 to 1
                batch_cat_weights = tf.nn.softmax(batch_predictions)
            
                all_predictions.append(batch_predictions)
                all_cat_weights.append(batch_cat_weights)
                all_indices.append(batch_indices)
            
                prog.item_progress(item, 0.4)
        
            # Concatenate all results
            predictions = tf.concat(all_predictions, axis=0)
            catWeights = tf.concat(all_cat_weights, axis=0)
            final_indices = tf.concat(all_indices, axis=0)
        
            return catWeights.numpy(), predictions.numpy(), final_indices.numpy().astype(np.int64)

    def findOptimalBatchSize(self, model, ds, training) -> int:
        if training and self.training_optimal_batchsize is not None:
            return self.training_optimal_batchsize
        if not training and self.prediction_optimal_batchsize is not None:
            return self.prediction_optimal_batchsize
        # Find an optimal batch_size
        maximum_batchSize: int = 2 * len(ds) - 1
        batchSize: int = 2
        # We are using a value greater than 0.0 for add_seconds so that small imprecise
        # timings for small batch sizes don't accidentally trip the time check.
        add_seconds: float = 0.05
        previous_time: float = 1e100
        while batchSize <= maximum_batchSize:
            try:
                small_ds = ds.take(batchSize).batch(batchSize)
                start_time = time.time()
                model.predict(small_ds, batch_size=batchSize)
                elapsed_time = time.time() - start_time
                if elapsed_time > 2 * previous_time + add_seconds:
                    batchSize //= 2
                    return self.cacheOptimalBatchSize(batchSize, model, training)
                previous_time = elapsed_time
            except tf.errors.OpError:
                batchSize //= 2
                return self.cacheOptimalBatchSize(batchSize, model, training)
            batchSize *= 2
        # Undo the last doubling; it was spurious
        batchSize //= 2
        return self.cacheOptimalBatchSize(batchSize, model, training)

    def cacheOptimalBatchSize(self, batchSize, model, training) -> int:
        if training:
            self.training_optimal_batchsize = batchSize
        else:
            self.prediction_optimal_batchsize = batchSize
        return batchSize

    def loadModel(self, modelPath):
        return tf.keras.models.load_model(modelPath)

    def saveModel(self, model, modelPath):
        model.save(modelPath)
