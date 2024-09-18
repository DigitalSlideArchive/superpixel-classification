import os
import time

import h5py
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
    def trainModelDetails(self, record, annotationName, batchSize, epochs, itemsAndAnnot, prog,
                          tempdir, trainingSplit):
        # print(f'Tensorflow trainModelDetails(batchSize={batchSize}, ...)')
        # make model
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
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes)])
        prog.progress(0.2)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        prog.progress(0.7)
        # generate split
        full_ds = tf.data.Dataset.from_tensor_slices((record['ds'], record['labelds']))
        full_ds = full_ds.shuffle(1000)  # add seed=123 ?
        count = len(full_ds)
        train_size = int(count * trainingSplit)
        if batchSize < 1:
            batchSize = self.findOptimalBatchSize(model, full_ds)
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

    def predictLabelsForItemDetails(
        self, batchSize, ds: h5py._hl.dataset.Dataset, item, model, prog,
    ):
        # print(f'Tensorflow predictLabelsForItemDetails(batchSize={batchSize}, ...)')
        if batchSize < 1:
            batchSize = self.findOptimalBatchSize(model, tf.data.Dataset.from_tensor_slices(ds))
            print(f'Optimal batch size for prediction = {batchSize}')
        predictions = model.predict(
            ds,
            batch_size=batchSize,
            callbacks=[_LogTensorflowProgress(
                prog, (ds.shape[0] + batchSize - 1) // batchSize, 0.05, 0.35, item)])
        prog.item_progress(item, 0.4)
        # softmax to scale to 0 to 1
        catWeights = tf.nn.softmax(predictions)
        return catWeights, predictions

    def findOptimalBatchSize(self, model, ds) -> int:
        # Find an optimal batch_size
        maximum_batchSize: int = len(ds)
        batchSize: int = 2
        # We are using a value greater than 0.0 for add_seconds so that small imprecise
        # timings for small batch sizes don't accidentally trip the time check.
        add_seconds: float = 0.5
        previous_time: float = 1e100
        while batchSize <= maximum_batchSize:
            try:
                small_ds = ds.take(batchSize).batch(batchSize)
                start_time = time.time()
                model.predict(small_ds, batch_size=batchSize)
                elapsed_time = time.time() - start_time
                if elapsed_time > 2 * previous_time + add_seconds:
                    return batchSize // 2
                previous_time = elapsed_time
            except tf.errors.OpError:
                return batchSize // 2
            batchSize *= 2
        # Undo the last doubling; it was spurious
        return batchSize // 2

    def loadModel(self, modelPath):
        return tf.keras.models.load_model(modelPath)

    def saveModel(self, model, modelPath):
        model.save(modelPath)
