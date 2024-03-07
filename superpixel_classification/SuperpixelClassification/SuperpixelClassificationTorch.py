import logging
import math
import os

import batchbald_redux as bbald
import batchbald_redux.consistent_mc_dropout
import numpy as np
import torch
from SuperpixelClassificationBase import (SuperpixelClassificationBase,
                                          summary_repr)

logger = logging.getLogger(__name__)


class _LogTorchProgress:
    def __init__(self, prog, total, start=0, width=1, item=None):
        """Pass a progress class and the total number of total"""
        self.prog = prog
        self.total = total
        self.start = start
        self.width = width
        self.item = item

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        val = ((epoch + 1) / self.total) * self.width + self.start
        if self.item is None:
            self.prog.progress(val)
        else:
            self.prog.item_progress(self.item, val)
        # Save logs information to report later!!!

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        val = ((batch + 1) / self.total) * self.width + self.start
        if self.item is None:
            self.prog.progress(val)
        else:
            self.prog.item_progress(self.item, val)


class _TorchModel(torch.nn.Module):
    def __init__(self, num_classes: int):
        super(_TorchModel, self).__init__()

        self.conv1: torch.Module = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv1_drop: torch.Module = bbald.consistent_mc_dropout.ConsistentMCDropout2d()
        self.conv2: torch.Module = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2_drop: torch.Module = bbald.consistent_mc_dropout.ConsistentMCDropout2d()
        self.conv3: torch.Module = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3_drop: torch.Module = bbald.consistent_mc_dropout.ConsistentMCDropout2d()
        self.fc1: torch.Module = torch.nn.Linear(1024, 128)
        self.fc1_drop: torch.Module = bbald.consistent_mc_dropout.ConsistentMCDropout()
        self.fc2: torch.Module = torch.nn.Linear(128, num_classes)

    def forward(self, input: torch.Tensor):
        input = torch.mul(input, 1.0 / 255)

        input = self.conv1(input)
        input = self.conv1_drop(input)
        input = torch.nn.functional.max_pool2d(input, 2)
        input = torch.nn.functional.relu(input)

        input = self.conv2(input)
        input = self.conv2_drop(input)
        input = torch.nn.functional.max_pool2d(input, 2)
        input = torch.nn.functional.relu(input)

        input = self.conv3(input)
        input = self.conv3_drop(input)
        input = torch.nn.functional.max_pool2d(input, 2)
        input = torch.nn.functional.relu(input)

        input = input.view(-1, 1024)

        input = self.fc1(input)
        input = self.fc1_drop(input)
        input = torch.nn.functional.relu(input)

        input = self.fc2(input)
        # To remain consistent with the Tensorflow implementation, we will not include
        # `input = torch.nn.functional.log_softmax(input, dim=1)` at this point.

        return input


class _ZipDataset(torch.utils.data.Dataset):
    def __init__(self, train_features, train_labels) -> None:
        torch.utils.data.Dataset.__init__(self)
        self.train_features = torch.from_numpy(train_features)
        self.train_labels = torch.from_numpy(train_labels)

    def __len__(self) -> int:
        return self.train_labels.shape[0]

    def __getitem__(self, index: int):
        return self.train_features[index, :], self.train_labels[index]


class SuperpixelClassificationTorch(SuperpixelClassificationBase):
    def trainModelDetails(self, record, annotationName, batchSize, epochs, itemsAndAnnot, prog,
                          tempdir, trainingSplit):
        # Make a data set and a data loader for each of training and validation
        count = len(record['ds'])
        # Split data into training and validation.  H5py requires that indices be
        # sorted.
        train_size = int(count * trainingSplit)
        shuffle = np.random.permutation(count)  # add seed=123 ?
        train_indices = np.sort(shuffle[0:train_size])
        val_indices = np.sort(shuffle[train_size:count])
        train_ds = _ZipDataset(record['ds'][train_indices].transpose((0, 3, 2, 1)),
                               record['labelds'][train_indices])
        val_ds = _ZipDataset(record['ds'][val_indices].transpose((0, 3, 2, 1)),
                             record['labelds'][val_indices])
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batchSize)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batchSize)
        print(batchSize, train_dl, val_dl)
        prog.progress(0.2)

        # make model
        num_classes = len(record['labels'])
        model: torch.nn.Module = _TorchModel(num_classes)
        prog.message('Training model')
        prog.progress(0)

        history = self.fitModel(model, train_dl, val_dl, epochs,
                                callbacks=[_LogTorchProgress(prog, epochs)])

        prog.message('Saving model')
        prog.progress(0)
        modelPath = os.path.join(tempdir, '%s Model Epoch %d.h5' % (
            annotationName, self.getCurrentEpoch(itemsAndAnnot)))
        self.saveModel(model, modelPath)
        return history, modelPath

    def fitModel(self, model: torch.nn.Module, train_dl: torch.utils.data.DataLoader,
                 val_dl: torch.utils.data.DataLoader, epochs: int, callbacks):
        model.train()  # Tell torch we will be training
        criterion = torch.nn.functional.nll_loss
        optimizer = torch.optim.Adam(model.parameters())

        num_validation_samples: int = 1  # Is this right?
        # Loop over the dataset multiple times
        for epoch in range(epochs):
            for cb in callbacks:
                cb.on_epoch_begin(epoch, logs=dict())
            train_loss: float = 0.0
            train_size: int = 0
            train_correct: float = 0.0
            for i, data in enumerate(train_dl):
                for cb in callbacks:
                    cb.on_train_batch_begin(i, logs=dict())
                inputs, labels = data
                info = f'inputs = {summary_repr(inputs)}'
                logger.info(info)
                info = f'labels = {summary_repr(labels)}'
                logger.info(info)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                # Apparently our criterion, nll_loss, requires that we squeeze our
                # `outputs` value here
                criterion_loss = criterion(outputs.squeeze(1), labels)
                criterion_loss.backward()
                optimizer.step()
                new_size: int = inputs.size(0)
                train_size += new_size
                new_loss: float = criterion_loss.item() * inputs.size(0)
                train_loss += new_loss
                new_correct_t: torch.Tensor
                new_correct_t = (torch.argmax(outputs, dim=1) == labels).float().sum()
                new_correct: float = new_correct_t.detach().cpu().numpy()
                train_correct += new_correct
                loss: float = new_loss / new_size
                accuracy: float = new_correct / new_size
                if not isinstance(accuracy, (int, float, np.float32, np.float64)):
                    accuracy = accuracy[()]
                logs = {'loss': loss, 'accuracy': accuracy}
                for cb in callbacks:
                    cb.on_train_batch_end(i, logs)
            loss = train_loss / train_size
            accuracy = train_correct / train_size
            if not isinstance(accuracy, (int, float, np.float32, np.float64)):
                accuracy = accuracy[()]
            logs = {'loss': loss, 'accuracy': accuracy}

            validation_loss: float = 0.0
            validation_size = 0
            validation_correct = 0.0
            with torch.no_grad():
                model.eval()  # Tell torch that we will be doing predictions
                for data in val_dl:
                    inputs, labels = data
                    outputs = model(inputs)
                    # Collapse multiple predictions into single one
                    outputs = torch.logsumexp(outputs, dim=1) - math.log(num_validation_samples)
                    # Apparently our criterion, nll_loss, requires that we squeeze
                    # our `labels` value here
                    criterion_loss = criterion(outputs, labels.squeeze(1), reduction='sum')
                    new_size = inputs.size(0)
                    validation_size += new_size
                    new_loss = criterion_loss.item() * inputs.size(0)
                    validation_loss += new_loss
                    new_correct_t = (torch.argmax(outputs, dim=1) == labels).float().sum()
                    new_correct = new_correct_t.detach().cpu().numpy()
                    validation_correct += new_correct
                val_loss: float = validation_loss / validation_size
                val_accuracy: float = validation_correct / validation_size
                more_logs = dict(val_loss=val_loss, val_accuracy=val_accuracy)
                logs = {**logs, **more_logs}
            for cb in callbacks:
                cb.on_epoch_end(epoch, logs)

        for cb in callbacks:
            cb.on_train_end(logs)  # `logs` is from the last epoch
        # Needs a meaningful return value in lieu of tensorflow.History object!!!
        return None

    def predictLabelsForItemDetails(self, batchSize, ds, item, model, prog):
        callbacks = [_LogTorchProgress(
            prog, (ds.shape[0] + batchSize - 1) // batchSize, 0.05, 0.35, item)]
        for cb in callbacks:
            cb.on_predict_begin(logs=dict())
        with torch.no_grad():
            model.eval()  # Tell torch that we will be doing predictions
            predictions_raw = model(torch.from_numpy(ds))
            predictions = predictions_raw.detach().cpu().numpy()
        for cb in callbacks:
            cb.on_predict_end({'outputs': predictions})
        prog.item_progress(item, 0.4)
        # scale to units
        cats = [np.argmax(r) for r in predictions]
        # softmax to scale to 0 to 1.
        catWeights = torch.nn.functional.softmax(predictions, dim=1)
        return cats, catWeights, predictions

    def loadModel(self, modelPath):
        model = torch.load(modelPath)
        model.eval()
        return model

    def saveModel(self, model, modelPath):
        torch.save(model, modelPath)
