import os

import batchbald_redux as bbald
import batchbald_redux.consistent_mc_dropout
import numpy as np
import torch
from SuperpixelClassificationBase import SuperpixelClassificationBase


class _LogTorchProgress:
    def __init__(self, prog, total, start=0, width=1, item=None) -> None:
        """Pass a progress class and the total number of total"""
        self.prog = prog
        self.total = total
        self.start = start
        self.width = width
        self.item = item

    def on_epoch_begin(self, epoch, logs=None) -> None:
        pass

    def on_epoch_end(self, epoch, logs=None) -> None:
        val = ((epoch + 1) / self.total) * self.width + self.start
        if self.item is None:
            self.prog.progress(val)
        else:
            self.prog.item_progress(self.item, val)
        # TODO: Save logs information to report later

    def on_train_begin(self, logs=None) -> None:
        pass

    def on_train_end(self, logs=None) -> None:
        pass

    def on_train_batch_begin(self, batch, logs=None) -> None:
        pass

    def on_train_batch_end(self, batch, logs=None) -> None:
        pass

    def on_predict_begin(self, logs=None) -> None:
        pass

    def on_predict_end(self, logs=None) -> None:
        pass

    def on_predict_batch_begin(self, batch, logs=None) -> None:
        pass

    def on_predict_batch_end(self, batch, logs=None) -> None:
        val = ((batch + 1) / self.total) * self.width + self.start
        if self.item is None:
            self.prog.progress(val)
        else:
            self.prog.item_progress(self.item, val)


class _BayesianTorchModel(bbald.consistent_mc_dropout.BayesianModule):
    def __init__(self, num_classes: int) -> None:
        self.device = torch.device(
            'cuda'
            if torch.cuda.is_available() and torch.cuda.device_count() > 0
            else 'cpu',
        )
        super(_BayesianTorchModel, self).__init__()
        self.to(self.device)

        self.conv1: torch.Module = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv1_drop: torch.Module = bbald.consistent_mc_dropout.ConsistentMCDropout2d()
        self.conv2: torch.Module = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2_drop: torch.Module = bbald.consistent_mc_dropout.ConsistentMCDropout2d()
        self.conv3: torch.Module = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3_drop: torch.Module = bbald.consistent_mc_dropout.ConsistentMCDropout2d()
        self.fc1: torch.Module = torch.nn.Linear(9216, 128)
        self.fc1_drop: torch.Module = bbald.consistent_mc_dropout.ConsistentMCDropout()
        self.fc2: torch.Module = torch.nn.Linear(128, num_classes)

        self.num_classes: int = num_classes
        self.bayesian_samples: int = 12

    def mc_forward_impl(self, input: torch.Tensor) -> torch.Tensor:
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

        input = input.view(-1, 9216)

        input = self.fc1(input)
        input = self.fc1_drop(input)
        input = torch.nn.functional.relu(input)

        input = self.fc2(input)

        # To remain consistent with the Tensorflow implementation, we will not include
        # `input = torch.nn.functional.log_softmax(input, dim=1)` at this point.

        return input


class SuperpixelClassificationTorch(SuperpixelClassificationBase):
    def trainModelDetails(self, record, annotationName, batchSize, epochs, itemsAndAnnot, prog,
                          tempdir, trainingSplit):
        # Make a data set and a data loader for each of training and validation
        count = len(record['ds'])
        # Split data into training and validation.  H5py requires that indices be
        # sorted.
        train_size = int(count * trainingSplit)
        shuffle = np.random.permutation(count)  # TODO: add seed=123?
        train_indices = np.sort(shuffle[0:train_size])
        val_indices = np.sort(shuffle[train_size:count])
        train_arg1 = torch.from_numpy(record['ds'][train_indices].transpose((0, 3, 2, 1)))
        train_arg2 = torch.from_numpy(record['labelds'][train_indices])
        val_arg1 = torch.from_numpy(record['ds'][val_indices].transpose((0, 3, 2, 1)))
        val_arg2 = torch.from_numpy(record['labelds'][val_indices])
        train_ds = torch.utils.data.TensorDataset(train_arg1, train_arg2)
        val_ds = torch.utils.data.TensorDataset(val_arg1, val_arg2)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batchSize)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batchSize)
        prog.progress(0.2)

        # make model
        num_classes = len(record['labels'])
        model: torch.nn.Module = _BayesianTorchModel(num_classes)
        prog.message('Training model')
        prog.progress(0)

        history = self.fitModel(
            model, train_dl, val_dl, epochs, callbacks=[_LogTorchProgress(prog, epochs)],
        )

        prog.message('Saving model')
        prog.progress(0)
        modelPath = os.path.join(tempdir, '%s Model Epoch %d.pth' % (
            annotationName, self.getCurrentEpoch(itemsAndAnnot)))
        self.saveModel(model, modelPath)
        return history, modelPath

    def fitModel(self, model: torch.nn.Module, train_dl: torch.utils.data.DataLoader,
                 val_dl: torch.utils.data.DataLoader, epochs: int, callbacks):
        model.train()  # Tell torch we will be training
        criterion = torch.nn.functional.nll_loss
        optimizer = torch.optim.Adam(model.parameters())

        # TODO: What are good values for num_samples here?!!!  Eg., simply 1?
        num_training_samples: int = 12
        num_validation_samples: int = 12
        # Loop over the dataset multiple times
        for epoch in range(epochs):
            for cb in callbacks:
                cb.on_epoch_begin(epoch, logs=dict())
            train_loss: float = 0.0
            train_size: int = 0
            train_correct: float = 0.0
            for batch, data in enumerate(train_dl):
                for cb in callbacks:
                    cb.on_train_batch_begin(batch, logs=dict())
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs, num_training_samples)
                outputs = torch.nn.functional.log_softmax(outputs, dim=-1)
                # outputs.shape == (batch_size, num_training_samples, num_classes).
                # labels.shape  == (batch_size).
                # Broadcast labels to the same shape as outputs.shape[0:2]
                labels = labels[:, None].expand(*outputs.shape[0:2])
                criterion_loss = criterion(
                    outputs.reshape(-1, outputs.shape[-1]),
                    labels.reshape(-1),
                )
                criterion_loss.backward()
                optimizer.step()
                new_size: int = inputs.size(0)
                # print(f'new_size[{epoch}, {batch}] = {new_size}')
                train_size += new_size
                new_loss: float = criterion_loss.item() * new_size
                # print(f'new_loss[{epoch}, {batch}] = {new_loss}')
                train_loss += new_loss
                new_correct_t: torch.Tensor
                new_correct_t = (torch.argmax(outputs, dim=-1) == labels).float().sum()
                new_correct: float = new_correct_t.detach().cpu().numpy()
                # print(f'new_correct[{epoch}, {batch}] = {new_correct}')
                train_correct += new_correct
                loss: float = new_loss / new_size
                accuracy: float = new_correct / new_size
                if not isinstance(accuracy, (int, float, np.float32, np.float64)):
                    accuracy = accuracy[()]
                logs = {'loss': loss, 'accuracy': accuracy}
                for cb in callbacks:
                    cb.on_train_batch_end(batch, logs)
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
                    outputs = model(inputs, num_validation_samples)
                    outputs = torch.nn.functional.log_softmax(outputs, dim=-1)
                    # outputs.shape == (batch_size, num_training_samples, num_classes).
                    # labels.shape  == (batch_size).
                    # Broadcast labels to the same shape as outputs.shape[0:2]
                    labels = labels[:, None].expand(*outputs.shape[0:2])
                    criterion_loss = criterion(
                        outputs.reshape(-1, outputs.shape[-1]),
                        labels.reshape(-1),
                        reduction='sum',
                    )
                    new_size = inputs.size(0)
                    validation_size += new_size
                    new_loss = criterion_loss.item() * inputs.size(0)
                    validation_loss += new_loss
                    new_correct_t = (torch.argmax(outputs, dim=-1) == labels).float().sum()
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
        history = []  # TODO: Perhaps return something meaningful?
        return history

    def predictLabelsForItemDetails(self, batchSize, ds_h5, item, model, prog):
        num_superpixels: int = ds_h5.shape[0]
        # print(f'{num_superpixels = }')
        bayesian_samples: int = model.bayesian_samples
        # print(f'{bayesian_samples = }')
        num_classes: int = model.num_classes
        # print(f'{num_classes = }')

        # TODO: Stop overriding batchSize!!! We are ignoring the supplied value of
        # batchSize because that is what SuperpixelClassificationTensorflow is doing.
        # Instead we should have the calling routine pass in a reasonable value, which
        # might be distinct from the batch size used during training.
        batchSize = 1 + (num_superpixels - 1) // bayesian_samples
        # batchSize = num_superpixels  # This can be too many

        callbacks = [_LogTorchProgress(
            prog, 1 + (num_superpixels - 1) // batchSize, 0.05, 0.35, item)]
        logs = dict(
            num_superpixels=num_superpixels,
            bayesian_samples=bayesian_samples,
            num_classes=num_classes,
        )
        for cb in callbacks:
            cb.on_predict_begin(logs=logs)

        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(ds_h5).transpose((0, 3, 2, 1))),
        )
        dl = torch.utils.data.DataLoader(ds, batch_size=batchSize)
        predictions = np.zeros((num_superpixels, bayesian_samples, num_classes))
        catWeights = np.zeros((num_superpixels, bayesian_samples, num_classes))
        with torch.no_grad():
            model.eval()  # Tell torch that we will be doing predictions
            row: int = 0
            for i, data in enumerate(dl):
                for cb in callbacks:
                    cb.on_predict_batch_begin(i)
                inputs = data[0]
                new_row = row + inputs.shape[0]
                # print(f'inputs[{i}].shape = {inputs.shape}')
                predictions_raw = model(inputs, bayesian_samples)
                catWeights_raw = torch.nn.functional.softmax(predictions_raw, dim=-1)
                predictions[row:new_row, :, :] = predictions_raw.detach().cpu().numpy()
                # softmax to scale to 0 to 1.
                catWeights[row:new_row, :, :] = catWeights_raw.detach().cpu().numpy()
                row = new_row
                for cb in callbacks:
                    cb.on_predict_batch_end(i)
        for cb in callbacks:
            cb.on_predict_end({'outputs': predictions})
        prog.item_progress(item, 0.4)
        # scale to units
        return catWeights, predictions

    def loadModel(self, modelPath):
        model = torch.load(modelPath)
        model.eval()
        return model

    def saveModel(self, model, modelPath):
        torch.save(model, modelPath)
