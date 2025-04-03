import os
import time
from typing import Any, Optional, Sequence

import batchbald_redux as bbald
import batchbald_redux.consistent_mc_dropout
import numpy as np
import torch
from numpy.typing import NDArray
from progress_helper import ProgressHelper
from SuperpixelClassificationBase import SuperpixelClassificationBase


class _LogTorchProgress:
    def __init__(
        self, prog: ProgressHelper, total: int, start: float = 0.0, width: float = 1.0, item=None,
    ) -> None:
        """Pass a progress class and the total number of total"""
        self.prog: ProgressHelper = prog
        self.total: int = total
        self.start: float = start
        self.width: float = width
        self.item = item

    def on_epoch_begin(self, epoch, logs=None) -> None:
        pass

    def on_epoch_end(self, epoch, logs=None) -> None:
        val: float = ((epoch + 1) / self.total) * self.width + self.start
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
        val: float = ((batch + 1) / self.total) * self.width + self.start
        if self.item is None:
            self.prog.progress(val)
        else:
            self.prog.item_progress(self.item, val)


class _BayesianPatchTorchModel(bbald.consistent_mc_dropout.BayesianModule):
    # A Bayesian model that takes patches (2-dimensional shape) rather than vectors
    # (1-dimensional shape) as input.  It is useful when feature != 'vector' and
    # SuperpixelClassificationBase.certainty == 'batchbald'.
    def __init__(self, num_classes: int) -> None:
        # Set `self.device` as early as possible so that other code does not lock out
        # what we want.
        self.device: str = torch.device(
            ('cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'),
        )
        # print(f'Initial model.device = {self.device}')
        super(_BayesianPatchTorchModel, self).__init__()

        self.conv1: torch.Module
        self.conv1_drop: torch.Module
        self.conv2: torch.Module
        self.conv2_drop: torch.Module
        self.conv3: torch.Module
        self.conv3_drop: torch.Module
        self.fc1: torch.Module
        self.fc1_drop: torch.Module
        self.fc2: torch.Module
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv1_drop = bbald.consistent_mc_dropout.ConsistentMCDropout2d()
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2_drop = bbald.consistent_mc_dropout.ConsistentMCDropout2d()
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3_drop = bbald.consistent_mc_dropout.ConsistentMCDropout2d()
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc1_drop = bbald.consistent_mc_dropout.ConsistentMCDropout()
        self.fc2 = torch.nn.Linear(128, num_classes)

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


class _VectorTorchModel(torch.nn.Module):
    # A non-Bayesian model that takes vectors (1-dimensional shape) rather than patches
    # (2-dimensional shape) as input.  It is useful when feature == 'vector' and
    # SuperpixelClassificationBase.certainty != 'batchbald'.

    def __init__(self, input_dim: int, num_classes: int) -> None:
        # Set `self.device` as early as possible so that other code does not lock out
        # what we want.
        self.device: str = torch.device(
            ('cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'),
        )
        # print(f'Initial model.device = {self.device}')
        super(_VectorTorchModel, self).__init__()

        self.input_dim: int = input_dim
        self.num_classes: int = num_classes
        self.fc: torch.Module = torch.nn.Linear(input_dim, num_classes)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # TODO: Is torch.mul appropriate here?
        input = torch.mul(input, 1.0 / 255)
        input = self.fc(input)
        # To remain consistent with the Tensorflow implementation, we will not include
        # `input = torch.nn.functional.log_softmax(input, dim=1)` at this point.
        return input


class _BayesianVectorTorchModel(bbald.consistent_mc_dropout.BayesianModule):
    # A Bayesian model that takes vectors (1-dimensional shape) rather than patches
    # (2-dimensional shape) as input.  It is useful when feature == 'vector' and
    # SuperpixelClassificationBase.certainty == 'batchbald'.

    def __init__(self, input_dim: int, num_classes: int) -> None:
        # Set `self.device` as early as possible so that other code does not lock out
        # what we want.
        self.device: str = torch.device(
            ('cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'),
        )
        # print(f'Initial model.device = {self.device}')
        super(_BayesianVectorTorchModel, self).__init__()

        self.input_dim: int = input_dim
        self.num_classes: int = num_classes
        self.bayesian_samples: int = 12
        self.fc: torch.Module = torch.nn.Linear(input_dim, num_classes)
        self.fc_drop: torch.Module = bbald.consistent_mc_dropout.ConsistentMCDropout()

    def mc_forward_impl(self, input: torch.Tensor) -> torch.Tensor:
        # TODO: Is torch.mul appropriate here?
        input = torch.mul(input, 1.0 / 255)
        input = self.fc(input)
        # TODO: Is it appropriate to have fc_drop as a last layer; we don't do that for
        # batchbald on patches?  More generally, is subclassing from bbald and using
        # self.bayesian_samples during training and/or prediction sufficient to make
        # this model properly Bayesian?
        input = self.fc_drop(input)
        # To remain consistent with the Tensorflow implementation, we will not include
        # `input = torch.nn.functional.log_softmax(input, dim=1)` at this point.
        return input


class SuperpixelClassificationTorch(SuperpixelClassificationBase):
    def __init__(self):
        self.training_optimal_batchsize: Optional[int] = None
        self.prediction_optimal_batchsize: Optional[int] = None

    def initializeCreateFeatureFromPatchAndMaskUNI(self):
        import timm
        import timm.data
        import timm.data.transforms_factory

        """
        To make the timm.create_model call succeed, be sure that a command like the following has
        already been run from a bash prompt on each system supporting the
        dsarchive/superpixel:latest docker image.  We need to run this command only if
        .cache/huggingface/hub does not already have the MahmoodLab/UNI model.  Instead of
        $HOME/.cache/huggingface/token in the following, use the actual location of your HuggingFace
        token; for security reasons it is better if the token is not within the mount point, which
        in this example is the tree rooted from the directory $HOME/.cache/huggingface/hub.

          docker run \
            --rm \
            --env=HF_TOKEN=$(cat $HOME/.cache/huggingface/token) \
            -v $HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub:rw \
            --entrypoint "" \
            dsarchive/superpixel:latest \
            huggingface-cli download MahmoodLab/UNI

        Additionally, make sure that your `docker-compose.override.yml` file includes something like

          services:
            worker:
              environment:
                GIRDER_WORKER_DOCKER_RUN_OPTIONS: >-
                  {"volumes":
                    ["/path_to_home_directory/.cache/huggingface/hub:/root/.cache/huggingface/hub"],
                    "environment": {"HF_HUB_OFFLINE": "1"}}
        """

        # pretrained=True needed to load UNI weights.  init_values need to be passed in to
        # successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
        model = timm.create_model(
            'hf-hub:MahmoodLab/UNI',
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )
        transform = timm.data.transforms_factory.create_transform(
            **timm.data.resolve_data_config(model.pretrained_cfg, model=model),
        )
        model.eval()
        self.UNI_model = model
        self.UNI_transform = transform

    def initializeCreateFeatureFromPatchAndMask(self):
        # This SuperpixelClassificationTorch implementation supports both the Simple and
        # UNI approaches.
        if self.feature_is_image:
            self.initializeCreateFeatureFromPatchAndMaskSimple()
        else:
            self.initializeCreateFeatureFromPatchAndMaskUNI()

    def createFeatureFromPatchAndMaskUNI(self, patch, mask, maskvals):
        return self.createFeatureListFromPatchAndMaskListUNI([patch], [mask], [maskvals])[0]

    def createFeatureListFromPatchAndMaskListUNI(self, patch_list, mask_list, maskvals_list):
        # As a first step, black out all pixels that are not part of the interior or border of the
        # superpixel, exactly as we do with for the simple approach.
        # Numpy order of dimensions is (element, height, width, channel) EHWC.  Torch order is ECHW.
        patch_stack = torch.stack(
            [
                torch.tensor(patch, dtype=torch.float)
                for patch in self.createFeatureListFromPatchAndMaskListSimple(
                    patch_list, mask_list, maskvals_list,
                )
            ],
            dim=0,
        ).permute(0, 3, 1, 2)
        # print(f'{patch_stack.shape = }', flush=True)
        # Image resizing and normalization (ImageNet parameters).
        # TODO: Is this scaling?  We should be centering and cropping.
        patch_stack = self.UNI_transform(patch_stack)
        # print(f'{patch_stack.shape = }', flush=True)
        with torch.inference_mode():
            feature_stack = self.UNI_model(patch_stack)
            # print(f'{feature_stack.shape = }', flush=True)
        feature_list = list(torch.unbind(feature_stack, dim=0))
        return feature_list

    def createFeatureFromPatchAndMask(self, patch, mask, maskvals):
        # This SuperpixelClassificationTorch implementation supports both the Simple and
        # UNI approaches.
        if self.feature_is_image:
            feature = self.createFeatureFromPatchAndMaskSimple(patch, mask, maskvals)
        else:
            feature = self.createFeatureFromPatchAndMaskUNI(patch, mask, maskvals)
        return feature

    def createFeatureListFromPatchAndMaskList(self, patch_list, mask_list, maskvals_list):
        # This SuperpixelClassificationTorch implementation supports both the Simple and
        # UNI approaches.
        if self.feature_is_image:
            feature_list = self.createFeatureListFromPatchAndMaskListSimple(
                patch_list, mask_list, maskvals_list,
            )
        else:
            feature_list = self.createFeatureListFromPatchAndMaskListUNI(
                patch_list, mask_list, maskvals_list,
            )
        return feature_list

    def trainModelDetails(
        self,
        record,
        annotationName: str,
        batchSize: int,
        epochs: int,
        itemsAndAnnot,
        prog: ProgressHelper,
        tempdir: str,
        trainingSplit: float,
    ):
        # make model
        num_classes: int = len(record['labels'])
        model: torch.nn.Module
        if self.feature_is_image:
            # Feature is patch
            if self.certainty == 'batchbald':
                model = _BayesianPatchTorchModel(num_classes)
            else:
                mesg = 'Expected torch model for input of type image to be Bayesian'
                raise ValueError(mesg)
        else:
            # Feature is vector
            input_dim: int = record['ds'].shape[1]
            if self.certainty == 'batchbald':
                model = _BayesianVectorTorchModel(input_dim, num_classes)
            else:
                model = _VectorTorchModel(input_dim, num_classes)
        model.to(model.device)

        # print(f'Torch trainModelDetails(batchSize={batchSize}, ...)')
        # Make a data set and a data loader for each of training and validation
        count: int = len(record['ds'])
        # Split data into training and validation.  H5py requires that indices be
        # sorted.
        train_size: int = int(count * trainingSplit)
        shuffle: NDArray[np.int_] = np.random.permutation(count)  # TODO: add seed=123?
        train_indices: NDArray[np.int_] = np.sort(shuffle[0:train_size])
        val_indices: NDArray[np.int_] = np.sort(shuffle[train_size:count])
        train_arg1: torch.Tensor
        train_arg2: torch.Tensor
        val_arg1: torch.Tensor
        val_arg2: torch.Tensor
        train_ds: torch.utils.data.TensorDataset
        val_ds: torch.utils.data.TensorDataset
        train_dl: torch.utils.data.DataLoader
        val_dl: torch.utils.data.DataLoader
        train_arg1 = (
            torch.from_numpy(record['ds'][train_indices].transpose((0, 3, 2, 1)))
            if self.feature_is_image
            else torch.from_numpy(record['ds'][train_indices])
        )
        train_arg2 = torch.from_numpy(record['labelds'][train_indices])
        val_arg1 = (
            torch.from_numpy(record['ds'][val_indices].transpose((0, 3, 2, 1)))
            if self.feature_is_image
            else torch.from_numpy(record['ds'][val_indices])
        )
        val_arg2 = torch.from_numpy(record['labelds'][val_indices])
        train_ds = torch.utils.data.TensorDataset(train_arg1, train_arg2)
        val_ds = torch.utils.data.TensorDataset(val_arg1, val_arg2)
        if batchSize < 1:
            batchSize = self.findOptimalBatchSize(model, train_ds, training=True)
            print(f'Optimal batch size for training (device = {model.device}) = {batchSize}')
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batchSize)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batchSize)
        prog.progress(0.2)

        prog.message('Training model')
        prog.progress(0)
        history = self.fitModel(
            model, train_dl, val_dl, epochs, callbacks=[_LogTorchProgress(prog, epochs)],
        )

        prog.message('Saving model')
        prog.progress(0)
        modelPath: str = os.path.join(
            tempdir,
            '%s Model Epoch %d.pth' % (annotationName, self.getCurrentEpoch(itemsAndAnnot)),
        )
        self.saveModel(model, modelPath)
        return history, modelPath

    def fitModel(
        self,
        model: torch.nn.Module,
        train_dl: torch.utils.data.DataLoader,
        val_dl: torch.utils.data.DataLoader,
        epochs: int,
        callbacks,
    ) -> Any:
        model.train()  # Tell torch we will be training
        criterion = torch.nn.functional.nll_loss
        optimizer = torch.optim.Adam(model.parameters())

        # TODO: Should training use as many bayesian samples as prediction does?
        num_training_samples: int = model.bayesian_samples if self.certainty == 'batchbald' else 1
        num_validation_samples: int = model.bayesian_samples if self.certainty == 'batchbald' else 1
        # Loop over the dataset multiple times
        epoch: int
        for epoch in range(epochs):
            for cb in callbacks:
                cb.on_epoch_begin(epoch, logs=dict())
            train_loss: float = 0.0
            train_size: int = 0
            train_correct: float = 0.0
            for batch, data in enumerate(train_dl):
                for cb in callbacks:
                    cb.on_train_batch_begin(batch, logs=dict())
                inputs: torch.Tensor
                labels: torch.Tensor
                inputs, labels = data
                inputs = inputs.to(model.device)
                labels = labels.to(model.device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = (
                    model(inputs, num_training_samples)
                    if self.certainty == 'batchbald'
                    else model(inputs)
                )
                if len(outputs.shape) == 2:
                    # Add a middle dimension, giving shape=(batch_size, 1, num_classes).
                    outputs = outputs.unsqueeze(1)
                outputs = torch.nn.functional.log_softmax(outputs, dim=-1)
                assert len(outputs.shape) == 3
                # outputs.shape == (batch_size, num_training_samples, num_classes).
                # labels.shape  == (batch_size).
                # Broadcast labels to the same shape as outputs.shape[0:2]
                labels = labels[:, None].expand(*outputs.shape[0:2])
                criterion_loss = criterion(
                    outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1),
                )
                criterion_loss.backward()
                optimizer.step()
                new_size: int = inputs.shape[0]
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
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    outputs = (
                        model(inputs, num_validation_samples)
                        if self.certainty == 'batchbald'
                        else model(inputs)
                    )
                    if len(outputs.shape) == 2:
                        outputs = outputs.unsqueeze(1)
                    outputs = torch.nn.functional.log_softmax(outputs, dim=-1)
                    assert len(outputs.shape) == 3
                    # outputs.shape == (batch_size, num_validation_samples, num_classes).
                    # labels.shape  == (batch_size).
                    # Broadcast labels to the same shape as outputs.shape[0:2]
                    labels = labels[:, None].expand(*outputs.shape[0:2])
                    criterion_loss = criterion(
                        outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1), reduction='sum',
                    )
                    new_size = inputs.shape[0]
                    validation_size += new_size
                    new_loss = criterion_loss.item() * new_size
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
        history: Sequence[Any] = []  # TODO: Perhaps return something meaningful?
        return history

    def predictLabelsForItemDetails(
        self, batchSize: int, ds_h5, item, model: torch.nn.Module, prog: ProgressHelper,
    ):
        # print(f'Torch predictLabelsForItemDetails(batchSize={batchSize}, ...)')
        num_superpixels: int = ds_h5.shape[0]
        # print(f'{num_superpixels = }')
        bayesian_samples: int = model.bayesian_samples if self.certainty == 'batchbald' else 1
        # print(f'{bayesian_samples = }')
        num_classes: int = model.num_classes
        # print(f'{num_classes = }')

        callbacks = [
            _LogTorchProgress(prog, 1 + (num_superpixels - 1) // batchSize, 0.05, 0.35, item),
        ]
        logs = (
            dict(
                num_superpixels=num_superpixels,
                bayesian_samples=bayesian_samples,
                num_classes=num_classes,
            )
            if self.certainty == 'batchbald'
            else dict(num_superpixels=num_superpixels, num_classes=num_classes)
        )
        for cb in callbacks:
            cb.on_predict_begin(logs=logs)

        ds: torch.utils.data.TensorDataset = torch.utils.data.TensorDataset(
            (
                torch.from_numpy(np.array(ds_h5).transpose((0, 3, 2, 1)))
                if self.feature_is_image
                else torch.from_numpy(np.array(ds_h5))
            ),
        )
        if batchSize < 1:
            batchSize = self.findOptimalBatchSize(model, ds, training=False)
            print(f'Optimal batch size for prediction (device = {model.device}) = {batchSize}')
        dl: torch.utils.data.DataLoader = torch.utils.data.DataLoader(ds, batch_size=batchSize)
        predictions: NDArray[np.float_] = np.zeros((num_superpixels, bayesian_samples, num_classes))
        catWeights: NDArray[np.float_] = np.zeros((num_superpixels, bayesian_samples, num_classes))
        with torch.no_grad():
            model.eval()  # Tell torch that we will be doing predictions
            row: int = 0
            for i, data in enumerate(dl):
                for cb in callbacks:
                    cb.on_predict_batch_begin(i)
                inputs = data[0]
                new_row = row + inputs.shape[0]
                inputs = inputs.to(model.device)
                # print(f'inputs[{i}].shape = {inputs.shape}')
                predictions_raw = (
                    model(inputs, bayesian_samples)
                    if self.certainty == 'batchbald'
                    else model(inputs)
                )
                if len(predictions_raw.shape) == 2:
                    # Add a middle dimension of size 1
                    predictions_raw = predictions_raw.unsqueeze(1)
                # softmax to scale to 0 to 1.
                catWeights_raw = torch.nn.functional.softmax(predictions_raw, dim=-1)
                predictions[row:new_row, :, :] = predictions_raw.detach().cpu().numpy()
                catWeights[row:new_row, :, :] = catWeights_raw.detach().cpu().numpy()
                row = new_row
                for cb in callbacks:
                    cb.on_predict_batch_end(i)
        for cb in callbacks:
            cb.on_predict_end({'outputs': predictions})
        prog.item_progress(item, 0.4)
        # scale to units
        return catWeights, predictions

    def findOptimalBatchSize(
        self, model: torch.nn.Module, ds: torch.utils.data.TensorDataset, training: bool,
    ) -> int:
        if training and self.training_optimal_batchsize is not None:
            return self.training_optimal_batchsize
        if not training and self.prediction_optimal_batchsize is not None:
            return self.prediction_optimal_batchsize
        # Find an optimal batch_size
        maximum_batchSize: int = 2 * ds.tensors[0].shape[0] - 1
        batchSize: int = 2
        # We are using a value greater than 0.0 for add_seconds so that small imprecise
        # timings for small batch sizes don't accidentally trip the time check.
        add_seconds: float = 0.05
        previous_time: float = 1e100
        while batchSize <= maximum_batchSize:
            try:
                dl: torch.utils.data.DataLoader
                dl = torch.utils.data.DataLoader(ds, batch_size=batchSize)
                start_time = time.time()
                with torch.no_grad():
                    model.eval()  # Tell torch that we will be doing predictions
                    data: Sequence[torch.Tensor] = next(iter(dl))
                    inputs: torch.Tensor = data[0]
                    inputs = inputs.to(model.device)
                    if self.certainty == 'batchbald':
                        model(inputs, model.bayesian_samples)
                    else:
                        model(inputs)
                elapsed_time = time.time() - start_time
                if elapsed_time > 2 * previous_time + add_seconds:
                    batchSize //= 2
                    return self.cacheOptimalBatchSize(batchSize, model, training)
                previous_time = elapsed_time
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    batchSize //= 2
                    return self.cacheOptimalBatchSize(batchSize, model, training)
                else:
                    raise e
            batchSize *= 2
        # Undo the last doubling; it was spurious
        batchSize //= 2
        return self.cacheOptimalBatchSize(batchSize, model, training)

    def cacheOptimalBatchSize(self, batchSize: int, model: torch.nn.Module, training: bool) -> int:
        if training:
            self.training_optimal_batchsize = batchSize
        else:
            self.prediction_optimal_batchsize = batchSize
        return batchSize

    def add_safe_globals(self):
        try:
            # If torch is new enough to recognize this command then the command is necessary, at
            # least for torch.load().
            torch.serialization.add_safe_globals(
                [
                    _BayesianPatchTorchModel,
                    _BayesianVectorTorchModel,
                    _VectorTorchModel,
                    torch.nn.Conv2d,
                    torch.nn.functional.log_softmax,
                    torch.nn.functional.max_pool2d,
                    torch.nn.functional.nll_loss,
                    torch.nn.functional.relu,
                    torch.nn.functional.softmax,
                    torch.nn.Linear,
                    torch.nn.Module,
                ],
            )
        except Exception:
            pass

    def loadModel(self, modelPath):
        self.add_safe_globals()
        model = torch.load(modelPath)
        model.eval()
        return model

    def saveModel(self, model, modelPath):
        self.add_safe_globals()
        torch.save(model, modelPath)
