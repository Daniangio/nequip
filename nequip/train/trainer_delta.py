""" Nequip.train.trainer_delta

"""

import sys
import inspect
import logging
from copy import deepcopy
from os.path import isfile
from time import perf_counter
from typing import Callable, Optional, Union, Tuple, List
from pathlib import Path

if sys.version_info[1] >= 7:
    import contextlib
else:
    # has backport of nullcontext
    import contextlib2 as contextlib

import numpy as np
import torch
from torch_ema import ExponentialMovingAverage
from torch.utils.data import ConcatDataset

from nequip.data import DataLoader, AtomicData, AtomicDataDict, ReferenceConcatDataset
from nequip.utils import (
    Output,
    Config,
    instantiate_from_cls_name,
    instantiate,
    save_file,
    load_file,
    load_callable,
    atomic_write,
    finish_all_writes,
    atomic_write_group,
    dtype_from_name,
)
from nequip.utils.versions import check_code_version
from nequip.model import model_from_config
from nequip.train.trainer import Trainer

from .loss import Loss, LossStat
from .metrics import Metrics
from ._key import ABBREV, LOSS_KEY, TRAIN, VALIDATION
from .early_stopping import EarlyStopping


class TrainerDelta(Trainer):


    @property
    def init_keys(self):
        return [
            key
            for key in list(inspect.signature(TrainerDelta.__init__).parameters.keys())
            if key not in (["self", "kwargs", "model"] + TrainerDelta.object_keys)
        ]


    def train(self):

        """Training"""
        if getattr(self, "dl_train", None) is None:
            raise RuntimeError("You must call `set_dataset()` before calling `train()`")
        if not self._initialized:
            self.init()

        for callback in self._init_callbacks:
            callback(self)

        self.init_log()
        self.wall = perf_counter()
        self.previous_cumulative_wall = self.cumulative_wall

        with atomic_write_group():
            if self.iepoch == -1:
                self.save()
            if self.iepoch in [-1, 0]:
                self.save_config()

        self.init_metrics()

        while not self.stop_cond:

            self.epoch_step()
            self.end_of_epoch_save()

        for callback in self._final_callbacks:
            callback(self)

        self.final_log()

        self.save()
        finish_all_writes()

    def batch_step(self, data, ref_data, validation=False):
        # no need to have gradients from old steps taking up memory
        self.optim.zero_grad(set_to_none=True)

        if validation:
            self.model.eval()
        else:
            self.model.train()

        # Do any target rescaling
        data, ref_data = data.to(self.torch_device), ref_data.to(self.torch_device)
        data, ref_data = AtomicData.to_AtomicDataDict(data), AtomicData.to_AtomicDataDict(ref_data)
        dataset_idcs = data[AtomicDataDict.DATASET_INDEX_KEY]

        data_unscaled = data
        ref_data_unscaled = ref_data
        for layer in self.rescale_layers:
            # This means that self.model is RescaleOutputs
            # this will normalize the targets
            # in validation (eval mode), it does nothing
            # in train mode, if normalizes the targets
            data_unscaled = layer.unscale(data_unscaled)
            ref_data_unscaled = layer.unscale(ref_data_unscaled)

        # Run model
        # We make a shallow copy of the input dict in case the model modifies it
        input_data = {
            k: v
            for k, v in data_unscaled.items()
            if k not in self._remove_from_model_input
        }
        out = self.model(input_data)
        del input_data

        ref_input_data = {
            k: v
            for k, v in ref_data_unscaled.items()
            if k not in self._remove_from_model_input
        }
        ref_out = self.model(ref_input_data)
        del ref_input_data

        # Apply deltas over specified output keys
        for key in self.loss.delta_keys:
            out[key] -= ref_out[key][dataset_idcs]
            data_unscaled[key] -= ref_data_unscaled[key][dataset_idcs]

        # If we're in evaluation mode (i.e. validation), then
        # data_unscaled's target prop is unnormalized, and out's has been rescaled to be in the same units
        # If we're in training, data_unscaled's target prop has been normalized, and out's hasn't been touched, so they're both in normalized units
        # Note that either way all normalization was handled internally by RescaleOutput

        if not validation:
            # Actually do an optimization step, since we're training:
            loss, loss_contrib = self.loss(pred=out, ref=data_unscaled)
            # see https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
            self.optim.zero_grad(set_to_none=True)
            loss.backward()

            # See https://stackoverflow.com/a/56069467
            # Has to happen after .backward() so there are grads to clip
            if self.max_gradient_norm < float("inf"):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_gradient_norm
                )

            self.optim.step()

            if self.use_ema:
                self.ema.update()

            if self.lr_scheduler_name == "CosineAnnealingWarmRestarts":
                self.lr_sched.step(self.iepoch + self.ibatch / self.n_batches)

        with torch.no_grad():
            if validation:
                scaled_out = out
                _data_unscaled = data
                for layer in self.rescale_layers:
                    # loss function always needs to be in normalized unit
                    scaled_out = layer.unscale(scaled_out, force_process=True)
                    _data_unscaled = layer.unscale(_data_unscaled, force_process=True)
                loss, loss_contrib = self.loss(pred=scaled_out, ref=_data_unscaled)
            else:
                # If we are in training mode, we need to bring the prediction
                # into real units
                for layer in self.rescale_layers[::-1]:
                    out = layer.scale(out, force_process=True)

            # save metrics stats
            self.batch_losses = self.loss_stat(loss, loss_contrib)
            # in validation mode, data is in real units and the network scales
            # out to be in real units interally.
            # in training mode, data is still in real units, and we rescaled
            # out to be in real units above.
            self.batch_metrics = self.metrics(pred=out, ref=data)


    def epoch_step(self):

        dataloaders = {TRAIN: self.dl_train, VALIDATION: self.dl_val}
        ref_dataloaders = {TRAIN: self.dl_ref_train, VALIDATION: self.dl_ref_val}
        categories = [TRAIN, VALIDATION] if self.iepoch >= 0 else [VALIDATION]
        dataloaders = [
            dataloaders[c] for c in categories
        ]
        ref_dataloaders = [
            ref_dataloaders[c] for c in categories
        ]  # get the right dataloaders for the catagories we actually run
        self.metrics_dict = {}
        self.loss_dict = {}

        for category, dataset, ref_dataset in zip(categories, dataloaders, ref_dataloaders):
            if category == VALIDATION and self.use_ema:
                cm = self.ema.average_parameters()
            else:
                cm = contextlib.nullcontext()

            with cm:
                self.reset_metrics()
                self.n_batches = len(dataset)
                for self.ibatch, (batch, ref_batch) in enumerate(zip(dataset, ref_dataset)):
                    self.batch_step(
                        data=batch,
                        ref_data=ref_batch,
                        validation=(category == VALIDATION),
                    )
                    self.end_of_batch_log(batch_type=category)
                    for callback in self._end_of_batch_callbacks:
                        callback(self)
                self.metrics_dict[category] = self.metrics.current_result()
                self.loss_dict[category] = self.loss_stat.current_result()

                if category == TRAIN:
                    for callback in self._end_of_train_callbacks:
                        callback(self)

        self.iepoch += 1

        self.end_of_epoch_log()

        # if the iepoch for the past epoch was -1, it will now be 0
        # for -1 (report_init_validation: True) we aren't training, so it's wrong
        # to step the LR scheduler even if it will have no effect with this particular
        # scheduler at the beginning of training.
        if self.iepoch > 0 and self.lr_scheduler_name == "ReduceLROnPlateau":
            self.lr_sched.step(metrics=self.mae_dict[self.metrics_key])

        for callback in self._end_of_epoch_callbacks:
            callback(self)


    def set_dataset(
        self,
        dataset: ConcatDataset,
        ref_dataset: ReferenceConcatDataset,
        validation_dataset: Optional[ConcatDataset] = None,
        ref_validation_dataset: Optional[ReferenceConcatDataset] = None,
    ) -> None:
        """Set the dataset(s) used by this trainer.

        Training and validation datasets will be sampled from
        them in accordance with the trainer's parameters.

        If only one dataset is provided, the train and validation
        datasets will both be sampled from it. Otherwise, if
        `validation_dataset` is provided, it will be used.
        """

        assert len(self.n_train) == len(dataset.datasets)
        assert len(self.n_val) == len(validation_dataset.datasets) if validation_dataset is not None else len(dataset.datasets)

        if self.train_idcs is None or self.val_idcs is None:
            self.train_idcs, self.val_idcs = [], []
            if validation_dataset is not None:
                for _validation_dataset, n_val in zip(validation_dataset.datasets, self.n_val):
                    total_n = len(_validation_dataset)
                    if n_val > total_n:
                        raise ValueError(
                            "too little data for validation. please reduce n_val"
                        )
                    if self.train_val_split == "random":
                        idcs = torch.randperm(total_n, generator=self.dataset_rng)
                    elif self.train_val_split == "sequential":
                        idcs = torch.arange(total_n)
                    else:
                        raise NotImplementedError(
                            f"splitting mode {self.train_val_split} not implemented"
                        )
                    self.val_idcs.append(idcs[:n_val])

            # If validation_dataset is None, Sample both from `dataset`
            for _dataset, n_train, n_val in zip(dataset.datasets, self.n_train, self.n_val):
                total_n = len(_dataset)
                if validation_dataset is None and (n_train + n_val) > total_n:
                    raise ValueError(
                        "too little data for training and validation. please reduce n_train and n_val"
                    )
                if n_train > total_n:
                    raise ValueError(
                        "too little data for training. please reduce n_train"
                    )

                if self.train_val_split == "random":
                    idcs = torch.randperm(total_n, generator=self.dataset_rng)
                elif self.train_val_split == "sequential":
                    idcs = torch.arange(total_n)
                else:
                    raise NotImplementedError(
                        f"splitting mode {self.train_val_split} not implemented"
                    )

                self.train_idcs.append(idcs[: n_train])
                if validation_dataset is None:
                    self.val_idcs.append(idcs[n_train : n_train + n_val])
        if validation_dataset is None:
            validation_dataset = dataset
            ref_validation_dataset = ref_dataset

        # torch_geometric datasets inherantly support subsets using `index_select`
        indexed_datasets_train = []
        for _dataset, train_idcs in zip(dataset.datasets, self.train_idcs):
            indexed_datasets_train.append(_dataset.index_select(train_idcs))
        self.dataset_train = ConcatDataset(indexed_datasets_train)
        self.ref_dataset_train = ref_dataset
        
        indexed_datasets_val = []
        for _dataset, val_idcs in zip(validation_dataset.datasets, self.val_idcs):
            indexed_datasets_val.append(_dataset.index_select(val_idcs))
        self.dataset_val = ConcatDataset(indexed_datasets_val)
        self.ref_dataset_val = ref_validation_dataset

        # based on recommendations from
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-async-data-loading-and-augmentation
        dl_kwargs = dict(
            exclude_keys=self.exclude_keys,
            num_workers=self.dataloader_num_workers,
            # keep stuff around in memory
            persistent_workers=(
                self.dataloader_num_workers > 0 and self.max_epochs > 1
            ),
            # PyTorch recommends this for GPU since it makes copies much faster
            pin_memory=(self.torch_device != torch.device("cpu")),
            # avoid getting stuck
            timeout=(10 if self.dataloader_num_workers > 0 else 0),
            # use the right randomness
            generator=self.dataset_rng,
        )
        self.dl_train = DataLoader(
            dataset=self.dataset_train,
            shuffle=self.shuffle,  # training should shuffle
            batch_size=self.batch_size,
            **dl_kwargs,
        )
        # validation, on the other hand, shouldn't shuffle
        # we still pass the generator just to be safe
        self.dl_val = DataLoader(
            dataset=self.dataset_val,
            batch_size=self.validation_batch_size,
            **dl_kwargs,
        )

        # reference dataloaders don't shuffle and pick always a batch of
        # n_datasets samples, one for each dataset
        dl_ref_kwargs = dict(
            batch_size=self.ref_dataset_train.n_datasets,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.torch_device != torch.device("cpu")),
        )
        self.dl_ref_train = DataLoader(dataset=self.ref_dataset_train, **dl_ref_kwargs)
        dl_ref_kwargs['batch_size'] = self.ref_dataset_val.n_datasets
        self.dl_ref_val = DataLoader(dataset=self.ref_dataset_val, **dl_ref_kwargs)
