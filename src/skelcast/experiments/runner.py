import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from skelcast.data.dataset import NTURGBDCollateFn


class Runner:
    def __init__(self,
                 train_set,
                 val_set,
                 train_batch_size,
                 val_batch_size,
                 block_size,
                 model: nn.Module,
                 optimizer=None,
                 n_epochs: int = 10) -> None:
        self.train_set = train_set
        self.val_set = val_set
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.block_size = block_size
        self._collate_fn = NTURGBDCollateFn(block_size=self.block_size)
        self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.train_batch_size, shuffle=True, collate_fn=self._collate_fn)
        self.val_loader = DataLoader(dataset=self.val_set, batch_size=self.val_batch_size, shuffle=False, collate_fn=self._collate_fn)
        self.model = model

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)

        self.training_loss_history = []
        self.training_loss_per_step = []
        self.validation_loss_history = []
        self.validation_loss_per_step = []

        self.n_epochs = n_epochs
        self._total_train_batches = len(self.train_set) // self.train_batch_size
        self._total_val_batches = len(self.val_set) // self.val_batch_size

        self._status_message = ''

    def setup(self):
        pass

    def fit(self):
        for epoch in range(self.n_epochs):
            self._status_message = ''
            self._status_message += f'\rEpoch: {epoch + 1}/{self.n_epochs}'
            for train_batch_idx, train_batch in enumerate(self.train_loader):
                self._status_message += f' - Training Batch: {train_batch_idx + 1}/{self._total_train_batches}'
                self.training_step()
                sys.stdout.write(self._status_message)
                sys.stdout.flush()
            self._status_message = ''
            self._status_message += f'\rEpoch: {epoch + 1}/{self.n_epochs}'
            for val_batch_idx, val_bathc in enumerate(self.val_loader):
                self._status_message += f' - Validation Batch: {val_batch_idx + 1}/{self._total_val_batches}'
                self.validation_step()
                sys.stdout.write(self._status_message)
                sys.stdout.flush()

    def training_step(self):
        pass

    def validation_step(self):
        pass
    