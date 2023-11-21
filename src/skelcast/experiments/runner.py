import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from skelcast.models import SkelcastModule
from skelcast.data.dataset import NTURGBDCollateFn, NTURGBDSample
from skelcast.callbacks.console import ConsoleCallback


class Runner:
    def __init__(self,
                 train_set: Dataset,
                 val_set: Dataset,
                 train_batch_size: int,
                 val_batch_size: int,
                 block_size: int,
                 model: SkelcastModule,
                 optimizer: torch.optim.Optimizer = None,
                 n_epochs: int = 10,
                 device: str = 'cpu') -> None:
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

        self._status_message = ''

        if device != 'cpu':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.console_callback = ConsoleCallback()

    def setup(self):
        self.model.to(self.device)
        self._total_train_batches = len(self.train_set) // self.train_batch_size
        self._total_val_batches = len(self.val_set) // self.val_batch_size
        self.console_callback.final_epoch = self.n_epochs


    def fit(self):
        for epoch in range(self.n_epochs):
            self.console_callback.on_epoch_start(epoch=epoch)
            for train_batch_idx, train_batch in enumerate(self.train_loader):
                self.training_step(train_batch=train_batch)
                self.console_callback.on_batch_end(batch_idx=train_batch_idx,
                                                   loss=self.training_loss_per_step[-1],
                                                   phase='train')
            epoch_loss = sum(self.training_loss_per_step[epoch * self._total_train_batches:(epoch + 1) * self._total_train_batches]) / self._total_train_batches
            self.console_callback.on_epoch_end(epoch=epoch,
                                               epoch_loss=epoch_loss, phase='train')
            self.training_loss_history.append(epoch_loss)
            for val_batch_idx, val_batch in enumerate(self.val_loader):
                self.validation_step(val_batch=val_batch)
                self.console_callback.on_batch_end(batch_idx=val_batch_idx,
                                                   loss=self.validation_loss_per_step[-1],
                                                   phase='val')
            epoch_loss = sum(self.validation_loss_per_step[epoch * self._total_val_batches:(epoch + 1) * self._total_val_batches]) / self._total_val_batches
            self.console_callback.on_epoch_end(epoch=epoch, epoch_loss=epoch_loss, phase='val')
            self.validation_loss_history.append(epoch_loss)

        return {
            'training_loss_history': self.training_loss_history,
            'training_loss_per_step': self.training_loss_per_step,
            'validation_loss_history': self.validation_loss_history,
            'validation_loss_per_step': self.validation_loss_per_step
        }

    def training_step(self, train_batch: NTURGBDSample):
        x, y = train_batch.x, train_batch.y
        # Cast them to a torch float32 and move them to the gpu
        x, y = x.to(torch.float32), y.to(torch.float32)
        x, y = x.to(self.device), y.to(self.device)

        out = self.model.training_step(x, y)
        loss = out['loss']
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Print the loss
        self.training_loss_per_step.append(loss.item())

    def validation_step(self, val_batch: NTURGBDSample):
        x, y = val_batch.x, val_batch.y
        # Cast them to a torch float32 and move them to the gpu
        x, y = x.to(torch.float32), y.to(torch.float32)
        x, y = x.to(self.device), y.to(self.device)

        out = self.model.validation_step(x, y)
        loss = out['loss']
        self.validation_loss_per_step.append(loss.item())
    