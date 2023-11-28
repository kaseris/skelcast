import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from skelcast.models import SkelcastModule
from skelcast.data.dataset import NTURGBDCollateFn, NTURGBDSample
from skelcast.callbacks.console import ConsoleCallback
from skelcast.callbacks.checkpoint import CheckpointCallback
from skelcast.logger.base import BaseLogger

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
                 device: str = 'cpu',
                 checkpoint_dir: str = None,
                 checkpoint_frequency: int = 1,
                 logger: BaseLogger = None) -> None:
        self.train_set = train_set
        self.val_set = val_set
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.block_size = block_size
        self._collate_fn = NTURGBDCollateFn(block_size=self.block_size, is_packed=True)
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
        
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency
        assert os.path.exists(self.checkpoint_dir), f'The designated checkpoint directory `{self.checkpoint_dir}` does not exist.'
        self.checkpoint_callback = CheckpointCallback(checkpoint_dir=self.checkpoint_dir,
                                                      frequency=self.checkpoint_frequency)
        self.logger = logger

    def setup(self):
        self.model.to(self.device)
        self._total_train_batches = len(self.train_set) // self.train_batch_size
        self._total_val_batches = len(self.val_set) // self.val_batch_size
        self.console_callback.final_epoch = self.n_epochs
        self.console_callback.training_batches = self._total_train_batches
        self.console_callback.validation_batches = self._total_val_batches


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
            self.logger.add_scalar(tag='train/epoch_loss', scalar_value=epoch_loss, global_step=epoch)
            self.training_loss_history.append(epoch_loss)
            for val_batch_idx, val_batch in enumerate(self.val_loader):
                self.validation_step(val_batch=val_batch)
                self.console_callback.on_batch_end(batch_idx=val_batch_idx,
                                                   loss=self.validation_loss_per_step[-1],
                                                   phase='val')
            epoch_loss = sum(self.validation_loss_per_step[epoch * self._total_val_batches:(epoch + 1) * self._total_val_batches]) / self._total_val_batches
            self.console_callback.on_epoch_end(epoch=epoch, epoch_loss=epoch_loss, phase='val')
            self.validation_loss_history.append(epoch_loss)
            self.checkpoint_callback.on_epoch_end(epoch=epoch, runner=self)
            self.logger.add_scalar(tag='val/epoch_loss', scalar_value=epoch_loss, global_step=epoch)

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
        # Log it to the logger
        if self.logger is not None:
            self.logger.add_scalar(tag='train/step_loss', scalar_value=loss.item(), global_step=len(self.training_loss_per_step))

    def validation_step(self, val_batch: NTURGBDSample):
        x, y = val_batch.x, val_batch.y
        # Cast them to a torch float32 and move them to the gpu
        x, y = x.to(torch.float32), y.to(torch.float32)
        x, y = x.to(self.device), y.to(self.device)

        out = self.model.validation_step(x, y)
        loss = out['loss']
        self.validation_loss_per_step.append(loss.item())
        # Log it to the logger
        if self.logger is not None:
            self.logger.add_scalar(tag='val/step_loss', scalar_value=loss.item(), global_step=len(self.validation_loss_per_step))
    
    def resume(self, checkpoint_path):
        """
        Resumes training from a saved checkpoint.

        Args:
        
        - checkpoint_path: Path to the checkpoint file.
        """
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Restore the previous' epoch's state
        self.model.load_state_dict(checkpoint.get('model_state_dict'))
        self.optimizer.load_state_dict(checkpoint.get('optimizer_state_dict'))
        self.training_loss_history = checkpoint.get('training_loss_history')
        self.validation_loss_history = checkpoint.get('validation_loss_history')
        self.training_loss_per_step = checkpoint.get('training_loss_per_step', [])
        self.validation_loss_per_step = checkpoint.get('validation_loss_per_step', [])
        
        # Set the current epoch to the loaded epoch and start from the next
        start_epoch = checkpoint.get('epoch', 0) + 1
        
        # resume the training
        for epoch in range(start_epoch, self.n_epochs):
            self.console_callback.on_epoch_start(epoch=epoch)
            for train_batch_idx, train_batch in enumerate(self.train_loader):
                self.training_step(train_batch=train_batch)
                self.console_callback.on_batch_end(batch_idx=train_batch_idx,
                                                   loss=self.training_loss_per_step[-1],
                                                   phase='train')
                if self.logger is not None:
                    self.logger.add_scalar(tag='train/step_loss', scalar_value=self.training_loss_per_step[-1], global_step=len(self.training_loss_per_step))
            epoch_loss = sum(self.training_loss_per_step[epoch * self._total_train_batches:(epoch + 1) * self._total_train_batches]) / self._total_train_batches
            self.console_callback.on_epoch_end(epoch=epoch,
                                               epoch_loss=epoch_loss, phase='train')
            self.training_loss_history.append(epoch_loss)
            if self.logger is not None:
                self.logger.add_scalar(tag='train/epoch_loss', scalar_value=epoch_loss, global_step=epoch)
            for val_batch_idx, val_batch in enumerate(self.val_loader):
                self.validation_step(val_batch=val_batch)
                self.console_callback.on_batch_end(batch_idx=val_batch_idx,
                                                   loss=self.validation_loss_per_step[-1],
                                                   phase='val')
                if self.logger is not None:
                    self.logger.add_scalar(tag='val/step_loss', scalar_value=self.validation_loss_per_step[-1], global_step=len(self.validation_loss_per_step))
            epoch_loss = sum(self.validation_loss_per_step[epoch * self._total_val_batches:(epoch + 1) * self._total_val_batches]) / self._total_val_batches
            self.console_callback.on_epoch_end(epoch=epoch, epoch_loss=epoch_loss, phase='val')
            self.validation_loss_history.append(epoch_loss)
            if self.logger is not None:
                self.logger.add_scalar(tag='val/epoch_loss', scalar_value=epoch_loss, global_step=epoch)
            self.checkpoint_callback.on_epoch_end(epoch=epoch, runner=self)

        return {
            'training_loss_history': self.training_loss_history,
            'training_loss_per_step': self.training_loss_per_step,
            'validation_loss_history': self.validation_loss_history,
            'validation_loss_per_step': self.validation_loss_per_step
        }
        