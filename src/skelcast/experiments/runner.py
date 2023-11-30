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
    """
    A training and validation runner for models in the Skelcast framework.

    This class handles the setup, training, validation, and checkpointing of SkelcastModule models. 
    It uses datasets for training and validation, and includes functionality for batch processing, 
    gradient logging, and checkpoint management.

    Args:
    ---
    -    `train_set` (Dataset): The dataset for training.
    -    `val_set` (Dataset): The dataset for validation.
    -    `train_batch_size` (int): Batch size for the training dataset.
    -    `val_batch_size` (int): Batch size for the validation dataset.
    -    `block_size` (int): Block size used for collating batch data.
    -    `model` (SkelcastModule): The model to be trained and validated.
    -    `optimizer` (torch.optim.Optimizer): Optimizer for model training.
    -    `n_epochs` (int): Number of epochs to train the model.
    -    `device` (str): The device ('cpu' or 'cuda') on which to run the model.
    -    `checkpoint_dir` (str): Directory to save checkpoints.
    -    `checkpoint_frequency` (int): Frequency (in epochs) at which to save checkpoints.
    -    `logger` (BaseLogger): Logger for recording training and validation metrics.
    -    `log_gradient_info` (bool): Flag to determine if gradient information is logged.

    Methods:
    ---
    -    `setup()`: Prepares the runner for training and validation.
    -    `fit()`: Starts the training process from epoch 0.
    -    `resume(checkpoint_path)`: Resumes training from a saved checkpoint.
    -    `training_step(train_batch)`: Executes a single training step.
    -    `validation_step(val_batch)`: Executes a single validation step.
    -    `_run_epochs(start_epoch)`: Runs training and validation for specified epochs.
    -    `_run_phase(phase, epoch)`: Runs a training or validation phase for a single epoch.
    -    `_log_epoch_loss(phase, epoch)`: Logs the loss for a completed epoch.
    -    `_restore_state(checkpoint)`: Restores the state of the model and optimizer from a checkpoint.
    -    `_compile_results()`: Compiles and returns training and validation results.

    Note:
    ---
        - This class requires a properly formatted SkelcastModule model and corresponding datasets.
        - The checkpoint directory must exist before initializing the Runner.
        - Logging and checkpointing are optional and can be configured as needed.

    Raises:
    ---
        `AssertionError`: If the checkpoint directory does not exist.
    """
    def __init__(self,
                 train_set: Dataset,
                 val_set: Dataset,
                 train_batch_size: int,
                 val_batch_size: int,
                 block_size: int,
                 model: SkelcastModule,
                 optimizer: torch.optim.Optimizer = None,
                 lr: float = 1e-4,
                 n_epochs: int = 10,
                 device: str = 'cpu',
                 checkpoint_dir: str = None,
                 checkpoint_frequency: int = 1,
                 logger: BaseLogger = None,
                 log_gradient_info: bool = False) -> None:
        self.train_set = train_set
        self.val_set = val_set
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.block_size = block_size
        self._collate_fn = NTURGBDCollateFn(block_size=self.block_size, is_packed=True)
        self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.train_batch_size, shuffle=True, collate_fn=self._collate_fn)
        self.val_loader = DataLoader(dataset=self.val_set, batch_size=self.val_batch_size, shuffle=False, collate_fn=self._collate_fn)
        self.model = model
        self.lr = lr

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

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
        self.log_gradient_info = log_gradient_info

    def setup(self):
        self.model.to(self.device)
        self._total_train_batches = len(self.train_set) // self.train_batch_size
        self._total_val_batches = len(self.val_set) // self.val_batch_size
        self.console_callback.final_epoch = self.n_epochs
        self.console_callback.training_batches = self._total_train_batches
        self.console_callback.validation_batches = self._total_val_batches

    def _run_epochs(self, start_epoch):
        for epoch in range(start_epoch, self.n_epochs):
            self.console_callback.on_epoch_start(epoch=epoch)
            self._run_phase('train', epoch)
            self._log_epoch_loss('train', epoch)
            self._run_phase('val', epoch)
            self._log_epoch_loss('val', epoch)
            self.checkpoint_callback.on_epoch_end(epoch=epoch, runner=self)

    def _run_phase(self, phase, epoch):
        loader = self.train_loader if phase == 'train' else self.val_loader
        step_method = self.training_step if phase == 'train' else self.validation_step
        loss_per_step = self.training_loss_per_step if phase == 'train' else self.validation_loss_per_step

        for batch_idx, batch in enumerate(loader):
            step_method(batch)
            self.console_callback.on_batch_end(batch_idx=batch_idx,
                                               loss=loss_per_step[-1],
                                               phase=phase)
            
    def _log_epoch_loss(self, phase, epoch):
        loss_per_step = self.training_loss_per_step if phase == 'train' else self.validation_loss_per_step
        total_batches = self._total_train_batches if phase == 'train' else self._total_val_batches
        epoch_loss = sum(loss_per_step[epoch * total_batches:(epoch + 1) * total_batches]) / total_batches
        self.console_callback.on_epoch_end(epoch=epoch, epoch_loss=epoch_loss, phase=phase)
        history = self.training_loss_history if phase == 'train' else self.validation_loss_history
        history.append(epoch_loss)
        self.logger.add_scalar(tag=f'{phase}/epoch_loss', scalar_value=epoch_loss, global_step=epoch)

    def resume(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self._restore_state(checkpoint)
        start_epoch = checkpoint.get('epoch', 0) + 1
        self._run_epochs(start_epoch)
        return self._compile_results()
    
    def _restore_state(self, checkpoint):
        self.model.load_state_dict(checkpoint.get('model_state_dict'))
        self.optimizer.load_state_dict(checkpoint.get('optimizer_state_dict'))
        self.training_loss_history = checkpoint.get('training_loss_history')
        self.validation_loss_history = checkpoint.get('validation_loss_history')
        self.training_loss_per_step = checkpoint.get('training_loss_per_step', [])
        self.validation_loss_per_step = checkpoint.get('validation_loss_per_step', [])

    def _compile_results(self):
        return {
            'training_loss_history': self.training_loss_history,
            'training_loss_per_step': self.training_loss_per_step,
            'validation_loss_history': self.validation_loss_history,
            'validation_loss_per_step': self.validation_loss_per_step
        }

    def fit(self):
        self._run_epochs(start_epoch=0)
        return self._compile_results()

    def training_step(self, train_batch: NTURGBDSample):
        x, y = train_batch.x, train_batch.y
        # Cast them to a torch float32 and move them to the gpu
        x, y = x.to(torch.float32), y.to(torch.float32)
        x, y = x.to(self.device), y.to(self.device)
        self.model.train()
        out = self.model.training_step(x, y)
        loss = out['loss']
        self.optimizer.zero_grad()
        loss.backward()
        if self.log_gradient_info:
        # Get the gradient flow and update norm ratio
            self.model.gradient_flow()
            self.model.compute_gradient_update_norm(lr=self.optimizer.param_groups[0]['lr'])
            grad_hists = self.model.get_gradient_histograms()
            # Log the gradient histograms to the logger
            if self.logger is not None:
                for name, hist in grad_hists.items():
                    self.logger.add_histogram(tag=f'gradient/hists/{name}_grad_hist', values=hist, global_step=len(self.training_loss_per_step))

            # Log the gradient updates to the logger
            if self.logger is not None:
                for name, ratio in self.model.gradient_update_ratios.items():
                    self.logger.add_scalar(tag=f'gradient/{name}_grad_update_norm_ratio', scalar_value=ratio, global_step=len(self.training_loss_per_step))

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
        self.model.eval()
        out = self.model.validation_step(x, y)
        loss = out['loss']
        self.validation_loss_per_step.append(loss.item())
        # Log it to the logger
        if self.logger is not None:
            self.logger.add_scalar(tag='val/step_loss', scalar_value=loss.item(), global_step=len(self.validation_loss_per_step))
        