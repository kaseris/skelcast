import os

import torch
import torch.distributed as dist

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

from skelcast.callbacks.console import ConsoleCallback
from skelcast.callbacks.checkpoint import CheckpointCallback
from skelcast.logger.base import BaseLogger
from skelcast.models import SkelcastModule
from skelcast.experiments import RUNNERS

@RUNNERS.register_module()
class DistributedRunner:
    
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
                 checkpoint_dir: str = None,
                 checkpoint_frequency: int = 1,
                 logger: BaseLogger = None,
                 log_gradient_info: bool = False,
                 collate_fn = None
                 ) -> None:
        
        self.train_set = train_set
        self.val_set = val_set
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.block_size = block_size
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.train_sampler = DistributedSampler(self.train_set)
        self.val_sampler = DistributedSampler(self.val_set)
        self.train_loader = DataLoader(self.train_set, batch_size=self.train_batch_size, sampler=self.train_sampler)
        self.val_loader = DataLoader(self.val_set, batch_size=self.val_batch_size, sampler=self.val_sampler)
        self.model = DistributedDataParallel(self.model, device_ids=[self.device])
        self.lr = lr

        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = optimizer

        self.training_loss_history = []
        self.training_loss_per_step = []
        self.validation_loss_history = []
        self.validation_loss_per_step = []

        self.n_epochs = n_epochs

        self._status_message = ''

        self.console_callback = ConsoleCallback()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency
        assert os.path.exists(self.checkpoint_dir), f'The designated checkpoint directory `{self.checkpoint_dir}` does not exist.'
        self.checkpoint_callback = CheckpointCallback(checkpoint_dir=self.checkpoint_dir,
                                                      frequency=self.checkpoint_frequency)
        self.logger = logger
        self.log_gradient_info = log_gradient_info
    
    def setup(self):
        init_process_group(backend='nccl')
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
        destroy_process_group()
        return self._compile_results()

    def training_step(self, train_batch):
        x, y, mask = train_batch.x, train_batch.y, train_batch.mask
        # Cast them to a torch float32 and move them to the gpu
        # TODO: Handle the mask None case
        x, y, mask = x.to(torch.float32), y.to(torch.float32), mask.to(torch.float32)
        x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
        self.model.train()
        out = self.model.training_step(x=x, y=y, mask=mask) # TODO: Make the other models accept a mask as well
        loss = out['loss']
        outputs = out['out']
        # Calculate the saturation of the tanh output
        saturated = (outputs.abs() > 0.95)
        saturation_percentage = saturated.sum(dim=(1, 2)).float() / (outputs.size(1) * outputs.size(2)) * 100
        # Calculate the dead neurons
        dead_neurons = (outputs.abs() < 0.05)
        dead_neurons_percentage = dead_neurons.sum(dim=(1, 2)).float() / (outputs.size(1) * outputs.size(2)) * 100
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

            if self.logger is not None:
                self.logger.add_scalar(tag='train/saturation', scalar_value=saturation_percentage.mean().item(), global_step=len(self.training_loss_per_step))
                self.logger.add_scalar(tag='train/dead_neurons', scalar_value=dead_neurons_percentage.mean().item(), global_step=len(self.training_loss_per_step))

        self.optimizer.step()
        # Print the loss
        self.training_loss_per_step.append(loss.item())
        # Log it to the logger
        if self.logger is not None:
            self.logger.add_scalar(tag='train/step_loss', scalar_value=loss.item(), global_step=len(self.training_loss_per_step))

    def validation_step(self, val_batch):
        x, y, mask = val_batch.x, val_batch.y, val_batch.mask
        # Cast them to a torch float32 and move them to the gpu
        x, y, mask = x.to(torch.float32), y.to(torch.float32), mask.to(torch.float32)
        x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
        self.model.eval()
        out = self.model.validation_step(x=x, y=y, mask=mask)
        loss = out['loss']
        self.validation_loss_per_step.append(loss.item())
        # Log it to the logger
        if self.logger is not None:
            self.logger.add_scalar(tag='val/step_loss', scalar_value=loss.item(), global_step=len(self.validation_loss_per_step))
        