from datetime import datetime

import torch

from skelcast.callbacks.callback import Callback


class CheckpointCallback(Callback):
    def __init__(self, checkpoint_dir: str, frequency: int) -> None:
        """
        Initialize the CheckpointCallback.

        Args:
        - checkpoint_dir: Directory where the checkpoints will be saved.
        - frequency: Frequency of epochs to save the checkpoints. Default is 1, meaning every epoch.
        """
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.frequency = frequency
        
    def on_epoch_start(self):
        pass
    
    def on_epoch_end(self, epoch, runner):
        if epoch % self.frequency == 0:
            self.save_checkpoint(runner=runner, epoch=epoch)
    
    def on_batch_start(self):
        pass
    
    def on_batch_end(self):
        pass
    
    def save_checkpoint(self, runner, epoch: int):
        """
        Saves the checkpoint.
        
        Args:
        
        - runner (Runner): The experiment runner instance
        - epoch (int): The current epoch
        """
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d_%H%M%S")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': runner.model.state_dict(),
            'optimizer_state_dict': runner.optimizer.state_dict(),
            'training_loss_history': runner.training_loss_history,
            'validation_loss_history': runner.validation_loss_history,
            'training_loss_per_step': runner.training_loss_per_step,
            'validation_loss_per_step': runner.validation_loss_per_step
        }
        
        checkpoint_path = f'{self.checkpoint_dir}/checkpoint_epoch_{epoch}_{formatted_time}.pt'
        torch.save(checkpoint, checkpoint_path)
        