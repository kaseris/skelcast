import sys
import time
from datetime import datetime

from skelcast.callbacks.callback import Callback


class ConsoleCallback(Callback):
    def __init__(self):
        self.current_epoch = 0
        self.latest_train_loss = "N/A"
        self.latest_val_loss = "N/A"
        self.current_batch = "N/A"
        self.final_epoch = 0
        self.validation_batches = 0
        self.training_batches = 0
        self.total_batches = 0

    def on_epoch_start(self, epoch):
        self.current_epoch = epoch
        # self.latest_train_loss = "N/A"  # Reset at the start of each epoch
        # self.latest_val_loss = "N/A"

    def on_batch_start(self):
        pass

    def on_batch_end(self, batch_idx, loss, phase):
        self.current_batch = batch_idx + 1  # Update current batch index

        if phase == 'train':
            self.latest_train_loss = f"{loss:.4f}"
            self.total_batches = self.training_batches
        elif phase == 'val':
            self.latest_val_loss = f"{loss:.4f}"
            self.total_batches = self.validation_batches
        
        self._print_status()

    def on_epoch_end(self, epoch, epoch_loss, phase):
        if phase == 'train':
            self.latest_train_loss = f"{epoch_loss:.4f}"
        elif phase == 'val':
            self.latest_val_loss = f"{epoch_loss:.4f}"
        
        self._print_status()

        if epoch == self.final_epoch - 1 and phase == 'val':
            print()

    def _print_status(self):
        now = datetime.now()
        now_formatted = now.strftime("[%Y-%m-%d %H:%M:%S]")
        clear_line = '\r' + ' ' * 80  # Create a line of 80 spaces
        message = f"{now_formatted} Epoch: {self.current_epoch + 1}/{self.final_epoch}, Batch: {self.current_batch}/{self.total_batches}, Train Loss: {self.latest_train_loss}, Val Loss: {self.latest_val_loss}"
        
        # First, print the clear_line to overwrite the previous output, then print your message
        print(f'{clear_line}\r{message}', end='')
        sys.stdout.flush()


if __name__ == '__main__':
    import random
    cb = ConsoleCallback()
    

    n_epochs = 100
    cb.final_epoch = n_epochs
    for epoch in range(n_epochs):
        cb.on_epoch_start(epoch=epoch)
        for batch_idx, train_batch in enumerate(range(100)):
            phase = 'train'
            loss = random.random()
            time.sleep(0.001)
            cb.on_batch_end(batch_idx=batch_idx, loss=loss, phase=phase)
        cb.on_epoch_end(epoch=epoch, epoch_loss=random.random(), phase='train')
        
        for batch_idx, val_batch in enumerate(range(100)):
            phase = 'val'
            loss = random.random()
            time.sleep(0.001)
            cb.on_batch_end(batch_idx=batch_idx, loss=loss, phase=phase)
        cb.on_epoch_end(epoch=epoch, epoch_loss=random.random(), phase='val')