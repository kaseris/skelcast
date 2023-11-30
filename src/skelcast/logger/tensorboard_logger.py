from torch.utils.tensorboard import SummaryWriter
from skelcast.logger.base import BaseLogger

from skelcast.logger import LOGGERS


@LOGGERS.register_module()
class TensorboardLogger(BaseLogger):
    def __init__(self, log_dir):
        super().__init__()
        self.writer = SummaryWriter(log_dir)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        self.writer.add_scalar(tag, scalar_value, global_step, walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step, walltime)

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None):
        self.writer.add_histogram(tag, values, global_step, bins, walltime, max_bins)

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        self.writer.add_image(tag, img_tensor, global_step, walltime, dataformats)

    # Implement other methods from SummaryWriter as needed

    def close(self):
        self.writer.close()
