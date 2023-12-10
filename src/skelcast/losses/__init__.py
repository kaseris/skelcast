from skelcast.core.registry import Registry

LOSSES = Registry()

from .logloss import LogLoss
from .torch_losses import PYTORCH_LOSSES