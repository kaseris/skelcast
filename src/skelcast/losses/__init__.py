from skelcast.core.registry import Registry

LOSSES = Registry()

from .logloss import LogLoss
from .euler_angle_loss import EulerAngleLoss
