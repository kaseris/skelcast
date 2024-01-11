__version__ = '0.0.1'

from .core.accel import Accelerator

accel = Accelerator()

available_devices = accel.available_devices