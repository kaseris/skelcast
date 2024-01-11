import torch
import platform

class Accelerator:
    
    def __init__(self):
        self._available_devices = ['cpu']
        os = platform.system()

        if os == 'Linux':
            if torch.cuda.is_available():
                self._available_devices.append('cuda')
        elif os == 'Darwin':  # macOS is identified as 'Darwin'
            if torch.backends.mps.is_available():
                self._available_devices.append('mps')
        else:
            raise NotImplementedError(f"OS {os} not supported.")

    @property
    def available_devices(self):
        """Returns the list of available devices."""
        return self._available_devices
    

class Device:

    def __init__(self, accel: Accelerator):
        self._accel = accel
        self._device_cache = {}

    def _get_device(self, device_type):
        """Returns a torch device object, using cache for efficiency."""
        if device_type not in self._device_cache:
            if device_type in self._accel.available_devices:
                self._device_cache[device_type] = torch.device(device_type)
            else:
                self._device_cache[device_type] = self.cpu()
        return self._device_cache[device_type]

    def cpu(self):
        """Returns a CPU torch device."""
        return torch.device('cpu')

    def cuda(self):
        """Returns a CUDA torch device if available, otherwise CPU."""
        return self._get_device('cuda')
        
    def mps(self):
        """Returns an MPS torch device if available, otherwise CPU."""
        return self._get_device('mps')
