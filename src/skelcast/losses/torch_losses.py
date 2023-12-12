import torch.nn as nn

PYTORCH_LOSSES = {name: getattr(nn, name) for name in dir(nn) if isinstance(getattr(nn, name), type) and issubclass(getattr(nn, name), nn.Module) and 'Loss' in name}
