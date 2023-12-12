import torch.optim as optim


PYTORCH_OPTIMIZERS = {name: getattr(optim, name) for name in dir(optim) if isinstance(getattr(optim, name), type) and issubclass(getattr(optim, name), optim.Optimizer)}
