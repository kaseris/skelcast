import abc
import torch.nn as nn


class SkelcastModule(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super(SkelcastModule, self).__init__()

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """
        Implements the prediction step of a module
        """
        pass

    @abc.abstractmethod
    def training_step(self, *args, **kwargs):
        """
        Implements a training step of a module
        """
        pass

    @abc.abstractmethod
    def validation_step(self, *args, **kwargs):
        """
        Implements a validation step of a module
        """
        pass
