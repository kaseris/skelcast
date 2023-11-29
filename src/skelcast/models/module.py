import abc
import torch.nn as nn


class SkelcastModule(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super(SkelcastModule, self).__init__()
        self.gradients = dict()

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

    def gradient_flow(self, *args, **kwargs):
        """
        Implements the gradient flow step of a module
        """
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.gradients[name] = param.grad.clone()
