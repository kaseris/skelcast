import abc
import torch.nn as nn


class SkelcastModule(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super(SkelcastModule, self).__init__()
        self.gradients = dict()
        self.gradient_update_ratios = dict()

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
    
    def from_pretrained(model_path=None):
        """
        Implements a method to load a pretrained model
        """
        pass

    def gradient_flow(self):
        """
        Implements the gradient flow step of a module
        """
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.gradients[name] = param.grad.clone().detach().cpu().numpy()

    def compute_gradient_update_norm(self, lr: float):
        """
        Computes the ratio of the parameter update to the parameter norm and stores it to the gradient_update_ratios
        dictionary. The gradient update is approximated as the vanilla gradient descent update.

        Args:
        -    lr (float): The optimizer's learning rate
        """
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.gradient_update_ratios[name] = (lr * param.grad.norm() / param.norm()).detach().cpu().numpy()

    def get_gradient_histograms(self):
        """
        Returns the flat gradients of the module's parameters from the gradients dictionary.
        """
        return {name: param.grad.clone().view(-1).detach().cpu().numpy() for name, param in self.named_parameters() if
                param.requires_grad and param.grad is not None}
    