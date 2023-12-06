import torch
import torch.nn as nn

from skelcast.losses import LOSSES


@LOSSES.register_module()
class LogLoss(nn.Module):
    """
    A custom loss function module in PyTorch that computes a modified logarithmic loss.

    Parameters:
    - alpha (float): A scaling factor for the difference between predictions and ground truth.
    - beta (float): An exponent factor for scaling the difference.
    - use_abs (bool): If True, uses the absolute value of the difference. Default is True.
    - reduction (str): Specifies the reduction to apply to the output: 'mean', 'sum'. Default is 'mean'.
    - batch_first (bool): If True, expects the batch size to be the first dimension of input tensors. Default is True.

    The forward method computes the loss given predictions and ground truth values.

    Usage:
    >>> loss_fn = LogLoss(alpha=1.0, beta=2.0, use_abs=True, reduction='mean', batch_first=True)
    >>> y_pred = torch.tensor([[0.2, 0.4], [0.6, 0.8]], dtype=torch.float32)
    >>> y_true = torch.tensor([[0.1, 0.3], [0.5, 0.7]], dtype=torch.float32)
    >>> loss = loss_fn(y_pred, y_true)
    """
    def __init__(self,  alpha, beta, use_abs=True, reduction='mean', batch_first=True) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.use_abs = use_abs
        if reduction not in ['mean', 'sum']:
            self.reduction = 'mean'
        else:
            self.reduction = reduction
        self.batch_first = batch_first

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_pred.shape[-1] == y_true.shape[-1], f'Predictions and ground truth labels must have the same dimensionality.'
        if self.batch_first:
            assert y_pred.shape[1] == y_true.shape[1], f'Predictions and ground truth labels must have the same context length.'
        else:
            assert y_pred.shape[0] == y_true.shape[0], f'Predictions and ground truth labels must have the same context length.'
        diff_ = y_pred - y_true
        if self.use_abs:
            diff_ = torch.abs(diff_)
        
        result = (torch.log(1.0 + self.alpha * diff_ ** self.beta))**2
        if self.reduction == 'mean':
            result = result.mean()
        elif self.reduction == 'sum':
            result = result.sum()
        return result
    