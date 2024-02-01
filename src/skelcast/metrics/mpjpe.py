import torch

from typing import Tuple

from skelcast.metrics import METRICS
from skelcast.metrics.metric import Metric


@METRICS.register_module()
class MeanPerJointPositionError(Metric):
    def __init__(self, keep_time_dim: bool = True):
        self.keep_time_dim = keep_time_dim
        self.reset()

    def reset(self):
        # Reset the state of the metric
        self.y = torch.tensor([])
        self.y_pred = torch.tensor([])

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        y_pred, y = output

        # Concatenate new predictions and targets to the stored tensors
        if self.y.numel() == 0:
            self.y = y
            self.y_pred = y_pred
        else:
            self.y = torch.cat([self.y, y], dim=0)
            self.y_pred = torch.cat([self.y_pred, y_pred], dim=0)

    def result(self):
        # Compute the Mean Per Joint Position Error
        if self.y.numel() == 0:
            raise ValueError('MeanPerJointPositionError must have at least one example before it can be computed.')

        error = (self.y - self.y_pred).norm(dim=-1)
        mean_error = error.mean(dim=[0, 2])
        
        if not self.keep_time_dim:
            mean_error = mean_error.mean()

        return mean_error