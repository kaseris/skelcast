import torch

from typing import Tuple

from skelcast.metrics import METRICS
from skelcast.metrics.metric import Metric


@METRICS.register_module()
class MeanPerJointPositionError(Metric):  # Inherits from our abstract Metric class
    def __init__(self, keep_time_dim: bool = True):
        self.keep_time_dim = keep_time_dim
        self.reset()  # Initialize/reset the state

    def reset(self):
        # Reset the state of the metric
        self.y = None
        self.y_pred = None

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        y_pred, y = output  # Unpack the output tuple, assuming output is already in the desired format

        # Initialize or update the stored tensors
        if self.y is None:
            self.y = y
            self.y_pred = y_pred
        else:
            self.y = torch.cat([self.y, y], dim=0)
            self.y_pred = torch.cat([self.y_pred, y_pred], dim=0)

    def result(self):
        # Compute the Mean Per Joint Position Error
        if self.y is None:
            raise ValueError('MeanPerJointPositionError must have at least one example before it can be computed.')

        error = (self.y - self.y_pred).norm(dim=-1)  # Calculate the L2 norm over the last dimension (joints)
        mean_error = error.mean(dim=[0, 2])  # Take the mean over the batch and time dimensions

        if not self.keep_time_dim:
            mean_error = mean_error.mean()  # Further reduce mean over all joints if time dimension is not kept

        return mean_error