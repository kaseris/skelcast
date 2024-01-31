import numpy as np
import torch
import torch.nn as nn

from skelcast.data.human36m.quaternion import qeuler
from skelcast.losses import LOSSES


@LOSSES.register_module()
class EulerAngleLoss(nn.Module):
    def __init__(self, order="xyz", reduction="mean"):
        super(EulerAngleLoss, self).__init__()
        self._order = order
        self._reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        # Check the shape of predictions and targets
        assert (
            predictions.shape == targets.shape
        ), f"Predictions and targets must have the same shape."
        assert (
            predictions.shape[-1] == 3
        ), f"Predictions and targets must have 3 channels in the last dimension."

        predicted_euler = qeuler(predictions, self._order, epsilon=1e-6)
        angle_distance = (
            torch.remainder(predicted_euler - targets + np.pi, 2 * np.pi) - np.pi
        )
        return torch.mean(torch.abs(angle_distance))
