import torch
from typing import Tuple

from skelcast.data import TRANSFORMS


@TRANSFORMS.register_module()
class MinMaxScaleTransform:

    def __init__(self, feature_scale: Tuple[float, float]) -> None:
        assert isinstance(feature_scale, tuple), '`feature_scale` must be a tuple.'
        self.min_, self.max_ = feature_scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f'`x` must be a 4-dimensional tensor. Found {x.ndim} dimension(s) instead.'

        # Pre-allocate tensors for min and max values for each axis
        min_vals = torch.empty(3, dtype=x.dtype, device=x.device)
        max_vals = torch.empty(3, dtype=x.dtype, device=x.device)

        # Calculate min and max for each axis
        for axis in range(3):
            min_vals[axis] = torch.amin(x[..., axis])
            max_vals[axis] = torch.amax(x[..., axis])

        # Scale each component of the last dimension separately
        scale = (self.max_ - self.min_) / (max_vals - min_vals)
        for axis in range(3):
            x[..., axis] = (x[..., axis] - min_vals[axis]) * scale[axis] + self.min_

        return x
