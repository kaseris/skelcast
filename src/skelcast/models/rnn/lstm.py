import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

from typing import Union

from skelcast.models import SkelcastModule

class SimpleLSTMRegressor(SkelcastModule):
    def __init__(self,
                 hidden_size,
                 linear_out,
                 num_layers,
                 batch_first: bool = True,
                 num_bodies: int = 1,
                 n_joints: int = 25,
                 n_dims: int = 3,
                 reduction: str = 'mean') -> None:
        super().__init__()
        self.num_bodies = num_bodies
        self.n_joints = n_joints
        self.n_dims = n_dims
        input_size = self.num_bodies * self.n_joints * self.n_dims
        self.linear_transform = nn.Linear(in_features=input_size, out_features=linear_out)
        self.lstm = nn.LSTM(input_size=linear_out, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=batch_first)
        self.linear = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.relu = nn.ReLU()
        self.criterion = nn.MSELoss(reduction=reduction)
    
    def forward(self, x: Union[torch.Tensor, PackedSequence],
                y: Union[torch.Tensor, PackedSequence] = None):
        if isinstance(x, torch.Tensor):
            assert x.ndim == 5, f'`x` must be a 5-dimensional tensor. Found {x.ndim} dimension(s).'
            batch_size, context_size, n_dim = x.shape
            assert n_dim == self.num_bodies * self.n_joints * self.n_dims, f'The number of bodies in the position 2 of the tensor is {n_dim}, but it should be {self.num_bodies * self.n_joints * self.n_dims}'
        else:
            assert x.data.ndim == 3, f'`x` must be a 5-dimensional tensor. Found {x.ndim} dimension(s).'
            batch_size, context_size, n_dim = x.data.shape
            assert n_dim == self.num_bodies * self.n_joints * self.n_dims, f'The number of bodies in the position 2 of the tensor is {n_dim}, but it should be {self.num_bodies * self.n_joints * self.n_dims}'

        x = self.linear_transform(x if isinstance(x, torch.Tensor) else x.data)
        out, _ = self.lstm(x)
        out = self.linear(out if isinstance(out, torch.Tensor) else out.data)
        out = self.relu(out if isinstance(out, torch.Tensor) else out.data)
        if y is not None:
            loss = self.criterion(out if isinstance(out, torch.Tensor) else out.data,
                                  y if isinstance(y, torch.Tensor) else y.data)
            return out, loss
        return out

    def training_step(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        out, loss = self(x, y)
        return {'out': out, 'loss': loss}
    
    @torch.no_grad()
    def validation_step(self, x, y) -> dict:
        out, loss = self(x, y)
        return {'out': out, 'loss': loss}
    

    @torch.no_grad()
    def predict(self):
        pass

