import torch
import torch.nn as nn

from skelcast.models import SkelcastModule

class SimpleLSTMRegressor(SkelcastModule):
    def __init__(self,
                 hidden_size,
                 linear_out,
                 num_layers,
                 batch_first: bool = True,
                 num_bodies: int = 1,
                 n_joints: int = 25,
                 n_dims: int = 3) -> None:
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
        self.criterion = nn.MSELoss(reduction='sum')
    
    def forward(self, x: torch.Tensor,
                y: torch.Tensor = None):
        assert x.ndim == 5, f'`x` must be a 5-dimensional tensor. Found {x.ndim} dimension(s).'
        batch_size, context_size, num_bodies, n_joints, n_dims = x.shape
        assert num_bodies == self.num_bodies, f'The number of bodies in the position 2 of the tensor is {num_bodies}, but it should be {self.num_bodies}'
        assert n_joints == self.n_joints, f'The number of bodies in the position 3 of the tensor is {n_joints}, but it should be {self.n_joints}'
        assert n_dims == self.n_dims, f'The number of bodies in the position 3 of the tensor is {n_dims}, but it should be {self.n_dims}'

        x = x.view(batch_size, context_size, num_bodies * n_joints * n_dims)
        x = self.linear_transform(x)
        out, _ = self.lstm(x)
        out = self.linear(out)
        out = self.relu(out)
        if y is not None:
            y = y.view(batch_size, context_size, num_bodies * n_joints * n_dims)
            loss = self.criterion(out, y)
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

