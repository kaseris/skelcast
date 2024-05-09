import torch
import torch.nn as nn

from typing import Tuple

from skelcast.models import MODELS


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, activation: str = "relu"):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x


@MODELS.register_module()
class TFlowDiag(nn.Module):
    def __init__(self,
                 skeleton_dim: int = 48,
                 embed_dim: int = 64,
                 z_dim: int = 128,
                 n_samples: int = 10,
                 encoder_tf_nhead: int = 8,
                 encoder_tf_dim_feedforward: int = 64,
                 encoder_tf_num_layers: int = 1,
                 input_shape_config: str = 'tbd'):
        super().__init__()
        
        self.skeleton_dim = skeleton_dim
        self.embed_dim = embed_dim
        self.z_dim = z_dim
        self.n_samples = n_samples
        self.encoder_tf_nhead = encoder_tf_nhead
        self.encoder_tf_dim_feedforward = encoder_tf_dim_feedforward
        self.encoder_tf_num_layers = encoder_tf_num_layers
        self.input_shape_config = input_shape_config

        self.project_input = nn.Linear(in_features=self.skeleton_dim, out_features=self.embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=self.encoder_tf_nhead,
                                                   dim_feedforward=self.encoder_tf_dim_feedforward,
                                                   activation='gelu',
                                                   batch_first=True)
        
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.encoder_tf_num_layers)
        
        self.mlp = MLP(in_dim=self.embed_dim, hidden_dim=self.embed_dim, out_dim=self.embed_dim)

        self.head_A = nn.Linear(in_features=self.embed_dim, out_features=self.z_dim * self.n_samples)
        self.head_b = nn.Linear(in_features=self.embed_dim, out_features=self.z_dim * self.n_samples)

    def encode_x(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.project_input(x)
        h_x = self.encoder(x)
        return h_x[:, -1, :], h_x
    
    def forward(self, x: torch.Tensor, z: torch.Tensor = None):
        if self.input_shape_config == 'tbd':
            x = x.permute(1, 0, 2)
        h_x, _ = self.encode_x(x)
        if z is None:
            z = torch.randn((h_x.shape[0] * self.n_samples, self.z_dim), device=x.device)
        
        h = self.mlp(h_x)
        a = self.head_A(h).view(-1, self.z_dim)
        b = self.head_b(h).view(-1, self.z_dim)
        a = torch.exp(0.5 * a)
        y = a * z + b
        return y, a, b
    
    def get_kl(self, a, b):
        var = a**2
        KLD = -0.5 * torch.sum(1 + var.log() - b.pow(2) - var)
        return KLD
    
    def sample(self, x, z=None):
        return self.forward(x, z)[0]