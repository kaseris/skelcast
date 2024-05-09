import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, Union, Optional, List

from skelcast.models import MODELS


def generate_square_subsequent_mask(sz) -> torch.Tensor:
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)

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

class Encoder(nn.Module):
    """
    An Encoder module designed for sequence encoding using Transformer architecture, particularly tailored for motion capture data.
    This class encapsulates the necessary components to encode input sequences of skeleton data into a latent space representation
    which can be further used in generative tasks.

    Parameters:
    - skeleton_dim (int): Dimensionality of the skeleton data in each frame. Default is 48.
    - embed_dim (int): Dimensionality of the embedding layer. Default is 64.
    - z_dim (int): Dimensionality of the latent space. Default is 128.
    - tf_nhead (int): Number of heads in each TransformerEncoderLayer. Default is 8.
    - tf_dim_feedforward (int): Dimensionality of the feed-forward network in each TransformerEncoderLayer. Default is 64.
    - tf_num_layers (int): Number of layers in the Transformer encoder. Default is 1.

    Attributes:
    - projection (nn.Linear): Linear layer that projects input data to the embedding dimension.
    - encoder_x (nn.TransformerEncoder): Transformer encoder for processing one part of the input.
    - encoder_y (nn.TransformerEncoder): Transformer encoder for processing the other part of the input.
    - h_mlp (MLP): Multi-layer perceptron that combines and processes the outputs from encoder_x and encoder_y.
    - mu_layer (nn.Linear): Linear layer that predicts the mean of the latent space distribution.
    - logvar_layer (nn.Linear): Linear layer that predicts the logarithm of the variance of the latent space distribution.

    Methods:
    - forward(x, y): Processes the input through the encoding layers and produces latent variables z, mu, and logvar.
    - encode(x, y): Encodes the input tensors x and y into the latent space parameters mu and logvar.
    - encode_x(x): Encodes the input tensor x using the encoder_x.
    - encode_y(y): Encodes the input tensor y using the encoder_y.
    - reparameterize(mu, logvar): Reparameterizes the encoded mean mu and logvar to produce latent variable z.

    Usage:
    - This encoder is typically used in tasks like variational autoencoders where the generation of new data instances is required
    based on learned distribution parameters.
    """
    def __init__(self,
                 skeleton_dim: int = 48,
                 embed_dim: int = 64,
                 z_dim: int = 128,
                 tf_nhead: int = 8,
                 tf_dim_feedforward: int = 64,
                 tf_num_layers: int = 1,
                 ):
        super().__init__()
        self.skeleton_dim = skeleton_dim
        self.embed_dim = embed_dim
        self.z_dim = z_dim


        self.projection = nn.Linear(in_features=skeleton_dim, out_features=embed_dim)

        encoder_x_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=tf_nhead, dim_feedforward=tf_dim_feedforward,
                                                     activation='gelu', batch_first=True)
        encoder_y_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=tf_nhead, dim_feedforward=tf_dim_feedforward,
                                                     activation='gelu', batch_first=True)
        self.encoder_x = nn.TransformerEncoder(encoder_layer=encoder_x_layer, num_layers=tf_num_layers)
        self.encoder_y = nn.TransformerEncoder(encoder_layer=encoder_y_layer, num_layers=tf_num_layers)
        self.h_mlp = MLP(in_dim=embed_dim * 2, hidden_dim=z_dim, out_dim=z_dim, activation='relu')

        self.mu_layer = nn.Linear(in_features=z_dim, out_features=z_dim)
        self.logvar_layer = nn.Linear(in_features=z_dim, out_features=z_dim)

        self.pos_enc = PositionalEncoding(d_model=skeleton_dim)

    def forward(self, x, y) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y  = self.pos_enc(x), self.pos_enc(y)
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.ndim == 3, f'x must have 3 dimensions. Found {x.ndim} dimensions'
        assert y.ndim == 3, f'y must have 3 dimensions. Found {y.ndim} dimensions'
        h_x, _ = self.encode_x(x)
        h_y = self.encode_y(y)
        assert (h_x.shape[-1] + h_y.shape[-1]) == 2 * self.embed_dim, f'In order to process the concatenation of h_x and h_y, it is required h_x.shape[-1] + h_y.shape[-1] == 2 * embed_dim. Found: lhs: {h_x.shape[-1] + h_y.shape[-1]} rhs: {2 * self.embed_dim}'
        assert h_x.ndim == 2, f'h_x must be a tensor of 2 dimensions. Found {h_x.ndim} dims'
        assert h_y.ndim == 2, f'h_y must be a tensor of 2 dimensions. Found {h_y.ndim} dims'
        h = torch.cat([h_x, h_y], dim=-1)
        h = self.h_mlp(h)
        return self.mu_layer(h), self.logvar_layer(h)
    
    def encode_x(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[-1] == self.skeleton_dim, f'The last dimension of x must be of shape {self.skeleton_dim}. Found {x.shape[-1]}'
        x = self.projection(x)
        x = self.encoder_x(x)
        return x[:, -1, :], x

    def encode_y(self, y: torch.Tensor) -> torch.Tensor:
        assert y.shape[-1] == self.skeleton_dim, f'The last dimension of y must be of shape {self.skeleton_dim}. Found {y.shape[-1]}'
        y = self.projection(y)
        y = self.encoder_y(y)
        return y[:, -1, :]
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        assert mu.shape[-1] == self.z_dim
        assert logvar.shape[-1] == self.z_dim
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

class Decoder(nn.Module):
    """
    A Decoder module for sequence prediction using a Transformer-based architecture. This module is designed to handle
    the decoding phase in sequence modeling tasks, particularly for applications involving skeleton-based motion prediction.

    Parameters:
    - skeleton_dim (int): Dimensionality of the skeleton representation in the input data. Default is 48.
    - embed_dim (int): Dimensionality of the embedding layer. Default is 64.
    - z_dim (int): Dimensionality of the latent space. Default is 128.
    - horizon (int): The number of future frames to predict. Default is 100.
    - history (int): The number of past frames used as context. Default is 25.
    - tf_nhead (int): Number of heads in the TransformerDecoderLayer. Default is 8.
    - tf_dim_feedforward (int): Dimensionality of the feed-forward network model in TransformerDecoderLayer. Default is 64.
    - tf_num_layers (int): Number of Transformer decoder layers. Default is 1.

    Attributes:
    - decoder_d_model (int): Combined dimension of the latent space and embedding, used in the decoder layers.
    - project_target (nn.Linear): Linear layer to project target skeleton data to decoder model dimension.
    - decoder (nn.TransformerDecoder): Transformer decoder to generate the output sequence.
    - regress (MLP): Multi-layer perceptron to regress the final output to the desired skeleton dimension.

    Methods:
    - forward(encoded_x, z, x, y=None): Processes input through the decoder and generates output sequence. Handles both
      autoregressive and direct decoding depending on the presence of target data `y`.
    - decode(encoded_x, z, x, y=None): Decodes the input context and latent variables into the predicted output sequence.
    - _autoregressive_decode(x, memory): Generates the sequence autoregressively using previously predicted output as the
      next input.
    - _decode(tgt, memory): Direct decoding from target tensor `tgt` and memory from the encoder.
    - prepare_decoder_inputs(encoded_x, z): Prepares combined memory tensor from encoded input and latent variables.
    - prepare_target_tensor(x, y): Prepares the target tensor by concatenating last pose of `x` and all `y`.

    Usage:
    - This module is typically used after an encoding phase where `encoded_x` and `z` are prepared, and aims to predict
      a sequence of future poses given past context `x` and optional direct future targets `y`.
    """
    def __init__(self,
                 skeleton_dim: int = 48,
                 embed_dim: int = 64,
                 z_dim: int = 128,
                 horizon: int = 100,
                 history: int = 25,
                 tf_nhead: int = 8,
                 tf_dim_feedforward: int = 64,
                 tf_num_layers: int = 1
                 ):
        super().__init__()
        self.skeleton_dim = skeleton_dim
        self.embed_dim = embed_dim
        self.z_dim = z_dim
        self.horizon = horizon
        self.history = history

        self.decoder_d_model = z_dim + embed_dim

        self.project_target = nn.Linear(in_features=self.skeleton_dim, out_features=self.decoder_d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.decoder_d_model, nhead=tf_nhead,
                                                   dim_feedforward=tf_dim_feedforward,
                                                   activation='gelu',
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=tf_num_layers)
        self.regress = MLP(in_dim=self.decoder_d_model, hidden_dim=self.skeleton_dim, out_dim=self.skeleton_dim, activation='tanh')

    def decode(self, encoded_x: torch.Tensor, z: torch.Tensor, x: torch.Tensor, y: Union[torch.Tensor, None] = None):
        assert encoded_x.ndim == 3, f'encoded_x must have 3 dims ([batch_size, seq_len, embed_dim]). Found {encoded_x.ndim} dims'
        assert encoded_x.shape[-1] + z.shape[-1] == self.decoder_d_model
        assert x.shape[-1] == self.skeleton_dim, f'input tensor `x` must have same dimensionality as skeleton_dim. Found x dimensionality: {x.shape[-1]}'
        assert x.shape[1] == self.history, f'The dimension at position 1 of the `x` tensor must be {self.history}. Found {x.shape[1]}'
        memory = self.prepare_decoder_inputs(encoded_x=encoded_x, z=z)
        
        if y is None:
            y_r = self._autoregressive_decode(x=x, memory=memory)
        else:
            tgt = self.prepare_target_tensor(x=x, y=y)
            
            y_r = self._decode(tgt=tgt, memory=memory)
        return y_r

    def _autoregressive_decode(self, x: torch.Tensor, memory: torch.Tensor):
        batch_size, seq_len, dim = x.shape
        # placeholder_target = torch.zeros(batch_size, self.horizon + 1, self.skeleton_dim, device=x.device)
        placeholder_target = torch.ones(batch_size, self.horizon + 1, dim).to(x.device)
        placeholder_target[:, 0, :] = x[:, -1, :]
        current_tgt = placeholder_target[:, 0:1, :]  # Start with the last known pose

        for i in range(1, self.horizon + 1):
            projected_tgt = self.project_target(current_tgt)
            decoder_output = self.decoder(tgt=projected_tgt, memory=memory,
                                        tgt_mask=generate_square_subsequent_mask(i).to(x.device))
            next_pose = self.regress(decoder_output[:, -1, :].unsqueeze(1))  # Regress only the last output

            # Detach all except the last step's output to prevent gradient computation
            if i < self.horizon:
                next_pose = next_pose.detach()  # Detach to stop gradients

            current_tgt = torch.cat([current_tgt, next_pose], dim=1)  # Append the predicted pose

        return current_tgt[:, 1:, :]


    
    def _decode(self, tgt: torch.Tensor, memory: torch.Tensor):
        decoder_output = self.decoder(tgt=tgt,
                                      memory=memory,
                                      tgt_mask=generate_square_subsequent_mask(self.horizon + 1).to(tgt.device))
        # We do not care about the first element, since it is the content of the history sequence
        # decoder_output = decoder_output[:, 1:, :]
        y_r = self.regress(decoder_output)[:, 1:, :]
        return y_r


    def prepare_decoder_inputs(self, encoded_x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z = z.unsqueeze(1).repeat(1, self.history, 1)
        memory = torch.cat([encoded_x, z], dim=-1)
        return memory
    
    def prepare_target_tensor(self, x: torch.Tensor, y: torch.Tensor):
        last_pose_of_x = x[:, -1, :].unsqueeze(1)
        tgt = torch.cat([last_pose_of_x, y], dim=1)
        return self.project_target(tgt)

    def forward(self, encoded_x: torch.Tensor, z: torch.Tensor, x: torch.Tensor, y: Union[torch.Tensor, None] = None):
        return self.decode(encoded_x=encoded_x,
                           z=z,
                           x=x,
                           y=y)


@MODELS.register_module()
class TFlowVAE(nn.Module):
    def __init__(self,
                 skeleton_dim: int = 48,
                 embed_dim: int = 64,
                 z_dim: int = 128,
                 encoder_tf_nhead: int = 8,
                 encoder_tf_dim_feedforward: int = 64,
                 tf_encoder_num_layers: int = 1,
                 horizon: int = 100,
                 history: int = 25,
                 decoder_tf_nhead: int = 8,
                 decoder_tf_dim_feedforward: int = 64,
                 decoder_tf_num_layers: int = 1,
                 input_shape_config: str = 'tbd',
                 mode: str = 'train'):
        super().__init__()

        self.skeleton_dim = skeleton_dim
        self.embed_dim = embed_dim
        self.z_dim = z_dim
        self.horizon = horizon
        self.history = history
        self.input_shape_config = input_shape_config
        self.mode = mode

        self.encoder_tf_nhead = encoder_tf_nhead
        self.encoder_tf_dim_feedforward = encoder_tf_dim_feedforward
        self.encoder_tf_num_layers = tf_encoder_num_layers

        self.decoder_tf_nhead = decoder_tf_nhead
        self.decoder_tf_dim_feedforward = decoder_tf_dim_feedforward
        self.decoder_tf_num_layers = decoder_tf_num_layers

        self.encoder = Encoder(skeleton_dim=self.skeleton_dim,
                               embed_dim=self.embed_dim,
                               z_dim=self.z_dim,
                               tf_nhead=self.encoder_tf_nhead,
                               tf_dim_feedforward=self.encoder_tf_dim_feedforward,
                               tf_num_layers=self.encoder_tf_num_layers)
        
        self.decoder = Decoder(skeleton_dim=self.skeleton_dim,
                               embed_dim=self.embed_dim,
                               z_dim=self.z_dim,
                               horizon=self.horizon,
                               history=self.history,
                               tf_nhead=self.decoder_tf_nhead,
                               tf_dim_feedforward=self.decoder_tf_dim_feedforward,
                               tf_num_layers=self.decoder_tf_num_layers)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.input_shape_config == 'tbd':
            x, y = self._permute_inputs(x, y)
        
        z, mu, logvar = self.encoder(x, y)
        _, encoded_x = self.encoder.encode_x(x)
        if self.mode == 'train':
            y_r = self.decoder(encoded_x, z, x, torch.zeros_like(y))
        else:
            y_r = self.decoder(encoded_x, z, x)
        
        if self.input_shape_config == 'tbd':
            # [batch_size, seq_len, dim] -> [seq_len, batch_size, dim]
            y_r = y_r.permute(1, 0, 2)
        return y_r, mu, logvar
    
    def _permute_inputs(*args) -> List[torch.Tensor]:
        out = []
        for _input in args:
            if isinstance(_input, torch.Tensor):
                out.append(_input.permute(1, 0, 2))
        return out


    def decode(self, x: torch.Tensor, z: torch.Tensor,  y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.input_shape_config == 'tbd':
            x = x.permute(1, 0, 2)
        
        _, encoded_x = self.encoder.encode_x(x)
        if y is None:
            bs, _, dim = x.shape
            y_r = self.decoder(encoded_x=encoded_x, z=z, x=x, y=torch.zeros(bs, self.horizon, dim))
        else:
            y_r = self.decoder(encoded_x=encoded_x, z=z, x=x, y=torch.zeros_like(y))

        if self.input_shape_config == 'tbd':
            y_r = y_r.permute(1, 0, 2)
        return y_r
    
    def sample_prior(self, x):
        z = torch.randn((x.shape[1], self.nz), device=x.device)
        return self.decode(x, z)
    
    @property
    def nz(self):
        return self.z_dim