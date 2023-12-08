import torch
import torch.nn as nn

from skelcast.models import MODELS, ENCODERS, DECODERS
from skelcast.models.module import SkelcastModule
from skelcast.models.transformers.base import PositionalEncoding


@ENCODERS.register_module()
class Encoder(nn.Module):
    def __init__(self, rnn_type: str = 'rnn',
                 input_dim: int = 75,
                 hidden_dim: int = 256,
                 batch_first: bool = True,
                 dropout: float = 0.2,
                 use_residual: bool = True) -> None:
        super().__init__()
        assert rnn_type in ['lstm', 'gru'], f'rnn_type must be one of rnn, lstm, gru, got {rnn_type}'
        self.rnn_type = rnn_type
        self.batch_first = batch_first
        self.use_residual = use_residual

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=batch_first)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=batch_first)
        self.linear = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, hidden = self.rnn(x)
        out = self.dropout(out)
        out = self.linear(out)
        if self.use_residual:
            out = out + x
        return out, hidden


@DECODERS.register_module()
class Decoder(nn.Module):
    def __init__(self,rnn_type: str = 'rnn',
                 input_dim: int = 75,
                 hidden_dim: int = 256,
                 batch_first: bool = True,
                 dropout: float = 0.2,
                 use_residual: bool = True) -> None:
        super().__init__()
        assert rnn_type in ['lstm', 'gru'], f'rnn_type must be one of rnn, lstm, gru, got {rnn_type}'
        self.rnn_type = rnn_type
        self.batch_first = batch_first
        self.use_residual = use_residual

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=batch_first)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=batch_first)
        self.linear = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None, timesteps_to_predict: int = 5) -> torch.Tensor:
        predictions = []
        for _ in range(timesteps_to_predict):
            out, hidden = self.rnn(x, hidden)
            out = self.dropout(out)
            out = self.linear(out)
            if self.use_residual:
                out = out + x
            predictions.append(out)
            x = out
        return torch.cat(predictions, dim=1)
        

@MODELS.register_module()
class PositionalVelocityRecurrentEncoderDecoder(SkelcastModule):
    """
    Positional Velocity Recurrent Encoder Decoder (PVRED) model.
    The model was proposed in the paper: https://arxiv.org/abs/1906.06514

    The model consists of an encoder and a decoder. The encoder is a recurrent neural network (RNN)
    that encodes the input sequence into a latent representation. The decoder is also an RNN that
    decodes the latent representation into a sequence of the same length as the input sequence.
    
    The authors of PVRED introduce the idea of positional encoding. Below, we describe how can the
    positional encoding be applied to the input sequence.

    Three ways of taking into account the positional encoding:
    1. Concatenate the positional encoding to the input
    2. Add the positional encoding to the input
    3. NOP (no positional encoding) - if pos_enc is None

    Args:
    ---
    - `input_dim` (`int`): Input dimension of the model
    - `enc_hidden_dim` (`int`): Hidden dimension of the encoder
    - `dec_hidden_dim` (`int`): Hidden dimension of the decoder
    - `enc_type` (`str`): Type of the encoder, one of lstm, gru
    - `dec_type` (`str`): Type of the decoder, one of lstm, gru
    - `include_velocity` (`bool`): Flag to indicate whether to include the velocity of the input
    - `pos_enc` (`str`): Type of the positional encoding, one of concat, add, None
    - `loss_fn` (`torch.nn.Module`): Loss function to be used for training
    - `batch_first` (`bool`): Flag to indicate whether the batch dimension is the first dimension
    - `std_thresh` (`float`): Threshold for the standard deviation of the input
    - `use_std_mask` (`bool`): Flag to indicate whether to use the standard deviation mask
    - `use_padded_len_mask` (`bool`): Flag to indicate whether to use the padded length mask
    - `observe_until` (`int`): The number of frames to observe before predicting the future

    Returns:
    ---
    - `dec_out` (`torch.Tensor`): Output of the decoder
    - `loss` (`torch.Tensor`): Loss of the model

    Examples:
    ---
    ```python
    import torch
    from skelcast.models.rnn.pvred import PositionalVelocityRecurrentEncoderDecoder

    model = PositionalVelocityRecurrentEncoderDecoder(input_dim = 75, observe_until=80)
    x = torch.randn(32, 100, 75)
    dec_out, loss = model(x)
    dec_out.shape
    >>> torch.Size([32, 20, 75])
    ```
    """
    def __init__(self, input_dim: int, enc_hidden_dim: int = 64,
                dec_hidden_dim: int = 64,
                enc_type: str = 'lstm',
                dec_type: str = 'lstm',
                include_velocity: bool = False,
                pos_enc: str = None,
                loss_fn: nn.Module = None,
                batch_first: bool = True,
                std_thresh: float = 1e-4,
                use_std_mask: bool = False,
                use_padded_len_mask: bool = False,
                observe_until: int = 30) -> None:
        assert pos_enc in ['concat', 'add', None], f'pos_enc must be one of concat, add, None, got {pos_enc}'
        assert isinstance(loss_fn, nn.Module), f'loss_fn must be an instance of torch.nn.Module, got {type(loss_fn)}'
        assert enc_type in ['lstm', 'gru'], f'enc_type must be one of lstm, gru, got {enc_type}'
        assert dec_type in ['lstm', 'gru'], f'dec_type must be one of lstm, gru, got {dec_type}'
        super().__init__()
        self.input_dim = input_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.enc_type = enc_type
        self.batch_first = batch_first
        self.std_thresh = std_thresh
        self.use_std_mask = use_std_mask
        self.use_padded_len_mask = use_padded_len_mask
        self.observe_until = observe_until

        self.include_velocity = include_velocity
        if self.include_velocity:
            # Double the input dimension because we are concatenating the xyz velocities to the input
            self.input_dim = self.input_dim * 2

        self.pos_enc_method = pos_enc
        if self.pos_enc_method == 'concat':
            self.pos_enc = PositionalEncoding(d_model=input_dim, mode='concat')
            self.input_dim = self.input_dim + input_dim
        elif self.pos_enc_method == 'add':
            self.pos_enc = PositionalEncoding(d_model=input_dim, mode='add')
        else:
            self.pos_enc = None
        
        self.loss_fn = loss_fn

        # Build encoder
        # TODO: Convert them to registry-type build
        self.encoder = Encoder(rnn_type=enc_type, input_dim=self.input_dim,
                               hidden_dim=enc_hidden_dim, batch_first=batch_first)
        # Build decoder
        self.decoder = Decoder(rnn_type=dec_type, input_dim=self.input_dim,
                               hidden_dim=dec_hidden_dim, batch_first=batch_first)


    def forward(self, x: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        # Calculate the velocity if the include_velocity flag is true
        if self.include_velocity:
            vel_inp = self._calculate_velocity(x)
            # Concatenate the velocity to the input and the targets
            x = torch.cat([x, vel_inp], dim=-1)

        # If the pos_enc is not None, apply the positional encoding, dependent on the pos_enc_method

        if self.pos_enc is not None:
            x = self.pos_enc(x)

        encoder_input, decoder_initial_value, targets = x[:, :self.observe_until, :], x[:, self.observe_until, :], x[:, self.observe_until:, :]
        mask_pred = torch.std(x, dim=1) > self.std_thresh if self.batch_first else torch.std(x, dim=0) > self.std_thresh
        # Encode the input
        enc_out, enc_hidden = self.encoder(encoder_input)
        # Decode the output
        dec_out = self.decoder(decoder_initial_value.unsqueeze(1), enc_hidden, timesteps_to_predict=targets.shape[1])

        # The decoder's output should have a shape of (batch_size, seq_len, input_dim)
        assert dec_out.shape == targets.shape, f'dec_out.shape must be equal to targets.shape, got {dec_out.shape} and {targets.shape}'
        # Apply the padded length masks to the prediction
        if self.use_padded_len_mask:
            dec_out = dec_out * masks.float()
        
        # Apply the std masks to the prediction
        if self.use_std_mask:
            dec_out = dec_out * mask_pred.float()

        # Calculate the loss
        loss = self.loss_fn(dec_out, targets)

        return dec_out, loss
    
    def _calculate_velocity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the velocity of the input tensor

        Args:
        ---
        - `x` (`torch.Tensor`): Input tensor of shape `(batch_size, seq_len, input_dim)`
        
        Returns:
        ---
        - Velocity tensor of shape (batch_size, seq_len, input_dim)
        """
        # Calculate the velocity
        velocity = torch.zeros_like(x)
        velocity[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        return velocity

    def training_step(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        self.encoder.train()
        self.decoder.train()
        # Forward pass
        dec_out, loss = self(x, y)
        return {'loss': loss, 'out': dec_out}
    
    @torch.no_grad()
    def validation_step(self, *args, **kwargs) -> dict:
        self.encoder.eval()
        self.decoder.eval()
        return self.training_step(*args, **kwargs)
    
    
    @torch.no_grad()
    def predict(self, *args, **kwargs):
        self.encoder.eval()
        self.decoder.eval()
    