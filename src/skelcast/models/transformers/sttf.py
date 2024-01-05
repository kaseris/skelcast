import torch
import torch.nn as nn
import torch.nn.functional as F

from skelcast.models.module import SkelcastModule
from skelcast.models.transformers.base import PositionalEncoding
from skelcast.models import MODELS


class TemporalMultiHeadAttentionBlock(nn.Module):
  '''
  Args:
    n_heads `int`: Number of heads
    d_model `int`: The input dimensionality
    d_head `int`: The per-head dimensionality
  '''
  def __init__(self, n_heads, d_head, n_joints, d_model, dropout=0.1, debug=False):
    super().__init__()
    self.n_heads = n_heads
    self.d_model = d_model
    self.d_head = d_head
    self.n_joints = n_joints
    self.debug = debug

    self.q = nn.Linear(in_features=d_model * n_joints, out_features=n_joints * d_head * n_heads, bias=False)
    self.k = nn.Linear(in_features=d_model * n_joints, out_features=n_joints * d_head * n_heads, bias=False)
    self.v = nn.Linear(in_features=d_model * n_joints, out_features=n_joints * d_head * n_heads, bias=False)

    self.back_proj = nn.Linear(in_features=d_head * n_heads, out_features=d_model, bias=False) # Project back to original dimensionality.
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor):
    batch_size, seq_len, n_joints, dims = x.shape
    assert n_joints == self.n_joints, f'Expected n_joints: {self.n_joints}. Got: n_joints: {n_joints}'
    assert dims == self.d_model, f'Expected d_model: {self.d_model}. Got: d_model: {dims}'
    x = x.view(batch_size, seq_len, self.n_joints * self.d_model)
    q_proj = self.q(x)
    k_proj = self.k(x)
    v_proj = self.v(x)
    mask = self.get_mask(seq_len, batch_size).to(x.device)
    attn_prod_ = torch.bmm(q_proj, k_proj.permute(0, 2, 1)) * (self.d_model) ** -0.5

    attn_temporal = F.softmax(attn_prod_ + mask, dim=-1)
    attn = attn_temporal @ v_proj
    out = self.back_proj(attn.view(batch_size, seq_len, n_joints, -1))
    out = self.dropout(out)
    if self.debug:
      return out.view(batch_size, seq_len, n_joints, dims), attn, attn_temporal
    return out.view(batch_size, seq_len, n_joints, dims)

  def get_mask(self, seq_len, batch_size):
    mask = torch.triu(torch.ones((seq_len, seq_len)) * float('-inf'), diagonal=1)
    return mask.repeat(batch_size, 1, 1)


class SpatialMultiHeadAttentionBlock(nn.Module):
  def __init__(self, n_heads, d_head, n_joints, d_model, dropout=0.1, debug=False):
    super().__init__()
    self.n_heads = n_heads
    self.d_model = d_model
    self.d_head = d_head
    self.n_joints = n_joints
    self.debug = debug

    # The query projection treats all joints differently, so stays as is
    self.q_spatial = nn.Linear(in_features=n_joints * d_model, out_features=n_joints * n_heads * d_head, bias=False)
    # The key and value projections are shared across all joints and time steps
    self.k_spatial = nn.Linear(in_features=d_model, out_features=n_heads * d_head, bias=False)
    self.v_spatial = nn.Linear(in_features=d_model, out_features=n_heads * d_head, bias=False)
    self.back_proj_spatial = nn.Linear(in_features=n_heads * d_head, out_features=d_model, bias=False)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    batch_size, seq_len, n_joints, dims = x.shape
    assert n_joints == self.n_joints, f'Expected n_joints: {self.n_joints}. Got: n_joints: {n_joints}'
    assert dims == self.d_model, f'Expected d_model: {self.d_model}. Got: d_model: {dims}'
    # Treat all joints individually to compute the query
    q_proj_spatial = self.q_spatial(x.view(batch_size, seq_len, n_joints * self.d_model))
    q_proj_spatial = q_proj_spatial.view(batch_size, n_joints, self.n_heads * self.d_head * seq_len)
    k_proj_spatial = self.k_spatial(x).view(batch_size, n_joints, self.n_heads * self.d_head * seq_len)
    v_proj_spatial = self.v_spatial(x).view(batch_size, n_joints, self.n_heads * self.d_head * seq_len)

    attn_prod_ = torch.bmm(q_proj_spatial, k_proj_spatial.permute(0, 2, 1)) * (self.d_model) ** -.5

    attn_spatial = F.softmax(attn_prod_, dim=-1)

    mha_attn_spatial = attn_spatial @ v_proj_spatial
    spatial_attn_out = self.back_proj_spatial(mha_attn_spatial.view(batch_size, n_joints, seq_len, self.n_heads * self.d_head).permute(0, 2, 1, 3))
    spatial_attn_out = self.dropout(spatial_attn_out)
    if self.debug:
      return spatial_attn_out, mha_attn_spatial, attn_spatial
    return spatial_attn_out


class PostNorm(nn.Module):
  def __init__(self, dim, fn):
    super().__init__()
    self.norm = nn.LayerNorm(dim)
    self.fn = fn

  def forward(self, x, **kwargs):
    return self.norm(self.fn(x) + x, **kwargs)


class MLP(nn.Module):
  def __init__(self, dim, embedding_dim, dropout=0.1):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(dim, embedding_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(embedding_dim, dim),
        nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)


class Transformer(nn.Module):
  def __init__(self, mlp_dim, dim, n_blocks, n_heads, d_head, dropout,
               n_joints):
    super().__init__()
    self.blocks = nn.ModuleList([])
    for _ in range(n_blocks):
      self.blocks.append(
          nn.ModuleList(
              [PostNorm(dim, TemporalMultiHeadAttentionBlock(n_heads=n_heads,
                                                         d_head=d_head,
                                                         n_joints=n_joints,
                                                         dropout=dropout,
                                                         d_model=dim)),
               PostNorm(dim, SpatialMultiHeadAttentionBlock(n_heads=n_heads,
                                                        d_head=d_head,
                                                        n_joints=n_joints,
                                                        d_model=dim,
                                                        dropout=dropout)),
               PostNorm(dim, MLP(dim=dim, embedding_dim=mlp_dim, dropout=dropout))])
      )

  def forward(self, x):
    for tmp_attn, spa_attn, mlp in self.blocks:
      o_1 = tmp_attn(x)
      o_2 = spa_attn(x)
      out = mlp(o_1 + o_2)
      return out
    

@MODELS.register_module()
class SpatioTemporalTransformer(SkelcastModule):
  """
  PyTorch implementation of the model proposed in the paper:
  "A Spatio-temporal Transformer for 3D Human Motion Prediction"
  https://arxiv.org/abs/2004.08692

  Args:
  -  n_joints `int`: Number of joints in the skeleton
  -  d_model `int`: The input dimensionality after the linear projection that computes the skeleton joints representation
  -  n_blocks `int`: Number of transformer blocks
  -  n_heads `int`: Number of self attention heads (for both temporal and spatial attention)
  -  d_head `int`: The per-head dimensionality
  -  mlp_dim `int`: The dimensionality of the MLP
  -  dropout `float`: Dropout probability
  """
  def __init__(self, n_joints,
               d_model,
               n_blocks,
               n_heads,
               d_head,
               mlp_dim,
               dropout,
               loss_fn: nn.Module = None,
               debug=False):
    super().__init__()
    self.n_joints = n_joints
    self.d_model = d_model
    self.n_blocks = n_blocks
    self.n_heads = n_heads
    self.d_head = d_head
    self.mlp_dim = mlp_dim
    self.dropout = dropout
    self.loss_fn = nn.SmoothL1Loss() if loss_fn is None else loss_fn
    self.debug = debug

    # Embedding projection before feeding into the transformer
    self.embedding = nn.Linear(in_features=3 * n_joints, out_features=d_model * n_joints, bias=False)
    self.pre_dropout = nn.Dropout(dropout)
    self.pe = PositionalEncoding(d_model=d_model * n_joints)

    self.transformer = Transformer(mlp_dim=mlp_dim,
                                   dim=d_model,
                                   n_blocks=n_blocks,
                                   n_heads=n_heads,
                                   d_head=d_head,
                                   dropout=dropout,
                                   n_joints=n_joints)
    
    self.linear_out = nn.Linear(in_features=d_model, out_features=3, bias=False)

  def forward(self, x: torch.Tensor):
    if x.ndim > 4:
      x = x.squeeze(2)
    batch_size, seq_len, n_joints, dims = x.shape
    input_ = x.view(batch_size, seq_len, n_joints * dims)
    o = self.embedding(input_)
    o = self.pe.pe.repeat(batch_size, 1, 1)[:, :seq_len, :] + o
    o = self.pre_dropout(o)
    o = o.view(batch_size, seq_len, n_joints, self.d_model)
    o = self.transformer(o)
    out = self.linear_out(o) + x
    return out
  
  def training_step(self, **kwargs) -> dict:
    # Retrieve the x and y from the keyword arguments
    x, y = kwargs['x'], kwargs['y']
    # Forward pass
    out = self(x)
    # Compute the loss
    loss = self.loss_fn(out, y.squeeze(2))
    return {'loss': loss, 'out': out}

  def validation_step(self, *args, **kwargs):
    with torch.no_grad():
      return self.training_step(*args, **kwargs)
    
  def predict(self, sample, n_steps=10, observe_from_to=[10]):
    """Predicts `n_steps` into the future given the sample.

    Args:
    -  sample `torch.Tensor`: The sample to predict from. The shape must be (seq_len, n_skels, n_joints, 3)
    -  n_steps `int`: The number of steps to predict into the future
    -  observe_from_to `list`: The start and end of the observation. If the list contains only one element, then it is the end of the observation.
    """
    if len(observe_from_to) == 1:
      from_ = 0
      to_ = observe_from_to[0]
    else:
      if len(observe_from_to) > 2:
            raise ValueError('`observe_from_to` must be a list of length 1 or 2.')
      if observe_from_to[0] > observe_from_to[1]:
          raise ValueError('The start of observation must be before the end of observation.')
      from_, to_ = observe_from_to

    sample = sample.squeeze(1).unsqueeze(0)
    sample_input = sample[:, from_:to_, ...]
    forecasted = []
    self.eval()
    with torch.no_grad():
        for _ in range(n_steps):
            prediction = self(sample_input.to(torch.float32))
            forecasted.append(prediction[:, -1:, ...].squeeze(1).detach())
            # Roll the sample input and replace the last element with the last element of the prediction
            # sample_input = torch.roll(sample_input, -1, dims=1)
            # print(sample_input)
            # sample_input[:, -1, ...] = prediction[:, -1:, ...].unsqueeze(1)
            from_ += 1
            to_ += 1
            sample_input = sample[:, from_:to_, ...]

    return torch.stack(forecasted, dim=1)