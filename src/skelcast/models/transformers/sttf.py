import torch
import torch.nn as nn
import torch.nn.functional as F

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
    print(f'q_proj, k_proj, v_proj shapes: {q_proj.shape, k_proj.shape, v_proj.shape}')
    mask = self.get_mask(seq_len, batch_size)
    attn_prod_ = torch.bmm(q_proj, k_proj.permute(0, 2, 1)) * (self.d_model) ** -0.5

    attn_temporal = F.softmax(attn_prod_ + mask, dim=-1)
    attn = attn_temporal @ v_proj
    print(f'attn shape: {attn.shape}')
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