import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreNorm(nn.Module):
    def __init__(self, dim, fn) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        self.fn(self.norm(x), **kwargs)


class MLP(nn.Module):
    def __init__(self, dim, embedding_dim, dropout=0.1) -> None:
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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, inner_head_dim=64, dropout=0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.inner_head_dim = inner_head_dim

        self.scale = self.inner_head_dim ** -.5

        self.per_head_dimensionality = self.inner_head_dim * self.n_heads
        self.pre_dropout = nn.Dropout(dropout)
        self.to_qkv_chunk = nn.Linear(self.d_model, self.per_head_dimensionality * 3,
                                      bias=False)
        self.out_proj = nn.Linear(self.per_head_dimensionality, self.d_model,
                                  bias=False)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        qkv = self.to_qkv_chunk(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(-1, self.n_heads, self.inner_head_dim).permute(1, 0, 2), qkv)
        attn = F.softmax(torch.matmul(q, k.permute(0, 2, 1) * self.scale), dim=-1)
        attn = self.pre_dropout(attn)
        attn = attn @ v
        attn = attn.permute(1, 0, 2).contiguous()
        attn = attn.view(-1, self.n_heads * self.inner_head_dim)
        out = self.out_proj(attn)
        return self.out_dropout(out)
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a long enough 'PE' matrix with position and dimension indexes
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Registers pe as a buffer that should not be considered a model parameter.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Adds the positional encoding vector to the input embedding vector
        x = x + self.pe[:x.size(0), :]
        return x
    

class Transformer(nn.Module):
    def __init__(self, dim, n_blocks, n_heads, dim_head, mlp_dim, dropout) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(n_blocks):
            self.blocks.append(
                nn.ModuleList([
                    PreNorm(dim, MultiHeadSelfAttention(dim,
                                                        n_heads,
                                                        dim_head,
                                                        dropout)),
                    PreNorm(dim, MLP(dim, mlp_dim, dropout))
                ])
            )
    
    def forward(self, x):
        for attn, mlp in self.blocks:
            out = attn(x) + x
            out = mlp(out) + out
        return out
