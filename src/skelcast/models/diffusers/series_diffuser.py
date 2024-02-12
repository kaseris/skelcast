import torch
import torch.nn as nn

from skelcast.models.diffusers.embedding import DiffusionEmbedding, PositionalEncoding


class Series_Denoiser(nn.Module):
    def __init__(self, input_dim, qkv_dim, num_layers, num_heads, prefix_len, pred_len, diff_steps):
        super().__init__()

        self.input_dim = input_dim
        self.qkv_dim = qkv_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prefix_len = prefix_len
        self.pred_len = pred_len
        self.diff_steps = diff_steps

        self.spatial_layer = nn.TransformerEncoderLayer(d_model=qkv_dim, nhead=num_heads,
                                                    dim_feedforward=qkv_dim, activation='gelu')
        self.spatial_TF = nn.TransformerEncoder(self.spatial_layer, num_layers=num_layers)
        self.spatial_inp_fc = nn.Linear(prefix_len+pred_len, qkv_dim)
        self.spatial_out_fc = nn.Linear(qkv_dim, prefix_len+pred_len)

        self.temporal_layer = nn.TransformerEncoderLayer(d_model=qkv_dim, nhead=num_heads,
                                                    dim_feedforward=qkv_dim, activation='gelu')
        self.temporal_TF = nn.TransformerEncoder(self.temporal_layer, num_layers=num_layers)
        self.temporal_inp_fc = nn.Linear(input_dim, qkv_dim)
        self.temporal_out_fc = nn.Linear(qkv_dim, input_dim)

        self.pos_encoder = PositionalEncoding(qkv_dim)
        
        self.step_temporal_encoder = DiffusionEmbedding(diff_steps, qkv_dim)
        self.step_spatial_encoder = DiffusionEmbedding(diff_steps, qkv_dim)

    def forward(self, noise, x, t):
        step_spatial_embed = self.step_spatial_encoder(t)
        step_temporal_embed = self.step_temporal_encoder(t)
        window = torch.cat([x, noise], dim=0)

        spatial = self.spatial_inp_fc(window.permute(2, 1, 0)) + step_spatial_embed # L B D -> D B L 
        spatial = self.pos_encoder(spatial)
        spatial = self.spatial_TF(spatial) 
        spatial = self.spatial_out_fc(spatial).permute(2, 1, 0)

        temporal = self.temporal_inp_fc(spatial) + step_temporal_embed
        temporal = self.pos_encoder(temporal)
        temporal = self.temporal_TF(temporal)
        temporal = self.temporal_out_fc(temporal)

        return temporal[x.shape[0]:]