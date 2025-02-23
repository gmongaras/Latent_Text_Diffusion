import torch
from torch import nn

import sys
sys.path.append("./")

from VAE.blocks.Attention import Attention
from VAE.blocks.MLP import MLP
from VAE.blocks.Norm import Norm
from xformers.ops.swiglu_op import SwiGLU
from VAE.blocks.PositionalEncoding import PositionalEncoding
import math



class Transformer_Block(nn.Module):
    def __init__(self, dim, hidden_scale=4.0, num_heads = 8, attn_type = "softmax", causal=False, positional_encoding="rotary", layer_idx=None):
        super().__init__()
        
        # MLP and attention blocks
        self.MLP = MLP(dim, hidden_scale)
        # self.MLP = SwiGLU(dim, int(dim*hidden_scale), dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_type=attn_type, causal=causal, positional_encoding=positional_encoding, layer_idx=layer_idx)
        
        # Two layer norms
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

        
    def forward(self, X, mask=None):
        # Attn layer
        X = self.attn(self.norm1(X), mask=mask) + X
        
        # MLP layer
        return self.MLP(self.norm2(X)) + X
    








class Transformer_Block_Encoder(nn.Module):
    def __init__(self, dim, seq_downsample_factor, out_dim, num_inner_blocks=1, hidden_scale=4.0, num_heads = 8, attn_type = "softmax", positional_encoding="rotary", layer_idx=None):
        super().__init__()

        self.seq_downsample_factor = seq_downsample_factor
        self.out_dim = out_dim

        # Inner blocks
        self.inner_blocks = nn.ModuleList([
            Transformer_Block(dim, hidden_scale, num_heads, attn_type, False, positional_encoding, layer_idx)
            for _ in range(num_inner_blocks)
        ])

        # Norm
        self.norm = nn.RMSNorm(dim)

        # QKV Projection
        self.query_proj = nn.Linear(out_dim, out_dim, bias = False)
        self.key_proj = nn.Linear(dim, out_dim, bias = False)
        self.value_proj = nn.Linear(dim, out_dim, bias = False)
        # Positional encodings to produce the query
        self.pos_enc = PositionalEncoding(out_dim)
        # Attention - Uses absolute positional encodings due to the queries being downsampled
        self.attn = Attention(out_dim, num_heads=num_heads, attn_type=attn_type, causal=False, positional_encoding="absolute", layer_idx=layer_idx, project_qkv=False)
        
    def forward(self, X, mask=None):
        # Forward each inner block
        for block in self.inner_blocks:
            X = block(X, mask=mask)

        # Norm
        X = self.norm(X)

        # Queries are just the input ids
        Q = self.query_proj(self.pos_enc(torch.arange(math.ceil(X.shape[1] / self.seq_downsample_factor)).unsqueeze(0).repeat(X.shape[0], 1).to(X.device).long()))
        # KV projection
        K = self.key_proj(X)
        V = self.value_proj(X)

        # Create a new, downsampled mask
        if mask is not None:
            new_mask = mask[:, ::self.seq_downsample_factor]

        # Attention
        return self.attn(None, Q, K, V, mask=mask), new_mask
    
















class Transformer_Block_Encoder_V2Conv(nn.Module):
    def __init__(self, dim, seq_downsample_factor, out_dim, num_inner_blocks=1, hidden_scale=4.0, num_heads = 8, attn_type = "softmax", positional_encoding="rotary", layer_idx=None):
        super().__init__()

        self.seq_downsample_factor = seq_downsample_factor
        self.out_dim = out_dim

        # Inner blocks
        self.inner_blocks = nn.ModuleList([
            Transformer_Block(dim, hidden_scale, num_heads, attn_type, False, positional_encoding, layer_idx)
            for _ in range(num_inner_blocks)
        ])

        # Norm
        self.norm = nn.RMSNorm(dim)

        # QKV Projection
        self.query_proj = nn.Linear(out_dim, out_dim, bias = False)
        self.key_proj = nn.Linear(dim, out_dim, bias = False)
        self.value_proj = nn.Linear(dim, out_dim, bias = False)
        # Positional encodings to produce the query
        self.pos_enc = PositionalEncoding(out_dim)
        # Attention - Uses absolute positional encodings due to the queries being downsampled
        self.attn = Attention(out_dim, num_heads=num_heads, attn_type=attn_type, causal=False, positional_encoding="absolute", layer_idx=layer_idx, project_qkv=False)
        
        # Used to pool the sequence
        self.pool = nn.Conv1d(dim, out_dim, 2*seq_downsample_factor+1, stride=seq_downsample_factor, padding=seq_downsample_factor, bias=False)

        # Norm and MLP
        self.norm2 = nn.RMSNorm(out_dim)
        self.MLP = MLP(out_dim, hidden_scale)

    def forward(self, X, mask=None):
        # Forward each inner block
        for block in self.inner_blocks:
            X = block(X, mask=mask)

        # Norm
        X = self.norm(X)

        # Pool the sequence to get the downsampled queries
        X = X * mask[:, :, None]
        X_downsample = self.pool(X.mT).mT
        # KV projection
        Q = self.query_proj(X_downsample)
        K = self.key_proj(X)
        V = self.value_proj(X)

        # Create a new, downsampled mask
        if mask is not None:
            new_mask = mask[:, ::self.seq_downsample_factor]

        # Attention (this won't work without the skip connection)
        X = self.attn(None, Q, K, V, mask=mask) + X_downsample

        # MLP
        return self.MLP(self.norm2(X)) + X, new_mask











class Transformer_Block_Decoder(nn.Module):
    def __init__(self, dim, cond_dim, total_downscale_factor, hidden_scale=4.0, num_heads = 8, attn_type = "softmax", positional_encoding="rotary", use_cross_attn=True, causal_VAE=False, causal_cheat=False, layer_idx=None):
        super().__init__()

        self.use_cross_attn = use_cross_attn
        self.causal_cheat = causal_cheat
        
        # MLP and attention blocks
        self.MLP = MLP(dim, hidden_scale)
        # self.MLP = SwiGLU(dim, int(dim*hidden_scale), dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_type=attn_type, causal=causal_VAE, causal_cheat=causal_cheat, total_downscale_factor=total_downscale_factor, positional_encoding=positional_encoding, layer_idx=layer_idx)
        if use_cross_attn:
            self.cross_attn = Attention(dim, num_heads=num_heads, attn_type=attn_type, causal=False, causal_cheat=False, positional_encoding=positional_encoding, layer_idx=layer_idx, project_qkv=False)
        
        # Cross attention qkv projection
        if use_cross_attn:
            self.query_proj = nn.Linear(dim, dim, bias = False)
            self.key_proj = nn.Linear(cond_dim, dim, bias = False)
            self.value_proj = nn.Linear(cond_dim, dim, bias = False)

        # Layer norms
        self.norm1 = nn.RMSNorm(dim)
        if use_cross_attn:
            self.norm_cond = nn.RMSNorm(cond_dim)
            self.norm2 = nn.RMSNorm(dim)
        self.norm3 = nn.RMSNorm(dim)

        
    def forward(self, X, cond=None, mask=None, mask_cond=None):
        # Attn layer
        X = self.attn(self.norm1(X), mask=mask) + X

        # Cross attn layer
        if self.use_cross_attn:
            cond = self.norm_cond(cond)
            q = self.query_proj(self.norm2(X))
            k = self.key_proj(cond)
            v = self.value_proj(cond)
            X = self.cross_attn(None, q, k, v, mask=mask_cond) + X
        
        # MLP layer
        return self.MLP(self.norm3(X)) + X










# class Transformer_Block_Decoder_V2Upsample(nn.Module):
#     def __init__(self, dim, seq_upsample_factor, out_dim, num_inner_blocks=1, hidden_scale=4.0, num_heads = 8, attn_type = "softmax", positional_encoding="rotary", layer_idx=None):
#         super().__init__()

#         self.seq_upsample_factor = seq_upsample_factor
#         self.out_dim = out_dim

#         # Inner blocks
#         self.inner_blocks = nn.ModuleList([
#             Transformer_Block(out_dim, hidden_scale, num_heads, attn_type, False, positional_encoding, layer_idx)
#             for _ in range(num_inner_blocks)
#         ])

#         # Norm
#         self.norm = nn.RMSNorm(dim)

#         # QKV Projection
#         # self.query_proj = nn.Linear(dim, out_dim, bias = False)
#         self.key_proj = nn.Linear(dim, out_dim, bias = False)
#         self.value_proj = nn.Linear(dim, out_dim, bias = False)
#         # Positional encodings to produce the query
#         self.pos_enc = PositionalEncoding(out_dim)
#         # Attention - Uses absolute positional encodings due to the queries being downsampled
#         self.attn = Attention(out_dim, num_heads=num_heads, attn_type=attn_type, causal=False, positional_encoding="absolute", layer_idx=layer_idx, project_qkv=False)
        
#         # Used to pool the sequence
#         self.pool = nn.ConvTranspose1d(dim, out_dim, 2*seq_upsample_factor+1, stride=seq_upsample_factor, padding=seq_upsample_factor, output_padding=seq_upsample_factor-1, bias=False)

#         # Second Norm
#         self.norm2 = nn.RMSNorm(out_dim)

#         # MLP
#         self.MLP = MLP(out_dim, hidden_scale)

#     def forward(self, X, mask=None, mask_next=None):
#         # Norm
#         X = self.norm(X)

#         # Upsample the sequence to get the upsampled queries
#         X = X * mask[:, :, None]
#         Q = self.pool(X.mT).mT
#         # KV projection
#         K = self.key_proj(X)
#         V = self.value_proj(X)

#         # Attention
#         X = self.attn(None, Q, K, V, mask=mask)

#         # MLP
#         X = self.MLP(self.norm2(X)) + X
        
#         # Forward each inner block
#         for block in self.inner_blocks:
#             X = block(X, mask=mask_next)

#         return X