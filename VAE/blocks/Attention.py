import torch
from torch import nn
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from VAE.blocks.rotary_embedding import RotaryEmbedding
from lightning_attn.ops import lightning_attn_func



class Attention(nn.Module):
    def __init__(self, dim, num_heads = 8, attn_type = "cosine", causal=False, emb_dim=None, positional_encoding="absolute", layer_idx=None, project_qkv=True):
        super().__init__()

        self.layer_idx = layer_idx
        self.positional_encoding = positional_encoding
        self.project_qkv = project_qkv
        self.causal = causal

        # The dimension must be divisible by the number of heads
        assert dim % num_heads == 0, "The dimension must be divisible by the number of heads"

        # If the attention type is "both", even indices are softmax while odd indices are cosine
        if attn_type == "both":
            attn_type = "softmax" if layer_idx % 4 == 0 else "cosine"
        
        # Projections
        if project_qkv:
            self.query_proj = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
            self.key_proj = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
            self.value_proj = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
        self.out_proj = nn.Linear(dim if emb_dim == None else emb_dim, dim if emb_dim == None else emb_dim, bias = False)
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = (dim if emb_dim == None else emb_dim) // num_heads
        if attn_type == "softmax":
            self.scale = self.head_dim ** -0.5

            # Softmax attention also needs q k norms
            self.q_norm = nn.RMSNorm(dim, dim)
            self.k_norm = nn.RMSNorm(dim, dim)

        elif attn_type == "cosine":
            self.norm_const = nn.Parameter(0.5*torch.ones(1, num_heads, 1, 1, dtype=self.out_proj.weight.dtype).to(self.out_proj.weight.device))
        else:
            raise RuntimeError(f"attn_type must be either softmax or cosine, but got {attn_type}")
        self.attn_type = attn_type
        self.causal = causal


        # Rotary embeddings
        if positional_encoding == "RoPE":
            self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        
        
        
    def forward(self, x, q=None, k=None, v=None, mask=None):
        # Reshape mask
        if mask is not None:
            mask = mask[:, None, None, :].bool()

        if self.project_qkv:
            N, S, d = x.shape
            T = S
        else:
            assert q is not None and k is not None and v is not None, "q, k, and v must be provided if project_qkv is True"
            assert x is None, "x must be None if project_qkv is True"
            N, S, d = q.shape
            N, T, d = k.shape


        # QKV projection
        if self.project_qkv:
            q = self.query_proj(x)
            k = self.key_proj(x)
            v = self.value_proj(x)



        # RMSNorm if softmax
        # Project the queries, keys, and values (N, C, d) --> (N, H, C, d//H)
        if self.attn_type == "softmax":
            # Add RMS norm if softmax
            queries = self.q_norm(q).reshape(N, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            keys = self.k_norm(k).reshape(N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            values = v.reshape(N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            queries = q.reshape(N, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            keys = k.reshape(N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            values = v.reshape(N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Normalize if cosine attention
        if self.attn_type == "cosine" or self.attn_type == "cosine2":
            queries = torch.nn.functional.normalize(queries, dim=-1, p=2)
            keys = torch.nn.functional.normalize(keys, dim=-1, p=2)





        # Apply rotary embeddings
        if self.positional_encoding == "RoPE":
            queries = self.rotary_emb.rotate_queries_or_keys(queries)
            keys = self.rotary_emb.rotate_queries_or_keys(keys)





            
        # Softmax attention
        if self.attn_type == "softmax":
            # Create mask
            if self.causal:
                with torch.no_grad():
                    mask = mask & torch.tril(torch.ones(N, self.num_heads, S, T, requires_grad=False, dtype=torch.bool, device=x.device))

            # Flash attention
            # attn = flash_attn_func(queries, keys, values, causal=self.causal)

            attn = (queries @ keys.mT) * self.scale
            
            if mask is not None:
                attn = attn.masked_fill(~mask, float('-inf')).softmax(dim=-1)
            else:
                attn = attn.softmax(dim=-1)

            attn = attn @ values










            
        # Cosine attention
        elif self.attn_type == "cosine":
            # We need to normalize the values
            values = values / (mask.sum(-1, keepdim=True)**self.norm_const.sigmoid().to(values.device))

            # Mask out the queries, keys, and values
            # queries = queries * mask.mT
            keys = keys * mask.mT
            values = values * mask.mT

            # Causual attention can use lightning attention
            if self.causal:
                with torch.no_grad():
                    mask = mask & torch.tril(torch.ones(N, self.num_heads, S, T, requires_grad=False, dtype=torch.bool, device=x.device))
                attn = ((queries @ keys.mT) * mask) @ values
                # attn_ = lightning_attn_func(queries, keys, values)
            else:
                attn = queries @ (keys.mT @ values)


        # Output projection
        return self.out_proj(attn.permute(0, 2, 1, 3).reshape(N, S, -1))
