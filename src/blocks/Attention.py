import torch
from torch import nn
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from src.blocks.patchify import patchify, unpatchify
from src.blocks.rotary_embedding import RotaryEmbedding


class Attention(nn.Module):
    def __init__(self, dim, num_heads = 8, attn_type = "cosine", causal=False, emb_dim=None, positional_encoding="absolute", layer_idx=None):
        super().__init__()

        self.layer_idx = layer_idx
        self.positional_encoding = positional_encoding

        # If the attention type is "both", even indices are softmax while odd indices are cosine
        if attn_type == "both":
            attn_type = "softmax" if layer_idx % 4 == 0 else "cosine"
        
        # Projections
        self.query_proj = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
        self.key_proj = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
        self.value_proj = nn.Linear(dim, dim if emb_dim == None else emb_dim, bias = False)
        self.out_proj = nn.Linear(dim if emb_dim == None else emb_dim, dim if emb_dim == None else emb_dim, bias = False)
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = (dim if emb_dim == None else emb_dim) // num_heads
        if attn_type == "softmax" or attn_type == "softmax_DK" or attn_type == "softmax_DV":
            self.scale = self.head_dim ** -0.5

            # Softmax attention also needs q k norms
            self.q_norm = nn.RMSNorm(dim, dim)
            self.k_norm = nn.RMSNorm(dim, dim)

        elif attn_type == "cosine":
            self.norm_const = nn.Parameter(0.5*torch.ones(1, num_heads, 1, 1, dtype=self.query_proj.weight.dtype).to(self.query_proj.weight.device))
        elif attn_type == "cosine2":
            pass
        elif attn_type == "cosine3":
            pass
        elif attn_type == "cosine4":
            pass
        elif attn_type == "cosine_norm":
            pass

        elif attn_type == "both_interp":
            self.scale = self.head_dim ** -0.5
            self.norm_const = nn.Parameter(0.5*torch.ones(1, num_heads, 1, 1, dtype=self.query_proj.weight.dtype).to(self.query_proj.weight.device))
        else:
            raise RuntimeError(f"attn_type must be either softmax or cosine, but got {attn_type}")
        self.attn_type = attn_type
        self.causal = causal


        # Rotary embeddings
        if positional_encoding == "RoPE":
            self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        
        
        
    def forward(self, x, mask=None):
        N, C, d = x.shape




        # RMSNorm if softmax
        # Project the queries, keys, and values (N, C, d) --> (N, H, C, d//H)
        if self.attn_type == "softmax" or self.attn_type == "softmax_DK" or self.attn_type == "softmax_DV":
            # Add RMS norm if softmax
            queries = self.q_norm(self.query_proj(x)).reshape(N, C, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            keys = self.k_norm(self.key_proj(x)).reshape(N, C, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            values = self.value_proj(x).reshape(N, C, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            queries = self.query_proj(x).reshape(N, C, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            keys = self.key_proj(x).reshape(N, C, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            values = self.value_proj(x).reshape(N, C, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Normalize if cosine attention
        if self.attn_type == "cosine" or self.attn_type == "cosine2":
            queries = torch.nn.functional.normalize(queries, dim=-1, p=2)
            keys = torch.nn.functional.normalize(keys, dim=-1, p=2)





        # Apply rotary embeddings
        if self.positional_encoding == "RoPE":
            queries = self.rotary_emb.rotate_queries_or_keys(queries)
            keys = self.rotary_emb.rotate_queries_or_keys(keys)




        # Add head and dim to mask
        mask = mask[:, None, :, None]


            
        # Softmax attention
        if self.attn_type == "softmax":
            # Create mask
            if self.causal:
                assert False, "Causal softmax attention not implemented"
                mask = torch.tril(torch.ones(N, self.num_heads, C, C, requires_grad=False)).bool().to(x.device)
                    
            # Flash attention
            # attn = flash_attn_func(queries.to(torch.bfloat16), keys.to(torch.bfloat16), values.to(torch.bfloat16), causal=self.causal).to(queries.dtype)

            attn = (queries @ keys.mT) * self.scale
            
            if self.causal or mask is not None:
                attn = attn.masked_fill(~mask.mT, float('-inf')).softmax(dim=-1)
            else:
                attn = attn.softmax(dim=-1)

            attn = attn @ values
        

        elif self.attn_type == "softmax_DK":
            attn = values @ (keys.mT @ queries).softmax(-2)


        elif self.attn_type == "softmax_DV":
            attn = queries @ (keys.mT @ values).softmax(-1)










            
        # Cosine attention
        elif self.attn_type == "cosine":
            """
            # Inner product
            # denom = (queries@keys.mT).sum(-1, keepdim=True) #(queries * keys.sum(-2, keepdim=True)).sum(-1, keepdim=True)
            inner = (queries@keys.mT + 1) / 2
            denom = (inner).sum(-1, keepdim=True)
            # sign = torch.sign(denom)
            # denom = denom.abs_().clamp_(1)
            # denom *= sign
            num = inner / denom
            attn = num @ values

            """

            # Mask queries and keys
            if mask is not None:
                queries = queries * mask
                keys = keys * mask

            if self.causal:
                # Create mask
                mask = torch.tril(torch.ones(N, self.num_heads, C, C, requires_grad=False)).bool().to(x.device)
                
                # We need to normalize the values
                values = values / ((mask).sum(-1, keepdims=True)**self.norm_const.sigmoid()).clamp(min=1)
                
                # Inner product
                attn = ((queries @ keys.mT) * mask) @ values

            # Can be optimized if not causal
            else:
                # Normalization term
                # v = ((values.shape[2]*torch.ones(values.shape[2]).unsqueeze(0).repeat(values.shape[1], 1)[None, :, :, None].to(values.device))**self.norm_const.sigmoid().to(values.device))
                
                # We need to normalize the values
                values = values / (values.shape[2]**self.norm_const.sigmoid().to(values.device))

                # Inner product
                attn = queries @ (keys.mT @ values)
        
        elif self.attn_type == "cosine2":
            # # Create mask
            # mask = torch.tril(torch.ones(N, self.num_heads, C, C, requires_grad=False)).bool().to(x.device) if self.causal \
            #         else torch.ones(N, self.num_heads, C, C, requires_grad=False).bool().to(x.device)

            prod = (((queries @ keys.mT)+1))# * mask

            attn = prod / prod.sum(-1, keepdim=True)

            attn = attn @ values


        elif self.attn_type == "cosine3":
            # Create mask
            mask = torch.tril(torch.ones(N, self.num_heads, C, C, requires_grad=False)).bool().to(x.device) if self.causal \
                    else torch.ones(N, self.num_heads, C, C, requires_grad=False).bool().to(x.device)

            prod = (((queries @ keys.mT))) * mask

            attn = prod / prod.abs().sum(-1, keepdim=True)

            attn = attn @ values

        elif self.attn_type == "cosine4":
            # Get norms of keys and queries
            keys_norm = keys.norm(dim=-1, keepdim=True)
            query_norm = queries.norm(dim=-1, keepdim=True)

            scale = 1/(self.head_dim**0.5)
            
            # Inner product
            attn = ((queries @ keys.mT) * scale) + (query_norm * keys_norm.mT) * scale

            # Normalize
            attn = attn / attn.sum(dim=-1, keepdim=True)

            attn = attn @ values

        elif self.attn_type == "cosine_norm":
            # Get norms of keys and queries
            key_norm = keys.norm(dim=-1, keepdim=True)
            query_norm = queries.norm(dim=-1, keepdim=True)

            # Inner product
            attn_weights = torch.matmul(queries, keys.mT)
            # attn_weights_denom = (query_norm * key_norm.mT).sum(-1, keepdim=True)
            attn_weights_denom = (query_norm * key_norm.sum(-2, keepdim=True))

            # Weight normalization
            attn_weights = attn_weights / attn_weights_denom

            attn = attn_weights @ values


        elif self.attn_type == "both_interp":
            # Compute the attention matrix
            attn = (queries @ keys.mT) * self.scale

            # We need to normalize the values
            values = values / (values.shape[2]**self.norm_const.sigmoid().to(values.device))

            # Inner product
            attn = queries @ (keys.mT @ values)



        # Output projection
        return self.out_proj(attn.permute(0, 2, 1, 3).reshape(N, C, -1))
