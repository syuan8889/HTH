import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm

class HashHead(nn.Module):
    def __init__(
        self,
        embedding_dim = 768,
        hash_bit = 256,
    ):
        super().__init__()
        self.hash_bit  = hash_bit
        self.embedding_dim = embedding_dim
        self.recon_layer = nn.Sequential(
            nn.Linear(hash_bit,embedding_dim),
        )
        self.norm = nn.LayerNorm(embedding_dim)
    # ys maybe u_hash_codes or hash_codes
    def forward(self, x):
        # eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        # x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.recon_layer(x)
        x = self.norm(x)
        return x
