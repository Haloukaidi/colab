"""EfficientSAM3 Lite50 modules; PyTorch only."""
import torch
from torch import nn
from torch.nn import functional as F

class LiteProjectionHead(nn.Sequential):
    def __init__(self, in_channels=512, mid_channels=256, out_channels=1024):
        super().__init__(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1,
                      groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels), nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, 1),
        )

class LiteFPNLevel(nn.Module):
    def __init__(self, in_channels=1024, out_channels=256, scale=1.0):
        super().__init__()
        if scale not in (4.0, 2.0, 1.0, 0.5):
            raise ValueError(scale)
        self.scale = float(scale)
        self.reduce = nn.Conv2d(in_channels, out_channels, 1)
        self.dw = nn.Conv2d(out_channels, out_channels, 3,
                            padding=1, groups=out_channels, bias=False)
        self.norm = nn.GroupNorm(8, out_channels)
        self.pw = nn.Conv2d(out_channels, out_channels, 1)
    def forward(self, x):
        x = self.reduce(x)
        if self.scale != 1:
            h, w = x.shape[-2:]
            x = F.interpolate(x, size=(round(h*self.scale), round(w*self.scale)),
                              mode="bilinear", align_corners=False)
        return self.pw(F.gelu(self.norm(self.dw(x))))

class FactorizedTokenEmbedding(nn.Module):
    def __init__(self, vocab_size=49408, embedding_dim=512, rank=64):
        super().__init__()
        assert 0 < rank <= embedding_dim
        self.codes = nn.Embedding(vocab_size, rank)
        self.projection = nn.Linear(rank, embedding_dim, bias=False)
    def forward(self, ids):
        return self.projection(self.codes(ids))
    @torch.no_grad()
    def initialize_from_dense(self, weights, niter=1):
        if tuple(weights.shape) != (self.codes.num_embeddings,
                                     self.projection.out_features):
            raise ValueError("Dense embedding shape mismatch")
        rank = self.codes.embedding_dim
        q = min(rank+8, min(weights.shape))
        u, s, v = torch.svd_lowrank(weights.detach().float(), q=q, niter=niter)
        self.codes.weight.copy_((u[:, :rank]*s[:rank]).to(self.codes.weight))
        self.projection.weight.copy_(v[:, :rank].to(self.projection.weight))
