import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Transformer-style sinusoidal positional embedding, adapted for
    diffusion timestep t.

    Appendix B: each residual block receives t via this embedding.
    The embedding follows the standard formula from [60]:

        PE(t, 2i)   = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))

    where d = dim and i indexes the embedding dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t : (B,) integer timestep tensor (0-indexed, range [0, T-1])
        Returns:
            emb : (B, dim)
        """
        device = t.device
        half   = self.dim // 2

        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(half, device=device).float() / (half - 1)
        )

        args    = t.float()[:, None] * freqs[None, :]
        sin_emb = torch.sin(args)
        cos_emb = torch.cos(args)

        return torch.cat([sin_emb, cos_emb], dim=-1)
