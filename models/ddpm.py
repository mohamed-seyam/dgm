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


class ResidualBlock(nn.Module):
    """
    Convolutional residual block with group normalization and time conditioning.

    Architecture:
        GroupNorm → SiLU → Conv → [+ time projection] → GroupNorm → SiLU →
        Dropout → Conv → [+ skip connection]

    Design choices:
    - GroupNorm (not weight norm): Appendix B, replaces weight normalization [49]
    - SiLU activation: standard in DDPM implementations; smooth gradient flow
    - Time embedding injected between the two convolutions: Appendix B
    - Dropout: Section 4, "We set the dropout rate on CIFAR10 to 0.1."
    - Residual connection: preserves gradient flow through depth (Wide ResNet [72])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_groups: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.act1  = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Appendix B: time embedding injected between the two convolutions
        self.time_act    = nn.SiLU()
        self.time_linear = nn.Linear(time_emb_dim, out_channels)

        self.norm2   = nn.GroupNorm(num_groups, out_channels)
        self.act2    = nn.SiLU()
        self.dropout = nn.Dropout(dropout)          # Section 4: p=0.1
        self.conv2   = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 1×1 conv when channel dimensions differ; identity otherwise
        self.skip_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x     : (B, in_channels,  H, W)
            t_emb : (B, time_emb_dim)        shared time vector from UNet.time_mlp
        Returns:
            (B, out_channels, H, W)
        """
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        t = self.time_act(t_emb)
        t = self.time_linear(t)
        t = t[:, :, None, None]     # broadcast over H, W
        h = h + t

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip_conv(x)