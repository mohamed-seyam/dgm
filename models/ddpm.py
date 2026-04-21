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
    

class AttentionBlock(nn.Module):
    """
    Single-head self-attention with group normalization.

    Appendix B:
        "We use self-attention at the 16×16 feature map resolution between
         the convolutional blocks."
    """

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        self.norm   = nn.GroupNorm(num_groups, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj   = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale  = channels ** -0.5      # 1/sqrt(d) — prevents softmax saturation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W)
        Returns:
            (B, C, H, W) — same shape, with residual connection
        """
        B, C, H, W = x.shape

        h   = self.norm(x)
        qkv = self.to_qkv(h).view(B, 3, C, H * W)
        q   = qkv[:, 0]                             # (B, C, H*W)
        k   = qkv[:, 1]
        v   = qkv[:, 2]

        scores = torch.bmm(q.permute(0, 2, 1), k) * self.scale  # (B, H*W, H*W)
        attn   = torch.softmax(scores, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        out = self.proj(out)

        return x + out

class UNet(nn.Module):
    """
    U-Net backbone that predicts the noise ε_θ(x_t, t).

    Architecture (CIFAR-10, Appendix B):
        base_channels=128, channel_mult=(1,2,2,2)
        Channels per level: 128, 256, 256, 256
        Self-attention at 16×16 only
        2 ResBlocks per encoder level, 3 per decoder level (extra one consumes
        the downsampling skip), dropout=0.1
        35.7 M parameters total
    """

    def __init__(
        self,
        img_channels: int = 3,
        base_channels: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        C   = base_channels          # 128
        C2  = base_channels * 2      # 256
        T   = base_channels * 4      # 512 — time embedding dimension

        # Appendix B: "sinusoidal position embedding projected through two FC layers"
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(C),
            nn.Linear(C, T),
            nn.SiLU(),
            nn.Linear(T, T),
        )

        # Stem: 3×32×32 → 128×32×32
        self.conv_in = nn.Conv2d(img_channels, C, kernel_size=3, padding=1)

        # ── Encoder — Level 0  (32×32, 128 ch) ──────────────────────────────
        self.enc0_res0 = ResidualBlock(C,  C,  T, dropout=dropout)
        self.enc0_res1 = ResidualBlock(C,  C,  T, dropout=dropout)
        self.enc0_down = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1)

        # ── Encoder — Level 1  (16×16, 256 ch, self-attention) ──────────────
        self.enc1_res0  = ResidualBlock(C,  C2, T, dropout=dropout)
        self.enc1_attn0 = AttentionBlock(C2)
        self.enc1_res1  = ResidualBlock(C2, C2, T, dropout=dropout)
        self.enc1_attn1 = AttentionBlock(C2)
        self.enc1_down  = nn.Conv2d(C2, C2, kernel_size=3, stride=2, padding=1)

        # ── Encoder — Level 2  (8×8, 256 ch) ────────────────────────────────
        self.enc2_res0 = ResidualBlock(C2, C2, T, dropout=dropout)
        self.enc2_res1 = ResidualBlock(C2, C2, T, dropout=dropout)
        self.enc2_down = nn.Conv2d(C2, C2, kernel_size=3, stride=2, padding=1)

        # ── Encoder — Level 3  (4×4, 256 ch) ────────────────────────────────
        self.enc3_res0 = ResidualBlock(C2, C2, T, dropout=dropout)
        self.enc3_res1 = ResidualBlock(C2, C2, T, dropout=dropout)

        # ── Bottleneck  (4×4, 256 ch): ResBlock → Self-Attn → ResBlock ──────
        self.mid_res1 = ResidualBlock(C2, C2, T, dropout=dropout)
        self.mid_attn = AttentionBlock(C2)
        self.mid_res2 = ResidualBlock(C2, C2, T, dropout=dropout)

        # ── Decoder — Level 3  (4×4 → 8×8) ─────────────────────────────────
        # Receives skips 11, 10, 9 — all 256 ch → cat gives 512 ch input
        self.dec3_res0 = ResidualBlock(C2 + C2, C2, T, dropout=dropout)
        self.dec3_res1 = ResidualBlock(C2 + C2, C2, T, dropout=dropout)
        self.dec3_res2 = ResidualBlock(C2 + C2, C2, T, dropout=dropout)
        # Nearest-neighbour upsample + conv avoids checkerboard artifacts
        self.dec3_up   = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(C2, C2, kernel_size=3, padding=1),
        )

        # ── Decoder — Level 2  (8×8 → 16×16) ───────────────────────────────
        # Receives skips 8, 7, 6 — all 256 ch
        self.dec2_res0 = ResidualBlock(C2 + C2, C2, T, dropout=dropout)
        self.dec2_res1 = ResidualBlock(C2 + C2, C2, T, dropout=dropout)
        self.dec2_res2 = ResidualBlock(C2 + C2, C2, T, dropout=dropout)
        self.dec2_up   = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(C2, C2, kernel_size=3, padding=1),
        )

        # ── Decoder — Level 1  (16×16 → 32×32, self-attention) ──────────────
        # Receives skips 5, 4 (256 ch → 512) and skip3 (128 ch → 384)
        self.dec1_res0  = ResidualBlock(C2 + C2, C2, T, dropout=dropout)
        self.dec1_attn0 = AttentionBlock(C2)
        self.dec1_res1  = ResidualBlock(C2 + C2, C2, T, dropout=dropout)
        self.dec1_attn1 = AttentionBlock(C2)
        self.dec1_res2  = ResidualBlock(C2 + C,  C2, T, dropout=dropout)
        self.dec1_attn2 = AttentionBlock(C2)
        self.dec1_up    = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(C2, C2, kernel_size=3, padding=1),
        )

        # ── Decoder — Level 0  (32×32, 128 ch) ──────────────────────────────
        # Receives skips 2, 1, 0 — all 128 ch
        self.dec0_res0 = ResidualBlock(C2 + C, C, T, dropout=dropout)
        self.dec0_res1 = ResidualBlock(C  + C, C, T, dropout=dropout)
        self.dec0_res2 = ResidualBlock(C  + C, C, T, dropout=dropout)

        # Output head: GroupNorm → SiLU → Conv2d
        self.norm_out = nn.GroupNorm(32, C)
        self.conv_out = nn.Conv2d(C, img_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict the noise ε_θ(x_t, t) added to the data at timestep t.

        Args:
            x : (B, 3, 32, 32)   noisy image x_t
            t : (B,)              integer timestep indices (0-indexed)
        Returns:
            (B, 3, 32, 32)  predicted noise ε_θ
        """
        t_emb = self.time_mlp(t)           # (B, 512)

        # ── Stem ─────────────────────────────────────────────────────────────
        h     = self.conv_in(x)            # (B, 128, 32, 32)
        skip0 = h

        # ── Encoder — Level 0  (32×32) ───────────────────────────────────────
        h     = self.enc0_res0(h, t_emb)
        skip1 = h
        h     = self.enc0_res1(h, t_emb)
        skip2 = h
        h     = self.enc0_down(h)          # (B, 128, 16, 16)
        skip3 = h

        # ── Encoder — Level 1  (16×16, attention) ────────────────────────────
        h     = self.enc1_res0(h, t_emb)
        h     = self.enc1_attn0(h)
        skip4 = h
        h     = self.enc1_res1(h, t_emb)
        h     = self.enc1_attn1(h)
        skip5 = h
        h     = self.enc1_down(h)          # (B, 256,  8,  8)
        skip6 = h

        # ── Encoder — Level 2  (8×8) ─────────────────────────────────────────
        h     = self.enc2_res0(h, t_emb)
        skip7 = h
        h     = self.enc2_res1(h, t_emb)
        skip8 = h
        h     = self.enc2_down(h)          # (B, 256,  4,  4)
        skip9 = h

        # ── Encoder — Level 3  (4×4) ─────────────────────────────────────────
        h      = self.enc3_res0(h, t_emb)
        skip10 = h
        h      = self.enc3_res1(h, t_emb)
        skip11 = h

        # ── Bottleneck ───────────────────────────────────────────────────────
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        # ── Decoder — Level 3  (4×4) ─────────────────────────────────────────
        h = torch.cat([h, skip11], dim=1)
        h = self.dec3_res0(h, t_emb)
        h = torch.cat([h, skip10], dim=1)
        h = self.dec3_res1(h, t_emb)
        h = torch.cat([h, skip9],  dim=1)
        h = self.dec3_res2(h, t_emb)
        h = self.dec3_up(h)                # (B, 256,  8,  8)

        # ── Decoder — Level 2  (8×8) ─────────────────────────────────────────
        h = torch.cat([h, skip8], dim=1)
        h = self.dec2_res0(h, t_emb)
        h = torch.cat([h, skip7], dim=1)
        h = self.dec2_res1(h, t_emb)
        h = torch.cat([h, skip6], dim=1)
        h = self.dec2_res2(h, t_emb)
        h = self.dec2_up(h)                # (B, 256, 16, 16)

        # ── Decoder — Level 1  (16×16, attention) ────────────────────────────
        h = torch.cat([h, skip5], dim=1)
        h = self.dec1_res0(h, t_emb)
        h = self.dec1_attn0(h)
        h = torch.cat([h, skip4], dim=1)
        h = self.dec1_res1(h, t_emb)
        h = self.dec1_attn1(h)
        h = torch.cat([h, skip3], dim=1)   # (B, 384, 16, 16)  ← 256+128
        h = self.dec1_res2(h, t_emb)
        h = self.dec1_attn2(h)
        h = self.dec1_up(h)                # (B, 256, 32, 32)

        # ── Decoder — Level 0  (32×32) ───────────────────────────────────────
        h = torch.cat([h, skip2], dim=1)   # (B, 384, 32, 32)  ← 256+128
        h = self.dec0_res0(h, t_emb)
        h = torch.cat([h, skip1], dim=1)
        h = self.dec0_res1(h, t_emb)
        h = torch.cat([h, skip0], dim=1)
        h = self.dec0_res2(h, t_emb)

        # ── Output ───────────────────────────────────────────────────────────
        return self.conv_out(F.silu(self.norm_out(h)))   # (B, 3, 32, 32)


class NoiseScheduler:
    """
    Precomputes all quantities needed for DDPM training (Algorithm 1)
    and sampling (Algorithm 2).

    All tensors are plain attributes (not nn.Parameters) since the forward
    process variances βt are fixed constants — Section 3.1: "we ignore the
    fact that the forward process variances βt are learnable by
    reparameterization and instead fix them to constants".
    """

    def __init__(
        self,
        T: int = 1000,              # Section 4
        beta_start: float = 1e-4,   # Section 4: β₁ = 10⁻⁴
        beta_end:   float = 0.02,   # Section 4: β_T = 0.02
    ):
        self.T = T

        # Linear variance schedule — Section 4
        betas      = torch.linspace(beta_start, beta_end, T)      # βt
        alphas     = 1.0 - betas                                  # αt = 1 − βt
        alpha_bars = torch.cumprod(alphas, dim=0)                 # ᾱt = ∏αs  [Eq. 4]

        # Precomputed for forward process q(x_t | x_0) — Eq. 4
        sqrt_ab   = torch.sqrt(alpha_bars)                        # √ᾱt
        sqrt_1mab = torch.sqrt(1.0 - alpha_bars)                  # √(1-ᾱt)

        # Precomputed for reverse mean μ_θ — Eq. 11
        sqrt_recip_a       = torch.sqrt(1.0 / alphas)             # 1/√αt
        beta_div_sqrt_1mab = betas / sqrt_1mab                    # βt/√(1-ᾱt)

        # Reverse noise scale — Section 3.2:
        # "both σ²t = βt and σ²t = β̃t had similar results … the first choice
        #  is optimal for x_0 ~ N(0,I)". We use σ_t = √βt.
        sigmas = torch.sqrt(betas)

        self.betas              = betas
        self.alphas             = alphas
        self.alpha_bars         = alpha_bars
        self.sqrt_ab            = sqrt_ab
        self.sqrt_1mab          = sqrt_1mab
        self.sqrt_recip_a       = sqrt_recip_a
        self.beta_div_sqrt_1mab = beta_div_sqrt_1mab
        self.sigmas             = sigmas

    def to(self, device):
        for attr in [
            "betas", "alphas", "alpha_bars",
            "sqrt_ab", "sqrt_1mab",
            "sqrt_recip_a", "beta_div_sqrt_1mab", "sigmas",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(
        self,
        x0:  torch.Tensor,
        t:   torch.Tensor,
        eps: torch.Tensor = None,
    ) -> tuple:
        """
        Sample x_t from x_0 in closed form — Section 2, Eq. 4:

            q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)

        Reparameterized as:
            x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,   ε ~ N(0, I)

        Used in Algorithm 1 (Training), lines 4-5.

        Returns:
            x_t  : noisy image at timestep t
            eps  : the noise that was added (the regression target)
        """
        if eps is None:
            eps = torch.randn_like(x0)

        sqrt_ab   = self.sqrt_ab[t].view(-1, 1, 1, 1)
        sqrt_1mab = self.sqrt_1mab[t].view(-1, 1, 1, 1)

        x_t = sqrt_ab * x0 + sqrt_1mab * eps
        return x_t, eps

    def compute_loss(
        self,
        model: nn.Module,
        x0:   torch.Tensor,
        t:    torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified training objective — Algorithm 1 / Eq. 14:

            L_simple(θ) = E_{t, x_0, ε} [ || ε − ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t)ε, t) ||² ]

        Section 3.4:
            "we found it beneficial to sample quality (and simpler to implement)
             to train on the following variant of the variational bound" [Eq. 14].

        Note: no time-dependent weighting — each timestep contributes equally.
        The paper shows this reweighting leads to better sample quality (Table 2).
        """
        eps               = torch.randn_like(x0)
        x_t, eps_target   = self.q_sample(x0, t, eps)
        eps_pred          = model(x_t, t)
        return F.mse_loss(eps_pred, eps_target)

    @torch.no_grad()
    def p_sample_step(
        self,
        model: nn.Module,
        x_t:  torch.Tensor,
        t_idx: int,
    ) -> torch.Tensor:
        """
        One reverse-process step — Algorithm 2:

            z ~ N(0, I)  if t > 1,  else z = 0              [line 3]
            x_{t-1} = (1/√αt)(x_t − βt/√(1-ᾱt) · ε_θ(x_t,t)) + σt·z  [line 4]

        The reverse mean μ_θ is from Eq. 11 (ε-prediction parameterization).
        σ_t = √βt (Section 3.2, first choice).

        Args:
            model : UNet  (ε_θ)
            x_t   : (B, C, H, W) noisy image at step t_idx
            t_idx : int, 0-indexed timestep (0 = last denoising step)
        Returns:
            x_{t-1} : (B, C, H, W)
        """
        B        = x_t.shape[0]
        t_tensor = torch.full((B,), t_idx, device=x_t.device, dtype=torch.long)

        eps_pred = model(x_t, t_tensor)

        c1   = self.sqrt_recip_a[t_idx]
        c2   = self.beta_div_sqrt_1mab[t_idx]
        mean = c1 * (x_t - c2 * eps_pred)

        # Algorithm 2, line 3: z = 0 at the final step (t_idx == 0)
        if t_idx == 0:
            return mean

        z     = torch.randn_like(x_t)
        sigma = self.sigmas[t_idx]
        return mean + sigma * z

    @torch.no_grad()
    def p_sample_loop(
        self,
        model:  nn.Module,
        shape:  tuple,
        device: torch.device,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Full reverse process — Algorithm 2:

            1. x_T ~ N(0, I)
            2. for t = T, ..., 1 do
            3.   z ~ N(0, I) if t > 1, else z = 0
            4.   x_{t-1} = (1/√αt)(x_t − βt/√(1-ᾱt)·ε_θ(x_t,t)) + σt·z
            5. end for
            6. return x_0

        Section 3.3:
            "At the end of sampling, we display μ_θ(x_1, 1) noiselessly."
            (handled by setting z = 0 when t_idx = 0)

        Args:
            model  : UNet (ε_θ)
            shape  : (B, C, H, W)
            device : torch device
        Returns:
            x_0 : (B, C, H, W) generated images in [-1, 1]
        """
        was_training = model.training
        model.eval()

        x = torch.randn(shape, device=device)

        for t in reversed(range(self.T)):
            if verbose and t % 100 == 0:
                print(f"  Sampling step {self.T - t}/{self.T}", end="\r")
            x = self.p_sample_step(model, x, t)

        if verbose:
            print()

        if was_training:
            model.train()

        return x
