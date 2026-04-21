"""
Central configuration for all models and training.
Change hyperparameters here — nowhere else.
"""

import torch

# ---------------------------------------------------------------------------
# DDPM — CIFAR-10 32×32 RGB
#
# All hyperparameters are taken directly from the paper:
#
# Architecture — Appendix B:
#   "Our neural network architecture follows the backbone of PixelCNN++ [52],
#    which is a U-Net [48] based on a Wide ResNet [72]."
#   "Our 32×32 models use four feature map resolutions (32×32 to 4×4)."
#   "All models have two convolutional residual blocks per resolution level."
#   "Self-attention at the 16×16 feature map resolution."
#   "We set the dropout rate on CIFAR10 to 0.1."
#
# Noise schedule — Section 4:
#   "We set T = 1000 for all experiments."
#   "We set the forward process variances to constants increasing linearly
#    from β₁ = 10⁻⁴ to β_T = 0.02."
#
# Optimization — Appendix B:
#   "We tried Adam and RMSProp … and chose the former."
#   "We set the learning rate to 2×10⁻⁴."
#   "We set the batch size to 128 for CIFAR10."
#   "We used random horizontal flips during training for CIFAR10."
# ---------------------------------------------------------------------------
DDPM_CONFIG = {
    # ── Architecture (Appendix B) ──────────────────────────────────────────
    "img_channels":    3,             # RGB
    "img_size":        32,            # CIFAR-10: 32×32
    "img_shape":       (3, 32, 32),
    "base_channels":   128,           # Appendix B: original paper value
    "channel_mult":    (1, 2, 2, 2),  # four resolutions: 32→16→8→4
    "num_res_blocks":  2,             # two residual blocks per resolution level
    "attn_resolutions": (16,),        # self-attention at 16×16 only
    "dropout":         0.1,           # Section 4: dropout rate on CIFAR-10

    # ── Noise schedule (Section 4) ─────────────────────────────────────────
    "T":               1000,          # total diffusion steps
    "beta_start":      1e-4,          # β₁ = 10⁻⁴
    "beta_end":        0.02,          # β_T = 0.02

    # ── Optimization (Appendix B) ──────────────────────────────────────────
    "lr":              2e-4,          # learning rate 2×10⁻⁴
    "batch_size":      128,           # Appendix B: original paper value
    "epochs":          500,           # Appendix B: original paper value
    # ── Sampling ───────────────────────────────────────────────────────────
    "sample_every":    10,            # generate samples every N epochs
    "n_samples":       16,            # Appendix B: original paper value 64 but we make 16 for speed
}

# ---------------------------------------------------------------------------
# Shared settings
# ---------------------------------------------------------------------------
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR   = "./data"
OUTPUT_DIR = "./outputs"
