"""
Quantitative evaluation on CIFAR-10.

Metrics:
  FID  — Fréchet Inception Distance  (lower is better)
  IS   — Inception Score             (higher is better)

Usage:
    python evaluate.py --model ddpm     --ckpt outputs/ddpm_cifar10/ddpm_best.pt
    python evaluate.py --model conv_vae --ckpt outputs/conv_vae_cifar10/conv_vae_best.pt
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from models import UNet, NoiseScheduler, DDPM, ConvVAE, VAE
from config import DDPM_CONFIG, CONV_VAE_CONFIG, VAE_CONFIG, DEVICE, DATA_DIR, OUTPUT_DIR

DDPM_CKPT_PATH     = os.path.join(OUTPUT_DIR, "ddpm_cifar10",      "ddpm_best.pt")
CONV_VAE_CKPT_PATH = os.path.join(OUTPUT_DIR, "conv_vae_cifar10",  "conv_vae_best.pt")
VAE_CKPT_PATH      = os.path.join(OUTPUT_DIR, "vae_mnist",         "vae_best.pt")


def load_ddpm(ckpt_path: str = None):
    cfg  = DDPM_CONFIG
    unet = UNet(
        img_channels=cfg["img_channels"],
        base_channels=cfg["base_channels"],
        dropout=0.0,
    ).to(DEVICE)

    scheduler = NoiseScheduler(
        T=cfg["T"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
    ).to(DEVICE)

    path = ckpt_path or DDPM_CKPT_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No DDPM checkpoint at {path}.\n"
            f"Run: python train.py --model ddpm --dataset cifar10"
        )

    ckpt = torch.load(path, map_location=DEVICE)
    unet.load_state_dict(ckpt["unet"])

    model = DDPM(unet, scheduler)
    model.eval()
    print(f"Loaded DDPM from {path}  (epoch {ckpt.get('epoch', '?')})")

    img_shape = cfg["img_shape"]
    return model, lambda n: model.sample(n, img_shape, DEVICE)


def load_conv_vae(ckpt_path: str = None):
    cfg  = CONV_VAE_CONFIG
    path = ckpt_path or CONV_VAE_CKPT_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No ConvVAE checkpoint at {path}.\n"
            f"Run: python train.py --model conv_vae --dataset cifar10"
        )

    model = ConvVAE(
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"],
    ).to(DEVICE)

    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded ConvVAE from {path}  (epoch {ckpt.get('epoch', '?')})")

    return model, lambda n: model.sample(n, DEVICE)


def load_vae(ckpt_path: str = None):
    cfg  = VAE_CONFIG
    path = ckpt_path or VAE_CKPT_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No VAE checkpoint at {path}.\n"
            f"Run: python train.py --model vae --dataset mnist"
        )

    model = VAE(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        recon_loss_type=cfg["recon_loss_type"],
    ).to(DEVICE)

    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded VAE from {path}  (epoch {ckpt.get('epoch', '?')})")

    # VAE outputs flat [0,1] vectors → reshape to (B,1,28,28) → repeat to RGB
    def sample_fn(n):
        flat = model.sample(n, DEVICE)                     # (B, 784) in [0,1]
        imgs = flat.view(n, 1, 28, 28).repeat(1, 3, 1, 1) # (B, 3, 28, 28)
        return imgs * 2 - 1                                # [0,1] → [-1,1]

    return model, sample_fn


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] float tensor to uint8 [0, 255] as required by torchmetrics."""
    return ((x.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)


def evaluate(sample_fn, n_samples: int = 5000, real_loader=None):
    """
    Compute FID and IS for any model.

    Args:
        sample_fn   : callable(n: int) -> (n, 3, H, W) tensor in [-1, 1]
        n_samples   : number of generated samples to use
        real_loader : DataLoader of real images in [0, 1]; defaults to CIFAR-10
    """
    fid_metric = FrechetInceptionDistance(normalize=False).to(DEVICE)
    is_metric  = InceptionScore(normalize=False).to(DEVICE)

    # ── Real images ─────────────────────────────────────────────────────
    if real_loader is None:
        print("Loading real CIFAR-10 images...")
        real_loader = DataLoader(
            datasets.CIFAR10(DATA_DIR, train=False, transform=transforms.ToTensor(), download=True),
            batch_size=128, shuffle=True,
        )
    collected = 0
    for x, _ in real_loader:
        # x may be grayscale (B,1,H,W) — repeat to RGB for InceptionV3
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        imgs = (x * 255).to(torch.uint8).to(DEVICE)
        fid_metric.update(imgs, real=True)
        collected += len(x)
        if collected >= n_samples:
            break

    # ── Generated images ────────────────────────────────────────────────
    print(f"Generating {n_samples} samples...")
    collected = 0
    while collected < n_samples:
        batch   = min(64, n_samples - collected)
        samples = to_uint8(sample_fn(batch))
        fid_metric.update(samples, real=False)
        is_metric.update(samples)
        collected += batch
        print(f"  {collected}/{n_samples}", end="\r")
    print()

    fid = fid_metric.compute().item()
    is_mean, is_std = is_metric.compute()
    return fid, is_mean.item(), is_std.item()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generative model metrics")
    parser.add_argument("--model",     choices=["ddpm", "conv_vae", "vae"], default="ddpm")
    parser.add_argument("--ckpt",      type=str, default=None, help="Path to checkpoint (.pt)")
    parser.add_argument("--n_samples", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Device : {DEVICE}")
    print(f"Model  : {args.model}\n")

    real_loader = None

    if args.model == "ddpm":
        model, sample_fn = load_ddpm(args.ckpt)
        out_dir = os.path.join(OUTPUT_DIR, "ddpm_cifar10")
    elif args.model == "conv_vae":
        model, sample_fn = load_conv_vae(args.ckpt)
        out_dir = os.path.join(OUTPUT_DIR, "conv_vae_cifar10")
    else:  # vae — MNIST
        model, sample_fn = load_vae(args.ckpt)
        out_dir = os.path.join(OUTPUT_DIR, "vae_mnist")
        print("Using MNIST test set as real distribution for FID/IS.")
        real_loader = DataLoader(
            datasets.MNIST(DATA_DIR, train=False, transform=transforms.ToTensor(), download=True),
            batch_size=128, shuffle=True,
        )

    fid, is_mean, is_std = evaluate(sample_fn, n_samples=args.n_samples, real_loader=real_loader)

    print(f"\nFID  : {fid:.2f}  (lower is better)")
    print(f"IS   : {is_mean:.2f} ± {is_std:.2f}  (higher is better)")
    print(f"\nNote: paper-comparable scores use --n_samples 50000\n")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics.txt")
    with open(out_path, "w") as f:
        f.write(f"model     : {args.model}\n")
        f.write(f"n_samples : {args.n_samples}\n")
        f.write(f"FID       : {fid:.2f}\n")
        f.write(f"IS        : {is_mean:.2f} +/- {is_std:.2f}\n")
    print(f"Saved → {out_path}")
