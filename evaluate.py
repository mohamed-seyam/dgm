"""
Quantitative evaluation of DDPM on CIFAR-10.

Metrics:
  FID  — Fréchet Inception Distance   (paper target: 3.17,       lower is better)
  IS   — Inception Score              (paper target: 9.46±0.11,  higher is better)

  Both from DDPM paper Table 1, computed on 50,000 samples (Appendix B).
  Use --n_samples 50000 for paper-comparable results; default 5,000 for speed.

Usage:
    python evaluate.py --n_samples 5000
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from models import UNet, NoiseScheduler, DDPM
from config import DDPM_CONFIG, DEVICE, DATA_DIR, OUTPUT_DIR

DDPM_CKPT_PATH = os.path.join(OUTPUT_DIR, "ddpm_cifar10", "ddpm_best.pt")


def load_ddpm(ckpt_path: str = None):
    
    cfg = DDPM_CONFIG

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
    print(f"Loaded DDPM from {DDPM_CKPT_PATH}  (epoch {ckpt.get('epoch', '?')})")
    return model


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] float tensor to uint8 [0, 255] as required by torchmetrics."""
    return ((x.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)


def evaluate(model: DDPM, n_samples: int = 5000):
    cfg = DDPM_CONFIG

    fid_metric = FrechetInceptionDistance(normalize=False).to(DEVICE)
    is_metric  = InceptionScore(normalize=False).to(DEVICE)

    # ── Real CIFAR-10 images ────────────────────────────────────────────
    print("Loading real CIFAR-10 images...")
    loader = DataLoader(
        datasets.CIFAR10(DATA_DIR, train=False, transform=transforms.ToTensor(), download=True),
        batch_size=128, shuffle=True,
    )
    collected = 0
    for x, _ in loader:
        imgs = (x * 255).to(torch.uint8).to(DEVICE)
        fid_metric.update(imgs, real=True)
        collected += len(x)
        if collected >= n_samples:
            break

    # ── Generated images ────────────────────────────────────────────────
    print(f"Generating {n_samples} samples...")
    collected = 0
    while collected < n_samples:
        batch = min(64, n_samples - collected)
        samples = to_uint8(model.sample(batch, cfg["img_shape"], DEVICE))
        fid_metric.update(samples, real=False)
        is_metric.update(samples)
        collected += batch
        print(f"  {collected}/{n_samples}", end="\r")
    print()

    fid = fid_metric.compute().item()
    is_mean, is_std = is_metric.compute()
    return fid, is_mean.item(), is_std.item()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DDPM metrics on CIFAR-10")
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (.pt)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Device: {DEVICE}")
    print("DDPM paper targets (Table 1): IS=9.46±0.11, FID=3.17\n")

    model = load_ddpm(args.ckpt)
    fid, is_mean, is_std = evaluate(model, n_samples=args.n_samples)

    print(f"\nFID  : {fid:.2f}  (paper: 3.17,  lower is better)")
    print(f"IS   : {is_mean:.2f} ± {is_std:.2f}  (paper: 9.46±0.11, higher is better)")
    print(f"\nNote: paper scores use 50,000 samples. Use --n_samples 50000 for comparable results.\n")

    out_path = os.path.join(OUTPUT_DIR, "ddpm_cifar10", "metrics.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"n_samples : {args.n_samples}\n")
        f.write(f"FID       : {fid:.2f}\n")
        f.write(f"IS        : {is_mean:.2f} +/- {is_std:.2f}\n")
    print(f"Saved → {out_path}")
