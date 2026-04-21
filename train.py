"""
Training entry point.

Usage:
    python train.py --model ddpm    --dataset cifar10
"""

import os
import argparse
import torch
import torch.optim as optim
from torchvision.utils import save_image

from models import DDPM, UNet, NoiseScheduler
from dataset import get_cifar10_loaders
from config import DDPM_CONFIG, DEVICE, DATA_DIR, OUTPUT_DIR


def train_one_epoch(model, optimizer, loader, epoch):
    model.train()
    total_loss = total_recon = total_kl = 0.0

    for x, _ in loader:
        x = x.to(DEVICE)
        x_recon, mu, log_var = model(x)
        losses = model.loss(x_recon, x, mu, log_var)

        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()

        total_loss  += losses["loss"].item()
        total_recon += losses["recon_loss"].item()
        total_kl    += losses["kl_loss"].item()

    n = len(loader)
    print(
        f"Epoch {epoch:3d} | "
        f"Loss: {total_loss/n:.2f} | "
        f"Recon: {total_recon/n:.2f} | "
        f"KL: {total_kl/n:.2f}"
    )
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, epoch, out_dir, img_shape):
    model.eval()
    total_loss = 0.0
    for x, _ in loader:
        x = x.to(DEVICE)
        x_recon, mu, log_var = model(x)
        total_loss += model.loss(x_recon, x, mu, log_var)["loss"].item()

    avg_loss = total_loss / len(loader)
    print(f"           Test  Loss: {avg_loss:.2f}")

    # Reconstruction grid
    x_sample, _ = next(iter(loader))
    x_sample = x_sample[:8].to(DEVICE)
    x_recon, _, _ = model(x_sample)
    comparison = torch.cat([
        x_sample.view(-1, *img_shape),
        x_recon.view(-1, *img_shape),
    ])
    save_image(comparison, f"{out_dir}/recon_epoch_{epoch:03d}.png", nrow=8)

    # Generation grid
    samples = model.sample(64, DEVICE).view(-1, *img_shape)
    save_image(samples, f"{out_dir}/sample_epoch_{epoch:03d}.png", nrow=8)

    return avg_loss



def train_ddpm():
    """
    Train DDPM on CIFAR-10 (32×32 RGB), following the paper exactly.

    ── Algorithm 1 (Training) — Section 3.4 / Eq. 14 ──────────────────────
    repeat:
        x_0  ~ q(x_0)                                # sample from data
        t    ~ Uniform({1, ..., T})                  # random timestep
        ε    ~ N(0, I)                               # random noise
        gradient step on ∇_θ ||ε − ε_θ(√ᾱt·x_0 + √(1-ᾱt)·ε, t)||²
    until converged

    ── Key hyperparameters (all from the paper) ──────────────────────────
    T           = 1000           (Section 4)
    β₁→β_T      = 1e-4 → 0.02    (Section 4, linear schedule)
    lr          = 2×10⁻⁴         (Appendix B)
    batch_size  = 128            (Appendix B)
    optimizer   = Adam           (Appendix B)
    dropout     = 0.1            (Section 4)
    horiz. flip = True           (Appendix B)

    ── Architecture ──────────────────────────────────────────────────────
    U-Net with base_channels=128, channel_mult=(1,2,2,2),
    two res blocks per level, self-attention at 16×16.  (Appendix B)
    """
    cfg     = DDPM_CONFIG
    out_dir = os.path.join(OUTPUT_DIR, "ddpm_cifar10")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Device     : {DEVICE}")
    print(f"Dataset    : CIFAR-10  (32×32 RGB)")
    print(f"T          : {cfg['T']}  steps  (Section 4)")
    print(f"β schedule : {cfg['beta_start']} → {cfg['beta_end']}  linear  (Section 4)")
    print(f"lr         : {cfg['lr']}  (Appendix B)")
    print(f"batch_size : {cfg['batch_size']}  (Appendix B)")

    # ── Data ────────────────────────────────────────────────────────────
    # Section 3.3: images scaled to [−1, 1] (done inside get_cifar10_loaders)
    # Appendix B: horizontal flips applied during training
    train_loader, _ = get_cifar10_loaders(DATA_DIR, cfg["batch_size"])

    # ── Noise scheduler ─────────────────────────────────────────────────
    # Section 4: linear β schedule, T=1000
    scheduler = NoiseScheduler(
        T=cfg["T"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
    ).to(DEVICE)

    # ── U-Net (ε_θ) ─────────────────────────────────────────────────────
    # Appendix B: architecture for 32×32 CIFAR-10
    unet = UNet(
        img_channels=cfg["img_channels"],
        base_channels=cfg["base_channels"],
        dropout=cfg["dropout"],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in unet.parameters())
    print(f"UNet params: {n_params / 1e6:.1f}M  "
          f"(paper CIFAR-10 model: 35.7M — Appendix B)\n")

    # ── DDPM wrapper ─────────────────────────────────────────────────────
    model = DDPM(unet, scheduler)

    # ── Optimizer ───────────────────────────────────────────────────────
    # Appendix B: "We tried Adam and RMSProp early on … and chose the former."
    # "We set the learning rate to 2×10⁻⁴."
    optimizer = optim.Adam(unet.parameters(), lr=cfg["lr"])

    best_loss = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        # ── One training epoch (Algorithm 1) ────────────────────────────
        unet.train()
        total_loss = 0.0

        for x, _ in train_loader:
            x = x.to(DEVICE)                           # x_0 ~ q(x_0)

            # t ~ Uniform({0, ..., T-1})
            # Paper uses 1-indexed Uniform({1,...,T}); we use 0-indexed equivalent
            t = torch.randint(0, cfg["T"], (x.shape[0],), device=DEVICE)

            # L_simple = ||ε − ε_θ(√ᾱt·x_0 + √(1-ᾱt)·ε, t)||²  [Eq. 14]
            loss = model(x, t)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping (common practice for stable diffusion training)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:4d}/{cfg['epochs']} | Loss: {avg_loss:.5f}")

        # ── Save checkpoint ─────────────────────────────────────────────
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = {
                "epoch":      epoch,
                "unet":       unet.state_dict(),

                "optimizer":  optimizer.state_dict(),
                "loss":       best_loss,
            }
            torch.save(ckpt, os.path.join(out_dir, "ddpm_best.pt"))

        # Save latest checkpoint every 10 epochs for resuming
        if epoch % 10 == 0:
            ckpt = {
                "epoch":      epoch,
                "unet":       unet.state_dict(),

                "optimizer":  optimizer.state_dict(),
                "loss":       avg_loss,
            }
            torch.save(ckpt, os.path.join(out_dir, "ddpm_latest.pt"))

        # ── Generate samples ─────────────────────────────────────────────
        # Algorithm 2: x_T ~ N(0,I), iteratively denoise to x_0
        if epoch % cfg["sample_every"] == 0 or epoch == 1:
            print(f"           Generating {cfg['n_samples']} samples "
                  f"(Algorithm 2)...")
            samples = model.sample(
                n=cfg["n_samples"],
                img_shape=cfg["img_shape"],
                device=DEVICE,
                verbose=True,
            )
            # Rescale from [−1, 1] → [0, 1] for saving
            # Section 3.3: data was scaled to [−1,1]; reverse here for display
            samples = (samples.clamp(-1, 1) + 1) / 2
            save_image(
                samples,
                os.path.join(out_dir, f"samples_epoch_{epoch:04d}.png"),
                nrow=8,
            )
            print(f"           Saved → samples_epoch_{epoch:04d}.png")

    print(f"\nDone. Best loss: {best_loss:.5f}")
    print(f"Outputs saved to: {out_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train generative models")
    parser.add_argument("--model",   choices=["ddpm"], default="ddpm")
    parser.add_argument("--dataset", choices=["cifar10"], default="cifar10")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model == "ddpm":
        train_ddpm()

    else:
        raise ValueError(f"Unknown model: {args.model}")

