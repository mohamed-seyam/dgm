import torch
import time
from models import ConvVAE, DDPM, UNet, NoiseScheduler
from config import DEVICE

print("Loading models...")

# Create models
vae = ConvVAE(latent_dim=128, hidden_dim=256).to(DEVICE).eval()
unet = UNet(img_channels=3, base_channels=128, dropout=0.1).to(DEVICE)
scheduler = NoiseScheduler(T=1000, beta_start=1e-4, beta_end=0.02).to(DEVICE)
ddpm = DDPM(unet, scheduler)

# ===== SIZE COMPARISON =====
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

vae_params = count_parameters(vae)
unet_params = count_parameters(unet)
vae_size_mb = vae_params * 4 / (1024**2)
unet_size_mb = unet_params * 4 / (1024**2)

print(f"\n{'='*60}")
print(f"MODEL SIZE COMPARISON")
print(f"{'='*60}")
print(f"ConvVAE:")
print(f"  Parameters:  {vae_params:,} ({vae_params/1e6:.1f}M)")
print(f"  Checkpoint:  ~{vae_size_mb:.1f} MB")
print(f"\nDDPM U-Net:")
print(f"  Parameters:  {unet_params:,} ({unet_params/1e6:.1f}M)")
print(f"  Checkpoint:  ~{unet_size_mb:.1f} MB")
print(f"\nRatio: DDPM is {unet_params/vae_params:.1f}x LARGER")
print(f"{'='*60}\n")

# ===== SPEED COMPARISON =====
n_samples = 16
print(f"Testing sampling speed ({n_samples} images)...\n")

# VAE speed
with torch.no_grad():
    start = time.time()
    z = torch.randn(n_samples, 128).to(DEVICE)
    vae_samples = vae.decode(z)
    vae_time = time.time() - start

# DDPM speed
with torch.no_grad():
    start = time.time()
    ddpm_samples = ddpm.sample(n_samples, (3, 32, 32), DEVICE, verbose=False)
    ddpm_time = time.time() - start

print(f"{'='*60}")
print(f"SAMPLING SPEED COMPARISON ({n_samples} images)")
print(f"{'='*60}")
print(f"VAE:   {vae_time:.3f}s ({vae_time/n_samples*1000:.1f}ms per image)")
print(f"DDPM:  {ddpm_time:.1f}s ({ddpm_time/n_samples:.2f}s per image)")
print(f"Speedup: DDPM is {ddpm_time/vae_time:.0f}x SLOWER than VAE")
print(f"{'='*60}\n")