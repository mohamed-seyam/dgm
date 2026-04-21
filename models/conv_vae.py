"""
Convolutional VAE for CIFAR-10 (32×32 RGB).

Same ELBO objective and reparameterization trick as the MLP VAE — only the
encoder/decoder architecture changes to handle spatial image structure.

Spatial progression (32×32 input, 3 stride-2 convolutions):
  Encoder: (B, 3, 32, 32) → (B, 32, 16, 16) → (B, 64, 8, 8) → (B, 128, 4, 4)
           → flatten (B, 2048) → Linear → mu, log_var
  Decoder: z → Linear → (B, 128, 4, 4) → (B, 64, 8, 8) → (B, 32, 16, 16)
           → (B, 3, 32, 32) → Tanh  (matches CIFAR-10 [-1, 1] normalization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.conv = nn.Sequential(
            # (B, 3, 32, 32) → (B, 32, 16, 16)
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # (B, 32, 16, 16) → (B, 64, 8, 8)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # (B, 64, 8, 8) → (B, 128, 4, 4)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc    = nn.Linear(128 * 4 * 4, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_lv = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        h = F.relu(self.fc(h))
        return self.fc_mu(h), self.fc_lv(h)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128 * 4 * 4),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            # (B, 128, 4, 4) → (B, 64, 8, 8)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # (B, 64, 8, 8) → (B, 32, 16, 16)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # (B, 32, 16, 16) → (B, 3, 32, 32)
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # output in [-1, 1] — matches CIFAR-10 normalization
        )

    def forward(self, z: torch.Tensor):
        h = self.fc(z)
        h = h.view(h.size(0), 128, 4, 4)
        return self.deconv(h)


class ConvVAE(nn.Module):
    """
    Convolutional VAE for 32×32 RGB images.

    Args:
        latent_dim : size of the latent space z
        hidden_dim : FC layer width between conv features and latent space
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.encoder    = ConvEncoder(latent_dim, hidden_dim)
        self.decoder    = ConvDecoder(latent_dim, hidden_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        z           = self.reparameterize(mu, log_var)
        x_recon     = self.decoder(z)
        return x_recon, mu, log_var

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)

    def loss(self, x_recon: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, log_var: torch.Tensor) -> dict:
        """
        ELBO = MSE reconstruction + KL divergence.

        MSE because CIFAR-10 pixels are continuous (normalized to [-1, 1]).
        KL closed form: -½ Σ(1 + log_var - mu² - exp(log_var))
        """
        batch_size = x.size(0)
        recon_loss = F.mse_loss(x_recon, x, reduction="sum") / batch_size
        kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
        total      = recon_loss + kl_loss
        return {"loss": total, "recon_loss": recon_loss, "kl_loss": kl_loss}
