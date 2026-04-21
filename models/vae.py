import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Maps an input image to the parameters (mu, log_var) of the latent Gaussian."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        # Two separate heads — one for mean, one for log-variance
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    """Maps a latent vector z back to the image space."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),   # pixel values in [0, 1]
        )

    def forward(self, z: torch.Tensor):
        return self.net(z)


class VAE(nn.Module):
    """
    Full VAE combining Encoder + reparameterization trick + Decoder.

    Args:
        input_dim:       flattened image size (e.g. 28*28 = 784 for MNIST)
        hidden_dim:      width of the hidden layers
        latent_dim:      size of the latent space z
        recon_loss_type: "bce" for binary data (MNIST),  "mse" for continuous data (Frey Face)

    Paper reference — Appendix C:
        C.1 MNIST:     models p(x|z) as Bernoulli  → BCE loss
        C.2 Frey Face: models p(x|z) as Gaussian   → MSE loss
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 400,
                 latent_dim: int = 20, recon_loss_type: str = "bce"):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim
        self.recon_loss_type = recon_loss_type

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        The reparameterization trick:
            z = mu + std * epsilon,  epsilon ~ N(0, I)

        Why? We need gradients to flow through the sampling step.
        By separating the randomness (epsilon) from the learnable
        parameters (mu, std), the gradient w.r.t. mu and std is
        well-defined and backprop works normally.
        """
        std = torch.exp(0.5 * log_var)          # sigma = exp(log_var / 2)
        eps = torch.randn_like(std)              # eps ~ N(0, I)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        # Flatten image: (B, C, H, W) → (B, input_dim)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        # Encode
        mu, log_var = self.encoder(x_flat)

        # Sample latent vector
        z = self.reparameterize(mu, log_var)

        # Decode
        x_recon = self.decoder(z)

        return x_recon, mu, log_var

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Generate n images by sampling z ~ N(0, I) and decoding."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)

    def loss(self, 
             x_recon: torch.Tensor, 
             x: torch.Tensor,
             mu: torch.Tensor, 
             log_var: torch.Tensor) -> dict:
        """
        ELBO loss = Reconstruction loss + KL divergence.

        Reconstruction loss depends on the dataset (paper Appendix C):
          - BCE for MNIST:     p(x|z) = Bernoulli(f(z))
                               pixels are binary → log-likelihood = BCE
          - MSE for Frey Face: p(x|z) = N(f(z), I)
                               pixels are continuous → log-likelihood = -MSE (up to constant)

        KL divergence (Appendix B — closed form for two Gaussians):
            KL(N(mu, sigma^2) || N(0,1)) =
                -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        if self.recon_loss_type == "bce":
            recon_loss = F.binary_cross_entropy(x_recon, x_flat, reduction="sum") / batch_size
        else:  # mse — for continuous data like Frey Face
            recon_loss = F.mse_loss(x_recon, x_flat, reduction="sum") / batch_size

        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size

        total = recon_loss + kl_loss
        return {"loss": total, "recon_loss": recon_loss, "kl_loss": kl_loss}
