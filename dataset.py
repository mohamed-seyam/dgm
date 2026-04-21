"""
Dataset loaders for all experiments.

Datasets:
    - MNIST      (VAE paper Appendix C.1)  — 28×28 grayscale → BCE loss
    - CIFAR-10   (DDPM paper Appendix B)   — 32×32 RGB → MSE loss + DDPM

"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(data_dir: str, batch_size: int):
    """
    MNIST: 60,000 train / 10,000 test, 28×28 grayscale, pixel values in [0, 1].
    ToTensor() handles [0, 255] uint8 → [0.0, 1.0] float32.
    """
    transform = transforms.ToTensor()
    train_loader = DataLoader(
        datasets.MNIST(data_dir, train=True,  transform=transform, download=True),
        batch_size=batch_size, shuffle=True,  num_workers=2,
    )
    test_loader = DataLoader(
        datasets.MNIST(data_dir, train=False, transform=transform, download=True),
        batch_size=batch_size, shuffle=False, num_workers=2,
    )
    print(f"MNIST: 60,000 train / 10,000 test  (28×28 grayscale, [0, 1])")
    return train_loader, test_loader


def get_cifar10_loaders(data_dir: str, batch_size: int):
    """
    CIFAR-10: 50,000 train / 10,000 test, 32×32 RGB.

    Train transform:
      - RandomHorizontalFlip: Appendix B augmentation for CIFAR-10
      - ToTensor:             [0,255] uint8 → [0.0,1.0] float32
      - Normalize(0.5, 0.5): [0,1] → [−1,1]  (Section 3.3)

    Test transform:
      - ToTensor + Normalize only (no flips for evaluation)
    """
    # Section 3.3: images scaled linearly to [−1, 1]
    normalize = transforms.Normalize(
        mean=(0.5, 0.5, 0.5),   # per-channel mean (maps 0.5 → 0)
        std=(0.5, 0.5, 0.5),    # per-channel std  (maps [0,1] → [−1,1])
    )

    train_transform = transforms.Compose([
        # Appendix B: "We used random horizontal flips during training for CIFAR10"
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_loader = DataLoader(
        datasets.CIFAR10(data_dir, train=True,  transform=train_transform, download=True),
        batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        datasets.CIFAR10(data_dir, train=False, transform=test_transform,  download=True),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    print(f"CIFAR-10: 50,000 train / 10,000 test  (32×32 RGB, normalized to [−1, 1])")
    return train_loader, test_loader
