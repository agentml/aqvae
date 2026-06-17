"""dSprites용 (β-)VAE 모델.

아키텍처는 Higgins et al. 2017 (β-VAE) / Burgess et al. 2018 의 표준 설정.
  - 입력: (B, 1, 64, 64) 이진 이미지.
  - 인코더: conv 4단(stride 2) -> FC -> (mu, logvar).
  - 디코더: 대칭 deconv. Bernoulli 출력(logits).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEOutput:
    recon_logits: torch.Tensor   # (B, 1, 64, 64)
    mu: torch.Tensor             # (B, latent_dim)
    logvar: torch.Tensor         # (B, latent_dim)
    z: torch.Tensor              # (B, latent_dim)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 10, hidden: int = 256,
                 in_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        # 인코더: 64 -> 32 -> 16 -> 8 -> 4
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU(True),    # 32x32
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True),   # 16x16
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),   # 8x8
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(True),   # 4x4
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, hidden), nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

        # 디코더: 4 -> 8 -> 16 -> 32 -> 64
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, 64 * 4 * 4), nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(True),   # 8x8
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),   # 16x16
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(True),   # 32x32
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),         # 64x64 (logits)
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_input(z).view(-1, 64, 4, 4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> VAEOutput:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z)
        return VAEOutput(recon_logits=recon_logits, mu=mu, logvar=logvar, z=z)


def vae_loss(out: VAEOutput, x: torch.Tensor, beta: float = 1.0,
             recon_type: str = "bernoulli"):
    """ELBO 음수. 재구성 + KL(표준정규 대비). 배치 평균으로 반환.

    recon_type:
      - "bernoulli": 이진/[0,1] 입력. BCE-with-logits(픽셀 합). (dSprites 기본)
      - "gaussian":  연속 RGB 입력. 고정분산 Gaussian = MSE(sigmoid(logits), x) 픽셀 합. (3D Shapes)
    recon/kl 개별 항도 함께 반환(로깅용).
    """
    B = x.size(0)
    if recon_type == "bernoulli":
        recon = F.binary_cross_entropy_with_logits(
            out.recon_logits, x, reduction="sum") / B
    elif recon_type == "gaussian":
        recon = F.mse_loss(
            torch.sigmoid(out.recon_logits), x, reduction="sum") / B
    else:
        raise ValueError(f"unknown recon_type: {recon_type}")
    # KL( q(z|x) || N(0,I) ) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * torch.sum(
        1 + out.logvar - out.mu.pow(2) - out.logvar.exp()
    ) / B
    loss = recon + beta * kl
    return loss, recon, kl
