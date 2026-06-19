"""VQ-VAE 베이스라인 (이산 표현 비교 앵커).

attractor 모델과 동일한 factor 구조로 맞춘 **product(grouped) 양자화**:
잠재 z(=32)를 n_groups=8 × group_dim=4 로 쪼개고, 각 그룹을 codebook(K=16, 4D)의
최근접 코드로 양자화. straight-through estimator로 gradient 통과(← 본 연구가 대체하려는 방식).

van den Oord et al. 2017. conv enc/dec 아키텍처는 src/model_vae.py 와 동일.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQVAEOutput:
    recon_logits: torch.Tensor   # (B, C, 64, 64)
    z_e: torch.Tensor            # (B, n_groups, group_dim)  인코더 연속 출력
    z_q: torch.Tensor            # (B, n_groups, group_dim)  양자화(STE) 결과
    indices: torch.Tensor        # (B, n_groups)  선택된 코드 인덱스


class ProductVQ(nn.Module):
    """n_groups 개의 독립 codebook. 각 그룹 group_dim 차원, 코드 K개."""

    def __init__(self, n_groups: int, group_dim: int, codebook_size: int):
        super().__init__()
        self.n_groups = n_groups
        self.group_dim = group_dim
        self.codebook_size = codebook_size
        # (n_groups, K, group_dim)
        self.codebook = nn.Parameter(torch.randn(n_groups, codebook_size, group_dim) * 0.1)

    def forward(self, z_e: torch.Tensor):
        # z_e: (B, n_groups, group_dim)
        # 그룹별 최근접 코드: dist (B, n_groups, K)
        cb = self.codebook                                   # (G, K, D)
        d = (z_e.unsqueeze(2) - cb.unsqueeze(0)).pow(2).sum(-1)  # (B, G, K)
        indices = d.argmin(-1)                               # (B, G)
        z_q = torch.gather(
            cb.unsqueeze(0).expand(z_e.size(0), -1, -1, -1), 2,
            indices[..., None, None].expand(-1, -1, 1, self.group_dim)
        ).squeeze(2)                                         # (B, G, D)
        return z_q, indices


class VQVAE(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden: int = 256, in_channels: int = 3,
                 n_groups: int = 8, codebook_size: int = 16, commitment: float = 0.25):
        super().__init__()
        assert latent_dim % n_groups == 0, "latent_dim 은 n_groups 로 나눠떨어져야 함"
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.n_groups = n_groups
        self.group_dim = latent_dim // n_groups
        self.commitment = commitment

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, hidden), nn.ReLU(True),
            nn.Linear(hidden, latent_dim),
        )
        self.vq = ProductVQ(n_groups, self.group_dim, codebook_size)
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, 64 * 4 * 4), nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> VQVAEOutput:
        B = x.size(0)
        z_e = self.encoder(x).view(B, self.n_groups, self.group_dim)
        z_q, indices = self.vq(z_e)
        # straight-through: forward는 z_q, backward는 z_e 로 gradient 복사
        z_st = z_e + (z_q - z_e).detach()
        recon_logits = self.decoder(self.decoder_input(z_st.view(B, -1)).view(-1, 64, 4, 4))
        return VQVAEOutput(recon_logits=recon_logits, z_e=z_e, z_q=z_q, indices=indices)


def vqvae_loss(out: VQVAEOutput, x: torch.Tensor, commitment: float = 0.25,
               recon_type: str = "gaussian"):
    """recon + codebook + commitment. KL 없음. 배치 평균."""
    B = x.size(0)
    if recon_type == "gaussian":
        recon = F.mse_loss(torch.sigmoid(out.recon_logits), x, reduction="sum") / B
    elif recon_type == "bernoulli":
        recon = F.binary_cross_entropy_with_logits(out.recon_logits, x, reduction="sum") / B
    else:
        raise ValueError(f"unknown recon_type: {recon_type}")
    # codebook: 코드를 z_e 로, commitment: z_e 를 코드로
    codebook_loss = F.mse_loss(out.z_q, out.z_e.detach(), reduction="sum") / B
    commit_loss = F.mse_loss(out.z_e, out.z_q.detach(), reduction="sum") / B
    vq = codebook_loss + commitment * commit_loss
    loss = recon + vq
    return loss, recon, vq
