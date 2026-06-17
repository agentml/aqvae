"""Attractor-AE (deterministic): 완전 미분가능 이산 표현.

conv encoder → context c(R^32) → FactorizedHopfield 고정점 y*(≈이산, 미분가능)
→ conv decoder. KL/샘플링 없음(VQ-VAE 와 같은 비변분 틀). 이산화는 attractor 동역학이
담당하고, 그 gradient 는 IFT 로 *정확*(STE 불필요).

비교: VQ-VAE 는 argmin+STE, 본 모델은 고정점 y* 를 그대로 디코딩(미분 가능).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fixed_point.multistable import FactorizedHopfield


@dataclass
class AttractorOutput:
    recon_logits: torch.Tensor   # (B, C, 64, 64)
    c: torch.Tensor              # (B, latent_dim)  encoder context
    y_star: torch.Tensor         # (B, latent_dim)  고정점(디코더 입력)
    indices: torch.Tensor        # (B, n_factors)   factor별 최근접 패턴(진단용, non-diff)
    disc: torch.Tensor           # 평균 dist→pat(이산성 진단; 작을수록 이산)
    info: dict                   # solver 정보
    w: torch.Tensor = None       # (B, n_factors, K) softmax 코드 = categorical posterior(변분용)


class AttractorVAE(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden: int = 256, in_channels: int = 3,
                 n_factors: int = 8, n_patterns: int = 16, beta: float = 2.0,
                 pattern_scale: float = 3.0, solver: str = "fpi",
                 solver_max_iter: int = 16, solver_tol: float = 1e-4,
                 bw_type: str = "onestep", adjoint_clip: float = 10.0,
                 context_norm: bool = False, context_scale: float = 3.0,
                 unit_patterns: bool = False, variational: bool = False):
        super().__init__()
        assert latent_dim % n_factors == 0, "latent_dim 은 n_factors 로 나눠떨어져야 함"
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.beta = beta
        self.variational = variational      # True면 w* 노출 + factor별 학습 prior(KL용)
        self.n_factors = n_factors
        self.n_patterns = n_patterns
        if variational:
            self.prior_logits = nn.Parameter(torch.zeros(n_factors, n_patterns))
        self.context_norm = context_norm    # factor별 c 를 고정 노름으로 묶어 collapse 방지
        self.context_scale = context_scale
        self.solver = solver
        self.solver_max_iter = solver_max_iter
        self.solver_tol = solver_tol
        self.bw_type = bw_type
        self.adjoint_clip = adjoint_clip

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, hidden), nn.ReLU(True),
            nn.Linear(hidden, latent_dim),
        )
        self.head = FactorizedHopfield(
            n_factors=n_factors, factor_dim=latent_dim // n_factors,
            n_patterns=n_patterns, beta=beta, learn_beta=False,
            pattern_scale=pattern_scale, unit_patterns=unit_patterns)
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

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.decoder_input(y).view(-1, 64, 4, 4))

    def _norm_context(self, c: torch.Tensor) -> torch.Tensor:
        """factor별 c 슬라이스를 고정 노름(context_scale)으로 정규화 — 크기 폭발 방지."""
        B = c.size(0)
        cf = c.view(B, self.head.n_factors, self.head.factor_dim)
        cf = cf / cf.norm(dim=-1, keepdim=True).clamp_min(1e-6) * self.context_scale
        return cf.view(B, self.latent_dim)

    def forward(self, x: torch.Tensor, beta=None) -> AttractorOutput:
        c = self.encoder(x)                                   # (B, latent_dim)
        if self.context_norm:
            c = self._norm_context(c)
        y_star, info = self.head.infer(
            c, bw_type=self.bw_type, adjoint_clip=self.adjoint_clip,
            beta=self.beta if beta is None else beta,
            solver=self.solver, max_iter=self.solver_max_iter, tol=self.solver_tol)
        recon_logits = self.decode(y_star)                    # y* 를 그대로 디코딩(미분가능)
        w = None
        if self.variational:
            b = self.beta if beta is None else beta
            w = self.head.soft_codes(y_star, c, beta=b)       # (B,F,K) 미분가능 posterior
        with torch.no_grad():
            indices = self.head.assign(y_star)                # 진단용(non-diff)
            disc = self._discreteness(y_star, indices)        # 평균 dist→pat
        return AttractorOutput(recon_logits=recon_logits, c=c, y_star=y_star,
                               indices=indices, disc=disc, info=info, w=w)

    @torch.no_grad()
    def generate(self, n: int, device, beta=None):
        """학습된 factor별 prior 에서 코드 샘플 → 패턴 디코딩 → 새 이미지 (변분 모델)."""
        assert self.variational, "generate 는 variational=True 에서만"
        pri = F.softmax(self.prior_logits, dim=-1)            # (F, K)
        idx = torch.multinomial(pri, n, replacement=True)     # (F, n)
        P = self.head._stacked_patterns()                     # (F, D, K)
        y = torch.stack([P[f, :, idx[f]].t() for f in range(self.n_factors)], dim=1)  # (n,F,D)
        return torch.sigmoid(self.decode(y.reshape(n, -1).to(device)))

    @torch.no_grad()
    def _discreteness(self, y_star: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """고정점이 최근접 저장패턴에서 떨어진 평균 거리(작을수록 이산)."""
        B = y_star.size(0)
        yf = y_star.view(B, self.head.n_factors, self.head.factor_dim)
        gathered = torch.stack([
            self.head.factors[i].eff_patterns.t()[indices[:, i]]
            for i in range(self.head.n_factors)], dim=1)
        return (yf - gathered).norm(dim=-1).mean()


def _recon_loss(out, x, recon_type):
    B = x.size(0)
    if recon_type == "gaussian":
        return F.mse_loss(torch.sigmoid(out.recon_logits), x, reduction="sum") / B
    if recon_type == "bernoulli":
        return F.binary_cross_entropy_with_logits(out.recon_logits, x, reduction="sum") / B
    raise ValueError(f"unknown recon_type: {recon_type}")


def attractor_loss(out: AttractorOutput, x: torch.Tensor, recon_type: str = "gaussian"):
    """deterministic AE: 재구성 손실만. aux = 평균 dist→pat(이산성 진단, 손실 아님)."""
    recon = _recon_loss(out, x, recon_type)
    return recon, recon, out.disc


def aqvae_loss(out: AttractorOutput, x: torch.Tensor, prior_logits: torch.Tensor,
               beta_kl: float, recon_type: str = "gaussian"):
    """Categorical AQ-VAE: recon(y* 디코딩) + β_KL·Σ_factor KL(Cat(w*) ‖ Cat(prior)).
    KL closed-form, w* 미분가능 → 샘플링 gradient 불필요. 반환 (loss, recon, kl)."""
    recon = _recon_loss(out, x, recon_type)
    w = out.w.clamp_min(1e-9)                                 # (B, F, K)
    logp = F.log_softmax(prior_logits, dim=-1).unsqueeze(0)   # (1, F, K)
    kl = (w * (w.log() - logp)).sum(-1)                       # (B, F)  per-factor KL
    kl = kl.sum(-1).mean()                                    # factor 합, 배치 평균
    return recon + beta_kl * kl, recon, kl
