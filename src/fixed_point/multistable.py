"""Modern Hopfield 기반 multi-stable attractor (이산 개념 생성).

Ramsauer et al. 2020. 상태 y∈R^D, 저장패턴 X∈R^{D×K}(=이산 개념).
  map:  T(y) = X · softmax(β · Xᵀ y)
  고정점 y* = T(y*) 는 (β 큰 경우) 저장패턴 중 하나로 수렴.
  β = stiffness 노브: 크면 K개 sharp attractor, 작으면 병합돼 적은 수.

factor 하나당 하나씩 둔다(예: D=4, K=16). 학습 시 patterns·beta 가 파라미터.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HopfieldAttractor(nn.Module):
    def __init__(self, dim: int, n_patterns: int, beta: float = 8.0,
                 learn_patterns: bool = True, learn_beta: bool = False,
                 pattern_scale: float = 1.0, unit_patterns: bool = False):
        super().__init__()
        self.dim = dim
        self.n_patterns = n_patterns
        self.unit_patterns = unit_patterns   # True면 패턴을 단위노름으로 고정(β가 단일 온도 노브)
        # 저장패턴 X: (D, K)
        X = torch.randn(dim, n_patterns) * (pattern_scale / math.sqrt(dim))
        self.patterns = nn.Parameter(X, requires_grad=learn_patterns)
        # β: 양수 유지 위해 log 파라미터(학습 시) 또는 buffer(고정)
        if learn_beta:
            self.log_beta = nn.Parameter(torch.tensor(float(math.log(beta))))
            self.register_buffer("_beta_const", None)
        else:
            self.register_parameter("log_beta", None)
            self.register_buffer("_beta_const", torch.tensor(float(beta)))

    @property
    def beta(self) -> torch.Tensor:
        return self._beta_const if self.log_beta is None else self.log_beta.exp()

    @property
    def eff_patterns(self) -> torch.Tensor:
        """실제 사용 패턴. unit_patterns 면 열(패턴)별 단위노름으로 정규화."""
        if self.unit_patterns:
            return self.patterns / self.patterns.norm(dim=0, keepdim=True).clamp_min(1e-6)
        return self.patterns

    def map(self, y: torch.Tensor, context=None, beta=None) -> torch.Tensor:
        """T(y; c) = X · softmax(β Xᵀ (y + c)). y:(B,D) -> (B,D).

        context=None  -> 순수 retrieval(query-as-init): y* 가 c 에 piecewise-constant
                         => ∂y*/∂c ≈ 0 (STE 문제와 동일, 인코더 gradient 죽음).
        context=c     -> persistent injection: 고정점이 c 에 매끄럽게 의존
                         => IFT 가 인코더로 정확한 nonzero gradient 전달.
        """
        b = self.beta if beta is None else beta
        P = self.eff_patterns
        q = y if context is None else y + context
        sim = q @ P                              # (B, K) = Xᵀ q per-sample
        w = F.softmax(b * sim, dim=-1)           # (B, K)
        return w @ P.t()                         # (B, D)

    def forward(self, y: torch.Tensor, context=None, beta=None) -> torch.Tensor:
        return self.map(y, context=context, beta=beta)

    @torch.no_grad()
    def set_patterns(self, X: torch.Tensor) -> None:
        """C1 검증용: 저장패턴을 고정값으로 주입."""
        assert X.shape == self.patterns.shape, (X.shape, self.patterns.shape)
        self.patterns.copy_(X.to(self.patterns))

    def assign(self, y_star: torch.Tensor) -> torch.Tensor:
        """고정점을 최근접 저장패턴 인덱스로 매핑. y_star:(B,D) -> (B,) long."""
        d = torch.cdist(y_star, self.eff_patterns.t())   # (B, K)
        return d.argmin(dim=-1)


class FactorizedHopfield(nn.Module):
    """factor 별 독립 Hopfield attractor ×n_factors.

    상태/컨텍스트는 (B, n_factors*factor_dim) 평면 벡터. factor i 의 동역학은
    자기 4차원 슬라이스에만 작용(block-diagonal) → 고정점/IFT 를 전체 벡터에서
    한 번에 처리하되 factor 간 독립.
    """

    def __init__(self, n_factors: int = 8, factor_dim: int = 4, n_patterns: int = 16,
                 beta: float = 2.0, learn_beta: bool = True, pattern_scale: float = 3.0,
                 unit_patterns: bool = False):
        super().__init__()
        self.n_factors = n_factors
        self.factor_dim = factor_dim
        self.dim = n_factors * factor_dim
        self.factors = nn.ModuleList([
            HopfieldAttractor(factor_dim, n_patterns, beta=beta, learn_beta=learn_beta,
                              pattern_scale=pattern_scale, unit_patterns=unit_patterns)
            for _ in range(n_factors)
        ])

    def _stacked_patterns(self) -> torch.Tensor:
        """모든 factor 의 eff_patterns 를 (n_factors, D, K) 로 적층 (벡터화용)."""
        return torch.stack([f.eff_patterns for f in self.factors], dim=0)

    def _beta(self, beta):
        if beta is not None:
            return beta                                   # 스칼라(브로드캐스트)
        return torch.stack([f.beta for f in self.factors]).view(1, self.n_factors, 1)

    def map(self, Y: torch.Tensor, context=None, beta=None) -> torch.Tensor:
        """T(Y;C) factor별 Hopfield 를 batched einsum 으로 일괄 계산 (Python 루프 제거)."""
        B = Y.size(0)
        Yf = Y.view(B, self.n_factors, self.factor_dim)
        q = Yf if context is None else Yf + context.view(B, self.n_factors, self.factor_dim)
        P = self._stacked_patterns()                      # (F, D, K)
        sim = torch.einsum("bfd,fdk->bfk", q, P)          # (B, F, K) = Xᵀq per-factor
        w = F.softmax(self._beta(beta) * sim, dim=-1)     # (B, F, K)
        out = torch.einsum("bfk,fdk->bfd", w, P)          # (B, F, D)
        return out.reshape(B, self.dim)

    def assign(self, Y: torch.Tensor) -> torch.Tensor:
        """factor 별 이산 인덱스. (B, dim) -> (B, n_factors) long."""
        B = Y.size(0)
        Yf = Y.view(B, self.n_factors, self.factor_dim)
        P = self._stacked_patterns()                      # (F, D, K)
        d = torch.cdist(Yf.transpose(0, 1), P.transpose(1, 2))  # (F, B, K)
        return d.argmin(-1).transpose(0, 1)               # (B, F)

    def grad_params(self, context):
        """IFT 라우팅 대상: context + 각 factor 의 patterns·log_beta."""
        ps = [context]
        for hop in self.factors:
            ps.append(hop.patterns)
            if hop.log_beta is not None:
                ps.append(hop.log_beta)
        return ps

    def infer(self, context: torch.Tensor, bw_type: str = "onestep", adjoint_clip=None,
              beta=None, solver: str = "anderson", max_iter: int = 50, tol: float = 1e-4):
        """context(B,dim) -> 고정점 y*(B,dim), IFT grad 부착. Phase D 에서 호출."""
        from .implicit import attach_ift_grad, solve_fixed_point
        cd = context.detach()
        y_ng, info = solve_fixed_point(lambda Y: self.map(Y, context=cd, beta=beta),
                                       cd.clone(), solver=solver, max_iter=max_iter, tol=tol)
        y_star = attach_ift_grad(lambda Y: self.map(Y, context=context, beta=beta),
                                 y_ng, self.grad_params(context),
                                 bw_type=bw_type, adjoint_clip=adjoint_clip)
        return y_star, info
