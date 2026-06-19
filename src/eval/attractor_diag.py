"""Attractor 고유 진단: (1) perturbation-복귀(basin), (2) internalization.

실행: python -m src.eval.attractor_diag --ckpt <attractor ckpt>
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..dataset.dataset_shapes3d import Shapes3D
from .interpret import load_model


@torch.no_grad()
def perturbation_return(m, c, beta, nus=(0.1, 0.25, 0.5, 1.0, 2.0)):
    """수렴한 y* 에 크기 ν 의 무작위 교란 후 같은 context 로 재반복 → 복귀 거리.

    basin 안이면 작은 ν 에서 y* 로 복귀(복귀거리 ≈ 0)."""
    head = m.head
    # 정상 고정점
    y_star, _ = head.infer(c, bw_type="onestep", beta=beta,
                           solver="fpi", max_iter=200, tol=1e-7)
    rows = []
    for nu in nus:
        y = y_star + nu * torch.randn_like(y_star)
        for _ in range(200):                       # 같은 context 로 재반복
            y = head.map(y, context=c, beta=beta)
        back = (y - y_star).norm(dim=-1).mean().item()      # 복귀 후 y* 와의 거리
        moved = (nu * torch.randn_like(y_star)).norm(dim=-1).mean().item()  # 교란 크기 참고
        rows.append((nu, back))
    return y_star, rows


@torch.no_grad()
def internalization(m, x, beta):
    """y*-c 이동량, solver step 수에 따른 recon 변화(추론 compute-정확도)."""
    c = m.encoder(x)
    if m.context_norm:
        c = m._norm_context(c)
    y_star, _ = m.head.infer(c, bw_type="onestep", beta=beta,
                             solver="fpi", max_iter=200, tol=1e-7)
    head_contrib = ((y_star - c).norm(dim=-1) / c.norm(dim=-1).clamp_min(1e-9)).mean().item()

    def recon_at(maxit):
        if maxit == 0:
            y = c                                   # solver 0회 = context 그대로 디코딩
        else:
            y, _ = m.head.infer(c, bw_type="onestep", beta=beta,
                                solver="fpi", max_iter=maxit, tol=1e-9)
        rl = m.decode(y)
        return torch.nn.functional.mse_loss(torch.sigmoid(rl), x, reduction="sum").item() / x.size(0)

    steps = [0, 1, 2, 4, 8, 16, 64]
    recons = [(s, recon_at(s)) for s in steps]
    return head_contrib, recons


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    m, a, epoch = load_model(args.ckpt, args.device)
    beta = a.get("beta_end", 8.0)
    ds = Shapes3D(a["data_dir"])
    x = next(iter(DataLoader(ds, batch_size=256, shuffle=True, num_workers=4))).to(args.device)
    print(f"[diag] epoch={epoch} beta={beta}")

    # (1) perturbation-복귀
    c = m.encoder(x)
    if m.context_norm:
        c = m._norm_context(c)
    _, rows = perturbation_return(m, c, beta)
    print("\n=== (1) perturbation-복귀 (basin 검증) ===")
    print("  교란크기 ν → 재반복 후 y* 와의 거리 (작으면 basin 복귀)")
    for nu, back in rows:
        print(f"   ν={nu:4.2f}  복귀거리={back:.3f}  {'복귀' if back < 0.3 else '이탈(다른 basin)'}")

    # (2) internalization
    hc, recons = internalization(m, x, beta)
    print("\n=== (2) internalization ===")
    print(f"  head contribution ||y*-c||/||c|| = {hc:.3f}  (작을수록 backbone 이 내재화)")
    print("  solver step 수 → recon (적은 step 에도 recon 유지되면 내재화)")
    base = recons[-1][1]
    for s, r in recons:
        print(f"   steps={s:2d}  recon={r:7.1f}  (Δ vs full={r-base:+.1f})")


if __name__ == "__main__":
    main()
