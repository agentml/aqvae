"""factor(4D)별 PCA 2D 투영: K개 attractor 존재 + step별 수렴 궤적 (matplotlib).

각 factor 패널:
  - 큰 마커(★) = K개 저장 attractor. 도착지는 진하게/테두리.
  - 선 = 시작점(encoder context)에서 fpi 반복 수렴 궤적. step 진행에 따라 색 진해짐.
  - 점 = 시작점, 화살표 = 마지막 step 방향.

실행: python -m src.eval.attractor_viz --ckpt <attractor ckpt> [--n-traj 60 --steps 25]
산출: outputs/interpret/attractor_traj_e<ep>.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..dataset.dataset_shapes3d import Shapes3D
from .interpret import load_model


def pca2(X):
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:2].T


@torch.no_grad()
def factor_data(factor, c_i, beta, steps):
    """반환: P2(K,2) attractor, T2(steps+1,B,2) 궤적, dest(B,), var(2,) PCA 설명분산비."""
    P = factor.eff_patterns.t().cpu().numpy()            # (K,4)
    y = c_i.clone()
    traj = [y.cpu().numpy()]
    for _ in range(steps):
        y = factor.map(y, context=c_i, beta=beta)
        traj.append(y.cpu().numpy())
    traj = np.stack(traj, 0)                             # (steps+1, B, 4)
    dest = factor.assign(y).cpu().numpy()

    # PCA 를 "도착지(used) attractor" 들에만 적합 → 2D 에서 도착 별들이 최대 분리.
    used = sorted(set(int(d) for d in dest))
    fitpts = P[used] if len(used) >= 2 else P
    mu, W = pca2(fitpts)
    sv = np.linalg.svd(fitpts - mu, compute_uv=False)
    var = (sv[:2] ** 2 / (sv ** 2).sum())
    return (P - mu) @ W, (traj - mu) @ W, dest, var


def panel(ax, P2, T2, dest, var, K, cmap, title):
    steps = T2.shape[0] - 1
    # 궤적: step 진행에 따라 alpha 증가, 도착 attractor 색
    for b in range(T2.shape[1]):
        col = cmap(dest[b] % K / max(K - 1, 1))
        seg = np.stack([T2[:-1, b], T2[1:, b]], axis=1)      # (steps,2,2)
        lc = LineCollection(seg, colors=[col], linewidths=0.9,
                            alpha=0.55, zorder=1)
        ax.add_collection(lc)
        ax.plot(T2[0, b, 0], T2[0, b, 1], '.', color='white', ms=2.5, zorder=2)  # 시작
        # 마지막 step 방향 화살표
        d = T2[-1, b] - T2[-2, b]
        if np.linalg.norm(d) > 1e-3:
            ax.annotate("", xy=T2[-1, b], xytext=T2[-2, b], zorder=2,
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.8, alpha=0.7))
    # attractor 마커: 도착지(used)는 크게/테두리, 아니면 작고 흐리게
    used = set(int(d) for d in dest)
    for k in range(K):
        col = cmap(k / max(K - 1, 1))
        if k in used:
            ax.plot(P2[k, 0], P2[k, 1], '*', color=col, ms=15,
                    markeredgecolor='k', markeredgewidth=0.8, zorder=3)
        else:
            ax.plot(P2[k, 0], P2[k, 1], '*', color=col, ms=8, alpha=0.35, zorder=3)
    ax.set_title(f"{title}  (active {len(used)}/{K}, PCA {var.sum()*100:.0f}%)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#0a0a0a")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-traj", type=int, default=60)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--beta", type=float, default=None, help="추론 β override (기본 ckpt beta_end)")
    p.add_argument("--out", default="outputs/interpret")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    m, a, epoch = load_model(args.ckpt, args.device)
    beta = args.beta if args.beta is not None else a.get("beta_end", 8.0)
    nf, fd = m.head.n_factors, m.head.factor_dim
    K = m.head.factors[0].n_patterns
    cmap = plt.get_cmap("hsv")
    print(f"[viz] epoch={epoch} beta={beta} nf={nf} fd={fd} K={K} n_traj={args.n_traj}")

    ds = Shapes3D(a["data_dir"])
    x = next(iter(DataLoader(ds, batch_size=args.n_traj, shuffle=True, num_workers=4))).to(args.device)
    c = m.encoder(x)
    if m.context_norm:
        c = m._norm_context(c)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for i in range(nf):
        c_i = c[:, i * fd:(i + 1) * fd]
        P2, T2, dest, var = factor_data(m.head.factors[i], c_i, beta, args.steps)
        panel(axes.flat[i], P2, T2, dest, var, K, cmap, f"factor {i}")
        print(f"  factor {i}: 활성 {len(set(int(d) for d in dest))}/{K}, PCA설명 {var.sum()*100:.0f}%")
    fig.suptitle(f"Per-factor attractor convergence  (e{epoch}, beta={beta})   "
                 f"star=attractor (outlined=reached), lines=trajectories, white dot=start(context)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(args.out, f"attractor_traj_e{epoch}_b{beta:g}.png")
    fig.savefig(path, dpi=110, facecolor="white")
    print(f"\n저장 -> {path}")


if __name__ == "__main__":
    main()
