"""C1 검증 (일회용): Hopfield 단독 multi-stability + β 노브 + 수축성.

실행: python -m src.fixed_point._verify_c1
산출: outputs/c1_basins/beta_*.png (2D basin 맵) + 콘솔 수치표.

Modern Hopfield 깨끗한 retrieval 조건: β·‖x‖²·(1-cos간격) ≫ 1.
=> 저장패턴 노름을 키우면(여기선 R=3) 적당한 β 에서 K개 attractor 가 또렷해진다.
"""
import math
import os

import torch

from ..attractor import HopfieldAttractor
from ..fixed_point.solvers import anderson_solve, fpi_solve

DT = torch.float64
torch.manual_seed(0)
OUT = "outputs/c1_basins"
R = 3.0  # 저장패턴 노름


def circle_patterns(K: int, r: float = R) -> torch.Tensor:
    ang = torch.arange(K, dtype=DT) * (2 * math.pi / K)
    return torch.stack([r * torch.cos(ang), r * torch.sin(ang)], dim=0)  # (2,K)


def hsv_palette(K: int) -> torch.Tensor:
    cols = []
    for k in range(K):
        h = k / K * 6.0
        c, x = 1.0, 1.0 * (1 - abs(h % 2 - 1))
        rgb = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)][int(h) % 6]
        cols.append(rgb)
    return torch.tensor(cols, dtype=torch.float32)


def count_distinct(pts: torch.Tensor, tol: float) -> int:
    reps = []
    for p in pts:
        if all((p - r).norm() > tol for r in reps):
            reps.append(p)
    return len(reps)


def spectral_radius(hop, y, beta) -> float:
    f = lambda v: hop.map(v.unsqueeze(0), beta=beta).squeeze(0)
    J = torch.autograd.functional.jacobian(f, y)
    return torch.linalg.eigvals(J).abs().max().item()


def eval_beta(hop, beta, D, n_init=400, span=None):
    """무작위 초기값 -> 고정점. 수렴한 것만으로 dist/distinct/specRad 계산."""
    if span is None:
        y0 = torch.randn(n_init, D, dtype=DT) * R
    else:
        y0 = (torch.rand(n_init, D, dtype=DT) * 2 - 1) * span
    ys, info = anderson_solve(lambda y: hop.map(y, beta=beta), y0, max_iter=500, tol=1e-9)
    resid = (hop.map(ys, beta=beta) - ys).norm(dim=-1)        # per-sample 잔차
    conv = resid < 1e-4
    frac = conv.float().mean().item()
    ysc = ys[conv]
    if len(ysc) == 0:
        return frac, float("nan"), 0, info["iters"], float("nan")
    dist = torch.cdist(ysc, hop.patterns.t()).min(dim=-1).values.max().item()
    nn_gap = 2 * R * math.sin(math.pi / hop.n_patterns)       # 인접 패턴 간격
    distinct = count_distinct(ysc, tol=0.25 * nn_gap)
    idx = hop.assign(ysc[:1])
    sr = spectral_radius(hop, hop.patterns.t()[idx[0]].clone(), beta)
    return frac, dist, distinct, info["iters"], sr


def basin_png(hop, beta, path, G=256, span=4.5):
    lin = torch.linspace(-span, span, G, dtype=DT)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    ys, _ = fpi_solve(lambda y: hop.map(y, beta=beta), grid, max_iter=200, tol=1e-8)
    assign = hop.assign(ys)
    resid = (hop.map(ys, beta=beta) - ys).norm(dim=-1)
    dnn = torch.cdist(ys, hop.patterns.t()).min(dim=-1).values  # 고정점→최근접패턴 거리
    nn_gap = 2 * R * math.sin(math.pi / hop.n_patterns)
    pal = hsv_palette(hop.n_patterns)
    img = pal[assign].reshape(G, G, 3).clone()
    # 패턴 근처로 수렴한 점만 색칠. 병합/중간점(centroid 등)은 회색.
    img[(dnn > 0.4 * nn_gap).reshape(G, G)] = 0.5
    img[(resid > 1e-2).reshape(G, G)] = 0.0                   # 미수렴 = 검정
    for p in hop.patterns.t():                                # 패턴 위치 = 흰 점
        ix = int((p[0].item() + span) / (2 * span) * (G - 1))
        iy = int((p[1].item() + span) / (2 * span) * (G - 1))
        img[max(iy-2, 0):iy+3, max(ix-2, 0):ix+3] = 1.0
    from ..utils import save_rgb_png
    save_rgb_png(img.numpy(), path)


def main():
    betas = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    os.makedirs(OUT, exist_ok=True)

    print(f"=== C1 2D (K=16, 원주 패턴 R={R}) ===")
    hop2 = HopfieldAttractor(2, 16, learn_patterns=False).double()
    hop2.set_patterns(circle_patterns(16))
    print(f"  {'beta':>6} {'conv%':>6} {'dist→pat':>9} {'distinct':>9} {'iters':>6} {'specRad':>8}")
    for b in betas:
        frac, dist, distinct, it, sr = eval_beta(hop2, b, 2, span=4.5)
        print(f"  {b:6.1f} {frac*100:5.0f}% {dist:9.2e} {distinct:9d} {it:6d} {sr:8.3f}")
        basin_png(hop2, b, os.path.join(OUT, f"beta_{b:05.1f}.png"))
    print(f"  basin 맵 -> {OUT}/beta_*.png")

    print(f"=== C1 4D (K=16, 단위패턴×{R}) — 실제 차원 ===")
    hop4 = HopfieldAttractor(4, 16, learn_patterns=False).double()
    X = torch.randn(4, 16, dtype=DT); X = X / X.norm(dim=0, keepdim=True) * R
    hop4.set_patterns(X)
    print(f"  {'beta':>6} {'conv%':>6} {'dist→pat':>9} {'distinct':>9} {'iters':>6} {'specRad':>8}")
    for b in betas:
        frac, dist, distinct, it, sr = eval_beta(hop4, b, 4)
        print(f"  {b:6.1f} {frac*100:5.0f}% {dist:9.2e} {distinct:9d} {it:6d} {sr:8.3f}")

    print("\n기대: β↑ 에 따라 distinct 1→K, conv% 높게 유지, 고β서 dist→pat 작아지고 specRad<1.")


if __name__ == "__main__":
    main()
