"""의미 분석: 각 팩터·attractor 가 무슨 GT 인자/값을 인코딩하나.

산출:
  (1) NMI 행렬 (8 모델팩터 × 6 GT팩터) — 팩터↔GT 정렬.
  (2) attractor purity 표 — 정렬된 팩터의 각 attractor 가 어느 GT 값인가.
  (3) swap traversal PNG — 팩터별로 attractor 를 갈아끼워 디코딩(각 attractor 의 모습).

실행: python -m src.eval.interpret --ckpt outputs/attractor_shapes3d_z32/ckpt_best.pt
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..dataset_shapes3d import Shapes3D, FACTOR_NAMES, FACTOR_SIZES
from ..model_attractor_vae import DAVAE
from ..utils import make_grid, save_rgb_png


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    in_ch = 3 if a["dataset"] == "shapes3d" else 1
    m = DAVAE(
        latent_dim=a["latent_dim"], hidden=a["hidden"], in_channels=in_ch,
        n_factors=a["n_factors"], n_patterns=a["n_patterns"], beta=a["beta_start"],
        context_norm=True, context_scale=a["context_scale"], pattern_scale=a["pattern_scale"],
        unit_patterns=True, solver=a.get("solver", "fpi"),
        solver_max_iter=a.get("solver_max_iter", 64),
        variational=a.get("variational", False))
    m.load_state_dict(ck["model"])
    m.to(device).eval()
    return m, a, ck.get("epoch")


def mi_nmi(a, b, na, nb):
    """이산 a,b 의 (normalized) mutual information. 반환 (NMI, MI[nats])."""
    c = np.zeros((na, nb))
    np.add.at(c, (a, b), 1)
    N = c.sum()
    p = c / N
    pa, pb = p.sum(1), p.sum(0)
    ii, jj = np.where(c > 0)
    I = float(np.sum(p[ii, jj] * np.log(p[ii, jj] / (pa[ii] * pb[jj]))))
    Ha = -float(np.sum(pa[pa > 0] * np.log(pa[pa > 0])))
    Hb = -float(np.sum(pb[pb > 0] * np.log(pb[pb > 0])))
    return I / (np.sqrt(Ha * Hb) + 1e-12), I


@torch.no_grad()
def collect(model, ds, n_samples, beta, device, bs=512):
    idx = np.unique(np.linspace(0, len(ds) - 1, n_samples).astype(int))
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=bs, num_workers=6)
    codes, gts = [], []
    for x, gt in loader:
        out = model(x.to(device), beta=beta)
        codes.append(out.indices.cpu().numpy())
        gts.append(gt.numpy())
    return np.concatenate(codes), np.concatenate(gts)   # (N,nf), (N,6)


@torch.no_grad()
def swap_traversal(model, ds, beta, device, base_idx=12345):
    """base 이미지의 y* 에서 팩터별로 attractor 패턴을 갈아끼워 디코딩."""
    nf, K = model.head.n_factors, model.head.factors[0].n_patterns
    fd = model.head.factor_dim
    x0 = ds[base_idx][0].unsqueeze(0).to(device)
    y0 = model(x0, beta=beta).y_star.clone()            # (1, nf*fd)
    rows = [torch.sigmoid(model(x0, beta=beta).recon_logits)[0]]  # 맨 앞: base 재구성
    rows += [torch.zeros_like(rows[0])] * (K - 1)       # 첫 행 패딩
    for i in range(nf):
        P = model.head.factors[i].eff_patterns          # (fd, K)
        for k in range(K):
            y = y0.clone()
            y[0, i * fd:(i + 1) * fd] = P[:, k]
            rows.append(torch.sigmoid(model.decode(y))[0])
    imgs = torch.stack(rows).permute(0, 2, 3, 1).cpu().numpy()   # ((nf+1)*K, H,W,3)
    return make_grid(imgs, nrow=K)                       # 행0=base, 행 i+1 = 팩터 i의 K attractor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-samples", type=int, default=30000)
    p.add_argument("--out", default="outputs/interpret")
    p.add_argument("--beta", type=float, default=None, help="기본: ckpt 의 beta_end")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, a, epoch = load_model(args.ckpt, args.device)
    beta = args.beta if args.beta is not None else a.get("beta_end", 8.0)
    nf, K = model.head.n_factors, model.head.factors[0].n_patterns
    print(f"[interpret] ckpt={args.ckpt} epoch={epoch} beta_eval={beta} "
          f"nf={nf} K={K} N={args.n_samples}")

    ds = Shapes3D(a["data_dir"], return_factors=True)
    codes, gts = collect(model, ds, args.n_samples, beta, args.device)
    print(f"[interpret] collected codes{codes.shape} gts{gts.shape}")

    # (1) NMI 행렬 8×6
    M = np.zeros((nf, 6))
    for i in range(nf):
        for j in range(6):
            M[i, j], _ = mi_nmi(codes[:, i], gts[:, j], K, FACTOR_SIZES[j])
    print("\n=== (1) NMI 행렬 (모델팩터 × GT팩터) ===")
    print("factor | " + " ".join(f"{n[:6]:>6}" for n in FACTOR_NAMES))
    for i in range(nf):
        star = M[i].argmax()
        cells = " ".join((f"\033[1m{M[i,j]:6.2f}\033[0m" if j == star else f"{M[i,j]:6.2f}")
                         for j in range(6))
        print(f"  f{i:<2}  | {cells}  -> {FACTOR_NAMES[star]} ({M[i,star]:.2f})")

    # (2) attractor purity: 각 팩터를 best GT 에 정렬, attractor→GT값 순도
    print("\n=== (2) attractor purity (팩터 → 정렬 GT) ===")
    for i in range(nf):
        j = int(M[i].argmax())
        cont = np.zeros((K, FACTOR_SIZES[j]))
        np.add.at(cont, (codes[:, i], gts[:, j]), 1)
        used = np.where(cont.sum(1) > 0)[0]
        dom = cont.argmax(1)
        purity = cont.max(1).sum() / cont.sum()
        ndomvals = len(np.unique(dom[used]))
        print(f"  f{i} → {FACTOR_NAMES[j]:11s}: 활성 attractor {len(used):2d}/{K}, "
              f"전체 purity {purity:.2f}, 서로 다른 dominant GT값 {ndomvals}/{FACTOR_SIZES[j]}")
        # 활성 attractor 중 일부 예시 (attractor k → dominant GT값, 그 순도)
        ex = used[:6]
        detail = "  ".join(f"a{k}→{FACTOR_NAMES[j][:4]}={dom[k]}({cont[k].max()/cont[k].sum():.2f})"
                           for k in ex)
        print(f"       예: {detail}")

    # (3) swap traversal PNG
    grid = swap_traversal(model, ds, beta, args.device)
    path = os.path.join(args.out, f"swap_traversal_e{epoch}.png")
    save_rgb_png(grid, path)
    print(f"\n=== (3) swap traversal -> {path} ===")
    print("  행0=base 재구성, 행 i+1 = 팩터 i 의 attractor 0..K-1 디코딩 (열=attractor).")


if __name__ == "__main__":
    main()
