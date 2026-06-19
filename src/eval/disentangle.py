"""Disentanglement 정량 평가: MIG · DCI(MI기반 D/C) · FactorVAE score.

세 모델(VAE / VQ-VAE / Attractor)을 동일 32-dim 연속 표현으로 비교.
  - VAE: mu,  VQ-VAE: z_e,  Attractor: y*.
의존성 numpy 만. (sklearn 없어 DCI 는 MI importance 기반.)

실행: python -m src.eval.disentangle --ckpts <vae> <vqvae> <attractor> [--n 40000]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..dataset_shapes3d import Shapes3D, FACTOR_NAMES, FACTOR_SIZES
from ..model_vae import VAE
from ..model_vqvae import VQVAE
from ..model_davae import DAVAE

NF_GT = 6


# ---------- 모델 로드 / 표현 추출 ----------
def load_any(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    arch = a["arch"]
    in_ch = 3 if a["dataset"] == "shapes3d" else 1
    if arch == "vae":
        m = VAE(latent_dim=a["latent_dim"], hidden=a["hidden"], in_channels=in_ch)
    elif arch == "vqvae":
        m = VQVAE(latent_dim=a["latent_dim"], hidden=a["hidden"], in_channels=in_ch,
                  n_groups=a["n_groups"], codebook_size=a["codebook_size"],
                  commitment=a["commitment"])
    else:
        m = DAVAE(latent_dim=a["latent_dim"], hidden=a["hidden"], in_channels=in_ch,
                         n_factors=a["n_factors"], n_patterns=a["n_patterns"],
                         beta=a["beta_start"], context_norm=True,
                         context_scale=a["context_scale"], pattern_scale=a["pattern_scale"],
                         unit_patterns=True, solver=a.get("solver", "fpi"),
                         solver_max_iter=a.get("solver_max_iter", 64),
                         variational=a.get("variational", False))
    m.load_state_dict(ck["model"])
    m.to(device).eval()
    return m, a, arch, ck.get("epoch")


@torch.no_grad()
def extract(m, x, arch, beta):
    """반환 (연속표현 rep(B,32), 이산 factor-code codes(B,nf) or None)."""
    if arch == "vae":
        mu, _ = m.encode(x)
        return mu, None
    if arch == "vqvae":
        out = m(x)
        return out.z_e.reshape(x.size(0), -1), out.indices
    out = m(x, beta=beta)
    return out.y_star, out.indices


@torch.no_grad()
def collect(m, arch, ds, n, beta, device, bs=512):
    idx = np.unique(np.linspace(0, len(ds) - 1, n).astype(int))
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=bs, num_workers=6)
    reps, gts, codes = [], [], []
    for x, gt in loader:
        r, c = extract(m, x.to(device), arch, beta)
        reps.append(r.cpu().numpy()); gts.append(gt.numpy())
        if c is not None:
            codes.append(c.cpu().numpy())
    cc = np.concatenate(codes) if codes else None
    return np.concatenate(reps), np.concatenate(gts), cc


# ---------- 정보량 유틸 ----------
def discretize(rep, bins=20):
    out = np.empty_like(rep, dtype=np.int64)
    for d in range(rep.shape[1]):
        col = rep[:, d]
        edges = np.quantile(col, np.linspace(0, 1, bins + 1))
        out[:, d] = np.clip(np.digitize(col, edges[1:-1]), 0, bins - 1)
    return out


def entropy_counts(x, n):
    c = np.bincount(x, minlength=n).astype(float)
    p = c / c.sum()
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def mi(a, b, na, nb):
    c = np.zeros((na, nb))
    np.add.at(c, (a, b), 1)
    p = c / c.sum()
    pa, pb = p.sum(1), p.sum(0)
    ii, jj = np.where(c > 0)
    return float(np.sum(p[ii, jj] * np.log(p[ii, jj] / (pa[ii] * pb[jj]))))


# ---------- 지표 ----------
def mig_and_R(rep_disc, gts, bins):
    """MIG + importance 행렬 R[d,j]=MI(latent_d; factor_j)."""
    D = rep_disc.shape[1]
    R = np.zeros((D, NF_GT))
    for d in range(D):
        for j in range(NF_GT):
            R[d, j] = mi(rep_disc[:, d], gts[:, j], bins, FACTOR_SIZES[j])
    migs = []
    for j in range(NF_GT):
        Hf = entropy_counts(gts[:, j], FACTOR_SIZES[j])
        s = np.sort(R[:, j])[::-1]
        migs.append((s[0] - s[1]) / (Hf + 1e-12))
    return float(np.mean(migs)), R


def factor_mig(codes, gts, K):
    """factor-level MIG: 8개 이산 factor-code 를 latent 단위로. (4dim/factor 설계에 공정)"""
    nf = codes.shape[1]
    migs = []
    for j in range(NF_GT):
        Hf = entropy_counts(gts[:, j], FACTOR_SIZES[j])
        mis = np.array([mi(codes[:, f], gts[:, j], K, FACTOR_SIZES[j]) for f in range(nf)])
        s = np.sort(mis)[::-1]
        migs.append((s[0] - s[1]) / (Hf + 1e-12))
    return float(np.mean(migs))


def dci_from_R(R):
    """MI importance 기반 DCI 의 Disentanglement·Completeness."""
    Rp = R + 1e-12
    D, J = Rp.shape
    # Disentanglement: 각 latent 이 한 factor 에만 정보를 갖는가
    P = Rp / Rp.sum(1, keepdims=True)
    Dd = 1 + np.sum(P * np.log(P) / np.log(J), axis=1)   # 1 - H_J(P_d)
    rho = Rp.sum(1) / Rp.sum()
    disent = float(np.sum(rho * Dd))
    # Completeness: 각 factor 가 한 latent 에만 의존하는가
    Q = Rp / Rp.sum(0, keepdims=True)
    Cj = 1 + np.sum(Q * np.log(Q) / np.log(D), axis=0)   # 1 - H_D(Q_j)
    comp = float(np.mean(Cj))
    return disent, comp


def factorvae_score(rep, gts, L=100, n_train=1000, n_test=1000, seed=0):
    rng = np.random.RandomState(seed)
    gstd = rep.std(0)
    active = np.where(gstd > 0.05 * gstd.max())[0]
    if len(active) == 0:
        return 0.0
    repn = (rep / (gstd + 1e-12))[:, active]
    idxmap = {(j, v): np.where(gts[:, j] == v)[0]
              for j in range(NF_GT) for v in range(FACTOR_SIZES[j])}
    valid = {j: [v for v in range(FACTOR_SIZES[j]) if len(idxmap[(j, v)]) >= L]
             for j in range(NF_GT)}

    def gen(n):
        out = []
        for _ in range(n):
            j = rng.randint(NF_GT)
            if not valid[j]:
                continue
            v = valid[j][rng.randint(len(valid[j]))]
            sel = idxmap[(j, v)][rng.randint(0, len(idxmap[(j, v)]), L)]
            d = repn[sel].var(0).argmin()
            out.append((int(d), j))
        return out

    train, test = gen(n_train), gen(n_test)
    V = np.zeros((len(active), NF_GT))
    for d, j in train:
        V[d, j] += 1
    classifier = V.argmax(1)            # local dim -> factor
    seen = V.sum(1) > 0
    if not test:
        return 0.0
    acc = np.mean([classifier[d] == j for d, j in test if seen[d]] or [0.0])
    return float(acc)


def evaluate(ckpt, ds, n, device, bins=20):
    m, a, arch, epoch = load_any(ckpt, device)
    beta = a.get("beta_end", 8.0)
    rep, gts, codes = collect(m, arch, ds, n, beta, device)
    rep_disc = discretize(rep, bins)
    migv, R = mig_and_R(rep_disc, gts, bins)
    disent, comp = dci_from_R(R)
    fvae = factorvae_score(rep, gts)
    ndim_active = int((rep.std(0) > 0.05 * rep.std(0).max()).sum())
    fmig = float("nan")
    if codes is not None:
        K = int(codes.max()) + 1
        fmig = factor_mig(codes, gts, max(K, 2))
    return dict(arch=arch, epoch=epoch, MIG=migv, fMIG=fmig, DCI_D=disent, DCI_C=comp,
                FactorVAE=fvae, ndim=rep.shape[1], ndim_active=ndim_active)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--n", type=int, default=40000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ds = Shapes3D("data", return_factors=True)
    rows = []
    for ck in args.ckpts:
        r = evaluate(ck, ds, args.n, args.device)
        rows.append(r)
        print(f"[done] {r['arch']:9s} e{r['epoch']}  MIG={r['MIG']:.3f} fMIG={r['fMIG']:.3f} "
              f"DCI-D={r['DCI_D']:.3f} DCI-C={r['DCI_C']:.3f} FactorVAE={r['FactorVAE']:.3f} "
              f"active_dim={r['ndim_active']}/{r['ndim']}", flush=True)

    print("\n=== Disentanglement 비교 (높을수록 좋음) ===")
    print(f"  {'model':10s} {'MIG':>6} {'fMIG':>6} {'DCI-D':>6} {'DCI-C':>6} {'FactorVAE':>10} {'active':>8}")
    for r in rows:
        fm = "  n/a" if r["fMIG"] != r["fMIG"] else f"{r['fMIG']:6.3f}"
        print(f"  {r['arch']:10s} {r['MIG']:6.3f} {fm} {r['DCI_D']:6.3f} {r['DCI_C']:6.3f} "
              f"{r['FactorVAE']:10.3f} {r['ndim_active']:>5}/{r['ndim']:<2}")
    print("  MIG=per-dim(VAE유리), fMIG=factor-code 단위(4dim/factor 설계에 공정),")
    print("  DCI=MI기반. 표현: VAE=mu, VQ-VAE=z_e, Attractor=y*.")


if __name__ == "__main__":
    main()
