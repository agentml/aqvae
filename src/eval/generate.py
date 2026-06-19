"""변분 AQ-VAE 생성 샘플 + 재구성 그리드.

실행: python -m src.eval.generate --ckpt outputs/aqvae_variational/ckpt_best.pt
산출: outputs/interpret/generated_e<ep>.png (prior 샘플), recon_e<ep>.png (입력|재구성).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..dataset.dataset_shapes3d import Shapes3D, prep_batch
from ..utils import make_grid, save_rgb_png
from .interpret import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--out", default="outputs/interpret")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    m, a, epoch = load_model(args.ckpt, args.device)
    beta = a.get("beta_end", 8.0)
    assert m.variational, "생성은 variational 모델에서만"
    print(f"[generate] epoch={epoch} beta={beta} n={args.n}")

    # (1) prior 샘플 생성
    with torch.no_grad():
        gen = m.generate(args.n, args.device, beta=beta)        # (n,3,64,64)
    grid = make_grid(gen.permute(0, 2, 3, 1).cpu().numpy(), nrow=int(args.n ** 0.5))
    gpath = os.path.join(args.out, f"generated_e{epoch}.png")
    save_rgb_png(grid, gpath)
    print(f"  생성 샘플 -> {gpath}")

    # (2) 재구성(입력 위 / 재구성 아래)
    ds = Shapes3D(a["data_dir"], uint8=True)
    x = prep_batch(next(iter(DataLoader(ds, batch_size=8, shuffle=True, num_workers=4))),
                   args.device, True)
    with torch.no_grad():
        rec = torch.sigmoid(m(x, beta=beta).recon_logits)
    imgs = torch.cat([x, rec], 0).permute(0, 2, 3, 1).cpu().numpy()
    rpath = os.path.join(args.out, f"recon_var_e{epoch}.png")
    save_rgb_png(make_grid(imgs, nrow=8), rpath)
    print(f"  재구성(위:원본/아래:재구성) -> {rpath}")

    # prior 분포 요약
    with torch.no_grad():
        pri = torch.softmax(m.prior_logits, dim=-1)             # (F,K)
        perpl = (-(pri.clamp_min(1e-9) * pri.clamp_min(1e-9).log()).sum(-1)).exp()
    print(f"  prior perplexity(factor별 유효 코드수): "
          f"{[round(x, 1) for x in perpl.tolist()]}")


if __name__ == "__main__":
    main()
