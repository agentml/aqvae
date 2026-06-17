"""(β-)VAE / VQ-VAE 학습 스크립트 — dSprites · 3D Shapes.

수렴(early stopping)까지 학습한다. epoch 평균 loss가 patience 에폭 동안
min-delta 이상 개선되지 않으면 종료. max-epochs 는 안전 상한.

사용 예:
    python -m src.train --arch vae   --dataset shapes3d --latent-dim 32   # 표준 VAE
    python -m src.train --arch vqvae --dataset shapes3d --latent-dim 32   # VQ-VAE
단일 GPU. 결과는 outputs/<run-name>/ 에 체크포인트 + 재구성 PNG 저장.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from .dataset import DSprites
from .dataset_shapes3d import Shapes3D
from .model import VAE, vae_loss
from .model_vqvae import VQVAE, vqvae_loss
from .model_attractor_vae import AttractorVAE, attractor_loss
from .utils import make_grid, save_gray_png, save_rgb_png


def parse_args():
    p = argparse.ArgumentParser(description="(beta-)VAE / VQ-VAE 학습 (수렴까지)")
    p.add_argument("--arch", choices=["vae", "vqvae", "attractor"], default="vae")
    p.add_argument("--dataset", choices=["dsprites", "shapes3d"], default="dsprites")
    p.add_argument("--recon", choices=["auto", "bernoulli", "gaussian"], default="auto",
                   help="auto: dsprites->bernoulli, shapes3d->gaussian")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--max-epochs", type=int, default=200, help="안전 상한")
    p.add_argument("--patience", type=int, default=10,
                   help="개선 없이 견디는 에폭 수 (early stopping)")
    p.add_argument("--min-delta", type=float, default=0.1,
                   help="개선으로 인정하는 epoch loss 최소 감소량")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=10)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--beta", type=float, default=1.0, help="VAE: 1.0=vanilla, >1=beta-VAE")
    # vqvae
    p.add_argument("--n-groups", type=int, default=8, help="VQ product 그룹 수")
    p.add_argument("--codebook-size", type=int, default=16, help="VQ 그룹별 코드 수 K")
    p.add_argument("--commitment", type=float, default=0.25)
    # attractor
    p.add_argument("--n-factors", type=int, default=8)
    p.add_argument("--n-patterns", type=int, default=16, help="factor별 attractor 수 K")
    p.add_argument("--context-scale", type=float, default=1.0)
    p.add_argument("--pattern-scale", type=float, default=1.0)
    p.add_argument("--beta-start", type=float, default=1.0, help="어닐링 시작 β(soft)")
    p.add_argument("--beta-end", type=float, default=8.0, help="어닐링 끝 β(discrete)")
    p.add_argument("--anneal-epochs", type=int, default=40, help="β를 끝까지 올리는 에폭")
    p.add_argument("--anneal-warmup-frac", type=float, default=0.3,
                   help="anneal-epochs 중 soft 유지 비율")
    p.add_argument("--solver", choices=["fpi", "anderson"], default="fpi")
    p.add_argument("--solver-max-iter", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def beta_at_epoch(args, epoch):
    """attractor β 어닐링: warmup 동안 beta_start, 이후 beta_end 까지 선형 상승."""
    if args.arch != "attractor":
        return None
    warm = int(args.anneal_warmup_frac * args.anneal_epochs)
    if epoch < warm:
        return args.beta_start
    if epoch >= args.anneal_epochs:
        return args.beta_end
    frac = (epoch - warm) / max(args.anneal_epochs - warm, 1)
    return args.beta_start + (args.beta_end - args.beta_start) * frac


@torch.no_grad()
def save_recon_grid(model, x, path, in_channels: int, n: int = 8, beta=None):
    model.eval()
    out = model(x[:n]) if beta is None else model(x[:n], beta=beta)
    recon = torch.sigmoid(out.recon_logits)
    imgs = torch.cat([x[:n], recon], dim=0)  # (2n, C, 64, 64), 위:원본 아래:재구성
    if in_channels == 3:
        grid = make_grid(imgs.permute(0, 2, 3, 1).cpu().numpy(), nrow=n)  # (N,H,W,3)
        save_rgb_png(grid, path)
    else:
        grid = make_grid(imgs[:, 0].cpu().numpy(), nrow=n)
        save_gray_png(grid, path)
    model.train()


def build_dataset(args):
    if args.dataset == "dsprites":
        return DSprites(args.data_dir), 1
    return Shapes3D(args.data_dir), 3


def build_model(args, in_channels):
    if args.arch == "vae":
        return VAE(latent_dim=args.latent_dim, hidden=args.hidden, in_channels=in_channels)
    if args.arch == "vqvae":
        return VQVAE(latent_dim=args.latent_dim, hidden=args.hidden, in_channels=in_channels,
                     n_groups=args.n_groups, codebook_size=args.codebook_size,
                     commitment=args.commitment)
    return AttractorVAE(latent_dim=args.latent_dim, hidden=args.hidden, in_channels=in_channels,
                        n_factors=args.n_factors, n_patterns=args.n_patterns,
                        beta=args.beta_start, context_norm=True,
                        context_scale=args.context_scale, pattern_scale=args.pattern_scale,
                        unit_patterns=True, solver=args.solver,
                        solver_max_iter=args.solver_max_iter)


def compute_loss(args, model, out, x, recon_type):
    """반환: (loss, recon, aux, aux_name)."""
    if args.arch == "vae":
        loss, recon, kl = vae_loss(out, x, beta=args.beta, recon_type=recon_type)
        return loss, recon, kl, "kl"
    if args.arch == "vqvae":
        loss, recon, vq = vqvae_loss(out, x, commitment=args.commitment, recon_type=recon_type)
        return loss, recon, vq, "vq"
    loss, recon, disc = attractor_loss(out, x, recon_type=recon_type)
    return loss, recon, disc, "dist"


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    recon_type = args.recon
    if recon_type == "auto":
        recon_type = "bernoulli" if args.dataset == "dsprites" else "gaussian"

    run_name = args.run_name or f"{args.arch}_{args.dataset}_z{args.latent_dim}"
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[train] run={run_name} arch={args.arch} dataset={args.dataset} "
          f"recon={recon_type} device={args.device} z={args.latent_dim} "
          f"patience={args.patience} min_delta={args.min_delta}")

    dataset, in_channels = build_dataset(args)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    print(f"[train] dataset size={len(dataset)} batches/epoch={len(loader)} "
          f"in_channels={in_channels}")

    model = build_model(args, in_channels).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"[train] params(M)={sum(p.numel() for p in model.parameters())/1e6:.3f}")

    fixed_x = next(iter(loader)).to(args.device)  # 재구성 시각화용 고정 배치

    best_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    history = []

    for epoch in range(args.max_epochs):
        model.train()
        t0 = time.time()
        beta = beta_at_epoch(args, epoch)                 # attractor 면 현재 β, 아니면 None
        fwd = {} if beta is None else {"beta": beta}
        run_loss = run_recon = run_aux = 0.0
        aux_name = "aux"
        for step, x in enumerate(loader):
            x = x.to(args.device, non_blocking=True)
            out = model(x, **fwd)
            loss, recon, aux, aux_name = compute_loss(args, model, out, x, recon_type)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            run_loss += loss.item(); run_recon += recon.item(); run_aux += aux.item()
            if (step + 1) % args.log_every == 0:
                print(f"  e{epoch} {step+1}/{len(loader)} loss={loss.item():.2f} "
                      f"recon={recon.item():.2f} {aux_name}={aux.item():.2f}", flush=True)

        nb = len(loader)
        ep_loss, ep_recon, ep_aux = run_loss / nb, run_recon / nb, run_aux / nb
        rec = {"epoch": epoch, "loss": ep_loss, "recon": ep_recon, aux_name: ep_aux}
        extra = ""
        if args.arch == "attractor":
            with torch.no_grad():
                oe = model(fixed_x, beta=beta)
                active = sum(int(oe.indices[:, j].unique().numel())
                             for j in range(model.head.n_factors))
            rec["beta"] = beta; rec["active"] = active
            extra = f" beta={beta:.2f} active={active}/{args.n_factors*args.n_patterns}"
        history.append(rec)
        print(f"[epoch {epoch}] loss={ep_loss:.3f} recon={ep_recon:.3f} "
              f"{aux_name}={ep_aux:.3f}{extra} ({time.time()-t0:.1f}s)", flush=True)

        save_recon_grid(model, fixed_x, os.path.join(out_dir, f"recon_e{epoch:03d}.png"),
                        in_channels, beta=beta)
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        # ---- 수렴 판정 (early stopping). 어닐링 중에는 정지 카운트 안 함 ----
        annealing = args.arch == "attractor" and epoch < args.anneal_epochs
        if ep_loss < best_loss - args.min_delta:
            best_loss = ep_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {"model": model.state_dict(), "args": vars(args),
                 "epoch": epoch, "loss": ep_loss},
                os.path.join(out_dir, "ckpt_best.pt"),
            )
        elif not annealing:
            epochs_no_improve += 1
            print(f"  no improve {epochs_no_improve}/{args.patience} "
                  f"(best={best_loss:.3f} @e{best_epoch})", flush=True)
            if epochs_no_improve >= args.patience:
                print(f"[train] converged — early stop at epoch {epoch} "
                      f"(best loss {best_loss:.3f} @ epoch {best_epoch})", flush=True)
                break

        torch.save(
            {"model": model.state_dict(), "args": vars(args), "epoch": epoch},
            os.path.join(out_dir, "ckpt_last.pt"),
        )

    print(f"[train] done -> {out_dir} (best loss {best_loss:.3f} @ epoch {best_epoch})")


if __name__ == "__main__":
    main()
