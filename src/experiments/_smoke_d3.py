"""D3: soft→discrete β 어닐링 검증 (일회용).

unit_patterns + context_norm 으로 effective temp = β·s 단일화. β 를 soft(1)에서
고온(β_end)으로 어닐링(앞 30% warmup) → 이산화(dist→pat↓)하며 active·recon 유지되는지.

실행: python -m src._smoke_d3
"""
import torch
from torch.utils.data import DataLoader

from ..dataset_shapes3d import Shapes3D
from ..model_attractor_vae import AttractorVAE, attractor_loss

torch.manual_seed(0)
STEPS = 1500
NP = 16


def stream(loader):
    while True:
        for x in loader:
            yield x


def sched(i, b0, b1, warmup=0.3):
    w = int(STEPS * warmup)
    if i < w:
        return b0
    return b0 + (b1 - b0) * (i - w) / max(STEPS - w - 1, 1)


@torch.no_grad()
def diag(m, xev, beta):
    out = m(xev, beta=beta)
    act = [int(out.indices[:, j].unique().numel()) for j in range(m.head.n_factors)]
    return sum(act), out.disc.item()


def run(tag, b0, b1):
    ds = Shapes3D("data")
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=6, drop_last=True)
    it = stream(loader)
    xev = next(it).cuda()
    m = AttractorVAE(latent_dim=32, in_channels=3, n_patterns=NP, beta=b0,
                     context_norm=True, context_scale=1.0, pattern_scale=1.0,
                     unit_patterns=True).cuda()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    print(f"\n[{tag}] β {b0}→{b1} (warmup 30%)")
    for i in range(STEPS):
        beta = sched(i, b0, b1)
        x = next(it).cuda()
        out = m(x, beta=beta)
        loss, recon, _ = attractor_loss(out, x)
        opt.zero_grad(); loss.backward(); opt.step()
        if (i + 1) % 250 == 0:
            tot, d = diag(m, xev, beta)
            print(f"  step{i+1:4d} β={beta:5.2f} recon={recon.item():7.1f} "
                  f"active={tot:3d}/128 dist→pat={d:.3f}")
    tot, d = diag(m, xev, b1)
    print(f"  FINAL active={tot}/128 dist→pat={d:.3f} recon={recon.item():.1f}")
    return tot, d, recon.item()


def main():
    res = {}
    for tag, b1 in [("anneal 1→4", 4.0), ("anneal 1→8", 8.0), ("anneal 1→16", 16.0)]:
        res[tag] = run(tag, 1.0, b1)
    print("\n=== 요약 ===")
    for k, (t, d, r) in res.items():
        print(f"  {k:14s}: active={t:3d}/128  dist→pat={d:.3f}  recon={r:.1f}")
    print("이상적: dist→pat↓(이산화) + active 높게 유지 + recon 낮게.")


if __name__ == "__main__":
    main()
