"""β 어닐링 가설 빠른 검증 (일회용).

가설: 고정 고β = collapse(소수 attractor), 저β/어닐링 = 인코더가 잠재 조직화 →
recon 개선 + 활성 attractor 다양성 유지.

실행: python -m src._smoke_anneal
"""
import torch
from torch.utils.data import DataLoader

from ..dataset_shapes3d import Shapes3D
from ..model_attractor_vae import DAVAE, attractor_loss

torch.manual_seed(0)
STEPS = 400
NP = 16   # n_patterns


def stream(loader):
    while True:
        for x in loader:
            yield x


@torch.no_grad()
def diversity(m, xev, beta):
    out = m(xev, beta=beta)
    idx = out.indices
    active = [int(idx[:, j].unique().numel()) for j in range(m.head.n_factors)]
    return sum(active), active, out.c.std().item(), out.disc.item()


def run(tag, sched):
    ds = Shapes3D("data")
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=6, drop_last=True)
    it = stream(loader)
    xev = next(it).cuda()[:2000] if False else None
    xev = next(it).cuda()  # 256 고정 eval 배치
    m = DAVAE(latent_dim=32, in_channels=3, n_patterns=NP, beta=2.0).cuda()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    print(f"\n[{tag}]")
    for i in range(STEPS):
        beta = sched(i)
        x = next(it).cuda()
        out = m(x, beta=beta)
        loss, recon, disc = attractor_loss(out, x)
        opt.zero_grad(); loss.backward(); opt.step()
        if (i + 1) % 100 == 0:
            tot, act, cstd, d = diversity(m, xev, beta)
            print(f"  step{i+1:4d} beta={beta:4.2f} recon={recon.item():7.1f} "
                  f"active_total={tot:3d}/{NP*8} c_std={cstd:.3f} dist→pat={d:.3f}")
    tot, act, cstd, d = diversity(m, xev, sched(STEPS - 1))
    print(f"  FINAL active per-factor={act} total={tot}/{NP*8} recon~ c_std={cstd:.3f}")
    return tot


def main():
    res = {}
    res["fixed_b2.0"] = run("fixed β=2.0", lambda i: 2.0)
    res["fixed_b0.5"] = run("fixed β=0.5", lambda i: 0.5)
    res["anneal_0.3->4"] = run("anneal β 0.3→4.0",
                               lambda i: 0.3 + (4.0 - 0.3) * i / (STEPS - 1))
    print("\n=== 요약 (최종 활성 attractor 총합, /128) ===")
    for k, v in res.items():
        print(f"  {k:16s}: {v}")
    print("가설 지지 = 어닐링이 고정β=2 보다 활성 수 많고 collapse 적으면.")


if __name__ == "__main__":
    main()
