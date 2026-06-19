"""context 정규화가 collapse 를 깨는지 검증 (일회용).

baseline(정규화 X)은 8/128 collapse. 정규화 on 에서 활성 attractor↑·recon↓ 이면 가설 지지.
실행: python -m src._smoke_ctxnorm
"""
import torch
from torch.utils.data import DataLoader

from ..dataset_shapes3d import Shapes3D
from ..model_davae import DAVAE, attractor_loss

torch.manual_seed(0)
STEPS = 300
NP = 16


def stream(loader):
    while True:
        for x in loader:
            yield x


@torch.no_grad()
def diversity(m, xev, beta):
    out = m(xev, beta=beta)
    act = [int(out.indices[:, j].unique().numel()) for j in range(m.head.n_factors)]
    # c 의 배치 다양성(인코더가 입력마다 다른 c 를 내는가)
    cbatch = out.c.std(dim=0).mean().item()
    return sum(act), act, cbatch, out.disc.item()


def run(tag, beta, **mkw):
    ds = Shapes3D("data")
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=6, drop_last=True)
    it = stream(loader)
    xev = next(it).cuda()
    m = DAVAE(latent_dim=32, in_channels=3, n_patterns=NP, beta=beta, **mkw).cuda()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    print(f"\n[{tag}] beta={beta} {mkw}")
    for i in range(STEPS):
        x = next(it).cuda()
        out = m(x, beta=beta)
        loss, recon, _ = attractor_loss(out, x)
        opt.zero_grad(); loss.backward(); opt.step()
        if (i + 1) % 100 == 0:
            tot, act, cbat, d = diversity(m, xev, beta)
            print(f"  step{i+1:4d} recon={recon.item():7.1f} active={tot:3d}/128 "
                  f"c_batch_std={cbat:5.2f} dist→pat={d:.3f}")
    tot, act, cbat, d = diversity(m, xev, beta)
    print(f"  FINAL active per-factor={act} total={tot}/128 dist→pat={d:.3f}")
    return tot, recon.item()


def main():
    # soft 영역 탐색: 작은 패턴노름·c·β -> 비포화 -> gradient 흐름?
    res = {}
    res["soft pat=1 s=1 β=1"] = run("soft pat=1 s=1 β=1", 1.0,
                                    context_norm=True, context_scale=1.0, pattern_scale=1.0)
    res["soft pat=1 s=1 β=0.5"] = run("soft pat=1 s=1 β=0.5", 0.5,
                                      context_norm=True, context_scale=1.0, pattern_scale=1.0)
    res["soft pat=0.5 s=0.5 β=2"] = run("soft pat=0.5 s=0.5 β=2", 2.0,
                                        context_norm=True, context_scale=0.5, pattern_scale=0.5)
    print("\n=== 요약 (활성 total/128, recon) ===")
    for k, (t, r) in res.items():
        print(f"  {k:22s}: active={t:3d}/128  recon={r:.1f}")
    print("baseline collapse=8/128 recon~1500. soft 영역서 active↑·recon↓·dist→pat>0 이면 학습 가능.")


if __name__ == "__main__":
    main()
