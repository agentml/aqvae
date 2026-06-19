"""C3 검증 (일회용): factorized head (8×4D Hopfield).

실행: python -m src.fixed_point._verify_c3
검증: (1) 형상, (2) factor 독립성(block-diagonal), (3) end-to-end 미분,
      (4) factor 별 이산 인덱스, (5) factor 별 multi-stability.
"""
import torch

from ..attractor import FactorizedHopfield

DT = torch.float64
torch.manual_seed(0)


def main():
    NF, FD, K = 8, 4, 16
    head = FactorizedHopfield(NF, FD, K, beta=2.0, learn_beta=True).double()
    B = 16
    c = torch.randn(B, NF * FD, dtype=DT, requires_grad=True)

    # 수렴 설정(β=8, fpi: element-wise 라 factor 자연 독립): (1)(2) 용
    conv_kw = dict(bw_type="onestep", beta=8.0, solver="fpi", max_iter=200, tol=1e-9)

    # (1) 형상
    y_star, info = head.infer(c, **conv_kw)
    print(f"(1) 형상: context {tuple(c.shape)} -> y* {tuple(y_star.shape)}  "
          f"solver iters={info['iters']} converged={info['converged']}")
    ok_shape = y_star.shape == c.shape and info["converged"]

    # (2) factor 독립성: factor 0 의 context 만 바꾸면 factor 0 출력만 변해야
    with torch.no_grad():
        c2 = c.detach().clone()
        c2[:, 0:FD] += 1.0                                  # factor 0 슬라이스만 교란
        y2, _ = head.infer(c2.requires_grad_(True), **conv_kw)
        diff = (y2 - y_star.detach()).view(B, NF, FD).norm(dim=(0, 2))  # factor별 변화량
    others = diff[1:].max().item()
    print(f"(2) 독립성: factor0 교란 -> Δfactor0={diff[0]:.3f}, max Δother={others:.2e}  "
          f"({'독립 OK' if others < 1e-6 else 'NOT independent'})")
    ok_indep = others < 1e-6

    # (3) end-to-end 미분 (moderate β=2 — C2 sweet spot, grad 흐름)
    cg = c.detach().clone().requires_grad_(True)
    yg, _ = head.infer(cg, bw_type="onestep", beta=2.0, solver="anderson", max_iter=100, tol=1e-5)
    g = torch.randn(B, NF * FD, dtype=DT)
    L = (yg * g).sum()
    params = head.grad_params(cg)
    grads = torch.autograd.grad(L, params, allow_unused=True)
    gc = grads[0]
    n_pat = sum(1 for p in params[1:] if p.dim() == 2)
    n_nonzero_pat = sum(1 for gr, p in zip(grads[1:], params[1:])
                        if p.dim() == 2 and gr is not None and gr.norm() > 0)
    print(f"(3) 미분: |grad_context|={gc.norm():.3f}, patterns grad nonzero "
          f"{n_nonzero_pat}/{n_pat}, beta grad 존재={grads[2] is not None}")
    ok_grad = gc.norm() > 1e-3 and n_nonzero_pat == NF

    # (4) factor 별 이산 인덱스
    idx = head.assign(y_star.detach())
    print(f"(4) assign: {tuple(idx.shape)} (B,NF), 값역 [{idx.min().item()},{idx.max().item()}] "
          f"(0..{K-1})")
    ok_assign = idx.shape == (B, NF) and idx.min() >= 0 and idx.max() < K

    # (5) factor 별 multi-stability: 무작위 context 많이 -> factor별 distinct 인덱스 수
    cbig = torch.randn(2000, NF * FD, dtype=DT, requires_grad=True)
    ybig, _ = head.infer(cbig, bw_type="onestep", beta=8.0)   # 고β로 또렷하게
    idxbig = head.assign(ybig.detach())
    distinct = [int(idxbig[:, i].unique().numel()) for i in range(NF)]
    print(f"(5) factor별 활성 attractor 수(β=8): {distinct}")

    ok = ok_shape and ok_indep and ok_grad and ok_assign
    print(f"\nC3 {'PASS ✓' if ok else 'FAIL ✗'}  "
          f"(shape={ok_shape}, indep={ok_indep}, grad={ok_grad}, assign={ok_assign})")


if __name__ == "__main__":
    main()
