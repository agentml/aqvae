"""Phase B 검증 (일회용): solver(B1) + IFT gradcheck(B2).

실행: python -m src.attractors._verify
정확성 게이트 — 통과해야 Phase C(Hopfield) 진행.
"""
import torch

from ..attractors.implicit import attach_ift_grad, solve_attractors

torch.manual_seed(0)
DT = torch.float64


def b1_solver():
    """T(y)=0.5 y + b 의 고정점 y*=2b 를 fpi/anderson 이 맞히는가."""
    B, n = 4, 5
    b = torch.randn(B, n, dtype=DT)
    T = lambda y: 0.5 * y + b
    y_target = 2 * b
    ok = True
    for name in ("fpi", "anderson"):
        y_star, info = solve_attractors(T, torch.zeros(B, n, dtype=DT),
                                         solver=name, max_iter=200, tol=1e-12)
        err = (y_star - y_target).abs().max().item()
        print(f"  [B1 {name:8s}] iters={info['iters']:3d} converged={info['converged']} "
              f"max|y*-2b|={err:.2e}")
        ok &= err < 1e-6   # tol=1e-12 상대잔차 => O(1) 값에서 ~1e-8 절대오차면 정상
    return ok


def b2_gradcheck():
    """선형 고정점 T(y)=c + W y (W 수축). 고정점 y*=(I-W)^{-1} c.
    exact-IFT gradient 를 (a) 닫힌형, (b) autograd-through-unroll 과 비교."""
    B, n = 4, 6
    W = (torch.randn(n, n, dtype=DT) * 0.12).requires_grad_(True)   # 작게 => 수축
    c = torch.randn(B, n, dtype=DT, requires_grad=True)

    # forward: no-grad solve
    Wd, cd = W.detach(), c.detach()
    y0 = cd.clone()
    y_star_ng, info = solve_attractors(lambda y: cd + y @ Wd.T, y0,
                                        solver="anderson", max_iter=300, tol=1e-13)

    # exact IFT (iterative backward)
    T = lambda y: c + y @ W.T
    y_star = attach_ift_grad(T, y_star_ng, [c, W], bw_type="iterative",
                             bw_max_iter=300, bw_tol=1e-13)
    L = y_star.sum()
    gc_ift, gW_ift = torch.autograd.grad(L, [c, W])

    # (a) 닫힌형: y*=(I-W)^{-1}c => dL/dc = (I-W)^{-T} 1
    I = torch.eye(n, dtype=DT)
    inv = torch.linalg.inv(I - Wd)
    gc_closed = (inv.T @ torch.ones(n, dtype=DT)).expand(B, n)
    err_c_closed = (gc_ift - gc_closed).abs().max().item()

    # (b) autograd-through-unroll
    c2 = cd.clone().requires_grad_(True)
    W2 = Wd.clone().requires_grad_(True)
    y = c2.clone()
    for _ in range(400):
        y = c2 + y @ W2.T
    gc_un, gW_un = torch.autograd.grad(y.sum(), [c2, W2])
    err_c_un = (gc_ift - gc_un).abs().max().item()
    err_W_un = (gW_ift - gW_un).abs().max().item()

    print(f"  [B2 forward ] anderson iters={info['iters']} converged={info['converged']} "
          f"resid={info['rel_residual']:.1e}")
    print(f"  [B2 grad_c ] IFT vs closed-form : max diff = {err_c_closed:.2e}")
    print(f"  [B2 grad_c ] IFT vs unroll      : max diff = {err_c_un:.2e}")
    print(f"  [B2 grad_W ] IFT vs unroll      : max diff = {err_W_un:.2e}")

    # onestep(Neumann-1) 은 근사 — exact 와 다름을 확인(정상 거동)
    y_star_os = attach_ift_grad(T, y_star_ng, [c, W], bw_type="onestep")
    gc_os, = torch.autograd.grad(y_star_os.sum(), [c], retain_graph=True)
    err_os = (gc_os - gc_closed).abs().max().item()
    print(f"  [B2 onestep] Neumann-1 vs closed-form (근사라 0 아님): {err_os:.2e}")

    return max(err_c_closed, err_c_un, err_W_un) < 1e-7


if __name__ == "__main__":
    print("=== B1: solver ===")
    ok1 = b1_solver()
    print("=== B2: IFT gradcheck ===")
    ok2 = b2_gradcheck()
    print(f"\nPhase B {'PASS ✓' if (ok1 and ok2) else 'FAIL ✗'}  (B1={ok1}, B2={ok2})")
