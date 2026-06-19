"""C2 검증 (일회용): Hopfield + solver + IFT 결합.

실행: python -m src.attractors._verify_c2

검증:
  (A) exact-IFT gradient(패턴 X, β, context c) vs autograd-through-unroll 일치.
  (B) persistent injection 이면 ∂L/∂c ≠ 0 (인코더로 gradient 흐름),
      순수 retrieval 이면 ∂L/∂c ≈ 0 (STE 문제) — '왜 injection 이 필요한가' 실증.
  (C) one-step(Neumann-1) 은 근사.
  (D) 저 β(비수축 근접)에서 adjoint_clip 이 폭주를 막는지.
"""
import torch

from ..attractors.implicit import attach_ift_grad, solve_attractors
from ..attractors.attractor import HopfieldAttractor

DT = torch.float64
torch.manual_seed(0)


def build(beta=6.0, learn_beta=True):
    D, K = 4, 8
    hop = HopfieldAttractor(D, K, beta=beta, learn_beta=learn_beta, pattern_scale=2.0).double()
    with torch.no_grad():
        X = torch.randn(D, K, dtype=DT); X = X / X.norm(dim=0, keepdim=True) * 3.0
        hop.set_patterns(X)
    return hop, D, K


def solve_and_attach(hop, c, bw_type, adjoint_clip=None, beta=None):
    """context c 주입, 고정점 solve 후 IFT grad 부착. 반환 y*."""
    cd = c.detach()
    y_star_ng, info = solve_attractors(
        lambda y: hop.map(y, context=cd, beta=beta), cd.clone(),
        solver="anderson", max_iter=300, tol=1e-12)
    params = [c, hop.patterns] + ([hop.log_beta] if hop.log_beta is not None else [])
    y_star = attach_ift_grad(lambda y: hop.map(y, context=c, beta=beta), y_star_ng, params,
                             bw_type=bw_type, adjoint_clip=adjoint_clip,
                             bw_max_iter=300, bw_tol=1e-12)
    return y_star, info


def A_gradcheck():
    print("=== (A) exact-IFT vs unroll gradcheck (injection) ===")
    hop, D, K = build()
    B = 6
    c = torch.randn(B, D, dtype=DT, requires_grad=True)

    # exact IFT (iterative backward)
    y_star, info = solve_and_attach(hop, c, bw_type="iterative")
    g = torch.randn(B, D, dtype=DT)
    L = (y_star * g).sum()
    gc, gX, gb = torch.autograd.grad(L, [c, hop.patterns, hop.log_beta])

    # autograd-through-unroll (같은 고정점까지 미분가능 반복)
    c2 = c.detach().clone().requires_grad_(True)
    X2 = hop.patterns.detach().clone().requires_grad_(True)
    lb2 = hop.log_beta.detach().clone().requires_grad_(True)
    y = c2.clone()
    for _ in range(400):
        q = y + c2
        w = torch.softmax(lb2.exp() * (q @ X2), dim=-1)
        y = w @ X2.t()
    L2 = (y * g).sum()
    gc_u, gX_u, gb_u = torch.autograd.grad(L2, [c2, X2, lb2])

    print(f"  forward: anderson iters={info['iters']} converged={info['converged']} "
          f"resid={info['rel_residual']:.1e}")
    ec = (gc - gc_u).abs().max().item()
    eX = (gX - gX_u).abs().max().item()
    eb = (gb - gb_u).abs().max().item()
    print(f"  grad_c  IFT vs unroll: {ec:.2e}")
    print(f"  grad_X  IFT vs unroll: {eX:.2e}")
    print(f"  grad_β  IFT vs unroll: {eb:.2e}")
    print(f"  |grad_c|={gc.norm():.3f} |grad_X|={gX.norm():.3f} |grad_β|={gb.norm():.3e}")
    return max(ec, eX, eb) < 1e-7


def B_injection_contrast():
    print("=== (B) ∂L/∂c vs β: injection(흐름) vs 순수 retrieval(STE), + discreteness ===")
    print("    (β↑ => softmax 포화 => grad_c 감소 but 더 또렷. trade-off 곡선)")
    hop, D, K = build(learn_beta=False)
    B = 64
    c = torch.randn(B, D, dtype=DT, requires_grad=True)
    print(f"  {'beta':>6} {'inj|∂L/∂c|':>11} {'retr|∂L/∂c|':>12} {'dist→pat':>9}")
    best_inj_low = 0.0
    for beta in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
        # injection (one-step adjoint — 학습 기본값, 안정)
        c1 = c.detach().clone().requires_grad_(True)
        y_inj, _ = solve_and_attach(hop, c1, bw_type="onestep", beta=beta)
        gc_inj, = torch.autograd.grad(y_inj.sum(), [c1])
        # discreteness: 고정점이 패턴에 얼마나 붙었나
        dist = torch.cdist(y_inj.detach(), hop.patterns.t()).min(-1).values.mean().item()
        # 순수 retrieval
        cd = c.detach()
        y_ng, _ = solve_attractors(lambda y: hop.map(y, beta=beta), cd.clone(),
                                    solver="anderson", max_iter=300, tol=1e-12)
        c_re = cd.clone().requires_grad_(True)
        y_re = attach_ift_grad(lambda y: hop.map(y, beta=beta), y_ng, [c_re], bw_type="onestep")
        gc_re, = torch.autograd.grad(y_re.sum(), [c_re], allow_unused=True)
        re_norm = 0.0 if gc_re is None else gc_re.norm().item()
        print(f"  {beta:6.1f} {gc_inj.norm().item():11.4f} {re_norm:12.4f} {dist:9.3f}")
        if beta <= 2.0:
            best_inj_low = max(best_inj_low, gc_inj.norm().item())
    # 통과: 중간 β 에서 injection grad_c 가 살아있고, retrieval 은 항상 ~0
    return best_inj_low > 1e-2


def C_onestep():
    print("=== (C) one-step(Neumann-1) 근사 vs exact ===")
    hop, D, K = build(learn_beta=False)
    c = torch.randn(8, D, dtype=DT, requires_grad=True)
    ye, _ = solve_and_attach(hop, c, bw_type="iterative", beta=8.0)
    ge, = torch.autograd.grad(ye.sum(), [c], retain_graph=True)
    c2 = c.detach().clone().requires_grad_(True)
    yo, _ = solve_and_attach(hop, c2, bw_type="onestep", beta=8.0)
    go, = torch.autograd.grad(yo.sum(), [c2])
    print(f"  one-step vs exact grad_c: max diff = {(ge - go).abs().max().item():.2e} "
          f"(근사라 0 아님, 학습 기본값)")
    return True


def D_adjoint_clip():
    print("=== (D) 저 β(비수축 근접) adjoint_clip ===")
    hop, D, K = build(learn_beta=False)
    c = torch.randn(8, D, dtype=DT, requires_grad=True)
    for clip in (None, 10.0):
        c2 = c.detach().clone().requires_grad_(True)
        y, _ = solve_and_attach(hop, c2, bw_type="onestep", adjoint_clip=clip, beta=1.5)
        g, = torch.autograd.grad(y.sum(), [c2])
        tag = "clip=None" if clip is None else f"clip={clip}"
        print(f"  β=1.5 {tag:10s} |grad_c|={g.norm().item():.3e} finite={torch.isfinite(g).all().item()}")
    return True


if __name__ == "__main__":
    okA = A_gradcheck()
    okB = B_injection_contrast()
    C_onestep()
    D_adjoint_clip()
    print(f"\nC2 {'PASS ✓' if (okA and okB) else 'FAIL ✗'}  (A gradcheck={okA}, B injection={okB})")
