"""고정점 y* = T(y*) 에 implicit-function-theorem(IFT) gradient 부착.

Solve-the-Loop 공식 소스(MIT) `attractor/models/attractor/attractor.py` 의
`_IFTAttach` / `_maybe_clip_jv` 를 단일 프로세스용으로 이식·단순화.

수식: y* = T(y*), J = ∂T/∂y|_{y*} 일 때
    ∂L/∂θ = u^T ∂T(y*)/∂θ,   u = (I - J^T)^{-1} v,   v = ∂L/∂y*.
- bw_type="onestep": u ≈ v + J^T v  (Neumann-1, 메모리 O(1) — 학습 기본).
- bw_type="iterative"/"anderson": u = J^T u + v 를 수렴까지 풀어 *정확* (검증·정밀용).

사용:
    y_star_ng, info = solve_attractors(lambda y: T(y, c.detach()), y0, ...)
    y_star = attach_ift_grad(lambda y: T(y, c), y_star_ng, [c, *head.parameters()])
"""
from __future__ import annotations

import torch

from .solvers import anderson_solve, get_solver


def _maybe_clip_jv(Jv: torch.Tensor, v: torch.Tensor, adjoint_clip) -> torch.Tensor:
    """J^T v 의 per-sample norm 이 adjoint_clip * ||v|| 를 넘으면 rescale.
    head 가 덜 수축적인 영역으로 흘러도 Neumann-1 adjoint 를 안전하게 유지."""
    if adjoint_clip is None:
        return Jv
    B = Jv.size(0)
    v_norm = v.reshape(B, -1).norm(dim=1).clamp_min(1e-12)
    Jv_norm = Jv.reshape(B, -1).norm(dim=1)
    bound = float(adjoint_clip) * v_norm
    scale = torch.where(Jv_norm > bound, bound / Jv_norm.clamp_min(1e-12),
                        torch.ones_like(Jv_norm))
    return Jv * scale.view(B, *([1] * (Jv.ndim - 1)))


def solve_attractors(T, y0, solver: str = "anderson", max_iter: int = 30,
                      tol: float = 1e-3, min_iter: int = 0, **solver_kwargs):
    """no-grad 로 y* = T(y*) 를 푼다. (T, y0 의 grad 는 무시 — forward 전용)"""
    name = solver
    if name == "anderson" and max_iter < 3:
        name = "fpi"
    fn = get_solver(name)
    kw = dict(max_iter=max_iter, tol=tol, min_iter=min_iter)
    if name == "anderson":
        kw.update(solver_kwargs)
    with torch.no_grad():
        y_star, info = fn(T, y0, **kw)
    return y_star, info


class _Holder:
    """autograd graph 를 든 텐서를 Function 입력 밖에서 운반(이중 backward 방지)."""
    def __init__(self, y_out, y_s, params, cfg):
        self.y_out = y_out
        self.y_s = y_s
        self.params = params
        self.cfg = cfg


class _IFTAttach(torch.autograd.Function):
    """forward 는 항등(고정점 값 그대로). backward 에서 adjoint
    (I - J^T) u = v 를 풀고 u 를 T 그래프로 흘려 params gradient 산출.
    params 를 Function 입력으로 받아 각 텐서의 .grad 가 채워지도록 함."""

    @staticmethod
    def forward(ctx, y_star_value, holder, *params):
        ctx.holder = holder
        ctx.n_params = len(params)
        return y_star_value

    @staticmethod
    def backward(ctx, grad_y_star):
        h: _Holder = ctx.holder
        y_out, y_s, cfg = h.y_out, h.y_s, h.cfg
        v = grad_y_star.contiguous()

        bw_type = cfg["bw_type"]
        if bw_type == "onestep":
            Jv, = torch.autograd.grad(y_out, y_s, v, retain_graph=True)
            Jv = _maybe_clip_jv(Jv, v, cfg["adjoint_clip"])
            u = v + Jv
        else:
            def T_op(u):
                Ju, = torch.autograd.grad(y_out, y_s, u, retain_graph=True)
                return Ju + v  # 고정점 u* = J^T u* + v  ==> u* = (I-J^T)^{-1} v
            if bw_type == "anderson":
                with torch.no_grad():
                    u, _ = anderson_solve(T_op, v.detach().clone(),
                                          max_iter=cfg["bw_max_iter"], tol=cfg["bw_tol"],
                                          min_iter=cfg["bw_min_iter"])
            else:  # "iterative"
                u = v.detach().clone()
                for it in range(cfg["bw_max_iter"]):
                    u_new = T_op(u)
                    diff = (u_new - u).reshape(u.size(0), -1).norm(dim=1)
                    ref = u_new.reshape(u.size(0), -1).norm(dim=1).clamp_min(1e-9)
                    u = u_new
                    if (it + 1) >= cfg["bw_min_iter"] and (diff / ref).max().item() < cfg["bw_tol"]:
                        break

        targets = tuple(p for p in h.params if p.requires_grad)
        grads = ()
        if targets and y_out.requires_grad:
            grads = torch.autograd.grad(y_out, targets, u, retain_graph=False,
                                        allow_unused=True)
        gmap = {id(t): g for t, g in zip(targets, grads)}

        def _g(p):
            if not p.requires_grad:
                return None
            g = gmap.get(id(p), None)
            return torch.zeros_like(p) if g is None else g

        param_grads = tuple(_g(p) for p in h.params)
        return (None, None) + param_grads


def attach_ift_grad(T, y_star, params, bw_type: str = "onestep", adjoint_clip=None,
                    bw_max_iter: int = 30, bw_tol: float = 1e-6, bw_min_iter: int = 0):
    """y_star(고정점 값, no-grad)에 IFT gradient 를 부착해 반환.

    T: y -> T(y) (closure; params 에 들어간 텐서들에 의존). y_star = T(y_star).
    params: gradient 가 필요한 텐서들(예: encoder context c, head.parameters()).
    """
    y_s = y_star.detach().requires_grad_(True)
    y_out = T(y_s)
    if not y_out.requires_grad:
        return y_star.detach()
    cfg = dict(bw_type=bw_type, adjoint_clip=adjoint_clip,
               bw_max_iter=int(bw_max_iter), bw_tol=float(bw_tol),
               bw_min_iter=int(bw_min_iter))
    holder = _Holder(y_out, y_s, tuple(params), cfg)
    return _IFTAttach.apply(y_star.detach(), holder, *params)
