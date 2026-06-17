"""고정점(attractor) 머신 — solver + implicit differentiation.

Solve-the-Loop (Fein-Ashley & Rashidinejad, 2026; MIT) 의
`attractor/models/attractor/{solvers.py, attractor.py}` 를 단일 GPU 용도로
이식·단순화한 것. 모델에 독립적인 재사용 부품.
"""
from .solvers import anderson_solve, fpi_solve, get_solver
from .implicit import attach_ift_grad, solve_fixed_point

__all__ = [
    "anderson_solve", "fpi_solve", "get_solver",
    "attach_ift_grad", "solve_fixed_point",
]
