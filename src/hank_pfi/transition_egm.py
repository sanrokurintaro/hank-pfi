"""Finite-horizon ("MIT shock") household transition, EGM version --
the counterpart to ``transition.py`` built on ``household_egm.py``
instead of DiscreteDP.

EGM is a *more* natural fit for this than the discrete solver: a single
EGM backward step already takes "next period's consumption policy" and
returns "this period's consumption policy" at fixed prices -- which is
exactly what a perfect-foresight transition needs, just with
period-specific (r_t, w_t) instead of one fixed (r, w). No separate
Bellman-max machinery is required the way ``transition.py`` needed for
the discrete solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distribution_egm import _lottery_transition
from .household_egm import EGMSolution, egm_backward_step
from .params import HouseholdParams


@dataclass
class TransitionResultEGM:
    C: np.ndarray
    K: np.ndarray


def solve_mit_shock_egm(
    p: HouseholdParams,
    steady: EGMSolution,
    dist0: np.ndarray,
    r_path: np.ndarray,
    w_path: np.ndarray,
    r_ss: float,
) -> TransitionResultEGM:
    """Perfect-foresight transition for a finite price path (r_path,
    w_path), under the same assumptions as ``transition.solve_mit_shock``:
    the economy has returned to ``steady`` by period T, and starts period
    0 at distribution ``dist0``. ``r_ss`` is the steady-state real rate,
    needed as the Euler-equation discount rate for the *last* period in
    the horizon (the return on savings made at T-1, realized at the
    steady state beyond T) -- see ``household_egm.egm_backward_step``'s
    docstring for why this has to be passed explicitly instead of reusing
    r_path.
    """
    T = len(r_path)
    a_grid, e_grid, P_e = steady.a_grid, steady.e_grid, steady.P_e

    c_next = steady.policy_c
    policies_a = [None] * T
    for t in range(T - 1, -1, -1):
        r_next = r_path[t + 1] if t + 1 < T else r_ss
        c_t = egm_backward_step(
            p, r_path[t], w_path[t], a_grid, e_grid, P_e, c_next, r_next=r_next
        )
        coh_t = (1.0 + r_path[t]) * a_grid[:, None] + w_path[t] * e_grid[None, :]
        policies_a[t] = coh_t - c_t
        c_next = c_t

    dist = np.asarray(dist0).reshape(-1).copy()
    n_a, n_e = len(a_grid), len(e_grid)

    C = np.empty(T)
    K = np.empty(T)
    for t in range(T):
        policy_a_t = policies_a[t]
        coh_t = (1.0 + r_path[t]) * a_grid[:, None] + w_path[t] * e_grid[None, :]
        policy_c_t = coh_t - policy_a_t
        dist_2d = dist.reshape(n_a, n_e)

        C[t] = float(np.sum(dist_2d * policy_c_t))
        K[t] = float(np.sum(dist_2d * policy_a_t))

        P_t = _lottery_transition(policy_a_t, a_grid, P_e)
        dist = np.asarray(dist @ P_t).ravel()

    return TransitionResultEGM(C=C, K=K)
