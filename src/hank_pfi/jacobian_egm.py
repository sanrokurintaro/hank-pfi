"""Household sequence-space Jacobians, EGM version -- same finite-
difference approach as ``jacobian.py``, built on
``transition_egm.solve_mit_shock_egm`` instead of the discrete-choice
``transition.solve_mit_shock``.

The whole point of this branch: because EGM's policy function is smooth
in the state (no discrete-grid snapping), a finite difference here should
not suffer the discretization artifacts documented in ``jacobian.py``'s
module docstring -- i.e. it should work with much smaller, more
economically meaningful perturbation sizes than the brute-force steps
(dr=0.02, dlogw=0.1) the DiscreteDP version needed to average out grid
noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .household_egm import EGMSolution
from .params import HouseholdParams
from .transition_egm import solve_mit_shock_egm


@dataclass
class HouseholdJacobiansEGM:
    J_r: np.ndarray
    J_w: np.ndarray
    T: int


def household_jacobians_egm(
    p: HouseholdParams,
    steady: EGMSolution,
    dist0: np.ndarray,
    r_ss: float,
    w_ss: float,
    C_ss: float,
    T: int,
    dr: float = 1e-4,
    dlogw: float = 1e-4,
) -> HouseholdJacobiansEGM:
    J_r = np.empty((T, T))
    J_w = np.empty((T, T))

    r_base = np.full(T, r_ss)
    w_base = np.full(T, w_ss)

    for s in range(T):
        r_up, r_dn = r_base.copy(), r_base.copy()
        r_up[s] += dr
        r_dn[s] -= dr
        C_up = solve_mit_shock_egm(p, steady, dist0, r_up, w_base, r_ss).C
        C_dn = solve_mit_shock_egm(p, steady, dist0, r_dn, w_base, r_ss).C
        J_r[:, s] = (C_up - C_dn) / (2 * dr) / C_ss

        w_up, w_dn = w_base.copy(), w_base.copy()
        w_up[s] *= np.exp(dlogw)
        w_dn[s] *= np.exp(-dlogw)
        C_up_w = solve_mit_shock_egm(p, steady, dist0, r_base, w_up, r_ss).C
        C_dn_w = solve_mit_shock_egm(p, steady, dist0, r_base, w_dn, r_ss).C
        J_w[:, s] = (C_up_w - C_dn_w) / (2 * dlogw) / C_ss

    return HouseholdJacobiansEGM(J_r=J_r, J_w=J_w, T=T)
