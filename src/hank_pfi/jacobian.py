"""The household block's sequence-space Jacobians (Auclert, Bardoczy,
Rognlie & Straub, 2021): T x T matrices J_r, J_w where

    J_r[t, s] = d log C_t / d r_s      (holding the wage path fixed)
    J_w[t, s] = d log C_t / d log w_s  (holding the real-rate path fixed)

computed by direct numerical differentiation: perturb a single period's
price, re-solve the household's finite-horizon transition
(``transition.solve_mit_shock``), and read off the resulting path of
aggregate consumption. This is the "brute-force" way to get a
sequence-space Jacobian -- T perturbations x 2 (central difference) x one
MIT-shock solve each -- as opposed to Auclert et al.'s "fake news
algorithm", which gets the same T x T matrix from O(T) object reuse
instead of O(T) full re-solves. The brute-force version is what is
implemented here because it is far simpler to get right, and at the grid
sizes and horizons this package targets (T ~ tens, n_a ~ hundreds) each
MIT shock solve takes well under a second, so the O(T) solves remain
fast enough. Swapping in the fake news algorithm later (same output
shape) is the natural efficiency upgrade if T needs to grow into the
hundreds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .household import HouseholdSolution
from .params import HouseholdParams
from .transition import solve_mit_shock


@dataclass
class HouseholdJacobians:
    J_r: np.ndarray   # (T, T), d log C_t / d r_s
    J_w: np.ndarray   # (T, T), d log C_t / d log w_s
    T: int


def household_jacobians(
    p: HouseholdParams,
    steady: HouseholdSolution,
    dist0: np.ndarray,
    r_ss: float,
    w_ss: float,
    C_ss: float,
    T: int,
    dr: float = 1e-4,
    dlogw: float = 1e-4,
) -> HouseholdJacobians:
    """Build the household block's consumption Jacobians w.r.t. the real
    rate path and the (log) wage path, both of length T, by central
    differences around the steady state (r_ss, w_ss).
    """
    J_r = np.empty((T, T))
    J_w = np.empty((T, T))

    r_base = np.full(T, r_ss)
    w_base = np.full(T, w_ss)

    for s in range(T):
        r_up, r_dn = r_base.copy(), r_base.copy()
        r_up[s] += dr
        r_dn[s] -= dr
        C_up = solve_mit_shock(p, steady, dist0, r_up, w_base).C
        C_dn = solve_mit_shock(p, steady, dist0, r_dn, w_base).C
        J_r[:, s] = (C_up - C_dn) / (2 * dr) / C_ss

        w_up, w_dn = w_base.copy(), w_base.copy()
        w_up[s] *= np.exp(dlogw)
        w_dn[s] *= np.exp(-dlogw)
        C_up_w = solve_mit_shock(p, steady, dist0, r_base, w_up).C
        C_dn_w = solve_mit_shock(p, steady, dist0, r_base, w_dn).C
        J_w[:, s] = (C_up_w - C_dn_w) / (2 * dlogw) / C_ss

    return HouseholdJacobians(J_r=J_r, J_w=J_w, T=T)
