"""Stationary distribution for the EGM household block.

``household_egm.py``'s policy function is continuous: the optimal a'
generally falls *between* two grid points, not exactly on one. So, unlike
``distribution.py`` (which reads an exact transition matrix off
``DiscreteDP``'s discrete policy), here we build the transition matrix
ourselves using a "lottery": each household's mass is split between the
two nearest grid points in proportion to how close its continuous choice
is to each one, so that the *expected* next-period asset holding exactly
equals the continuous policy (Young, 2010-style non-stochastic
simulation, same idea as ``distribution.py``'s power-iteration choice,
now also doing double duty to handle off-grid landings).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse

from .household_egm import EGMSolution

_TOP_GRID_WARNING_THRESHOLD = 1e-3


def _lottery_transition(policy_a: np.ndarray, a_grid: np.ndarray, P_e: np.ndarray) -> sparse.csr_matrix:
    n_a, n_e = policy_a.shape
    n_states = n_a * n_e

    a_flat = policy_a.reshape(-1)
    idx_hi = np.searchsorted(a_grid, a_flat, side="left").clip(1, n_a - 1)
    idx_lo = idx_hi - 1
    a_lo, a_hi = a_grid[idx_lo], a_grid[idx_hi]
    span = np.where(a_hi > a_lo, a_hi - a_lo, 1.0)
    w_hi = np.clip((a_flat - a_lo) / span, 0.0, 1.0)
    w_lo = 1.0 - w_hi

    e_of_s = np.arange(n_states) % n_e
    col_offsets = np.arange(n_e)

    indices_lo = (idx_lo[:, None] * n_e + col_offsets[None, :])
    indices_hi = (idx_hi[:, None] * n_e + col_offsets[None, :])
    data_lo = w_lo[:, None] * P_e[e_of_s, :]
    data_hi = w_hi[:, None] * P_e[e_of_s, :]

    indices = np.concatenate([indices_lo, indices_hi], axis=1).ravel()
    data = np.concatenate([data_lo, data_hi], axis=1).ravel()
    indptr = np.arange(0, (n_states + 1) * 2 * n_e, 2 * n_e)

    P = sparse.csr_matrix((data, indices, indptr), shape=(n_states, n_states))
    return P


def stationary_distribution_egm(
    sol: EGMSolution, tol: float = 1e-12, max_iter: int = 100_000
) -> np.ndarray:
    """Power iteration from the uniform distribution, same rationale as
    ``distribution.stationary_distribution`` (see its docstring): avoids
    depending on an eigenvector solver picking an arbitrary invariant
    measure when the chain is reducible.
    """
    n_a, n_e = sol.policy_a.shape
    P = _lottery_transition(sol.policy_a, sol.a_grid, sol.P_e)

    n = n_a * n_e
    dist = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        dist_next = np.asarray(dist @ P).ravel()
        if np.max(np.abs(dist_next - dist)) < tol:
            dist = dist_next
            break
        dist = dist_next
    else:
        warnings.warn(
            f"EGM stationary distribution did not converge to tol={tol} "
            f"within {max_iter} iterations.",
            stacklevel=2,
        )

    dist = dist.reshape(n_a, n_e)
    top_mass = dist[-1, :].sum()
    if top_mass > _TOP_GRID_WARNING_THRESHOLD:
        warnings.warn(
            f"{top_mass:.2%} of the EGM stationary distribution's mass sits "
            "at the top asset grid point -- a_max is likely binding. "
            "Increase HouseholdParams.a_max.",
            stacklevel=2,
        )
    return dist
