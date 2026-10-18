"""Layer (2): the idiosyncratic household Bellman problem.

This is the core of "Proposal B": instead of writing a value/policy
iteration loop by hand, the (a, e) state space is discretized and handed
to QuantEcon's ``DiscreteDP``, whose ``method='policy_iteration'`` *is*
Howard's policy improvement algorithm (PFI). We only build the reward and
transition arrays; the iteration itself is QuantEcon's, not ours.

The state-action pair (SA) formulation is used with a sparse transition
matrix: for a fixed asset choice a', the only uncertainty is next period's
income, so each feasible (state, action) row has only n_e nonzero
transition probabilities out of n_a * n_e columns. Representing this
densely (n_a*n_e, n_a, n_a*n_e) would blow up memory for any reasonably
fine grid; the sparse SA formulation is what makes this usable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import quantecon as qe
from scipy import sparse

from .grids import asset_grid, income_process
from .params import HouseholdParams

_CONSUMPTION_FLOOR = 1e-10  # below this, consumption is treated as infeasible


def _utility(c: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return np.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)


@dataclass
class RewardTransition:
    """SA-pair reward/transition arrays for one period's prices (r, w).

    Shared by both the stationary PFI solver (``solve_household``, which
    hands this to ``DiscreteDP``) and the finite-horizon backward Bellman
    recursion used for MIT-shock / Jacobian calculations in
    ``transition.py``, so the (somewhat fiddly) sparse-matrix construction
    logic lives in exactly one place.
    """

    s_indices: np.ndarray   # (L,) int, state index of each feasible pair
    a_indices: np.ndarray   # (L,) int, action (a') index of each feasible pair
    R: np.ndarray           # (L,) float, flow utility
    Q: sparse.csr_matrix    # (L, n_states) transition probabilities
    coh: np.ndarray         # (n_a, n_e) cash on hand, for recovering c from a'
    n_a: int
    n_e: int

    @property
    def n_states(self) -> int:
        return self.n_a * self.n_e


def build_reward_transition(
    p: HouseholdParams,
    r: float,
    w: float,
    a_grid: np.ndarray,
    e_grid: np.ndarray,
    P_e: np.ndarray,
) -> RewardTransition:
    """Build the SA-pair reward vector and sparse transition matrix for a
    single period's prices (r, w), given pre-computed grids.

    For a fixed asset choice a', next period's uncertainty is only over
    income, so each feasible (state, action) row has exactly n_e nonzero
    transition probabilities out of n_a * n_e columns -- representing this
    densely as (n_a*n_e, n_a, n_a*n_e) would blow up memory for any
    reasonably fine grid, so we build the sparse SA formulation directly.
    """
    n_a, n_e = p.n_a, p.n_e
    n_states = n_a * n_e

    coh = (1.0 + r) * a_grid[:, None] + w * e_grid[None, :]
    c = coh[:, :, None] - a_grid[None, None, :]
    feasible = c > _CONSUMPTION_FLOOR

    feasible_flat = feasible.reshape(n_states, n_a)
    s_indices, a_indices = np.nonzero(feasible_flat)
    R = _utility(c.reshape(n_states, n_a)[s_indices, a_indices], p.gamma)

    e_of_row = s_indices % n_e
    L = len(s_indices)
    col_offsets = np.arange(n_e)
    indices = (a_indices[:, None] * n_e + col_offsets[None, :]).ravel()
    data = P_e[e_of_row, :].ravel()
    indptr = np.arange(0, (L + 1) * n_e, n_e)
    Q = sparse.csr_matrix((data, indices, indptr), shape=(L, n_states))

    return RewardTransition(
        s_indices=s_indices, a_indices=a_indices, R=R, Q=Q,
        coh=coh, n_a=n_a, n_e=n_e,
    )


@dataclass
class HouseholdSolution:
    a_grid: np.ndarray            # (n_a,)
    e_grid: np.ndarray            # (n_e,)
    P_e: np.ndarray               # (n_e, n_e) income transition matrix
    v: np.ndarray                 # (n_a, n_e) value function
    policy_a_index: np.ndarray    # (n_a, n_e) int, index into a_grid of a'
    policy_a: np.ndarray          # (n_a, n_e) float, a' in levels
    policy_c: np.ndarray          # (n_a, n_e) float, consumption
    ddp_result: object            # raw QuantEcon DPSolveResult (has .mc, etc.)
    rt: RewardTransition           # the SA-pair arrays used to build it


def solve_household(p: HouseholdParams, r: float, w: float) -> HouseholdSolution:
    """Solve the household's problem for given prices (r, w) via PFI
    (Howard improvement), delegated to ``quantecon.markov.DiscreteDP``.
    """
    a_grid = asset_grid(p)
    e_grid, P_e = income_process(p)
    n_a, n_e = p.n_a, p.n_e

    rt = build_reward_transition(p, r, w, a_grid, e_grid, P_e)

    ddp = qe.markov.DiscreteDP(rt.R, rt.Q, p.beta, rt.s_indices, rt.a_indices)
    res = ddp.solve(method="policy_iteration")

    v = res.v.reshape(n_a, n_e)
    policy_a_index = res.sigma.reshape(n_a, n_e)
    policy_a = a_grid[policy_a_index]
    policy_c = rt.coh - policy_a

    return HouseholdSolution(
        a_grid=a_grid,
        e_grid=e_grid,
        P_e=P_e,
        v=v,
        policy_a_index=policy_a_index,
        policy_a=policy_a,
        policy_c=policy_c,
        ddp_result=res,
        rt=rt,
    )
