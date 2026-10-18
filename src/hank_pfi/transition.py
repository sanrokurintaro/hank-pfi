"""Finite-horizon ("MIT shock") household transition: given a path of
prices {r_t, w_t} for t=0..T-1 that returns to the steady state by t=T,
solve the household's time-varying Bellman problem by plain backward
induction (one Bellman backup per period, starting from the steady-state
value function as the terminal condition), then push the distribution
forward through the resulting time-varying policies.

This is deliberately *not* ``DiscreteDP``/Howard improvement: Howard
improvement solves the fixed point of a *stationary* infinite-horizon
problem. Here the problem is non-stationary (prices differ period to
period) and has a known terminal value (the steady state households
converge back to), so a single backward pass is exact and is the standard
"MIT shock" / perfect-foresight transition method. ``household.py``'s
``build_reward_transition`` is reused for each period's (r_t, w_t) so the
reward/transition construction logic is not duplicated.

This module is what ``jacobian.py`` calls (twice per shock date, for a
central difference) to build the household's sequence-space Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .household import HouseholdSolution, build_reward_transition
from .params import HouseholdParams


@dataclass
class TransitionResult:
    C: np.ndarray   # (T,) aggregate consumption path
    K: np.ndarray   # (T,) aggregate asset-holdings path (end of period t)


def _quadratic_refine_a(
    dense: np.ndarray, policy_a_index: np.ndarray, a_grid: np.ndarray
) -> np.ndarray:
    """Refine the discrete argmax a'-choice to a continuous value via local
    quadratic interpolation of the Bellman objective around the argmax.

    Why this exists: with a *pure* discrete choice over a finite grid (no
    interpolation), a household whose true continuous optimum sits between
    two grid points gets rounded to whichever grid point is closest. For
    small price perturbations used in finite-difference Jacobians, this
    rounding is a step function: the realized consumption doesn't move at
    all until the perturbation is large enough to flip the argmax to a
    neighboring grid point, at which point it jumps by an amount of order
    the *grid spacing* rather than the (much smaller) perturbation size.
    That jump swamps the genuine marginal response and can even flip its
    sign in the aggregate -- confirmed empirically while building the
    Jacobian in ``jacobian.py`` (see also: this is exactly why production
    sequence-space-Jacobian toolkits solve for policy functions via EGM and
    interpolate between grid points, rather than using a raw discrete
    choice). Fitting a quadratic through the three objective values around
    the discrete argmax and taking its vertex as a continuous "shadow"
    choice removes most of this artifact without touching the discrete
    argmax itself (which still governs the distribution's transition
    matrix, exactly as in the stationary PFI solve).
    """
    n_states, n_a = dense.shape
    idx = policy_a_index
    idx_lo = np.clip(idx - 1, 0, n_a - 1)
    idx_hi = np.clip(idx + 1, 0, n_a - 1)
    rows = np.arange(n_states)

    x0, x1, x2 = a_grid[idx_lo], a_grid[idx], a_grid[idx_hi]
    y0, y1, y2 = dense[rows, idx_lo], dense[rows, idx], dense[rows, idx_hi]

    interior = (idx_lo < idx) & (idx_hi > idx) & np.isfinite(y0) & np.isfinite(y2)

    # quadratic through 3 (possibly unequally spaced) points via divided
    # differences: p(x) = A(x-x0)(x-x1) + B(x-x0) + y0, vertex at p'=0.
    with np.errstate(divide="ignore", invalid="ignore"):
        d01 = (y1 - y0) / np.where(x1 != x0, x1 - x0, np.nan)
        d12 = (y2 - y1) / np.where(x2 != x1, x2 - x1, np.nan)
        A = (d12 - d01) / np.where(x2 != x0, x2 - x0, np.nan)
        B = d01 - A * (x0 - x1)
        vertex = x1 - B / (2.0 * A)

    refine_ok = interior & (A < 0) & np.isfinite(vertex)
    refine_ok &= (vertex >= x0) & (vertex <= x2)

    a_refined = a_grid[idx].copy()
    a_refined[refine_ok] = vertex[refine_ok]
    return a_refined


def _bellman_backup(
    rt, v_next: np.ndarray, beta: float, n_a: int, n_e: int, a_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One backward Bellman step: given next period's value function (flat,
    length n_states) and this period's reward/transition (SA-pair form),
    return this period's (policy_a_index, v, a_refined) as (n_a, n_e)
    arrays. ``a_refined`` is the quadratic-interpolation-refined continuous
    choice (see ``_quadratic_refine_a``); ``policy_a_index`` is the exact
    discrete argmax used everywhere else (distribution dynamics, steady
    state).

    Infeasible (s, a') pairs are left at -inf via a dense scatter; with a
    sensible grid every state has at least a' = a_min feasible, so no row
    is ever entirely -inf.
    """
    n_states = n_a * n_e
    continuation = rt.Q @ v_next  # (L,), sparse mat-vec
    rhs = rt.R + beta * continuation  # (L,)

    dense = np.full((n_states, n_a), -np.inf)
    dense[rt.s_indices, rt.a_indices] = rhs
    policy_a_index = np.argmax(dense, axis=1)
    v = dense[np.arange(n_states), policy_a_index]
    a_refined = _quadratic_refine_a(dense, policy_a_index, a_grid)
    return (
        policy_a_index.reshape(n_a, n_e),
        v.reshape(n_a, n_e),
        a_refined.reshape(n_a, n_e),
    )


def _induced_row_transition(
    policy_a_index: np.ndarray, P_e: np.ndarray, n_a: int, n_e: int
):
    """Sparse (n_states, n_a*n_e) one-step transition matrix implied by a
    given policy a'(a, e): deterministic in a', stochastic in e via P_e.
    Same structure DiscreteDP builds internally for ``res.mc``, but we
    need it standalone here since the policy changes every period.
    """
    from scipy import sparse

    n_states = n_a * n_e
    a_idx_flat = policy_a_index.reshape(n_states)  # a' index per state s
    e_of_s = np.arange(n_states) % n_e

    col_offsets = np.arange(n_e)
    indices = (a_idx_flat[:, None] * n_e + col_offsets[None, :]).ravel()
    data = P_e[e_of_s, :].ravel()
    indptr = np.arange(0, (n_states + 1) * n_e, n_e)
    return sparse.csr_matrix((data, indices, indptr), shape=(n_states, n_states))


def solve_mit_shock(
    p: HouseholdParams,
    steady: HouseholdSolution,
    dist0: np.ndarray,
    r_path: np.ndarray,
    w_path: np.ndarray,
) -> TransitionResult:
    """Solve the household's perfect-foresight transition for a given
    finite path of prices (r_path, w_path), each of length T, under the
    assumption that the economy has returned to the ``steady`` state by
    period T (used as the terminal value function) and that the economy
    starts period 0 at distribution ``dist0`` (an (n_a, n_e) array, usually
    the steady-state stationary distribution).

    ``dist0`` is taken as an argument rather than recomputed here because
    ``distribution.stationary_distribution`` is itself an iterative solve;
    recomputing it on every one of the O(T) calls this function gets from
    ``jacobian.py`` would dominate the total runtime for no reason, since
    it never changes across shock dates.
    """
    T = len(r_path)
    n_a, n_e = p.n_a, p.n_e
    a_grid, e_grid, P_e = steady.a_grid, steady.e_grid, steady.P_e

    v_next = steady.v.reshape(-1)
    policies = [None] * T
    refined = [None] * T
    cohs = [None] * T

    for t in range(T - 1, -1, -1):
        rt = build_reward_transition(p, r_path[t], w_path[t], a_grid, e_grid, P_e)
        policy_a_index, v, a_refined = _bellman_backup(
            rt, v_next, p.beta, n_a, n_e, a_grid
        )
        policies[t] = policy_a_index
        refined[t] = a_refined
        cohs[t] = rt.coh
        v_next = v.reshape(-1)

    dist = np.asarray(dist0).reshape(-1).copy()

    C = np.empty(T)
    K = np.empty(T)
    for t in range(T):
        policy_a_index = policies[t]
        policy_a = a_grid[policy_a_index]
        # consumption uses the quadratic-refined (continuous) choice to
        # avoid the discrete-grid artifact described in
        # ``_quadratic_refine_a``; the distribution still transitions on
        # the exact discrete grid, so K (reported for diagnostics only)
        # uses the discrete choice to stay consistent with that transition.
        policy_c = cohs[t] - refined[t]
        dist_2d = dist.reshape(n_a, n_e)

        C[t] = float(np.sum(dist_2d * policy_c))
        K[t] = float(np.sum(dist_2d * policy_a))

        P_t = _induced_row_transition(policy_a_index, P_e, n_a, n_e)
        dist = np.asarray(dist @ P_t).ravel()

    return TransitionResult(C=C, K=K)
