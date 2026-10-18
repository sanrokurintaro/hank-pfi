"""EGM (Endogenous Grid Method) household solver -- an alternative to
``household.py``'s DiscreteDP/PFI solver, built to test whether avoiding
discrete choice fixes the Jacobian artifacts documented in
``jacobian.py`` / ``transition.py`` (see README "このマイルストーンで
分かったこと" on the ``master`` branch, and the git history of this
``egm-household`` branch for the motivating discussion).

Standard Carroll (2006, *Economics Letters*) EGM for the income
fluctuation problem, applied here to the Aiyagari household (CRRA
utility, CRRA = constant relative risk aversion, Euler equation only).

Unlike ``household.py``, which searches over a *discrete* set of a'
candidates and always lands exactly on a grid point, EGM works backward
from a grid of a' values via the Euler equation and interpolates the
resulting consumption function back onto the original asset grid. The
policy this produces is piecewise-linear and *continuous* in the state,
not a step function -- that continuity is the entire point of this
branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grids import asset_grid, income_process
from .params import HouseholdParams


def _marginal_utility(c: np.ndarray, gamma: float) -> np.ndarray:
    return c ** (-gamma)


def _inverse_marginal_utility(mu: np.ndarray, gamma: float) -> np.ndarray:
    return mu ** (-1.0 / gamma)


@dataclass
class EGMSolution:
    a_grid: np.ndarray       # (n_a,) fixed exogenous grid (same as household.py's)
    e_grid: np.ndarray       # (n_e,)
    P_e: np.ndarray          # (n_e, n_e)
    policy_c: np.ndarray     # (n_a, n_e) consumption, continuous (not grid-snapped)
    policy_a: np.ndarray     # (n_a, n_e) = coh - policy_c, continuous
    n_iter: int
    converged: bool


def egm_backward_step(
    p: HouseholdParams,
    r: float,
    w: float,
    a_grid: np.ndarray,
    e_grid: np.ndarray,
    P_e: np.ndarray,
    c_next: np.ndarray,
    r_next: float | None = None,
) -> np.ndarray:
    """One EGM backward step: given next period's consumption policy
    ``c_next`` evaluated at ``a_grid`` (i.e. c'(a', e') for a' on the
    fixed grid), return this period's consumption policy, also
    re-interpolated onto the fixed ``a_grid`` so the representation is
    stable across iterations and periods.

    Two *different* interest rates matter here, and collapsing them into
    one is a timing bug that only shows up once prices start varying over
    time (it's invisible in the stationary problem, where they're equal
    by construction -- confirmed the hard way while building
    ``transition_egm.py``'s Jacobian, where it showed up as the "income
    effect" and "substitution effect" rows of J_r having the same sign
    and magnitude instead of opposite signs as expected):

    - ``r`` is the return *realized this period* on wealth carried in
      from last period, entering the budget constraint
      coh_t = (1+r)*a_t + w_t*e_t -- this is what prices the endogenous
      grid (the a_endog formula below).
    - ``r_next`` is the return *on savings made this period*, realized
      next period, which is what actually prices the Euler equation
      (beta*(1+r_next)*E[u'(c_{t+1})]) -- i.e. the rate relevant to the
      the t -> t+1 saving decision, the NK-style "r_t" discussed for
      ``jacobian.py`` on the master branch. Defaults to ``r`` for the
      stationary problem, where the two coincide.
    """
    if r_next is None:
        r_next = r
    n_a, n_e = len(a_grid), len(e_grid)

    # E[u'(c_{t+1})|e] for each candidate a' on the grid: treat a_grid
    # itself as the set of end-of-period asset choices a'.
    mu_next = _marginal_utility(c_next, p.gamma)         # (n_a, n_e'), indexed by a'
    emu = mu_next @ P_e.T                                 # (n_a, n_e), E[.|e]

    rhs = p.beta * (1.0 + r_next) * emu
    c_today = _inverse_marginal_utility(rhs, p.gamma)     # consumption today implied by Euler eq.

    # Budget constraint c + a' = (1+r)a + w*e  =>  a = (c + a' - w*e)/(1+r)
    # This is the "endogenous grid": for each candidate a' (= a_grid[i]) and
    # e, the current wealth a that makes a' optimal.
    a_endog = (c_today + a_grid[:, None] - w * e_grid[None, :]) / (1.0 + r)

    policy_c = np.empty((n_a, n_e))
    for j in range(n_e):
        order = np.argsort(a_endog[:, j])
        ae = a_endog[order, j]
        ce = c_today[order, j]

        c_interior = np.interp(a_grid, ae, ce)

        # Below ae[0] (the smallest wealth level at which the Euler
        # equation's *unconstrained* solution is valid), the borrowing
        # constraint a' = a_min binds and the household simply consumes
        # all its cash on hand down to a_min. np.interp would otherwise
        # hold ce[0] constant there, which is wrong -- the constrained
        # consumption function is c = coh(a) - a_min, not a flat line.
        # (Earlier version of this function spliced a synthetic
        # (a_min, coh(a_min, e) - a_min) point into the *same* array
        # before interpolating; when the constraint doesn't actually bind
        # at a_min for this e (ae[0] can be negative, i.e. the
        # unconstrained solution already applies at a = 0), that splice
        # point landed in the middle of an otherwise-smooth, increasing
        # segment and corrupted the interpolation right around a = 0 --
        # confirmed by inspecting the sorted (ae, ce) arrays directly.)
        below = a_grid <= ae[0]
        if np.any(below):
            coh_below = (1.0 + r) * a_grid[below] + w * e_grid[j]
            c_interior[below] = coh_below - p.a_min

        # np.interp holds the boundary value constant beyond the known
        # range; for a above the largest endogenous point this
        # underestimates consumption (true CRRA consumption functions
        # keep rising). Patch the top of the grid with linear
        # extrapolation from the last two endogenous points instead.
        above = a_grid > ae[-1]
        if np.any(above) and len(ae) >= 2:
            slope = (ce[-1] - ce[-2]) / (ae[-1] - ae[-2])
            c_interior[above] = ce[-1] + slope * (a_grid[above] - ae[-1])

        policy_c[:, j] = c_interior

    coh = (1.0 + r) * a_grid[:, None] + w * e_grid[None, :]
    return np.clip(policy_c, 1e-10, coh - p.a_min - 1e-12)


def solve_household_egm(
    p: HouseholdParams,
    r: float,
    w: float,
    tol: float = 1e-9,
    max_iter: int = 2000,
) -> EGMSolution:
    """Solve the stationary household problem via EGM time iteration:
    repeatedly apply the Euler-equation backward step at *fixed* prices
    (r, w) until the consumption policy stops changing. This is the EGM
    analogue of ``household.solve_household``'s Howard-improved policy
    iteration -- a different fixed-point algorithm for the same
    stationary infinite-horizon problem.
    """
    a_grid = asset_grid(p)
    e_grid, P_e = income_process(p)
    n_a, n_e = p.n_a, p.n_e

    coh = (1.0 + r) * a_grid[:, None] + w * e_grid[None, :]
    c = coh.copy()  # initial guess: consume everything

    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        c_new = egm_backward_step(p, r, w, a_grid, e_grid, P_e, c)
        diff = np.max(np.abs(c_new - c))
        c = c_new
        if diff < tol:
            converged = True
            break

    policy_c = c
    policy_a = coh - policy_c
    policy_a = np.clip(policy_a, p.a_min, p.a_max)

    return EGMSolution(
        a_grid=a_grid, e_grid=e_grid, P_e=P_e,
        policy_c=policy_c, policy_a=policy_a,
        n_iter=n_iter, converged=converged,
    )
