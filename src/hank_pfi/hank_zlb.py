"""The genuine HANK replacement for ``nk_zlb.py``'s representative-agent IS
curve: aggregate demand is now driven by the household block's own
sequence-space Jacobians (``jacobian.py``) instead of a single exogenous
``sigma`` parameter, while the ZLB is still solved by the same OccBin-style
regime-switching guess-and-verify loop as before.

Timing convention note (important, and non-obvious -- discovered while
validating this module): in ``household.py``/``transition.py``, ``r_path[t]``
is the return realized *at* t on assets already held entering t (a's choice
is made at t-1). So ``jacobian.J_r[t, s]`` decomposes into a *positive*
diagonal (pure income effect: more interest income this period raises this
period's consumption) and a *negative* super-diagonal ``J_r[t, t+1]`` (the
standard substitution effect: a higher return on saving made *this* period,
realized next period, lowers this period's consumption). The textbook NK IS
curve's semi-elasticity corresponds to the *latter*, since its ``r_t`` means
"the rate relevant to the t -> t+1 saving decision" (equivalently, the
Fisher equation ``r_t = i_t - pi_{t+1}``). So the Jacobian this module uses
is ``J_r[:, 1:]`` shifted left by one column relative to the raw household
Jacobian -- see ``_shift_jacobian``.

Closure: the household Jacobians alone only say how consumption responds
to a *given* path of (r, w). To close the model we need wages to respond to
aggregate activity. We use the standard linear-technology closure
``log w_t - log w_ss = y_t`` (wages move one-for-one with the output gap),
which turns ``y = J_r_shift @ r_dev + J_w @ y`` into a fixed point solved by
a single matrix inversion -- see ``_effective_jacobian``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .household import HouseholdSolution
from .jacobian import HouseholdJacobians, household_jacobians
from .params import HouseholdParams, NKParams


def _shift_jacobian(J_r_raw: np.ndarray) -> np.ndarray:
    """J_r_raw has shape (T_raw, T_raw) with T_raw = T + 1 periods of
    shocks. Returns the (T, T) NK-convention Jacobian J_r_shift[t, s] =
    J_r_raw[t, s+1] (see module docstring): d log C_t / d (rate relevant
    to the s -> s+1 saving decision).
    """
    T = J_r_raw.shape[0] - 1
    return J_r_raw[:T, 1 : T + 1]


def _effective_jacobian(J: HouseholdJacobians) -> np.ndarray:
    """Combine the r- and w-Jacobians under the w_t = Y_t closure into a
    single (T, T) matrix J_eff with y = J_eff @ r_dev.
    """
    J_r_shift = _shift_jacobian(J.J_r)
    T = J_r_shift.shape[0]
    J_w = J.J_w[:T, :T]
    return np.linalg.solve(np.eye(T) - J_w, J_r_shift)


def _nkpc_forward_matrix(T: int, beta: float) -> np.ndarray:
    """L[t, s] = beta**(s - t) for s >= t, else 0, so that pi = kappa * L @ y
    solves pi_t = kappa*y_t + beta*pi_{t+1} with terminal pi_T = y_T = 0.
    """
    t_idx = np.arange(T)[:, None]
    s_idx = np.arange(T)[None, :]
    power = s_idx - t_idx
    L = np.where(power >= 0, beta ** np.maximum(power, 0), 0.0)
    return L


def _shift_down_matrix(T: int) -> np.ndarray:
    """ShiftDown @ pi gives pi_{t+1} (0 at the terminal t = T-1, matching
    the pi_T = 0 boundary condition used throughout)."""
    S = np.zeros((T, T))
    for t in range(T - 1):
        S[t, t + 1] = 1.0
    return S


@dataclass
class HankTransitionPath:
    y: np.ndarray
    pi: np.ndarray
    i: np.ndarray
    regime_binds: np.ndarray
    converged: bool
    n_iter: int
    J_eff: np.ndarray  # the combined (r, w-closure) Jacobian actually used


def build_hank_jacobian(
    hh_p: HouseholdParams,
    steady_household: HouseholdSolution,
    dist0: np.ndarray,
    r_ss: float,
    w_ss: float,
    C_ss: float,
    T: int,
    dr: float = 0.02,
    dlogw: float = 0.1,
) -> np.ndarray:
    """Compute the household block's Jacobians (needs T+1 periods of raw
    shocks to produce a T-period NK-convention-shifted Jacobian, see
    ``_shift_jacobian``) and combine them into J_eff via the w=Y closure.
    """
    J = household_jacobians(
        hh_p, steady_household, dist0, r_ss, w_ss, C_ss, T + 1, dr=dr, dlogw=dlogw
    )
    return _effective_jacobian(J)


def _solve_given_regime(
    regime_binds: np.ndarray,
    e_shock_path: np.ndarray,
    J_eff: np.ndarray,
    L: np.ndarray,
    S: np.ndarray,
    I: np.ndarray,
    r_star_vec: np.ndarray,
    p: NKParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    D = np.diag((~regime_binds).astype(float))  # 1 where unconstrained
    M = J_eff @ (D @ (p.phi_pi * p.kappa * L + p.phi_y * I) - S @ (p.kappa * L))
    rhs = J_eff @ (D - I) @ r_star_vec - J_eff @ e_shock_path
    y = np.linalg.solve(I - M, rhs)
    pi = p.kappa * (L @ y)
    i_unconstrained = p.r_star + p.phi_pi * pi + p.phi_y * y
    return y, pi, i_unconstrained


def solve_hank_transition_path(
    e_shock_path: np.ndarray,
    J_eff: np.ndarray,
    p: NKParams,
    max_iter: int = 200,
) -> HankTransitionPath:
    """Solve for the ZLB-constrained transition path given the household
    block's J_eff in place of the representative-agent IS curve's sigma.

    For a *given* regime guess (which periods the ZLB binds), the whole
    system is linear in y: substituting the NKPC and the (regime-dependent)
    Taylor rule into the HANK IS curve ``y = J_eff @ r_dev`` yields one
    (T, T) linear system, solved exactly via ``numpy.linalg.solve`` -- no
    period-by-period backward recursion is possible here (unlike the
    representative-agent ``nk_zlb.py``) because y_t now depends on the
    *entire* r_dev path through J_eff, not just next period's (y, pi).

    Finding the *right* regime: naive OccBin-style fixed-point iteration
    (guess regime -> solve -> recompute regime -> repeat) is what
    ``nk_zlb.solve_transition_path`` uses, and it works there because the
    representative-agent IS curve is a clean, smooth 2-equation system.
    With a real household-implied J_eff (noisier, shaped by whatever the
    Jacobian's finite-difference/discretization error looks like) that
    iteration can fail to settle -- confirmed empirically while building
    this module. Since a one-time negative demand shock is expected to push
    the ZLB to bind over a *single contiguous* window [0, k) and then
    release for good (never rebinding once the economy recovers), we search
    directly over candidate window lengths k = 0..T and accept the first
    one that is self-consistent (Taylor-rule-implied rate < 0 throughout
    the window, >= 0 after it). This is both more robust and, for T
    candidates, no more expensive than a handful of fixed-point iterations.
    The naive iteration is kept as a fallback for the (untested, but
    possible) case of a non-contiguous bind set.
    """
    T = len(e_shock_path)
    L = _nkpc_forward_matrix(T, p.beta)
    S = _shift_down_matrix(T)
    I = np.eye(T)
    r_star_vec = np.full(T, p.r_star)

    # Try windows anchored at t=0 first (the generic case for a one-time
    # demand shock), then fall back to scanning *every* contiguous window
    # [k1, k2) -- a real household Jacobian's hump-shaped intertemporal-MPC
    # dynamics (see jacobian.py's module docstring and the single-shock
    # diagnostics in the package's development notes) can make the output
    # gap and inflation move non-monotonically enough that the ZLB episode
    # isn't anchored at t=0. O(T^2) candidate windows is still cheap: each
    # is one (T, T) linear solve.
    n_tries = 0
    candidates = [(0, k) for k in range(T + 1)]
    candidates += [
        (k1, k2)
        for k1 in range(1, T)
        for k2 in range(k1 + 1, T + 1)
    ]
    for k1, k2 in candidates:
        n_tries += 1
        regime_binds = np.zeros(T, dtype=bool)
        regime_binds[k1:k2] = True
        y, pi, i_unconstrained = _solve_given_regime(
            regime_binds, e_shock_path, J_eff, L, S, I, r_star_vec, p
        )
        ok_bind = bool(np.all(i_unconstrained[k1:k2] < 0.0)) if k2 > k1 else True
        ok_free = bool(
            np.all(i_unconstrained[:k1] >= 0.0) and np.all(i_unconstrained[k2:] >= 0.0)
        )
        if ok_bind and ok_free:
            i = np.where(regime_binds, 0.0, i_unconstrained)
            return HankTransitionPath(
                y=y, pi=pi, i=i, regime_binds=regime_binds,
                converged=True, n_iter=n_tries, J_eff=J_eff,
            )

    regime_binds = np.zeros(T, dtype=bool)
    for n_iter in range(1, max_iter + 1):
        y, pi, i_unconstrained = _solve_given_regime(
            regime_binds, e_shock_path, J_eff, L, S, I, r_star_vec, p
        )
        new_regime = i_unconstrained < 0.0
        if np.array_equal(new_regime, regime_binds):
            i = np.where(regime_binds, 0.0, i_unconstrained)
            return HankTransitionPath(
                y=y, pi=pi, i=i, regime_binds=regime_binds,
                converged=True, n_iter=n_iter, J_eff=J_eff,
            )
        regime_binds = new_regime

    y, pi, i_unconstrained = _solve_given_regime(
        regime_binds, e_shock_path, J_eff, L, S, I, r_star_vec, p
    )
    i = np.where(regime_binds, 0.0, i_unconstrained)
    return HankTransitionPath(
        y=y, pi=pi, i=i, regime_binds=regime_binds,
        converged=False, n_iter=max_iter, J_eff=J_eff,
    )
