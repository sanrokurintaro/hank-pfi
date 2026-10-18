"""Layer (6): a 3-equation NK aggregate block with an occasionally
binding ZLB, solved by OccBin-style regime switching (Guerrieri &
Iacoviello, 2015).

This layer is **reduced-form / representative-agent** for now: the IS
curve's semi-elasticity ``sigma`` and the NKPC slope ``kappa`` are
calibrated parameters (see ``params.NKParams``), not yet derived from the
household block in ``household.py``. Wiring the two together -- i.e.
replacing ``sigma`` with the semi-elasticity of aggregate consumption to
the real rate implied by the stationary Aiyagari economy (a
sequence-space-Jacobian-style object) -- is the natural next milestone
and is flagged in the README.

What *is* already the real thing here is the regime-switching algorithm:
exactly the same "guess which periods the constraint binds -> solve the
unconstrained linear system in each regime -> verify -> update the guess"
logic will be needed once the IS curve is replaced by the household
block's aggregate Euler relation, so this module is the template for
layer (6) regardless of which aggregate block eventually feeds it.

Why not just use SSJ/OccBin's own code? Because with a HANK household
block (layer 2) in the loop, the problem stops being a small linear
system in (y, pi) and becomes "re-solve the household PFI problem and
re-aggregate for every candidate regime", which the reduced-form
RANK tools are not set up to do while remaining in this package's
QuantEcon-first dependency footprint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import NKParams


@dataclass
class TransitionPath:
    y: np.ndarray        # (T,) output gap
    pi: np.ndarray        # (T,) inflation
    i: np.ndarray         # (T,) nominal policy rate (>= 0 by construction)
    regime_binds: np.ndarray  # (T,) bool, True where the ZLB binds
    converged: bool
    n_iter: int


def natural_rate_shock(
    T: int, size: float, duration: int, rho: float = 0.6
) -> np.ndarray:
    """A plateau-then-decay natural-rate shock path: rstar_t sits at
    ``size`` (typically negative, a demand shock) for ``duration``
    periods, then decays geometrically at rate ``rho``.

    ``T`` should be set well beyond the point where the shock has decayed
    to ~0 (a few multiples of 1/(1-rho) after ``duration``); the
    transition-path solver imposes "back to steady state" as a terminal
    condition at t=T, and an unrealistically short T biases the dynamics
    near the end of the sample.
    """
    shock = np.zeros(T)
    plateau = min(duration, T)
    shock[:plateau] = size
    for t in range(plateau, T):
        shock[t] = rho * shock[t - 1]
    return shock


def _backward_solve(
    rstar_path: np.ndarray, regime_binds: np.ndarray, p: NKParams
) -> tuple[np.ndarray, np.ndarray]:
    """Given a regime guess (which periods have i_t pinned at 0) and the
    exogenous natural-rate shock path, solve the 2x2 linear system for
    (y_t, pi_t) backward from the terminal condition (y_T, pi_T) = (0, 0),
    i.e. the economy has returned to steady state after T periods.
    """
    T = len(rstar_path)
    y = np.zeros(T + 1)
    pi = np.zeros(T + 1)

    for t in range(T - 1, -1, -1):
        y_next, pi_next = y[t + 1], pi[t + 1]
        if regime_binds[t]:
            # i_t = 0 imposed. Full IS curve: y_t = y_{t+1} - sigma*(i_t - pi_{t+1}
            # - (r_star + rstar_path[t])), i.e. the natural rate is r_star plus
            # the shock, and i_t = 0 removes the policy-rate term entirely:
            #   y_t = y_{t+1} + sigma*pi_{t+1} + sigma*(r_star + rstar_path[t])
            # NKPC (same in both regimes): pi_t = kappa*y_t + beta*pi_{t+1}
            y_t = y_next + p.sigma * pi_next + p.sigma * (p.r_star + rstar_path[t])
            pi_t = p.kappa * y_t + p.beta * pi_next
        else:
            # Taylor rule i_t = r_star + phi_pi*pi_t + phi_y*y_t substituted into
            # the same IS curve. Solve the resulting 2x2 linear system in
            # (y_t, pi_t):
            #   (1 + sigma*phi_y) y_t + sigma*phi_pi pi_t = y_{t+1} + sigma*pi_{t+1}
            #                                                + sigma*rstar_path[t]
            #   -kappa*y_t + pi_t = beta*pi_{t+1}
            # (the r_star terms from the Taylor rule and the natural rate cancel,
            # leaving only the shock rstar_path[t] as forcing -- this is the
            # usual representation of policy relative to its own steady state.)
            A = np.array(
                [[1.0 + p.sigma * p.phi_y, p.sigma * p.phi_pi], [-p.kappa, 1.0]]
            )
            b = np.array(
                [
                    y_next + p.sigma * pi_next + p.sigma * rstar_path[t],
                    p.beta * pi_next,
                ]
            )
            y_t, pi_t = np.linalg.solve(A, b)

        y[t], pi[t] = y_t, pi_t

    return y[:T], pi[:T]


def solve_transition_path(
    rstar_path: np.ndarray,
    p: NKParams,
    max_iter: int = 200,
) -> TransitionPath:
    """OccBin-style guess-and-verify solver for the ZLB-constrained
    perfect-foresight transition path.

    1. Guess a vector of which periods the ZLB binds (initially: none).
    2. Solve the linear system implied by that regime guess.
    3. Compute the Taylor-rule-implied rate in every period; a period
       should bind iff that rate would be negative.
    4. If the implied regime matches the guess, stop; otherwise update
       the guess and repeat.
    """
    T = len(rstar_path)
    regime_binds = np.zeros(T, dtype=bool)

    for n_iter in range(1, max_iter + 1):
        y, pi = _backward_solve(rstar_path, regime_binds, p)
        i_unconstrained = p.r_star + p.phi_pi * pi + p.phi_y * y
        new_regime = i_unconstrained < 0.0

        if np.array_equal(new_regime, regime_binds):
            i = np.where(regime_binds, 0.0, i_unconstrained)
            return TransitionPath(
                y=y, pi=pi, i=i, regime_binds=regime_binds,
                converged=True, n_iter=n_iter,
            )
        regime_binds = new_regime

    y, pi = _backward_solve(rstar_path, regime_binds, p)
    i_unconstrained = p.r_star + p.phi_pi * pi + p.phi_y * y
    i = np.where(regime_binds, 0.0, i_unconstrained)
    return TransitionPath(
        y=y, pi=pi, i=i, regime_binds=regime_binds,
        converged=False, n_iter=max_iter,
    )
