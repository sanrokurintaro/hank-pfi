"""Layer (1): grid construction and income-process discretization.

Reuses QuantEcon's Rouwenhorst discretization instead of reimplementing
Markov-chain approximation of an AR(1) process.
"""

from __future__ import annotations

import warnings

import numpy as np
import quantecon as qe

from .params import HouseholdParams


def asset_grid(p: HouseholdParams) -> np.ndarray:
    """Exponentially spaced asset grid concentrated near the borrowing
    constraint, where the household's policy function is most curved."""
    lin = np.linspace(0.0, 1.0, p.n_a) ** p.grid_curvature
    return p.a_min + (p.a_max - p.a_min) * lin


def income_process(p: HouseholdParams) -> tuple[np.ndarray, np.ndarray]:
    """Discretize a log-AR(1) income process via Rouwenhorst's method.

    Returns
    -------
    e_grid : (n_e,) array of income *levels*, normalized so that the
        stationary-distribution-weighted mean is 1.
    P : (n_e, n_e) Markov transition matrix.
    """
    with warnings.catch_warnings():
        # quantecon 0.13.x unconditionally warns about a pre-0.6.0 signature
        # change even when called with the current (n, rho, sigma, mu) order;
        # we already use the current signature, so this is a known false
        # positive from the installed quantecon version, not our bug.
        warnings.filterwarnings("ignore", message="The API of rouwenhorst")
        mc = qe.markov.rouwenhorst(p.n_e, rho=p.rho_e, sigma=p.sigma_e, mu=0.0)
    e_levels = np.exp(mc.state_values)
    stat = mc.stationary_distributions[0]
    e_grid = e_levels / np.dot(stat, e_levels)
    return e_grid, mc.P
