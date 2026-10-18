"""Layer (5): the Aiyagari general equilibrium (steady state).

Closes the model with a representative Cobb-Douglas firm and finds the
interest rate r* that clears the capital market, i.e. aggregate household
asset holdings (layer 4, built on the PFI household solver of layer 2)
equal the firm's capital demand. Solved by bisection (``scipy.optimize
.brentq``) on the excess-demand function -- no custom root finder needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .aggregate import Aggregates, aggregate
from .distribution import stationary_distribution
from .household import HouseholdSolution, solve_household
from .params import FirmParams, HouseholdParams

_L_SUPPLY = 1.0  # aggregate effective labor is normalized to 1 by the
                  # income-grid normalization in grids.income_process


def capital_labor_ratio(r: float, firm: FirmParams) -> float:
    """Firm FOC for r: r + delta = alpha * tfp * (K/L)^(alpha-1)."""
    return ((r + firm.delta) / (firm.alpha * firm.tfp)) ** (1.0 / (firm.alpha - 1.0))


def wage(r: float, firm: FirmParams) -> float:
    """Firm FOC for w, given the capital-labor ratio implied by r."""
    k_l = capital_labor_ratio(r, firm)
    return (1.0 - firm.alpha) * firm.tfp * k_l ** firm.alpha


@dataclass
class SteadyState:
    r: float
    w: float
    household: HouseholdSolution
    distribution: np.ndarray
    aggregates: Aggregates
    K_demand: float


def excess_capital(r: float, hh_p: HouseholdParams, firm: FirmParams) -> float:
    """K_supply(r) - K_demand(r). Root of this is the Aiyagari equilibrium."""
    w = wage(r, firm)
    sol = solve_household(hh_p, r, w)
    dist = stationary_distribution(sol)
    agg = aggregate(sol, dist)
    k_demand = capital_labor_ratio(r, firm) * _L_SUPPLY
    return agg.K - k_demand


def solve_steady_state(
    hh_p: HouseholdParams,
    firm: FirmParams,
    r_lo: float | None = None,
    r_hi: float | None = None,
    xtol: float = 1e-6,
) -> SteadyState:
    """Bisect on r for the Aiyagari stationary equilibrium.

    Bounds default to (-delta + eps, 1/beta - 1 - eps): below -delta the
    firm's capital demand is undefined, and at or above 1/beta - 1
    precautionary-saving households want to accumulate assets without
    bound, so no stationary equilibrium exists in that range.
    """
    if r_lo is None:
        r_lo = -firm.delta + 1e-4
    if r_hi is None:
        r_hi = 1.0 / hh_p.beta - 1.0 - 1e-4

    r_star = brentq(excess_capital, r_lo, r_hi, args=(hh_p, firm), xtol=xtol)

    w_star = wage(r_star, firm)
    sol = solve_household(hh_p, r_star, w_star)
    dist = stationary_distribution(sol)
    agg = aggregate(sol, dist)
    k_demand = capital_labor_ratio(r_star, firm) * _L_SUPPLY

    return SteadyState(
        r=r_star,
        w=w_star,
        household=sol,
        distribution=dist,
        aggregates=agg,
        K_demand=k_demand,
    )
