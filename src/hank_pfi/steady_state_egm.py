"""Layer (5), EGM version: the Aiyagari general equilibrium, built on
``household_egm.py`` / ``distribution_egm.py`` instead of the
DiscreteDP/PFI household block. Mirrors ``steady_state.py`` exactly --
same firm closure, same bisection on r -- only the household layer
underneath differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .aggregate import Aggregates, aggregate
from .distribution_egm import stationary_distribution_egm
from .household_egm import EGMSolution, solve_household_egm
from .params import FirmParams, HouseholdParams
from .steady_state import capital_labor_ratio, wage

_L_SUPPLY = 1.0


@dataclass
class SteadyStateEGM:
    r: float
    w: float
    household: EGMSolution
    distribution: np.ndarray
    aggregates: Aggregates
    K_demand: float


def excess_capital_egm(r: float, hh_p: HouseholdParams, firm: FirmParams) -> float:
    w = wage(r, firm)
    sol = solve_household_egm(hh_p, r, w)
    dist = stationary_distribution_egm(sol)
    agg = aggregate(sol, dist)
    k_demand = capital_labor_ratio(r, firm) * _L_SUPPLY
    return agg.K - k_demand


def solve_steady_state_egm(
    hh_p: HouseholdParams,
    firm: FirmParams,
    r_lo: float | None = None,
    r_hi: float | None = None,
    xtol: float = 1e-6,
) -> SteadyStateEGM:
    if r_lo is None:
        r_lo = -firm.delta + 1e-4
    if r_hi is None:
        r_hi = 1.0 / hh_p.beta - 1.0 - 1e-4

    r_star = brentq(excess_capital_egm, r_lo, r_hi, args=(hh_p, firm), xtol=xtol)

    w_star = wage(r_star, firm)
    sol = solve_household_egm(hh_p, r_star, w_star)
    dist = stationary_distribution_egm(sol)
    agg = aggregate(sol, dist)
    k_demand = capital_labor_ratio(r_star, firm) * _L_SUPPLY

    return SteadyStateEGM(
        r=r_star, w=w_star, household=sol, distribution=dist,
        aggregates=agg, K_demand=k_demand,
    )
