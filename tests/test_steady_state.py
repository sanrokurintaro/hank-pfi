import numpy as np

from hank_pfi.params import FirmParams, HouseholdParams
from hank_pfi.steady_state import excess_capital, solve_steady_state


def test_excess_capital_changes_sign():
    # Existence of an Aiyagari equilibrium requires excess capital demand
    # to cross zero somewhere in (-delta, 1/beta - 1).
    hh_p = HouseholdParams(n_a=80, n_e=5)
    firm = FirmParams()
    r_lo = -firm.delta + 1e-3
    r_hi = 1.0 / hh_p.beta - 1.0 - 1e-3
    ed_lo = excess_capital(r_lo, hh_p, firm)
    ed_hi = excess_capital(r_hi, hh_p, firm)
    assert ed_lo < 0 < ed_hi


def test_steady_state_clears_the_capital_market():
    hh_p = HouseholdParams(n_a=100, n_e=5)
    firm = FirmParams()
    ss = solve_steady_state(hh_p, firm)

    assert -firm.delta < ss.r < 1.0 / hh_p.beta - 1.0
    # With a discrete asset grid, K_supply(r) is a step function (each
    # household's optimal a' jumps between grid points as r crosses
    # thresholds), so excess_capital(r) is not smooth at the grid scale.
    # brentq's xtol bounds the r-bracket width, not the residual K-gap,
    # and with n_a=100 the local slope of excess_capital is steep enough
    # that a tight r-bracket still leaves a few-percent K-gap. A tighter
    # match requires a finer grid (see examples/, which uses n_a=300+).
    assert np.isclose(ss.aggregates.K, ss.K_demand, rtol=0.02)
    # the precautionary-savings motive implies the Aiyagari equilibrium
    # rate sits below the complete-markets rate 1/beta - 1
    assert ss.r < 1.0 / hh_p.beta - 1.0
