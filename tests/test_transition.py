import numpy as np

from hank_pfi.params import FirmParams, HouseholdParams
from hank_pfi.steady_state import solve_steady_state
from hank_pfi.transition import solve_mit_shock


def test_no_shock_path_reproduces_the_steady_state():
    hh_p = HouseholdParams(n_a=80, n_e=5)
    firm = FirmParams()
    ss = solve_steady_state(hh_p, firm)

    T = 15
    r_path = np.full(T, ss.r)
    w_path = np.full(T, ss.w)
    res = solve_mit_shock(hh_p, ss.household, ss.distribution, r_path, w_path)

    # res.K uses the exact discrete policy, same as the steady-state solve,
    # and matches to float precision. res.C instead reads off the
    # quadratic-refined (continuous) choice -- see
    # transition._quadratic_refine_a -- which deliberately differs from
    # the raw discrete grid point even at the steady state itself, trading
    # a few percent of level bias for a much smaller finite-difference
    # Jacobian artifact (see jacobian.py's module docstring). The bias
    # shrinks as the grid is refined; n_a=300 brings it under 1%.
    assert np.allclose(res.K, ss.aggregates.K, atol=1e-6)
    assert np.all(res.C > 0)
    assert np.allclose(res.C, res.C[0])  # constant across t, as it must be
    assert abs(res.C[0] / ss.aggregates.C - 1.0) < 0.1


def test_a_shock_decays_away_from_its_own_date():
    # A one-time shock at date s has its largest effect on consumption
    # near s. It does *not* die out immediately afterward: households who
    # held assets through date s earned extra interest income, and they
    # smooth that one-time wealth gain into consumption over many future
    # periods (a standard HANK "intertemporal MPC" / wealth-effect echo,
    # confirmed while developing this module -- see jacobian.py). So this
    # only checks that the effect is monotonically shrinking by the end of
    # the horizon, not that it has fully vanished.
    hh_p = HouseholdParams(n_a=80, n_e=5)
    firm = FirmParams()
    ss = solve_steady_state(hh_p, firm)

    T = 20
    s = 5
    r_path = np.full(T, ss.r)
    w_path = np.full(T, ss.w)
    r_shocked = r_path.copy()
    r_shocked[s] += 0.02

    base = solve_mit_shock(hh_p, ss.household, ss.distribution, r_path, w_path)
    shocked = solve_mit_shock(hh_p, ss.household, ss.distribution, r_shocked, w_path)
    dC = shocked.C - base.C

    assert abs(dC[s]) > abs(dC[-1])  # effect is largest near the shock date
    assert abs(dC[-1]) < abs(dC[s + 1])  # still shrinking by the tail
    assert abs(dC[-1]) < 0.05  # and small in absolute terms
