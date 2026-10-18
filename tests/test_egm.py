"""Tests for the EGM (Endogenous Grid Method) branch of the household
block: ``household_egm.py``, ``distribution_egm.py``,
``steady_state_egm.py``, ``transition_egm.py``, ``jacobian_egm.py``, and
their hookup into ``hank_zlb.py``.

The headline comparison against the DiscreteDP/PFI branch (``master``):
EGM's policy function is continuous, so finite-difference Jacobians built
from it should not need the large, noise-averaging perturbation sizes
(dr=0.02, dlogw=0.1) the DiscreteDP branch needed -- see
``test_jacobian_egm_handles_tiny_perturbations_cleanly`` below, which is
exactly the scenario that gave wrong-signed, noisy results on ``master``.
"""

import numpy as np

from hank_pfi.hank_zlb import build_hank_jacobian_egm, solve_hank_transition_path
from hank_pfi.household_egm import solve_household_egm
from hank_pfi.distribution_egm import stationary_distribution_egm
from hank_pfi.aggregate import aggregate
from hank_pfi.jacobian_egm import household_jacobians_egm
from hank_pfi.nk_zlb import natural_rate_shock
from hank_pfi.params import FirmParams, HouseholdParams, NKParams
from hank_pfi.steady_state_egm import solve_steady_state_egm
from hank_pfi.transition_egm import solve_mit_shock_egm


def test_household_egm_policy_is_budget_feasible():
    p = HouseholdParams(n_a=100, n_e=5)
    sol = solve_household_egm(p, r=0.01, w=1.0)
    assert sol.converged
    assert np.all(sol.policy_c > 0)
    assert np.all(sol.policy_a >= p.a_min - 1e-8)
    assert np.all(sol.policy_a <= p.a_max + 1e-8)


def test_household_egm_roughly_matches_discretedp():
    # Different numerical methods for the same economic problem should
    # give similar (not identical -- EGM is continuous, DiscreteDP snaps
    # to grid points) policy functions.
    from hank_pfi.household import solve_household

    p = HouseholdParams(n_a=150, n_e=5)
    r, w = 0.012, 1.35
    sol_egm = solve_household_egm(p, r, w)
    sol_dp = solve_household(p, r, w)
    # compare at a handful of interior states; tolerance reflects
    # DiscreteDP's grid-snapping error, not EGM's accuracy
    diff = np.abs(sol_egm.policy_a - sol_dp.policy_a)
    assert np.median(diff) < 1.0


def test_distribution_egm_sums_to_one_and_is_a_fixed_point():
    p = HouseholdParams(n_a=100, n_e=5)
    sol = solve_household_egm(p, r=0.0, w=1.0)
    dist = stationary_distribution_egm(sol)
    assert np.isclose(dist.sum(), 1.0, atol=1e-8)
    assert np.all(dist >= -1e-12)


def test_steady_state_egm_clears_the_capital_market():
    hh_p = HouseholdParams(n_a=120, n_e=5)
    firm = FirmParams()
    ss = solve_steady_state_egm(hh_p, firm)
    assert -firm.delta < ss.r < 1.0 / hh_p.beta - 1.0
    assert np.isclose(ss.aggregates.K, ss.K_demand, rtol=0.02)


def test_transition_egm_no_shock_reproduces_steady_state():
    hh_p = HouseholdParams(n_a=100, n_e=5)
    firm = FirmParams()
    ss = solve_steady_state_egm(hh_p, firm)
    T = 15
    r_path = np.full(T, ss.r)
    w_path = np.full(T, ss.w)
    res = solve_mit_shock_egm(hh_p, ss.household, ss.distribution, r_path, w_path, ss.r)
    # unlike the DiscreteDP+quadratic-refinement branch, EGM has no
    # systematic level bias here: the no-shock path should match the
    # steady state to numerical precision, not just "within a few percent"
    assert np.allclose(res.C, ss.aggregates.C, atol=1e-6)


def test_jacobian_egm_handles_tiny_perturbations_cleanly():
    # This is the key test: dr=dlogw=1e-4 is the exact perturbation size
    # that gave wrong-signed, noisy Jacobians on the DiscreteDP branch
    # (see master's jacobian.py docstring and README). EGM should handle
    # it smoothly since its policy function has no discrete-grid snapping.
    hh_p = HouseholdParams(n_a=120, n_e=5)
    firm = FirmParams()
    ss = solve_steady_state_egm(hh_p, firm)
    T = 10
    J = household_jacobians_egm(
        hh_p, ss.household, ss.distribution, ss.r, ss.w, ss.aggregates.C, T,
        dr=1e-4, dlogw=1e-4,
    )
    r_diag = np.diag(J.J_r)
    r_super_diag = np.diag(J.J_r, k=1)
    w_diag = np.diag(J.J_w)

    # income effect (own-period): raising r_t raises income this period
    assert np.all(r_diag > 0)
    # substitution effect (one period ahead): a higher return on saving
    # made this period, realized next period, lowers this period's
    # consumption
    assert np.all(r_super_diag < 0)
    # positive same-period MPC out of labor income
    assert np.all(w_diag > 0)
    # smoothly varying, not noisy -- consecutive entries shouldn't jump by
    # more than a modest fraction of their own scale
    assert np.max(np.abs(np.diff(r_diag))) < 0.2 * np.max(np.abs(r_diag))


def test_hank_zlb_with_egm_jacobian_matches_no_shock_steady_state():
    hh_p = HouseholdParams(n_a=120, n_e=5)
    firm = FirmParams()
    ss = solve_steady_state_egm(hh_p, firm)
    T = 15
    J_eff = build_hank_jacobian_egm(
        hh_p, ss.household, ss.distribution, ss.r, ss.w, ss.aggregates.C, T
    )
    nk_p = NKParams()
    path0 = solve_hank_transition_path(np.zeros(T), J_eff, nk_p)
    assert path0.converged
    assert np.allclose(path0.y, 0.0, atol=1e-8)
    assert not path0.regime_binds.any()


def test_hank_zlb_with_egm_jacobian_handles_a_demand_shock():
    hh_p = HouseholdParams(n_a=120, n_e=5)
    firm = FirmParams()
    ss = solve_steady_state_egm(hh_p, firm)
    T = 20
    J_eff = build_hank_jacobian_egm(
        hh_p, ss.household, ss.distribution, ss.r, ss.w, ss.aggregates.C, T
    )
    nk_p = NKParams()
    shock = natural_rate_shock(T=T, size=-0.02, duration=4, rho=0.7)
    path = solve_hank_transition_path(shock, J_eff, nk_p)
    assert path.converged
    assert np.all(path.i >= -1e-9)
    assert path.y[0] < 0  # output gap opens up on impact
