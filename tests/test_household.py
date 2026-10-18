import numpy as np

from hank_pfi.grids import asset_grid, income_process
from hank_pfi.household import solve_household
from hank_pfi.params import HouseholdParams


def test_asset_grid_monotone_and_bounded():
    p = HouseholdParams(n_a=50)
    a = asset_grid(p)
    assert a[0] == p.a_min
    assert a[-1] == p.a_max
    assert np.all(np.diff(a) > 0)


def test_income_process_normalized_to_mean_one():
    p = HouseholdParams(n_e=7)
    e_grid, P = income_process(p)
    import quantecon as qe

    mc = qe.markov.MarkovChain(P)
    stat = mc.stationary_distributions[0]
    assert np.isclose(np.dot(stat, e_grid), 1.0)
    assert np.allclose(P.sum(axis=1), 1.0)


def test_household_policy_is_budget_feasible():
    p = HouseholdParams(n_a=60, n_e=5)
    sol = solve_household(p, r=0.01, w=1.0)
    assert np.all(sol.policy_c > 0)
    assert np.all(sol.policy_a >= p.a_min - 1e-8)
    assert np.all(sol.policy_a <= p.a_max + 1e-8)


def test_higher_r_weakly_raises_savings_policy():
    # Standard Aiyagari comparative static: at a higher interest rate,
    # households save (weakly) more at every (a, e).
    p = HouseholdParams(n_a=60, n_e=5)
    sol_lo = solve_household(p, r=0.0, w=1.0)
    sol_hi = solve_household(p, r=0.02, w=1.0)
    assert np.all(sol_hi.policy_a >= sol_lo.policy_a - 1e-8)
