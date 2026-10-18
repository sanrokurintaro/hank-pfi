import numpy as np

from hank_pfi.distribution import stationary_distribution
from hank_pfi.household import solve_household
from hank_pfi.params import HouseholdParams


def test_stationary_distribution_sums_to_one_and_is_a_fixed_point():
    p = HouseholdParams(n_a=60, n_e=5)
    sol = solve_household(p, r=0.0, w=1.0)
    dist = stationary_distribution(sol)

    assert np.isclose(dist.sum(), 1.0, atol=1e-8)
    assert np.all(dist >= -1e-12)

    n_a, n_e = dist.shape
    P = sol.ddp_result.mc.P
    dist_next = np.asarray(dist.reshape(-1) @ P).ravel().reshape(n_a, n_e)
    assert np.allclose(dist, dist_next, atol=1e-6)
