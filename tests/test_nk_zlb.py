import numpy as np

from hank_pfi.nk_zlb import natural_rate_shock, solve_transition_path
from hank_pfi.params import NKParams


def test_no_shock_path_is_the_steady_state_everywhere():
    p = NKParams()
    path = solve_transition_path(np.zeros(50), p)
    assert np.allclose(path.y, 0.0, atol=1e-10)
    assert np.allclose(path.pi, 0.0, atol=1e-10)
    assert np.allclose(path.i, p.r_star, atol=1e-10)
    assert not path.regime_binds.any()
    assert path.converged


def test_large_negative_demand_shock_triggers_a_transitory_zlb_episode():
    p = NKParams()
    shock = natural_rate_shock(T=100, size=-0.02, duration=4, rho=0.7)
    path = solve_transition_path(shock, p)

    assert path.converged
    assert path.regime_binds[0]              # binds on impact
    assert not path.regime_binds[-1]          # has released well before the horizon ends
    assert np.all(path.i >= -1e-10)           # ZLB respected everywhere
    assert np.all(path.i[path.regime_binds] == 0.0)
    assert path.y[0] < 0                      # output gap opens up on impact
    # returns to steady state well within the horizon
    assert abs(path.y[-1]) < 1e-6
    assert abs(path.pi[-1]) < 1e-6


def test_natural_rate_shock_shape():
    shock = natural_rate_shock(T=20, size=-0.03, duration=3, rho=0.5)
    assert np.allclose(shock[:3], -0.03)
    assert shock[3] == -0.03 * 0.5
    assert abs(shock[-1]) < abs(shock[3])  # decays toward 0 after the plateau
