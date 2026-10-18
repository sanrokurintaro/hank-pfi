import numpy as np

from hank_pfi.hank_zlb import solve_hank_transition_path
from hank_pfi.nk_zlb import natural_rate_shock, solve_transition_path
from hank_pfi.params import NKParams


def test_matches_the_representative_agent_solver_for_an_equivalent_jacobian():
    # The representative-agent IS curve y_t = y_{t+1} - sigma*r_dev_t,
    # iterated forward from the terminal condition y_T = 0, is exactly
    # y_t = -sigma * sum_{s=t}^{T-1} r_dev_s -- i.e. J[t, s] = -sigma for
    # s >= t, else 0. Feeding this synthetic "RANK Jacobian" into the HANK
    # machinery must reproduce nk_zlb.solve_transition_path exactly (up to
    # floating point), since the two are different implementations of the
    # same underlying linear system. This is the main correctness check for
    # the matrix-based OccBin solver in hank_zlb.py.
    p = NKParams()
    T = 30
    shock = natural_rate_shock(T=T, size=-0.02, duration=4, rho=0.7)

    J_rank = np.zeros((T, T))
    for t in range(T):
        J_rank[t, t:] = -p.sigma

    hank_path = solve_hank_transition_path(shock, J_rank, p)
    rank_path = solve_transition_path(shock, p)

    assert np.allclose(hank_path.y, rank_path.y, atol=1e-9)
    assert np.allclose(hank_path.pi, rank_path.pi, atol=1e-9)
    assert np.allclose(hank_path.i, rank_path.i, atol=1e-9)
    assert np.array_equal(hank_path.regime_binds, rank_path.regime_binds)
    assert hank_path.converged


def test_no_shock_is_the_steady_state_regardless_of_jacobian():
    p = NKParams()
    T = 20
    J_arbitrary = np.random.default_rng(0).normal(scale=0.05, size=(T, T))
    path = solve_hank_transition_path(np.zeros(T), J_arbitrary, p)

    assert np.allclose(path.y, 0.0, atol=1e-8)
    assert np.allclose(path.pi, 0.0, atol=1e-8)
    assert np.allclose(path.i, p.r_star, atol=1e-8)
    assert not path.regime_binds.any()
    assert path.converged
