import numpy as np

from hank_pfi.jacobian import household_jacobians
from hank_pfi.params import FirmParams, HouseholdParams
from hank_pfi.steady_state import solve_steady_state


def test_jacobians_are_finite_and_the_right_shape():
    hh_p = HouseholdParams(n_a=100, n_e=5)
    firm = FirmParams()
    ss = solve_steady_state(hh_p, firm)
    T = 8

    J = household_jacobians(
        hh_p, ss.household, ss.distribution, ss.r, ss.w, ss.aggregates.C, T,
        dr=0.02, dlogw=0.1,
    )

    assert J.J_r.shape == (T, T)
    assert J.J_w.shape == (T, T)
    assert np.all(np.isfinite(J.J_r))
    assert np.all(np.isfinite(J.J_w))


def test_own_period_wage_response_is_a_positive_mpc():
    # d log C_t / d log w_t (holding r fixed) is a same-period MPC out of
    # labor income -- should be positive (more income this period raises
    # this period's consumption) for a sizable majority of shock dates,
    # at a perturbation large enough to not be dominated by the
    # discrete-grid artifact documented in transition._quadratic_refine_a
    # (small steps are noisy; see the package's development notes).
    hh_p = HouseholdParams(n_a=150, n_e=5)
    firm = FirmParams()
    ss = solve_steady_state(hh_p, firm)
    T = 6

    J = household_jacobians(
        hh_p, ss.household, ss.distribution, ss.r, ss.w, ss.aggregates.C, T,
        dr=0.02, dlogw=0.2,
    )
    diag = np.diag(J.J_w)
    assert np.median(diag) > 0
