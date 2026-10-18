"""Example (egm-household branch): solve the ZLB transition using the
EGM-based household block instead of DiscreteDP, and compare against the
DiscreteDP version for the same shock.

This is the direct counterpart of
``examples/run_hank_zlb_transition.py`` (master branch) -- same shock,
same NK block, only the household layer underneath differs. The
headline difference: the EGM Jacobian is built with perturbation sizes
three orders of magnitude smaller (dr=dlogw=1e-4 vs 0.02/0.1) and still
comes out clean, because EGM's policy function doesn't have the
discrete-grid snapping that corrupted small-step finite differences on
``master``.

Run with:
    poetry run python examples/run_hank_zlb_transition_egm.py
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from hank_pfi.hank_zlb import build_hank_jacobian, build_hank_jacobian_egm, solve_hank_transition_path
from hank_pfi.nk_zlb import natural_rate_shock
from hank_pfi.params import FirmParams, HouseholdParams, NKParams
from hank_pfi.steady_state import solve_steady_state
from hank_pfi.steady_state_egm import solve_steady_state_egm


def main() -> None:
    hh_p = HouseholdParams(n_a=150, n_e=5)
    firm = FirmParams()
    nk_p = NKParams()
    T = 30
    shock = natural_rate_shock(T=T, size=-0.02, duration=4, rho=0.7)

    print("Solving the DiscreteDP steady state and Jacobian...")
    ss_dp = solve_steady_state(hh_p, firm)
    t0 = time.time()
    J_dp = build_hank_jacobian(
        hh_p, ss_dp.household, ss_dp.distribution, ss_dp.r, ss_dp.w, ss_dp.aggregates.C, T,
    )
    print(f"  DiscreteDP Jacobian (dr=0.02, dlogw=0.1): {time.time() - t0:.1f}s")

    print("Solving the EGM steady state and Jacobian...")
    ss_egm = solve_steady_state_egm(hh_p, firm)
    t0 = time.time()
    J_egm = build_hank_jacobian_egm(
        hh_p, ss_egm.household, ss_egm.distribution, ss_egm.r, ss_egm.w, ss_egm.aggregates.C, T,
    )
    print(f"  EGM Jacobian (dr=dlogw=1e-4): {time.time() - t0:.1f}s")

    path_dp = solve_hank_transition_path(shock, J_dp, nk_p)
    path_egm = solve_hank_transition_path(shock, J_egm, nk_p)
    print(f"DiscreteDP: converged={path_dp.converged}, ZLB periods={path_dp.regime_binds.sum()}")
    print(f"EGM:        converged={path_egm.converged}, ZLB periods={path_egm.regime_binds.sum()}")

    t_plot = T * 2 // 3
    t = np.arange(t_plot)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].plot(np.arange(T), np.diag(J_dp), label="DiscreteDP (dr=0.02)")
    axes[0].plot(np.arange(T), np.diag(J_egm), label="EGM (dr=1e-4)")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_title("J_eff diagonal (own-period IS slope)")
    axes[0].set_xlabel("period")
    axes[0].legend(fontsize=8)

    axes[1].plot(t, path_dp.y[:t_plot], label="DiscreteDP")
    axes[1].plot(t, path_egm.y[:t_plot], label="EGM")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title("Output gap")
    axes[1].set_xlabel("period")
    axes[1].legend(fontsize=8)

    axes[2].plot(t, path_dp.i[:t_plot], label="DiscreteDP")
    axes[2].plot(t, path_egm.i[:t_plot], label="EGM")
    axes[2].axhline(0, color="r", lw=0.8, ls="--", label="ZLB")
    axes[2].set_title("Policy rate")
    axes[2].set_xlabel("period")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("examples/hank_egm_vs_discretedp.png", dpi=150)
    print("\nSaved examples/hank_egm_vs_discretedp.png")


if __name__ == "__main__":
    main()
