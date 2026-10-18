"""Example: solve the ZLB transition using the *genuine* HANK IS curve
(household-block sequence-space Jacobian, ``hank_zlb.py``) and compare it
against the representative-agent version (``nk_zlb.py``) for the same
shock.

Headline finding (see README for the full discussion): the household
block's Jacobian has a *much smaller* own-period semi-elasticity than the
representative-agent calibration's sigma=1. At the same shock size used
in the representative-agent demo, the HANK economy barely moves and never
hits the ZLB; a noticeably larger shock is needed to produce a comparable
ZLB episode, and even then the implied output gap is large enough to be
outside the range where the log-linear approximation underlying the NKPC
and Taylor rule is really trustworthy. This is reported honestly rather
than papered over with a recalibration chosen to "look nice".

Run with:
    .venv/Scripts/python.exe examples/run_hank_zlb_transition.py
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from hank_pfi.hank_zlb import build_hank_jacobian, solve_hank_transition_path
from hank_pfi.nk_zlb import natural_rate_shock, solve_transition_path
from hank_pfi.params import FirmParams, HouseholdParams, NKParams
from hank_pfi.steady_state import solve_steady_state


def main() -> None:
    hh_p = HouseholdParams(n_a=200, n_e=5)
    firm = FirmParams()
    nk_p = NKParams()
    T = 20

    print("Solving the Aiyagari steady state...")
    ss = solve_steady_state(hh_p, firm)
    print(f"  r* = {ss.r:.4%}, C* = {ss.aggregates.C:.4f}")

    print(f"Building the household block's sequence-space Jacobian (T={T})...")
    t0 = time.time()
    J_eff = build_hank_jacobian(
        hh_p, ss.household, ss.distribution, ss.r, ss.w, ss.aggregates.C, T,
        dr=0.02, dlogw=0.1,
    )
    print(f"  done in {time.time() - t0:.1f}s. J_eff diag: {np.round(np.diag(J_eff), 4)}")
    print(f"  (representative-agent sigma for comparison: {nk_p.sigma})")

    same_size_shock = natural_rate_shock(T=T, size=-0.02, duration=4, rho=0.7)
    hank_same = solve_hank_transition_path(same_size_shock, J_eff, nk_p)
    rank_same = solve_transition_path(same_size_shock, nk_p)
    print(
        f"\nAt the representative-agent demo's shock size (-0.02): "
        f"RANK ZLB periods={rank_same.regime_binds.sum()}, "
        f"HANK ZLB periods={hank_same.regime_binds.sum()}"
    )

    large_shock = natural_rate_shock(T=T, size=-0.05, duration=4, rho=0.7)
    hank_large = solve_hank_transition_path(large_shock, J_eff, nk_p)
    rank_large = solve_transition_path(large_shock, nk_p)
    print(
        f"At a larger shock (-0.05): "
        f"RANK ZLB periods={rank_large.regime_binds.sum()}, "
        f"HANK ZLB periods={hank_large.regime_binds.sum()}, "
        f"HANK y[0]={hank_large.y[0]:.3f} (log gap -- large, see caveat in module docstring)"
    )

    # The matrix-based HANK solve anchors y_T = 0 exactly at the horizon's
    # end; with the empirical (noisy) household Jacobian this can show a
    # visible finite-horizon boundary artifact in the last few periods
    # (confirmed while building this example: a hockey-stick-shaped
    # overshoot right before t=T that is not present with the clean
    # representative-agent Jacobian). Plotting only the first part of the
    # horizon keeps the figure focused on the economically relevant
    # near-term response; T itself is kept long enough that this window is
    # unaffected. See the README's "このマイルストーンで分かったこと"
    # section for the full discussion.
    t_plot = T * 3 // 5
    t = np.arange(t_plot)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    axes[0, 0].plot(t, rank_same.y[:t_plot], label="RANK (sigma=1)")
    axes[0, 0].plot(t, hank_same.y[:t_plot], label="HANK (household Jacobian)")
    axes[0, 0].set_title("Output gap, shock = -0.02 (RANK demo size)")
    axes[0, 0].axhline(0, color="k", lw=0.5)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(t, rank_same.i[:t_plot], label="RANK")
    axes[0, 1].plot(t, hank_same.i[:t_plot], label="HANK")
    axes[0, 1].axhline(0, color="r", lw=0.8, ls="--", label="ZLB")
    axes[0, 1].set_title("Policy rate, shock = -0.02")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(t, rank_large.y[:t_plot], label="RANK")
    axes[1, 0].plot(t, hank_large.y[:t_plot], label="HANK")
    axes[1, 0].set_title("Output gap, shock = -0.05")
    axes[1, 0].axhline(0, color="k", lw=0.5)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(t, rank_large.i[:t_plot], label="RANK")
    axes[1, 1].plot(t, hank_large.i[:t_plot], label="HANK")
    axes[1, 1].axhline(0, color="r", lw=0.8, ls="--", label="ZLB")
    axes[1, 1].set_title("Policy rate, shock = -0.05")
    axes[1, 1].legend(fontsize=8)

    for ax in axes.ravel():
        ax.set_xlabel("period")

    fig.tight_layout()
    fig.savefig("examples/hank_vs_rank_zlb.png", dpi=150)
    print("\nSaved examples/hank_vs_rank_zlb.png")


if __name__ == "__main__":
    main()
