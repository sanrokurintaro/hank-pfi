"""Example: solve the Aiyagari steady state (layers 1-5) and plot the
wealth distribution and savings policy.

Run with:
    .venv/Scripts/python.exe examples/run_aiyagari.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from hank_pfi.params import FirmParams, HouseholdParams
from hank_pfi.steady_state import solve_steady_state


def main() -> None:
    hh_p = HouseholdParams(n_a=300, n_e=7)
    firm = FirmParams()

    print("Solving the Aiyagari steady state (PFI via QuantEcon DiscreteDP)...")
    ss = solve_steady_state(hh_p, firm)

    print(f"r*       = {ss.r:.4%}")
    print(f"w*       = {ss.w:.4f}")
    print(f"K supply = {ss.aggregates.K:.4f}")
    print(f"K demand = {ss.K_demand:.4f}")
    print(f"C        = {ss.aggregates.C:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    marginal_wealth = ss.distribution.sum(axis=1)
    axes[0].plot(ss.household.a_grid, marginal_wealth)
    axes[0].set_xlabel("assets a")
    axes[0].set_ylabel("density")
    axes[0].set_title("Stationary wealth distribution")
    axes[0].set_xlim(0, ss.household.a_grid[marginal_wealth > 1e-5][-1] * 1.2)

    for j, e in enumerate(ss.household.e_grid):
        axes[1].plot(ss.household.a_grid, ss.household.policy_a[:, j], label=f"e={e:.2f}")
    axes[1].plot(ss.household.a_grid, ss.household.a_grid, "k--", lw=0.8, label="45°")
    axes[1].set_xlabel("assets a")
    axes[1].set_ylabel("a'(a, e)")
    axes[1].set_title("Savings policy (Howard-improved PFI)")
    axes[1].legend(fontsize=7)
    axes[1].set_xlim(0, ss.household.a_grid[marginal_wealth > 1e-5][-1] * 1.2)
    axes[1].set_ylim(0, ss.household.a_grid[marginal_wealth > 1e-5][-1] * 1.2)

    fig.tight_layout()
    fig.savefig("examples/aiyagari_steady_state.png", dpi=150)
    print("Saved examples/aiyagari_steady_state.png")


if __name__ == "__main__":
    main()
