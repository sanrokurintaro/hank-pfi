"""Example: solve a ZLB-constrained transition path (layer 6) in response
to a temporary negative natural-rate shock, via OccBin-style regime
switching, and plot the impulse response.

Run with:
    .venv/Scripts/python.exe examples/run_zlb_transition.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from hank_pfi.nk_zlb import natural_rate_shock, solve_transition_path
from hank_pfi.params import NKParams


def main() -> None:
    p = NKParams()
    T = 60
    rstar_path = natural_rate_shock(T=T, size=-0.02, duration=4, rho=0.7)

    path = solve_transition_path(rstar_path, p)
    print(f"converged: {path.converged} (n_iter={path.n_iter})")
    binds = np.where(path.regime_binds)[0]
    print(f"ZLB binds in periods {binds[0]}..{binds[-1]} ({len(binds)} periods)")

    t = np.arange(T)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].plot(t, rstar_path)
    axes[0, 0].set_title("Natural-rate shock")
    axes[0, 0].axhline(0, color="k", lw=0.5)

    axes[0, 1].plot(t, path.y)
    axes[0, 1].set_title("Output gap")
    axes[0, 1].axhline(0, color="k", lw=0.5)

    axes[1, 0].plot(t, path.pi)
    axes[1, 0].set_title("Inflation")
    axes[1, 0].axhline(0, color="k", lw=0.5)

    axes[1, 1].plot(t, path.i, label="nominal rate i_t")
    axes[1, 1].axhline(0, color="r", lw=0.8, ls="--", label="ZLB")
    axes[1, 1].set_title("Policy rate")
    axes[1, 1].legend(fontsize=8)

    for ax in axes.ravel():
        ax.set_xlabel("period")

    fig.tight_layout()
    fig.savefig("examples/zlb_transition.png", dpi=150)
    print("Saved examples/zlb_transition.png")


if __name__ == "__main__":
    main()
