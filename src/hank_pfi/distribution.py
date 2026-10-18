"""Layer (3): the stationary cross-sectional distribution.

The induced Markov chain (``res.mc``, built by QuantEcon's DiscreteDP from
the PFI-optimal policy) is driven forward by power iteration from a
uniform initial distribution, rather than solved via
``MarkovChain.stationary_distributions`` (an eigenvector-based solver).

Why: with a finite asset grid and a *pure* discrete choice (no lottery /
interpolation between grid points), the discretized optimal policy can
map many (a, e) states onto the same handful of a' grid points. This
makes the induced chain reducible into several disconnected recurrent
classes even though the underlying economic problem is not degenerate --
a well-known artifact of naive grid-based discretization (see e.g. the
discussion around Young (2010) non-stochastic simulation). Power
iteration from a non-degenerate (uniform) starting distribution picks out
the long-run distribution of a population that starts spread across the
whole state space, which is the economically meaningful object for an
Aiyagari economy and does not depend on an arbitrary choice among
multiple eigenvectors.
"""

from __future__ import annotations

import warnings

import numpy as np

from .household import HouseholdSolution

_TOP_GRID_WARNING_THRESHOLD = 1e-3


def stationary_distribution(
    sol: HouseholdSolution, tol: float = 1e-12, max_iter: int = 100_000
) -> np.ndarray:
    """Return the stationary distribution over (a, e) states as an
    (n_a, n_e) array that sums to 1, via power iteration on the induced
    Markov chain starting from the uniform distribution.
    """
    P = sol.ddp_result.mc.P  # sparse (n, n), rows sum to 1
    n = P.shape[0]
    dist = np.full(n, 1.0 / n)

    for _ in range(max_iter):
        dist_next = dist @ P
        dist_next = np.asarray(dist_next).ravel()
        if np.max(np.abs(dist_next - dist)) < tol:
            dist = dist_next
            break
        dist = dist_next
    else:
        warnings.warn(
            f"stationary distribution did not converge to tol={tol} "
            f"within {max_iter} iterations.",
            stacklevel=2,
        )

    n_a, n_e = sol.v.shape
    dist = dist.reshape(n_a, n_e)

    top_mass = dist[-1, :].sum()
    if top_mass > _TOP_GRID_WARNING_THRESHOLD:
        warnings.warn(
            f"{top_mass:.2%} of the stationary distribution's mass sits "
            "at the top asset grid point -- a_max is likely binding. "
            "Increase HouseholdParams.a_max.",
            stacklevel=2,
        )

    return dist
