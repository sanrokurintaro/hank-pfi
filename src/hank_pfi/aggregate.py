"""Layer (4): aggregation of micro policies into macro quantities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .household import HouseholdSolution


@dataclass
class Aggregates:
    K: float   # aggregate asset holdings (= capital supply)
    C: float   # aggregate consumption
    L: float   # aggregate effective labor supply


def aggregate(sol: HouseholdSolution, dist: np.ndarray) -> Aggregates:
    K = float(np.sum(dist * sol.a_grid[:, None]))
    C = float(np.sum(dist * sol.policy_c))
    L = float(np.sum(dist * sol.e_grid[None, :]))
    return Aggregates(K=K, C=C, L=L)
