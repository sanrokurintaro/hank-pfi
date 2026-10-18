"""hank_pfi: HANK models with occasionally-binding constraints, solved via
Policy Function Iteration (Howard improvement) on a discretized state space.

Architecture (see README.md for the full layer table):
    grids        -- asset grid & income process discretization
    household    -- idiosyncratic Bellman problem via QuantEcon DiscreteDP
    distribution -- stationary cross-sectional distribution
    aggregate    -- aggregation of micro policies into macro quantities
    steady_state -- Aiyagari-style general equilibrium (bisection on r)
    nk_zlb       -- representative-agent NK block with an occasionally
                    binding ZLB, solved by OccBin-style regime switching
    transition   -- finite-horizon ("MIT shock") household transition
    jacobian     -- household block's sequence-space Jacobians
    hank_zlb     -- the genuine HANK replacement for nk_zlb, using the
                    household block's own Jacobians as the IS curve
"""

from . import (
    grids,
    household,
    distribution,
    aggregate,
    steady_state,
    nk_zlb,
    transition,
    jacobian,
    hank_zlb,
)

__all__ = [
    "grids",
    "household",
    "distribution",
    "aggregate",
    "steady_state",
    "nk_zlb",
    "transition",
    "jacobian",
    "hank_zlb",
]
