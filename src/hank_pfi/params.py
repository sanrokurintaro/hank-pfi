"""Calibration primitives shared across layers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HouseholdParams:
    """Preferences, asset grid and income-process parameters for the
    idiosyncratic household problem (Aiyagari-Bewley-Huggett block)."""

    beta: float = 0.96          # discount factor
    gamma: float = 2.0          # CRRA risk aversion
    a_min: float = 0.0          # borrowing constraint
    a_max: float = 200.0        # upper bound of the asset grid
    n_a: int = 200              # number of asset grid points
    grid_curvature: float = 1.7 # >1 concentrates points near a_min
    rho_e: float = 0.9          # persistence of log income
    sigma_e: float = 0.3        # std. dev. of income innovations
    n_e: int = 7                # number of income states (Rouwenhorst)


@dataclass(frozen=True)
class FirmParams:
    """Representative Cobb-Douglas firm, standard Aiyagari general
    equilibrium closure."""

    alpha: float = 0.36   # capital share
    delta: float = 0.08   # depreciation rate
    tfp: float = 1.0      # total factor productivity


@dataclass(frozen=True)
class NKParams:
    """Reduced-form 3-equation NK block used for the ZLB transition-path
    layer. sigma/kappa are placeholders for the semi-elasticities that a
    later milestone should replace with values implied by the household
    block (sequence-space-Jacobian style), see README 'Next steps'."""

    sigma: float = 1.0     # intertemporal elasticity of substitution (IS slope)
    kappa: float = 0.1     # NKPC slope
    beta: float = 0.99     # discount factor of the NK block (quarterly)
    phi_pi: float = 1.5    # Taylor rule response to inflation
    phi_y: float = 0.5     # Taylor rule response to the output gap
    r_star: float = 0.01   # steady-state nominal rate (net, per period).
    # NOTE: keep this comfortably above 0. At r_star=0 the steady state
    # sits exactly on the ZLB, so even an infinitesimal negative deviation
    # is flagged as "binding" forever as the shock decays asymptotically
    # to 0 -- a degenerate edge case of the boundary itself, not a
    # genuine prolonged liquidity trap. r_star > 0 (e.g. a real rate plus
    # an inflation target) avoids this.
