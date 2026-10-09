"""plume_inv: unsteady, polydisperse inversion of deep-sea tephra deposits.

Modules
-------
constants    : physical constants and benchmark parameter values, with provenance.
stem         : plume-stem closures (point and planar sources) and the heat flux.
settling     : settling velocity of basaltic tephra in cold seawater.
umbrella     : umbrella-cloud dynamics, quasi-steady and unsteady (front-limited).
kernel       : deposition kernels and the nu(Q) representation of the source.
spacetime    : space-time structure of the front-limited deposit.
forward      : deposit prediction for all size classes at all core locations.
data         : loaders for the NESCA push-core data, with provenance checks.
svd          : identifiability, the number of source modes a deposit resolves.
inverse      : Bayesian inversion for the measure nu(Q).
maps         : map-view deposit and vent-position misfit.
shape        : clast shape from the spread of dispersal length scales.
sources      : candidate heat sources for a megaplume, with bounded parameters.
heat_context : the megaplume heat flux against other geophysical heat fluxes.
"""

__version__ = "1.0.0"

from . import constants, forward, kernel, settling, stem, umbrella  # noqa: F401

__all__ = ["constants", "stem", "settling", "umbrella", "kernel", "forward"]
