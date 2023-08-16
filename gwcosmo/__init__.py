"""Top-level package for gwcosmo."""
from __future__ import absolute_import
from .utilities import cosmology, luminosity_function, posterior_utilities, redshift_utilities, cache
from .likelihood import posterior_samples, skymap
from .prior import catalog, priors
from .plotting import plot
from .maps import create_mth_map, create_norm_map

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("gwcosmo")
except PackageNotFoundError:
    # package is not installed
    pass
