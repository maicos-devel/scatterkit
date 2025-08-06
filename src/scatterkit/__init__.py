"""scatterkit: SAXS analysis of MD trajectories"""

__authors__ = "MAICoS Developer Team"

import warnings

from ._version import get_versions

__authors__ = "MAICoS Developer Team"
#: Version information for MAICoS, following :pep:`440`
#: and `semantic versioning <http://semver.org/>`_.
__version__ = get_versions()["version"]
del get_versions

from .saxs import Saxs

__all__ = [
    "Saxs",
]
