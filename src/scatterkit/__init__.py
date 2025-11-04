"""scatterkit: SAXS analysis of MD trajectories."""

__authors__ = "MAICoS Developer Team"

from ._version import get_versions
from .diporderstructurefactor import DiporderStructureFactor
from .rdfdiporder import RDFDiporder
from .saxs import Saxs

__authors__ = "MAICoS Developer Team"
#: Version information for MAICoS, following :pep:`440`
#: and `semantic versioning <http://semver.org/>`_.
__version__ = get_versions()["version"]
del get_versions

__all__ = ["Saxs", "DiporderStructureFactor", "RDFDiporder"]
