from . import barycentric
from . import correspond
from . import io
from . import rasterize
from . import unwrap
from . import visualize
from . import vtkutils

try:
    from ._version import __version__
except ImportError:  # not installed, e.g. a source checkout that was never built
    from importlib.metadata import PackageNotFoundError, version as _version

    try:
        __version__ = _version("menpo3d")
    except PackageNotFoundError:
        __version__ = "unknown"
