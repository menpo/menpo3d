from . import barycentric
from . import correspond
from . import io
from . import rasterize
from . import unwrap
from . import visualize
from . import vtkutils

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
