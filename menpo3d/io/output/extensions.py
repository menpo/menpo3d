from menpo.io.output.extensions import LJSONExporter
from .mesh import OBJExporter


landmark_types = {
    '.ljson': LJSONExporter
}

mesh_types = {
    '.obj': OBJExporter
}