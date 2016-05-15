from menpo.io.output.extensions import ljson_exporter
from .mesh import obj_exporter


landmark_types = {
    '.ljson': ljson_exporter
}

mesh_types = {
    '.obj': obj_exporter
}