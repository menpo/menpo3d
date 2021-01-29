from menpo.io.output.extensions import ljson_exporter
from .mesh import obj_exporter, ply_exporter


landmark_types = {".ljson": ljson_exporter}

mesh_types_buffer_support = {".obj": obj_exporter}

mesh_types_paths_only = {".ply": ply_exporter}

mesh_types = {}
mesh_types.update(mesh_types_buffer_support)
mesh_types.update(mesh_types_paths_only)
