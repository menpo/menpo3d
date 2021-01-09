from menpo.io.input.landmark import ljson_importer
from .mesh import wrl_importer, mjson_importer, obj_importer, stl_importer, ply_importer
from .landmark_mesh import bnd_importer, lan_importer, lm3_importer, pts_mesh_importer
from .lsfm import lsfm_model_importer

# TODO: Add PLY (ASCII and binary) and OFF importers
mesh_types = {
    ".obj": obj_importer,
    ".stl": stl_importer,
    ".ply": ply_importer,
    ".wrl": wrl_importer,
    ".mjson": mjson_importer,
}

mesh_landmark_types = {
    ".pts3": pts_mesh_importer,
    ".lm3": lm3_importer,
    ".lan": lan_importer,
    ".bnd": bnd_importer,
    ".ljson": ljson_importer,
}

lsfm_types = {".mat": lsfm_model_importer}
