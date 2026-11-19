import numpy as np

from menpo.io.output.base import _enforce_only_paths_supported
from menpo.shape.mesh import TexturedTriMesh


def obj_exporter(mesh, file_handle, **kwargs):
    r"""
    Given a file handle to write in to (which should act like a Python `file`
    object), write out the mesh data. No value is returned.

    Note that this does not save out textures of textured images, and so should
    not be used in isolation.

    Parameters
    ----------
    file_handle : `str`
        The full path where the obj will be saved out.
    mesh : :map:`TriMesh`
        Any subclass of :map:`TriMesh`. If :map:`TexturedTriMesh` texture
        coordinates will be saved out. Note that :map:`ColouredTriMesh`
        will only have shape data saved out, as .OBJ doesn't robustly support
        per-vertex colour information.
    """
    for v in mesh.points:
        file_handle.write("v {} {} {}\n".format(*v).encode("utf-8"))
    file_handle.write(b"\n")
    if isinstance(mesh, TexturedTriMesh):
        for tc in mesh.tcoords.points:
            file_handle.write("vt {} {}\n".format(*tc).encode("utf-8"))
        file_handle.write(b"\n")
        # triangulation of points and tcoords is identical
        for t in mesh.trilist + 1:
            file_handle.write("f {0}/{0} {1}/{1} {2}/{2}\n".format(*t).encode("utf-8"))
    else:
        # no tcoords - so triangulation is straight forward
        for t in mesh.trilist + 1:
            file_handle.write("f {} {} {}\n".format(*t).encode("utf-8"))


def ply_exporter(mesh, file_path, binary=False, **kwargs):
    r"""
    Given a file path to write in to write out the mesh data.
    No value is returned. Only file paths are supported and if a file handle
    is passed it will be ignored and a warning will be raised.

    Note that this does not save out textures of textured images, and so should
    not be used in isolation.

    Parameters
    ----------
    file_path : `str`
        The full path where the obj will be saved out.
    mesh : :map:`TriMesh`
        Any subclass of :map:`TriMesh`. If :map:`TexturedTriMesh` texture
        coordinates will be saved out. Note that :map:`ColouredTriMesh`
        will only have shape data saved out, as .PLY doesn't robustly support
        per-vertex colour information.
    binary: `bool`, optional
        Specify whether to format output in binary or ascii, defaults to False
    """
    file_path = _enforce_only_paths_supported(file_path, "PLY")

    # float32 vertex data, matching what the previous (VTK based) writer produced
    columns = [mesh.points.astype(np.float32)]
    names = ["x", "y", "z"]
    if isinstance(mesh, TexturedTriMesh):
        columns.append(mesh.tcoords.points.astype(np.float32))
        names += ["u", "v"]
    vertices = np.hstack(columns)
    trilist = np.asarray(mesh.trilist, dtype=np.int32)

    header = ["ply"]
    header.append("format binary_little_endian 1.0" if binary else "format ascii 1.0")
    header.append("comment menpo3d generated PLY File")
    header.append("element vertex {}".format(len(vertices)))
    header += ["property float {}".format(name) for name in names]
    header.append("element face {}".format(len(trilist)))
    header.append("property list uchar int vertex_indices")
    header.append("end_header")

    with open(str(file_path), "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        if binary:
            f.write(vertices.astype("<f4").tobytes())
            faces = np.empty(len(trilist), dtype=[("n", "u1"), ("i", "<i4", (3,))])
            faces["n"] = 3
            faces["i"] = trilist
            f.write(faces.tobytes())
        else:
            np.savetxt(f, vertices, fmt="%.9g")
            np.savetxt(f, np.hstack([np.full((len(trilist), 1), 3), trilist]), fmt="%d")
