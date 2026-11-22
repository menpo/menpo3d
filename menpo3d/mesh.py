import numpy as np

from menpo.base import copy_landmarks_and_path
from menpo.shape import ColouredTriMesh, TexturedTriMesh, TriMesh

from .io.input.mesh.stl import merge_points


def merge_duplicate_points(mesh, return_index=False):
    r"""
    Merges the points of a mesh that are identical, and returns a new mesh.

    Some files give a mesh one point per triangle corner: an OBJ whose texture
    coordinates or normals are indexed differently from its vertices is imported
    that way (as ``vtkOBJReader`` did), so a mesh of 6,000 vertices has 36,000
    points. Two points are merged when everything the mesh stores for a point is
    the same: its position, and its texture coordinates or colour if it has any.
    So the seams of a texture are kept (a vertex whose triangles use different
    texture coordinates stays as many points as it has of them) and nothing else
    about the surface changes.

    The result has the points in the order of their first appearance, and the
    triangles in the same order, with their indices changed. The landmarks and
    the path are kept. Triangles are not removed, so a triangle with two corners
    that were merged becomes degenerate, as it was in the file. Points that no
    triangle uses are kept.

    Parameters
    ----------
    mesh : :map:`TriMesh` or subclass
        The mesh. It is not changed.
    return_index : `bool`, optional
        If ``True`` the index of the new point of every point of the mesh is
        returned too, to bring other per-point data over with it: the data of
        the new points is ``values[np.unique(index, return_index=True)[1]]``.

    Returns
    -------
    merged : ``type(mesh)``
        The new mesh.
    index : ``(mesh.n_points,)`` `ndarray`
        For each point of ``mesh`` the index of its point in ``merged``. Only if
        ``return_index`` is ``True``.

    Raises
    ------
    ValueError
        ``mesh`` has no triangle list (it is a :map:`PointCloud`).
    """
    if getattr(mesh, "trilist", None) is None:
        raise ValueError(
            "merge_duplicate_points needs a mesh with a triangle list, not a {}".format(
                type(mesh).__name__
            )
        )
    columns = [mesh.points]
    if isinstance(mesh, TexturedTriMesh):
        columns.append(mesh.tcoords.points)
    if isinstance(mesh, ColouredTriMesh):
        columns.append(mesh.colours)
    _, index = merge_points(np.hstack(columns).astype(np.float64, copy=False))
    # the new points are numbered in order of first appearance: this is the
    # first point of the mesh that has each number
    first_of_each = np.unique(index, return_index=True)[1]

    points = mesh.points[first_of_each]
    trilist = index[mesh.trilist]
    if isinstance(mesh, TexturedTriMesh):
        merged = TexturedTriMesh(
            points,
            mesh.tcoords.points[first_of_each],
            mesh.texture.copy(),
            trilist=trilist,
            copy=False,
        )
    elif isinstance(mesh, ColouredTriMesh):
        merged = ColouredTriMesh(
            points, trilist=trilist, colours=mesh.colours[first_of_each], copy=False
        )
    else:
        merged = TriMesh(points, trilist=trilist, copy=False)
    copy_landmarks_and_path(mesh, merged)
    return (merged, index) if return_index else merged
