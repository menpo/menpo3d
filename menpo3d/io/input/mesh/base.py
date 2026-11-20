import warnings
import json

import numpy as np
import menpo.io as mio
from menpo.shape import ColouredTriMesh, TexturedTriMesh, TriMesh, PointCloud


def _construct_shape_type(points, trilist, tcoords, texture, colour_per_vertex):
    r"""
    Construct the correct Shape subclass given the inputs. TexturedTriMesh
    can only be created when tcoords and texture are available. ColouredTriMesh
    can only be created when colour_per_vertex is non None and TriMesh
    can only be created when trilist is non None. Worst case fall back is
    PointCloud.

    Parameters
    ----------
    points : ``(N, D)`` `ndarray`
        The N-D points.
    trilist : ``(N, 3)`` `ndarray`` or ``None``
        Triangle list or None.
    tcoords : ``(N, 2)`` `ndarray` or ``None``
        Texture coordinates.
    texture : :map:`Image` or ``None``
        Texture.
    colour_per_vertex : ``(N, 1)`` or ``(N, 3)`` `ndarray` or ``None``
        The colour per vertex.

    Returns
    -------
    shape : :map:`PointCloud` or subclass
        The correct shape for the given inputs.
    """
    # Four different outcomes - either we have a textured mesh, a coloured
    # mesh or just a plain mesh or we fall back to a plain pointcloud.
    if trilist is None:
        obj = PointCloud(points, copy=False)
    elif tcoords is not None and texture is not None:
        obj = TexturedTriMesh(points, tcoords, texture, trilist=trilist, copy=False)
    elif colour_per_vertex is not None:
        obj = ColouredTriMesh(
            points, trilist=trilist, colours=colour_per_vertex, copy=False
        )
    else:
        # TriMesh fall through
        obj = TriMesh(points, trilist=trilist, copy=False)

    if tcoords is not None and texture is None:
        warnings.warn(
            "tcoords were found, but no texture was recovered, "
            "reverting to an untextured mesh."
        )
    if texture is not None and tcoords is None:
        warnings.warn(
            "texture was found, but no tcoords were recovered, "
            "reverting to an untextured mesh."
        )

    return obj


def wrl_importer(filepath, asset=None, texture_resolver=None, **kwargs):
    r"""Allows importing VRML 2.0 meshes.

    Reproduces the topology ``vtkVRMLImporter`` produced and, like it, uses the first
    ``IndexedFaceSet`` in the scene (``Transform`` nodes are ignored). If the texture
    coordinate or normal indices differ from ``coordIndex`` every face corner becomes its
    own point. Polygons are split into triangles the way VTK did (with a warning).

    Parameters
    ----------
    asset : `object`, optional
        An optional asset that may help with loading. This is unused for this
        implementation.
    texture_resolver : `callable`, optional
        A callable that recieves the mesh filepath and returns a single
        path to the texture to load.
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    shape : :map:`PointCloud` or subclass
        The correct shape for the given inputs.
    """
    from .triangulate import triangulate_polygons
    from .wrl import read_wrl

    data = read_wrl(filepath)
    points = data.points.astype(np.float64)
    trilist = triangulate_polygons(points, data.counts, data.connectivity)

    texture = None
    if texture_resolver is not None:
        texture_path = texture_resolver(filepath)
        if texture_path is not None and texture_path.exists():
            texture = mio.import_image(texture_path)

    tcoords = data.tcoords
    if tcoords is not None and len(tcoords) != len(points):
        tcoords = None

    colour_per_vertex = data.colours
    if colour_per_vertex is not None and len(colour_per_vertex) != len(points):
        colour_per_vertex = None

    return _construct_shape_type(points, trilist, tcoords, texture, colour_per_vertex)


def obj_importer(filepath, asset=None, texture_resolver=None, **kwargs):
    r"""Allows importing Wavefront (OBJ) files.

    The topology is the one ``vtkOBJReader`` produced, which datasets built with
    earlier versions of menpo3d rely on: if every ``vt``/``vn`` index equals its ``v``
    index the points are the ``v`` list (unused vertices included), otherwise every
    face corner becomes its own point. Polygons are split into triangles the way
    VTK did (with a warning). Merging duplicated points is a separate step that
    belongs after import.

    Parameters
    ----------
    asset : `object`, optional
        An optional asset that may help with loading. This is unused for this
        implementation.
    texture_resolver : `callable`, optional
        A callable that recieves the mesh filepath and returns a single
        path to the texture to load.
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    shape : :map:`PointCloud` or subclass
        The correct shape for the given inputs.
    """
    from .obj import read_obj
    from .triangulate import triangulate_polygons

    data = read_obj(filepath)
    points = data.points
    trilist = triangulate_polygons(points, data.counts, data.connectivity)

    texture = None
    if texture_resolver is not None:
        texture_path = texture_resolver(filepath)
        if texture_path is not None and texture_path.exists():
            texture = mio.import_image(texture_path)

    tcoords = data.tcoords if texture is not None else None
    colour_per_vertex = None
    return _construct_shape_type(points, trilist, tcoords, texture, colour_per_vertex)


def ply_importer(filepath, asset=None, texture_resolver=None, **kwargs):
    r"""Allows importing Polygon File Format (PLY) files.

    Positions and texture coordinates are float32, as they were when VTK read them.
    Polygons are split into triangles the way VTK did (with a warning).

    Parameters
    ----------
    asset : `object`, optional
        An optional asset that may help with loading. This is unused for this
        implementation.
    texture_resolver : `callable`, optional
        A callable that recieves the mesh filepath and returns a single
        path to the texture to load.
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    shape : :map:`PointCloud` or subclass
        The correct shape for the given inputs.
    """
    from .ply import read_ply
    from .triangulate import triangulate_polygons

    data = read_ply(filepath)
    points = data.points.astype(np.float64)
    trilist = triangulate_polygons(points, data.counts, data.connectivity)

    texture = None
    if texture_resolver is not None:
        texture_path = texture_resolver(filepath)
        if texture_path is not None and texture_path.exists():
            texture = mio.import_image(texture_path)

    tcoords = data.tcoords if texture is not None else None

    scalar_per_vertex = data.colours
    if scalar_per_vertex is not None:
        scalar_per_vertex = scalar_per_vertex[:, :3]
        # Convert to [0, 1] floats
        scalar_per_vertex = scalar_per_vertex * (1 / 255.0)

    return _construct_shape_type(points, trilist, tcoords, texture, scalar_per_vertex)


def stl_importer(filepath, asset=None, **kwargs):
    r"""Allows importing Stereolithography CAD (STL) files.

    Identical vertices are merged (in order of first appearance) and triangles that
    collapse are dropped, as ``vtkSTLReader`` did.

    Parameters
    ----------
    asset : `object`, optional
        An optional asset that may help with loading. This is unused for this
        implementation.
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    shape : :map:`PointCloud` or subclass
        The correct shape for the given inputs.
    """
    from .stl import read_stl

    points, trilist = read_stl(filepath)
    return _construct_shape_type(points.astype(np.float64), trilist, None, None, None)


def mjson_importer(filepath, asset=None, texture_resolver=True, **kwargs):
    r"""
    Import meshes that are in a simple JSON format.

    Parameters
    ----------
    asset : `object`, optional
        An optional asset that may help with loading. This is unused for this
        implementation.
    texture_resolver : `callable`, optional
        A callable that recieves the mesh filepath and returns a single
        path to the texture to load.
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    shape : :map:`PointCloud` or subclass
        The correct shape for the given inputs.
    """
    with open(str(filepath), "rb") as f:
        mesh_json = json.load(f)

    texture = None
    if texture_resolver is not None:
        texture_path = texture_resolver(filepath)
        if texture_path is not None and texture_path.exists():
            texture = mio.import_image(texture_path)

    points = mesh_json["points"]
    trilist = mesh_json["trilist"]
    tcoords = (mesh_json.get("tcoords"),)
    colour_per_vertex = mesh_json.get("colour_per_vertex")

    return _construct_shape_type(points, trilist, tcoords, texture, colour_per_vertex)
