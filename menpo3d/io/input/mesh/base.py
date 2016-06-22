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
        obj = TexturedTriMesh(points, tcoords, texture,
                              trilist=trilist, copy=False)
    elif colour_per_vertex is not None:
        obj = ColouredTriMesh(points, trilist=trilist,
                              colours=colour_per_vertex, copy=False)
    else:
        # TriMesh fall through
        obj = TriMesh(points, trilist=trilist, copy=False)

    if tcoords is not None and texture is None:
        warnings.warn('tcoords were found, but no texture was recovered, '
                      'reverting to an untextured mesh.')
    if texture is not None and tcoords is None:
        warnings.warn('texture was found, but no tcoords were recovered, '
                      'reverting to an untextured mesh.')

    return obj


def vtk_ensure_trilist(polydata):
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        trilist = vtk_to_numpy(polydata.GetPolys().GetData())

        # 5 is the triangle type - if we have another type we need to
        # use a vtkTriangleFilter
        c = vtk.vtkCellTypes()
        polydata.GetCellTypes(c)

        if c.GetNumberOfTypes() != 1 or polydata.GetCellType(0) != 5:
            warnings.warn('Non-triangular mesh connectivity was detected - '
                          'this is currently unsupported and thus the '
                          'connectivity is being coerced into a triangular '
                          'mesh. This may have unintended consequences.')
            t_filter = vtk.vtkTriangleFilter()
            t_filter.SetInput(polydata)
            t_filter.Update()
            trilist = vtk_to_numpy(t_filter.GetOutput().GetPolys().GetData())

        return trilist.reshape([-1, 4])[:, 1:]
    except Exception as e:
        print(e)
        return None


def wrl_importer(filepath, asset=None, texture_resolver=None, **kwargs):
    """Allows importing VRML 2.0 meshes.

    Uses VTK and assumes that the first actor in the scene is the one
    that we want.

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
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    vrml_importer = vtk.vtkVRMLImporter()
    vrml_importer.SetFileName(str(filepath))
    vrml_importer.Update()

    # Get the first actor.
    actors = vrml_importer.GetRenderer().GetActors()
    actors.InitTraversal()
    mapper = actors.GetNextActor().GetMapper()
    mapper_dataset = mapper.GetInput()

    if actors.GetNextActor():
        # There was more than one actor!
        warnings.warn('More than one actor was detected in the scene. Only '
                      'single scene actors are currently supported.')

    # Get the Data
    polydata = vtk.vtkPolyData.SafeDownCast(mapper_dataset)
    polydata.Update()

    # We must have point data!
    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float)

    trilist = vtk_ensure_trilist(polydata)

    texture = None
    if texture_resolver is not None:
        texture_path = texture_resolver(filepath)
        if texture_path is not None and texture_path.exists():
            texture = mio.import_image(texture_path)

    # Three different outcomes - either we have a textured mesh, a coloured
    # mesh or just a plain mesh. Let's try each in turn.

    # Textured
    tcoords = None
    try:
        tcoords = vtk_to_numpy(polydata.GetPointData().GetTCoords())
    except Exception:
        pass

    if isinstance(tcoords, np.ndarray) and tcoords.size == 0:
        tcoords = None

    # Colour-per-vertex
    try:
        colour_per_vertex = vtk_to_numpy(mapper.GetLookupTable().GetTable()) / 255.
    except Exception:
        pass

    if isinstance(colour_per_vertex, np.ndarray) and colour_per_vertex.size == 0:
        colour_per_vertex = None

    return _construct_shape_type(points, trilist, tcoords, texture,
                                 colour_per_vertex)


def obj_importer(filepath, asset=None, texture_resolver=None, **kwargs):
    """Allows importing Wavefront (OBJ) files.

    Uses VTK.

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
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    obj_importer = vtk.vtkOBJReader()
    obj_importer.SetFileName(str(filepath))
    obj_importer.Update()

    # Get the output
    polydata = obj_importer.GetOutput()

    # We must have point data!
    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float)

    trilist = np.require(vtk_ensure_trilist(polydata), requirements=['C'])

    texture = None
    if texture_resolver is not None:
        texture_path = texture_resolver(filepath)
        if texture_path is not None and texture_path.exists():
            texture = mio.import_image(texture_path)

    tcoords = None
    if texture is not None:
        try:
            tcoords = vtk_to_numpy(polydata.GetPointData().GetTCoords())
        except Exception:
            pass

        if isinstance(tcoords, np.ndarray) and tcoords.size == 0:
            tcoords = None

    colour_per_vertex = None
    return _construct_shape_type(points, trilist, tcoords, texture,
                                 colour_per_vertex)


def stl_importer(filepath, asset=None, **kwargs):
    """Allows importing Stereolithography CAD (STL) files.

    Uses VTK.

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
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    stl_importer = vtk.vtkSTLReader()
    stl_importer.SetFileName(str(filepath))
    stl_importer.Update()

    # Get the output
    polydata = stl_importer.GetOutput()

    # We must have point data!
    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float)
    trilist = np.require(vtk_ensure_trilist(polydata), requirements=['C'])

    colour_per_vertex = None
    tcoords = None
    texture = None
    return _construct_shape_type(points, trilist, tcoords, texture,
                                 colour_per_vertex)


def mjson_importer(filepath, asset=None, texture_resolver=True, **kwargs):
    """
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
    with open(str(filepath), 'rb') as f:
        mesh_json = json.load(f)

    texture = None
    if texture_resolver is not None:
        texture_path = texture_resolver(filepath)
        if texture_path is not None and texture_path.exists():
            texture = mio.import_image(texture_path)

    points = mesh_json['points']
    trilist = mesh_json['trilist']
    tcoords = mesh_json.get('tcoords'),
    colour_per_vertex = mesh_json.get('colour_per_vertex')

    return _construct_shape_type(points, trilist, tcoords, texture,
                                 colour_per_vertex)
