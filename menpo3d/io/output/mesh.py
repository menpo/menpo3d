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
        file_handle.write('v {} {} {}\n'.format(*v).encode('utf-8'))
    file_handle.write(b'\n')
    if isinstance(mesh, TexturedTriMesh):
        for tc in mesh.tcoords.points:
            file_handle.write('vt {} {}\n'.format(*tc).encode('utf-8'))
        file_handle.write(b'\n')
        # triangulation of points and tcoords is identical
        for t in (mesh.trilist + 1):
            file_handle.write('f {0}/{0} {1}/{1} {2}/{2}\n'.format(*t).encode('utf-8'))
    else:
        # no tcoords - so triangulation is straight forward
        for t in (mesh.trilist + 1):
            file_handle.write('f {} {} {}\n'.format(*t).encode('utf-8'))


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
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    file_path = _enforce_only_paths_supported(file_path, 'PLY')

    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(mesh.points))
    polydata.SetPoints(points)

    cells = vtk.vtkCellArray()
    counts = np.empty((mesh.trilist.shape[0], 1), dtype=np.int)
    counts.fill(3)
    tris = np.concatenate((counts, mesh.trilist), axis=1)
    cells.SetCells(mesh.trilist.shape[0], numpy_to_vtkIdTypeArray(tris))
    polydata.SetPolys(cells)

    if isinstance(mesh, TexturedTriMesh):
        pointdata = polydata.GetPointData()
        pointdata.SetTCoords(numpy_to_vtk(mesh.tcoords.points))

    ply_writer = vtk.vtkPLYWriter()
    ply_writer.SetFileName(str(file_path))
    ply_writer.SetInputData(polydata)
    if not binary:
        ply_writer.SetFileTypeToASCII()
    else:
        ply_writer.SetFileTypeToBinary()
    ply_writer.Update()
    ply_writer.Write()
