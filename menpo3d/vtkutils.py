import numpy as np
from menpo.shape import TriMesh


def trimesh_to_vtk(trimesh):
    r"""Return a `vtkPolyData` representation of a :map:`TriMesh` instance

    Parameters
    ----------
    trimesh : :map:`TriMesh`
        The menpo :map:`TriMesh` object that needs to be converted to a
        `vtkPolyData`

    Returns
    -------
    `vtk_mesh` : `vtkPolyData`
        A VTK mesh representation of the Menpo :map:`TriMesh` data

    Raises
    ------
    ValueError:
        If the input trimesh is not 3D.
    """
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    if trimesh.n_dims != 3:
        raise ValueError('trimesh_to_vtk() only works on 3D TriMesh instances')

    mesh = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(trimesh.points, deep=1))
    mesh.SetPoints(points)

    cells = vtk.vtkCellArray()

    # Seemingly, VTK may be compiled as 32 bit or 64 bit.
    # We need to make sure that we convert the trilist to the correct dtype
    # based on this. See numpy_to_vtkIdTypeArray() for details.
    isize = vtk.vtkIdTypeArray().GetDataTypeSize()
    req_dtype = np.int32 if isize == 4 else np.int64
    cells.SetCells(trimesh.n_tris,
                   numpy_to_vtkIdTypeArray(
                       np.hstack((np.ones(trimesh.n_tris)[:, None] * 3,
                                  trimesh.trilist)).astype(req_dtype).ravel(),
                       deep=1))
    mesh.SetPolys(cells)
    return mesh


def trimesh_from_vtk(vtk_mesh):
    r"""Return a :map:`TriMesh` representation of a `vtkPolyData` instance

    Parameters
    ----------
    vtk_mesh : `vtkPolyData`
        The VTK mesh representation that needs to be converted to a
        :map:`TriMesh`

    Returns
    -------
    trimesh : :map:`TriMesh`
        A menpo :map:`TriMesh` representation of the VTK mesh data
    """
    from vtk.util.numpy_support import vtk_to_numpy
    points = vtk_to_numpy(vtk_mesh.GetPoints().GetData())
    trilist = vtk_to_numpy(vtk_mesh.GetPolys().GetData())
    return TriMesh(points, trilist=trilist.reshape([-1, 4])[:, 1:])


class VTKClosestPointLocator(object):
    r"""A callable that can be used to find the closest point on a given
    `vtkPolyData` for a query point.

    Parameters
    ----------
    vtk_mesh : `vtkPolyData`
        The VTK mesh that will be queried for finding closest points. A
        data structure will be initialized around this mesh which will enable
        efficient future lookups.
    """
    def __init__(self, vtk_mesh):
        import vtk
        cell_locator = vtk.vtkCellLocator()
        cell_locator.SetDataSet(vtk_mesh)
        cell_locator.BuildLocator()
        self.cell_locator = cell_locator

        # prepare some private properties that will be filled in for us by VTK
        self._c_point = [0., 0., 0.]
        self._cell_id = vtk.mutable(0)
        self._sub_id = vtk.mutable(0)
        self._distance = vtk.mutable(0.0)

    def __call__(self, points):
        r"""Return the nearest points on the mesh and the index of the nearest
        triangle for a collection of points. This is a lower-level algorithm
        and operates directly on a numpy array rather than an pointcloud.

        Parameters
        ----------
        points : ``(n_points, 3)`` `ndarray`
            Query points

        Returns
        -------
        `nearest_points`, `tri_indices` : ``(n_points, 3)`` `ndarray`, ``(n_points,)`` `ndarray`
            A tuple of the nearest points on the `vtkPolyData` and the triangle
            indices of the triangles that the nearest point is located inside of.
        """
        snapped_points, indices = [], []
        for p in points:
            snapped, index = self._find_single_closest_point(p)
            snapped_points.append(snapped)
            indices.append(index)

        return np.array(snapped_points), np.array(indices)

    def _find_single_closest_point(self, point):
        r"""Return the nearest point on the mesh and the index of the nearest
        triangle

        Parameters
        ----------
        point : ``(3,)`` `ndarray`
            Query point

        Returns
        -------
        `nearest_point`, `tri_index` : ``(3,)`` `ndarray`, ``int``
            A tuple of the nearest point on the `vtkPolyData` and the triangle
            index of the triangle that the nearest point is located inside of.
        """
        self.cell_locator.FindClosestPoint(point, self._c_point,
                                           self._cell_id,
                                           self._sub_id,
                                           self._distance)
        return self._c_point[:], self._cell_id.get()


def decimate_mesh(mesh, reduction=0.75, type_reduction='quadric', **kwargs):
    """
    Decimate this mesh specifying the percentage (0,1) of triangles to
    be removed

    Parameters
    ----------
    reduction: float (default: 0.75) 
               The percentage of triangles to be removed. 
               It should be in (0, 1)

    type_reduction : str (default: quadric)
                     The type of decimation as:
                         'quadric' : Quadric decimation
                         'progressive : Progressive decimation
    Returns
    -------
    mesh : :map:`TriMesh`
        A new mesh that has been decimated.
    """
    import vtk
    if type_reduction == 'quadric':
        decimate = vtk.vtkQuadricDecimation()
    elif type_reduction == 'progressive':
        decimate = vtk.vtkDecimatePro()
    else:
        raise Exception('Wrong type of reduction. It should be quadric or progressive')

    inputPolyData = trimesh_to_vtk(mesh)
    decimate.SetInputData(inputPolyData)
    decimate.SetTargetReduction(reduction)

    if kwargs.get('preserve_topology', False) and type_reduction == 'progressive':
        decimate.PreserveTopologyOn()

    decimate.Update()

    return trimesh_from_vtk(decimate.GetOutput())
