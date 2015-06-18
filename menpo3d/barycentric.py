from collections import namedtuple
import numpy as np
from menpo.shape import PointCloud, TriMesh
from .vtkutils import trimesh_to_vtk, VTKClosestPointLocator


def barycentric_coordinates(point, a, b, c):
    T = np.hstack(((a - c)[:, None], (b - c)[:, None]))
    return np.linalg.lstsq(T, point - c)[0]


def barycentric_coordinates_for_indices(mesh, tri_index, point):
    a, b, c = mesh.points[mesh.trilist[tri_index]]
    return barycentric_coordinates(point, a, b, c)


def snap_pointcloud_to_surface(self, pointcloud):
    r"""Constrain points in a :map:`PointCloud: to lie as close as possible to
    this :map:`TriMesh`.

    Each point in the provided `pointcloud` will be projected to the surface
    of this mesh in the minimum possible distance.

    Parameters
    ----------
    pointcloud : :map:`PointCloud`
        The pointcloud that will be projected onto this mesh.

    Returns
    -------
    `snapped_pointcloud, tri_indices)` : :map:`PointCloud`, ``(n_points, )`` `ndarray`
        A tuple of the snapped :map`PointCloud` and the indices of the
        triangles involved in the snapping.
    """
    vtk_mesh = trimesh_to_vtk(self)
    locator = VTKClosestPointLocator(vtk_mesh)
    snapped_points, indices = [], []
    for p in pointcloud.points:
        snapped, index = locator(p)
        snapped_points.append(snapped)
        indices.append(index)

    return PointCloud(np.array(snapped_points), copy=False), np.array(indices)



def barycentric_points_from_contained_points(self, pointcloud, indices):
    bcoords = np.array([barycentric_coordinates_for_indices(self, i, p)
                        for i, p in zip(indices, pointcloud.points)])
    # compute the third (degenerate) coordinate as well
    bcoords_full = np.hstack((bcoords, (1 - bcoords.sum(axis=1))[:, None]))
    return bcoords_full


def barycentric_coordinates_of_pointcloud(self, pointcloud):
    p, i = self.snap_pointcloud_to_surface(pointcloud)
    bc = barycentric_points_from_contained_points(self, p, i)
    return i, bc


def project_barycentric_coordinates(self, indices, bcoords):
    t = self.points[self.trilist[indices]]
    points = np.sum(t * bcoords[..., None], axis=1)
    return PointCloud(points)


TriMesh.snap_pointcloud_to_surface = snap_pointcloud_to_surface
TriMesh.barycentric_coordinates_of_pointcloud = barycentric_coordinates_of_pointcloud
TriMesh.project_barycentric_coordinates = project_barycentric_coordinates
