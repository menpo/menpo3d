import numpy as np
from menpo.shape import PointCloud, TriMesh, ColouredTriMesh, TexturedTriMesh
from .vtkutils import trimesh_to_vtk, VTKClosestPointLocator


def barycentric_coordinates(point, a, b, c):
    T = np.hstack(((a - c)[:, None], (b - c)[:, None]))
    return np.linalg.lstsq(T, point - c)[0]


def barycentric_coordinates_for_indices(mesh, tri_index, point):
    a, b, c = mesh.points[mesh.trilist[tri_index]]
    return barycentric_coordinates(point, a, b, c)


def barycentric_points_from_contained_points(self, pointcloud, tri_index):
    # http://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    abc_per_tri = self.points[self.trilist[tri_index]]

    a = abc_per_tri[:, 0, :]
    b = abc_per_tri[:, 1, :]
    c = abc_per_tri[:, 2, :]
    p = pointcloud.points

    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = (v0 * v0).sum(axis=1)
    d01 = (v0 * v1).sum(axis=1)
    d11 = (v1 * v1).sum(axis=1)
    d20 = (v2 * v0).sum(axis=1)
    d21 = (v2 * v1).sum(axis=1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w

    return np.vstack([u, v, w]).T


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
    snapped_pointcloud, tri_indices : :map:`PointCloud`, ``(n_points, )`` `ndarray`
        A tuple of the snapped :map`PointCloud` and the indices of the
        triangles involved in the snapping.
    """
    vtk_mesh = trimesh_to_vtk(self)
    locator = VTKClosestPointLocator(vtk_mesh)
    snapped_points, indices = locator(pointcloud.points)
    return PointCloud(snapped_points, copy=False), indices


def barycentric_coordinates_of_pointcloud(self, pointcloud):
    r"""Return the barycentric coordinates of points in a :map:`PointCloud:
    that have been constrained to lie as close as possible to this
    :map:`TriMesh`.

    Each point in the provided `pointcloud` will be projected to the surface
    of this mesh in the minimum possible distance. The barycentric coordinates
    and the triangle index of each point will be returned.

    Parameters
    ----------
    pointcloud : :map:`PointCloud`
        The pointcloud that will be projected onto this mesh.

    Returns
    -------
    bcoords, tri_indices : ``(n_points, 3)`` `ndarray`, ``(n_points, )`` `ndarray`
        A tuple of the barycentric coordinates and the indices of the
        triangles involved for each snapped point.
    """
    p, i = self.snap_pointcloud_to_surface(pointcloud)
    bc = barycentric_points_from_contained_points(self, p, i)
    return bc, i


def barycentric_coordinate_interpolation(self, per_vertex_interpolant,
                                         bcoords, tri_indices):
    r"""Interpolate some per-vertex value on this mesh using barycentric coordinates.

    Parameters
    ----------
    per_vertex_interpolant : ``(n_points, k)`` `ndarray`
        Any array of per-vertex data. This will be linearly blended over every
         triangle referenced in ``tri_indices`` with the relevent weights
         given in ``bcoords``.
    bcoords : ``(n_samples, 3)`` `ndarray`
        The barycentric coordinates that will be used in the projection
    tri_indices : ``(n_samples, )`` `ndarray`
        The index of the triangle that the above ``bcoords`` correspond to.

    Returns
    -------
    `ndarray` : ``(n_samples, k)``
        The interpolated values of ``per_vertex_interpolant``.
    """
    shape = per_vertex_interpolant.shape
    if not len(shape) == 2 or shape[0] != self.n_points:
        raise ValueError("per_vertex_interpolant must be of shape (n_points, k)")
    t = per_vertex_interpolant[self.trilist[tri_indices]]
    return np.sum(t * bcoords[..., None], axis=1)


def project_barycentric_coordinates(self, bcoords, tri_indices):
    r"""Projects a set of barycentric coordinates onto this mesh surface,
    returning a :map:`PointCloud`.

    Parameters
    ----------
    bcoords: ``(n_samples, 3)`` `ndarray`
        The barycentric coordinates that will be used in the projection
    tri_indices : ``(n_samples, )`` `ndarray`
        The index of the triangle that the above ``bcoords`` correspond to.

    Returns
    -------
    pointcloud : :map:`PointCloud`
        A :map:`PointCloud` representation of the provided barycentric
        coordinates on this :map:`TriMesh`
    """
    # Interpolate self.points using the more generic
    # barycentric_coordinate_interpolation method
    interped_points = self.barycentric_coordinate_interpolation(self.points,
                                                                bcoords,
                                                                tri_indices)
    return PointCloud(interped_points, copy=False)


def sample_texture_with_barycentric_coordinates_colour(self, bcoords,
                                                       tri_indices):
    return self.barycentric_coordinate_interpolation(
        self.colours, bcoords, tri_indices)


def sample_texture_with_barycentric_coordinates_texture(self, bcoords,
                                                        tri_indices, order=1):
    sample_points = self.barycentric_coordinate_interpolation(
            self.tcoords_pixel_scaled().points, bcoords, tri_indices)
    texture = self.texture
    if hasattr(texture, 'as_unmasked'):
        # TODO this as_unmasked should not be needed, but it is (we can fall
        # off the texture at bcoords). This means the sampled texture contains
        # wrong (black) values.
        texture = texture.as_unmasked(copy=False)
    return texture.sample(sample_points, order=order).T


TriMesh.snap_pointcloud_to_surface = snap_pointcloud_to_surface
TriMesh.barycentric_coordinates_of_pointcloud = barycentric_coordinates_of_pointcloud
TriMesh.barycentric_coordinate_interpolation = barycentric_coordinate_interpolation
TriMesh.project_barycentric_coordinates = project_barycentric_coordinates
ColouredTriMesh.sample_texture_with_barycentric_coordinates = sample_texture_with_barycentric_coordinates_colour
TexturedTriMesh.sample_texture_with_barycentric_coordinates = sample_texture_with_barycentric_coordinates_texture
