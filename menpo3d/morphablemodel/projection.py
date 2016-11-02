from __future__ import division
import numpy as np

from menpo.transform import Homogeneous
from menpo3d.camera import PerspectiveCamera


def compute_rotation_matrices(phi, theta, varphi):
    r"""
    Compute the rotation matrix of a perspective camera model.

    Parameters
    ----------
    phi : `float`
        Angle over X axis.
    theta : `float`
        Angle over Y axis.
    varphi : `float`
        Angle over Z axis.

    Returns
    -------
    rotation_phi : `menpo.transform.Homogeneous`
        The rotation matrix of wrt the X axis.
    rotation_theta : `menpo.transform.Homogeneous`
        The rotation matrix of wrt the Y axis.
    rotation_varphi : `menpo.transform.Homogeneous`
        The rotation matrix of wrt the Z axis.
    rotation : `menpo.transform.Homogeneous`
        The rotation matrix of the perspective camera model.
    """
    # Create R_phi
    rot_phi = np.eye(4)
    rot_phi[1:3, 1:3] = [[ np.cos(phi), np.sin(phi)],
                         [-np.sin(phi), np.cos(phi)]]
    rot_phi = Homogeneous(rot_phi)

    # Create R_theta
    rot_theta = np.eye(4)
    rot_theta[:3, :3] = [[np.cos(theta), 0, -np.sin(theta)],
                         [0,             1,              0],
                         [np.sin(theta), 0,  np.cos(theta)]]
    rot_theta = Homogeneous(rot_theta)

    # Create R_varphi
    rot_varphi = np.eye(4)
    rot_varphi[:2, :2] = [[ np.cos(varphi), np.sin(varphi)],
                          [-np.sin(varphi), np.cos(varphi)]]
    rot_varphi = Homogeneous(rot_varphi)

    # Compose transforms
    rotation = np.dot(np.dot(rot_varphi.h_matrix, rot_theta.h_matrix),
                      rot_phi.h_matrix)
    rotation = Homogeneous(rotation)
    # rotation = rot_theta.compose_after(rot_phi)
    # rotation.compose_before_inplace(rot_varphi)

    return rot_phi, rot_theta, rot_varphi, rotation


def weak_projection_matrix(width, height, mesh_camera_space):
    r"""
    Function that creates a weak projection matrxi.

    Parameters
    ----------
    width : `int`
        The image plane width.
    height : `int`
        The image plane height.
    mesh_camera_space : ``menpo.shape.TriMesh`` or `menpo.shape.ColouredTriMesh`
        The camera mesh object.

    Returns
    -------
    projection_transform : `menpo.transform.Homogeneous`
        The weak projection transform object.
    """
    # Identify how far and near the mesh is the camera. We want to ensure
    # that the near and far planes are set so that the whole mesh object is
    # visible.
    near_bounds, far_bounds = mesh_camera_space.bounds()

    # Rather than just use the bounds, we add 10% at each direction
    # just to avoid any numerical errors.
    average_plane = (near_bounds[-1] + far_bounds[-1]) * 0.5
    padded_range = mesh_camera_space.range()[-1] * 1.1
    near_plane = average_plane - padded_range
    far_plane = average_plane + padded_range

    # Create weak projection matrix
    plane_sum = far_plane + near_plane
    plane_prod = far_plane * near_plane
    denom = far_plane - near_plane
    max_d = max(width, height)
    proj = np.array([[2.0 * max_d / width, 0,                    0,                    0],
                     [0,                   2.0 * max_d / height, 0,                    0],
                     [0,                   0,                    -(plane_sum) / denom, (-2.0 * plane_prod) / denom],
                     [0,                   0,                    -1,                   0]])

    return Homogeneous(proj)


def compute_view_projection_transforms(image, mesh, image_pointcloud,
                                       mesh_pointcloud,
                                       distortion_coeffs=None):
    r"""
    Function that estimates the camera pose (view and projection transforms)
    given a mesh, an image, the 3D sparse pointcloud on the mesh and its
    2D projection on the image plane. Note that the function empoloys a weak
    projection transform.

    Parameters
    ----------
    image : `menpo.image.Image`
        The input image object.
    mesh : ``menpo.shape.TriMesh`` or `menpo.shape.ColouredTriMesh`
        The input mesh object.
    image_pointcloud : `menpo.shape.PointCloud` or subclass
        The 2D projection of the 3D sparse shape of the mesh on the image
        coordinate space.
    mesh_pointcloud : `menpo.shape.PointCloud` or subclass
        The 3D sparse shape on the mesh coordinate space.
    distortion_coeffs : ``(4,)`` or ``(5,)`` or ``(8,)`` `ndarray` or ``None``, optional
        The distortion coefficients of 4, 5, or 8 elements. If ``None``,
        then the distortion coefficients are set to zero.

    Returns
    -------
    view_transform : `menpo.transform.Homogeneous`
        The view transform object.
    projection_transform : `menpo.transform.Homogeneous`
        The projection transform object.
    rotation_transform : `menpo.transform.Rotation`
        The rotation transform object.
    """
    camera = PerspectiveCamera.init_from_2d_projected_shape(
        mesh_pointcloud, image_pointcloud, image.shape,
        distortion_coeffs=distortion_coeffs)

    r = camera.rotation_transform
    t = camera.translation_transform

    # This is equivalent to rotating y by 180, then rotating z by 180.
    # This is also equivalent to glulookat to lookat the origin
    #   gluLookAt(0,0,0,0,0,1,0,-1,0);
    axes_flip_matrix = np.eye(4)
    axes_flip_matrix[1, 1] = -1
    axes_flip_matrix[2, 2] = -1
    axes_flip_t = Homogeneous(axes_flip_matrix)

    # Create view transform and projection transform
    view_t_flipped = r.compose_before(t)
    view_t = view_t_flipped.compose_before(axes_flip_t)
    proj_t = weak_projection_matrix(image.width, image.height,
                                    view_t_flipped.apply(mesh))
    return view_t, proj_t, r


def get_camera_parameters(projection_t, view_t):
    r"""
    Function that returns a vector of camera parameters, given the camera
    view and projection matrices.

    Parameters
    ----------
    projection_t : `menpo.transform.Homogeneous`
        The projection transform object.
    view_t : `menpo.transform.Homogeneous`
        The view transform object.

    Returns
    -------
    params : ``(6,)`` `ndarray`
        The camera parameters:
        focal length, varphi, theta, phi, translation X, translation Y
    """
    # There are siz parameters:
    # focal length, varphi, theta, phi, translation X, translation Y
    params = np.zeros(6)

    # PROJECTION MATRIX PARAMETERS
    # The focal length is the first diagonal element of the projection matrix
    # For the moment this is not optimised.
    params[0] = projection_t.h_matrix[0, 0]

    # VIEW MATRIX PARAMETERS
    # Euler angles
    # For the case of cos(theta) != 0, we have two triplets of Euler angles
    # we will only give one of the two solutions
    if view_t.h_matrix[2, 0] != 1 or view_t.h_matrix[2, 0] != -1:
        theta = np.pi + np.arcsin(view_t.h_matrix[2, 0])
        varphi = np.arctan2(view_t.h_matrix[2, 1] / np.cos(theta),
                            view_t.h_matrix[2, 2] / np.cos(theta))
        phi = np.arctan2(view_t.h_matrix[1, 0] / np.cos(theta),
                         view_t.h_matrix[0, 0] / np.cos(theta))
        params[1] = varphi
        params[2] = theta
        params[3] = phi

    # Translations
    params[4] = - view_t.h_matrix[0, 3]  # tw x
    params[5] = - view_t.h_matrix[1, 3]  # tw y

    return params
