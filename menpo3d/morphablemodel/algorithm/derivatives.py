import numpy as np

from menpo.transform import Homogeneous


def d_perspective_projection_d_shape_parameters(shape_pc_uv, warped_uv, camera):
    """
    Calculates the derivative of the perspective projection wrt.
    to the shape parameters.

    Parameters
    ----------
    shape_pc_uv : ``(n_points, 3, n_parameters)`` `ndarray`
        The (sampled) basis of the shape model.
    warped_uv : ``(n_points, 3)`` `ndarray`
        The view transformed shape instance.
    camera: `PerspectiveCamera` object that is responsible of
     projecting the model to the image plane.

    Returns
    -------
    dw_da: the computed derivative.
    """
    n_points, n_dims, n_parameters = shape_pc_uv.shape
    assert n_dims == 3

    # Compute constant
    # (focal length divided by squared Z dimension of warped shape)
    z = warped_uv[:, 2]

    # n_dims, n_parameters, n_points
    dw_da = camera.rotation_transform.apply(shape_pc_uv.transpose(0, 2, 1)).T

    dw_da[:2] -= warped_uv[:, :2].T[:, None] * dw_da[2] / z

    return camera.projection_transform.focal_length * dw_da[:2] / z


def d_orthographic_projection_d_shape_parameters(shape_pc_uv, camera):
    """
    Calculates the derivative of the orthographic projection wrt.
    to the shape parameters.

    Parameters
    ----------
    shape_pc_uv : ``(n_points, 3, n_parameters)`` `ndarray`
        The (sampled) basis of the shape model.
    camera: `PerspectiveCamera` object that is responsible of
     projecting the model to the image plane.

    Returns
    -------
    dw_da: the computed derivative.
    """
    n_points, n_dims, n_parameters = shape_pc_uv.shape
    assert n_dims == 3

    # n_dims, n_parameters, n_points
    dp_da_uv = camera.rotation_transform.apply(shape_pc_uv.transpose(0, 2, 1)).T

    return camera.projection_transform.focal_length * dp_da_uv[:2]


def d_orthographic_projection_d_warp_parameters(shape_uv, warped_uv, camera):

    pt = camera.rotation_transform.apply(shape_uv).T

    r1 = 2 * np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]]).dot(pt)
    r2 = 2 * np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]).dot(pt)
    r3 = 2 * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]).dot(pt)
    zeros = np.zeros_like(r1)

    tx = np.array([zeros[0] + 1, zeros[0], zeros[0]])
    ty = np.array([zeros[0], zeros[0] + 1, zeros[0]])
    tz = zeros

    return np.array([zeros, zeros, r1, r2, r3, tx, ty, tz]).transpose(1, 0, 2)[:2]


def d_perspective_projection_d_warp_parameters(shape_uv, warped_shape, camera):
    pt = camera.rotation_transform.apply(shape_uv).T

    z = warped_shape[:, 2]

    r1 = 2 * np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]]).dot(pt)
    r2 = 2 * np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]).dot(pt)
    r3 = 2 * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]).dot(pt)
    zeros = np.zeros_like(r1)

    tx = np.array([zeros[0] + 1, zeros[0], zeros[0]])
    ty = np.array([zeros[0], zeros[0] + 1, zeros[0]])
    tz = zeros

    grad = np.array([zeros, zeros, r1, r2, r3, tx, ty, tz]).transpose(1, 0, 2)

    grad[:2] -= warped_shape[:, :2].T[:, None] * grad[2] / z

    return camera.projection_transform.focal_length * grad[:2]