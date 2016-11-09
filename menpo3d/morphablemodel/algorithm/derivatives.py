import numpy as np


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


def d_perspective_projection_d_camera_parameters(warped_uv, camera):
    n_points, n_dims = warped_uv.shape
    assert n_dims == 3

    # Initialize derivative
    # WE DO NOT COMPUTE THE DERIVATIVE WITH RESPECT TO THE FIRST QUATERNION
    dw_dp = np.zeros((2, camera.n_parameters - 1, n_points))

    # Get z-component of warped
    z = warped_uv[:, 2]

    # Focal length
    dw_dp[:, 0] = (warped_uv[:, :2] / z[..., None]).T

    # Quaternions
    centered_warped_uv = camera.translation_transform.pseudoinverse().apply(
        warped_uv).T
    r1 = 2 * np.array([[0, 0,  0],
                       [0, 0, -1],
                       [0, 1,  0]]).dot(centered_warped_uv).T
    r2 = 2 * np.array([[ 0, 0, 1],
                       [ 0, 0, 0],
                       [-1, 0, 0]]).dot(centered_warped_uv).T
    r3 = 2 * np.array([[0, -1, 0],
                       [1,  0, 0],
                       [0,  0, 0]]).dot(centered_warped_uv).T
    # WE DO NOT COMPUTE THE DERIVATIVE WITH RESPECT TO THE FIRST QUATERNION
    # q_2
    dw_dp[:, 1] = r1[:, :2].T - r1[:, 2] * warped_uv[:, :2].T / z
    # q_3
    dw_dp[:, 2] = r2[:, :2].T - r2[:, 2] * warped_uv[:, :2].T / z
    # q_4
    dw_dp[:, 3] = r3[:, :2].T - r3[:, 2] * warped_uv[:, :2].T / z
    # constant multiplication
    dw_dp[:, 1:4] *= camera.projection_transform.focal_length / z

    # Translations
    # t_x
    dw_dp[0, 4] = camera.projection_transform.focal_length / z
    # t_y
    dw_dp[1, 5] = camera.projection_transform.focal_length / z
    # t_z
    dw_dp[:, 6] = (- camera.projection_transform.focal_length *
                   warped_uv[:, :2] / z[..., None] ** 2).T

    return dw_dp


def d_orthographic_projection_d_camera_parameters(warped_uv, camera):
    n_points, n_dims = warped_uv.shape
    assert n_dims == 3

    # Initialize derivative
    # WE DO NOT COMPUTE THE DERIVATIVE WITH RESPECT TO THE FIRST QUATERNION
    dw_dp = np.zeros((2, camera.n_parameters - 1, n_points))

    # Focal length
    dw_dp[:, 0] = warped_uv[:, :2].T

    # Quaternions
    centered_warped_uv = camera.translation_transform.pseudoinverse().apply(
        warped_uv).T
    r1 = 2 * np.array([[0, 0,  0],
                       [0, 0, -1],
                       [0, 1,  0]]).dot(centered_warped_uv).T
    r2 = 2 * np.array([[ 0, 0, 1],
                       [ 0, 0, 0],
                       [-1, 0, 0]]).dot(centered_warped_uv).T
    r3 = 2 * np.array([[0, -1, 0],
                       [1,  0, 0],
                       [0,  0, 0]]).dot(centered_warped_uv).T
    # WE DO NOT COMPUTE THE DERIVATIVE WITH RESPECT TO THE FIRST QUATERNION
    # q_2
    dw_dp[:, 1] = camera.projection_transform.focal_length * r1[:, :2].T
    # q_3
    dw_dp[:, 2] = camera.projection_transform.focal_length * r2[:, :2].T
    # q_4
    dw_dp[:, 3] = camera.projection_transform.focal_length * r3[:, :2].T

    # Translations
    # t_x
    dw_dp[0, 4] = camera.projection_transform.focal_length
    # t_y
    dw_dp[1, 5] = camera.projection_transform.focal_length
    return dw_dp
