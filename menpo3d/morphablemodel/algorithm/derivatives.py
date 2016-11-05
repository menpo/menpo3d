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


def d_orthographic_projection_d_warp_parameters(shape_uv, warped_uv,
                                                warp_parameters, r_phi,
                                                r_theta, r_varphi):
    # Initialize
    n_parameters = len(warp_parameters)
    n_points = shape_uv.shape[0]
    dp_dr = np.zeros((2, n_parameters, n_points))

    focal_length, qw, qx, qy, qz, tx, ty = warp_parameters

    # Compute constant
    const = np.vstack((8 * focal_length / 2,
                       8 * focal_length / 2))

    # DERIVATIVE WRT FOCAL LENGTH
    dp_dr[:, 0, :] = np.vstack(([0.5 * warped_uv[0, :].T,
                                 0.5 * warped_uv[1, :].T]))

    # DERIVATIVE WRT PHI
    # Derivative of phi rotation wrt angle
    dr_phi_dphi = np.eye(4, 4)
    phi = warp_parameters[1]
    dr_phi_dphi[1:3, 1:3] = np.array([[-np.sin(phi), np.cos(phi)],
                                      [-np.cos(phi), -np.sin(phi)]])
    dr_phi_dphi = Homogeneous(dr_phi_dphi)

    # Derivative of warped shape wrt phi
    dW_dphi_uv = dr_phi_dphi.apply(r_theta.apply(r_varphi.apply(shape_uv))).T
    dp_dr[:, 1, :] = dW_dphi_uv[:2, :] * const

    # DERIVATIVE WRT THETA
    # Derivative of theta rotation  wrt theta
    dr_theta_dtheta = np.eye(4, 4)
    theta = warp_parameters[2]
    dr_theta_dtheta[:3, :3] = np.array([[-np.sin(theta), 0, -np.cos(theta)],
                                        [             0, 0,              0],
                                        [ np.cos(theta), 0, -np.sin(theta)]])
    dr_theta_dtheta = Homogeneous(dr_theta_dtheta)

    # Derivative of warped shape wrt theta
    dW_dtheta_uv = r_phi.apply(
        dr_theta_dtheta.apply(r_varphi.apply(shape_uv))).T
    dp_dr[:, 2, :] = dW_dtheta_uv[:2, :] * const

    # DERIVATIVE WRT VARPHI
    # Derivative of varphi rotation wrt angle
    dr_varphi_dvarphi = np.eye(4, 4)
    varphi = warp_parameters[3]
    dr_varphi_dvarphi[:2, :2] = np.array([[-np.sin(varphi), -np.cos(varphi)],
                                          [ np.cos(varphi), -np.sin(varphi)]])
    dr_varphi_dvarphi = Homogeneous(dr_varphi_dvarphi)

    # Derivative of warped shape wrt varphi
    dW_dvarphi_uv = r_phi.apply(
        r_theta.apply(dr_varphi_dvarphi.apply(shape_uv))).T
    dp_dr[:, 3, :] = dW_dvarphi_uv[:2, :] * const

    # DERIVATIVE WRT TRANSLATION X
    dp_dtx_uv = np.vstack((np.ones([1, n_points]), np.zeros([1, n_points])))
    dp_dr[:, 4, :] = dp_dtx_uv * const

    # DERIVATIVE WRT TRANSLATION Y
    dp_dty_uv = np.vstack((np.zeros([1, n_points]), np.ones([1, n_points])))
    dp_dr[:, 5, :] = dp_dty_uv * const

    return dp_dr


def d_perspective_projection_d_warp_parameters(shape_uv, warped_uv,
                                               warp_parameters, r_phi,
                                               r_theta, r_varphi):
    # Compute constant
    # w = warped_uv[:, 2] or 1
    const = 4 * warp_parameters[0] / (w ** 2)

    # DERIVATIVE WRT FOCAL LENGTH
    dp_dr[:, 0, :] = np.vstack(([0.5 * warped_uv[:, 0] / w,
                                 0.5 * warped_uv[:, 1] / w]))

    # DERIVATIVE WRT PHI
    # Derivative of phi rotation wrt angle
    dr_phi_dphi = np.eye(4, 4)
    phi = warp_parameters[1]
    dr_phi_dphi[1:3, 1:3] = np.array([[-np.sin(phi),  np.cos(phi)],
                                      [-np.cos(phi), -np.sin(phi)]])
    dr_phi_dphi = Homogeneous(dr_phi_dphi)

    # Derivative of warped shape wrt phi
    dW_dphi_uv = dr_phi_dphi.apply(r_theta.apply(r_varphi.apply(shape_uv))).T

    # Derivative of projection wrt phi
    dp_dphi_uv = np.vstack(
        (dW_dphi_uv[0, :] * w - warped_uv[:, 0] * dW_dphi_uv[2, :],
         dW_dphi_uv[1, :] * w - warped_uv[:, 1] * dW_dphi_uv[2, :]))
    dp_dr[:, 1, :] = const * dp_dphi_uv

    # DERIVATIVE WRT THETA
    # Derivative of theta rotation wrt angle
    dr_theta_dtheta = np.eye(4, 4)
    theta = warp_parameters[2]
    dr_theta_dtheta[:3, :3] = np.array([[-np.sin(theta), 0, -np.cos(theta)],
                                        [             0, 0,              0],
                                        [ np.cos(theta), 0, -np.sin(theta)]])
    dr_theta_dtheta = Homogeneous(dr_theta_dtheta)

    # Derivative of warped shape wrt theta
    dW_dtheta_uv = r_phi.apply(
        dr_theta_dtheta.apply(r_varphi.apply(shape_uv))).T

    # Derivative of projection wrt theta
    dp_dtheta_uv = np.vstack(
        (dW_dtheta_uv[0, :] * w - warped_uv[:, 0] * dW_dtheta_uv[2, :],
         dW_dtheta_uv[1, :] * w - warped_uv[:, 1] * dW_dtheta_uv[2, :]))
    dp_dr[:, 2, :] = const * dp_dtheta_uv

    # DERIVATIVE WRT VARPHI
    # Derivative of varphi rotation wrt angle
    dr_varphi_dvarphi = np.eye(4, 4)
    varphi = warp_parameters[3]
    dr_varphi_dvarphi[:2, :2] = np.array([[-np.sin(varphi), -np.cos(varphi)],
                                          [ np.cos(varphi), -np.sin(varphi)]])
    dr_varphi_dvarphi = Homogeneous(dr_varphi_dvarphi)

    # Derivative of warped shape wrt varphi
    dW_dvarphi_uv = r_phi.apply(
        r_theta.apply(dr_varphi_dvarphi.apply(shape_uv))).T

    # Derivative of projection wrt varphi
    dp_dvarphi_uv = np.vstack(
        (dW_dvarphi_uv[0, :] * w - warped_uv[:, 0] * dW_dvarphi_uv[2, :],
         dW_dvarphi_uv[1, :] * w - warped_uv[:, 1] * dW_dvarphi_uv[2, :]))
    dp_dr[:, 3, :] = const * dp_dvarphi_uv

    # DERIVATIVE WRT TRANSLATION X
    dp_dtx_uv = np.vstack((warped_uv[:, 2], np.zeros(n_points)))
    dp_dr[:, 4, :] = const * dp_dtx_uv

    # DERIVATIVE WRT TRANSLATION Y
    dp_dty_uv = np.vstack((np.zeros(n_points), w))
    dp_dr[:, 5, :] = const * dp_dty_uv

    return dp_dr
