import numpy as np
from menpo.transform import Homogeneous


def compute_projection_derivatives_shape_parameters(shape_pc_uv, rho_array, warped_uv,
                                                    R, shape_eigenvalues,
                                                    projection_type):
    if projection_type == 1:
        # Perspective projection derivative wrt shape parameters
        dp_dalpha = compute_pers_projection_derivatives_shape_parameters(warped_uv, shape_pc_uv,
                                                                         rho_array, R,
                                                                         shape_eigenvalues)
    else:
        # Orthographic projection derivative wrt shape parameters 
        dp_dalpha = compute_ortho_projection_derivatives_shape_parameters(shape_pc_uv, rho_array,
                                                                          R, shape_eigenvalues)

    return dp_dalpha


def compute_projection_derivatives_warp_parameters(shape_uv, warped_uv, rho_array, r_phi, r_theta, r_varphi,
                                                   projection_type):
    
    if projection_type == 1:
        # Orthographic projection derivative wrt warp parameters
        dp_drho = compute_ortho_projection_derivatives_warp_parameters(shape_uv, warped_uv, rho_array,
                                                                       r_phi, r_theta, r_varphi)
    else:
        # Perspective projection derivative wrt warp parameters
        dp_drho = compute_pers_projection_derivatives_warp_parameters(shape_uv, warped_uv.T, rho_array,
                                                                      r_phi, r_theta, r_varphi)
    return dp_drho


def compute_ortho_projection_derivatives_shape_parameters(s_pc_uv, rho, r_tot, shape_ev):
    # Precomputations
    n_parameters = np.size(s_pc_uv, 2)
    n_points = np.size(s_pc_uv, 0)
    dp_dalpha = np.zeros([2, n_parameters, n_points])

    const = rho[0]

    for k in range(n_parameters):
        dw_dalpha_k_uv = r_tot.apply((shape_ev[k] * s_pc_uv[:, :, k])).T
        dp_dalpha_k_uv = np.vstack((dw_dalpha_k_uv[0], dw_dalpha_k_uv[1]))
        dp_dalpha[:, k, :] = const*dp_dalpha_k_uv

    return dp_dalpha


def compute_pers_projection_derivatives_shape_parameters(w_uv, s_pc_uv, rho, r_tot, shape_ev):
    # Precomputations
    n_parameters = s_pc_uv.shape[2]
    n_points = s_pc_uv.shape[0]
    dp_dalpha = np.zeros([2, n_parameters, n_points])
    
    w = w_uv[:, 2]
    const = rho[0]/(w**2)

    for k in range(n_parameters):
        dw_dalpha_k_uv = r_tot.apply(shape_ev[k] * s_pc_uv[:, :, k]).T
        dp_dalpha_k_uv = np.vstack(
            (dw_dalpha_k_uv[0]*w - w_uv[:, 0]*dw_dalpha_k_uv[2],
             dw_dalpha_k_uv[1]*w - w_uv[:, 1]*dw_dalpha_k_uv[2]))
        dp_dalpha[:, k, :] = const*dp_dalpha_k_uv

    return dp_dalpha


def compute_pers_projection_derivatives_warp_parameters(s_uv, w_uv, rho, r_phi, r_theta, r_varphi):
    # Precomputations
    n_parameters = len(rho)
    n_points = np.size(s_uv, 0)
    dp_dgamma = np.zeros([2, n_parameters, n_points])

    w = w_uv[:, 2]

    const = 4*rho[0]/(w**2)

    # Compute the derivative of the perspective projection wrt focal length
    dp_dgamma[:, 0, :] = np.vstack(([0.5 * w_uv[:, 0]/w,
                                     0.5 * w_uv[:, 1]/w]))

    # Compute the derivative of the phi rotation matrix
    dr_phi_dphi = np.eye(4, 4)
    dr_phi_dphi[1:3, 1:3] = np.array([[-np.sin(rho[1]), -np.cos(rho[1])],
                                      [np.cos(rho[1]), -np.sin(rho[1])]])
    dr_phi_dphi = Homogeneous(dr_phi_dphi)

    # Compute the derivative of the warp wrt phi
    dW_dphi_uv = dr_phi_dphi.apply(r_theta.apply(r_varphi.apply(s_uv))).T

    # Compute the derivative of the projection wrt phi
    dp_dphi_uv = np.vstack(
    (dW_dphi_uv[0, :]*w - w_uv[:, 0]*dW_dphi_uv[2, :],
     dW_dphi_uv[1, :]*w - w_uv[:, 1]*dW_dphi_uv[2, :]))

    dp_dgamma[:, 1, :] = const*dp_dphi_uv

    # Compute the derivative of the theta rotation matrix
    dr_theta_dtheta = np.eye(4, 4)
    dr_theta_dtheta[:3, :3] = np.array([[-np.sin(rho[2]), 0, -np.cos(rho[2])],
                                        [0, 0, 0],
                                        [np.cos(rho[2]), 0, -np.sin(rho[2])]])
    dr_theta_dtheta = Homogeneous(dr_theta_dtheta)

    # Compute the derivative of the warp wrt theta
    dW_dtheta_uv = r_phi.apply(dr_theta_dtheta.apply(r_varphi.apply(s_uv))).T

    # Compute the derivative of the projection wrt theta
    dp_dtheta_uv = np.vstack(
        (dW_dtheta_uv[0, :]*w - w_uv[:, 0]*dW_dtheta_uv[2, :],
         dW_dtheta_uv[1, :]*w - w_uv[:, 1]*dW_dtheta_uv[2, :]))

    dp_dgamma[:, 2, :] = const*dp_dtheta_uv

    # Compute the derivative of the varphi rotation matrix
    dr_varphi_dvarphi = np.eye(4, 4)
    dr_varphi_dvarphi[:2, :2] = np.array([[-np.sin(rho[3]), -np.cos(rho[3])],
                                          [np.cos(rho[3]), -np.sin(rho[3])]])
    dr_varphi_dvarphi = Homogeneous(dr_varphi_dvarphi)

    # Compute the derivative of the warp wrt varphi
    dW_dvarphi_uv = r_phi.apply(r_theta.apply(dr_varphi_dvarphi.apply(s_uv))).T

    # Compute the derivative of the projection wrt varphi
    dp_dvarphi_uv = np.vstack(
        (dW_dvarphi_uv[0, :]*w - w_uv[:, 0]*dW_dvarphi_uv[2, :],
         dW_dvarphi_uv[1, :]*w - w_uv[:, 1]*dW_dvarphi_uv[2, :]))

    dp_dgamma[:, 3, :] = const*dp_dvarphi_uv

    # Compute the derivative of the projection function wrt tx
    dp_dtx_uv = np.vstack((w_uv[:, 2], np.zeros(n_points)))

    dp_dgamma[:, 4, :] = const*dp_dtx_uv

    # Compute the derivative of the projection function wrt ty
    dp_dty_uv = np.vstack((np.zeros(n_points), w))

    dp_dgamma[:, 5, :] = const*dp_dty_uv

    # Compute the derivative of the projection function wrt tz
    #dp_dtz_uv = np.vstack((-w_uv[:, 0], -w_uv[:, 1]))

    #dp_dgamma[:, 6, :] = const*dp_dtz_uv

    return dp_dgamma  


def compute_ortho_projection_derivatives_warp_parameters(s_uv, w_uv, rho, r_phi, r_theta, r_varphi):
    # Precomputations
    n_parameters = len(rho)
    n_points = np.size(s_uv, 0)
    dp_dgamma = np.zeros([2, n_parameters, n_points])

    const_term = np.vstack((8*rho[0]/2, 8*rho[0]/2))

    # Compute the derivative of the perspective projection wrt focal length
    dp_dgamma[:, 0, :] = np.vstack(([0.5 * w_uv[0, :].T,
                                     0.5 * w_uv[1, :].T]))

    # Compute the derivative of the phi rotation matrix
    dr_phi_dphi = np.eye(4, 4)
    dr_phi_dphi[1:3, 1:3] = np.array([[-np.sin(rho[1]), -np.cos(rho[1])],
                                      [np.cos(rho[1]), -np.sin(rho[1])]])
    dr_phi_dphi = Homogeneous(dr_phi_dphi)

    # Compute the derivative of the warp wrt phi
    dW_dphi_uv = dr_phi_dphi.apply(r_theta.apply(r_varphi.apply(s_uv))).T

    dp_dgamma[:, 1, :] = dW_dphi_uv[:2, :]*const_term

    # Compute the derivative of the theta rotation matrix
    dr_theta_dtheta = np.eye(4, 4)
    dr_theta_dtheta[:3, :3] = np.array([[-np.sin(rho[2]), 0, -np.cos(rho[2])],
                                        [0, 0, 0],
                                        [np.cos(rho[2]), 0, -np.sin(rho[2])]])
    dr_theta_dtheta = Homogeneous(dr_theta_dtheta)

    # Compute the derivative of the warp wrt theta
    dW_dtheta_uv = r_phi.apply(dr_theta_dtheta.apply(r_varphi.apply(s_uv))).T

    dp_dgamma[:, 2, :] = dW_dtheta_uv[:2, :]*const_term

    # Compute the derivative of the varphi rotation matrix
    dr_varphi_dvarphi = np.eye(4, 4)
    dr_varphi_dvarphi[:2, :2] = np.array([[-np.sin(rho[3]), -np.cos(rho[3])],
                                          [np.cos(rho[3]), -np.sin(rho[3])]])
    dr_varphi_dvarphi = Homogeneous(dr_varphi_dvarphi)

    # Compute the derivative of the warp wrt varphi
    dW_dvarphi_uv = r_phi.apply(r_theta.apply(dr_varphi_dvarphi.apply(s_uv))).T

    dp_dgamma[:, 3, :] = dW_dvarphi_uv[:2, :]*const_term

    # Compute the derivative of the projection function wrt tx
    dp_dtx_uv = np.vstack((np.ones([1, n_points]), np.zeros([1, n_points])))

    dp_dgamma[:, 4, :] = dp_dtx_uv*const_term

    # Define the derivative of the projection function wrt ty
    dp_dty_uv = np.vstack((np.zeros([1, n_points]), np.ones([1, n_points])))

    dp_dgamma[:, 5, :] = dp_dty_uv*const_term

    return dp_dgamma


def compute_texture_derivatives_texture_parameters(t_pc, texture_ev):
    # Initialization
    n_parameters = np.size(t_pc, 2)
    n_points = np.size(t_pc, 0)
    dt_dbeta = np.zeros([3, n_parameters, n_points])

    # Computations
    for k in range(n_parameters):
        # Compute the derivative of the linear texture model wrt beta k
        dt_dbeta[:, k, :] = texture_ev[k]*t_pc[:, :, k].T

    return dt_dbeta
