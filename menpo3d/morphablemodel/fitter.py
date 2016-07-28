import numpy as np
import numpy.matlib
import menpo.io as mio
from menpo.image import Image
from menpofit.fitter import MultiScaleParametricFitter
from lk import SimultaneousForwardAdditive
from model import Model
import scipy.io as spio


class MorphableModelFitter(object):
    r"""
    Abstract class for defining an 3DMM fitter.

    """
    def __init__(self, mm):
        self._model = mm

    @property
    def mm(self):
        r"""
        The 3DMM model.

        """
        return self._model

    def _precompute(self):
        return homogenize(self._model.shape_pc)

    def _compute_shape(self, alpha_c):

        r"""

        Parameters
        ----------
        alpha_c

        Returns
        -------

        """
        mm = self._model
        shape = model_to_object(alpha_c, mm.shape_mean, mm.shape_pc, mm.shape_ev)
        shape = homogenize(shape)
        # Subtracting the mean from the shape matrix
        # to express the shape as linear combination of principal components
        s_m = np.matlib.repmat(np.mean(shape[0:3, :], 1), 1, np.size(shape, 1))
        shape[0:3, :, 0] -= s_m
        return shape

    # TODO
    def fit(self, anchors_pf):

        r"""

        Parameters
        ----------
        anchors_pf

        Returns
        -------

        """

        # Define control parameters
        ctrl_params = control_parameters(pt=1, vb=True)

        # Define standard fitting parameters
        std_fit_params = standard_fitting_parameters(ctrl_params)

        # Define fitting parameters
        fit_params = fitting_parameters(20, -1, -1, [0, 1, 4, 5], -1, -1, 10 ** -3)

        alpha_c = std_fit_params['alpha_array']  # Shape parameters
        beta_c = std_fit_params['beta_array']  # Texture parameters
        rho_c = std_fit_params['rho_array']  # Shape transformation parameters
        iota_c = std_fit_params['iota_array']  # Rendering parameters

        [n_alphas, n_betas, n_iotas, n_rhos] = [0]*4
        if fit_params['n_alphas'] != -1:
            n_alphas = len(fit_params['n_alphas'])
        if fit_params['n_betas'] != -1:
            n_betas = len(fit_params['n_betas'])
        if fit_params['n_rhos'] != -1:
            n_rhos = len(fit_params['n_rhos'])
        if fit_params['n_iotas'] != -1:
            n_iotas = len(fit_params['n_iotas'])

        n_parameters = n_alphas + n_betas + n_iotas + n_rhos

        # Precomputations
        s_pc = self._precompute()

        # Simultaneous Forwards Additive Algorithm
        for i in xrange(fit_params['max_iters']):
            # Alignment on anchor points only

            # Compute shape and texture
            shape = self._compute_shape(alpha_c)

            # Compute warp and projection matrices
            [r_phi, r_theta, r_varphi, rot, view_matrix, projection_matrix] = \
                compute_warp_and_projection_matrices(rho_c, ctrl_params['projection_type'])

            # Import anchor points
            [img, resolution, anchor_points, model_triangles] = import_anchor_points(anchors_pf)

            # Compute anchor points warp and projection
            anchor_array = self._model.triangle_array[:, model_triangles]
            warped = np.dot(view_matrix, shape[:, :, 0])
            projection = project(warped, projection_matrix, ctrl_params['projection_type'])

            # [uv_anchor, yx_anchor] = compute_anchor_points_projection(anchor_array, projection)
            uv_anchor = np.vstack((map(int, np.arange(0, 10)), np.matlib.repmat(0.333, 3, 10)))
            yx_anchor = np.array([[241, 243, 245, 241, 298, 372, 371, 223, 223, 462],
                                  [181, 233, 299, 351, 268, 216, 315, 152, 378, 267],
                                  [92401, 119027, 179441, 137002, 110452, 161139, 77535, 193247, 136654]])

            # Compute anchor points error
            a = np.array(yx_anchor[:2][:].tolist())
            b = anchor_points[:2][:]
            anchor_error_pixel = compute_anchor_points_error(b, a)
            anchor_error = np.zeros(anchor_error_pixel.shape)

            for j in xrange(2):
                anchor_error[j] = ["{:.5f}".format(x*(2/resolution[j])) for x in anchor_error_pixel[j]]

            # Compute the derivatives
            # Shape sampling
            s_uv_anchor = sample_object_at_uv(shape, anchor_array, uv_anchor)
            # Warp sampling
            w_uv_anchor = sample_object_at_uv(warped, anchor_array, uv_anchor)
            # Shape principal components sampling
            s_pc_uv_anchor = sample_object_at_uv(s_pc, anchor_array, uv_anchor)

            dp_dalpha = []
            dp_dbeta = []
            dp_diota = []
            dp_drho = []

            if n_alphas > 0:
                dp_dalpha = compute_projection_derivatives_shape_parameters(s_uv_anchor, w_uv_anchor, rho_c,
                                                                            rot, s_pc_uv_anchor, ctrl_params,
                                                                            self._model.shape_ev)

            if n_rhos > 0:
                dp_drho = compute_projection_derivatives_warp_parameters(s_uv_anchor, w_uv_anchor, rho_c,
                                                                         r_phi, r_theta, r_varphi, ctrl_params)

            # Compute steepest descent matrix and hessian
            # print map(lambda x: -x, dp_dalpha)
            # sd_anchor = np.hstack((-dp_dalpha, -dp_drho, dp_dbeta, dp_diota))
            # print sd_anchor
            # return
            # h_anchor = hessian(sd_anchor)
            # sd_error_product_anchor = compute_sd_error_product(sd_anchor, anchor_error_pixel)
            #
            # # Visualize
            # visualize(image, anchors, yx_anchor)
            #
            # # Update parameters
            # delta_sigma = update_parameters(h_anchor, sd_error_product_anchor)
            # [alpha_c, beta_c, rho_c, iota_c] = update(delta_sigma, fit_params)
            #
            # # Check for convergence
            # if np.linalg.norm(delta_sigma) < fit_params['cvg_thresh']:
            #     break

        # Save final parameters
        # fit_params['n_alphas'] = alpha_c
        # fit_params['n_betas'] = beta_c
        # fit_params['n_rhos'] = rho_c
        # fit_params['n_iotas'] = iota_c


def hessian(sd):
    # Computes the hessian as defined in the Lucas Kanade Algorithm
    n_channels = np.size(sd[:, 0, 1])
    n_params = np.size(sd[0, :, 0])
    h = np.zeros((n_params, n_params))
    sd = np.transpose(sd, [2, 1, 0])
    for i in xrange(n_channels):
        h_i = np.dot(np.transpose(sd[:, :, i]), sd[:, :, i])
        h += h_i
    return h


def rho_from_view_projection_matrices(proj_t, view_t):

    r"""

    Function which computes the rho array from the projection and view matrices

    Parameters
    ----------
    proj_t : projection matrix
    view_t : view matrix

    Returns
    -------
    rho : array of rendering parameters

    """

    rho = np.zeros(6)

    # PROJECTION MATRIX PARAMETERS
    # The focal length is the first diagonal element of the projection matrix
    # At the moment this is not optimised
    rho[0] = proj_t[0, 0]

    # VIEW MATRIX PARAMETERS
    # Euler angles
    # For the case of cos(theta) != 0, we have two triplets of Euler angles
    # we will only give one of the two solutions
    if view_t[2, 0] != 0:
        theta = -np.arcsin(view_t[1, 0])
        phi = np.arctan2(view_t[1, 1], view_t[1, 1])
        varphi = np.arctan2(view_t[1, 0], view_t[0, 0])
        rho[1] = phi
        rho[2] = theta
        rho[3] = varphi

    # Translations
    rho[4] = -view_t[0, 3]  # tw x
    rho[5] = -view_t[1, 3]  # tw y

    return rho


def standard_fitting_parameters(params):

    r"""

    Parameters
    ----------
    params

    Returns
    -------

    """

    if params['projection_type'] == 0:
        # Define projection parameters
        # focal length, phi, theta, varphi, tw_x, tw_y, (tw_z)
        rho_array = np.zeros(6)
        rho_array[0] = 1.2  # focal length
    else:
        rho_array = np.zeros(7)
        rho_array[0] = 30  # focal length
    rho_array[4] = 0.045  # tw_x
    rho_array[5] = 0.306  # tw_y

    # Define illumination and color correction parameters:
    # in order: gamma (red, green, blue), contrast, offset (red, green, blue),
    # ambiant light intensity (red, green, blue),
    # Directional light intensity (red, green, blue), directional light direction (theta, phi), Ks, v
    iota_array = [1, 1, 1, 1, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0., 0., 30, 40]

    # Define shape parameters
    alpha_array = np.zeros(101)

    # Define texture parameters
    beta_array = np.zeros(101)

    std_fit_params = {
        'rho_array': rho_array,
        'iota_array': iota_array,
        'alpha_array': alpha_array,
        'beta_array': beta_array
    }
    return std_fit_params


def fitting_parameters(max_iters, n_points, n_alphas, n_rhos, n_betas, n_iotas,
                       cvg_thresh):
    fit_params = {
        'max_iters': max_iters,  # number of iteration: -1 : until convergence
        'n_points': n_points,  # number of points
        # parameters
        'n_alphas': n_alphas,
        'n_rhos': n_rhos,
        'n_betas': n_betas,
        'n_iotas': n_iotas,
        'cvg_thresh': cvg_thresh,
    }
    return fit_params


def control_parameters(pt=0, vb=False, vis=False):
    ctrl_params = {
        'projection_type': pt,  # 0: weak perspective, 1: perspective
        'verbose': vb,  # Console information: False:  off, True: on
        'visualize': vis  # Visualize the fitting
    }
    return ctrl_params


def compute_warp_and_projection_matrices(rho_array, projection_type):

    # 3D Rotation
    r_phi = np.eye(4)

    r_phi[1:3, 1:3] = np.array([[np.cos(rho_array[1]), -np.sin(rho_array[1])],
                                [np.sin(rho_array[1]), np.cos(rho_array[1])]])
    r_theta = np.eye(4)
    r_theta[0:3, 0:3] = np.array([[np.cos(rho_array[2]), 0, -np.sin(rho_array[2])],
                                  [0, 1, 0],
                                  [np.sin(rho_array[2]), 0, np.cos(rho_array[2])]])
    r_varphi = np.eye(4)
    r_varphi[0:2, 0:2] = np.array([[np.cos(rho_array[3]), -np.sin(rho_array[3])],
                                   [np.sin(rho_array[3]), np.cos(rho_array[3])]])

    rot_total = np.dot(np.dot(r_varphi, r_theta), r_phi)

    # 3D Translation
    if projection_type == 0:
        tw = [rho_array[4], rho_array[5], 0, 1]
    else:
        tw = [rho_array[4], rho_array[5], rho_array[6], 1]

    to = [0, 0, 0, 1]
    tc = [0, 0, -20, 1]

    translation_tw = np.eye(4)
    translation_tw[:, 3] = np.dot(rot_total, to) + tw - tc

    # View matrix and projection matrix calculations
    view_matrix = np.dot(translation_tw, rot_total)

    far = 50
    near = rho_array[0]
    u_max = 2
    u_min = -2
    v_max = 2
    v_min = -2

    # Projection matrix computation as in graphics course
    m_persla = np.eye(4)
    # m_persla[0, 2] = (u_max + u_min) / (u_max - u_min)
    # m_persla[1, 2] = (v_max + v_min) / (v_max - v_min)

    m_perslb = np.eye(4)
    m_perslb[0, 0] = 2 * rho_array[0] / (u_max - u_min)
    m_perslb[1, 1] = 2 * rho_array[0] / (v_max - v_min)

    if projection_type == 0:
        m_pers2 = np.eye(4)
        m_pers2[2, 2] = 2 * rho_array[0] / (far - near)
        m_pers2[2, 3] = -(far + near) / (far - near)
    else:
        m_pers2 = np.eye(4)
        m_pers2[2, 2] = (far + near) / (far - near)
        m_pers2[2, 3] = -2 * far * near / (far - near)
        m_pers2[3, 2] = 1
        m_pers2[3, 3] = 0

    projection_matrix = np.dot(np.dot(m_pers2, m_perslb), m_persla)

    return [r_phi, r_theta, r_varphi, rot_total, view_matrix, projection_matrix]


def compute_anchor_points_projection(self, anchor_array, projection):
    [uv_anchor, yx_anchor] = [0] * 2
    return [uv_anchor, yx_anchor]


def compute_anchor_points_error(img_anchor_points, yx_model_anchor_points):
    return img_anchor_points - yx_model_anchor_points


def sample_object_at_uv(obj, triangle_array, uv):
    n_points = np.size(uv, 1)
    uv_indices = map(int, uv[0, :])
    triangle_ind = np.copy(triangle_array)
    for i in xrange(3):
        triangle_ind[i] = map(lambda x: x - 1, triangle_ind[i])

    if obj.ndim < 3:
        nobjects = 1
        sample = np.vstack((obj[:3, triangle_ind[0, uv_indices]],
                            obj[:3, triangle_ind[1, uv_indices]],
                            obj[:3, triangle_ind[2, uv_indices]]))
        sample = np.tile(sample[:, :, None], (1, 1, 1))
    else:
        nobjects = np.size(obj, 2)
        sample = np.vstack((obj[:3, triangle_ind[0, uv_indices], :],
                            obj[:3, triangle_ind[1, uv_indices], :],
                            obj[:3, triangle_ind[2, uv_indices], :]))

    if isinstance(obj[3, 0], float):
        sampled = obj[3, 0] * np.ones([4, n_points, nobjects])
    else:
        sampled = obj[3, 0, 0] * np.ones([4, n_points, nobjects])

    # The None is there to have the same behaviour of tile as repmat in matlab
    sampled[0:3, :, :] = \
        np.multiply(np.tile(uv[1, :, None], (3, 1, nobjects)), sample[:3]) \
        + np.multiply(np.tile(uv[2, :, None], (3, 1, nobjects)), sample[3:6]) \
        + np.multiply(np.tile(uv[3, :, None], (3, 1, nobjects)), sample[6:9])
    return sampled


def compute_ortho_projection_derivatives_shape_parameters(s_uv, s_pc_uv, rho, r_tot, shape_ev):
    # Precomputations
    n_parameters = np.size(s_pc_uv, 2)
    n_points = np.size(s_uv, 1)
    dp_dgamma = np.zeros([2, n_parameters, n_points])

    u_max = 2
    u_min = -2
    v_max = 2
    v_min = -2

    const_x = (2 * rho[0]) / (u_max - u_min)
    const_y = (2 * rho[0]) / (v_max - v_min)
    const_term = np.vstack((const_x, const_y))

    for k in xrange(n_parameters):
        dw_dalpha_k_uv = np.dot(r_tot, shape_ev[k] * s_pc_uv[:, :, k])
        dp_dalpha_k_uv = np.vstack((dw_dalpha_k_uv[0, :], dw_dalpha_k_uv[1, :]))
        dp_dgamma[:, k, :] = np.multiply(np.tile(const_term, (1, n_points)), dp_dalpha_k_uv)

    return dp_dgamma


# DONE
def compute_pers_projection_derivatives_shape_parameters(s_uv, w_uv, s_pc_uv, rho, r_tot, shape_ev):
    # Precomputations
    n_parameters = np.size(s_pc_uv, 2)
    n_points = np.size(s_uv, 1)
    dp_dgamma = np.zeros([2, n_parameters, n_points])

    u_max = 2
    u_min = -2
    v_max = 2
    v_min = -2

    w = w_uv[2, :].transpose()

    const_x = np.divide((2 * rho[0]) / (u_max - u_min), np.power(w, 2))
    const_y = np.divide((2 * rho[0]) / (v_max - v_min), np.power(w, 2))
    const_term = np.vstack((const_x, const_y))

    for k in xrange(n_parameters):
        dw_dalpha_k_uv = np.dot(r_tot, shape_ev[k] * s_pc_uv[:, :, k])
        dp_dalpha_k_uv = np.vstack(
            (np.multiply(dw_dalpha_k_uv[0, :], w) - np.multiply(w_uv[0, :].transpose(), dw_dalpha_k_uv[2, :]),
             np.multiply(dw_dalpha_k_uv[1, :], w) - np.multiply(w_uv[1, :].transpose(), dw_dalpha_k_uv[2, :])))
        dp_dgamma[:, k, :] = np.multiply(const_term, dp_dalpha_k_uv)

    return dp_dgamma


# TODO
def compute_projection_derivatives_shape_parameters(s_uv_anchor, w_uv_anchor, rho_c, rot,
                                                    s_pc_uv_anchor, ctrl_params, shape_ev):
    if ctrl_params['projection_type'] == 0:
        dp_dgamma = compute_ortho_projection_derivatives_shape_parameters(s_uv_anchor, s_pc_uv_anchor,
                                                                          rho_c, rot, shape_ev)
    else:
        dp_dgamma = compute_pers_projection_derivatives_shape_parameters(s_uv_anchor, w_uv_anchor,
                                                                         s_pc_uv_anchor, rho_c, rot,
                                                                         shape_ev)
    return dp_dgamma


# TODO
def compute_projection_derivatives_warp_parameters(s_uv, w_uv, rho_c,
                                                   r_phi, r_theta, r_varphi, ctrl_params):
    if ctrl_params['projection_type'] == 0:
        dp_dgamma = compute_ortho_projection_derivatives_warp_parameters(s_uv, w_uv, rho_c, r_phi, r_theta, r_varphi)
    else:
        dp_dgamma = compute_pers_projection_derivatives_warp_parameters(s_uv, w_uv, rho_c, r_phi, r_theta, r_varphi)
    return dp_dgamma


# TODO
def compute_sd_error_product(sd_anchor, error_uv):
    return 0


# TODO: doc + comments
def project(w, projection_matrix, projection_type):
    projected = np.dot(projection_matrix, w)

    # Perspective projection type
    if projection_type == 1:
        projected = np.divide(projected, np.matlib.repmat(projected[3, :], 4, 1))
    return projected


# TODO: doc + comments
def model_to_object(coeff, mean, pc, ev):
    # Maybe add these lines directly in the model import_from_basel
    # as the loadmat imports a list of lists for a 1D array
    mean = np.ndarray.flatten(mean)
    ev = np.ndarray.flatten(ev)
    # Reconstruction
    nseg = 1
    ndim = np.size(coeff, 0)
    obj = mean * np.ones([1, nseg]) + np.dot(pc[:, 0:ndim], np.multiply(coeff, ev[0:ndim]))
    return np.transpose(np.array(obj))

    # Blending
    # nver = np.size(obj, 0) / 3
    # allver = np.zeros([nseg * nver, 3])
    # k = 0
    # for i in xrange(nseg):
    #     allver[k + 1:k + nver, :] = np.reshape(obj[:, i], (3, nver), order='F')
    #     k += nver
    #
    # obj = np.transpose(np.linalg.inv(mm) * mb * allver)
    #
    # return obj


# TODO: model2object
def compute_texture():
    texture = []
    return texture


# TODO: doc + comments
def homogenize(obj):
    obj = np.array(obj)
    npoints = np.size(obj, 0) / 3
    nobjects = np.size(obj, 1)

    out = np.ones([4, npoints, nobjects])
    out[0:3, :, :] = np.reshape(obj, (3, npoints, nobjects), order='F')

    return out


# TODO
def visualize(image, anchors, yx_anchor):
    print "visualize"


# TODO
def update_parameters(hess, sd_error_product):
    return np.dot(-np.linalg.inv(hess), sd_error_product)


def update(delta_sigma, fit_params):
    [alpha_c, beta_c, rho_c, iota_c] = [0]*4
    return [alpha_c, beta_c, rho_c, iota_c]


# DONE
def compute_pers_projection_derivatives_warp_parameters(s_uv, w_uv, rho, r_phi, r_theta, r_varphi):
    # Precomputations
    n_parameters = np.size(rho, 0)
    n_points = np.size(s_uv, 1)
    dp_dgamma = np.zeros([2, n_parameters, n_points])

    u_max = 2
    u_min = -2
    v_max = 2
    v_min = -2

    w = w_uv[2, :].transpose()

    const_x = np.divide((2 * rho[0]) / (u_max - u_min), np.power(w, 2))
    const_y = np.divide((2 * rho[0]) / (v_max - v_min), np.power(w, 2))
    const_term = np.vstack((const_x, const_y))

    # Compute the derivative of the perspective projection wrt focal length
    dp_dgamma[:, 0, :] = np.vstack(([(2 / (u_max - u_min)) * np.divide(w_uv[0, :].transpose(), w),
                                     (2 / (u_max - u_min)) * np.divide(w_uv[1, :].transpose(), w)]))

    # Compute the derivative of the phi rotation matrix
    dr_phi_dphi = np.eye(4, 4)
    dr_phi_dphi[1:3, 1:3] = np.array([[-np.sin(rho[1]), -np.cos(rho[1])],
                                      [np.cos(rho[1]), -np.sin(rho[1])]])

    # Compute the derivative of the warp wrt phi
    dW_dphi_uv = np.dot(dr_phi_dphi, np.dot(r_theta, np.dot(r_varphi, s_uv[:, :, 0])))

    # Compute the derivative of the projection wrt phi
    dp_dphi_uv = np.vstack(
    (np.multiply(dW_dphi_uv[0, :], w) - np.multiply(w_uv[0, :].transpose(), dW_dphi_uv[2, :]),
     np.multiply(dW_dphi_uv[1, :], w) - np.multiply(w_uv[1, :].transpose(), dW_dphi_uv[2, :])))

    dp_dgamma[:, 1, :] = np.multiply(const_term, dp_dphi_uv)

    # Compute the derivative of the theta rotation matrix
    dr_theta_dtheta = np.eye(4, 4)
    dr_theta_dtheta[:3, :3] = np.array([[-np.sin(rho[2]), 0, -np.cos(rho[2])],
                                        [0, 0, 0],
                                        [np.cos(rho[2]), 0, -np.sin(rho[2])]])

    # Compute the derivative of the warp wrt theta
    dW_dtheta_uv = np.dot(r_phi, np.dot(dr_theta_dtheta, np.dot(r_varphi, s_uv[:, :, 0])))

    # Compute the derivative of the projection wrt theta
    dp_dtheta_uv = np.vstack(
        (np.multiply(dW_dtheta_uv[0, :], w) - np.multiply(w_uv[0, :].transpose(), dW_dtheta_uv[2, :]),
         np.multiply(dW_dtheta_uv[1, :], w) - np.multiply(w_uv[1, :].transpose(), dW_dtheta_uv[2, :])))

    dp_dgamma[:, 2, :] = np.multiply(const_term, dp_dtheta_uv)

    # Compute the derivative of the varphi rotation matrix
    dr_varphi_dvarphi = np.eye(4, 4)
    dr_varphi_dvarphi[:2, :2] = np.array([[-np.sin(rho[3]), -np.cos(rho[3])],
                                          [np.cos(rho[3]), -np.sin(rho[3])]])

    # Compute the derivative of the warp wrt varphi
    dW_dvarphi_uv = np.dot(r_phi, np.dot(r_theta, np.dot(dr_varphi_dvarphi, s_uv[:, :, 0])))

    # Compute the derivative of the projection wrt varphi
    dp_dvarphi_uv = np.vstack(
        (np.multiply(dW_dvarphi_uv[0, :], w) - np.multiply(w_uv[0, :].transpose(), dW_dvarphi_uv[2, :]),
         np.multiply(dW_dvarphi_uv[1, :], w) - np.multiply(w_uv[1, :].transpose(), dW_dvarphi_uv[2, :])))

    dp_dgamma[:, 3, :] = np.multiply(const_term, dp_dvarphi_uv)

    # Compute the derivative of the projection function wrt tx
    dp_dtx_uv = np.vstack((w_uv[2, :].transpose(), np.zeros(n_points)))

    dp_dgamma[:, 4, :] = np.multiply(const_term, dp_dtx_uv)

    # Compute the derivative of the projection function wrt ty
    dp_dty_uv = np.vstack((np.zeros(n_points), w))

    dp_dgamma[:, 5, :] = np.multiply(const_term, dp_dty_uv)

    # Compute the derivative of the projection function wrt tz
    dp_dtz_uv = np.vstack((-w_uv[0, :].transpose(), -w_uv[1, :].transpose()))

    dp_dgamma[:, 5, :] = np.multiply(const_term, dp_dtz_uv)

    return dp_dgamma


# DONE
def compute_ortho_projection_derivatives_warp_parameters(s_uv, w_uv, rho, r_phi, r_theta, r_varphi):
    # Precomputations
    n_parameters = np.size(rho, 0)
    n_points = np.size(s_uv, 1)
    dp_dgamma = np.zeros([2, n_parameters, n_points])

    u_max = 2
    u_min = -2
    v_max = 2
    v_min = -2

    const_x = (2 * rho[0]) / (u_max - u_min)
    const_y = (2 * rho[0]) / (v_max - v_min)
    const_term = np.vstack((const_x, const_y))

    # Compute the derivative of the perspective projection wrt focal length
    dp_dgamma[:, 0, :] = np.vstack(([(2 / (u_max - u_min)) * w_uv[0, :].transpose(),
                                     (2 / (u_max - u_min)) * w_uv[1, :].transpose()]))

    # Compute the derivative of the phi rotation matrix
    dr_phi_dphi = np.eye(4, 4)
    dr_phi_dphi[1:3, 1:3] = np.array([[-np.sin(rho[1]), -np.cos(rho[1])],
                                      [np.cos(rho[1]), -np.sin(rho[1])]])

    # Compute the derivative of the warp wrt phi
    dW_dphi_uv = np.dot(dr_phi_dphi, np.dot(r_theta, np.dot(r_varphi, s_uv[:, :, 0])))

    dp_dgamma[:, 1, :] = np.multiply(np.tile(const_term, (1, n_points)), dW_dphi_uv[:2, :])

    # Compute the derivative of the theta rotation matrix
    dr_theta_dtheta = np.eye(4, 4)
    dr_theta_dtheta[:3, :3] = np.array([[-np.sin(rho[2]), 0, -np.cos(rho[2])],
                                        [0, 0, 0],
                                        [np.cos(rho[2]), 0, -np.sin(rho[2])]])

    # Compute the derivative of the warp wrt theta
    dW_dtheta_uv = np.dot(r_phi, np.dot(dr_theta_dtheta, np.dot(r_varphi, s_uv[:, :, 0])))

    dp_dgamma[:, 2, :] = np.multiply(np.tile(const_term, (1, n_points)), dW_dtheta_uv[:2, :])

    # Compute the derivative of the varphi rotation matrix
    dr_varphi_dvarphi = np.eye(4, 4)
    dr_varphi_dvarphi[:2, :2] = np.array([[-np.sin(rho[3]), -np.cos(rho[3])],
                                          [np.cos(rho[3]), -np.sin(rho[3])]])

    # Compute the derivative of the warp wrt varphi
    dW_dvarphi_uv = np.dot(r_phi, np.dot(r_theta, np.dot(dr_varphi_dvarphi, s_uv[:, :, 0])))

    dp_dgamma[:, 3, :] = np.multiply(np.tile(const_term, (1, n_points)), dW_dvarphi_uv[:2, :])

    # Compute the derivative of the projection function wrt tx
    dp_dtx_uv = np.vstack((np.ones([1, n_points]), np.zeros([1, n_points])))

    dp_dgamma[:, 4, :] = np.multiply(np.tile(const_term, (1, n_points)), dp_dtx_uv)

    # Define the derivative of the projection function wrt ty
    dp_dty_uv = np.vstack((np.zeros([1, n_points]), np.ones([1, n_points])))

    dp_dgamma[:, 5, :] = np.multiply(np.tile(const_term, (1, n_points)), dp_dty_uv)

    return dp_dgamma


# Raises warnings about boolean array-likes
def compute_texture_derivatives_shape_parameters(s_pc, T, shadow_index, W, triangle_array, S, uv, N, light_vector,
                                                 shape_ev, r_total, s_pc_uv, iota, V, R, M, L_dir, L_spec):

    n_parameters = np.size(s_pc, 2)
    n_points = np.size(T, 1)
    dt_dalpha = np.zeros([3, n_parameters, n_points])
    illuminated = [x != 0 for x in shadow_index]
    n_illuminated = np.sum(illuminated)
    shadow_degree = np.vstack((np.tile(shadow_index[illuminated], (3, 1)), np.zeros([1, n_illuminated])))
    W[3, :] = [0] * len(W[3, :])
    s_pc_uv[3, :, :] = np.zeros(s_pc_uv[3, :, :].shape)

    # Euclidean norm of the warp
    W_norm = np.sqrt(np.sum(np.power(W[:, illuminated], 2), axis=0))
    W_norm_vec = np.zeros([4, n_illuminated])
    W_norm_vec[:3, :] = np.tile(W_norm, (3, 1))

    # Compute f = v1 x v2 = (s1 - S2) x (s1- s3) and g = || f ||
    v1 = S[:, triangle_array[0, uv[0, illuminated]]] - S[:, triangle_array[1, uv[0, illuminated]]]
    v2 = S[:, triangle_array[0, uv[0, illuminated]]] - S[:, triangle_array[2, uv[0, illuminated]]]
    first = np.transpose(v1[:3, :])
    second = np.transpose(v2[:3, :])
    f = np.zeros([4, n_illuminated])
    f[:3, :] = np.transpose(np.cross(first, second))
    g = np.sqrt(np.sum(np.square(f), axis=0))
    g_vec = np.zeros([4, n_illuminated])
    g_vec[:3, :] = np.tile(g, (3, 1))

    # Compute the dot product of the surface normal and the light direction
    n_dot_d = np.sum(np.multiply(np.transpose(N[:, illuminated]), np.tile(light_vector, (n_illuminated, 1))), axis=1)
    n_dot_d = np.transpose(n_dot_d)
    # Clamping
    n_dot_d[n_dot_d < 0] = 0
    n_dot_d_vec = np.zeros([4, n_illuminated])
    n_dot_d_vec[:3, :] = np.tile(n_dot_d, (3, 1))

    # Compute the dot product of the reflection and the viewing direction
    r_dot_v = np.sum(np.multiply(R[:, illuminated], V[:, illuminated]), axis=0)
    # Clamping
    r_dot_v[n_dot_d < 0] = 0
    r_dot_v[r_dot_v < 0] = 0

    for k in xrange(n_parameters):
        # Compute derivatives of v1 and v2 with respect the kth element of alpha
        dv1_dalpha_k = shape_ev[k]*(s_pc[:, triangle_array[0, uv[0, illuminated]], k] - \
                                    s_pc[:, triangle_array[1, uv[0, illuminated]], k])
        dv2_dalpha_k = shape_ev[k]*(s_pc[:, triangle_array[0, uv[0, illuminated]], k] - \
                                    s_pc[:, triangle_array[2, uv[0, illuminated]], k])

        # Compute the derivatives of f and g wrt the kth element of alpha
        df_dalpha_k = np.zeros([4, n_illuminated])
        df_dalpha_k[:3, :] = np.transpose(np.cross(np.transpose(dv1_dalpha_k[:3, :]), second) + \
                                   np.cross(first, np.transpose(dv2_dalpha_k[:3, :])))

        dg_dalpha_k = np.divide(np.sum(np.multiply(f, df_dalpha_k), axis=0), g)
        # dg_dalpha_k = np.multiply(np.divide(np.transpose(f), g), df_dalpha_k)  # Where is the transpose in Joan's code
                                                                                 # and why is there a sum ?
        dg_dalpha_k_vec = np.zeros([4, n_illuminated])
        dg_dalpha_k_vec[:3, :] = np.tile(dg_dalpha_k, (3, 1))

        # Compute the derivative of the normal wrt the kth element of alpha
        # derivative of the normal with object centered coordinates
        dn_obj_dalpha_k = np.divide(np.multiply(df_dalpha_k, g_vec) - np.multiply(f, dg_dalpha_k_vec),
                                    np.square(g_vec))
        dn_obj_dalpha_k[3, :] = np.zeros(len(dn_obj_dalpha_k[3, :]))
        dn_dalpha_k = np.dot(r_total, dn_obj_dalpha_k)

        # Compute the derivative of the reflection (r) wrt alpha k
        # Compute the derivative of the dot product of the normal and the direction of the light wrt alpha k
        dn_dot_d_dalpha_k = np.zeros([4, n_illuminated])
        dot_prod = np.sum(np.multiply(np.transpose(dn_dalpha_k), np.tile(light_vector, (n_illuminated, 1))), axis=1)
        dot_prod = np.transpose(dot_prod)
        dot_prod[n_dot_d < 0] = 0
        dn_dot_d_dalpha_k[:3, :] = np.tile(dot_prod, (3, 1))

        dr_dalpha_k = 2 * (np.multiply(dn_dot_d_dalpha_k, N[:, illuminated]) + np.multiply(n_dot_d_vec, dn_dalpha_k))

        # Compute the derivative of the warp (W) and its euclidean norm wrt alpha k
        dW_dalpha_k = np.dot(r_total, shape_ev[k]*s_pc_uv[:, illuminated, k])
        dW_norm_dalpha_k = np.divide(np.sum(np.multiply(W[:, illuminated], dW_dalpha_k), axis=0), W_norm)
        dW_norm_dalpha_k_vec = np.zeros([4, n_illuminated])
        dW_norm_dalpha_k_vec[:3, :] = np.tile(dW_norm_dalpha_k, (3, 1))

        # Compute the derivative of the viewing vector (v) wrt alpha k
        dv_dalpha_k = -np.divide(np.multiply(dW_dalpha_k, W_norm_vec) - \
                                 np.multiply(W[:, illuminated], dW_norm_dalpha_k_vec),
                                 np.square(W_norm_vec))

        # Compute the derivative of the dot product of r and v wrt alpha k
        dr_dot_v_dalpha_k = np.zeros([4, n_illuminated])
        summed = iota[16] * np.sum(np.multiply(np.tile(np.power(r_dot_v, iota[16] - 1), (4, 1)),
                                               np.sum(np.multiply(dr_dalpha_k, V[:, illuminated]), axis=0) + \
                                               np.sum(np.multiply(R[:, illuminated], dv_dalpha_k), axis=0)), axis=0)
        summed[n_dot_d < 0] = 0
        summed[r_dot_v < 0] = 0
        dr_dot_v_dalpha_k[:3, :] = np.tile(summed, (3, 1))

        # Compute the derivative of the color transformation function wrt alpha_k
        dt_dalpha_k = np.dot(M, np.multiply(shadow_degree,
                                            np.multiply(np.dot(L_dir, dn_dot_d_dalpha_k),
                                                        T[:, illuminated] + np.dot(
                                                            L_spec, iota[15] * np.multiply(
                                                                dr_dot_v_dalpha_k, np.ones([4, n_illuminated]))))))
        dt_dalpha[:, k, illuminated] = dt_dalpha_k[:3, :]

        return dt_dalpha


def compute_texture_derivatives_warp_parameters(rho, T, shadow_index, W, light_vector, R, V, r_theta, r_phi, r_varphi,
                                                N_uv, S, iota, M, L_dir, L_spec):

    # Pre-computations
    n_parameters = np.size(rho, 0)
    n_points = np.size(T, 1)
    dt_drho = np.zeros([3, n_parameters, n_points])

    illuminated = [x != 0 for x in shadow_index]
    n_illuminated = np.sum(illuminated)
    shadow_degree = np.vstack((np.tile(shadow_index[illuminated], (3, 1)), np.zeros([1, n_illuminated])))
    W[3, :] = [0] * len(W[3, :])

    # Euclidean norm of the warp
    W_norm = np.sqrt(np.sum(np.power(W[:, illuminated], 2), axis=0))
    W_norm_vec = np.zeros([4, n_illuminated])
    W_norm_vec[:3, :] = np.tile(W_norm, (3, 1))

    # Compute the dot product of the surface normal and the light direction
    n_dot_d = np.sum(np.multiply(np.transpose(N_uv[:, illuminated]), np.tile(light_vector, (n_illuminated, 1))), axis=1)
    n_dot_d = np.transpose(n_dot_d)
    # Clamping
    n_dot_d[n_dot_d < 0] = 0
    n_dot_d_vec = np.zeros([4, n_illuminated])
    n_dot_d_vec[:3, :] = np.tile(n_dot_d, (3, 1))

    # Compute the dot product of the reflection and the viewing direction
    r_dot_v = np.sum(np.multiply(R[:, illuminated], V[:, illuminated]), axis=0)
    # Clamping
    r_dot_v[n_dot_d < 0] = 0
    r_dot_v[r_dot_v < 0] = 0

    # Compute the derivative of the texture wrt phi
    # ---------------------------------------------

    # Compute the derivative of the phi rotation matrix
    dr_phi_dphi = np.eye(4)
    dr_phi_dphi[1:3, 1:3] = np.array([[-np.sin(rho[1]), -np.cos(rho[1])],
                  [np.cos(rho[1]), -np.sin(rho[1])]])

    # Compute the derivative of the normal wrt phi
    dn_dphi = np.dot(dr_phi_dphi, np.dot(r_theta, np.dot(r_varphi, N_uv[:, illuminated])))

    # Compute the derivative of the dot product of the normal and the direction of the light wrt phi
    dn_dot_d_dphi = np.zeros([4, n_illuminated])
    dot_prod = np.sum(np.multiply(np.transpose(dn_dphi), np.tile(light_vector, (n_illuminated, 1))), axis=1)
    dot_prod = np.transpose(dot_prod)
    dot_prod[n_dot_d < 0] = 0
    dn_dot_d_dphi[:3, :] = np.tile(dot_prod, (3, 1))

    # Compute the derivative of the reflection wrt phi
    dr_dphi = 2 * (np.multiply(dn_dot_d_dphi, N_uv[:, illuminated]) + np.multiply(n_dot_d_vec, dn_dphi))

    # Compute the derivative of the warp wrt phi
    dW_dphi = np.dot(dr_phi_dphi, np.dot(r_theta, np.dot(r_varphi, S[:, illuminated])))
    dW_norm_dphi = np.divide(np.sum(np.multiply(W[:, illuminated], dW_dphi), axis=0), W_norm)
    dW_norm_dphi_vec = np.zeros([4, n_illuminated])
    dW_norm_dphi_vec[:3, :] = np.tile(dW_norm_dphi, (3, 1))

    # Compute the derivative of the viewing vector (v) wrt phi
    dv_dphi = - np.divide(np.multiply(dW_dphi, W_norm_vec) -
      np.multiply(W[:, illuminated], dW_norm_dphi_vec),
      np.power(W_norm_vec, 2))

    # Compute the derivative of the dot product of r and v wrt phi
    dr_dot_v_dphi = np.zeros([4, n_illuminated])
    summed = np.sum(iota[16] * np.multiply(np.tile(np.power(r_dot_v, iota[16] - 1), (4, 1)),
                       np.sum(np.multiply(dr_dphi, V[:, illuminated]), axis=0) +
                       np.sum(np.multiply(R[:, illuminated], dv_dphi), axis=0)), axis=0)
    summed[n_dot_d < 0] = 0
    summed[r_dot_v < 0] = 0
    dr_dot_v_dphi[:3, :] = np.tile(summed, (3, 1))

    # Compute the derivative of the color transformation function wrt phi
    dt_dphi = np.dot(M, np.multiply(shadow_degree, np.dot(L_dir, np.multiply(
        dn_dot_d_dphi, T[:, illuminated] + np.dot(L_spec, iota[16] * np.multiply(
            dr_dot_v_dphi, np.ones([4, n_illuminated])))))))
    dt_drho[:, 1, illuminated] = dt_dphi[:3, :]

    # Compute the derivative of the texture wrt theta
    # -----------------------------------------------

    # Compute the derivative of the theta rotation matrix
    dr_theta_dtheta = np.eye(4)
    dr_theta_dtheta[:3, :3] = np.array([[-np.sin(rho[2]), 0, -np.cos(rho[2])],
                    [0, 0, 0],
                    [np.cos(rho[2]), 0, -np.sin(rho[2])]])

    # Compute the derivative of the normal wrt theta
    dn_dtheta = np.dot(r_phi, np.dot(dr_theta_dtheta, np.dot(r_varphi, N_uv[:, illuminated])))

    # Compute the derivative of the dot product of the normal and the direction of the light wrt theta
    dn_dot_d_dtheta = np.zeros([4, n_illuminated])
    dot_prod = np.sum(np.multiply(np.transpose(dn_dtheta), np.tile(light_vector, (n_illuminated, 1))), axis=1)
    dot_prod = np.transpose(dot_prod)
    dot_prod[n_dot_d < 0] = 0
    dn_dot_d_dtheta[:3, :] = np.tile(dot_prod, (3, 1))

    # Compute the derivative of the reflection wrt theta
    dr_dtheta = 2 * (np.multiply(dn_dot_d_dtheta, N_uv[:, illuminated]) + np.multiply(n_dot_d_vec, dn_dtheta))

    # Compute the derivative of the warp (W) and its euclidean norm wrt theta
    dW_dtheta = np.dot(r_phi, np.dot(dr_theta_dtheta, np.dot(r_varphi, S[:, illuminated])))
    dW_norm_dtheta = np.divide(np.sum(np.multiply(W[:, illuminated], dW_dtheta), axis=0), W_norm)
    dW_norm_dtheta_vec = np.zeros([4, n_illuminated])
    dW_norm_dtheta_vec[:3, :] = np.tile(dW_norm_dtheta, (3, 1))

    # Compute the derivative of the viewing vector (v) wrt theta
    dv_dtheta = - np.divide(np.multiply(dW_dtheta, W_norm_vec) -
        np.multiply(W[:, illuminated], dW_norm_dtheta_vec),
        np.square(W_norm_vec))

    # Compute the derivative of the dot product of r and v wrt theta
    dr_dot_v_dtheta = np.zeros([4, n_illuminated])
    summed = np.sum(iota[16] * np.multiply(np.tile(np.power(r_dot_v, iota[16] - 1), (4, 1)),
                                           np.sum(np.multiply(dr_dtheta, V[:, illuminated]), axis=0) +
                                           np.sum(np.multiply(R[:, illuminated], dv_dtheta), axis=0)), axis=0)
    summed[n_dot_d < 0] = 0
    summed[r_dot_v < 0] = 0
    dr_dot_v_dtheta[:3, :] = np.tile(summed, (3, 1))

    # Compute the derivative of the color transformation function wrt phi
    dt_dtheta = np.dot(M, np.multiply(shadow_degree, np.dot(L_dir, np.multiply(
    dn_dot_d_dtheta, T[:, illuminated] + np.dot(L_spec, iota[16] * np.multiply(
    dr_dot_v_dtheta, np.ones([4, n_illuminated])))))))
    dt_drho[:, 2, illuminated] = dt_dtheta[:3, :]

    # Compute the derivative of the texture wrt varphi
    # -----------------------------------------------

    # Compute the derivative of the varphi rotation matrix
    dr_varphi_dvarphi = np.eye(4)
    dr_varphi_dvarphi[:2, :2] = np.array([[-np.sin(rho[3]), -np.cos(rho[3])],
                                          [np.cos(rho[3]), -np.sin(rho[3])]])

    # Compute the derivative of the normal wrt varphi
    dn_dvarphi = np.dot(r_phi, np.dot(r_theta, np.dot(dr_varphi_dvarphi, N_uv[:, illuminated])))

    # Compute the derivative of the dot product of the normal and the direction of the light wrt varphi
    dn_dot_d_dvarphi = np.zeros([4, n_illuminated])
    dot_prod = np.sum(np.multiply(np.transpose(dn_dvarphi), np.tile(light_vector, (n_illuminated, 1))), axis=1)
    dot_prod = np.transpose(dot_prod)
    dot_prod[n_dot_d < 0] = 0
    dn_dot_d_dvarphi[:3, :] = np.tile(dot_prod, (3, 1))

    # Compute the derivative of the reflection wrt varphi
    dr_dvarphi = 2 * (np.multiply(dn_dot_d_dvarphi, N[:, illuminated]) + np.multiply(n_dot_d_vec, dn_dvarphi))

    # Compute the derivative of the warp (W) and its euclidean norm wrt varphi
    dW_dvarphi = np.dot(r_phi, np.dot(r_theta, np.dot(dr_varphi_dvarphi, S[:, illuminated])))
    dW_norm_dvarphi = np.divide(np.sum(np.multiply(W[:, illuminated], dW_dvarphi), axis=0), W_norm)
    dW_norm_dvarphi_vec = np.zeros([4, n_illuminated])
    dW_norm_dvarphi_vec[:3, :] = np.tile(dW_norm_dvarphi, (3, 1))

    # Compute the derivative of the viewing vector (v) wrt varphi
    dv_dvarphi = - np.divide(np.multiply(dW_dvarphi, W_norm_vec) -
         np.multiply(W[:, illuminated], dW_norm_dvarphi_vec),
         np.square(W_norm_vec))

    # Compute the derivative of the dot product of r and v wrt varphi
    dr_dot_v_dvarphi = np.zeros([4, n_illuminated])
    summed = np.sum(iota[16] * np.multiply(np.tile(np.power(r_dot_v, iota[16] - 1), (4, 1)),
                                           np.sum(np.multiply(dr_dvarphi, V[:, illuminated]), axis=0) +
                                           np.sum(np.multiply(R[:, illuminated], dv_dvarphi), axis=0)), axis=0)
    summed[n_dot_d < 0] = 0
    summed[r_dot_v < 0] = 0
    dr_dot_v_dvarphi[:3, :] = np.tile(summed, (3, 1))

    # Compute the derivative of the color transformation function wrt varphi
    dt_dvarphi = np.dot(M, np.multiply(shadow_degree, np.dot(L_dir, np.multiply(
        dn_dot_d_dvarphi, T[:, illuminated] + np.dot(L_spec, iota[16] * np.multiply(
            dr_dot_v_dvarphi, np.ones([4, n_illuminated])))))))
    dt_drho[:, 3, illuminated] = dt_dvarphi[:3, :]

    return dt_drho


def compute_texture_derivatives_illumination_parameters(iota, T, shadow_index, N, light_vector, R, V, C, I,
                                                        G, M, diffuse_term, specular_term, L_dir, L_spec):
    # Pre-computations
    n_parameters = np.size(iota, 0)
    n_points = np.size(T, 1)
    dt_diota = np.zeros([3, n_parameters, n_points])

    illuminated = [x != 0 for x in shadow_index]
    n_illuminated = np.sum(illuminated)
    shadow_degree = np.vstack((np.tile(shadow_index[illuminated], (3, 1)), np.zeros([1, n_illuminated])))

    # Compute the dot product of the surface normal and the light direction
    n_dot_d = np.sum(np.multiply(N[:, illuminated], np.transpose(np.tile(light_vector, (n_illuminated, 1)))), axis=0)
    # Clamping
    n_dot_d[n_dot_d < 0] = 0

    # Compute the dot product of the reflection and the viewing direction
    r_dot_v = np.sum(np.multiply(R[:, illuminated], V[:, illuminated]), axis=0)
    # Clamping
    r_dot_v[n_dot_d < 0] = 0
    r_dot_v[r_dot_v < 0] = 0

    # Compute the derivative of the gain matrix wrt the gain g_r
    dG_dgr = np.zeros([4, 4])
    dG_dgr[0, 0] = 1
    dG_dgr[3, 3] = 1

    # Compute the derivative of the texture function wrt the gain g_r
    dt_dgr = np.dot(np.dot(dG_dgr, C), I)
    dt_diota[:, 0, :] = dt_dgr[:3, :]

    # Compute the derivative of the gain matrix wrt the gain g_g
    dG_dgg = np.zeros([4, 4])
    dG_dgg[1, 1] = 1
    dG_dgg[3, 3] = 1

    # Compute the derivative of the texture function wrt the gain g_g
    dt_dgg = np.dot(np.dot(dG_dgg, C), I)
    dt_diota[:, 1, :] = dt_dgg[:3, :]

    # Compute the derivative of the gain matrix wrt the gain g_b
    dG_dgb = np.zeros([4, 4])
    dG_dgb[2, 2] = 1
    dG_dgb[3, 3] = 1

    # Compute the derivative of the texture function wrt the gain g_b
    dt_dgb = np.dot(np.dot(dG_dgb, C), I)
    dt_diota[:, 2, :] = dt_dgb[:3, :]

    # Compute the derivative of the contrast matrix wrt c
    dC_dc = - np.array([[0.3, 0.59, 0.11, 0],
                        [0.3, 0.59, 0.11, 0],
                        [0.3, 0.59, 0.11, 0],
                        [0, 0, 0, 1]])

    # Compute the derivative of the texture function wrt c
    dt_dc = np.dot(G, np.dot(dC_dc, I))

    dt_diota[:, 3, :] = dt_dc[:3, :]

    # Compute the derivative of the projection function wrt tx
    dt_dor = np.vstack((np.ones(n_points), np.zeros([2, n_points])))

    dt_diota[:, 4, :] = dt_dor

    # Compute the derivative of the projection function wrt ty
    dt_dog = np.vstack((np.zeros(n_points), np.ones(n_points), np.zeros(n_points)))

    dt_diota[:, 5, :] = dt_dog

    # Compute the derivative of the projection function wrt tz
    dt_dob = np.vstack((np.zeros([2, n_points]), np.ones(n_points)))

    dt_diota[:, 6, :] = dt_dob

    # Compute the derivative of the ambient light matrix wrt Lamb_r
    dLamb_dLambr = np.zeros([4, 4])
    dLamb_dLambr[0, 1] = 1
    dLamb_dLambr[3, 3] = 1

    # Compute the derivative of the color transformation function wrt Lamb_r
    dt_dLambr = np.dot(M, np.dot(dLamb_dLambr, T))

    dt_diota[:, 7, :] = dt_dLambr[:3, :]

    # Compute the derivative of the ambient light matrix wrt Lamb_g
    dLamb_dLambg = np.zeros([4, 4])
    dLamb_dLambg[1, 1] = 1
    dLamb_dLambg[3, 3] = 1

    # Compute the derivative of the color transformation function wrt Lamb_g
    dt_dLambg = np.dot(np.dot(M, dLamb_dLambg), T)

    dt_diota[:, 8, :] = dt_dLambg[:3, :]

    # Compute the derivative of the ambient light matrix wrt Lamb_b
    dLamb_dLambb = np.zeros([4, 4])
    dLamb_dLambb[2, 2] = 1
    dLamb_dLambb[3, 3] = 1

    # Compute the derivative of the color transformation function wrt Lamb_b
    dt_dLambb = np.dot(np.dot(M, dLamb_dLambb), T)

    dt_diota[:, 9, :] = dt_dLambb[:3, :]

    # Compute the derivative of the directed light matrix wrt Ldir_r
    dLdir_dLdirr = np.zeros([4, 4])
    dLdir_dLdirr[0, 0] = 1
    dLdir_dLdirr[3, 3] = 1

    dLspec_dLdirr = np.zeros([4, 4])
    dLspec_dLdirr[0, 0] = -1
    dLspec_dLdirr[3, 3] = -1

    # Compute the derivative of the color transformation function wrt Ldir_r
    dt_dLdirr = np.dot(M, np.multiply(shadow_degree, np.dot(dLdir_dLdirr, diffuse_term) + \
                                      np.dot(dLspec_dLdirr, specular_term)))

    dt_diota[:, 10, illuminated] = dt_dLdirr[:3, :]

    # Compute the derivative of the directed light matrix wrt Ldir_g
    dLdir_dLdirg = np.zeros([4, 4])
    dLdir_dLdirg[1, 1] = 1
    dLdir_dLdirg[3, 3] = 1

    dLspec_dLdirg = np.zeros([4, 4])
    dLspec_dLdirg[1, 1] = -1
    dLspec_dLdirg[3, 3] = -1

    # Compute the derivative of the color transformation function wrt Ldir_g
    dt_dLdirg = np.dot(M, np.multiply(shadow_degree, np.dot(dLdir_dLdirg, diffuse_term) + \
                                      np.dot(dLspec_dLdirg, specular_term)))

    dt_diota[:, 11, illuminated] = dt_dLdirg[:3, :]

    # Compute the derivative of the directed light matrix wrt Ldir_b
    dLdir_dLdirb = np.zeros([4, 4])
    dLdir_dLdirb[2, 2] = 1
    dLdir_dLdirb[3, 3] = 1

    dLspec_dLdirb = np.zeros([4, 4])
    dLspec_dLdirb[2, 2] = -1
    dLspec_dLdirb[3, 3] = -1

    # Compute the derivative of the color transformation function wrt Ldir_b
    dt_dLdirb = np.dot(M, np.multiply(shadow_degree, np.dot(dLdir_dLdirb, diffuse_term) + \
                                      np.dot(dLspec_dLdirb, specular_term)))

    dt_diota[:, 12, illuminated] = dt_dLdirb[:3, :]

    # The derivative of the color transformation function wrt phil
    # ------------------------------------------------------------

    # Compute the derivative of the direction of the light wrt phi_l
    dd_dphil = np.array([np.cos(iota[14]) * np.cos(iota[13]), 0, -np.cos(iota[14]) * np.sin(iota[13]), 0])

    # Compute the derivative of the dot product between the normal and the direction of the light wrt phi_l
    dn_dot_d_dphil = np.zeros([4, n_illuminated])
    dot_prod = np.sum(np.multiply(N[:, illuminated], np.transpose(np.tile(dd_dphil, (n_illuminated, 1)))), axis=0)
    dot_prod[n_dot_d < 0] = 0
    dn_dot_d_dphil[:3, :] = np.tile(dot_prod, (3, 1))

    # Compute the derivative of the reflection of the light wrt phi l
    dr_dphil = 2 * np.multiply(dn_dot_d_dphil, N[:, illuminated] - np.transpose(np.tile(dd_dphil, (n_illuminated, 1))))

    # Compute the derivative of the dot product of r and v wrt phi l
    dr_dot_v_dphil = np.zeros([4, n_illuminated])
    summed = np.sum(iota[16] * np.multiply(np.tile(np.power(r_dot_v, iota[16] - 1), (4, 1)),
                                           np.sum(np.multiply(dr_dphil, V[:, illuminated]), axis=0)), axis=0)
    summed[n_dot_d < 0] = 0
    summed[r_dot_v < 0] = 0
    dr_dot_v_dphil[:3, :] = np.tile(summed, (3, 1))

    # Compute the derivative of the color transformation function wrt phil
    dt_dphil = np.dot(M, np.multiply(shadow_degree, np.multiply(np.dot(L_dir, dn_dot_d_dphil),
                                                                T[:, illuminated] + np.dot(L_spec,
                                                                                           iota[15] * np.multiply(
                                                                                               dr_dot_v_dphil, np.ones(
                                                                                                   [4, n_illuminated]))))))
    dt_diota[:, 13, illuminated] = dt_dphil[:3, :]

    # The derivative of the color transformation function wrt thetal
    # --------------------------------------------------------------

    # Compute the derivative of the direction of the light wrt thetal
    dd_dthetal = np.array([-np.sin(iota[14]) * np.sin(iota[13]), np.cos(iota[14]), -np.sin(iota[14]) * np.cos(iota[13]), 0])

    # Compute the derivative of the dot product between the normal and the direction of the light wrt thetal
    dn_dot_d_dthetal = np.zeros([4, n_illuminated])
    dot_prod = np.sum(np.multiply(N[:, illuminated], np.transpose(np.tile(dd_dthetal, (n_illuminated, 1)))), axis=0)
    dot_prod[n_dot_d < 0] = 0
    dn_dot_d_dphil[:3, :] = np.tile(dot_prod, (3, 1))

    # Compute the derivative of the reflection of the light wrt thetal
    dr_dthetal = 2 * np.multiply(dn_dot_d_dthetal, N[:, illuminated] - np.transpose(np.tile(dd_dthetal, (n_illuminated, 1))))

    # Compute the derivative of the dot product of r and v wrt thetal
    dr_dot_v_dthetal = np.zeros([4, n_illuminated])
    summed = np.sum(iota[16] * np.multiply(np.tile(np.power(r_dot_v, iota[16] - 1), (4, 1)),
                                           np.sum(np.multiply(dr_dthetal, V[:, illuminated]), axis=0)), axis=0)
    summed[n_dot_d < 0] = 0
    summed[r_dot_v < 0] = 0
    dr_dot_v_dphil[:3, :] = np.tile(summed, (3, 1))

    # Compute the derivative of the color transformation function wrt thetal
    dt_dthetal = np.dot(M, np.multiply(shadow_degree, np.multiply(np.dot(L_dir, dn_dot_d_dthetal),
                                                                  T[:, illuminated] + np.dot(L_spec, iota[15]
                                                                                             * np.multiply(
                                                                      dr_dot_v_dthetal, np.ones([4, n_illuminated]))))))
    dt_diota[:, 14, illuminated] = dt_dthetal[:3, :]

    # Compute the dot product of the reflection and the viewing direction
    r_dot_v = np.sum(np.multiply(R[:, illuminated], V[:, illuminated]), axis=0)
    # Clamping
    r_dot_v[n_dot_d < 0] = 0
    r_dot_v[r_dot_v < 0] = 0

    r_dot_v_v = np.zeros([4, n_illuminated])
    aux = np.power(r_dot_v, iota[16])
    aux[n_dot_d <= 0] = 0
    aux[r_dot_v <= 0] = 0
    r_dot_v_v[:3, :] = np.tile(aux, (3, 1))

    dt_dks = np.dot(M, np.multiply(shadow_degree, np.dot(L_spec, np.multiply(r_dot_v_v, np.ones([4, n_illuminated])))))

    dt_diota[:, 15, illuminated] = dt_dks[:3, :]

    dr_dot_v_dv = np.zeros([4, n_illuminated])
    aux = np.multiply(np.power(r_dot_v, iota[16]), np.log(r_dot_v))
    aux[n_dot_d <= 0] = 0
    aux[r_dot_v <= 0] = 0
    dr_dot_v_dv[:3, :] = np.tile(aux, (3, 1))

    dt_dv = np.dot(M, np.multiply(shadow_degree, np.dot(L_spec, np.multiply(r_dot_v_v, np.ones([4, n_illuminated])))))

    dt_diota[:, 16, illuminated] = dt_dv[:3, :]

    return dt_diota


# NOT SURE ABOUT THE MATH
def compute_texture_derivatives_texture_parameters(t_pc, shadow_index, texture_ev, M,
                                                   L_amb, L_dir, diffuse_dot, specular_term):
    # Initialization

    n_parameters = np.size(t_pc, 2)
    n_points = np.size(t_pc, 1)
    dt_dbeta = np.zeros([3, n_parameters, n_points])

    # Precomputations
    illuminated = [x != 0 for x in shadow_index]
    n_illuminated = np.sum(illuminated)
    shadow_degree = np.vstack((np.tile(shadow_index[illuminated], (3, 1)), np.zeros([1, n_illuminated])))

    # Computations
    for k in xrange(n_parameters):
        # Compute the derivative of the linear texture model wrt beta k
        dT_dbetak = texture_ev[k]*t_pc[:, :, k]

        # Compute the derivative of the ambient part of the texture function wrt beta k
        dt_dbetak = np.dot(M, np.dot(L_amb, dT_dbetak))

        # Compute the derivative of the diffuse and specular parts of the texture function wrt beta k
        dt_dbetak[:, illuminated] = dt_dbetak[:, illuminated] + np.multiply(
                shadow_degree, np.dot(M, np.dot(L_dir, np.multiply(diffuse_dot,
                                                                   dT_dbetak[:, illuminated]) + specular_term)))

        dt_dbeta[:, k, :] = dt_dbetak[:3, :]

    return dt_dbeta


# TODO: doc + comments
def import_anchor_points(anchors_pf):
    # Loading the anchor points file
    anchors = spio.loadmat(anchors_pf)["I_input"]

    img = anchors["img"][0, 0]
    resolution = anchors["resolution"][0, 0]
    resolution = np.ndarray.flatten(resolution)
    anchor_points = anchors["anchorPoints"][0, 0]
    model_triangles = anchors["modelTriangles"][0, 0]
    # Adapt Matlab indices to Python
    model_triangles = [x-1 for x in np.ndarray.flatten(model_triangles)]

    return [img, resolution, anchor_points, model_triangles]


# TODO
def compute_derivatives(n_alphas, n_rhos, n_betas, n_iotas,
                        s_uv, w_uv, rho, rot, s_pc_uv, ctrl, shape_ev,
                        s_pc, T_uv, shadow_index, triangle_array, S, uv, light_vector, iota, V, R, M, L_dir, L_spec,
                        r_phi, r_theta, r_varphi, N_uv, t_pc, texture_ev, L_amb, diffuse_dot, specular_term,
                        diffuse_term, C, I, G):
    r"""

    Parameters
    ----------
    n_alphas : `int`
        The number of shape parameters we want to update
    n_rhos : `int`
        The number of camera parameters we want to update
    n_betas : `int`
        The number of texture parameters we want to update
    n_iotas : `int`
        The number of rendering parameters we want to update
    s_uv : ``(4, n_dims)`` `ndarray`
        The sampling of the shape matrix using the selected uv triangles
    w_uv : ``(4, n_dims)`` `ndarray`
        The sampling of the warp matrix using the selected uv triangles
    rho : ``(6,)`` `ndarray`
        The camera parameters array
    rot : ``(4, 4)`` `ndarray`
        The total rotation matrix
    s_pc_uv : ``(4, n_dims, n_ev)`` `ndarray`
        The sampling of the principal components of the shape matrix using the selected uv triangles
    ctrl : ``(3,)`` `ndarray`
        The control parameters`
    shape_ev : ``(n_ev,)`` `ndarray`
        The shape eigenvalues
    s_pc : ``(160470, n_ev)`` `ndarray`
        The shape principal components
    T_uv : ``(4, n_dims)`` `ndarray`
        The uv sampling of the texture matrix
    shadow_index : ``(n_dims,)`` `ndarray`
        The points in shadow matrix
    triangle_array : ``(3, n_triangles)`` `ndarray`
        The triangle list of the shape
    S :  ``(4, 53490)`` `ndarray`
        The shape matrix
    uv : ``(4, n_dims)`` `ndarray`
        The selected triangles from the projected vertices
    N_uv : ``(4, n_dims)`` `ndarray`
        The selected uv normals
    light_vector : ``(4,)`` `ndarray`
        The light vector
    iota : ``(17,)`` `ndarray`
        The rendering parameters array
    V : ``(4, n_dims)`` `ndarray`
        The viewing vector
    R : ``(4, n_dims)`` `ndarray`
        The reflection vector
    M : ``(4, 4)`` `ndarray`
        The color transformation matrix G*C
    L_dir : ``(4, 4)`` `ndarray`
        The directional light matrix
    L_spec : ``(4, 4)`` `ndarray`
        The specular light matrix
    r_phi : ``(4, 4)`` `ndarray`
        The phi rotation matrix
    r_theta : ``(4, 4)`` `ndarray`
        The theta rotation matrix
    r_varphi : ``(4, 4)`` `ndarray`
        the varphi rotation matrix
    t_pc : ``(160470, n_ev)`` `ndarray`
        The texture principal components
    texture_ev : ``(n_ev,)`` `ndarray`
        The eigenvalues of texture
    L_amb : ``(4, 4)`` `ndarray`
        The ambient light matrix
    diffuse_dot : ``(4, n_dims)`` `ndarray`
        The dot product between the normals and the light vectors
    specular_term : ``(4, n_dims)`` `ndarray`
        The specular term
    diffuse_term : ``(4, n_dims)`` `ndarray`
        The diffuse term
    C : ``(4, 4)`` `ndarray`
        Color contrast matrix
    I : ``(4, n_dims)`` `ndarray`
       Color intensity matrix
    G : ``(4, 4)`` `ndarray`
        Gamma matrix

    Returns
    -------
    dp_dalpha : `(2, n_parameters, n_points)`` `ndarray`
        The derivative of the projection wrt the shape parameters
    dt_dalpha : `(3, n_parameters, n_points)`` `ndarray`
        The derivative of the texture wrt the shape parameters
    dp_drho : `(2, n_parameters, n_points)`` `ndarray`
        The derivative of the projection wrt the warp parameters
    dt_drho : `(3, n_parameters, n_points)`` `ndarray`
        The derivative of the texture wrt the warp parameters
    dt_dbeta : `(3, n_parameters, n_points)`` `ndarray`
        The derivative of the texture wrt the texture parameters
    dt_diota : `(3, n_parameters, n_points)`` `ndarray`
        The derivative of the texture wrt the illumination parameters
    """
    # Function that takes all the parameters and computes all the derivatives within the color cost function
    dp_dalpha, dt_dalpha, dp_drho, dt_drho, dt_dbeta, dt_diota = ([],)*6
    if n_alphas > 0:
        dp_dalpha = compute_projection_derivatives_shape_parameters(s_uv, w_uv, rho, rot, s_pc_uv, ctrl,
                                                                    shape_ev)
        dt_dalpha = compute_texture_derivatives_shape_parameters(s_pc, T, shadow_index, w_uv, triangle_array,
                                                                 S, uv, N_uv, light_vector, shape_ev, rot, s_pc_uv,
                                                                 iota, V, R, M, L_dir, L_spec)
    if n_rhos > 0:
        dp_drho = compute_projection_derivatives_warp_parameters(s_uv, w_uv, rho,
                                                                 r_phi, r_theta, r_varphi, ctrl)
        dt_drho = compute_texture_derivatives_warp_parameters(rho, T, shadow_index, w_uv, light_vector, R, V,
                                                              r_theta, r_phi, r_varphi, N_uv, S, iota, M,
                                                              L_dir, L_spec)
    if n_betas > 0:
        dt_dbeta = compute_texture_derivatives_texture_parameters(t_pc, shadow_index, texture_ev, M,
                                                                  L_amb, L_dir, diffuse_dot, specular_term)
    if n_iotas > 0:
        dt_diota = compute_texture_derivatives_illumination_parameters(iota, T, shadow_index, N_uv, light_vector, R, V,
                                                                       C, I, G, M, diffuse_term, specular_term, L_dir,
                                                                       L_spec)

    return dp_dalpha, dt_dalpha, dp_drho, dt_drho, dt_dbeta, dt_diota


if __name__ == "__main__":
    model = Model()
    mm = model.init_from_basel("model.mat")
    mmf = MorphableModelFitter(mm)

    # TESTS WITH DUMMY VALUES
    mmf.fit("bale.mat")
    S_uv_rand = np.random.rand(4, 1000, 1)
    W_uv_rand = np.random.rand(4, 1000)
    s_pc_uv_rand = np.random.rand(4, 1000, 199)
    # Define control parameters
    ctrl_params = control_parameters(pt=1, vb=True)

    # Define standard fitting parameters
    std_fit_params = standard_fitting_parameters(ctrl_params)
    rho = std_fit_params['rho_array']  # Shape transformation parameters
    iota = std_fit_params['iota_array']

    # Compute warp and projection matrices
    [r_phi, r_theta, r_varphi, rot, view_matrix, projection_matrix] = \
        compute_warp_and_projection_matrices(rho, ctrl_params['projection_type'])

    # print compute_ortho_projection_derivatives_shape_parameters(S_uv_rand, s_pc_uv_rand, rho,
    #                                                             rot, mm.shape_ev)
    # print compute_pers_projection_derivatives_shape_parameters(S_uv_rand, W_uv_rand, s_pc_uv_rand,
    #                                                            rho, rot, mm.shape_ev)
    # print compute_ortho_projection_derivatives_warp_parameters(S_uv_rand, W_uv_rand, rho, r_phi, r_theta, r_varphi)
    # print compute_pers_projection_derivatives_warp_parameters(S_uv_rand, W_uv_rand, rho, r_phi, r_theta, r_varphi)

    shadow_index = np.random.rand(1000)
    uv = np.random.randint(100000, size=(4, 1000))
    N = np.random.rand(4, 1000)
    V = np.random.rand(4, 1000)
    R = np.random.rand(4, 1000)
    T = np.random.rand(4, 1000)
    light_vector = np.random.rand(4)
    M = np.random.rand(4, 4)
    L_dir = np.random.rand(4, 4)
    L_spec = np.random.rand(4, 4)
    L_amb = np.random.rand(4, 4)
    s_pc = np.random.rand(4, 53490, 199)
    s = np.random.rand(4, 53490)
    C = np.random.rand(4, 4)
    I = np.random.rand(4, 1000)
    G = np.random.rand(4, 4)
    diffuse_term = np.random.rand(4, 1000)
    specular_term = np.random.rand(4, 1000)

    # res = compute_texture_derivatives_shape_parameters(s_pc, T, shadow_index, W_uv_rand, mm.triangle_array,
    #                                                    s, uv, N, light_vector, mm.shape_ev, rot, s_pc_uv_rand,
    #                                                    iota, V, R, M, L_dir, L_spec)
    # res2 = compute_texture_derivatives_warp_parameters(rho, T, shadow_index, W_uv_rand, light_vector, R, V,
    #                                           r_theta, r_phi, r_varphi,
    #                                           N, N, s, iota, M, L_dir, L_spec)

    # res3 = compute_texture_derivatives_illumination_parameters(iota, T, shadow_index, N, light_vector, R, V,
    #                                                           C, I, G, M, diffuse_term, specular_term, L_dir, L_spec)

    t_pc = np.random.rand(4, 61104, 199)
    diffuse_dot = np.random.rand(4, 1000)
    res4 = compute_texture_derivatives_texture_parameters(t_pc, shadow_index, mm.texture_ev, M,
                                                          L_amb, L_dir, diffuse_dot, specular_term)

    print(res4)




