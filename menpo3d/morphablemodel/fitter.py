import numpy as np
import menpo.io as mio
from menpo.image import Image
from menpofit.fitter import MultiScaleParametricFitter
from lk import SimultaneousForwardAdditive


class MMFitter(MultiScaleParametricFitter):
    r"""
    Abstract class for defining an 3DMM fitter.

    """
    def __init__(self, mm, algorithms):
        self._model = mm
        # Call superclass
        super(MMFitter, self).__init__(
            scales=mm.scales, reference_shape=mm.reference_shape,
            holistic_features=mm.holistic_features, algorithms=algorithms)

    @property
    def mm(self):
        r"""
        The trained 3DMM model.

        """
        return self._model

    def _fitter_result(self, image, algorithm_results, affine_transforms,
                       scale_transforms, gt_shape=None):
        r"""
        Function the creates the multi-scale fitting result object.
        """
        return 0


class LucasKanadeMMFitter(MMFitter):

    # The fitter does not take the image in the constructor as it is supposed
    # to run fit using an input image on an object constructed without the information about the image
    def __init__(self, model):
        # Assign attributes
        self.model = model
        self.rot = {
            'rot_phi': [],
            'rot_theta': [],
            'rot_varphi': [],
            'rot_total': []
        }
        self.view_matrix = []
        self.projection_matrix = []

    @staticmethod
    def control_parameters(pt=0, vb=False, vl=0):
        ctrl_params = {
            'projection_type': pt,  # 0: weak perspective, 1: perspective
            'verbose': vb,  # Console information: False:  off, True: on
            'visualize': vl  # Visualize fitting: 0: off, 1: on, 3: renders all visible points
        }
        return ctrl_params

    @staticmethod
    def fitting_parameters(max_iters, n_points, img_cost_func,
                           anchor_cost_func, prior_alpha_cost_func, prior_beta_cost_func,
                           texture_cost_func, n_alphas, n_rhos, n_betas, n_iotas,
                           cvg_thresh, w_img, w_anchor, w_alpha, w_beta, w_texture):

        # FITTING 1 - Alignment
        fit_params = {
            'max_iters': max_iters,  # number of iteration: -1 : until convergence
            'n_points': n_points,  # number of points
            # Cost function activation
            'img_cost_func': img_cost_func,
            'anchor_cost_func': anchor_cost_func,
            'prior_alpha_cost_func': prior_alpha_cost_func,
            'prior_beta_cost_func': prior_beta_cost_func,
            'texture_cost_func': texture_cost_func,
            # parameters
            'n_alphas': n_alphas,
            'n_rhos': n_rhos,
            'n_betas': n_betas,
            'n_iotas': n_iotas,
            'cvg_thresh': cvg_thresh,
            # weights
            'w_img': w_img,
            'w_anchor': w_anchor,
            'w_alpha': w_alpha,
            'w_beta': w_beta,
            'w_texture': w_texture
        }
        return fit_params

    def basic_alignment_fitting(self, image, ctrl_params, std_fit_params):
        fit_params = self.fitting_parameters(20, -1, False, True, False, False, False, -1, [1, 2, 5, 6], -1,
                                             -1, 10 ** -3, 1, 1, 1, 1, 1)
        self.run_lucas_kanade(image, std_fit_params, fit_params, ctrl_params)

    # TODO
    def basic_shape_fitting(self, image, ctrl_params, std_fit_params):
        fit_params = self.fitting_parameters(20, -1, False, True, False, False, False, -1, [1, 2, 5, 6], -1,
                                             -1, 10 ** -3, 1, 1, 1, 1, 1)
        self.run_lucas_kanade(image, std_fit_params, fit_params, ctrl_params)

    # TODO
    def lighting_fitting(self, image, ctrl_params, std_fit_params):
        fit_params = self.fitting_parameters(20, -1, False, True, False, False, False, -1, [1, 2, 5, 6], -1,
                                             -1, 10 ** -3, 1, 1, 1, 1, 1)
        self.run_lucas_kanade(image, std_fit_params, fit_params, ctrl_params)

    # TODO
    def texture_fitting(self, image, ctrl_params, std_fit_params):
        fit_params = self.fitting_parameters(20, -1, False, True, False, False, False, -1, [1, 2, 5, 6], -1,
                                             -1, 10 ** -3, 1, 1, 1, 1, 1)
        self.run_lucas_kanade(image, std_fit_params, fit_params, ctrl_params)

    def precompute(self):
        [vi_dx, vi_dy, s_pc, t_pc] = [0]*4
        return [vi_dx, vi_dy, s_pc, t_pc]

    # TODO: model2object
    def compute_shape(self):
        shape = []
        return shape

    # TODO: model2object
    def compute_texture(self):
        texture = []
        return texture

    # TODO
    def compute_warp_and_projection_matrices(self):
        self.rot = {}
        self.view_matrix = []
        self.projection_matrix = []

    # TODO
    def visualize(self):
        print self.view_matrix

    # TODO
    def update_parameters(self):
        self.view_matrix = []
        delta_sigma = 0
        return delta_sigma

    # TODO
    def save_final_params(self):
        self.view_matrix = []

    # TODO
    def img_cost_func(self):
        print "image cost function"

    # TODO
    def compute_anchor_points_projection(self, anchor_array, projection):
        [uv_anchor, yx_anchor] = [0]*2
        return [uv_anchor, yx_anchor]

    # TODO
    def compute_anchor_points_error(self, yx_anchor):
        return 0

    # TODO
    def sample_object_at_uv(self, shape, anchor_array, uv_anchor):
        return 0

    # TODO
    def compute_ortho_proj_derivatives_shape_params(self):
        return 0

    # TODO
    def compute_pers_proj_derivatives_shape_params(self):
        return 0

    # TODO
    def compute_ortho_warp_derivatives_shape_params(self):
        return 0

    # TODO
    def compute_pers_warp_derivatives_shape_params(self):
        return 0

    # TODO
    def compute_projection_derivatives_shape_parameters(self, ctrl_params):
        if ctrl_params['projection_type'] == 0:
            dp_dgamma = self.compute_ortho_proj_derivatives_shape_params()
        else:
            dp_dgamma = self.compute_pers_proj_derivatives_shape_params()
        return dp_dgamma

    # TODO
    def compute_projection_derivatives_warp_parameters(self, ctrl_params):
        if ctrl_params['projection_type'] == 0:
            dp_dgamma = self.compute_ortho_warp_derivatives_shape_params()
        else:
            dp_dgamma = self.compute_pers_warp_derivatives_shape_params()
        return dp_dgamma

    # TODO
    def compute_sd_error_product(self, sd_anchor, error_uv):
        return 0

    # TODO
    def warp(self, image):
        r"""
            Warps an image into the template's mask.

            Parameters
            ----------
            image : `menpo.image.Image` or subclass
                The input image to be warped.

            Returns
            -------
            warped_image : `menpo.image.Image` or subclass
                The warped image.
        """
        return 0

    # TODO
    def project(self, warp):
        return 0

    # TODO
    def anchor_cost_func(self, shape, s_pc, ctrl_params):
        # Compute anchor points projection
        anchor_array = self.model.triangle_array
        warp = self.warp(shape)
        projection = self.project(warp)
        [uv_anchor, yx_anchor] = self.compute_anchor_points_projection(anchor_array, projection)

        # Compute anchor points error
        anchor_error_pixel = self.compute_anchor_points_error(yx_anchor)

        # Compute the derivatives
        s_uv_anchor = self.sample_object_at_uv(shape, anchor_array, uv_anchor)
        w_uv_anchor = self.sample_object_at_uv(warp, anchor_array, uv_anchor)
        s_pc_uv_anchor = self.sample_object_at_uv(s_pc, anchor_array, uv_anchor)

        dp_alpha = self.compute_projection_derivatives_shape_parameters(ctrl_params)
        dp_rho = self.compute_projection_derivatives_warp_parameters(ctrl_params)
        dp_beta = []
        dp_iota = []

        # Compute steepest descent matrix and hessian
        sd_anchor = []
        h_anchor = hessian(sd_anchor)
        sd_error_product_anchor = self.compute_sd_error_product(sd_anchor, anchor_error_pixel)

        return [sd_anchor, h_anchor, sd_error_product_anchor]

    # TODO
    def run(self, image, std_fit_params, fit_params, ctrl_params):

        [vi_dx, vi_dy, s_pc, t_pc] = self.precompute()

        # Simultaneous Forwards Additive Algorithm
        for i in xrange(fit_params['max_iters']):
            # Compute shape and texture
            shape = self.compute_shape()
            texture = self.compute_texture()

            # Compute warp and projection matrices
            self.compute_warp_and_projection_matrices()

            if fit_params['img_cost_func']:
                pass
            if fit_params['anchor_cost_func']:
                self.anchor_cost_func(shape, s_pc, ctrl_params)
            if fit_params['prior_alpha_cost_func']:
                pass
            if fit_params['prior_beta_cost_func']:
                pass
            if fit_params['texture_cost_func']:
                pass

            if ctrl_params['visualize'] > 0:
                self.visualize()

            # Update parameters
            delta_sigma = self.update_parameters()

            # Check for convergence
            if np.linalg.norm(delta_sigma) < fit_params['cvg_thresh']:
                break

        self.save_final_params()

    # TODO
    def fit(self, image):

        # Define common fitting and control parameters
        ctrl_params = self.control_parameters(pt=1, vb=True, vl=3)

        # Initialize fitting parameters
        std_fit_params = standard_fitting_parameters(ctrl_params)

        # Basic alignment fitting
        self.basic_alignment_fitting(image, ctrl_params, std_fit_params)

        # Basic shape fitting
        self.basic_shape_fitting(image, ctrl_params, std_fit_params)

        # Lighting fitting
        self.lighting_fitting(image, ctrl_params, std_fit_params)

        # Texture fitting
        self.texture_fitting(image, ctrl_params, std_fit_params)


def hessian(sd):
    # Computes the hessian as defined in the Lucas Kanade Algorithm
    """n_channels = np.size(sd[:, 0, 1])
    n_params = np.size(sd[0, :, 0])
    h = np.zeros((n_params, n_params))
    sd = np.transpose(sd, [2, 1, 0])
    for i in xrange(n_channels):
        h_i = np.dot(np.transpose(sd[:, :, i]), sd[:, :, i])
        h += h_i"""
    return 0


def standard_fitting_parameters(params):

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
    iota_array = [1, 1, 1, 1, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 30, 40]

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
