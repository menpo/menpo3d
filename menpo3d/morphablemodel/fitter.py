import numpy as np
import menpo.io as mio


class Fitter:

    # The fitter does not take the image in the constructor as it is supposed
    # to run fit using an input image on an object constructed without the information about the image
    def __init__(self, model, std_fit_params, fit_params, ctrl_params):
        self.model = model
        self.fit_params = fit_params
        self.std_fit_params = std_fit_params
        self.ctrl_params = ctrl_params
        self.rot = {
            'rot_phi': [],
            'rot_theta': [],
            'rot_varphi': [],
            'rot_total': []
        }
        self.view_matrix = []
        self.projection_matrix = []

    def pre_calculations(self):
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
        print self.fit_params

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
    def compute_projection_derivatives_shape_parameters(self):
        if self.ctrl_params['projection_type'] == 0:
            dp_dgamma = self.compute_ortho_proj_derivatives_shape_params()
        else:
            dp_dgamma = self.compute_pers_proj_derivatives_shape_params()
        return dp_dgamma

    # TODO
    def compute_projection_derivatives_warp_parameters(self):
        if self.ctrl_params['projection_type'] == 0:
            dp_dgamma = self.compute_ortho_warp_derivatives_shape_params()
        else:
            dp_dgamma = self.compute_pers_warp_derivatives_shape_params()
        return dp_dgamma

    # TODO
    def compute_sd_error_product(self, sd_anchor, error_uv):
        return 0

    # TODO
    def warp(self, shape):
        return 0

    # TODO
    def project(self, warp):
        return 0

    # TODO
    def anchor_cost_func(self, shape, s_pc):
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

        dp_alpha = self.compute_projection_derivatives_shape_parameters()
        dp_rho = self.compute_projection_derivatives_warp_parameters()
        dp_beta = []
        dp_iota = []

        # Compute steepest descent matrix and hessian
        sd_anchor = []
        h_anchor = hessian(sd_anchor)
        sd_error_product_anchor = self.compute_sd_error_product(sd_anchor, anchor_error_pixel)

        return [sd_anchor, h_anchor, sd_error_product_anchor]

    # TODO
    def fit(self, image):
        [vi_dx, vi_dy, s_pc, t_pc] = self.pre_calculations()

        # Simultaneous Forwards Additive Algorithm
        for i in xrange(self.fit_params['max_iters']):
            # Compute shape and texture
            shape = self.compute_shape()
            texture = self.compute_texture()

            # Compute warp and projection matrices
            self.compute_warp_and_projection_matrices()

            if self.fit_params['img_cost_func']:
                pass
            if self.fit_params['anchor_cost_func']:
                self.anchor_cost_func(shape, s_pc)
            if self.fit_params['prior_alpha_cost_func']:
                pass
            if self.fit_params['prior_beta_cost_func']:
                pass
            if self.fit_params['texture_cost_func']:
                pass

            if self.ctrl_params['visualize'] > 0:
                self.visualize()

            # Update parameters
            delta_sigma = self.update_parameters()

            # Check for convergence
            if np.linalg.norm(delta_sigma) < self.fit_params['cvg_thresh']:
                break

        self.save_final_params()


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
