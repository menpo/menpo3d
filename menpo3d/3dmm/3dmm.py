import numpy as np
import menpo.io as mio
from scipy.io import loadmat
from model import Model
from fitter import Fitter


class Menpo3DMM:
    # TODO
    def __init__(self, image, template, fitting, epsilon):
        self.image = image
        self.template = template
        self.fitting = fitting
        self.epsilon = epsilon

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
            'n_points': n_points,   # number of points
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

    def basic_alignment_fitting(self, model, ctrl_params, std_fit_params):
        fit_params = self.fitting_parameters(20, -1, False, True, False, False, False, -1, [1, 2, 5, 6], -1,
                                             -1, 10**-3, 1, 1, 1, 1, 1)
        fitter = Fitter(self.image, model, std_fit_params, fit_params, ctrl_params)
        self.fitting = fitter.fit()

    # TODO
    def basic_shape_fitting(self, model, ctrl_params, std_fit_params):
        fit_params = self.fitting_parameters(20, -1, False, True, False, False, False, -1, [1, 2, 5, 6], -1,
                                             -1, 10 ** -3, 1, 1, 1, 1, 1)
        fitter = Fitter(self.image, model, std_fit_params, fit_params, ctrl_params)
        self.fitting = fitter.fit()

    # TODO
    def lighting_fitting(self, model, ctrl_params, std_fit_params):
        fit_params = self.fitting_parameters(20, -1, False, True, False, False, False, -1, [1, 2, 5, 6], -1,
                                             -1, 10 ** -3, 1, 1, 1, 1, 1)
        fitter = Fitter(self.image, model, std_fit_params, fit_params, ctrl_params)
        self.fitting = fitter.fit()

    # TODO
    def texture_fitting(self, model, ctrl_params, std_fit_params):
        fit_params = self.fitting_parameters(20, -1, False, True, False, False, False, -1, [1, 2, 5, 6], -1,
                                             -1, 10 ** -3, 1, 1, 1, 1, 1)
        fitter = Fitter(self.image, model, std_fit_params, fit_params, ctrl_params)
        self.fitting = fitter.fit()

    # TODO
    def run(self, anchors_pf, model_pf):
        # Perform the Lucas-Kanade algorithm on the 3D Morphable Mode

        # Load model and image
        self.image = loadmat(anchors_pf)
        model = Model(model_pf)
        self.template = model

        # Define common fitting and control parameters
        ctrl_params = self.control_parameters(pt=1, vb=True, vl=3)

        # Initialize fitting parameters
        std_fit_params = standard_fitting_parameters(ctrl_params)

        # Basic alignment fitting
        self.basic_alignment_fitting(model, ctrl_params, std_fit_params)

        # Basic shape fitting
        self.basic_shape_fitting(model, ctrl_params, std_fit_params)

        # Lighting fitting
        self.lighting_fitting(model, ctrl_params, std_fit_params)

        # Texture fitting
        self.texture_fitting(model, ctrl_params, std_fit_params)

        print self.fitting


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


if __name__=="__main__":
    menpo3d = Menpo3DMM(0, 0.1, 0, 0)
    menpo3d.run("bale.mat", "model.mat")


















