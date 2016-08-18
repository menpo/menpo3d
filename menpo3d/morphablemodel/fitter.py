import sys
import menpo.io as mio
import numpy as np
import matplotlib.pyplot as plt
from .lmalign import retrieve_view_projection_transforms
from .detectandfit import detect_and_fit
#from .irenderer import InverseRenderer, params_from_ir
import menpo3d.io as m3dio
from menpo.feature import gradient
from menpo.image import Image
from menpo3d.rasterize import GLRasterizer
from menpo.transform import Homogeneous, UniformScale
from .derivatives import compute_pers_projection_derivatives_shape_parameters, \
                              compute_ortho_projection_derivatives_shape_parameters, \
                              compute_ortho_projection_derivatives_warp_parameters, \
                              compute_pers_projection_derivatives_warp_parameters, \
                              compute_texture_derivatives_texture_parameters


class MMFitter(object):
    r"""
    Class for defining a 3DMM fitter.

    """
    def __init__(self, mm):
        self._model = mm
        self.result = None
        self.errors = []

    @property
    def mm(self):
        r"""
        The 3DMM model.

        """
        return self._model

    @property
    def get_result(self):
        r"""
        The 3DMM model fitting result.

        """
        return self.result

    # TODO
    def fit(self, image_path, n_alphas, n_betas, n_tris, max_iters=50):

        # PRE-COMPUTATIONS
        # ----------------
        # Constants
        threshold = 1e-2

        # Import and fit aam on image
        image = mio.import_image(image_path)
        detect_and_fit(image)

        # Compute gradient
        image.pixels = image.pixels.astype(np.double)
        # print image.pixels
        grad = gradient(image)
        # scaling by image resolution solves the not plausible face shapes problem
        VI_dy = grad.pixels[:3] * image.shape[1] / 2
        VI_dx = grad.pixels[3:] * image.shape[0] / 2

        # Generate instance
        instance = self._model.instance()

        # Get view projection rotation matrices
        view_t, proj_t, R = retrieve_view_projection_transforms(image, instance, group='ibug68')

        # Get camera parameters array
        rho_array = rho_from_view_projection_matrices(proj_t.h_matrix, R.h_matrix)

        # Prior probabilities constants
        alpha = np.zeros(n_alphas)
        beta = np.zeros(n_betas)

        dalpha_prior_dalpha = 2. / np.square(self._model.shape_model.eigenvalues[:n_alphas])
        #dalpha_prior_drho = np.zeros(len(rho_array))
        dalpha_prior_dbeta = np.zeros(n_betas)
        #SD_alpha_prior = np.concatenate((dalpha_prior_dalpha, dalpha_prior_drho, dalpha_prior_dbeta))
        SD_alpha_prior = np.concatenate((dalpha_prior_dalpha, dalpha_prior_dbeta))
        H_alpha_prior = np.diag(SD_alpha_prior)

        dbeta_prior_dalpha = np.zeros(n_alphas)
        #dbeta_prior_drho = np.zeros(len(rho_array))
        dbeta_prior_dbeta = 2. / np.square(self._model.texture_model.eigenvalues[:n_betas])
        #SD_beta_prior = np.concatenate((dbeta_prior_dalpha, dbeta_prior_drho, dbeta_prior_dbeta))
        SD_beta_prior = np.concatenate((dbeta_prior_dalpha, dbeta_prior_dbeta))
        H_beta_prior = np.diag(SD_beta_prior)
        
        # Initilialize inverse renderer
        rasterizer = GLRasterizer(height=image.height, width=image.width,
                                  view_matrix=view_t.h_matrix,
                                  projection_matrix=proj_t.h_matrix)
        
        errors = []
        eps = np.inf
        k = 0
        

        while k < max_iters and eps > threshold:
            
            # Progress bar
            progress = k*100/max_iters
            sys.stdout.write("\r%d%%" % progress)
            sys.stdout.flush()

            r_phi, r_theta, r_varphi = compute_rotation_matrices(rho_array)

            # Pre-computations
            tri_index_img, b_coords_img = rasterizer.rasterize_barycentric_coordinate_image(instance)
            tri_indices = tri_index_img.as_vector() 
            b_coords = b_coords_img.as_vector(keep_channels=True) 
            yx = tri_index_img.mask.true_indices() 

            #ir_image = ir.inverse_render(instance)

            #inverse_image = ir_image[1]
            #yx = inverse_image.mask.true_indices()

            #b_coords, tri_indices = params_from_ir(inverse_image)
            
            rand = np.random.permutation(b_coords.shape[1])
            b_coords = b_coords[:, rand[:n_tris]]
            yx = yx[rand[:n_tris]]
            tri_indices = tri_indices[rand[:n_tris]]

            # build the vertex indices (3 per pixel)
            # for the visible triangle
            vertex_indices = instance.trilist[tri_indices]

            W = view_t.apply(instance.points)
            shape_pc = self._model.shape_model.components.T
            tex_pc = self._model.texture_model.components.T

            # Reshape before sampling
            shape_pc = shape_pc.reshape([instance.n_points, -1])
            tex_pc = tex_pc.reshape([instance.n_points, -1])

            # Compute samples
            #norms_uv = sample_object(instance.vertex_normals(), vertex_indices, b_coords)  # normals
            shape_uv = sample_object(instance.points, vertex_indices, b_coords)  # shape
            tex_uv = sample_object(instance.colours, vertex_indices, b_coords)  # texture
            warped_uv = sample_object(W, vertex_indices, b_coords)  # shape multiplied by view matrix
            shape_pc_uv = sample_object(shape_pc, vertex_indices, b_coords)  # shape eigenvectors
            tex_pc_uv = sample_object(tex_pc, vertex_indices, b_coords)  # texture eigenvectors
            img_uv = sample_image(image, yx)  # image
            VI_dx_uv = Image(VI_dx).sample(yx)
            VI_dy_uv = Image(VI_dy).sample(yx)

            # Reshape after sampling
            new_shape = tex_pc_uv.shape
            shape_pc_uv = shape_pc_uv.reshape([new_shape[0], 3, -1])
            tex_pc_uv = tex_pc_uv.reshape([new_shape[0], 3, -1])

            # DERIVATIVES

            # Orthographic projection derivative wrt shape parameters
            dop_dalpha = compute_ortho_projection_derivatives_shape_parameters(shape_uv, shape_pc_uv, rho_array,
                                                                               R, self._model.shape_model.eigenvalues)
            
            dop_dalpha = dop_dalpha[:, :n_alphas, :]
            

            # Perspective projection derivative wrt shape parameters
            # dpp_dalpha = compute_pers_projection_derivatives_shape_parameters(shape_uv, warped_uv.T, shape_pc_uv,
            #                                                                   rho_array, R, shape_model.eigenvalues)

            # Texture derivative wrt texture parameters
            dt_dbeta = compute_texture_derivatives_texture_parameters(tex_pc_uv, self._model.texture_model.eigenvalues)
            dt_dbeta = dt_dbeta[:, :n_betas, :]

            # Orthographic projection derivative wrt warp parameters
            dop_drho = compute_ortho_projection_derivatives_warp_parameters(shape_uv, warped_uv.T, rho_array,
                                                                            r_phi, r_theta, r_varphi)
            

            # Perspective projection derivative wrt warp parameters
            # dpp_drho = compute_pers_projection_derivatives_warp_parameters(shape_uv, warped_uv.T, rho_array,
            #                                                               r_phi, r_theta, r_varphi)

            # Compute sd matrix and hessian
            #dp_dgamma = np.hstack((dop_dalpha, dop_drho))
            dt = -dt_dbeta
            SD_gamma = compute_sd(dop_dalpha, VI_dx_uv, VI_dy_uv)
            SD_img = np.hstack((SD_gamma, dt))
            H_img = compute_hessian(SD_img)

            # Compute error
            img_error_uv = img_uv - tex_uv.T
            eps = (img_error_uv ** 2).mean()
            errors.append(eps)
            SD_error_img = compute_sd_error(SD_img, img_error_uv)

            # Prior probabilities over shape parameters
            #alpha_prior_error = np.concatenate((alpha, rho_array, beta))
            alpha_prior_error = np.concatenate((alpha, beta))
            SD_error_alpha_prior = np.multiply(SD_alpha_prior, alpha_prior_error)

            # Prior probabilities over texture parameters
            #beta_prior_error = np.concatenate((alpha, rho_array, beta))
            beta_prior_error = np.concatenate((alpha, beta))
            SD_error_beta_prior = np.multiply(SD_beta_prior, beta_prior_error)

            # Final hessian and SD error matrix
            H = H_img + 0.005*H_alpha_prior + 10e6 * H_beta_prior
            SD_error = SD_error_img + 0.005*SD_error_alpha_prior + 10e6 * SD_error_beta_prior

            # Compute update
            delta_sigma = -np.dot(np.linalg.inv(H), SD_error)

            # Update parameters
            alpha += delta_sigma[:n_alphas]
            #rho_array += delta_sigma[len(alpha):len(alpha) + len(rho_array)]
            #beta += delta_sigma[len(alpha) + len(rho_array):len(alpha) + len(rho_array) + len(beta)]
            beta += delta_sigma[n_alphas:]
            

            # Generate the updated instance
            instance = self._model.instance(alpha=alpha, beta=5e2*beta)

            # Generate the new view and projection matrices (the projection matrix is fixed for now)
            #view_t_flipped, R = compute_view_matrix(rho_array)
            #view_t.h_matrix[1:3, :3] = -R.h_matrix[1:3, :3]
            #view_t.h_matrix[0, :3] = R.h_matrix[0, :3]
            
            # Update the inverse renderer
            #rasterizer.set_view_matrix(view_t.h_matrix)
            
            # View per-iteration result 
            #result = rasterizer.rasterize_mesh(instance)
            
            # Hacky blue pixels fix 
            #result.pixels = np.clip(result.pixels, 0, 1)
            #result.view()
            
            k += 1
        
        sys.stdout.write('\rSuccessfully fitted.')
        #render = GLRasterizer(image.height, image.width, view_matrix=view_t.h_matrix,
        #                         projection_matrix=proj_t.h_matrix)
        result = rasterizer.rasterize_mesh(instance)
        result.pixels = np.clip(result.pixels, 0, 1)
        self.result = result
        #self.result = instance
        self.errors = errors
        

    def visualize_result(self):
        self.result.view()
    
    def plot_error(self, image_name):
        plt.plot(self.errors)
        plt.ylabel('Error')
        plt.title(image_name)
        plt.show()


def sample_object(x, vertex_indices, b_coords):
    per_vert_per_pixel = x[vertex_indices]
    return np.sum(per_vert_per_pixel *
                  b_coords.T[..., None], axis=1)


def sample_image(image, yx):
    # how do we do this?
    # [img_uv_rand] = sampleImageAtUV(img, yx_rand);
    image_uv = image.sample(yx)
    return image_uv


def rho_from_view_projection_matrices(proj_t, view_t):
    rho = np.zeros(6)

    # PROJECTION MATRIX PARAMETERS
    # The focal length is the first diagonal element of the projection matrix
    # For the moment this is not optimised
    rho[0] = proj_t[0, 0]

    # VIEW MATRIX PARAMETERS
    # Euler angles
    # For the case of cos(theta) != 0, we have two triplets of Euler angles
    # we will only give one of the two solutions
    if view_t[2, 0] != 1 or view_t[2, 0] != -1:
        theta = np.pi + np.arcsin(view_t[2, 0])
        varphi = np.arctan2(view_t[2, 1] / np.cos(theta), view_t[2, 2] / np.cos(theta))
        phi = np.arctan2(view_t[1, 0] / np.cos(theta), view_t[0, 0] / np.cos(theta))
        rho[1] = varphi
        rho[2] = theta
        rho[3] = phi

    # Translations
    rho[4] = -view_t[0, 3]  # tw x
    rho[5] = -view_t[1, 3]  # tw y

    return rho


def compute_rotation_matrices(rho_array):
    rot_varphi = np.eye(4)

    rot_varphi[1:3, 1:3] = np.array([[np.cos(rho_array[1]), np.sin(rho_array[1])],
                                     [-np.sin(rho_array[1]), np.cos(rho_array[1])]])
    rot_theta = np.eye(4)
    rot_theta[0:3, 0:3] = np.array([[np.cos(rho_array[2]), 0, np.sin(rho_array[2])],
                                    [0, 1, 0],
                                    [-np.sin(rho_array[2]), 0, np.cos(rho_array[2])]])
    rot_phi = np.eye(4)
    rot_phi[0:2, 0:2] = np.array([[np.cos(rho_array[3]), -np.sin(rho_array[3])],
                                  [np.sin(rho_array[3]), np.cos(rho_array[3])]])

    r_phi = Homogeneous(rot_phi)
    r_theta = Homogeneous(rot_theta)
    r_varphi = Homogeneous(rot_varphi)

    return r_phi, r_theta, r_varphi


def compute_view_matrix(rho):
    axes_flip_matrix = np.eye(4)
    axes_flip_matrix[1, 1] = -1
    axes_flip_matrix[2, 2] = -1
    axes_flip_t = Homogeneous(axes_flip_matrix)

    view_t = np.eye(4)
    view_t[0, 3] = -rho[4]  # tw x
    view_t[1, 3] = -rho[5]  # tw y
    r_phi, r_theta, r_varphi = compute_rotation_matrices(rho)
    rotation = np.dot(np.dot(r_phi.h_matrix, r_varphi.h_matrix), r_theta.h_matrix)
    view_t[:3, :3] = rotation[:3, :3]

    # view_t = Homogeneous(view_t).compose_before(axes_flip_t)

    return Homogeneous(view_t), Homogeneous(rotation)


def compute_sd(dp_dgamma, VI_dx_uv, VI_dy_uv):
    # SD_x
    expanded_vi_dx = np.expand_dims(VI_dx_uv, axis=2)
    permuted_vi_dx = np.transpose(expanded_vi_dx, (0, 2, 1))
    new_vi_dx = np.tile(permuted_vi_dx, (1, dp_dgamma.shape[1], 1))
    SD_x = np.multiply(new_vi_dx, np.tile(dp_dgamma[0, :, :], (3, 1, 1)))

    # SD_y
    expanded_vi_dy = np.expand_dims(VI_dy_uv, axis=2)
    permuted_vi_dy = np.transpose(expanded_vi_dy, (0, 2, 1))
    new_vi_dy = np.tile(permuted_vi_dy, (1, dp_dgamma.shape[1], 1))
    SD_y = np.multiply(new_vi_dy, np.tile(dp_dgamma[1, :, :], (3, 1, 1)))

    return SD_x + SD_y


def compute_hessian(sd):
    # Computes the hessian as defined in the Lucas Kanade Algorithm
    n_channels = np.size(sd[:, 0, 1])
    n_params = np.size(sd[0, :, 0])
    h = np.zeros([n_params, n_params])
    sd = np.transpose(sd, [2, 1, 0])
    for i in xrange(n_channels):
        h_i = np.dot(np.transpose(sd[:, :, i]), sd[:, :, i])
        h += h_i
    return h


def compute_sd_error(sd, error_uv):
    n_channels = np.size(sd, 0)
    n_parameters = np.size(sd, 1)

    sd = np.transpose(sd, (2, 1, 0))
    sd_error_product = np.zeros(n_parameters)

    for i in xrange(n_channels):
        sd_error_prod_i = np.dot(error_uv[i, :], sd[:, :, i])
        sd_error_product += sd_error_prod_i

    return sd_error_product

