from __future__ import division
import sys
import numpy as np

from menpo.feature import gradient
from menpo.image import Image
from menpo.transform import Homogeneous
from menpo3d.rasterize import GLRasterizer

from .lmalign import retrieve_view_projection_transforms
from .derivatives import (compute_texture_derivatives_texture_parameters,
                          compute_projection_derivatives_warp_parameters,
                          compute_projection_derivatives_shape_parameters)


LM_GROUP = '__3dmm_fit'


class MMFitter(object):
    r"""
    Class for defining a 3DMM fitter.

    """
    def __init__(self, mm):
        self.model = mm

    def fit_from_shape(self, image, shape, n_alphas=100, n_betas=100,
                       n_tris=1000, camera_update=False, max_iters=100):

        # PRE-COMPUTATIONS
        # ----------------
        # Constants
        threshold = 1e-3
        # Projection type: 1 is for perspective and 0 for orthographic
        projection_type = 1

        # store the landmarks
        image.landmarks[LM_GROUP] = shape

        # Compute the gradient
        grad = gradient(image)
        
        # scaling the gradient by image resolution solves the non human-like
        # faces problem
        if image.shape[1] > image.shape[0]:
            scale = image.shape[1]/2
        else:
            scale = image.shape[0]/2

        VI_dy = grad.pixels[:3] * scale
        VI_dx = grad.pixels[3:] * scale

        # Generate instance
        instance = self.model.instance(landmark_group=LM_GROUP)

        # Get view projection rotation matrices
        view_t, proj_t, R = retrieve_view_projection_transforms(image, instance,
                                                                group=LM_GROUP)

        # Get camera parameters array
        rho = rho_from_view_projection_matrices(proj_t.h_matrix,
                                                      R.h_matrix)

        # Prior probabilities constants
        alpha = np.zeros(n_alphas)
        beta = np.zeros(n_betas)

        da_prior_db = np.zeros(n_betas)
        da_prior_da = 2. / (self.model.shape_model.eigenvalues[:n_alphas] ** 2)
        
        db_prior_da = np.zeros(n_alphas)
        db_prior_db = 2. / (self.model.texture_model.eigenvalues[:n_betas] ** 2)
        
        if camera_update:
            prior_drho = np.zeros(len(rho))
            SD_alpha_prior = np.concatenate((da_prior_da, prior_drho,
                                             da_prior_db))
            SD_beta_prior = np.concatenate((db_prior_da, prior_drho,
                                            db_prior_db))
        else:
            SD_alpha_prior = np.concatenate((da_prior_da,
                                             da_prior_db))
            SD_beta_prior = np.concatenate((db_prior_da,
                                            db_prior_db))
            
        H_alpha_prior = np.diag(SD_alpha_prior)
        H_beta_prior = np.diag(SD_beta_prior)
        
        # Initilialize rasterizer
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

            # Rotation matrices 
            r_phi, r_theta, r_varphi = compute_rotation_matrices(rho)
            
            # Inverse rendering
            tri_index_img, b_coords_img = (
                rasterizer.rasterize_barycentric_coordinate_image(instance))
            tri_indices = tri_index_img.as_vector() 
            b_coords = b_coords_img.as_vector(keep_channels=True) 
            yx = tri_index_img.mask.true_indices()

            # Select triangles randomly
            rand = np.random.permutation(b_coords.shape[1])
            b_coords = b_coords[:, rand[:n_tris]]
            yx = yx[rand[:n_tris]]
            tri_indices = tri_indices[rand[:n_tris]]

            # Build the vertex indices (3 per pixel)
            # for the visible triangle
            vertex_indices = instance.trilist[tri_indices]
            
            # Warp the shape witht the view matrix
            W = view_t.apply(instance.points)
            
            # This solves the perspective projection problems
            # It cancels the axes flip done in the view matrix before the
            # rasterization
            W[:, 1:] *= -1
            
            # Shape and texture principal components are reshaped before
            # sampling
            shape_pc = self.model.shape_model.components.T
            tex_pc = self.model.texture_model.components.T
            shape_pc = shape_pc.reshape([instance.n_points, -1])
            tex_pc = tex_pc.reshape([instance.n_points, -1])

            # Sampling
            # norms_uv = sample_object(instance.vertex_normals(),
            #                          vertex_indices, b_coords)
            shape_uv = sample_object(instance.points, vertex_indices, b_coords)
            tex_uv = sample_object(instance.colours, vertex_indices, b_coords)
            warped_uv = sample_object(W, vertex_indices, b_coords)
            shape_pc_uv = sample_object(shape_pc, vertex_indices, b_coords)
            tex_pc_uv = sample_object(tex_pc, vertex_indices, b_coords)
            img_uv = image.sample(yx)  # image
            VI_dx_uv = Image(VI_dx).sample(yx)  # gradient along x
            VI_dy_uv = Image(VI_dy).sample(yx)  # gradient along y

            # Reshape after sampling
            new_shape = tex_pc_uv.shape
            shape_pc_uv = shape_pc_uv.reshape([new_shape[0], 3, -1])
            tex_pc_uv = tex_pc_uv.reshape([new_shape[0], 3, -1])       

            # DERIVATIVES
            dop_dalpha = []
            dpp_dalpha = []
            dt_dbeta = []
            dop_drho = []
            dpp_drho = []
            
            if n_alphas > 0:

                # Projection derivative wrt shape parameters
                dp_dalpha = compute_projection_derivatives_shape_parameters(
                    shape_pc_uv, rho, warped_uv, R,
                    self.model.shape_model.eigenvalues, projection_type)
                dp_dalpha = dp_dalpha[:, :n_alphas, :]
                
            if n_betas >0:
            
                # Texture derivative wrt texture parameters
                dt_dbeta = compute_texture_derivatives_texture_parameters(
                    tex_pc_uv, self.model.texture_model.eigenvalues)
                dt_dbeta = dt_dbeta[:, :n_betas, :]
                
            if camera_update:

                # Projection derivative wrt warp parameters
                dp_drho = compute_projection_derivatives_warp_parameters(
                    shape_uv, warped_uv.T, rho, r_phi, r_theta, r_varphi,
                    projection_type)

            # Compute sd matrix and hessian
            if camera_update:
                dp_dgamma = np.hstack((dp_dalpha, dp_drho))
            else: 
                dp_dgamma = dp_dalpha
            
            if n_betas > 0 and n_alphas > 0:
                dt = -dt_dbeta
                SD_gamma = compute_sd(dp_dgamma, VI_dx_uv, VI_dy_uv)
                SD_img = np.hstack((SD_gamma, dt))
            elif n_alphas > 0 and n_betas == 0:
                SD_img = compute_sd(dp_dgamma, VI_dx_uv, VI_dy_uv)
            else:
                SD_img = -dt_dbeta              
                
            # Hessian approximation
            H_img = compute_hessian(SD_img)

            # Compute error
            img_error_uv = img_uv - tex_uv.T

            # Compute steepest descent matrix
            SD_error_img = compute_sd_error(SD_img, img_error_uv)
            
            # Compute and store the error for future plots
            eps = (img_error_uv ** 2).mean()
            errors.append(eps)

            # Prior probabilities over shape parameters
            if camera_update:
                prior_error = np.concatenate((alpha, rho, beta))
            else:
                prior_error = np.concatenate((alpha, beta))
                
            # Prior probability SD matrices
            SD_error_alpha_prior = SD_alpha_prior*prior_error
            SD_error_beta_prior = SD_beta_prior*prior_error

            # Final hessian and SD error matrix
            H = H_img + 1e-2 * H_alpha_prior + H_beta_prior
            SD_error = (SD_error_img + 1e-2 * SD_error_alpha_prior +
                        SD_error_beta_prior)

            # Compute increment
            delta_sigma = -np.dot(np.linalg.inv(H), SD_error)

            # Update parameters
            if camera_update:
                alpha += delta_sigma[:n_alphas]
                rho += delta_sigma[n_alphas:n_alphas+len(rho)]
                beta += delta_sigma[n_alphas+len(rho):n_alphas+len(rho)+n_betas]
            else:
                alpha += delta_sigma[:n_alphas]
                beta += delta_sigma[n_alphas:]

            # Generate the updated instance
            # The texture is scaled by 255 to cancel the 1./255 scaling in the
            # model class
            instance = self.model.instance(alpha=alpha, beta=255. * beta)
            
            # Clip to avoid out of range pixels
            instance.colours = np.clip(instance.colours, 0, 1)

            if camera_update:
                # Compute new view matrix
                _, R = compute_view_matrix(rho)
                view_t.h_matrix[1:3, :3] = -R.h_matrix[1:3, :3]
                view_t.h_matrix[0, :3] = R.h_matrix[0, :3]
            
                # Update the rasterizer
                rasterizer.set_view_matrix(view_t.h_matrix)
            
            k += 1
            
        # Rasterize the final mesh
        rasterized_result = rasterizer.rasterize_mesh(instance)
        
        # Save final values
        sys.stdout.write('\rSuccessfully fitted.')

        return {
            'rasterized_result': rasterized_result,
            'result': instance,
            'errors': errors
        }


def sample_object(x, vertex_indices, b_coords):
    per_vert_per_pixel = x[vertex_indices]
    return np.sum(per_vert_per_pixel *
                  b_coords.T[..., None], axis=1)


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
        varphi = np.arctan2(view_t[2, 1] / np.cos(theta),
                            view_t[2, 2] / np.cos(theta))
        phi = np.arctan2(view_t[1, 0] / np.cos(theta),
                         view_t[0, 0] / np.cos(theta))
        rho[1] = varphi
        rho[2] = theta
        rho[3] = phi

    # Translations
    rho[4] = -view_t[0, 3]  # tw x
    rho[5] = -view_t[1, 3]  # tw y

    return rho


def compute_rotation_matrices(rho):
    rot_phi = np.eye(4)

    rot_phi[1:3, 1:3] = np.array([[np.cos(rho[1]), -np.sin(rho[1])],
                                  [np.sin(rho[1]), np.cos(rho[1])]])
    rot_theta = np.eye(4)
    rot_theta[0:3, 0:3] = np.array([[np.cos(rho[2]), 0, np.sin(rho[2])],
                                    [0, 1, 0],
                                    [-np.sin(rho[2]), 0, np.cos(rho[2])]])
    rot_varphi = np.eye(4)
    rot_varphi[0:2, 0:2] = np.array([[np.cos(rho[3]), -np.sin(rho[3])],
                                     [np.sin(rho[3]), np.cos(rho[3])]])

    r_phi = Homogeneous(rot_phi)
    r_theta = Homogeneous(rot_theta)
    r_varphi = Homogeneous(rot_varphi)

    return r_phi, r_theta, r_varphi


def compute_view_matrix(rho):

    view_t = np.eye(4)
    view_t[0, 3] = -rho[4]  # tw x
    view_t[1, 3] = -rho[5]  # tw y
    r_phi, r_theta, r_varphi = compute_rotation_matrices(rho)
    rotation = np.dot(np.dot(r_varphi.h_matrix, r_theta.h_matrix),
                      r_phi.h_matrix)
    view_t[:3, :3] = rotation[:3, :3]

    return Homogeneous(view_t), Homogeneous(rotation)


def compute_sd(dp_dgamma, VI_dx_uv, VI_dy_uv):
    permuted_vi_dx = np.transpose(VI_dx_uv[..., None], (0, 2, 1))
    permuted_vi_dy = np.transpose(VI_dy_uv[..., None], (0, 2, 1))
    return permuted_vi_dx*dp_dgamma[0] + permuted_vi_dy*dp_dgamma[1]


def compute_hessian(sd):
    # Computes the hessian as defined in the Lucas Kanade Algorithm
    n_channels = sd.shape[0]
    n_params = sd.shape[1]
    h = np.zeros((n_params, n_params))
    sd = sd.T
    for i in range(n_channels):
        h += np.dot(sd[:, :, i].T, sd[:, :, i])
    return h


def compute_sd_error(sd, error_uv):
    n_channels = sd.shape[0]
    n_parameters = sd.shape[1]
    sd_error_product = np.zeros(n_parameters)
    sd = sd.T
 
    for i in range(n_channels):
        sd_error_product += np.dot(error_uv[i, :], sd[:, :, i])
        
    return sd_error_product.T
