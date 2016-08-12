import numpy as np
from menpo.transform import Homogeneous


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
    
    #view_t = Homogeneous(view_t).compose_before(axes_flip_t)
    
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