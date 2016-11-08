import numpy as np

from menpo.feature import gradient as fast_gradient
from menpo.image import Image
from menpo.visualize import print_dynamic

from menpo3d.rasterize import rasterize_barycentric_coordinates

from .derivatives import (d_orthographic_projection_d_shape_parameters,
                          d_perspective_projection_d_shape_parameters,
                          d_orthographic_projection_d_camera_parameters,
                          d_perspective_projection_d_camera_parameters)
from ..result import MMAlgorithmResult


class LucasKanade(object):
    def __init__(self, model, n_samples, projection_type, eps=1e-3):
        self.model = model
        self.eps = eps
        self.n_samples = n_samples
        self.projection_type = projection_type
        # Call precompute
        self._precompute()

    @property
    def n(self):
        r"""
        Returns the number of active components of the shape model.

        :type: `int`
        """
        return self.model.shape_model.n_active_components

    @property
    def m(self):
        r"""
        Returns the number of active components of the texture model.

        :type: `int`
        """
        return self.model.texture_model.n_active_components

    @property
    def n_vertices(self):
        r"""
        Returns the number of vertices of the shape model's trimesh.

        :type: `int`
        """
        return self.model.shape_model.template_instance.n_points

    @property
    def n_channels(self):
        r"""
        Returns the number of channels of the texture model.

        :type: `int`
        """
        return self.model.n_channels

    def visible_sample_points(self, instance_in_img, image_shape):
        # Inverse rendering
        yx, tri_indices, b_coords = rasterize_barycentric_coordinates(
            instance_in_img, image_shape)

        # Select triangles randomly
        rand = np.random.permutation(b_coords.shape[0])
        b_coords = b_coords[rand[:self.n_samples]]
        yx = yx[rand[:self.n_samples]]
        tri_indices = tri_indices[rand[:self.n_samples]]

        # Build the vertex indices (3 per pixel) for the visible triangles
        vertex_indices = instance_in_img.trilist[tri_indices]

        return vertex_indices, tri_indices, b_coords, yx

    def sample(self, x, vertex_indices, b_coords):
        per_vert_per_pixel = x[vertex_indices]
        return np.sum(per_vert_per_pixel * b_coords[..., None], axis=1)

    def gradient(self, image):
        # Compute the gradient of the image
        grad = fast_gradient(image)

        # Create gradient image for X and Y
        grad_y = Image(grad.pixels[:self.n_channels])
        grad_x = Image(grad.pixels[self.n_channels:])

        return grad_x, grad_y

    def d_projection_d_shape_parameters(self, warped_uv, shape_pc_uv, camera):
        if self.projection_type == 'perspective':
            return d_perspective_projection_d_shape_parameters(
                shape_pc_uv, warped_uv, camera)
        else:
            return d_orthographic_projection_d_shape_parameters(
                shape_pc_uv, camera)

    def d_projection_d_camera_parameters(self, warped_uv, camera):
        if self.projection_type == 'perspective':
            return d_perspective_projection_d_camera_parameters(warped_uv,
                                                                 camera)
        else:
            return d_orthographic_projection_d_camera_parameters(warped_uv,
                                                                  camera)

    def compute_steepest_descent(self, dp_da, grad_x_uv, grad_y_uv):
        permuted_grad_x = np.transpose(grad_x_uv[..., None], (0, 2, 1))
        permuted_grad_y = np.transpose(grad_y_uv[..., None], (0, 2, 1))
        return permuted_grad_x * dp_da[0] + permuted_grad_y * dp_da[1]

    def compute_hessian(self, sd):
        n_channels = sd.shape[0]
        n_params = sd.shape[1]
        h = np.zeros((n_params, n_params))
        sd = sd.T
        for i in range(n_channels):
            h += np.dot(sd[:, :, i].T, sd[:, :, i])
        return h

    def compute_sd_error(self, sd, error_uv):
        n_channels = sd.shape[0]
        n_parameters = sd.shape[1]
        sd_error_product = np.zeros(n_parameters)
        sd = sd.T
        for i in range(n_channels):
            sd_error_product += np.dot(error_uv[i, :], sd[:, :, i])
        return sd_error_product.T

    def _precompute(self):
        # Rescale shape and appearance components to have size:
        # n_vertices x (n_active_components * n_dims)
        shape_pc = self.model.shape_model.components.T
        self.shape_pc = shape_pc.reshape([self.n_vertices, -1])

        # Priors
        c = 1000  # SMALL VALUES FOR MORE CONSTRAINED SHAPE MODEL
        shape_prior_weight = 1. / (c * self.model.shape_model.noise_variance())
        texture_prior_weight = 1.
        self.J_shape_prior = (shape_prior_weight * 1. /
                              self.model.shape_model.eigenvalues)
        self.J_texture_prior = (texture_prior_weight * 1. /
                                self.model.texture_model.eigenvalues)
        self.H_shape_prior = np.hstack(
            (self.J_shape_prior, np.zeros_like(self.J_texture_prior)))
        self.H_texture_prior = np.hstack(
            (np.zeros_like(self.J_shape_prior), self.J_texture_prior))


class Simultaneous(LucasKanade):
    r"""
    Class for defining Simultaneous Morphable Model optimization algorithm.
    """
    def run(self, image, initial_mesh, camera, gt_mesh=None, use_priors=True,
            camera_update=False, max_iters=20, return_costs=False):
        # Define cost closure
        def cost_closure(x):
            return x.T.dot(x)

        # Retrieve camera parameters from the provided view and projection
        # transforms.
        camera_parameters = camera.as_vector()
        shape_parameters = self.model.shape_model.project(initial_mesh)
        texture_parameters = self.model.project_instance_on_texture_model(
            initial_mesh)

        # Reconstruct provided instance
        instance = self.model.instance(shape_weights=shape_parameters,
                                       texture_weights=texture_parameters)

        # Compute input image gradient
        grad_x, grad_y = self.gradient(image)

        # Initialize lists
        a_list = [shape_parameters]
        b_list = [texture_parameters]
        r_list = [camera]
        instances = [instance.with_clipped_texture()]
        costs = None
        if return_costs:
            costs = []

        # Initialize iteration counter and epsilon
        k = 0
        eps = np.Inf
        while k < max_iters and eps > self.eps:
            print_dynamic("{}/{}".format(k + 1, max_iters))

            instance_in_image = camera.apply(instance)

            # Compute indices locations for warping
            (vertex_indices, tri_indices,
             b_coords, yx) = self.visible_sample_points(instance_in_image,
                                                        image.shape)

            # Warp the mesh with the view matrix
            W = camera.view_transform.apply(instance.points)

            # Sample all the terms we need at our sample locations.
            warped_uv = self.sample(W, vertex_indices, b_coords)
            texture_uv = instance.sample_texture_with_barycentric_coordinates(
                b_coords, tri_indices)
            texture_pc_uv = self.model.sample_texture_model(b_coords,
                                                            tri_indices)
            shape_pc_uv = self.sample(self.shape_pc, vertex_indices, b_coords)
            # Reshape shape basis after sampling
            shape_pc_uv = shape_pc_uv.reshape([self.n_samples, 3, -1])
            img_uv = image.sample(yx)
            grad_x_uv = grad_x.sample(yx)
            grad_y_uv = grad_y.sample(yx)

            # Compute derivative of projection wrt shape parameters
            dp_da_dr = self.d_projection_d_shape_parameters(warped_uv,
                                                            shape_pc_uv, camera)

            # Compute derivative of projection wrt camera parameters
            if camera_update:
                dp_dr = self.d_projection_d_camera_parameters(warped_uv, camera)
                # Concatenate it with the derivative wrt shape parameters
                dp_da_dr = np.hstack((dp_da_dr, dp_dr))

            # Derivative of texture wrt texture parameters
            dt_db = - np.rollaxis(texture_pc_uv, 0, 3)

            # Compute steepest descent
            sd_da_dr = self.compute_steepest_descent(dp_da_dr, grad_x_uv,
                                                     grad_y_uv)
            sd = np.hstack((sd_da_dr, dt_db))

            print()
            print("sd_da_dr: {:.4f} - {:.4f} -> {:.4f}".format(
                sd_da_dr.min(), sd_da_dr.max(), np.linalg.norm(
                    sd_da_dr.ravel())))
            print("dt_db: {:.4f} - {:.4f} -> {:.4f}".format(
                dt_db.min(), dt_db.max(), np.linalg.norm(dt_db.ravel())))

            # Compute hessian
            hessian = self.compute_hessian(sd)
            if use_priors:
                hessian += (np.diag(self.H_shape_prior) +
                            np.diag(self.H_texture_prior))

            # Compute error
            img_error_uv = img_uv - texture_uv.T

            # Compute steepest descent matrix
            sd_error_img = self.compute_sd_error(sd, img_error_uv)

            # Apply priors
            if use_priors:
                if camera_update:
                    all_parameters = np.concatenate(
                        (shape_parameters, camera_parameters,
                         texture_parameters))
                else:
                    all_parameters = np.concatenate(
                        (shape_parameters, texture_parameters))
                sd_shape = self.H_shape_prior * all_parameters
                sd_texture = self.H_texture_prior * all_parameters
                sd_error_img += sd_shape + sd_texture

            # Update costs
            if return_costs:
                costs.append(cost_closure(img_error_uv.ravel()))

            # Compute increment
            delta_s = - np.linalg.solve(hessian, sd_error_img)

            # Update parameters
            shape_parameters += delta_s[:self.n]
            if camera_update:
                camera_parameters += delta_s[self.n:self.n+len(camera_parameters)]
                camera = camera.from_vector(camera_parameters)
                texture_parameters += delta_s[self.n+len(camera_parameters):]
            else:
                texture_parameters += delta_s[self.n:]
            a_list.append(shape_parameters)
            b_list.append(texture_parameters)
            r_list.append(camera)

            # Generate the updated instance
            instance = self.model.instance(
                shape_weights=shape_parameters,
                texture_weights=texture_parameters)
            instances.append(instance.with_clipped_texture())

            # Increase iteration counter
            k += 1

        return MMAlgorithmResult(
            shape_parameters=a_list, texture_parameters=b_list,
            meshes=instances, camera_transforms=r_list, image=image,
            initial_mesh=initial_mesh.with_clipped_texture(),
            initial_camera_transform=r_list[0], gt_mesh=gt_mesh, costs=costs)

    def __str__(self):
        return "Simultaneous Lucas-Kanade"


def parameters_prior(params, eigenvalues):
    c = params / eigenvalues
    norm = np.sqrt(len(eigenvalues)) / c.dot(c)
    return norm * params
