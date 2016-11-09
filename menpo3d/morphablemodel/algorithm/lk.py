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

    def d_projection_d_shape_camera_parameters(self, warped_uv, shape_pc_uv,
                                               camera, camera_update):
        # Compute derivative of projection wrt shape parameters
        if self.projection_type == 'perspective':
            dp_da_dr = d_perspective_projection_d_shape_parameters(
                shape_pc_uv, warped_uv, camera)
        else:
            dp_da_dr = d_orthographic_projection_d_shape_parameters(
                shape_pc_uv, camera)

        # Compute derivative of projection wrt camera parameters
        if camera_update:
            if self.projection_type == 'perspective':
                dp_dr = d_perspective_projection_d_camera_parameters(
                    warped_uv, camera)
            else:
                dp_dr = d_orthographic_projection_d_camera_parameters(
                    warped_uv, camera)

            # Concatenate it with the derivative wrt shape parameters
            dp_da_dr = np.hstack((dp_da_dr, dp_dr))

        return dp_da_dr

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
        c = 500  # SMALL VALUES FOR MORE CONSTRAINED SHAPE MODEL
        shape_prior_weight = 1. / (c * self.model.shape_model.noise_variance())
        texture_prior_weight = 1.
        self.J_shape_prior = (shape_prior_weight * 1. /
                              np.array(self.model.shape_model.eigenvalues))
        self.J_texture_prior = (texture_prior_weight * 1. /
                                np.array(self.model.texture_model.eigenvalues))


class Simultaneous(LucasKanade):
    r"""
    Class for defining Simultaneous Morphable Model optimization algorithm.
    """
    def run(self, image, initial_mesh, camera, gt_mesh=None, max_iters=20,
            landmarks_prior=None, parameters_priors=True, camera_update=False,
            focal_length_update=False, return_costs=False):
        # Define cost closure
        def cost_closure(x):
            return x.T.dot(x)

        # Retrieve camera parameters from the provided view and projection
        # transforms.
        camera_parameters = camera.as_vector()
        shape_parameters = self.model.shape_model.project(initial_mesh)
        texture_parameters = self.model.project_instance_on_texture_model(
            initial_mesh)

        # Get starting point of Hessian slicing for texture
        texture_slice = self.n
        if camera_update:
            # Note that we do not optimize with respect to the 1st quaternion
            texture_slice = self.n + camera.n_parameters - 1
            if not focal_length_update:
                texture_slice -= 1
        total_texture_slice = self.n + camera.n_parameters

        # Reconstruct provided instance
        instance = self.model.instance(shape_weights=shape_parameters,
                                       texture_weights=texture_parameters)

        # Compute input image gradient
        grad_x, grad_y = self.gradient(image)

        # Initialize lists
        shape_parameters_per_iter = [shape_parameters]
        texture_parameters_per_iter = [texture_parameters]
        camera_per_iter = [camera]
        instance_per_iter = [instance.with_clipped_texture()]
        costs = None
        if return_costs:
            costs = []

        # Initialize iteration counter and epsilon
        k = 0
        eps = np.Inf
        while k < max_iters and eps > self.eps:
            print_dynamic("{}/{}".format(k + 1, max_iters))
            # Apply camera projection on current instance
            instance_in_image = camera.apply(instance)

            # Compute indices locations for warping
            (vertex_indices, tri_indices,
             b_coords, yx) = self.visible_sample_points(instance_in_image,
                                                        image.shape)

            # Warp the mesh with the view matrix (rotation + translation)
            instance_w = camera.view_transform.apply(instance.points)

            # Sample all the terms from the model part at the sample locations
            warped_uv = self.sample(instance_w, vertex_indices, b_coords)
            texture_uv = instance.sample_texture_with_barycentric_coordinates(
                b_coords, tri_indices)
            texture_pc_uv = self.model.sample_texture_model(b_coords,
                                                            tri_indices)
            shape_pc_uv = self.sample(self.shape_pc, vertex_indices, b_coords)
            # Reshape shape basis after sampling
            shape_pc_uv = shape_pc_uv.reshape([self.n_samples, 3, -1])

            # Sample all the terms from the image part at the sample locations
            img_uv = image.sample(yx)
            grad_x_uv = grad_x.sample(yx)
            grad_y_uv = grad_y.sample(yx)

            # Compute derivative of projection wrt shape and camera parameters
            dp_da_dr = self.d_projection_d_shape_camera_parameters(
                warped_uv, shape_pc_uv, camera, camera_update)

            # Compute steepest descent
            sd_da_dr = self.compute_steepest_descent(dp_da_dr, grad_x_uv,
                                                     grad_y_uv)

            # Computer derivative of texture wrt texture parameters
            dt_db = - np.rollaxis(texture_pc_uv, 0, 3)

            # Formulation the total Jacobian
            sd = np.hstack((sd_da_dr, dt_db))

            # Slice off the Jacobian of focal length, if asked to not optimise
            # wrt the focal length
            if not focal_length_update:
                sd = np.delete(sd, self.n, 1)

            # Compute hessian
            hessian = self.compute_hessian(sd)
            if parameters_priors:
                hessian[:self.n, :self.n] += np.diag(self.J_shape_prior)
                hessian[texture_slice:, texture_slice:] += np.diag(
                    self.J_texture_prior)

            # Compute error
            img_error_uv = img_uv - texture_uv.T

            # Compute steepest descent matrix
            sd_error_img = self.compute_sd_error(sd, img_error_uv)

            # Apply priors
            if parameters_priors:
                sd_error_img[:self.n] += self.J_shape_prior * shape_parameters
                sd_error_img[texture_slice:] += (self.J_texture_prior *
                                                 texture_parameters)

            # Update costs
            if return_costs:
                costs.append(cost_closure(img_error_uv.ravel()))

            # Compute increment
            ds = - np.linalg.solve(hessian, sd_error_img)

            # If focal length is not updated, then set its increment to zero
            if not focal_length_update:
                ds = np.insert(ds, self.n, [0.])

            # Set increment of the 1st quaternion to one
            ds = np.insert(ds, self.n + 1, [1.])

            # Update parameters
            shape_parameters += ds[:self.n]
            if camera_update:
                camera_parameters = camera_parameters_update(
                    camera_parameters, ds[self.n:total_texture_slice])
                camera = camera.from_vector(camera_parameters)
            texture_parameters += ds[total_texture_slice:]

            # Generate the updated instance
            instance = self.model.instance(
                shape_weights=shape_parameters,
                texture_weights=texture_parameters)

            # Update lists
            shape_parameters_per_iter.append(shape_parameters)
            texture_parameters_per_iter.append(texture_parameters)
            camera_per_iter.append(camera)
            instance_per_iter.append(instance.with_clipped_texture())

            # Increase iteration counter
            k += 1

        return MMAlgorithmResult(
            shape_parameters=shape_parameters_per_iter,
            texture_parameters=texture_parameters_per_iter,
            meshes=instance_per_iter, camera_transforms=camera_per_iter,
            image=image, initial_mesh=initial_mesh.with_clipped_texture(),
            initial_camera_transform=camera_per_iter[0], gt_mesh=gt_mesh,
            costs=costs)

    def __str__(self):
        return "Simultaneous Forward Additive"


def quaternion_multiply(current_q, increment_q):
    # Make sure that the q increment has unit norm
    increment_q /= np.linalg.norm(increment_q)
    # Update
    w0, x0, y0, z0 = current_q
    w1, x1, y1, z1 = increment_q
    return np.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                      x1*w0 + y1*z0 - z1*y0 + w1*x0,
                     -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                      x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=np.float64)


def camera_parameters_update(c, dc):
    # Add for focal length and translation parameters, but multiply for
    # quaternions
    new = c + dc
    new[1:5] = quaternion_multiply(c[1:5], dc[1:5])
    return new
