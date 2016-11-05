import numpy as np

from menpo.feature import gradient as fast_gradient
from menpo.image import Image
from menpo.visualize import print_dynamic

from menpo3d.rasterize import rasterize_barycentric_coordinates

from .derivatives import (d_orthographic_projection_d_shape_parameters,
                          d_perspective_projection_d_shape_parameters,
                          d_orthographic_projection_d_warp_parameters,
                          d_perspective_projection_d_warp_parameters)
from ..projection import compute_rotation_matrices
from ..result import MMAlgorithmResult


class LucasKanade(object):
    def __init__(self, model, n_samples, projection_type, eps=1e-3):
        self.model = model
        self.eps = eps
        self.n_samples = n_samples
        self.projection_type = projection_type

        # Call precomputation
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
        r"""
        Computes the warping map.

        Parameters
        ----------
        instance : `menpo.shape.ColouredTriMesh`
            The input mesh instance.

        Returns
        -------
        vertex_indices : ``(n_samples, 3)`` `ndarray`
            The vertices indices per sample.
        b_coords : ``(n_samples, 3)`` `ndarray`
            The barycentric coordinates per sample.
        yx : ``(n_samples,)`` `ndarray`
            The indices of the true points.
        """
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
        r"""
        Method that samples an object at locations specified by the
        barycentric coordinates.

        Parameters
        ----------
        x : `ndarray`
            The object to sample.
        vertex_indices : ``(n_samples, 3)`` `ndarray`
            The vertices indices per sample.
        b_coords : ``(n_samples, 3)`` `ndarray`
            The barycentric coordinates per sample.

        Returns
        -------
        sampled_x : ``(n_samples, ...)`` `ndarray`
            The sampled object.
        """
        per_vert_per_pixel = x[vertex_indices]
        return np.sum(per_vert_per_pixel * b_coords[..., None], axis=1)

    def gradient(self, image):
        r"""
        Computes the gradient of an image.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The input image.

        Returns
        -------
        gradient_X : ``(n_channels, height, width)`` `menpo.image.Image`
            The image gradient over the X direction.
        gradient_Y : ``(n_channels, height, width)`` `menpo.image.Image`
            The image gradient over the Y direction.
        """
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

        return d_orthographic_projection_d_shape_parameters(
                shape_pc_uv, camera)

    def d_projection_d_warp_parameters(self, shape_uv, warped_uv,
                                       warp_parameters):
        r_phi, r_theta, r_varphi, _= compute_rotation_matrices(
            warp_parameters[1], warp_parameters[2], warp_parameters[3])
        if self.projection_type == 'perspective':
            dp_dr = d_perspective_projection_d_warp_parameters(
                shape_uv, warped_uv, warp_parameters, r_phi, r_theta, r_varphi)
        else:
            dp_dr = d_orthographic_projection_d_warp_parameters(
                shape_uv, warped_uv, warp_parameters, r_phi, r_theta, r_varphi)
        return dp_dr

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

        # Parameters priors
        self.sd_alpha_prior = np.zeros(self.n + self.m)
        self.sd_alpha_prior[:self.n] = 1. / np.sqrt(self.model.shape_model.eigenvalues)
        #self.sd_alpha_prior[:self.n] = 2. / (
        # self.model.shape_model.eigenvalues ** 2)
        self.sd_beta_prior = np.zeros(self.n + self.m)
        self.sd_beta_prior[self.n:] = 1. / np.sqrt(self.model.texture_model.eigenvalues)
        #self.sd_beta_prior[self.n:] = 2. / (
        # self.model.texture_model.eigenvalues ** 2)
        self.H_alpha_prior = np.diag(self.sd_alpha_prior)
        self.H_beta_prior = np.diag(self.sd_beta_prior)


class Simultaneous(LucasKanade):
    r"""
    Class for defining Simultaneous Morphable Model optimization algorithm.
    """
    def run(self, image, initial_mesh, camera, gt_mesh=None,
            camera_update=False, max_iters=20, return_costs=False):
        # Define cost closure
        def cost_closure(x):
            return x.T.dot(x)

        # Retrieve warp (camera) parameters from the provided view and
        # projection transforms.
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
            shape_uv = self.sample(instance.points, vertex_indices, b_coords)
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


            # print('texture_uv.shape: {}, texture_pc_uv.shape: {}'.format(texture_uv.shape, texture_pc_uv.shape))
            # print('shape_uv.shape: {}, shape_pc_uv.shape: {}'.format(shape_uv.shape, shape_pc_uv.shape))

            # Compute derivative of projection wrt shape parameters
            dp_da_dr = self.d_projection_d_shape_parameters(
                warped_uv, shape_pc_uv, camera)

            # Compute derivative of projection wrt warp parameters
            if camera_update:
                dp_dr = self.d_projection_d_warp_parameters(
                    shape_uv, warped_uv, camera_parameters)
                # Concatenate it with the derivative wrt shape parameters
                dp_da_dr = np.hstack((dp_da_dr, dp_dr))

            # Derivative of texture wrt texture parameters
            dt_db = np.rollaxis(texture_pc_uv, 0, 3)

            # Compute steepest descent
            sd_da_dr = self.compute_steepest_descent(dp_da_dr, grad_x_uv,
                                                     grad_y_uv)
            sd = np.hstack((sd_da_dr, -dt_db))

            # Compute hessian
            hessian = self.compute_hessian(sd) + 1e-1 * self.H_alpha_prior + self.H_beta_prior

            # Compute error
            img_error_uv = img_uv - texture_uv.T

            # Compute steepest descent matrix
            sd_error_img = self.compute_sd_error(sd, img_error_uv)

            # Apply priors
            sd_error_alpha_prior = self.sd_alpha_prior * \
                                   np.concatenate((shape_parameters,
                                                   texture_parameters))
            sd_error_beta_prior = self.sd_beta_prior * \
                                  np.concatenate((shape_parameters,
                                                  texture_parameters))
            sd_error_img = sd_error_img + 1e-1 * sd_error_alpha_prior + sd_error_beta_prior

            # Update costs
            if return_costs:
                costs.append(cost_closure(img_error_uv.ravel()))

            # Compute increment
            delta_s = - np.linalg.solve(hessian, sd_error_img)

            # Update parameters
            shape_parameters += delta_s[:self.n]
            a_list.append(shape_parameters)
            if camera_update:
                camera_parameters += delta_s[self.n:self.n+len(camera_parameters)]
                texture_parameters += delta_s[(self.n+len(camera_parameters)):]
            else:
                texture_parameters += delta_s[self.n:]
            b_list.append(texture_parameters)

            # print("Image UV: {:.5f} - {:.5f}".format(img_uv.min(),
            #                                          img_uv.max()))
            # print("Instance UV: {:.5f} - {:.5f}".format(texture_uv.min(),
            #                                             texture_uv.max()))
            # print("Grad X: {:.5f} - {:.5f}".format(grad_x_uv.min(),
            #                                        grad_x_uv.max()))
            # print("Grad Y: {:.5f} - {:.5f}".format(grad_y_uv.min(),
            #                                        grad_y_uv.max()))
            # print("SD 1.1: {:.5f} - {:.5f}".format(sd_da_dr.min(),
            #                                        sd_da_dr.max()))
            # print("SD 1.2: {:.5f} - {:.5f}".format(dp_da_dr.min(),
            #                                        dp_da_dr.max()))
            # print("SD 2: {:.5f} - {:.5f}".format(dt_db.min(),
            #                                      dt_db.max()))

            # Generate the updated instance
            instance = self.model.instance(
                shape_weights=shape_parameters,
                texture_weights=texture_parameters)
            instances.append(instance.with_clipped_texture())

            if camera_update:
                camera = camera.from_vector(camera_parameters)
            r_list.append(camera)

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
