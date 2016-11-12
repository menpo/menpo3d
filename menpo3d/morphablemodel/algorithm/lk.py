import numpy as np

from menpo.feature import gradient as fast_gradient
from menpo.image import Image
from menpo.visualize import print_dynamic

from menpo3d.rasterize import rasterize_barycentric_coordinates

from .derivatives import (d_camera_d_camera_parameters,
                          d_camera_d_shape_parameters)
from ..result import MMAlgorithmResult


DEBUG = False


class LucasKanade(object):
    def __init__(self, model, n_samples, eps=1e-3):
        self.model = model
        self.eps = eps
        self.n_samples = n_samples
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

    def compute_cost(self, data_error, lms_error, shape_parameters,
                     texture_parameters, shape_prior_weight,
                     texture_prior_weight, landmarks_prior_weight):
        # Cost of data term
        current_cost = data_error.T.dot(data_error)

        # Cost of shape prior
        if shape_prior_weight is not None:
            current_cost += (shape_prior_weight *
                             np.sum((shape_parameters ** 2) *
                                    self.J_shape_prior))

        # Cost of texture prior
        if texture_prior_weight is not None:
            current_cost += (texture_prior_weight *
                             np.sum((texture_parameters ** 2) *
                                    self.J_texture_prior))

        # Cost of landmarks prior
        if landmarks_prior_weight is not None:
            current_cost += landmarks_prior_weight * lms_error.T.dot(lms_error)

        return current_cost

    def _precompute(self):
        # Rescale shape and appearance components to have size:
        # n_vertices x (n_active_components * n_dims)
        shape_pc = self.model.shape_model.components.T
        self.shape_pc = shape_pc.reshape([self.n_vertices, -1])

        # Priors
        self.J_shape_prior = 1. / np.array(self.model.shape_model.eigenvalues)
        self.J_texture_prior = 1. / np.array(self.model.texture_model.eigenvalues)
        self.shape_pc_lms = shape_pc.reshape([self.n_vertices, 3, -1])[
            self.model.model_landmarks_index]


class SimultaneousForwardAdditive(LucasKanade):
    r"""
    Class for defining Simultaneous Forward Additive Morphable Model
    optimization algorithm.
    """
    def run(self, image, initial_mesh, camera, gt_mesh=None, max_iters=20,
            camera_update=False, focal_length_update=False,
            reconstruction_weight=1., shape_prior_weight=1.,
            texture_prior_weight=1., landmarks=None, landmarks_prior_weight=1.,
            return_costs=False, verbose=True):
        # Parse landmarks prior options
        if landmarks is None or landmarks_prior_weight is None:
            landmarks_prior_weight = None
            landmarks = None
        lms_points = None
        if landmarks is not None:
            lms_points = landmarks.points[:, [1, 0]]

        # Retrieve camera parameters from the provided camera object.
        # Project provided instance to retrieve shape and texture parameters.
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
        shape_parameters_per_iter = [shape_parameters]
        texture_parameters_per_iter = [texture_parameters]
        camera_per_iter = [camera]
        instance_per_iter = [instance.with_rescaled_texture(0., 1.)]
        costs = None
        if return_costs:
            costs = []

        # Initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Main loop
        while k < max_iters and eps > self.eps:
            if verbose:
                print_dynamic("{}/{}".format(k + 1, max_iters))
                print()
            # Apply camera projection on current instance
            instance_in_image = camera.apply(instance)

            # Compute indices locations for sampling
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

            # Compute error
            img_error_uv = (img_uv - texture_uv.T).ravel()

            # Compute Jacobian, SD and Hessian of data term
            if reconstruction_weight is not None:
                sd, n_camera_parameters = self.J_data(
                    camera, warped_uv, shape_pc_uv, texture_pc_uv, grad_x_uv,
                    grad_y_uv, camera_update, focal_length_update,
                    reconstruction_weight)
                hessian = sd.dot(sd.T)
                sd_error = sd.dot(img_error_uv)
            else:
                n_camera_parameters = 0
                if camera_update:
                    if focal_length_update:
                        n_camera_parameters = camera.n_parameters - 1
                    else:
                        n_camera_parameters = camera.n_parameters - 2
                hessian = np.zeros((self.n+n_camera_parameters,
                                    self.n+n_camera_parameters))
                sd_error = np.zeros(self.n+n_camera_parameters)

            if DEBUG:
                from pandas import DataFrame
                idx = self.n + n_camera_parameters
                data_j = {'J shape': [sd[:, :self.n, :].min(),
                                      sd[:, :self.n, :].max(),
                                      np.linalg.norm(sd[:, :self.n, :])],
                          'J texture': [sd[:, idx:, :].min(),
                                        sd[:, idx:, :].max(),
                                        np.linalg.norm(sd[:, idx:, :])]}
                columns_j = ['J shape', 'J texture']
                if camera_update:
                    data_j['J camera'] = [sd[:, self.n:idx, :].min(),
                                          sd[:, self.n:idx, :].max(),
                                          np.linalg.norm(sd[:, self.n:idx, :])]
                    columns_j.append('J camera')
                data_h = {'H shape': [hessian[:self.n, :self.n].min(),
                                      hessian[:self.n, :self.n].max(),
                                      np.linalg.norm(hessian[:self.n, :self.n])],
                          'H texture': [hessian[idx:, idx:].min(),
                                        hessian[idx:, idx:].max(),
                                        np.linalg.norm(hessian[idx:, idx:])]}
                columns_h = ['H shape', 'H texture']
                if camera_update:
                    data_h['H camera'] = [hessian[self.n:idx, self.n:idx].min(),
                                          hessian[self.n:idx, self.n:idx].max(),
                                          np.linalg.norm(hessian[self.n:idx, self.n:idx])]
                    columns_h.append('H camera')

            # Compute Jacobian, update SD and Hessian wrt shape prior
            if shape_prior_weight is not None:
                sd_shape = shape_prior_weight * self.J_shape_prior
                hessian[:self.n, :self.n] += np.diag(sd_shape)
                sd_error[:self.n] += sd_shape * shape_parameters

            # Compute Jacobian, update SD and Hessian wrt texture prior
            if texture_prior_weight is not None:
                idx = self.n + n_camera_parameters
                sd_texture = texture_prior_weight * self.J_texture_prior
                hessian[idx:, idx:] += np.diag(sd_texture)
                sd_error[idx:] += sd_texture * texture_parameters

            if DEBUG:
                if shape_prior_weight is not None:
                    data_j['J shape prior'] = [sd_shape.min(),
                                               sd_shape.max(),
                                               np.linalg.norm(sd_shape)]
                    columns_j.append('J shape prior')
                    data_h['H shape prior'] = [sd_shape.min(),
                                               sd_shape.max(),
                                               np.linalg.norm(sd_shape)]
                    columns_h.append('H shape prior')

                if texture_prior_weight is not None:
                    data_j['J texture prior'] = [sd_texture.min(),
                                                 sd_texture.max(),
                                                 np.linalg.norm(sd_texture)]
                    columns_j.append('J texture prior')
                    data_h['H texture prior'] = [sd_texture.min(),
                                                 sd_texture.max(),
                                                 np.linalg.norm(sd_texture)]
                    columns_h.append('H texture prior')

            # Compute Jacobian, update SD and Hessian wrt landmarks prior
            lms_error = None
            if landmarks_prior_weight is not None:
                # Get projected instance on landmarks and error term
                warped_lms = instance_in_image.points[
                    self.model.model_landmarks_index]
                lms_error = (warped_lms[:, [1, 0]] - lms_points).T.ravel()

                # Jacobian and Hessian wrt shape parameters
                warped_view_lms = instance_w[self.model.model_landmarks_index]
                sd_lms_shape = d_camera_d_shape_parameters(
                    camera, warped_view_lms, self.shape_pc_lms)
                sd_lms_shape = np.transpose(sd_lms_shape,
                                            (1, 0, 2)).reshape(self.n, -1)
                hessian[:self.n, :self.n] += (landmarks_prior_weight *
                                              sd_lms_shape.dot(sd_lms_shape.T))
                sd_error[:self.n] += (landmarks_prior_weight *
                                      sd_lms_shape.dot(lms_error))

                # Jacobian and Hessian wrt camera parameters
                if camera_update:
                    sd_lms_camera = d_camera_d_camera_parameters(
                        camera, warped_view_lms,
                        with_focal_length=focal_length_update)
                    n_camera_parameters = sd_lms_camera.shape[1]
                    sd_lms_camera = np.transpose(
                        sd_lms_camera, (1, 0, 2)).reshape(
                        n_camera_parameters, -1)
                    idx = self.n + n_camera_parameters
                    hessian[self.n:idx, self.n:idx] += (
                        landmarks_prior_weight *
                        sd_lms_camera.dot(sd_lms_camera.T))
                    sd_error[self.n:idx] += (landmarks_prior_weight *
                                             sd_lms_camera.dot(lms_error))

            if return_costs:
                costs.append(self.compute_cost(
                    img_error_uv, lms_error, shape_parameters,
                    texture_parameters, shape_prior_weight,
                    texture_prior_weight, landmarks_prior_weight))

            if DEBUG:
                if landmarks_prior_weight is not None:
                    data_j['J shape lms'] = [sd_lms_shape.min(),
                                             sd_lms_shape.max(),
                                             np.linalg.norm(sd_lms_shape)]
                    columns_j.append('J shape lms')
                    tmp = landmarks_prior_weight * self.compute_hessian(sd_lms_shape)
                    data_h['H shape lms'] = [tmp.min(),
                                             tmp.max(),
                                             np.linalg.norm(tmp)]
                    columns_h.append('H shape lms')
                    if camera_update:
                        data_j['J camera lms'] = [sd_lms_camera.min(),
                                                  sd_lms_camera.max(),
                                                  np.linalg.norm(sd_lms_camera)]
                        columns_j.append('J camera lms')
                        tmp = landmarks_prior_weight * self.compute_hessian(sd_lms_camera)
                        data_h['H camera lms'] = [tmp.min(),
                                                  tmp.max(),
                                                  np.linalg.norm(tmp)]
                        columns_h.append('H camera lms')

                print(DataFrame(data_j, columns=columns_j,
                                index=['min', 'max', 'norm']))
                print(DataFrame(data_h, columns=columns_h,
                                index=['min', 'max', 'norm']))

            # Solve to find the increment of parameters
            d_shape, d_camera, d_texture = self.solve(
                hessian, sd_error, reconstruction_weight, camera_update,
                focal_length_update, camera)

            # Update parameters
            shape_parameters += d_shape
            if camera_update:
                camera_parameters = camera_parameters_update(
                    camera_parameters, d_camera)
                camera = camera.from_vector(camera_parameters)
            texture_parameters += d_texture

            # Generate the updated instance
            instance = self.model.instance(shape_weights=shape_parameters,
                                           texture_weights=texture_parameters)

            # Update lists
            shape_parameters_per_iter.append(shape_parameters)
            texture_parameters_per_iter.append(texture_parameters)
            camera_per_iter.append(camera)
            instance_per_iter.append(instance.with_rescaled_texture(0., 1.))

            # Increase iteration counter
            k += 1

            # shape_parameters, texture_parameters, camera_parameters = yield \
            #     shape_parameters, texture_parameters, camera_parameters

        return MMAlgorithmResult(
            shape_parameters=shape_parameters_per_iter,
            texture_parameters=texture_parameters_per_iter,
            meshes=instance_per_iter, camera_transforms=camera_per_iter,
            image=image, initial_mesh=initial_mesh.with_rescaled_texture(0., 1.),
            initial_camera_transform=camera_per_iter[0], gt_mesh=gt_mesh,
            costs=costs)

    def solve(self, hessian, sd_error, reconstruction_prior_weight,
              camera_update, focal_length_update, camera):
        # Solve
        ds = - np.linalg.solve(hessian, sd_error)

        # Get shape parameters increment
        d_shape = ds[:self.n]

        # Initialize texture parameters update
        d_texture = np.zeros(self.m)

        # Get camera parameters increment
        if camera_update:
            # Keep the rest
            ds = ds[self.n:]

            # If focal length is not updated, then set its increment to zero
            if not focal_length_update:
                ds = np.insert(ds, 0, [0.])

            # Set increment of the 1st quaternion to one
            ds = np.insert(ds, 1, [1.])

            # Get camera parameters update
            d_camera = ds[:camera.n_parameters]

            # Get texture parameters increment
            if reconstruction_prior_weight is not None:
                d_texture = ds[camera.n_parameters:]
        else:
            d_camera = None
            if reconstruction_prior_weight is not None:
                d_texture = ds[self.n:]

        return d_shape, d_camera, d_texture

    def J_data(self, camera, warped_uv, shape_pc_uv, texture_pc_uv, grad_x_uv,
               grad_y_uv, camera_update, focal_length_update,
               reconstruction_prior_weight):
        # Compute derivative of camera wrt shape and camera parameters
        dp_da_dr = d_camera_d_shape_parameters(camera, warped_uv, shape_pc_uv)
        n_camera_parameters = 0
        if camera_update:
            dp_dr = d_camera_d_camera_parameters(
                camera, warped_uv, with_focal_length=focal_length_update)
            dp_da_dr = np.hstack((dp_da_dr, dp_dr))
            n_camera_parameters = dp_dr.shape[1]

        # Multiply image gradient with camera derivative
        permuted_grad_x = np.transpose(grad_x_uv[..., None], (0, 2, 1))
        permuted_grad_y = np.transpose(grad_y_uv[..., None], (0, 2, 1))
        J = permuted_grad_x * dp_da_dr[0] + permuted_grad_y * dp_da_dr[1]

        # Computer derivative of texture wrt texture parameters
        dt_db = - np.rollaxis(texture_pc_uv, 0, 3)

        # Concatenate to create the data term steepest descent
        J = np.hstack((J, dt_db))

        # Reshape to : n_params x (2 * N)
        n_params = J.shape[1]
        J = np.transpose(J, (1, 0, 2)).reshape(n_params, -1)
        return reconstruction_prior_weight * J, n_camera_parameters

    def __str__(self):
        return "Simultaneous Forward Additive"


class WibergForwardAdditive(LucasKanade):
    r"""
    Class for defining Wiberg Forward Additive Morphable Model optimization
    algorithm.
    """
    def run(self, image, initial_mesh, camera, gt_mesh=None, max_iters=20,
            camera_update=False, focal_length_update=False,
            shape_prior_weight=1., texture_prior_weight=1., landmarks=None,
            landmarks_prior_weight=1., return_costs=False, verbose=True):
        # Parse landmarks prior options
        if landmarks is None or landmarks_prior_weight is None:
            landmarks_prior_weight = None
            landmarks = None
        if landmarks is not None:
            lms_points = landmarks.points[:, [1, 0]]

        # Retrieve camera parameters from the provided camera object.
        # Project provided instance to retrieve shape and texture parameters.
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
        shape_parameters_per_iter = [shape_parameters]
        texture_parameters_per_iter = [texture_parameters]
        camera_per_iter = [camera]
        instance_per_iter = [instance.with_rescaled_texture(0., 1.)]
        costs = None
        if return_costs:
            costs = []

        # Initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Main loop
        while k < max_iters and eps > self.eps:
            if verbose:
                print_dynamic("{}/{}".format(k + 1, max_iters))
                print()
            # Apply camera projection on current instance
            instance_in_image = camera.apply(instance)

            # Compute indices locations for sampling
            (vertex_indices, tri_indices,
             b_coords, yx) = self.visible_sample_points(instance_in_image,
                                                        image.shape)

            # Warp the mesh with the view matrix (rotation + translation)
            instance_w = camera.view_transform.apply(instance.points)

            # Sample all the terms from the model part at the sample locations
            warped_uv = self.sample(instance_w, vertex_indices, b_coords)
            m_texture_uv = self.model.instance().sample_texture_with_barycentric_coordinates(b_coords, tri_indices)
            shape_pc_uv = self.sample(self.shape_pc, vertex_indices, b_coords)
            # Reshape shape basis after sampling
            shape_pc_uv = shape_pc_uv.reshape([self.n_samples, 3, -1])

            # Sample all the terms from the image part at the sample locations
            img_uv = image.sample(yx)
            grad_x_uv = grad_x.sample(yx)
            grad_y_uv = grad_y.sample(yx)

            # Compute error
            img_error_uv = img_uv - m_texture_uv.T

            # Compute Jacobian, SD and Hessian of data term
            sd, n_camera_parameters = self.J_data(
                camera, warped_uv, shape_pc_uv, grad_x_uv, grad_y_uv,
                camera_update, focal_length_update)
            hessian = sd.T.dot(sd)
            sd_error = sd.T.dot(img_error_uv.ravel())

            # Compute Jacobian, update SD and Hessian wrt shape prior
            if shape_prior_weight is not None:
                sd_shape = shape_prior_weight * self.J_shape_prior
                hessian[:self.n, :self.n] += np.diag(sd_shape)
                sd_error[:self.n] += sd_shape * shape_parameters

            # Compute Jacobian, update SD and Hessian wrt landmarks prior
            lms_error = None
            if landmarks_prior_weight is not None:
                # Get projected instance on landmarks and error term
                warped_lms = instance_in_image.points[
                    self.model.model_landmarks_index]
                lms_error = (warped_lms[:, [1, 0]] - lms_points).T

                # Jacobian and Hessian wrt shape parameters
                warped_view_lms = instance_w[self.model.model_landmarks_index]
                sd_lms_shape = d_camera_d_shape_parameters(
                    camera, warped_view_lms, self.shape_pc_lms)
                hessian[:self.n, :self.n] += (
                    landmarks_prior_weight * self.compute_hessian(sd_lms_shape))
                sd_error[:self.n] += (
                    landmarks_prior_weight * self.compute_sd_error(sd_lms_shape,
                                                                   lms_error))

                # Jacobian and Hessian wrt camera parameters
                if camera_update:
                    sd_lms_camera = d_camera_d_camera_parameters(
                        camera, warped_view_lms,
                        with_focal_length=focal_length_update)
                    hessian[self.n:, self.n:] += (
                        landmarks_prior_weight *
                        self.compute_hessian(sd_lms_camera))
                    sd_error[self.n:] += (
                        landmarks_prior_weight *
                        self.compute_sd_error(sd_lms_camera, lms_error))

            if return_costs:
                costs.append(self.compute_cost(
                    img_error_uv, lms_error, shape_parameters,
                    texture_parameters, shape_prior_weight,
                    texture_prior_weight, landmarks_prior_weight))

            # Solve to find the increment of parameters
            d_shape, d_camera = self.solve(hessian, sd_error, camera_update,
                                           focal_length_update)

            # Update parameters
            shape_parameters += d_shape
            if camera_update:
                camera_parameters = camera_parameters_update(
                    camera_parameters, d_camera)
                camera = camera.from_vector(camera_parameters)

            # Generate the updated instance
            instance = self.model.instance(shape_weights=shape_parameters,
                                           texture_weights=texture_parameters)

            # Update lists
            shape_parameters_per_iter.append(shape_parameters)
            texture_parameters_per_iter.append(texture_parameters)
            camera_per_iter.append(camera)
            instance_per_iter.append(instance.with_rescaled_texture(0., 1.))

            # Increase iteration counter
            k += 1

            # shape_parameters, texture_parameters, camera_parameters = yield \
            #     shape_parameters, texture_parameters, camera_parameters

        return MMAlgorithmResult(
            shape_parameters=shape_parameters_per_iter,
            texture_parameters=texture_parameters_per_iter,
            meshes=instance_per_iter, camera_transforms=camera_per_iter,
            image=image, initial_mesh=initial_mesh.with_rescaled_texture(0., 1.),
            initial_camera_transform=camera_per_iter[0], gt_mesh=gt_mesh,
            costs=costs)

    def project_out(self, J):
        tmp = self.model.texture_model.components.dot(J)
        return J - self.texture_T.dot(tmp)

    def compute_hessian(self, sd):
        n_params = sd.shape[1]
        sd = np.transpose(sd, (1, 0, 2)).reshape(n_params, -1)
        return sd.dot(sd.T)

    def J_data(self, camera, warped_uv, shape_pc_uv, grad_x_uv, grad_y_uv,
               camera_update, focal_length_update):
        # Compute derivative of camera wrt shape and camera parameters
        dp_da_dr = d_camera_d_shape_parameters(camera, warped_uv, shape_pc_uv)
        n_camera_parameters = 0
        if camera_update:
            dp_dr = d_camera_d_camera_parameters(
                camera, warped_uv, with_focal_length=focal_length_update)
            dp_da_dr = np.hstack((dp_da_dr, dp_dr))
            n_camera_parameters = dp_dr.shape[1]

        # Multiply image gradient with camera derivative
        permuted_grad_x = np.transpose(grad_x_uv[..., None], (0, 2, 1))
        permuted_grad_y = np.transpose(grad_y_uv[..., None], (0, 2, 1))
        J = permuted_grad_x * dp_da_dr[0] + permuted_grad_y * dp_da_dr[1]

        # Project-out
        n_params = J.shape[1]
        J = np.transpose(J, (1, 0, 2)).reshape(n_params, -1)
        PJ = self.project_out(J)

        # Concatenate to create the data term steepest descent
        return PJ, n_camera_parameters

    def solve(self, hessian, sd_error, camera_update, focal_length_update):
        # Solve
        ds = - np.linalg.solve(hessian, sd_error)

        # Get shape parameters increment
        d_shape = ds[:self.n]

        # Get camera parameters increment
        if camera_update:
            # Keep the rest
            ds = ds[self.n:]

            # If focal length is not updated, then set its increment to zero
            if not focal_length_update:
                ds = np.insert(ds, 0, [0.])

            # Set increment of the 1st quaternion to one
            ds = np.insert(ds, 1, [1.])

            # Get camera parameters update
            d_camera = ds
        else:
            d_camera = None

        return d_shape, d_camera

    def _precompute(self):
        # call super method
        super(WibergForwardAdditive, self)._precompute()
        self.texture_T = self.model.texture_model.components.T

    def __str__(self):
        return "Wiberg Forward Additive"


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
