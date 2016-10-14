import numpy as np

from menpo.feature import gradient as fast_gradient
from menpo.image import Image
from menpo.visualize import print_dynamic

from menpo3d.rasterize import GLRasterizer

from .derivatives import (d_orthographic_projection_d_shape_parameters,
                          d_perspective_projection_d_shape_parameters,
                          d_orthographic_projection_d_warp_parameters,
                          d_perspective_projection_d_warp_parameters,
                          d_texture_d_texture_parameters)
from ..projection import (compute_view_projection_transforms,
                          get_camera_parameters, compute_rotation_matrices)


class LucasKanade(object):
    def __init__(self, model, n_samples, projection_type, eps=1e-3):
        self.model = model
        self.eps = eps
        self.n_samples = n_samples
        self.projection_type = projection_type
        self.rasterizer = None

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

    def initialize_camera_parameters(self, image, shape,
                                     distortion_coeffs=None):
        r"""
        Retrieves the camera model properties, i.e. view transform,
        projection transform, rotation transform and the warp parameters
        vector (focal length, varphi, theta, phi, translation X, translation Y).

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The input image.
        shape : `menpo.shape.PointCloud` or subclass
            The input 2D pointcloud defined on the image coordinate space.
        distortion_coeffs : ``(4,)`` or ``(5,)`` or ``(8,)`` `ndarray` or ``None``, optional
            The distortion coefficients of 4, 5, or 8 elements. If ``None``,
            then the distortion coefficients are set to zero.

        Returns
        -------
        view_transform : `menpo.transform.Homogeneous`
            The view transform object.
        projection_transform : `menpo.transform.Homogeneous`
            The projection transform object.
        rotation_transform : `menpo.transform.Rotation`
            The rotation transform object.
        params : ``(6,)`` `ndarray`
            The camera parameters:
            focal length, varphi, theta, phi, translation X, translation Y
        """
        # Estimate view transform, projection transform and rotation
        view_t, projection_t, rotation_t = compute_view_projection_transforms(
            image=image, mesh=self.model.shape_model.mean(),
            image_pointcloud=shape, mesh_pointcloud=self.model.landmarks,
            distortion_coeffs=distortion_coeffs)

        # Get warp (camera) parameters vector
        warp_parameters = get_camera_parameters(projection_t, view_t)

        return view_t, projection_t, rotation_t, warp_parameters

    def initialize_rasterizer(self, image, view_transform,
                              projection_transform):
        r"""
        Initializes the rasterizer using :map:`GLRasterizer`.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The input image.
        view_transform : `menpo.transform.Homogeneous`
            The view transform object of the camera.
        projection_transform : `menpo.transform.Homogeneous`
            The projection transform object of the camera.
        """
        self.rasterizer = GLRasterizer(
            height=image.height, width=image.width,
            view_matrix=view_transform.h_matrix,
            projection_matrix=projection_transform.h_matrix)

    def compute_warp_indices(self, instance):
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
        true_indices : ``(n_samples,)`` `ndarray`
            The indices of the true points.
        """
        # Inverse rendering
        tri_index_img, b_coords_img = \
            self.rasterizer.rasterize_barycentric_coordinate_image(instance)
        print(tri_index_img)
        print(b_coords_img)
        tri_indices = tri_index_img.as_vector()
        b_coords = b_coords_img.as_vector(keep_channels=True)
        true_indices = tri_index_img.mask.true_indices()

        # Select triangles randomly
        rand = np.random.permutation(b_coords.shape[1])
        b_coords = b_coords[:, rand[:self.n_samples]]
        true_indices = true_indices[rand[:self.n_samples]]
        tri_indices = tri_indices[rand[:self.n_samples]]

        # Build the vertex indices (3 per pixel) for the visible triangles
        vertex_indices = instance.trilist[tri_indices]

        return vertex_indices, b_coords, true_indices

    def sample(self, x, vertex_indices, b_coords):
        r"""
        Method tht samples an object at locations specified by the
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
        return np.sum(per_vert_per_pixel * b_coords.T[..., None], axis=1)

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

        # Scale the gradient by the image resolution
        if image.shape[1] > image.shape[0]:
            scale = image.shape[1] / 2
        else:
            scale = image.shape[0] / 2

        # Create gradient image for X and Y
        grad_y = Image(grad.pixels[:self.n_channels] * scale)
        grad_x = Image(grad.pixels[self.n_channels:] * scale)

        return grad_x, grad_y

    def d_projection_d_shape_parameters(self, warped_uv, shape_pc_uv,
                                        shape_eigenvalues, focal_length,
                                        rotation_transform):
        if self.projection_type == 'perspective':
            dp_da = d_perspective_projection_d_shape_parameters(
                shape_pc_uv, shape_eigenvalues, focal_length,
                rotation_transform, warped_uv)
        else:
            dp_da = d_orthographic_projection_d_shape_parameters(
                shape_pc_uv, shape_eigenvalues, focal_length,
                rotation_transform)
        return dp_da

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
        texture_pc = self.model.texture_model.components.T
        self.shape_pc = shape_pc.reshape([self.n_vertices, -1])
        self.texture_pc = texture_pc.reshape([self.n_vertices, -1])


class Simultaneous(LucasKanade):
    r"""
    Class for defining Simultaneous Morphable Model optimization algorithm.
    """
    def run(self, image, initial_shape, camera_update=False, max_iters=20,
            return_costs=False):
        r"""
        Execute the optimization algorithm.

        Parameters
        ----------
        image : `menpo.image.Image`
            The input test image.
        initial_shape : `menpo.shape.PointCloud`
            The initial shape from which the optimization will start.
        camera_update : `bool`, optional
            If ``False``, then the camera (warp) parameters are not updated.
        max_iters : `int`, optional
            The maximum number of iterations. Note that the algorithm may
            converge, and thus stop, earlier.
        return_costs : `bool`, optional
            If ``True``, then the cost function values will be computed
            during the fitting procedure. Then these cost values will be
            assigned to the returned `fitting_result`. *Note that the costs
            computation increases the computational cost of the fitting. The
            additional computation cost depends on the fitting method. Only
            use this option for research purposes.*

        Returns
        -------
        fitting_result : :map:`AAMAlgorithmResult`
            The parametric iterative fitting result.
        """
        # Define cost closure
        def cost_closure(x):
            return x.T.dot(x)

        # Retrieve warp (camera) parameters, view transform, projection
        # transform and rotation transform
        view_t, projection_t, rotation_t, warp_parameters = \
            self.initialize_camera_parameters(image, initial_shape)

        # Initialize parameters lists
        shape_parameters = np.zeros(self.n)
        texture_parameters = np.zeros(self.m)
        a_list = [shape_parameters]
        b_list = [texture_parameters]
        r_list = [warp_parameters]
        costs = []

        # Compute input image gradient
        grad_x, grad_y = self.gradient(image)

        # Initialize the rasterizer
        self.initialize_rasterizer(image, view_t, projection_t)

        # Generate instance
        instance = self.model.instance()

        # Prior
        da_prior_db = np.zeros(self.m)
        da_prior_da = 2. / (self.model.shape_model.eigenvalues ** 2)
        db_prior_da = np.zeros(self.n)
        db_prior_db = 2. / (self.model.texture_model.eigenvalues ** 2)
        if camera_update:
            prior_drho = np.zeros(len(warp_parameters))
            sd_alpha_prior = np.concatenate((da_prior_da, prior_drho,
                                             da_prior_db))
            sd_beta_prior = np.concatenate((db_prior_da, prior_drho,
                                            db_prior_db))
        else:
            sd_alpha_prior = np.concatenate((da_prior_da,
                                             da_prior_db))
            sd_beta_prior = np.concatenate((db_prior_da,
                                            db_prior_db))
        h_alpha_prior = np.diag(sd_alpha_prior)
        h_beta_prior = np.diag(sd_beta_prior)

        # Initialize iteration counter and epsilon
        k = 0
        eps = np.Inf
        while k < max_iters and eps > self.eps:
            print_dynamic("{}/{}".format(k, max_iters))
            # Compute indices locations for warping
            vertex_indices, b_coords, true_indices = self.compute_warp_indices(
                instance)

            # Warp the mesh with the view matrix
            W = view_t.apply(instance.points)

            # This solves the perspective projection problems
            # It cancels the axes flip done in the view matrix before the
            # rasterization
            W[:, 1:] *= -1

            # Sample to UV space
            shape_uv = self.sample(instance.points, vertex_indices, b_coords)
            texture_uv = self.sample(instance.colours, vertex_indices, b_coords)
            warped_uv = self.sample(W, vertex_indices, b_coords)
            shape_pc_uv = self.sample(self.shape_pc, vertex_indices, b_coords)
            print(self.texture_pc.shape)
            texture_pc_uv = self.sample(self.texture_pc, vertex_indices, b_coords)
            print(texture_pc_uv.shape)
            img_uv = image.sample(true_indices)
            grad_x_uv = grad_x.sample(true_indices)
            grad_y_uv = grad_y.sample(true_indices)

            # Reshape bases after sampling
            new_shape = texture_pc_uv.shape
            shape_pc_uv = shape_pc_uv.reshape([new_shape[0], 3, -1])
            texture_pc_uv = texture_pc_uv.reshape(
                [new_shape[0], self.model.n_channels, -1])
            print(texture_pc_uv.shape)

            # Compute derivative of projection wrt shape parameters
            dp_da_dr = self.d_projection_d_shape_parameters(
                warped_uv, shape_pc_uv, self.model.shape_model.eigenvalues,
                warp_parameters[0], rotation_t)

            # Compute derivative of projection wrt warp parameters
            if camera_update:
                dp_dr = self.d_projection_d_warp_parameters(
                    shape_uv, warped_uv, warp_parameters)
                # Concatenate it with the derivative wrt shape parameters
                dp_da_dr = np.hstack((dp_da_dr, dp_dr))

            # Compute derivative of texture wrt texture parameters
            dt_db = d_texture_d_texture_parameters(
                texture_pc_uv, self.model.texture_model.eigenvalues)

            # Compute steepest descent
            sd_da_dr = self.compute_steepest_descent(dp_da_dr, grad_x_uv,
                                                     grad_y_uv)
            sd = np.hstack((sd_da_dr, -dt_db))

            # Compute hessian
            h = self.compute_hessian(sd)

            # Compute error
            img_error_uv = img_uv - texture_uv.T

            # Compute steepest descent matrix
            sd_error_img = self.compute_sd_error(sd, img_error_uv)

            # Update costs
            eps = cost_closure(sd_error_img)
            if return_costs:
                costs.append(eps)

            # Prior probabilities over shape parameters
            if camera_update:
                prior_error = np.concatenate((shape_parameters,
                                              warp_parameters,
                                              texture_parameters))
            else:
                prior_error = np.concatenate((shape_parameters,
                                              texture_parameters))

            # Prior probability SD matrices
            sd_error_alpha_prior = sd_alpha_prior * prior_error
            sd_error_beta_prior = sd_beta_prior * prior_error

            # Final hessian and SD error matrix
            h += 0 * h_alpha_prior + h_beta_prior
            sd_error_img += 0 * sd_error_alpha_prior + sd_error_beta_prior

            # Compute increment
            delta_s = -np.dot(np.linalg.inv(h), sd_error_img)

            # Update parameters
            shape_parameters += delta_s[:self.n]
            a_list.append(shape_parameters)
            if camera_update:
                warp_parameters += delta_s[self.n:self.n+len(warp_parameters)]
                r_list.append(warp_parameters)
                texture_parameters += delta_s[(self.n+len(warp_parameters)):]
            else:
                texture_parameters += delta_s[self.n:]
            b_list.append(texture_parameters)

            # Generate the updated instance
            # The texture is scaled by 255 to cancel the 1./255 scaling in the
            # model class
            instance = self.model.instance(
                shape_weights=shape_parameters,
                texture_weights=255. * texture_parameters)

            # Clip to avoid out of range pixels
            instance.colours = np.clip(instance.colours, 0, 1)

            # Update rasterizer
            if camera_update:
                # Compute new view matrix
                _, _, _, rot_t = compute_rotation_matrices(warp_parameters[1],
                                                           warp_parameters[2],
                                                           warp_parameters[3])
                view_t.h_matrix[1:3, :3] = -rot_t.h_matrix[1:3, :3]
                view_t.h_matrix[0, :3] = rot_t.h_matrix[0, :3]

                # Update the rasterizer
                self.rasterizer.set_view_matrix(view_t.h_matrix)

            # Increase iteration counter
            k += 1

        # Rasterize the final mesh
        rasterized_result = self.rasterizer.rasterize_mesh(instance)

        del self.rasterizer

        return rasterized_result, instance, costs, shape_parameters, \
               texture_parameters, warp_parameters

    def __str__(self):
        return "Simultaneous Lucas-Kanade"
