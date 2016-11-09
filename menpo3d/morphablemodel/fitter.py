import numpy as np

from menpo.transform import AlignmentAffine, AlignmentSimilarity, Rotation
from menpo.shape import PointCloud

import menpo3d.checks as checks
from menpo3d.camera import PerspectiveCamera

from .algorithm import Simultaneous
from .result import MMResult
from .shapemodel import ShapeModel


class MMFitter(object):
    r"""
    Abstract class for defining a multi-scale Morphable Model fitter.

    Parameters
    ----------
    mm : :map:`ColouredMorphableModel` or :map:`TexturedMorphableModel`
        The trained Morphable Model.
    algorithms : `list` of `class`
        The list of algorithm objects that will perform the fitting per scale.
    """
    def __init__(self, mm, algorithms):
        # Assign model and algorithms
        self._model = mm
        self.algorithms = algorithms
        self.n_scales = len(self.algorithms)

    @property
    def mm(self):
        r"""
        The trained Morphable Model.

        :type: :map:`ColouredMorphableModel` or `:map:`TexturedMorphableModel`
        """
        return self._model

    @property
    def holistic_features(self):
        r"""
        The features that are extracted from the input image.

        :type: `function`
        """
        return self.mm.holistic_features

    @property
    def diagonal(self):
        r"""
        The diagonal used to rescale the image.

        :type: `int`
        """
        return self.mm.diagonal

    def _prepare_image(self, image, initial_shape):
        r"""
        Function the performs pre-processing on the image to be fitted. This
        involves the following steps:

            1. Rescale image so that the provided initial_shape has the
               specified diagonal.
            2. Compute features
            3. Estimate the affine transform introduced by the rescale to
               diagonal and features extraction

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image to be fitted.
        initial_shape : `menpo.shape.PointCloud`
            The initial shape estimate from which the fitting procedure
            will start.

        Returns
        -------
        image : `menpo.image.Image`
            The feature-based image.
        initial_shape : `menpo.shape.PointCloud`
            The rescaled initial shape.
        affine_transform : `menpo.transform.Affine`
            The affine transform that is the inverse of the transformations
            introduced by the rescale wrt diagonal as well as the feature
            extraction.
        """
        # Attach landmarks to the image, in order to make transforms easier
        image.landmarks['__initial_shape'] = initial_shape

        if self.diagonal is not None:
            # Rescale image so that initial_shape matches the provided diagonal
            tmp_image = image.rescale_landmarks_to_diagonal_range(
                self.diagonal, group='__initial_shape')
        else:
            tmp_image = image
        # Extract features
        feature_image = self.holistic_features(tmp_image)

        # Get final transformed landmarks
        new_initial_shape = feature_image.landmarks['__initial_shape'].lms

        # Now we have introduced an affine transform that consists of the image
        # rescaled based on the diagonal, as well as potential rescale
        # (down-sampling) caused by features. We need to store this transform
        # (estimated by AlignmentAffine) in order to be able to revert it at
        # the final fitting result.
        affine_transform = AlignmentAffine(new_initial_shape, initial_shape)

        # Detach added landmarks from image
        del image.landmarks['__initial_shape']

        return feature_image, new_initial_shape, affine_transform

    def _align_mean_shape_with_bbox(self, bbox):
        # Convert 3D landmarks to 2D by removing the Z axis
        template_shape = PointCloud(self.mm.landmarks.points[:, [1, 0]])

        # Rotation that flips over x axis
        rot_matrix = np.eye(template_shape.n_dims)
        rot_matrix[0, 0] = -1
        template_shape = Rotation(rot_matrix,
                                  skip_checks=True).apply(template_shape)

        # Align the 2D landmarks' bbox with the provided bbox
        return AlignmentSimilarity(template_shape.bounding_box(),
                                   bbox).apply(template_shape)

    def _fit(self, image, camera, instance=None, gt_mesh=None, max_iters=50,
             landmarks_prior=None, parameters_priors=True, camera_update=False,
             focal_length_update=False, return_costs=False):
        # Check provided instance
        if instance is None:
            instance = self.mm.instance()

        # Check arguments
        max_iters = checks.check_max_iters(max_iters, self.n_scales)

        # Initialize list of algorithm results
        algorithm_results = []

        # Main loop at each scale level
        for i in range(self.n_scales):
            # Run algorithm
            algorithm_result = self.algorithms[i].run(
                image, instance, camera, gt_mesh=gt_mesh,
                max_iters=max_iters[i], landmarks_prior=landmarks_prior,
                parameters_priors=parameters_priors,
                camera_update=camera_update,
                focal_length_update=focal_length_update,
                return_costs=return_costs)

            # Get current instance
            instance = algorithm_result.final_mesh
            camera = algorithm_result.final_camera_transform

            # Add algorithm result to the list
            algorithm_results.append(algorithm_result)

        return algorithm_results

    def fit_from_bb(self, image, initial_bb, gt_mesh=None, max_iters=50,
                    parameters_priors=True, camera_update=False,
                    focal_length_update=False, return_costs=False,
                    distortion_coeffs=None):
        initial_shape = self._align_mean_shape_with_bbox(initial_bb)
        return self.fit_from_shape(
            image=image, initial_shape=initial_shape, gt_mesh=gt_mesh,
            max_iters=max_iters, parameters_priors=parameters_priors,
            camera_update=camera_update, landmarks_prior=False,
            focal_length_update=focal_length_update,
            return_costs=return_costs, distortion_coeffs=distortion_coeffs)

    def fit_from_shape(self, image, initial_shape, gt_mesh=None, max_iters=50,
                       landmarks_prior=True, parameters_priors=True,
                       camera_update=False, focal_length_update=False,
                       return_costs=False, distortion_coeffs=None,
                       init_shape_params_from_lms=True):
        # Check that the provided initial shape has the same number of points
        # as the landmarks of the model
        if initial_shape.n_points != self.mm.landmarks.n_points:
            raise ValueError(
                "The provided 2D initial shape must have {} landmark "
                "points.".format(self.mm.landmarks.n_points))

        # Rescale image and extract features
        rescaled_image, rescaled_initial_shape, affine_transform = \
            self._prepare_image(image, initial_shape)

        # Estimate view, projection and rotation transforms from the
        # provided initial shape
        camera = PerspectiveCamera.init_from_2d_projected_shape(
            self.mm.landmarks, rescaled_initial_shape, rescaled_image.shape,
            distortion_coeffs=distortion_coeffs)

        if init_shape_params_from_lms:
            # Wrap the shape model in a container that allows us to mask the
            # PCA basis spatially
            sm = ShapeModel(self.mm.shape_model)
            # Only keep the landmark points in the basis
            sm_lms_3d = sm.mask_points(self.mm.model_landmarks_index)

            # Warp the basis with the camera and retain only the first two dims
            sm_lms_2d = camera.apply(sm_lms_3d).mask_dims([0, 1])
            # Project onto the first few shape components to give an initial
            # shape
            shape_weights = sm_lms_2d.project(rescaled_initial_shape,
                                              n_components=10)
            instance = self.mm.instance(shape_weights)
        else:
            instance = None

        # Set landmarks prior argument
        landmarks_prior = rescaled_initial_shape if landmarks_prior else None

        # Execute multi-scale fitting
        algorithm_results = self._fit(
            rescaled_image, camera, gt_mesh=gt_mesh, max_iters=max_iters,
            landmarks_prior=landmarks_prior, instance=instance,
            parameters_priors=parameters_priors, camera_update=camera_update,
            focal_length_update=focal_length_update, return_costs=return_costs)

        # Return multi-scale fitting result
        return self._fitter_result(
            image=image, algorithm_results=algorithm_results,
            affine_transform=affine_transform, gt_mesh=gt_mesh)

    def _fitter_result(self, image, algorithm_results, affine_transform,
                       gt_mesh=None):
        return MMResult(algorithm_results, [affine_transform] * self.n_scales,
                        self.n_scales, image=image, gt_mesh=gt_mesh,
                        model_landmarks_index=self.mm.model_landmarks_index)


class LucasKanadeMMFitter(MMFitter):
    def __init__(self, mm, lk_algorithm_cls=Simultaneous, n_scales=1,
                 n_shape=1.0, n_texture=1.0, n_samples=1000,
                 projection_type='perspective'):
        # Check parameters
        n_shape = checks.check_multi_scale_param(n_scales, int, 'n_shape',
                                                 n_shape)
        n_texture = checks.check_multi_scale_param(n_scales, int,
                                                   'n_texture', n_texture)
        self.n_samples = checks.check_multi_scale_param(n_scales, int,
                                                        'n_samples', n_samples)
        if projection_type in ['orthographic', 'perspective']:
            self.projection_type = projection_type
        else:
            raise ValueError("Projection type can be either 'perspective' or "
                             "'orthographic'")

        # Create list of algorithms
        algorithms = []
        for i in range(n_scales):
            mm_copy = mm.copy()
            set_model_components(mm_copy.shape_model, n_shape[i])
            set_model_components(mm_copy.texture_model, n_texture[i])
            algorithms.append(lk_algorithm_cls(mm_copy, self.n_samples[i],
                                               self.projection_type))

        # Call superclass
        super(LucasKanadeMMFitter, self).__init__(mm=mm, algorithms=algorithms)

    def __str__(self):
        scales_info = []
        lvl_str_tmplt = r"""   - Scale {}
     - {} active shape components
     - {} active texture components"""
        for k in range(self.n_scales):
            scales_info.append(lvl_str_tmplt.format(
                k,
                self.algorithms[k].model.shape_model.n_active_components,
                self.algorithms[k].model.texture_model.n_active_components))
        scales_info = '\n'.join(scales_info)

        cls_str = r"""{class_title}
 - {scales} scales
{scales_info}
    """.format(class_title=self.algorithms[0].__str__(),
               scales=self.n_scales, scales_info=scales_info)
        return self.mm.__str__() + cls_str


def set_model_components(model, n_components):
    r"""
    Function that sets the number of active components to the provided model.

    Parameters
    ----------
    model : `menpo.model.PCAVectorModel` or `menpo.model.PCAModel` or subclass
        The PCA model.
    n_components : `int` or `float` or ``None``
        The number of active components to set.

    Raises
    ------
    ValueError
        n_components can be an integer or a float or None
    """
    if n_components is not None:
        if type(n_components) is int or type(n_components) is float:
            model.n_active_components = n_components
        else:
            raise ValueError('n_components can be an integer or a float or '
                             'None')
