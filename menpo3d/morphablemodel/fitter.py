from menpo3d.rasterize import GLRasterizer
import menpo3d.checks as checks

from .algorithm import Simultaneous
from .projection import compute_view_projection_transforms


class MMFitter(object):
    r"""
    Abstract class for defining a multi-scale Morphable Model fitter.

    Parameters
    ----------
    mm : :map:`ColouredMorphableModel` or `subclass`
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

        :type: :map:`ColouredMorphableModel` or `subclass`
        """
        return self._model

    def _fit(self, image, view_transform, projection_transform,
             rotation_transfrom, instance=None, camera_update=False,
             max_iters=50, return_costs=False):
        # Check provided instance
        if instance is None:
            instance = self.mm.instance()

        # Check arguments
        max_iters = checks.check_max_iters(max_iters, self.n_scales)

        # Initialize the rasterizer
        rasterizer = GLRasterizer(
            height=image.height, width=image.width,
            view_matrix=view_transform.h_matrix,
            projection_matrix=projection_transform.h_matrix)

        # Initialize list of algorithm results
        algorithm_results = []

        # Main loop at each scale level
        for i in range(self.n_scales):
            # Run algorithm
            algorithm_result = self.algorithms[i].run(
                image, instance, rasterizer, view_transform,
                projection_transform, rotation_transfrom,
                camera_update=camera_update, max_iters=max_iters[i],
                return_costs=return_costs)

            # Get current instance
            instance = algorithm_result[1]

            # Add algorithm result to the list
            algorithm_results.append(algorithm_result)

        # Destroy rasterizer
        del rasterizer

        return algorithm_results

    def fit_from_shape(self, image, initial_shape, camera_update=False,
                       max_iters=50, return_costs=False,
                       distortion_coeffs=None, gt_shape=None):
        # Check that the provided initial shape has the same number of points
        # as the landmarks of the model
        if initial_shape.n_points != self.mm.landmarks.n_points:
            raise ValueError(
                "The provided 2D initial shape must have {} landmark "
                "points.".format(self.mm.landmarks.n_points))

        # Estimate view, projection and rotation transforms from the
        # provided initial shape
        view_t, projection_t, rotation_t = compute_view_projection_transforms(
            image=image, mesh=self.mm.shape_model.mean(),
            image_pointcloud=initial_shape, mesh_pointcloud=self.mm.landmarks,
            distortion_coeffs=distortion_coeffs)

        # Execute multi-scale fitting
        algorithm_results = self._fit(
            image=image, view_transform=view_t,
            projection_transform=projection_t, rotation_transfrom=rotation_t,
            camera_update=camera_update, max_iters=max_iters,
            return_costs=return_costs)

        # Return multi-scale fitting result
        return self._fitter_result(
            image=image, algorithm_results=algorithm_results, gt_shape=gt_shape)

    def _fitter_result(self, image, algorithm_results, gt_shape=None):
        rasterized_results = []
        instances = []
        costs = []
        a_list = []
        b_list = []
        r_list = []
        for r in algorithm_results:
            rasterized_results += r[0]
            instances.append(r[1])
            costs += r[2]
            a_list += r[3]
            b_list += r[4]
            r_list += r[5]
        return rasterized_results, instances, costs, a_list, b_list, r_list


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
