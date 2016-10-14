from .algorithm import Simultaneous


class MMFitter(object):
    r"""
    Abstract class for defining an Morphable Model fitter.

    Parameters
    ----------
    mm : :map:`ColouredMorphableModel` or `subclass`
        The trained Morphable Model.
    algorithm : `class`
        The algorithm object that will perform the fitting.
    projection_type : ``{'orthographic', 'perspective'}``, optional
        The type of projection from 3D to 2D. It can be either
        `'orthographic'` or `'perspective'`.
    """
    def __init__(self, mm, algorithm):
        # Assign model and algorithm objects
        self._model = mm
        self.algorithm = algorithm

    @property
    def mm(self):
        r"""
        The trained Morphable Model.

        :type: :map:`ColouredMorphableModel` or `subclass`
        """
        return self._model

    def fit_from_shape(self, image, initial_shape, camera_update=False,
                       max_iters=100, return_costs=False):
        return self.algorithm.run(image, initial_shape,
                                  camera_update=camera_update,
                                  max_iters=max_iters,
                                  return_costs=return_costs)

    def _fitter_result(self, image, algorithm_result, gt_shape=None):
        pass


class LucasKanadeMMFitter(MMFitter):
    def __init__(self, mm, lk_algorithm_cls=Simultaneous,
                 n_shape=None, n_texture=None, n_samples=1000,
                 projection_type='perspective'):
        # Check parameters
        set_model_components(mm.shape_model, n_shape)
        set_model_components(mm.texture_model, n_texture)
        self.n_samples = n_samples
        if projection_type in ['orthographic', 'perspective']:
            self.projection_type = projection_type
        else:
            raise ValueError("Projection type can be either 'perspective' or "
                             "'orthographic'")

        # Get algorithm object
        algorithm = lk_algorithm_cls(mm, n_samples, self.projection_type)

        # Call superclass
        super(LucasKanadeMMFitter, self).__init__(mm=mm, algorithm=algorithm)

    def __str__(self):
        cls_str = r"""{}
 - {} active shape components
 - {} active texture components""".format(
            self.algorithm.__str__(), self.mm.shape_model.n_active_components,
            self.mm.texture_model.n_active_components)
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
