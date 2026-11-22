import numpy as np


# TODO: Should inherit from PCAModel (?)
class MultilinearModel(object):
    """Usage for a morphable model:

    shape_model = MultilinearModel(identity_model, expression_model)
    mm = TexturedMorphableModel(shape_model, texture_model, landmarks, tcoords, no_op, None)
    """

    def __init__(self, *models):
        self.models = models

    @property
    def n_components(self):
        r"""
        The number of components of all the models together.

        :type: `int`
        """
        return int(np.sum([mdl.n_components for mdl in self.models]))

    def trim_components(self, n_components):
        r"""
        Returns a new multilinear model whose models are trimmed (see
        `menpo.model.PCAVectorModel.trim_components`: they share the memory of these
        models, and leave them as they are).

        Parameters
        ----------
        n_components : `int` or `float` or a sequence of those
            A number of components for each model, or one number for the first
            model only. The models it does not give a number for keep all their
            components.

        Returns
        -------
        model : :map:`MultilinearModel`
            The trimmed model.
        """
        try:
            numbers = list(n_components)
        except TypeError:
            numbers = [n_components]
        numbers += [None] * (len(self.models) - len(numbers))
        return MultilinearModel(
            *[
                mdl.trim_components(mdl.n_components if n is None else n)
                for mdl, n in zip(self.models, numbers)
            ]
        )

    def noise_variance(self):
        return self.models[0].noise_variance()

    def mean(self):
        v = np.sum([mdl.mean_vector for mdl in self.models], 0)

        return self.template_instance.from_vector(v)

    def instance(self, weights, normalized_weights=False):
        n_active = np.cumsum([0] + [mdl.n_components for mdl in self.models])
        slices = (slice(*s) for s in zip(n_active, n_active[1:]))

        v = np.sum(
            [
                mdl.instance_vector(weights[start_end], normalized_weights)
                for mdl, start_end in zip(self.models, slices)
            ],
            0,
        )

        return self.template_instance.from_vector(v)

    @property
    def components(self):
        return np.concatenate([mdl.components for mdl in self.models], 0)

    @property
    def template_instance(self):
        # TODO: Assert all models are of the same template instance.
        return self.models[0].template_instance

    @property
    def eigenvalues(self):
        return np.concatenate([mdl.eigenvalues for mdl in self.models], 0)

    def project(self, v):
        # Warning: Hacky way to project. Only projects onto the first
        # models components and sets the rest to zero.

        n_components = [mdl.n_components for mdl in self.models]
        e = np.zeros(sum(n_components[1:]))
        return np.hstack([self.models[0].project(v), e])
