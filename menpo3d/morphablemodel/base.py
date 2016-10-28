import numpy as np

from menpo.base import name_of_callable
from menpo.shape import ColouredTriMesh, TexturedTriMesh


class MorphableModel(object):
    r"""
    Class for defining a Coloured Morphable Model. Please see the references
    for a basic list of relevant papers.

    Parameters
    ----------
    shape_model : `menpo.model.PCAModel`
        The PCA model of the 3D shape. It is assumed that a shape instance is
        defined as a `menpo.shape.TriMesh`.
    texture_model : `menpo.model.PCAVectorModel`
        The PCA model of the 3D texture. It is assumed that a texture
        instance is defined as an ``(n_vertices * n_channels,)`` vector,
        where `n_vertices` should be the same as in the case of the
        `shape_model`.
    landmarks : `menpo.shape.PointUndirectedGraph`
        The set of sparse landmarks defined in the 3D space.

    References
    ----------
    .. [1] V. Blanz, T. Vetter. "A morphable model for the synthesis of 3D
        faces", Conference on Computer Graphics and Interactive Techniques,
        pp. 187-194, 1999.
    .. [2] P. Paysan, R. Knothe, B. Amberg, S. Romdhani, T. Vetter. "A 3D
        face model for pose and illumination invariant face recognition",
        IEEE International Conference on Advanced Video and Signal Based
        Surveillance, pp. 296-301, 2009.
    """
    def __init__(self, shape_model, texture_model, landmarks):
        self.shape_model = shape_model
        self.texture_model = texture_model
        self.landmarks = landmarks

    @property
    def n_vertices(self):
        """
        Returns the number of vertices of the shape model's trimesh.

        :type: `int`
        """
        return self.shape_model.template_instance.n_points

    @property
    def n_triangles(self):
        """
        Returns the number of triangles of the shape model's trimesh.

        :type: `int`
        """
        return self.shape_model.template_instance.n_tris

    @property
    def n_channels(self):
        """
        Returns the number of channels of the texture model.

        :type: `int`
        """
        return int(self.texture_model.n_features / self.n_vertices)

    @property
    def _str_title(self):
        return 'Coloured Morphable Model'

    def instance(self, shape_weights=None, texture_weights=None,
                 landmark_group='landmarks'):
        r"""
        Generates a novel Morphable Model instance given a set of shape and
        texture weights. If no weights are provided, then the mean Morphable
        Model instance is returned.

        Parameters
        ----------
        shape_weights : ``(n_weights,)`` `ndarray` or `list` or ``None``, optional
            The weights of the shape model that will be used to create a novel
            shape instance. If ``None``, the weights are assumed to be zero,
            thus the mean shape is used.
        texture_weights : ``(n_weights,)`` `ndarray` or `list` or ``None``, optional
            The weights of the texture model that will be used to create a
            novel texture instance. If ``None``, the weights are assumed
            to be zero, thus the mean appearance is used.
        landmark_group : `str`, optional
            The group name that will be used for the sparse landmarks that
            will be attached to the returned instance.

        Returns
        -------
        instance : `menpo.shape.ColouredTriMesh`
            The coloured trimesh instance.
        """
        if shape_weights is None:
            shape_weights = np.zeros(self.shape_model.n_active_components)
        if texture_weights is None:
            texture_weights = np.zeros(self.texture_model.n_active_components)

        # Generate instance
        shape_instance = self.shape_model.instance(shape_weights)
        texture_instance = self.texture_model.instance(texture_weights)

        # Create and return trimesh
        return self._instance(shape_instance, texture_instance, landmark_group)

    def random_instance(self, landmark_group='__landmarks__'):
        r"""
        Generates a random instance of the Morphable Model.

        Parameters
        ----------
        landmark_group : `str`, optional
            The group name that will be used for the sparse landmarks that
            will be attached to the returned instance. Default is
            ``'__landmarks__'``.

        Returns
        -------
        instance : `menpo.shape.ColouredTriMesh`
            The coloured trimesh instance.
        """
        # TODO: this bit of logic should to be transferred down to PCAModel
        shape_weights = np.random.randn(self.shape_model.n_active_components)
        shape_instance = self.shape_model.instance(shape_weights)
        texture_weights = np.random.randn(
            self.texture_model.n_active_components)
        texture_instance = self.texture_model.instance(texture_weights)

        return self._instance(shape_instance, texture_instance, landmark_group)

    def view_shape_model_widget(self, n_parameters=5,
                                parameters_bounds=(-15.0, 15.0),
                                mode='multiple'):
        r"""
        Visualizes the shape model of the Morphable Model using an interactive
        widget.

        Parameters
        ----------
        n_parameters : `int` or `list` of `int` or ``None``, optional
            The number of shape principal components to be used for the
            parameters sliders. If `int`, then the number of sliders per
            scale is the minimum between `n_parameters` and the number of
            active components per scale. If `list` of `int`, then a number of
            sliders is defined per scale. If ``None``, all the active
            components per scale will have a slider.
        parameters_bounds : ``(float, float)``, optional
            The minimum and maximum bounds, in std units, for the sliders.
        mode : {``single``, ``multiple``}, optional
            If ``'single'``, only a single slider is constructed along with a
            drop down menu. If ``'multiple'``, a slider is constructed for
            each parameter.
        """
        try:
            from menpowidgets import visualize_shape_model_3d
            visualize_shape_model_3d(self.shape_model,
                                     n_parameters=n_parameters,
                                     parameters_bounds=parameters_bounds,
                                     mode=mode)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def view_mm_widget(self, n_shape_parameters=5, n_texture_parameters=5,
                       parameters_bounds=(-15.0, 15.0), mode='multiple'):
        r"""
        Visualizes the Morphable Model using an interactive widget.

        Parameters
        ----------
        n_shape_parameters : `int` or `list` of `int` or ``None``, optional
            The number of shape principal components to be used for the
            parameters sliders. If `int`, then the number of sliders per
            scale is the minimum between `n_parameters` and the number of
            active components per scale. If `list` of `int`, then a number of
            sliders is defined per scale. If ``None``, all the active
            components per scale will have a slider.
        n_texture_parameters : `int` or `list` of `int` or ``None``, optional
            The number of texture principal components to be used for the
            parameters sliders. If `int`, then the number of sliders per
            scale is the minimum between `n_parameters` and the number of
            active components per scale. If `list` of `int`, then a number of
            sliders is defined per scale. If ``None``, all the active
            components per scale will have a slider.
        parameters_bounds : ``(float, float)``, optional
            The minimum and maximum bounds, in std units, for the sliders.
        mode : {``single``, ``multiple``}, optional
            If ``'single'``, only a single slider is constructed along with a
            drop down menu. If ``'multiple'``, a slider is constructed for
            each parameter.
        """
        try:
            from menpowidgets import visualize_morphable_model
            visualize_morphable_model(
                self, n_shape_parameters=n_shape_parameters,
                n_texture_parameters=n_texture_parameters,
                parameters_bounds=parameters_bounds, mode=mode)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def __str__(self):
        cls_str = r"""{}
 - Shape model class: {}
   - {} vertices, {} triangles
   - {} shape components
   - Instance class: {}
 - Texture model class: {}
   - {} texture components
   - {} channels
 - Sparse landmarks class: {}
   - {} landmarks
""".format(self._str_title, name_of_callable(self.shape_model),
           self.n_vertices, self.n_triangles, self.shape_model.n_components,
           name_of_callable(self.shape_model.template_instance),
           name_of_callable(self.texture_model),
           self.texture_model.n_components, self.n_channels,
           name_of_callable(self.landmarks), self.landmarks.n_points)
        return cls_str


class ColouredMorphableModel(MorphableModel):

    def _instance(self, shape_instance, texture_instance, landmark_group):
        # Reshape the texture instance
        texture_instance = texture_instance.reshape([-1, self.n_channels])
        # Create trimesh
        trimesh = ColouredTriMesh(shape_instance.points,
                                  trilist=shape_instance.trilist,
                                  colours=texture_instance)
        # Attach landmarks to trimesh
        trimesh.landmarks[landmark_group] = self.landmarks
        # Return trimesh
        return trimesh


class TexturedMorphableModel(MorphableModel):

    def __init__(self, shape_model, texture_model, landmarks, tcoords):
        super(TexturedMorphableModel, self).__init__(shape_model,
                                                     texture_model, landmarks)
        self.tcoords = tcoords

    def _instance(self, shape_instance, texture_instance, landmark_group):
        # Create trimesh
        trimesh = TexturedTriMesh(shape_instance.points,
                                  trilist=shape_instance.trilist,
                                  tcoords=tcoords,
                                  texture=texture_instance)
        # Attach landmarks to trimesh
        trimesh.landmarks[landmark_group] = self.landmarks
        # Return trimesh
        return trimesh
