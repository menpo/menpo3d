import numpy as np
from collections import Iterable

from menpo.base import LazyList
from menpo.image import Image
from menpo.transform import Homogeneous
from menpo.shape import PointCloud

from menpo3d.rasterize import rasterize_mesh


def error_function(mesh, gt_mesh):
    return 1.


class Result(object):
    r"""
    Class for defining a basic fitting result. It holds the final mesh of a
    fitting process and, optionally, the initial mesh, ground truth mesh
    and the image object.

    Parameters
    ----------
    final_mesh : `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh`
        The final mesh of the fitting process.
    final_camera_transform : `menpo3d.camera.PerspectiveCamera`
        The final camera transform object.
    image : `menpo.image.Image` or `subclass` or ``None``, optional
        The image on which the fitting process was applied. Note that a copy
        of the image will be assigned as an attribute. If ``None``, then no
        image is assigned.
    initial_mesh : `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh` or ``None``, optional
        The initial mesh that was provided to the fitting method to
        initialise the fitting process. If ``None``, then no initial mesh is
        assigned.
    initial_camera_transform : `menpo3d.camera.PerspectiveCamera` or ``None``, optional
        The initial camera transform object. If ``None``, then no initial
        camera is assigned.
    gt_mesh : `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh` or ``None``, optional
        The ground truth mesh associated with the image. If ``None``, then no
        ground truth mesh is assigned.
    """
    def __init__(self, final_mesh, final_camera_transform, image=None,
                 initial_mesh=None, initial_camera_transform=None,
                 gt_mesh=None):
        self._final_mesh = final_mesh
        self._final_camera_transform = final_camera_transform
        self._initial_mesh = initial_mesh
        self._initial_camera_transform = initial_camera_transform
        self._gt_mesh = gt_mesh
        # If image is provided, create a copy
        self._image = None
        if image is not None:
            self._image = Image(image.pixels)

    @property
    def is_iterative(self):
        r"""
        Flag whether the object is an iterative fitting result.

        :type: `bool`
        """
        return False

    @property
    def final_mesh(self):
        r"""
        Returns the final mesh of the fitting process.

        :type: `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh`
        """
        return self._final_mesh

    @property
    def final_camera_transform(self):
        r"""
        Returns the final camera transform of the fitting process.

        :type: `menpo3d.camera.PerspectiveCamera`
        """
        return self._final_camera_transform

    @property
    def initial_mesh(self):
        r"""
        Returns the initial mesh that was provided to the fitting method to
        initialise the fitting process. In case the initial mesh does not
        exist, then ``None`` is returned.

        :type: `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh` or ``None``
        """
        return self._initial_mesh

    @property
    def initial_camera_transform(self):
        r"""
        Returns the initial camera transform of the fitting process.

        :type: `menpo3d.camera.PerspectiveCamera` or ``None``
        """
        return self._initial_camera_transform

    @property
    def gt_mesh(self):
        r"""
        Returns the ground truth mesh associated with the image. In case there
        is not an attached ground truth mesh, then ``None`` is returned.

        :type: `menpo.shape.TriMesh` or ``None``
        """
        return self._gt_mesh

    @property
    def image(self):
        r"""
        Returns the image that the fitting was applied on, if it was provided.
        Otherwise, it returns ``None``.

        :type: `menpo.shape.Image` or `subclass` or ``None``
        """
        return self._image

    def final_error(self, compute_error):
        r"""
        Returns the final error of the fitting process, if the ground truth
        mesh exists. This is the error computed based on the `final_mesh`.

        Parameters
        ----------
        compute_error: `callable` or ``None``, optional
            Callable that computes the error between the fitted and
            ground truth meshes.

        Returns
        -------
        final_error : `float`
            The final error at the end of the fitting process.

        Raises
        ------
        ValueError
            Ground truth mesh has not been set, so the final error cannot be
            computed
        """
        if compute_error is None:
            compute_error = error_function
        if self.gt_mesh is not None:
            return compute_error(self.final_mesh, self.gt_mesh)
        else:
            raise ValueError('Ground truth mesh has not been set, so the '
                             'final error cannot be computed')

    def initial_error(self, compute_error):
        r"""
        Returns the initial error of the fitting process, if the ground truth
        mesh and initial mesh exist. This is the error computed based on the
        `initial_mesh`.

        Parameters
        ----------
        compute_error: `callable` or ``None``, optional
            Callable that computes the error between the initial and
            ground truth meshes.

        Returns
        -------
        initial_error : `float`
            The initial error at the beginning of the fitting process.

        Raises
        ------
        ValueError
            Initial mesh has not been set, so the initial error cannot be
            computed
        ValueError
            Ground truth mesh has not been set, so the initial error cannot be
            computed
        """
        if compute_error is None:
            compute_error = error_function
        if self.initial_mesh is None:
            raise ValueError('Initial shape has not been set, so the initial '
                             'error cannot be computed')
        elif self.gt_mesh is None:
            raise ValueError('Ground truth shape has not been set, so the '
                             'initial error cannot be computed')
        else:
            return compute_error(self.initial_mesh, self.gt_mesh)

    def rasterized_final_mesh(self, shape=None):
        r"""
        Returns the rasterized final mesh. The image's shape will be used in
        case `shape` is not provided.

        Parameters
        ----------
        shape: `(int, int)` or ``None``, optional
            The shape of the rasterized image. If ``None``, then the fitted
            image's shape will be used.

        Returns
        -------
        rasterized_final_mesh : `menpo.image.Image`
            The image with the rasterized final mesh.

        Raises
        ------
        ValueError
            The final camera transform does not exist.
        ValueError
            You need to provide an image shape, since the image does not exist.
        """
        if self.final_camera_transform is None:
            raise ValueError("The final camera transform does not exist.")
        if shape is None:
            if self.image is None:
                raise ValueError("You need to provide an image shape, "
                                 "since the image does not exist.")
            else:
                shape = self.image.shape
        return rasterize_mesh(
            self.final_camera_transform.apply(self.final_mesh), shape)

    def rasterized_initial_mesh(self, shape=None):
        r"""
        Returns the rasterized initial mesh, if it exists. The image's shape
        will be used in case `shape` is not provided.

        Parameters
        ----------
        shape: `(int, int)` or ``None``, optional
            The shape of the rasterized image. If ``None``, then the fitted
            image's shape will be used.

        Returns
        -------
        rasterized_initial_mesh : `menpo.image.Image`
            The image with the rasterized initial mesh.

        Raises
        ------
        ValueError
            The initial mesh does not exist.
        ValueError
            The initial camera transform does not exist.
        ValueError
            You need to provide an image shape, since the image does not exist.
        """
        if self.initial_mesh is None:
            raise ValueError("The initial mesh does not exist.")
        if self.initial_camera_transform is None:
            raise ValueError("The initial camera transform does not exist.")
        if shape is None:
            if self.image is None:
                raise ValueError("You need to provide an image shape, "
                                 "since the image does not exist.")
            else:
                shape = self.image.shape
        return rasterize_mesh(
            self.initial_camera_transform.apply(self.initial_mesh), shape)

    def view_final_mesh(self, figure_id=None, new_figure=False,
                        textured=True, mesh_type='surface',
                        mesh_colour=(1, 0, 0), line_width=2,
                        ambient_light=0.0, specular_light=0.0, step=None,
                        alpha=1.0):
        """
        Visualize the final mesh.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        textured : `bool`, optional
            If ``True``, then the texture is rendered. If ``False``, then only
            the TriMesh is rendered with the specified `colour`.
        mesh_type : ``{'surface', 'wireframe'}``, optional
            The representation type to be used for the mesh.
        mesh_colour : `(float, float, float)`, optional
            The colour of the mesh as a tuple of RGB values. It only applies if
            `textured` is ``False``.
        line_width : `float`, optional
            The width of the lines, if there are any.
        ambient_light : `float`, optional
            The ambient light intensity. It must be in range ``[0., 1.]``.
        specular_light : `float`, optional
            The specular light intensity. It must be in range ``[0., 1.]``.
        step : `int` or ``None``, optional
            If `int`, then one every `step` normals will be rendered.
            If ``None``, then all vertexes will be rendered. It only applies if
            `normals` is not ``None``.
        alpha : `float`, optional
            Defines the transparency (opacity) of the object.

        Returns
        -------
        renderer : `menpo3d.visualize.TexturedTriMeshViewer3D`
            The Menpo3D rendering object.
        """
        return self.final_mesh.view(
            figure_id=figure_id, new_figure=new_figure, textured=textured,
            mesh_type=mesh_type, mesh_colour=mesh_colour,
            line_width=line_width, ambient_light=ambient_light,
            specular_light=specular_light, step=step, alpha=alpha)

    def view_initial_mesh(self, figure_id=None, new_figure=False,
                          textured=True, mesh_type='surface',
                          mesh_colour=(1, 0, 0), line_width=2,
                          ambient_light=0.0, specular_light=0.0, step=None,
                          alpha=1.0):
        """
        Visualize the initial mesh, if it exists.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        textured : `bool`, optional
            If ``True``, then the texture is rendered. If ``False``, then only
            the TriMesh is rendered with the specified `colour`.
        mesh_type : ``{'surface', 'wireframe'}``, optional
            The representation type to be used for the mesh.
        mesh_colour : `(float, float, float)`, optional
            The colour of the mesh as a tuple of RGB values. It only applies if
            `textured` is ``False``.
        line_width : `float`, optional
            The width of the lines, if there are any.
        ambient_light : `float`, optional
            The ambient light intensity. It must be in range ``[0., 1.]``.
        specular_light : `float`, optional
            The specular light intensity. It must be in range ``[0., 1.]``.
        step : `int` or ``None``, optional
            If `int`, then one every `step` normals will be rendered.
            If ``None``, then all vertexes will be rendered. It only applies if
            `normals` is not ``None``.
        alpha : `float`, optional
            Defines the transparency (opacity) of the object.

        Returns
        -------
        renderer : `menpo3d.visualize.TexturedTriMeshViewer3D`
            The Menpo3D rendering object.
        """
        if self.initial_mesh is None:
            raise ValueError("The initial mesh does not exist.")
        else:
            return self.initial_mesh.view(
                figure_id=figure_id, new_figure=new_figure, textured=textured,
                mesh_type=mesh_type, mesh_colour=mesh_colour,
                line_width=line_width, ambient_light=ambient_light,
                specular_light=specular_light, step=step, alpha=alpha)

    def view_gt_mesh(self, figure_id=None, new_figure=False,
                     mesh_type='wireframe', line_width=2, colour=(1, 0, 0),
                     marker_style='sphere', marker_size=0.05,
                     marker_resolution=8, step=None, alpha=1.0):
        """
        Visualize the ground truth mesh, if it exists.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        mesh_type : `str`, optional
            The representation type to be used for the mesh.
            Example options ::

                {surface, wireframe, points, mesh, fancymesh}

        line_width : `float`, optional
            The width of the lines, if there are any.
        colour : `(float, float, float)`, optional
            The colour of the mesh as a tuple of RGB values.
        marker_style : `str`, optional
            The style of the markers.
            Example options ::

                {2darrow, 2dcircle, 2dcross, 2ddash, 2ddiamond, 2dhooked_arrow,
                 2dsquare, 2dthick_arrow, 2dthick_cross, 2dtriangle, 2dvertex,
                 arrow, axes, cone, cube, cylinder, point, sphere}

        marker_size : `float`, optional
            The size of the markers. This size can be seen as a scale factor
            applied to the size markers, which is by default calculated from
            the inter-marker spacing. It only applies for the 'fancymesh'.
        marker_resolution : `int`, optional
            The resolution of the markers. For spheres, for instance, this is
            the number of divisions along theta and phi. It only applies for
            the 'fancymesh'.
        step : `int` or ``None``, optional
            If `int`, then one every `step` markers will be rendered.
            If ``None``, then all vertexes will be rendered. It only applies for
            the 'fancymesh' and if `normals` is not ``None``.
        alpha : `float`, optional
            Defines the transparency (opacity) of the object.

        Returns
        -------
        renderer : `menpo3d.visualize.TriMeshViewer3D`
            The Menpo3D rendering object.
        """
        if self.initial_mesh is None:
            raise ValueError("The ground truth mesh does not exist.")
        else:
            return self.gt_mesh.view(
                figure_id=figure_id, new_figure=new_figure, mesh_type=mesh_type,
                line_width=line_width, colour=colour, marker_style=marker_style,
                marker_size=marker_size, marker_resolution=marker_resolution,
                step=step, alpha=alpha)

    def __str__(self):
        out = "Fitting result of mesh with {} points.".format(
            self.final_mesh.n_points)
        if self.gt_mesh is not None:
            if self.initial_mesh is not None:
                out += "\nInitial error: {:.4f}".format(self.initial_error())
            out += "\nFinal error: {:.4f}".format(self.final_error())
        return out


class NonParametricIterativeResult(Result):
    r"""
    Class for defining a non-parametric iterative fitting result, i.e. the
    result of a method that does not optimize over a parametric shape model.
    It holds the meshes of all the iterations of the fitting procedure.
    It can optionally store the image on which the fitting was applied,
    as well as its ground truth mesh.

    Parameters
    ----------
    meshes : `list` of `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh`
        The `list` of meshes per iteration. Note that the list does not
        include the initial mesh. However, it includes the reconstruction of
        the initial mesh. The last member of the list is the final mesh.
    camera_transforms : `list` of `menpo3d.camera.PerspectiveCamera`
        The `list` of camera transform objects per iteration. Note that the
        list does not include the initial camera transform. The last member
        of the list is the final camera transform.
    image : `menpo.image.Image` or `subclass` or ``None``, optional
        The image on which the fitting process was applied. Note that a copy
        of the image will be assigned as an attribute. If ``None``, then no
        image is assigned.
    initial_mesh : `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh` or ``None``, optional
        The initial mesh from which the fitting process started. If
        ``None``, then no initial mesh is assigned.
    initial_camera_transform : `menpo3d.camera.PerspectiveCamera` or ``None``, optional
        The initial camera transform. If ``None``, then no initial camera
        is assigned.
    gt_mesh : `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh` or ``None``, optional
        The ground truth mesh associated with the image. If ``None``, then
        no ground truth mesh is assigned.
    costs : `list` of `float` or ``None``, optional
        The `list` of cost per iteration. If ``None``, then it is assumed
        that the cost function cannot be computed for the specific
        algorithm. It must have the same length as `meshes`.
    """
    def __init__(self, meshes, camera_transforms, image=None, initial_mesh=None,
                 initial_camera_transform=None, gt_mesh=None, costs=None):
        super(NonParametricIterativeResult, self).__init__(
            final_mesh=meshes[-1], final_camera_transform=camera_transforms[-1],
            image=image, initial_mesh=initial_mesh,
            initial_camera_transform=initial_camera_transform, gt_mesh=gt_mesh)
        self._n_iters = len(meshes)
        # If initial mesh is provided, then add it in the beginning of meshes
        self._meshes = meshes
        if self.initial_mesh is not None:
            self._meshes = [self.initial_mesh] + self._meshes
        # If initial camera transform is provided, then add it in the beginning
        # of meshes
        self._camera_transforms = camera_transforms
        if self.initial_camera_transform is not None:
            self._camera_transforms = ([self.initial_camera_transform] +
                                       self._camera_transforms)
        # Add costs as property
        self._costs = costs

    @property
    def is_iterative(self):
        r"""
        Flag whether the object is an iterative fitting result.

        :type: `bool`
        """
        return True

    @property
    def meshes(self):
        r"""
        Returns the `list` of meshes obtained at each iteration of the fitting
        process. The `list` includes the `initial_mesh` (if it exists) and
        `final_mesh`.

        :type: `list` of `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh`
        """
        return self._meshes

    @property
    def camera_transforms(self):
        r"""
        Returns the `list` of camera transforms per iteration of the fitting
        process. The `list` includes the `initial_camera_transform` (if it
        exists) and the camera transform of the final mesh.

        :type: `list` of `menpo3d.camera.PerspectiveCamera`
        """
        return self._camera_transforms

    @property
    def n_iters(self):
        r"""
        Returns the total number of iterations of the fitting process.

        :type: `int`
        """
        return self._n_iters

    def to_result(self, pass_image=True, pass_initial_mesh=True,
                  pass_gt_mesh=True):
        r"""
        Returns a :map:`Result` instance of the object, i.e. a fitting result
        object that does not store the iterations. This can be useful for
        reducing the size of saved fitting results.

        Parameters
        ----------
        pass_image : `bool`, optional
            If ``True``, then the image will get passed (if it exists).
        pass_initial_mesh : `bool`, optional
            If ``True``, then the initial mesh will get passed (if it exists).
        pass_gt_mesh : `bool`, optional
            If ``True``, then the ground truth mesh will get passed (if it
            exists).

        Returns
        -------
        result : :map:`Result`
            The final "lightweight" fitting result.
        """
        image = None
        if pass_image:
            image = self.image
        initial_mesh = None
        initial_camera_transform = None
        if pass_initial_mesh:
            initial_mesh = self.initial_mesh
            initial_camera_transform = self.initial_camera_transform
        gt_mesh = None
        if pass_gt_mesh:
            gt_mesh = self.gt_mesh
        return Result(self.final_mesh, self.final_camera_transform,
                      image=image, initial_mesh=initial_mesh,
                      initial_camera_transform=initial_camera_transform,
                      gt_mesh=gt_mesh)

    def errors(self, compute_error=None):
        r"""
        Returns a list containing the error at each fitting iteration, if the
        ground truth mesh exists.

        Parameters
        ----------
        compute_error: `callable`, optional
            Callable that computes the error between the mesh at each
            iteration and the ground truth mesh.

        Returns
        -------
        errors : `list` of `float`
            The error at each iteration of the fitting process.

        Raises
        ------
        ValueError
            Ground truth mesh has not been set, so the final error cannot be
            computed
        """
        if compute_error is None:
            compute_error = error_function
        if self.gt_mesh is not None:
            return [compute_error(t, self.gt_mesh)
                    for t in self.meshes]
        else:
            raise ValueError('Ground truth shape has not been set, so the '
                             'errors per iteration cannot be computed')

    def plot_errors(self, compute_error=None, figure_id=None,
                    new_figure=False, render_lines=True, line_colour='b',
                    line_style='-', line_width=2, render_markers=True,
                    marker_style='o', marker_size=4, marker_face_colour='b',
                    marker_edge_colour='k', marker_edge_width=1.,
                    render_axes=True, axes_font_name='sans-serif',
                    axes_font_size=10, axes_font_style='normal',
                    axes_font_weight='normal', axes_x_limits=0.,
                    axes_y_limits=None, axes_x_ticks=None,
                    axes_y_ticks=None, figure_size=(10, 6),
                    render_grid=True, grid_line_style='--',
                    grid_line_width=0.5):
        r"""
        Plot of the error evolution at each fitting iteration.

        Parameters
        ----------
        compute_error: `callable`, optional
            Callable that computes the error between the shape at each
            iteration and the ground truth shape.
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        render_lines : `bool`, optional
            If ``True``, the line will be rendered.
        line_colour : `colour` or ``None`` (See below), optional
            The colour of the line. If ``None``, the colour is sampled from
            the jet colormap.
            Example `colour` options are ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        line_style : `str` (See below), optional
            The style of the lines. Example options::

                {-, --, -., :}

        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : `str` (See below), optional
            The style of the markers.
            Example `marker` options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int`, optional
            The size of the markers in points.
        marker_face_colour : `colour` or ``None``, optional
            The face (filling) colour of the markers. If ``None``, the colour
            is sampled from the jet colormap.
            Example `colour` options are ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_colour : `colour` or ``None``, optional
            The edge colour of the markers. If ``None``, the colour
            is sampled from the jet colormap.
            Example `colour` options are ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : `str` (See below), optional
            The font of the axes.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : `str` (See below), optional
            The font style of the axes.
            Example options ::

                {normal, italic, oblique}

        axes_font_weight : `str` (See below), optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the graph as a percentage of the curves' width. If
            `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_y_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the graph as a percentage of the curves' height.
            If `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) or ``None``, optional
            The size of the figure in inches.
        render_grid : `bool`, optional
            If ``True``, the grid will be rendered.
        grid_line_style : ``{'-', '--', '-.', ':'}``, optional
            The style of the grid lines.
        grid_line_width : `float`, optional
            The width of the grid lines.

        Returns
        -------
        renderer : `menpo.visualize.GraphPlotter`
            The renderer object.
        """
        from menpo.visualize import plot_curve
        errors = self.errors(compute_error=compute_error)
        return plot_curve(
            x_axis=list(range(len(errors))), y_axis=[errors], figure_id=figure_id,
            new_figure=new_figure, title='Fitting Errors per Iteration',
            x_label='Iteration', y_label='Fitting Error',
            axes_x_limits=axes_x_limits, axes_y_limits=axes_y_limits,
            axes_x_ticks=axes_x_ticks, axes_y_ticks=axes_y_ticks,
            render_lines=render_lines, line_colour=line_colour,
            line_style=line_style, line_width=line_width,
            render_markers=render_markers, marker_style=marker_style,
            marker_size=marker_size, marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width, render_legend=False,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, figure_size=figure_size,
            render_grid=render_grid,  grid_line_style=grid_line_style,
            grid_line_width=grid_line_width)

    def displacements(self):
        r"""
        A list containing the displacement between the mesh of each iteration
        and the mesh of the previous one.

        :type: `list` of `ndarray`
        """
        return [np.linalg.norm(s1.points - s2.points, axis=1)
                for s1, s2 in zip(self.meshes, self.meshes[1:])]

    def displacements_stats(self, stat_type='mean'):
        r"""
        A list containing a statistical metric on the displacements between
        the mesh of each iteration and the mesh of the previous one.

        Parameters
        ----------
        stat_type : ``{'mean', 'median', 'min', 'max'}``, optional
            Specifies a statistic metric to be extracted from the displacements.

        Returns
        -------
        displacements_stat : `list` of `float`
            The statistical metric on the points displacements for each
            iteration.

        Raises
        ------
        ValueError
            type must be 'mean', 'median', 'min' or 'max'
        """
        if stat_type == 'mean':
            return [np.mean(d) for d in self.displacements()]
        elif stat_type == 'median':
            return [np.median(d) for d in self.displacements()]
        elif stat_type == 'max':
            return [np.max(d) for d in self.displacements()]
        elif stat_type == 'min':
            return [np.min(d) for d in self.displacements()]
        else:
            raise ValueError("type must be 'mean', 'median', 'min' or 'max'")

    def plot_displacements(self, stat_type='mean', figure_id=None,
                           new_figure=False, render_lines=True, line_colour='b',
                           line_style='-', line_width=2, render_markers=True,
                           marker_style='o', marker_size=4,
                           marker_face_colour='b', marker_edge_colour='k',
                           marker_edge_width=1., render_axes=True,
                           axes_font_name='sans-serif', axes_font_size=10,
                           axes_font_style='normal', axes_font_weight='normal',
                           axes_x_limits=0., axes_y_limits=None,
                           axes_x_ticks=None, axes_y_ticks=None,
                           figure_size=(10, 6), render_grid=True,
                           grid_line_style='--', grid_line_width=0.5):
        r"""
        Plot of a statistical metric of the displacement between the mesh of
        each iteration and the mesh of the previous one.

        Parameters
        ----------
        stat_type : {``mean``, ``median``, ``min``, ``max``}, optional
            Specifies a statistic metric to be extracted from the displacements
            (see also `displacements_stats()` method).
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        render_lines : `bool`, optional
            If ``True``, the line will be rendered.
        line_colour : `colour` or ``None`` (See below), optional
            The colour of the line. If ``None``, the colour is sampled from
            the jet colormap.
            Example `colour` options are ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        line_style : `str` (See below), optional
            The style of the lines. Example options::

                {-, --, -., :}

        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : `str` (See below), optional
            The style of the markers.
            Example `marker` options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int`, optional
            The size of the markers in points.
        marker_face_colour : `colour` or ``None``, optional
            The face (filling) colour of the markers. If ``None``, the colour
            is sampled from the jet colormap.
            Example `colour` options are ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_colour : `colour` or ``None``, optional
            The edge colour of the markers. If ``None``, the colour
            is sampled from the jet colormap.
            Example `colour` options are ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : `str` (See below), optional
            The font of the axes.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : `str` (See below), optional
            The font style of the axes.
            Example options ::

                {normal, italic, oblique}

        axes_font_weight : `str` (See below), optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the graph as a percentage of the curves' width. If
            `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_y_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the graph as a percentage of the curves' height.
            If `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) or ``None``, optional
            The size of the figure in inches.
        render_grid : `bool`, optional
            If ``True``, the grid will be rendered.
        grid_line_style : ``{'-', '--', '-.', ':'}``, optional
            The style of the grid lines.
        grid_line_width : `float`, optional
            The width of the grid lines.

        Returns
        -------
        renderer : `menpo.visualize.GraphPlotter`
            The renderer object.
        """
        from menpo.visualize import plot_curve
        # set labels
        if stat_type == 'max':
            name = 'Maximum'
        elif stat_type == 'min':
            name = 'Minimum'
        elif stat_type == 'mean':
            name = 'Mean'
        elif stat_type == 'median':
            name = 'Median'
        else:
            raise ValueError('stat_type must be one of {max, min, mean, '
                             'median}.')
        y_label = '{} Displacement'.format(name)
        title = '{} displacement per Iteration'.format(name)

        # plot
        displacements = self.displacements_stats(stat_type=stat_type)
        return plot_curve(
            x_axis=list(range(len(displacements))), y_axis=[displacements],
            figure_id=figure_id, new_figure=new_figure, title=title,
            x_label='Iteration', y_label=y_label,
            axes_x_limits=axes_x_limits, axes_y_limits=axes_y_limits,
            axes_x_ticks=axes_x_ticks, axes_y_ticks=axes_y_ticks,
            render_lines=render_lines, line_colour=line_colour,
            line_style=line_style, line_width=line_width,
            render_markers=render_markers, marker_style=marker_style,
            marker_size=marker_size, marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width, render_legend=False,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, figure_size=figure_size,
            render_grid=render_grid,  grid_line_style=grid_line_style,
            grid_line_width=grid_line_width)

    @property
    def costs(self):
        r"""
        Returns a `list` with the cost per iteration. It returns ``None`` if
        the costs are not computed.

        :type: `list` of `float` or ``None``
        """
        return self._costs

    def plot_costs(self, figure_id=None, new_figure=False, render_lines=True,
                   line_colour='b', line_style='-', line_width=2,
                   render_markers=True, marker_style='o', marker_size=4,
                   marker_face_colour='b', marker_edge_colour='k',
                   marker_edge_width=1., render_axes=True,
                   axes_font_name='sans-serif', axes_font_size=10,
                   axes_font_style='normal', axes_font_weight='normal',
                   axes_x_limits=0., axes_y_limits=None, axes_x_ticks=None,
                   axes_y_ticks=None, figure_size=(10, 6),
                   render_grid=True, grid_line_style='--',
                   grid_line_width=0.5):
        r"""
        Plot of the cost function evolution at each fitting iteration.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        render_lines : `bool`, optional
            If ``True``, the line will be rendered.
        line_colour : `colour` or ``None``, optional
            The colour of the line. If ``None``, the colour is sampled from
            the jet colormap.
            Example `colour` options are ::

                    {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                    or
                    (3, ) ndarray

        line_style : ``{'-', '--', '-.', ':'}``, optional
            The style of the lines.
        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : `marker`, optional
            The style of the markers.
            Example `marker` options ::

                    {'.', ',', 'o', 'v', '^', '<', '>', '+', 'x', 'D', 'd', 's',
                     'p', '*', 'h', 'H', '1', '2', '3', '4', '8'}

        marker_size : `int`, optional
            The size of the markers in points.
        marker_face_colour : `colour` or ``None``, optional
            The face (filling) colour of the markers. If ``None``, the colour
            is sampled from the jet colormap.
            Example `colour` options are ::

                    {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                    or
                    (3, ) ndarray

        marker_edge_colour : `colour` or ``None``, optional
            The edge colour of the markers.If ``None``, the colour
            is sampled from the jet colormap.
            Example `colour` options are ::

                    {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                    or
                    (3, ) ndarray

        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See below, optional
            The font of the axes.
            Example options ::

                {'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : ``{'normal', 'italic', 'oblique'}``, optional
            The font style of the axes.
        axes_font_weight : See below, optional
            The font weight of the axes.
            Example options ::

                {'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
                 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
                 'extra bold', 'black'}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the graph as a percentage of the curves' width. If
            `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_y_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the graph as a percentage of the curves' height.
            If `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) or ``None``, optional
            The size of the figure in inches.
        render_grid : `bool`, optional
            If ``True``, the grid will be rendered.
        grid_line_style : ``{'-', '--', '-.', ':'}``, optional
            The style of the grid lines.
        grid_line_width : `float`, optional
            The width of the grid lines.

        Returns
        -------
        renderer : `menpo.visualize.GraphPlotter`
            The renderer object.
        """
        from menpo.visualize import plot_curve
        costs = self.costs
        if costs is not None:
            return plot_curve(
                x_axis=list(range(len(costs))), y_axis=[costs],
                figure_id=figure_id, new_figure=new_figure,
                title='Cost per Iteration', x_label='Iteration',
                y_label='Cost Function', axes_x_limits=axes_x_limits,
                axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
                axes_y_ticks=axes_y_ticks, render_lines=render_lines,
                line_colour=line_colour, line_style=line_style,
                line_width=line_width, render_markers=render_markers,
                marker_style=marker_style, marker_size=marker_size,
                marker_face_colour=marker_face_colour,
                marker_edge_colour=marker_edge_colour,
                marker_edge_width=marker_edge_width, render_legend=False,
                render_axes=render_axes, axes_font_name=axes_font_name,
                axes_font_size=axes_font_size,
                axes_font_style=axes_font_style,
                axes_font_weight=axes_font_weight, figure_size=figure_size,
                render_grid=render_grid,  grid_line_style=grid_line_style,
                grid_line_width=grid_line_width)
        else:
            raise ValueError('costs are either not returned or not well '
                             'defined for the selected fitting algorithm')


class ParametricIterativeResult(NonParametricIterativeResult):
    r"""
    Class for defining a parametric iterative fitting result, i.e. the
    result of a method that optimizes the parameters of a shape model. It holds
    the shape parameters and camera transforms of all the iterations of the
    fitting procedure. It can optionally store the meshes, image on which the
    fitting was applied, as well as its ground truth mesh.

    .. note:: When using a method with a parametric shape model, the first step
              is to **reconstruct the initial mesh** using the shape model. The
              generated reconstructed mesh is then used as initialisation for
              the iterative optimisation. This step is not counted in the number
              of iterations.

    Parameters
    ----------
    shape_parameters : `list` of `ndarray`
        The `list` of shape parameters per iteration. Note that the list
        includes the parameters of the projection of the initial mesh. The last
        member of the list corresponds to the final mesh's parameters. It must
        have the same length as `camera_transforms` and `meshes`.
    meshes : `list` of `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh`
        The `list` of meshes per iteration. Note that the list does not
        include the initial mesh. However, it includes the reconstruction of
        the initial mesh. The last member of the list is the final mesh.
    camera_transforms : `list` of `menpo3d.camera.PerspectiveCamera`
        The `list` of camera transform objects per iteration. Note that the
        list does not include the initial camera transform. The last member
        of the list is the final camera transform.
    image : `menpo.image.Image` or `subclass` or ``None``, optional
        The image on which the fitting process was applied. Note that a copy
        of the image will be assigned as an attribute. If ``None``, then no
        image is assigned.
    initial_mesh : `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh` or ``None``, optional
        The initial mesh from which the fitting process started. If
        ``None``, then no initial mesh is assigned.
    initial_camera_transform : `menpo3d.camera.PerspectiveCamera` or ``None``, optional
        The initial camera transform. If ``None``, then no initial camera
        is assigned.
    gt_mesh : `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh` or ``None``, optional
        The ground truth mesh associated with the image. If ``None``, then
        no ground truth mesh is assigned.
    costs : `list` of `float` or ``None``, optional
        The `list` of cost per iteration. If ``None``, then it is assumed
        that the cost function cannot be computed for the specific
        algorithm. It must have the same length as `meshes`.
    """
    def __init__(self, shape_parameters, meshes, camera_transforms,
                 image=None, initial_mesh=None,
                 initial_camera_transform=None, gt_mesh=None, costs=None):
        # Assign shape parameters
        self._shape_parameters = shape_parameters
        # Get reconstructed initial shape
        self._reconstructed_initial_mesh = meshes[0]
        # Call superclass
        super(ParametricIterativeResult, self).__init__(
            meshes=meshes, camera_transforms=camera_transforms, image=image,
            initial_mesh=initial_mesh,
            initial_camera_transform=initial_camera_transform,
            gt_mesh=gt_mesh, costs=costs)
        # Correct n_iters. The initial mesh's reconstruction should not count
        # in the number of iterations.
        self._n_iters -= 1

    @property
    def meshes(self):
        r"""
        Returns the `list` of meshes obtained at each iteration of the fitting
        process. The `list` includes the `initial_mesh` (if it exists),
        `reconstructed_initial_mesh` and `final_mesh`.

        :type: `list` of `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh`
        """
        return self._meshes

    @property
    def camera_transforms(self):
        r"""
        Returns the `list` of camera transforms per iteration of the fitting
        process. The `list` includes the `initial_camera_transform` (if it
        exists) and the camera transform of the reconstructed initial mesh and
        final mesh.

        :type: `list` of `menpo3d.camera.PerspectiveCamera`
        """
        return self._camera_transforms

    @property
    def shape_parameters(self):
        r"""
        Returns the `list` of shape parameters obtained at each iteration of
        the fitting process. The `list` includes the parameters of the
        `reconstructed_initial_mesh` and `final_mesh`.

        :type: `list` of ``(n_params,)`` `ndarray`
        """
        return self._shape_parameters

    @property
    def reconstructed_initial_mesh(self):
        r"""
        Returns the initial mesh's reconstruction with the shape model that was
        used to initialise the iterative optimisation process.

        :type: `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh`
        """
        if self.initial_mesh is not None:
            return self.meshes[1]
        else:
            return self.meshes[0]

    @property
    def reconstructed_initial_camera_transform(self):
        r"""
        Returns the camera transform of the reconstructed initial mesh.

        :type: `menpo3d.camera.PerspectiveCamera`
        """
        if self.initial_camera_transform is not None:
            return self.camera_transforms[1]
        else:
            return self.camera_transforms[0]

    def rasterized_reconstructed_initial_mesh(self, shape=None):
        r"""
        Returns the rasterized reconstructed initial mesh, if it exists. The
        image's shape will be used in case `shape` is not provided.

        Parameters
        ----------
        shape: `(int, int)` or ``None``, optional
            The shape of the rasterized image. If ``None``, then the fitted
            image's shape will be used.

        Returns
        -------
        rasterized_reconstructed_initial_mesh : `menpo.image.Image`
            The image with the rasterized reconstructed initial mesh.

        Raises
        ------
        ValueError
            You need to provide an image shape, since the image does not exist.
        """
        if shape is None:
            if self.image is None:
                raise ValueError("You need to provide an image shape, "
                                 "since the image does not exist.")
            else:
                shape = self.image.shape
        return rasterize_mesh(
            self.reconstructed_initial_camera_transform.apply(
                self.reconstructed_initial_mesh), shape)

    @property
    def _reconstruction_indices(self):
        r"""
        Returns a list with the indices of reconstructed meshes in the `meshes`
        list.

        :type: `list` of `int`
        """
        if self.initial_mesh is not None:
            return [1]
        else:
            return [0]

    def reconstructed_initial_error(self, compute_error=None):
        r"""
        Returns the error of the reconstructed initial mesh of the fitting
        process, if the ground truth mesh exists. This is the error computed
        based on the `reconstructed_initial_mesh`.

        Parameters
        ----------
        compute_error: `callable`, optional
            Callable that computes the error between the reconstructed initial
            and ground truth meshes.

        Returns
        -------
        reconstructed_initial_error : `float`
            The error that corresponds to the initial mesh's reconstruction.

        Raises
        ------
        ValueError
            Ground truth mesh has not been set, so the reconstructed initial
            error cannot be computed
        """
        if compute_error is None:
            compute_error = error_function
        if self.gt_mesh is None:
            raise ValueError('Ground truth mesh has not been set, so the '
                             'reconstructed initial error cannot be computed')
        else:
            return compute_error(self.reconstructed_initial_mesh, self.gt_mesh)


def _affine_2d_to_3d(transform):
    h_matrix = np.eye(4)
    h_matrix[:2, :2] = transform.h_matrix[:2, :2]
    h_matrix[:2, 3] = transform.h_matrix[:2, 2]
    h_matrix[3, :2] = transform.h_matrix[2, :2]
    return Homogeneous(h_matrix)


class MultiScaleNonParametricIterativeResult(NonParametricIterativeResult):
    r"""
    Class for defining a multi-scale non-parametric iterative fitting result,
    i.e. the result of a multi-scale method that does not optimise over a
    parametric shape model. It holds the meshes of all the iterations of
    the fitting procedure, as well as the scales. It can optionally store the
    image on which the fitting was applied, as well as its ground truth mesh.

    Parameters
    ----------
    results : `list` of :map:`NonParametricIterativeResult`
        The `list` of non parametric iterative results per scale.
    n_scales : `int`
        The number of scales.
    affine_transforms : `list` of `menpo.transform.Affine`
        The list of affine transforms per scale that transform where used
        dutring the image's pre-processing.
    image : `menpo.image.Image` or `subclass` or ``None``, optional
        The image on which the fitting process was applied. Note that a copy
        of the image will be assigned as an attribute. If ``None``, then no
        image is assigned.
    gt_mesh : `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh` or ``None``, optional
        The ground truth mesh associated with the image. If ``None``, then no
        ground truth mesh is assigned.
    model_landmarks_index : `list` or ``None``, optional
        It `list`, then it is supposed to provide indices for vertices of the
        model that have some kind of semantic meaning. These points will be
        used in order to generate 2D pointclouds projected in the image plane.
        If ``None``, then the 2D pointclouds will not be generated.
    """
    def __init__(self, results, affine_transforms, n_scales, image=None,
                 gt_mesh=None, model_landmarks_index=None):
        # Make sure results are iterable with the correct length
        if not isinstance(results, Iterable):
            results = [results]
        if len(results) != n_scales:
            raise ValueError("The provided results and n_scales do not match.")
        # Make sure affine_transforms are iterable with the correct length
        if not isinstance(affine_transforms, Iterable):
            affine_transforms = [affine_transforms]
        if len(affine_transforms) != n_scales:
            raise ValueError("The provided affine_transforms and n_scales do "
                             "not match.")
        # Get initial mesh and initial camera transform
        initial_mesh = None
        if results[0].initial_mesh is not None:
            initial_mesh = results[0].initial_mesh
        initial_camera_transform = None
        if results[0].initial_camera_transform is not None:
            initial_camera_transform = results[0].initial_camera_transform
        # Create meshes list and n_iters_per_scale
        # If the result object has an initial shape, then it has to be
        # removed from the final meshes list
        n_iters_per_scale = []
        meshes = []
        camera_transforms = []
        self._affine_transforms = []
        for i in list(range(n_scales)):
            n_iters_per_scale.append(results[i].n_iters)
            if results[i].initial_mesh is None:
                meshes += results[i].meshes
            else:
                meshes += results[i].meshes[1:]
            if results[i].initial_camera_transform is None:
                camera_transforms += results[i].camera_transforms
                self._affine_transforms += (
                    [_affine_2d_to_3d(affine_transforms[i])] *
                    len(results[i].camera_transforms))
            else:
                camera_transforms += results[i].camera_transforms[1:]
                self._affine_transforms += (
                    [_affine_2d_to_3d(affine_transforms[i])] *
                    len(results[i].camera_transforms[1:]))
        # Call superclass
        super(MultiScaleNonParametricIterativeResult, self).__init__(
            meshes=meshes, camera_transforms=camera_transforms, image=image,
            initial_mesh=initial_mesh,
            initial_camera_transform=initial_camera_transform, gt_mesh=gt_mesh)
        # Get attributes
        self._n_iters_per_scale = n_iters_per_scale
        self._n_scales = n_scales
        # Correct affine transforms, if neccessary
        if self.initial_camera_transform is not None:
            self._affine_transforms = ([self._affine_transforms[0]] +
                                       self._affine_transforms)
        # Create costs list. We assume that if the costs of the first result
        # object is None, then the costs property of all objects is None.
        # Similarly, if the costs property of the the first object is not
        # None, then the same stands for the rest.
        self._costs = None
        if results[0].costs is not None:
            self._costs = []
            for r in results:
                self._costs += r.costs
        # Assing model_landmarks_index
        self._model_landmarks_index = model_landmarks_index

    @property
    def n_iters_per_scale(self):
        r"""
        Returns the number of iterations per scale of the fitting process.

        :type: `list` of `int`
        """
        return self._n_iters_per_scale

    @property
    def n_scales(self):
        r"""
        Returns the number of scales used during the fitting process.

        :type: `int`
        """
        return self._n_scales

    def rasterized_final_mesh(self, shape=None):
        r"""
        Returns the rasterized final mesh. The image's shape will be used in
        case `shape` is not provided.

        Parameters
        ----------
        shape: `(int, int)` or ``None``, optional
            The shape of the rasterized image. If ``None``, then the fitted
            image's shape will be used.

        Returns
        -------
        rasterized_final_mesh : `menpo.image.Image`
            The image with the rasterized final mesh.

        Raises
        ------
        ValueError
            The final camera transform does not exist.
        ValueError
            You need to provide an image shape, since the image does not exist.
        """
        if self.final_camera_transform is None:
            raise ValueError("The final camera transform does not exist.")
        if shape is None:
            if self.image is None:
                raise ValueError("You need to provide an image shape, "
                                 "since the image does not exist.")
            else:
                shape = self.image.shape
        instance_in_img = self._affine_transforms[-1].apply(
            self.final_camera_transform.apply(self.final_mesh))
        return rasterize_mesh(instance_in_img, shape)

    def rasterized_initial_mesh(self, shape=None):
        r"""
        Returns the rasterized initial mesh, if it exists. The image's shape
        will be used in case `shape` is not provided.

        Parameters
        ----------
        shape: `(int, int)` or ``None``, optional
            The shape of the rasterized image. If ``None``, then the fitted
            image's shape will be used.

        Returns
        -------
        rasterized_initial_mesh : `menpo.image.Image`
            The image with the rasterized initial mesh.

        Raises
        ------
        ValueError
            The initial mesh does not exist.
        ValueError
            The initial camera transform does not exist.
        ValueError
            You need to provide an image shape, since the image does not exist.
        """
        if self.initial_mesh is None:
            raise ValueError("The initial mesh does not exist.")
        if self.initial_camera_transform is None:
            raise ValueError("The initial camera transform does not exist.")
        if shape is None:
            if self.image is None:
                raise ValueError("You need to provide an image shape, "
                                 "since the image does not exist.")
            else:
                shape = self.image.shape
        instance_in_img = self._affine_transforms[0].apply(
            self.initial_camera_transform.apply(self.initial_mesh))
        return rasterize_mesh(instance_in_img, shape)

    def rasterized_meshes(self, shape=None):
        r"""
        Returns the rasterized meshes. The image's shape will be used in
        case `shape` is not provided.

        Parameters
        ----------
        shape: `(int, int)` or ``None``, optional
            The shape of the rasterized image. If ``None``, then the fitted
            image's shape will be used.

        Returns
        -------
        rasterized_meshes : `list` of `menpo.image.Image`
            The list of images with the rasterized meshes.

        Raises
        ------
        ValueError
            You need to provide an image shape, since the image does not exist.
        """
        if shape is None:
            if self.image is None:
                raise ValueError("You need to provide an image shape, "
                                 "since the image does not exist.")
            else:
                shape = self.image.shape

        def rast(x):
            affine, camera, mesh = x
            instance_in_img = affine.compose_after(
                camera.camera_transform).apply(mesh)
            return rasterize_mesh(instance_in_img, shape)

        xs = list(zip(self._affine_transforms, self.camera_transforms,
                      self.meshes))
        return LazyList.init_from_iterable(xs, f=rast)

    def sparse_final_mesh_projected_in_2d(self):
        r"""
        Returns the sparse final mesh projected in 2D.

        :type: `menpo.shape.PointCloud`
            The sparse final mesh projected in the image plane.

        Raises
        ------
        ValueError
            There is not a sparse landmarks index mapping.
        ValueError
            The final camera transform does not exist.
        """
        if self._model_landmarks_index is None:
            raise ValueError("There is not a sparse landmarks index mapping.")
        if self.final_camera_transform is None:
            raise ValueError("The final camera transform does not exist.")
        sparse_instance = PointCloud(
            self.final_mesh.points[self._model_landmarks_index])
        instance_in_img = self._affine_transforms[-1].apply(
            self.final_camera_transform.apply(sparse_instance))
        return PointCloud(instance_in_img.points[:, :2])

    def sparse_initial_mesh_projected_in_2d(self):
        r"""
        Returns the sparse initial mesh projected in 2D.

        :type: `menpo.shape.PointCloud`
            The sparse initial mesh projected in the image plane.

        Raises
        ------
        ValueError
            There is not a sparse landmarks index mapping.
        ValueError
            The initial camera transform does not exist.
        """
        if self._model_landmarks_index is None:
            raise ValueError("There is not a sparse landmarks index mapping.")
        if self.initial_mesh is None:
            raise ValueError("The initial mesh does not exist.")
        if self.initial_camera_transform is None:
            raise ValueError("The initial camera transform does not exist.")
        sparse_instance = PointCloud(
            self.initial_mesh.points[self._model_landmarks_index])
        instance_in_img = self._affine_transforms[-1].apply(
            self.final_camera_transform.apply(sparse_instance))
        return PointCloud(instance_in_img.points[:, :2])

    def sparse_meshes_projected_in_2d(self):
        r"""
        Returns the list of sparse meshes projected in 2D.

        :type: `list` of `menpo.shape.PointCloud`
            The list of sparse meshes projected in the image plane.
        """
        def project(x):
            mesh, affine, camera = x
            sparse_instance = PointCloud(
                mesh.points[self._model_landmarks_index])
            instance_in_img = affine.apply(camera.apply(sparse_instance))
            return PointCloud(instance_in_img.points[:, :2])

        xs = list(zip(self.meshes, self._affine_transforms,
                      self.camera_transforms))
        return LazyList.init_from_iterable(xs, f=project)


class MultiScaleParametricIterativeResult(MultiScaleNonParametricIterativeResult):
    r"""
    Class for defining a multi-scale parametric iterative fitting result, i.e.
    the result of a multi-scale method that optimizes over a parametric shape
    model. It holds the meshes of all the iterations of the fitting procedure,
    as well as the scales. It can optionally store the image on which the
    fitting was applied, as well as its ground truth mesh.

    .. note:: When using a method with a parametric shape model, the first step
              is to **reconstruct the initial mesh** using the shape model. The
              generated reconstructed mesh is then used as initialisation for
              the iterative optimisation. This step is not counted in the number
              of iterations.

    Parameters
    ----------
    results : `list` of :map:`ParametricIterativeResult`
        The `list` of non parametric iterative results per scale.
    n_scales : `int`
        The number of scales.
    affine_transforms : `list` of `menpo.transform.Affine`
        The list of affine transforms per scale that transform where used
        dutring the image's pre-processing.
    image : `menpo.image.Image` or `subclass` or ``None``, optional
        The image on which the fitting process was applied. Note that a copy
        of the image will be assigned as an attribute. If ``None``, then no
        image is assigned.
    gt_mesh : `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh` or ``None``, optional
        The ground truth mesh associated with the image. If ``None``, then no
        ground truth mesh is assigned.
    model_landmarks_index : `list` or ``None``, optional
        It `list`, then it is supposed to provide indices for vertices of the
        model that have some kind of semantic meaning. These points will be
        used in order to generate 2D pointclouds projected in the image plane.
        If ``None``, then the 2D pointclouds will not be generated.
    """
    def __init__(self, results, affine_transforms, n_scales, image=None,
                 gt_mesh=None, model_landmarks_index=None):
        # Call superclass
        super(MultiScaleParametricIterativeResult, self).__init__(
            results=results, affine_transforms=affine_transforms,
            n_scales=n_scales, image=image, gt_mesh=gt_mesh,
            model_landmarks_index=model_landmarks_index)
        # Create shape parameters
        self._shape_parameters = []
        for r in results:
            self._shape_parameters += r.shape_parameters
        # Correct n_iters
        self._n_iters -= n_scales

    @property
    def shape_parameters(self):
        r"""
        Returns the `list` of shape parameters obtained at each iteration of
        the fitting process. The `list` includes the parameters of the
        `initial_mesh` (if it exists) and `final_mesh`.

        :type: `list` of ``(n_params,)`` `ndarray`
        """
        return self._shape_parameters

    @property
    def reconstructed_initial_meshes(self):
        r"""
        Returns the result of the reconstruction step that takes place at each
        scale before applying the iterative optimisation.

        :type: `list` of `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh`
        """
        ids = self._reconstruction_indices
        return [self.meshes[i] for i in ids]

    @property
    def reconstructed_initial_camera_transforms(self):
        r"""
        Returns the camera transforms of the reconstructed initial meshes.

        :type: `list` of `menpo3d.camera.PerspectiveCamera`
        """
        ids = self._reconstruction_indices
        return [self.camera_transforms[i] for i in ids]

    def rasterized_reconstructed_initial_meshes(self, shape=None):
        r"""
        Returns the rasterized reconstructed initial meshes. The image's shape
        will be used in case `shape` is not provided.

        Parameters
        ----------
        shape: `(int, int)` or ``None``, optional
            The shape of the rasterized image. If ``None``, then the fitted
            image's shape will be used.

        Returns
        -------
        rasterized_reconstructed_initial_meshes : `list` of `menpo.image.Image`
            The list of images with the rasterized reconstructed initial meshes.

        Raises
        ------
        ValueError
            You need to provide an image shape, since the image does not exist.
        """
        if shape is None:
            if self.image is None:
                raise ValueError("You need to provide an image shape, "
                                 "since the image does not exist.")
            else:
                shape = self.image.shape
        ids = self._reconstruction_indices
        rasterized_meshes = []
        for i in ids:
            instance_in_img = self._affine_transforms[i].apply(
                self.camera_transforms[i].apply(self.meshes[i]))
            rasterized_meshes.append(rasterize_mesh(instance_in_img, shape))
        return rasterized_meshes

    @property
    def _reconstruction_indices(self):
        r"""
        Returns a list with the indices of reconstructed meshes in the `meshes`
        list.

        :type: `list` of `int`
        """
        initial_val = 0
        if self.initial_mesh is not None:
            initial_val = 1
        ids = []
        for i in list(range(self.n_scales)):
            if i == 0:
                ids.append(initial_val)
            else:
                previous_val = ids[i - 1]
                ids.append(previous_val + self.n_iters_per_scale[i - 1] + 1)
        return ids

    def reconstructed_initial_error(self, compute_error=None):
        r"""
        Returns the error of the reconstructed initial mesh of the fitting
        process, if the ground truth mesh exists. This is the error computed
        based on the `reconstructed_initial_meshes[0]`.

        Parameters
        ----------
        compute_error: `callable`, optional
            Callable that computes the error between the reconstructed initial
            and ground truth meshes.

        Returns
        -------
        reconstructed_initial_error : `float`
            The error that corresponds to the initial mesh's reconstruction.

        Raises
        ------
        ValueError
            Ground truth mesh has not been set, so the reconstructed initial
            error cannot be computed
        """
        if compute_error is None:
            compute_error = error_function
        if self.gt_mesh is None:
            raise ValueError('Ground truth mesh has not been set, so the '
                             'reconstructed initial error cannot be computed')
        else:
            return compute_error(self.reconstructed_initial_meshes[0],
                                 self.gt_mesh)
