import numpy as np
from menpo.visualize import Renderer
from k3d import (Plot, mesh as k3d_mesh, points as k3d_points,
                 text as k3d_text, vectors as k3d_vectors,
                 line as k3d_line)
from k3d.colormaps import matplotlib_color_maps
from io import BytesIO
from ipywidgets import GridBox, Layout, Widget
from collections import defaultdict
# The colour map used for all lines and markers
GLOBAL_CMAP = 'jet'


def dict_figures():
    dict_fig = defaultdict(list)
    for x in Widget.widgets.values():
        if hasattr(x, 'figure_id'):
            dict_fig[x.figure_id].append(x.model_id)
    return dict_fig


def list_figures():
    list_figures = list(dict_figures().keys())
    for figure_id in list_figures:
        print(figure_id)


def clear_figure(figure_id=None):
    # TODO remove figures, clear memory
    dict_fig = dict_figures()


def _parse_marker_size(marker_size, points):
    if marker_size is None:
        from menpo.shape import PointCloud
        pc = PointCloud(points, copy=False)
        # This is the way that mayavi automatically computes the scale factor
        # in  case the user passes scale_factor = 'auto'. We use it for both
        # the  marker_size as well as the numbers_size.
        xyz_min, xyz_max = pc.bounds()
        x_min, y_min, z_min = xyz_min
        x_max, y_max, z_max = xyz_max
        distance = np.sqrt(((x_max - x_min) ** 2 +
                            (y_max - y_min) ** 2 +
                            (z_max - z_min) ** 2) /
                           (4 * pc.n_points ** 0.33))
        if distance == 0:
            marker_size = 1
        else:
            marker_size = 0.1 * distance
    return marker_size


def _parse_colour(colour):
    from matplotlib.colors import rgb2hex
    if isinstance(colour, int):
        return colour
    else:
        return int(rgb2hex(colour)[1:], base=16)


def _check_colours_list(render_flag, colours_list, n_objects, error_str):
    from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
    if render_flag:
        if colours_list is None:
            # sample colours from jet colour map
            colours_list = sample_colours_from_colourmap(n_objects,
                                                         GLOBAL_CMAP)
            colours_list = list(map(_parse_colour, colours_list))

        if isinstance(colours_list, list):
            if len(colours_list) == 1:
                colours_list[0] = _parse_colour(colours_list[0])
                colours_list *= n_objects
            elif len(colours_list) != n_objects:
                raise ValueError(error_str)
        else:
            colours_list = [_parse_colour(colours_list)] * n_objects
    else:
        colours_list = [0x00FF00] * n_objects
    return colours_list


def _check_figure_id(obj, figure_id, new_figure):
    if figure_id is None:
        if new_figure:
            # A new figure is created but with no figure_id
            # we should create an id of 'Figure_n form'
            list_ids = []
            for x in obj.widgets.values():
                if hasattr(x, 'figure_id') and x is not obj:
                    if x.figure_id is not None and 'Figure_' in str(x.figure_id):
                        try:
                            n_figure_id = int(x.figure_id.split('Figure_')[1])
                        except ValueError:
                            continue
                        list_ids.append(n_figure_id)
            if len(list_ids):
                figure_id = 'Figure_{}'.format(sorted(list_ids)[-1] + 1)
            else:
                figure_id = 'Figure_0'

        else:
            obj.remove_widget()
            raise ValueError('You cannot plot a figure with no id and new figure False')
    else:
        if new_figure:
            for x in obj.widgets.values():
                if hasattr(x, 'figure_id') and x is not obj:
                    if x.figure_id == figure_id:
                        obj.remove_widget()
                        raise ValueError('Figure id is already given')
    return figure_id


class K3dwidgetsRenderer(Plot, Renderer):
    """ Abstract class for performing visualizations using K3dwidgets.

    Parameters
    ----------
    figure_id : str or `None`
        A figure name or `None`.
    new_figure : bool
        If `True`, creates a new figure on the cell.
    """
    def __init__(self, figure_id, new_figure):
        super(K3dwidgetsRenderer, self).__init__()

        self.figure_id = _check_figure_id(self, figure_id, new_figure)
        self.new_figure = new_figure
        self.grid_visible = False
        self.camera = [-0.02, -0.12, 3.32,
                       0.00, -0.16, 0.58,
                       0.02, 1.00, 0.04]

    def _render(self):
        widg_to_draw = self
        if not self.new_figure:
            for widg in self.widgets.values():
                if isinstance(widg, K3dwidgetsRenderer):
                    if widg.figure_id == self.figure_id and widg.model_id != self.model_id and widg.new_figure:
                        widg_to_draw = widg
                        return widg_to_draw
            self.remove_widget()
            raise Exception('Figure with id {} was not found '.format(self.figure_id))

        return widg_to_draw

    def remove_widget(self):
        super(K3dwidgetsRenderer, self).close()
        # copy from close from ipywidgets.widget.Widget
        self.widgets.pop(self.model_id, None)
        self.comm.close()
        self.comm = None
        self._repr_mimebundle_ = None
        # TODO
        # Why the following atributes don't change

    def get_figure(self):
        r"""
        Gets the figure specified by the combination of `self.figure_id` and
        `self.new_figure`. If `self.figure_id == None` then `mlab.gcf()`
        is used. `self.figure_id` is also set to the correct id of the figure
        if a new figure is created.

        Returns
        -------
        figure : Mayavi figure object
            The figure we will be rendering on.
        """
        # return self.figure
        pass

#    def save_figure(self, filename, format='png', size=None,
#                    magnification='auto', overwrite=False):
#        r"""
#        Method for saving the figure of the current `figure_id` to file.
#
#        Parameters
#        ----------
#        filename : `str` or `file`-like object
#            The string path or file-like object to save the figure at/into.
#        format : `str`
#            The format to use. This must match the file path if the file path is
#            a `str`.
#        size : `tuple` of `int` or ``None``, optional
#            The size of the image created (unless magnification is set,
#            in which case it is the size of the window used for rendering). If
#            ``None``, then the figure size is used.
#        magnification :	`double` or ``'auto'``, optional
#            The magnification is the scaling between the pixels on the screen,
#            and the pixels in the file saved. If you do not specify it, it will
#            be calculated so that the file is saved with the specified size.
#            If you specify a magnification, Mayavi will use the given size as a
#            screen size, and the file size will be ``magnification * size``.
#            If ``'auto'``, then the magnification will be set automatically.
#        overwrite : `bool`, optional
#            If ``True``, the file will be overwritten if it already exists.
#        """
#        from menpo.io.output.base import _export
#        savefig_args = {'size': size, 'figure': self.figure,
#                        'magnification': magnification}
#        # Use the export code so that we have a consistent interface
#        _export(savefig_args, filename, self._extensions_map, format,
#                overwrite=overwrite)

    @property
    def _width(self):
        r"""
        The width of the scene in pixels.  An underscore has been added in the
        begining of the name due to conflict with K3d Plot class
        :type: `int`
        """
        pass

    @property
    def _height(self):
        r"""
        The height of the scene in pixels.  An underscore has been added in the
        begining of the name due to conflict with K3d Plot class

        :type: `int`
        """
        pass

    @property
    def modelview_matrix(self):
        r"""
        Retrieves the modelview matrix for this scene.

        :type: ``(4, 4)`` `ndarray`
        """
        # camera = self.figure.scene.camera
        # return camera.view_transform_matrix.to_array().astype(np.float32)
        pass

    @property
    def projection_matrix(self):
        r"""
        Retrieves the projection matrix for this scene.

        :type: ``(4, 4)`` `ndarray`
        """
#         scene = self.figure.scene
#         camera = scene.camera
#         scene_size = tuple(scene.get_size())
#         aspect_ratio = float(scene_size[0]) / float(scene_size[1])
#         p = camera.get_projection_transform_matrix(
#             aspect_ratio, -1, 1).to_array().astype(np.float32)
#         return p
        pass

    @property
    def renderer_settings(self):
        r"""
        Returns all the information required to construct an identical
        renderer to this one.

        Returns
        -------
        settings : `dict`
            The dictionary with the following keys:

                * ``'width'`` (`int`) : The width of the scene.
                * ``'height'`` (`int`) : The height of the scene.
                * ``'model_matrix'`` (`ndarray`) : The model array (identity).
                * ``'view_matrix'`` (`ndarray`) : The view array.
                * ``'projection_matrix'`` (`ndarray`) : The projection array.

        """
        return {'width': self.width,
                'height': self.height,
                'model_matrix': np.eye(4, dtype=np.float32),
                'view_matrix': self.modelview_matrix,
                'projection_matrix': self.projection_matrix}

    def force_draw(self):
        r"""
        Method for forcing the current figure to render. This is useful for
        the widgets animation.
        """
        self.render()


class K3dwidgetsVectorViewer3d(K3dwidgetsRenderer):
    def __init__(self, figure_id, new_figure, points, vectors):
        super(K3dwidgetsVectorViewer3d, self).__init__(figure_id, new_figure)
        non_zero_indices = np.unique(np.nonzero(vectors.reshape(-1, 3))[0])
        self.points = points[non_zero_indices].astype(np.float32)
        self.vectors = vectors[non_zero_indices].astype(np.float32)

    def _render(self, colour='r', line_width=2, marker_size=None):
        marker_size = _parse_marker_size(marker_size, self.points)
        colour = _parse_colour(colour)

        widg_to_draw = super(K3dwidgetsVectorViewer3d, self)._render()
        vectors_to_add = k3d_vectors(self.points, self.vectors,
                                     color=colour, head_size=marker_size,
                                     line_width=line_width)
        widg_to_draw += vectors_to_add
        return widg_to_draw


class K3dwidgetsPointGraphViewer3d(K3dwidgetsRenderer):
    def __init__(self, figure_id, new_figure, points, edges):
        super(K3dwidgetsPointGraphViewer3d, self).__init__(figure_id,
                                                           new_figure)
        self.points = points.astype(np.float32)
        self.edges = edges

    def _render(self, render_lines=True, line_colour='r', line_width=2,
                render_markers=True, marker_style='flat', marker_size=10,
                marker_colour='g', render_numbering=False,
                numbers_colour='k', numbers_size=None):

        widg_to_draw = super(K3dwidgetsPointGraphViewer3d, self)._render()
        # Render the lines if requested
        if render_lines:
            if isinstance(line_colour, list):
                line_colour = [_parse_colour(i_color) for i_color in
                               line_colour]
            else:
                line_colour = _parse_colour(line_colour)

            lines_to_add = None
            for edge in self.edges:
                if isinstance(line_colour, list):
                    if len(line_colour):
                        color_this_line = line_colour.pop()
                    else:
                        color_this_line = 0xFF0000
                else:
                    color_this_line = line_colour

                if lines_to_add is None:
                    lines_to_add = k3d_line(self.points[edge],
                                            color=color_this_line)
                else:
                    lines_to_add += k3d_line(self.points[edge],
                                             color=color_this_line)
            widg_to_draw += lines_to_add

        # Render the markers if requested
        if render_markers:
            marker_size = _parse_marker_size(marker_size, self.points)
            marker_colour = _parse_colour(marker_colour)

            if marker_style == 'sphere':
                marker_style = 'mesh'

            points_to_add = k3d_points(self.points, color=marker_colour,
                                       point_size=marker_size,
                                       shader=marker_style)
            widg_to_draw += points_to_add

            if render_numbering:
                text_to_add = None
                for i, point in enumerate(self.points):
                    if text_to_add is None:
                        text_to_add = k3d_text(str(i), position=point,
                                               label_box=False)
                    else:
                        text_to_add += k3d_text(str(i), position=point,
                                                label_box=False)
                widg_to_draw += text_to_add

        return widg_to_draw


class K3dwidgetsTriMeshViewer3d(K3dwidgetsRenderer):
    def __init__(self, figure_id, new_figure, points, trilist, landmarks=None):
        super(K3dwidgetsTriMeshViewer3d, self).__init__(figure_id, new_figure)
        self.points = points.astype(np.float32)
        self.trilist = trilist.astype(np.uint32)
        self.landmarks = landmarks

    def _render_mesh(self, line_width, colour, marker_style, marker_size):
        marker_size = _parse_marker_size(marker_size, self.points)
        colour = _parse_colour(colour)

        widg_to_draw = super(K3dwidgetsTriMeshViewer3d, self)._render()
        mesh_to_add = k3d_mesh(self.points, self.trilist.flatten(),
                               flat_shading=False, color=colour, side='double')
        widg_to_draw += mesh_to_add

        if hasattr(self.landmarks, 'points'):
            self.landmarks.view(inline=True, new_figure=False,
                                figure_id=self.figure_id)
        return widg_to_draw

    def _render(self, line_width=2, colour='r',
                marker_style='sphere', marker_size=None,
                normals=None, normals_colour='k', normals_line_width=2,
                normals_marker_size=None):

        widg_to_draw = self._render_mesh(line_width, colour,
                                         marker_style, marker_size)
        if normals is not None:
            tmp_normals_widget = K3dwidgetsVectorViewer3d(self.figure_id,
                                                          False, self.points,
                                                          normals)
            tmp_normals_widget._render(colour=normals_colour,
                                       line_width=normals_line_width,
                                       marker_size=normals_marker_size)

        return widg_to_draw


class K3dwidgetsTexturedTriMeshViewer3d(K3dwidgetsRenderer):
    def __init__(self, figure_id, new_figure, points, trilist, texture,
                 tcoords, landmarks):
        super(K3dwidgetsTexturedTriMeshViewer3d, self).__init__(figure_id,
                                                                new_figure)
        self.points = points
        self.trilist = trilist
        self.texture = texture
        self.tcoords = tcoords
        self.landmarks = landmarks
        self.lighting = 0

    def _render_mesh(self, mesh_type='surface', ambient_light=0.0,
                     specular_light=0.0, alpha=1.0):

        widg_to_draw = super(K3dwidgetsTexturedTriMeshViewer3d, self)._render()

        uvs = self.tcoords.points
        tmp_img = self.texture.mirror(axis=0).as_PILImage()
        img_byte_arr = BytesIO()
        tmp_img.save(img_byte_arr, format='PNG')
        texture = img_byte_arr.getvalue()
        texture_file_format = 'png'

        mesh_to_add = k3d_mesh(self.points.astype(np.float32),
                               self.trilist.flatten().astype(np.uint32),
                               flat_shading=False,
                               color=0xFFFFFF, side='front', texture=texture,
                               uvs=uvs,
                               texture_file_format=texture_file_format)

        widg_to_draw += mesh_to_add

        if hasattr(self.landmarks, 'points'):
            self.landmarks.view(inline=True, new_figure=False,
                                figure_id=self.figure_id)

        self.camera = [-0.02, -0.12, 3.32,
                       0.00, -0.16, 0.58,
                       0.02, 1.00, 0.04]

        return widg_to_draw

    def _render(self, normals=None, normals_colour='k',
                normals_line_width=2, normals_marker_size=None):

        if normals is not None:
            tmp_normals_widget = K3dwidgetsVectorViewer3d(self.figure_id,
                                                          False, self.points,
                                                          normals)
            tmp_normals_widget._render(colour=normals_colour,
                                       line_width=normals_line_width,
                                       marker_size=normals_marker_size)

        self._render_mesh()
        return self


class K3dwidgetsColouredTriMeshViewer3d(K3dwidgetsRenderer):
    # TODO
    def __init__(self, figure_id, new_figure, points, trilist,
                 colour_per_point, landmarks):
        super(K3dwidgetsColouredTriMeshViewer3d, self).__init__(figure_id,
                                                                new_figure)
        self.points = points
        self.trilist = trilist
        self.colour_per_point = colour_per_point
        self.colorbar_object_id = False
        self.landmarks = landmarks

    def _render_mesh(self):
        widg_to_draw = super(K3dwidgetsColouredTriMeshViewer3d, self)._render()

        mesh_to_add = k3d_mesh(self.points.astype(np.float32),
                               self.trilist.flatten().astype(np.uint32),
                               attribute=self.colour_per_point,
                               )
        widg_to_draw += mesh_to_add

        if hasattr(self.landmarks, 'points'):
            self.landmarks.view(inline=True, new_figure=False,
                                figure_id=self.figure_id)

    def _render(self, normals=None, normals_colour='k', normals_line_width=2,
                normals_marker_size=None):
        if normals is not None:
            K3dwidgetsVectorViewer3d(self.figure_id, False,
                                     self.points, normals).render(
                colour=normals_colour, line_width=normals_line_width, step=step,
                marker_style=normals_marker_style,
                marker_resolution=normals_marker_resolution,
                marker_size=normals_marker_size, alpha=alpha)
        self._render_mesh()
        return self


class K3dwidgetsSurfaceViewer3d(K3dwidgetsRenderer):
    def __init__(self, figure_id, new_figure, values, mask=None):
        super(K3dwidgetsSurfaceViewer3d, self).__init__(figure_id, new_figure)
        if mask is not None:
            values[~mask] = np.nan
        self.values = values

    def render(self, colour=(1, 0, 0), line_width=2, step=None,
               marker_style='2darrow', marker_resolution=8, marker_size=0.05,
               alpha=1.0):
        # warp_scale = kwargs.get('warp_scale', 'auto')
        # mlab.surf(self.values, warp_scale=warp_scale)
        return self


class K3dwidgetsLandmarkViewer3d(K3dwidgetsRenderer):
    def __init__(self, figure_id, new_figure, group, landmark_group):
        super(K3dwidgetsLandmarkViewer3d, self).__init__(figure_id, new_figure)
        self.group = group
        self.landmark_group = landmark_group

    def _render(self, render_lines=True, line_colour='r', line_width=2,
                render_markers=True, marker_style='sphere', marker_size=None,
                marker_colour='r', marker_resolution=8, step=None, alpha=1.0,
                render_numbering=False, numbers_colour='k', numbers_size=None):
        # Regarding the labels colours, we may get passed either no colours (in
        # which case we generate random colours) or a single colour to colour
        # all the labels with
        # TODO: All marker and line options could be defined as lists...
        n_labels = self.landmark_group.n_labels
        line_colour = _check_colours_list(
            render_lines, line_colour, n_labels,
            'Must pass a list of line colours with length n_labels or a single '
            'line colour for all labels.')
        marker_colour = _check_colours_list(
            render_markers, marker_colour, n_labels,
            'Must pass a list of marker colours with length n_labels or a '
            'single marker face colour for all labels.')
        marker_size = _parse_marker_size(marker_size, self.landmark_group.points)
        numbers_size = _parse_marker_size(numbers_size,
                                          self.landmark_group.points)

        # get pointcloud of each label
        sub_pointclouds = self._build_sub_pointclouds()

        widg_to_draw = super(K3dwidgetsLandmarkViewer3d, self)._render()

        if marker_style == 'sphere':
            marker_style = 'mesh'

        for i, (label, pc) in enumerate(sub_pointclouds):
            points_to_add = k3d_points(pc.points.astype(np.float32),
                                       color=marker_colour[i],
                                       point_size=marker_size,
                                       shader=marker_style)
            widg_to_draw += points_to_add
        if render_numbering:
            text_to_add = None
            for i, point in enumerate(self.landmark_group.points):
                if text_to_add is None:
                    text_to_add = k3d_text(str(i), position=point,
                                           label_box=False)
                else:
                    text_to_add += k3d_text(str(i), position=point,
                                            label_box=False)
            widg_to_draw += text_to_add
        return widg_to_draw

    def _build_sub_pointclouds(self):
        return [(label, self.landmark_group.get_label(label))
                for label in self.landmark_group.labels]


class K3dwidgetsHeatmapViewer3d(K3dwidgetsRenderer):
    def __init__(self, figure_id, new_figure, points, trilist, landmarks=None):
        super(K3dwidgetsHeatmapViewer3d, self).__init__(figure_id, new_figure)
        self.points = points
        self.trilist = trilist
        self.landmarks = landmarks

    def _render_mesh(self, distances_between_meshes, type_cmap,
                     scalar_range, show_statistics=False):

        marker_size = _parse_marker_size(None, self.points)

        widg_to_draw = super(K3dwidgetsHeatmapViewer3d, self)._render()

        try:
            color_map = getattr(matplotlib_color_maps, type_cmap)
        except AttributeError:
            print('Could not find colormap {}. Hot_r is going to be used instead'.format(type_cmap))
            color_map = getattr(matplotlib_color_maps, 'hot_r')

        mesh_to_add = k3d_mesh(self.points.astype(np.float32),
                               self.trilist.flatten().astype(np.uint32),
                               color_map=color_map,
                               attribute=distances_between_meshes,
                               color_range=scalar_range
                               )
        widg_to_draw += mesh_to_add

        if hasattr(self.landmarks, 'points'):
            self.landmarks.view(inline=True, new_figure=False,
                                figure_id=self.figure_id)

        if show_statistics:
            text = '\\begin{{matrix}} \\mu &  {:.3} \\\\ \\sigma^2 & {:.3} \\\\ \\max & {:.3}  \\end{{matrix}}'\
                          .format(distances_between_meshes.mean(),
                                  distances_between_meshes.std(),
                                  distances_between_meshes.max())
            min_b = np.min(self.points, axis=0)
            max_b = np.max(self.points, axis=0)
            text_position = (max_b-min_b)/2
            widg_to_draw += k3d_text(text, position=text_position,
                                     color=0xff0000, size=1)

        return widg_to_draw

    def _render(self, distances_between_meshes, type_cmap='hot_r',
                scalar_range=[0, 2], show_statistics=False):
        return self._render_mesh(distances_between_meshes, type_cmap,
                                 scalar_range, show_statistics)


class K3dwidgetsPCAModelViewer3d(GridBox):
    def __init__(self, figure_id, new_figure, points, trilist,
                 components, eigenvalues, n_parameters, parameters_bound,
                 landmarks_indices, widget_style):

        try:
            from menpowidgets.options import LinearModelParametersWidget
        except ImportError as e:
            from menpo.visualize import MenpowidgetsMissingError
            raise MenpowidgetsMissingError(e)

        self.figure_id = _check_figure_id(self, figure_id, new_figure)
        self.new_figure = new_figure
        self.points = points.astype(np.float32)
        self.trilist = trilist.astype(np.uint32)
        self.components = components.astype(np.float32)
        self.eigenvalues = eigenvalues.astype(np.float32)
        self.n_parameters = n_parameters
        self.landmarks_indices = landmarks_indices
        self.layout = Layout(grid_template_columns='1fr 1fr')
        self.wid = LinearModelParametersWidget(n_parameters=n_parameters,
                                               render_function=self.render_function,
                                               params_str='Parameter ',
                                               mode='multiple',
                                               params_bounds=parameters_bound,
                                               plot_variance_visible=False,
                                               style=widget_style)
        self.mesh_window = K3dwidgetsTriMeshViewer3d(self.figure_id, False,
                                                     self.points, self.trilist)
        super(K3dwidgetsPCAModelViewer3d, self).__init__(children=[self.wid, self.mesh_window],
                                                         layout=Layout(grid_template_columns='1fr 1fr'))

    def _render_mesh(self, mesh_type, line_width, colour, marker_size,
                     marker_resolution, marker_style, step, alpha):
        marker_size = _parse_marker_size(marker_size, self.points)
        colour = _parse_colour(colour)

        mesh_to_add = k3d_mesh(self.points, self.trilist.flatten(),
                               flat_shading=False, color=colour,
                               name='Instance', side='double')

        self.mesh_window += mesh_to_add

        if self.landmarks_indices is not None:
            landmarks_to_add = k3d_points(self.points[self.landmarks_indices],
                                          color=0x00FF00, name='landmarks',
                                          point_size=marker_size,
                                          shader='mesh')
            self.mesh_window += landmarks_to_add
        return self

    def render_function(self, change):
        weights = np.asarray(self.wid.selected_values).astype(np.float32)
        weighted_eigenvalues = weights * self.eigenvalues[:self.n_parameters]**0.5
        new_instance = (self.components[:self.n_parameters, :].T@weighted_eigenvalues).reshape(-1, 3)
        mesh = self.points + new_instance

        self.mesh_window.objects[0].vertices = mesh
        if self.landmarks_indices is not None:
            self.mesh_window.objects[1].positions = mesh[self.landmarks_indices]

    def _render(self, mesh_type='wireframe', line_width=2, colour='r',
                marker_style='sphere', marker_size=None, marker_resolution=8,
                normals=None, normals_colour='k', normals_line_width=2,
                normals_marker_resolution=8, step=None, alpha=1.0):

        return self._render_mesh(mesh_type, line_width, colour, marker_size,
                                 marker_resolution, marker_style, step, alpha)

    def remove_widget(self):
        super(K3dwidgetsPCAModelViewer3d, self).close()
