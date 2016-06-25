import numpy as np
from menpo.visualize.base import Renderer


class MayaviViewer(Renderer):
    """
    Abstract class for performing visualizations using Mayavi.

    Parameters
    ----------
    figure_id : str or `None`
        A figure name or `None`. `None` assumes we maintain the Mayavi
        state machine and use `mlab.gcf()`.
    new_figure : bool
        If `True`, creates a new figure to render on.
    """

    def __init__(self, figure_id, new_figure):
        try:
            import mayavi
        except ImportError:
            raise ImportError("mayavi is required for viewing 3D objects "
                              "(consider 'conda/pip install mayavi')")
        super(MayaviViewer, self).__init__(figure_id, new_figure)

        self._supported_ext = ['.png', '.jpg', '.bmp', '.tiff',  # 2D
                               '.ps', '.eps', '.pdf',  # 2D
                               '.rib', '.oogl', 'iv', '.vrml', '.obj']  # 3D
        n_ext = len(self._supported_ext)
        func_list = [lambda obj, fp: mayavi.mlab.savefig(fp, **obj)] * n_ext
        self._extensions_map = dict(zip(['.' + s for s in self._supported_ext],
                                    func_list))

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
        import mayavi.mlab as mlab
        if self.new_figure or self.figure_id is not None:
            self.figure = mlab.figure(self.figure_id)
        else:
            self.figure = mlab.gcf()

        self.figure_id = self.figure.name

        return self.figure

    def save_figure(self, filename, format='png', size=None,
                    magnification='auto', overwrite=False):
        r"""
        Method for saving the figure of the current `figure_id` to file.

        Parameters
        ----------
        filename : `str` or `file`-like object
            The string path or file-like object to save the figure at/into.
        format : `str`
            The format to use. This must match the file path if the file path is
            a `str`.
        size : `tuple` of `int`, optional
            The size of the image created (unless magnification is set,
            in which case it is the size of the window used for rendering).
        magnification :	`double`, optional
            The magnification is the scaling between the pixels on the screen,
            and the pixels in the file saved. If you do not specify it, it will
            be calculated so that the file is saved with the specified size.
            If you specify a magnification, Mayavi will use the given size as a
            screen size, and the file size will be ``magnification * size``.
        overwrite : `bool`, optional
            If ``True``, the file will be overwritten if it already exists.
        """
        from menpo.io.output.base import _export
        savefig_args = {'size': size, 'figure': self.figure,
                        'magnification': magnification}
        # Use the export code so that we have a consistent interface
        _export(savefig_args, filename, self._extensions_map, format,
                overwrite=overwrite)

    @property
    def width(self):
        r"""The width scene in pixels
        """
        return self.figure.scene.get_size()[0]

    @property
    def height(self):
        return self.figure.scene.get_size()[1]

    @property
    def modelview_matrix(self):
        r"""Retrieves the modelview matrix for this scene.
        """
        camera = self.figure.scene.camera
        return camera.view_transform_matrix.to_array().astype(np.float32)

    @property
    def projection_matrix(self):
        r"""Retrieves the projection matrix for this scene.
        """
        scene = self.figure.scene
        camera = scene.camera
        scene_size = tuple(scene.get_size())
        aspect_ratio = float(scene_size[0]) / float(scene_size[1])
        p = camera.get_projection_transform_matrix(
            aspect_ratio, -1, 1).to_array().astype(np.float32)
        return p

    @property
    def renderer_settings(self):
        r"""Returns all the information required to construct an identical
        renderer to this one

        Returns

        width: int
            The width of the render window

        height: int
            The height of the render window
        model_matrix: ndarray of shape (4,4)
            The model array - always identity
        view_matrix: ndarray of shape (4,4)
            The view array - actually combined modelview
        projection_matrix: ndarray of shape (4,4)
            The projection array.
        """
        return {'width': self.width,
                'height': self.height,
                'model_matrix': np.eye(4, dtype=np.float32),
                'view_matrix': self.modelview_matrix,
                'projection_matrix': self.projection_matrix}


class MayaviPointCloudViewer3d(MayaviViewer):

    def __init__(self, figure_id, new_figure, points):
        super(MayaviPointCloudViewer3d, self).__init__(figure_id, new_figure)
        self.points = points

    def render(self, marker_size=1, marker_face_colour=(1, 1, 1), **kwargs):
        from mayavi import mlab
        mlab.points3d(
            self.points[:, 0], self.points[:, 1], self.points[:, 2],
            figure=self.figure, scale_factor=marker_size,
            color=marker_face_colour)
        return self


class MayaviPointGraphViewer3d(MayaviViewer):

    def __init__(self, figure_id, new_figure, points, adjacency_array):
        super(MayaviPointGraphViewer3d, self).__init__(figure_id, new_figure)
        self.points = points
        self.adjacency_array = adjacency_array

    def render(self, line_width=4, render_points=True, points_color=(0, 0, 1),
               points_scale=2, line_color=(0, 0, 1), **kwargs):
        from mayavi import mlab
        # Create the points
        src = mlab.pipeline.scalar_scatter(self.points[:, 0], self.points[:, 1],
                                           self.points[:, 2])
        # Connect them
        src.mlab_source.dataset.lines = self.adjacency_array
        # The stripper filter cleans up connected lines
        lines = mlab.pipeline.stripper(src)

        # Finally, display the set of lines
        mlab.pipeline.surface(lines, line_width=line_width, color=line_color)
        if render_points:
            mlab.points3d(self.points[:, 0], self.points[:, 1],
                          self.points[:, 2], scale_factor=points_scale,
                          color=points_color)
        return self


class MayaviSurfaceViewer3d(MayaviViewer):

    def __init__(self, figure_id, new_figure, values, mask=None):
        super(MayaviSurfaceViewer3d, self).__init__(figure_id, new_figure)
        if mask is not None:
            values[~mask] = np.nan
        self.values = values

    def render(self, **kwargs):
        from mayavi import mlab
        warp_scale = kwargs.get('warp_scale', 'auto')
        mlab.surf(self.values, warp_scale=warp_scale)
        return self


class MayaviLandmarkViewer3d(MayaviViewer):

    def __init__(self, figure_id, new_figure, pointcloud, lmark_group):
        super(MayaviLandmarkViewer3d, self).__init__(figure_id, new_figure)
        self.pointcloud = pointcloud
        self.lmark_group = lmark_group

    def render(self, scale_factor=1.0, text_scale=1.0, **kwargs):
        import mayavi.mlab as mlab
        # disabling the rendering greatly speeds up this for loop
        self.figure.scene.disable_render = True
        positions = []
        for label in self.lmark_group:
            p = self.lmark_group[label]
            for i, p in enumerate(p.points):
                positions.append(p)
                l = '%s_%d' % (label, i)
                # TODO: This is due to a bug in mayavi that won't allow
                # rendering text to an empty figure
                mlab.points3d(p[0], p[1], p[2], scale_factor=scale_factor)
                mlab.text3d(p[0], p[1], p[2], l, figure=self.figure,
                            scale=text_scale)
        positions = np.array(positions)
        os = np.zeros_like(positions)
        os[:, 2] = 1
        mlab.quiver3d(positions[:, 0], positions[:, 1], positions[:, 2],
                      os[:, 0], os[:, 1], os[:, 2], figure=self.figure)
        self.figure.scene.disable_render = False

        # Ensure everything fits inside the camera viewport
        mlab.get_engine().current_scene.scene.reset_zoom()

        return self


class MayaviTriMeshViewer3d(MayaviViewer):

    def __init__(self, figure_id, new_figure, points, trilist):
        super(MayaviTriMeshViewer3d, self).__init__(figure_id, new_figure)
        self.points = points
        self.trilist = trilist

    def _render_mesh(self):
        import mayavi.mlab as mlab
        mlab.triangular_mesh(self.points[:, 0],
                             self.points[:, 1],
                             self.points[:, 2],
                             self.trilist,
                             color=(0.5, 0.5, 0.5),
                             figure=self.figure)

    def render(self, normals=None, **kwargs):
        if normals is not None:
            MayaviVectorViewer3d(self.figure_id, False,
                                 self.points, normals).render(**kwargs)
        self._render_mesh()
        return self


class MayaviTexturedTriMeshViewer3d(MayaviTriMeshViewer3d):

    def __init__(self, figure_id, new_figure, points,
                 trilist, texture, tcoords_per_point):
        super(MayaviTexturedTriMeshViewer3d, self).__init__(figure_id,
                                                            new_figure,
                                                            points,
                                                            trilist)
        self.texture = texture
        self.tcoords_per_point = tcoords_per_point

    def _render_mesh(self):
        from tvtk.api import tvtk
        pd = tvtk.PolyData()
        pd.points = self.points
        pd.polys = self.trilist
        pd.point_data.t_coords = self.tcoords_per_point
        mapper = tvtk.PolyDataMapper()
        mapper.set_input_data(pd)
        actor = tvtk.Actor(mapper=mapper)
        # Get the pixels from our image class which are [0, 1] and scale
        # back to valid pixels. Then convert to tvtk ImageData.
        texture = self.texture.pixels_with_channels_at_back(out_dtype=np.uint8)
        if self.texture.n_channels == 1:
            texture = np.stack([texture] * 3, axis=-1)
        image_data = np.flipud(texture).ravel()
        image_data = image_data.reshape([-1, 3])
        image = tvtk.ImageData()
        image.point_data.scalars = image_data
        image.dimensions = self.texture.width, self.texture.height, 1
        texture = tvtk.Texture()
        texture.set_input_data(image)
        actor.texture = texture
        self.figure.scene.add_actors(actor)


class MayaviColouredTriMeshViewer3d(MayaviTriMeshViewer3d):

    def __init__(self, figure_id, new_figure, points,
                 trilist, colour_per_point):
        super(MayaviColouredTriMeshViewer3d, self).__init__(figure_id,
                                                            new_figure,
                                                            points,
                                                            trilist)
        self.colour_per_point = colour_per_point

    def _render_mesh(self):
        from tvtk.api import tvtk
        pd = tvtk.PolyData()
        pd.points = self.points
        pd.polys = self.trilist
        pd.point_data.scalars = (self.colour_per_point * 255.).astype(np.uint8)
        mapper = tvtk.PolyDataMapper(input=pd)
        actor = tvtk.Actor(mapper=mapper)
        self.figure.scene.add_actors(actor)


class MayaviVectorViewer3d(MayaviViewer):

    def __init__(self, figure_id, new_figure, points, vectors):
        super(MayaviVectorViewer3d, self).__init__(figure_id,
                                                   new_figure)
        self.points = points
        self.vectors = vectors

    def render(self, **kwargs):
        from mayavi import mlab
        # Only get every nth vector. 1 means get every vector.
        mask_points = kwargs.get('mask_points', 1)
        mlab.quiver3d(self.points[:, 0],
                      self.points[:, 1],
                      self.points[:, 2],
                      self.vectors[:, 0],
                      self.vectors[:, 1],
                      self.vectors[:, 2],
                      mask_points=mask_points,
                      figure=self.figure)
        return self
