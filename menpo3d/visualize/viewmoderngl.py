"""The moderngl 3D viewers behind ``mesh.view()``, ``landmarks.view()`` and the widgets (the constructor arguments and
``render`` options are those the Mayavi viewers had, which menpo's ``view`` methods call).

A *figure* is a scene plus a camera. ``figure_id``/``new_figure`` select one the way ``mlab.figure`` did.
Drawing is cheap (it adds items to the scene); pixels are made on demand: by ``save_figure``, by the notebook widget
or by the standalone window, always with the same renderer.
"""

import io
import os
import threading

import numpy as np
from menpo.visualize import Renderer

from .viewer.scene import Scene

GLOBAL_CMAP = "jet"
DEFAULT_SIZE = (640, 480)

_figures = {}  # name -> Figure
_current = None
_lock = threading.Lock()


def _parse_colour(colour):
    from matplotlib.colors import ColorConverter

    return ColorConverter().to_rgb(colour)


def _parse_marker_size(marker_size, points):
    if marker_size is None:
        from .viewer.gl import _auto_marker_size

        return _auto_marker_size(np.asarray(points, float))
    return marker_size


def _check_colours_list(render_flag, colours_list, n_objects, error_str):
    from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap

    if render_flag:
        if colours_list is None:
            colours_list = sample_colours_from_colourmap(n_objects, GLOBAL_CMAP)
        if isinstance(colours_list, list):
            if len(colours_list) == 1:
                colours_list[0] = _parse_colour(colours_list[0])
                colours_list *= n_objects
            elif len(colours_list) != n_objects:
                raise ValueError(error_str)
        else:
            colours_list = [_parse_colour(colours_list)] * n_objects
    else:
        colours_list = [None] * n_objects
    return colours_list


def _draw_labels(rgb, labels, std, camera, w, h):
    """Text anchored in 3D (point numbering). The browser draws the same labels over the live stream; offline they are
    drawn into the picture so ``save_figure`` shows them too."""
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    path = font_manager.findfont("DejaVu Sans")
    focal = 1 / np.tan(np.radians(camera.fov) / 2)
    for label in labels:
        clip = std @ np.array([*label["p"], 1.0])
        if clip[3] <= 0:
            continue
        x, y = (clip[0] / clip[3] + 1) * w / 2, (1 - clip[1] / clip[3]) * h / 2
        px = int(np.clip(label["size"] * focal * (h / 2) / clip[3] * 1.5, 7, 48))
        colour = tuple(int(255 * c) for c in label["colour"])
        draw.text(
            (x, y),
            label["text"],
            fill=colour,
            font=ImageFont.truetype(path, px),
            anchor="ls",
        )
    return np.asarray(image)


class Figure:
    """A scene and its camera. Displayed in a notebook it becomes the interactive widget; ``save`` renders it offline."""

    def __init__(self, name, size=DEFAULT_SIZE):
        self.name = name
        self.scene = Scene(
            name, background=(1, 1, 1)
        )  # menpo's Mayavi figures have a white background
        self.size = tuple(size)
        self._widget = None
        # Keep the last still as an ordinary image output in the saved notebook. Turn off when the figure is shown inside an
        # ipywidgets Output widget (it keeps its own state, and a display() while formatting would overwrite the widget view).
        self.persist_still = True

    # ---- pixels
    def render(self, size=None):
        """RGB uint8 array (height, width, 3) of the scene from the camera's current pose."""
        from .viewer.worker import get_worker

        w, h = (int(v) for v in (size or self.size))
        self.scene.ensure_camera()
        rgb, std = get_worker().render(self.scene, self.scene.camera, w, h)
        labels = self.scene.labels()
        return (
            _draw_labels(rgb, labels, std, self.scene.camera, w, h) if labels else rgb
        )

    def save(self, filename, size=None, magnification=1):
        from PIL import Image

        w, h = size or self.size
        rgb = self.render((int(w * magnification), int(h * magnification)))
        Image.fromarray(rgb).save(filename)

    # ---- display
    def _repr_mimebundle_(self, **kwargs):
        try:
            from .viewer import require_viewer

            require_viewer()
            from scene_streamer.widget import figure_widget

            return figure_widget(self)._repr_mimebundle_(**kwargs)
        except (
            ImportError
        ) as error:  # no streaming extras: a picture is still better than nothing
            import base64
            import warnings

            from PIL import Image

            warnings.warn(
                f"showing a still picture, not the interactive viewer: {error}",
                stacklevel=2,
            )
            buf = io.BytesIO()
            Image.fromarray(self.render()).save(buf, "PNG")
            return {"image/png": base64.b64encode(buf.getvalue()).decode()}

    def get_pose(self):
        """The camera pose (azimuth, elevation, distance, target, fov): place the mesh interactively, then read it here."""
        return self.scene.camera.get_pose()

    def set_pose(self, **pose):
        """Put the camera at a pose from ``get_pose``. With content in the scene it is kept: the scene is not re-framed."""
        self.scene.camera.set_pose(**pose)
        if self.scene.bounds() is not None:
            self.scene.camera_fitted, self.scene.needs_fit = True, False
        self.scene._changed()

    def reset_camera(self):
        """Frame the current content, like Mayavi's reset view (keeps the viewing direction)."""
        self.scene.fit_camera()
        self.scene._changed()

    def show(self, block=True, window=None):
        """Open the standalone viewer window on this figure and return when it is closed.

        ``window="native"`` is a real window (needs ``pip install "menpo3d[viz_native_window]"`` and a display); ``"browser"`` is a browser window
        in app mode (works over SSH with a port forward; needs ``menpo3d[viz_notebook]``). The default is native when moderngl-window
        and a display are there, otherwise the browser.
        """
        choice = window or os.environ.get("MENPO3D_WINDOW")
        native_error = "not requested"
        if choice in (None, "native"):
            try:
                from .viewer.native import require_window, show

                require_window()
                return show(self, block=block)
            except (ImportError, RuntimeError) as error:
                if choice == "native":
                    raise
                native_error = error
        if choice not in (None, "browser", "native"):
            raise ValueError(f"window must be 'native' or 'browser', not {choice!r}")
        from .viewer import require_viewer

        try:
            require_viewer()
        except ImportError as error:
            raise ImportError(
                f"no native window ({native_error}); and {error}"
            ) from None
        from scene_streamer.window import show

        return show(self, block=block)


def get_figure(figure_id, new_figure):
    global _current
    with _lock:
        if new_figure or (figure_id is None and _current is None):
            name = figure_id if figure_id is not None else f"Figure {len(_figures) + 1}"
            fig = _figures[name] = Figure(name)
        elif figure_id is not None:
            fig = _figures.get(figure_id) or _figures.setdefault(
                figure_id, Figure(figure_id)
            )
        else:
            fig = _current
        _current = fig
        return fig


def gcf():
    return get_figure(None, False)


def show(block=True, window=None):
    """Like ``mlab.show()``: open the window on the current figure."""
    return gcf().show(block=block, window=window)


class ModernGLRenderer(Renderer):
    """Base class of the moderngl viewers."""

    def __init__(self, figure_id, new_figure):
        super().__init__(figure_id, new_figure)
        self._supported_ext = ["png", "jpg", "jpeg", "bmp", "tiff"]

    def get_figure(self):
        self.figure = get_figure(self.figure_id, self.new_figure)
        self.figure_id = self.figure.name
        return self.figure

    @property
    def scene(self):
        return self.figure.scene

    def save_figure(
        self, filename, format="png", size=None, magnification="auto", overwrite=False
    ):
        from menpo.io.output.base import _export

        fig = self.figure

        def save(obj, fp, **kwargs):
            from PIL import Image

            w, h = obj["size"] or fig.size
            mag = 1 if obj["magnification"] == "auto" else obj["magnification"]
            rgb = fig.render((int(w * mag), int(h * mag)))
            image = Image.fromarray(rgb)
            image.save(
                fp,
                format="JPEG" if format.lower() in ("jpg", "jpeg") else format.upper(),
            )

        extensions = {"." + e: save for e in self._supported_ext}
        _export(
            dict(size=size, magnification=magnification),
            filename,
            extensions,
            format,
            overwrite=overwrite,
        )

    @property
    def width(self):
        return self.figure.size[0]

    @property
    def height(self):
        return self.figure.size[1]

    @property
    def modelview_matrix(self):
        return self.scene.camera.view_matrix().astype(np.float32)

    @property
    def projection_matrix(self):
        return self.scene.camera.projection_matrix(self.width / self.height).astype(
            np.float32
        )

    @property
    def renderer_settings(self):
        return {
            "width": self.width,
            "height": self.height,
            "model_matrix": np.eye(4, dtype=np.float32),
            "view_matrix": self.modelview_matrix,
            "projection_matrix": self.projection_matrix,
        }

    def clear_figure(self):
        self.figure.scene.clear()

    def add_colorbar(self, cmap="jet", vmin=0.0, vmax=1.0, title=None, **kwargs):
        """A colour bar over the picture (position='right'|'left'|'top'|'bottom'); drawn into every render, saved images included."""
        self.scene.add_colorbar(cmap, vmin, vmax, title=title, **kwargs)
        return self

    def add_text(self, text, position="top-left", **kwargs):
        """Text over the picture, e.g. statistics: a corner name or (x, y) as image fractions."""
        self.scene.add_text(text, position, **kwargs)
        return self

    def force_draw(self):
        """Nothing to force: changing the scene already tells every attached viewer to redraw."""

    def _repr_mimebundle_(self, **kwargs):
        return self.figure._repr_mimebundle_(**kwargs)


def _numbers(scene, points, render_numbering, numbers_size, numbers_colour):
    if render_numbering:
        size = _parse_marker_size(numbers_size, points)
        scene.add_labels(
            points,
            [str(k) for k in range(len(points))],
            colour=_parse_colour(numbers_colour),
            size=size,
        )


class ModernGLVectorViewer3d(ModernGLRenderer):
    def __init__(self, figure_id, new_figure, points, vectors):
        super().__init__(figure_id, new_figure)
        self.points, self.vectors = points, vectors

    def render(
        self,
        colour="r",
        line_width=2,
        marker_style="2darrow",
        marker_resolution=8,
        marker_size=None,
        step=None,
        alpha=1.0,
    ):
        scale = _parse_marker_size(marker_size, self.points)
        self.scene.add_vectors(
            self.points,
            self.vectors,
            colour=_parse_colour(colour),
            scale=scale,
            width=line_width,
            alpha=alpha,
            step=step,
        )
        return self


class ModernGLPointGraphViewer3d(ModernGLRenderer):
    def __init__(self, figure_id, new_figure, points, edges):
        super().__init__(figure_id, new_figure)
        self.points, self.edges = points, edges

    def render(
        self,
        render_lines=True,
        line_colour="r",
        line_width=2,
        render_markers=True,
        marker_style="sphere",
        marker_size=None,
        marker_colour="r",
        marker_resolution=8,
        step=None,
        alpha=1.0,
        render_numbering=False,
        numbers_colour="k",
        numbers_size=None,
    ):
        if render_lines and len(self.edges):
            self.scene.add_lines(
                self.points,
                self.edges,
                colour=_parse_colour(line_colour),
                width=line_width,
                alpha=alpha,
            )
        if render_markers:
            size = _parse_marker_size(marker_size, self.points)
            self.scene.add_markers(
                self.points,
                size,
                colour=_parse_colour(marker_colour),
                style=marker_style,
                alpha=alpha,
                step=step,
            )
        _numbers(
            self.scene, self.points, render_numbering, numbers_size, numbers_colour
        )
        return self


def _normals(viewer, normals, step, colour, line_width, style, resolution, size, alpha):
    if normals is not None:
        ModernGLVectorViewer3d(viewer.figure_id, False, viewer.points, normals).render(
            colour=colour,
            line_width=line_width,
            step=step,
            marker_style=style,
            marker_resolution=resolution,
            marker_size=size,
            alpha=alpha,
        )


class ModernGLTriMeshViewer3d(ModernGLRenderer):
    def __init__(self, figure_id, new_figure, points, trilist):
        super().__init__(figure_id, new_figure)
        self.points, self.trilist = points, trilist

    def render(
        self,
        mesh_type="wireframe",
        line_width=2,
        colour="r",
        marker_style="sphere",
        marker_size=None,
        marker_resolution=8,
        normals=None,
        normals_colour="k",
        normals_line_width=2,
        normals_marker_style="2darrow",
        normals_marker_size=None,
        normals_marker_resolution=8,
        step=None,
        alpha=1.0,
    ):
        _normals(
            self,
            normals,
            step,
            normals_colour,
            normals_line_width,
            normals_marker_style,
            normals_marker_resolution,
            normals_marker_size,
            alpha,
        )
        self.scene.add_mesh(
            self.points,
            self.trilist,
            colour=_parse_colour(colour),
            mode=mesh_type,
            alpha=alpha,
            line_width=line_width,
            marker_size=None if marker_size is None else marker_size,
        )
        return self


class ModernGLColouredTriMeshViewer3d(ModernGLRenderer):
    def __init__(self, figure_id, new_figure, points, trilist, colour_per_point):
        super().__init__(figure_id, new_figure)
        self.points, self.trilist, self.colour_per_point = (
            points,
            trilist,
            colour_per_point,
        )

    def render(
        self,
        mesh_type="surface",
        ambient_light=0.0,
        specular_light=0.0,
        normals=None,
        normals_colour="k",
        normals_line_width=2,
        normals_marker_style="2darrow",
        normals_marker_resolution=8,
        normals_marker_size=None,
        step=None,
        alpha=1.0,
    ):
        _normals(
            self,
            normals,
            step,
            normals_colour,
            normals_line_width,
            normals_marker_style,
            normals_marker_resolution,
            normals_marker_size,
            alpha,
        )
        self.scene.add_mesh(
            self.points,
            self.trilist,
            mode=mesh_type,
            alpha=alpha,
            colours=np.clip(self.colour_per_point, 0, 1),
            ambient=ambient_light,
            specular=specular_light,
        )
        return self


class ModernGLTexturedTriMeshViewer3d(ModernGLRenderer):
    def __init__(
        self, figure_id, new_figure, points, trilist, texture, tcoords_per_point
    ):
        super().__init__(figure_id, new_figure)
        self.points, self.trilist, self.texture, self.tcoords_per_point = (
            points,
            trilist,
            texture,
            tcoords_per_point,
        )

    def render(
        self,
        mesh_type="surface",
        ambient_light=0.0,
        specular_light=0.0,
        normals=None,
        normals_colour="k",
        normals_line_width=2,
        normals_marker_style="2darrow",
        normals_marker_resolution=8,
        normals_marker_size=None,
        step=None,
        alpha=1.0,
    ):
        _normals(
            self,
            normals,
            step,
            normals_colour,
            normals_line_width,
            normals_marker_style,
            normals_marker_resolution,
            normals_marker_size,
            alpha,
        )
        pixels = self.texture.pixels_with_channels_at_back(out_dtype=np.uint8)
        if self.texture.n_channels == 1:
            pixels = np.stack([pixels] * 3, axis=-1)
        self.scene.add_mesh(
            self.points,
            self.trilist,
            mode=mesh_type,
            alpha=alpha,
            tcoords=self.tcoords_per_point,
            texture=pixels,
            ambient=ambient_light,
            specular=specular_light,
        )
        return self


class ModernGLSurfaceViewer3d(ModernGLRenderer):
    """A 2D array drawn as a height field (``mlab.surf``): rows and columns are x and y, the values are z and the colour."""

    def __init__(self, figure_id, new_figure, values, mask=None):
        super().__init__(figure_id, new_figure)
        values = np.array(values, dtype=float)
        if mask is not None:
            values[~np.asarray(mask, dtype=bool)] = np.nan
        self.values = values

    def render(
        self,
        cmap="jet",
        warp_scale="auto",
        mesh_type="surface",
        line_width=1.0,
        alpha=1.0,
        scalar_range=None,
        colorbar=False,
        colorbar_title=None,
        statistics=None,
        **kwargs,
    ):
        """``warp_scale='auto'`` gives the z extent 0.3 * (x span + y span), like Mayavi; a number scales the values.
        ``colorbar`` adds a colour bar; ``statistics`` is a list of names (mean, std, min, max, median, rms) printed in a corner."""
        from .viewer.colormap import scalars_to_colours, statistics_lines

        v = self.values
        rows, cols = v.shape
        x, y = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        finite = np.isfinite(v)
        lo, hi = (
            scalar_range
            if scalar_range is not None
            else (np.nanmin(v), np.nanmax(v))
            if finite.any()
            else (0.0, 1.0)
        )
        if warp_scale == "auto":
            warp_scale = 0.3 * ((rows - 1) + (cols - 1)) / (hi - lo) if hi > lo else 1.0
        z = np.nan_to_num(v, nan=lo) * warp_scale
        points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(float)
        index = np.arange(rows * cols).reshape(rows, cols)
        a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
        ok = (
            finite[:-1, :-1] & finite[1:, :-1] & finite[1:, 1:] & finite[:-1, 1:]
        ).ravel()
        trilist = np.concatenate(
            [
                np.stack([a.ravel(), b.ravel(), c.ravel()], 1)[ok],
                np.stack([a.ravel(), c.ravel(), d.ravel()], 1)[ok],
            ]
        )
        colours, lo, hi = scalars_to_colours(v.ravel(), cmap, lo, hi)
        self.scene.add_mesh(
            points,
            trilist,
            mode=mesh_type,
            alpha=alpha,
            line_width=line_width,
            colours=colours,
        )
        if colorbar:
            self.add_colorbar(cmap, lo, hi, title=colorbar_title)
        if statistics:
            self.add_text("\n".join(statistics_lines(v, statistics)), "top-left")
        return self


class ModernGLLandmarkViewer3d(ModernGLRenderer):
    def __init__(self, figure_id, new_figure, group, landmark_group):
        super().__init__(figure_id, new_figure)
        self.group, self.landmark_group = group, landmark_group

    def render(
        self,
        render_lines=True,
        line_colour="r",
        line_width=2,
        render_markers=True,
        marker_style="sphere",
        marker_size=None,
        marker_colour="r",
        marker_resolution=8,
        step=None,
        alpha=1.0,
        render_numbering=False,
        numbers_colour="k",
        numbers_size=None,
    ):
        n_labels = self.landmark_group.n_labels
        line_colour = _check_colours_list(
            render_lines,
            line_colour,
            n_labels,
            "Must pass a list of line colours with length n_labels or a single line colour for all labels.",
        )
        marker_colour = _check_colours_list(
            render_markers,
            marker_colour,
            n_labels,
            "Must pass a list of marker colours with length n_labels or a single marker face colour for all labels.",
        )
        marker_size = _parse_marker_size(marker_size, self.landmark_group.points)
        numbers_size = _parse_marker_size(numbers_size, self.landmark_group.points)
        for i, label in enumerate(self.landmark_group.labels):
            self.landmark_group.get_label(label).view(
                figure_id=self.figure_id,
                new_figure=False,
                render_lines=render_lines,
                line_colour=line_colour[i],
                line_width=line_width,
                render_markers=render_markers,
                marker_style=marker_style,
                marker_size=marker_size,
                marker_colour=marker_colour[i],
                marker_resolution=marker_resolution,
                step=step,
                alpha=alpha,
                render_numbering=render_numbering,
                numbers_colour=numbers_colour,
                numbers_size=numbers_size,
            )
        return self
