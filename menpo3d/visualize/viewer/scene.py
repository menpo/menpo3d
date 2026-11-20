"""A scene is plain NumPy data: a list of items (meshes, line sets, markers, vectors, labels) and a camera.

Nothing here touches OpenGL, so a scene can be built anywhere (a notebook kernel, a script) and drawn by whichever
renderer is attached. Items are immutable once added and the list is replaced, never edited, so the render thread can
read a scene while another thread changes it without a lock.
"""

import itertools
import threading

import numpy as np

from .camera import OrbitCamera

_ids = itertools.count(1)

DEFAULT_COLOUR = (0.85, 0.75, 0.65)


def _f4(a, shape_tail):
    a = np.ascontiguousarray(a, dtype="f4")
    if a.ndim != 2 or a.shape[1] != shape_tail:
        raise ValueError(f"expected an (n, {shape_tail}) array, got {a.shape}")
    return a


def _rgb(colour):
    if isinstance(colour, str):
        from matplotlib.colors import to_rgb

        colour = to_rgb(colour)
    return tuple(float(c) for c in colour)


class Scene:
    def __init__(self, name=None, background=(1.0, 1.0, 1.0)):
        self.id = f"scene{next(_ids)}"
        self.name = name or self.id
        self.background = _rgb(background)
        self.items = ()  # replaced, never mutated
        self.version = 0
        self.camera = OrbitCamera()
        self.camera_fitted = False
        self.needs_fit = True  # new content after a clear re-frames the camera only if it is somewhere else or another size
        self.listeners = []  # called with the scene after every change (the hub wakes its viewers)
        self._lock = threading.Lock()

    # ---- editing
    def _changed(self):
        self.version += 1
        for listener in list(self.listeners):
            listener(self)

    def _add(self, item):
        with self._lock:
            self.items = self.items + (item,)
        self._changed()
        return item

    def clear(self):
        with self._lock:
            self.items = ()
        self.needs_fit = True
        self._changed()

    def add_mesh(
        self,
        points,
        trilist,
        colour=DEFAULT_COLOUR,
        mode="surface",
        alpha=1.0,
        line_width=1.0,
        colours=None,
        tcoords=None,
        texture=None,
        ambient=0.0,
        specular=0.0,
        marker_size=None,
    ):
        """mode: 'surface' | 'wireframe' | 'points' | 'fancymesh'. colours: per-vertex RGB in 0..1. texture: (h, w, 3) uint8."""
        points = _f4(points, 3)
        item = dict(
            kind="mesh",
            points=points,
            trilist=np.ascontiguousarray(trilist, dtype="i4"),
            colour=_rgb(colour),
            mode=mode,
            alpha=float(alpha),
            line_width=float(line_width),
            colours=None if colours is None else _f4(colours, 3),
            tcoords=None if tcoords is None else _f4(tcoords, 2),
            texture=None
            if texture is None
            else np.ascontiguousarray(texture, np.uint8),
            ambient=float(ambient),
            specular=float(specular),
            marker_size=marker_size,
        )
        return self._add(item)

    def add_lines(self, points, edges, colour=(1, 0, 0), width=2.0, alpha=1.0):
        return self._add(
            dict(
                kind="lines",
                points=_f4(points, 3),
                edges=np.ascontiguousarray(edges, dtype="i4").reshape(-1, 2),
                colour=_rgb(colour),
                width=float(width),
                alpha=float(alpha),
            )
        )

    def add_markers(
        self, points, size, colour=(1, 0, 0), style="sphere", alpha=1.0, step=None
    ):
        """size is the sphere's diameter in scene units (Mayavi's scale_factor); style 'point' draws square pixels."""
        points = _f4(points, 3)
        if step:
            points = points[:: int(step)]
        return self._add(
            dict(
                kind="markers",
                points=points,
                size=float(size),
                colour=_rgb(colour),
                style=style,
                alpha=float(alpha),
            )
        )

    def add_vectors(
        self,
        points,
        vectors,
        colour=(1, 0, 0),
        scale=1.0,
        width=2.0,
        alpha=1.0,
        step=None,
    ):
        """Arrows from points along vectors; the arrow's length is |vector| * scale (Mayavi's quiver3d)."""
        points, vectors = _f4(points, 3), _f4(vectors, 3)
        if step:
            points, vectors = points[:: int(step)], vectors[:: int(step)]
        return self._add(
            dict(
                kind="vectors",
                points=points,
                vectors=vectors,
                colour=_rgb(colour),
                scale=float(scale),
                width=float(width),
                alpha=float(alpha),
            )
        )

    def add_labels(self, points, texts, colour=(0, 0, 0), size=None):
        """Text anchored at 3D points, drawn by the client (size is the text height in scene units)."""
        return self._add(
            dict(
                kind="labels",
                points=_f4(points, 3),
                texts=[str(t) for t in texts],
                colour=_rgb(colour),
                size=size,
            )
        )

    # ---- queries
    def bounds(self):
        boxes = [
            (i["points"].min(0), i["points"].max(0))
            for i in self.items
            if len(i.get("points", ()))
        ]
        if not boxes:
            return None
        return np.min([b[0] for b in boxes], 0), np.max([b[1] for b in boxes], 0)

    def ensure_camera(self):
        """Frame new content. The first content is always framed. After a clear, a mesh of the same size in about the same
        place keeps the user's view (browsing through similar shapes), anything else is re-framed (like Mayavi's reset)."""
        if not self.needs_fit:
            return
        box = self.bounds()
        if box is None:
            return
        lo, hi = box
        centre, radius = (lo + hi) / 2, max(float(np.linalg.norm(hi - lo)) / 2, 1e-9)
        cam = self.camera
        if (
            not self.camera_fitted
            or not (0.33 <= radius / cam.radius <= 3)
            or np.linalg.norm(centre - cam.target) > 2 * cam.radius
        ):
            cam.fit(lo, hi)
        self.camera_fitted = True
        self.needs_fit = False

    def fit_camera(self):
        box = self.bounds()
        if box is not None:
            self.camera.fit(*box)
            self.camera_fitted = True
            self.needs_fit = False

    def labels(self):
        """What the browser draws on top of the frame: [{text, x, y, z, colour, size}] in scene units."""
        out = []
        for item in self.items:
            if item["kind"] == "labels":
                size = item["size"] or float(self.camera.radius) * 0.03
                for text, (x, y, z) in zip(item["texts"], item["points"]):
                    out.append(
                        dict(
                            text=text,
                            p=[float(x), float(y), float(z)],
                            colour=item["colour"],
                            size=size,
                        )
                    )
        return out

    def n_triangles(self):
        return int(sum(len(i["trilist"]) for i in self.items if i["kind"] == "mesh"))
