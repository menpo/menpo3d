"""An orbit camera. The whole state is a handful of numbers, so a pose can be saved, sent to a browser and replayed
offline: that is what keeps the live view, the notebook widget and ``save_figure`` pixel-identical."""

import numpy as np


def perspective(fov, aspect, near, far):
    f = 1 / np.tan(np.radians(fov) / 2)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1], m[2, 2], m[2, 3], m[3, 2] = (
        f / aspect,
        f,
        (far + near) / (near - far),
        2 * far * near / (near - far),
        -1,
    )
    return m


def look_at(eye, target, up=(0, 1, 0)):
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = x, y, z
    m[:3, 3] = -m[:3, :3] @ eye
    return m


class OrbitCamera:
    """Looks at ``target`` from ``distance``; y is up, azimuth 0 looks down -z from +z (a face looks back at you)."""

    POSE_KEYS = ("azimuth", "elevation", "distance", "target", "fov")

    def __init__(
        self, azimuth=0.0, elevation=0.0, distance=2.6, target=(0, 0, 0), fov=35.0
    ):
        self.azimuth, self.elevation, self.distance, self.fov = (
            azimuth,
            elevation,
            distance,
            fov,
        )
        self.target = np.asarray(target, float)
        self.radius = 1.0  # size of what is being looked at: sets the clipping planes

    def copy(self):
        c = OrbitCamera(
            self.azimuth, self.elevation, self.distance, self.target.copy(), self.fov
        )
        c.radius = self.radius
        return c

    def get_pose(self):
        return dict(
            azimuth=float(self.azimuth),
            elevation=float(self.elevation),
            distance=float(self.distance),
            target=[float(x) for x in self.target],
            fov=float(self.fov),
            radius=float(self.radius),
        )

    def set_pose(self, azimuth, elevation, distance, target, fov=35.0, radius=None):
        self.azimuth, self.elevation, self.distance = azimuth, elevation, distance
        self.target, self.fov = np.asarray(target, float), fov
        if (
            radius is not None
        ):  # the size of what the pose frames: it sets the clipping planes
            self.radius = radius

    def fit(self, lo, hi):
        """Frame the box ``lo..hi`` (what Mayavi's reset view does): same direction, new target and distance."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        self.target = (lo + hi) / 2
        self.radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-9)
        self.distance = self.radius / np.sin(np.radians(self.fov) / 2)

    def orbit(self, dx, dy):
        self.azimuth -= dx * 0.01
        self.elevation = float(np.clip(self.elevation + dy * 0.01, -1.5, 1.5))

    def zoom(self, dy):
        self.distance = float(
            np.clip(
                self.distance * (1 + 0.001 * dy), self.radius * 0.05, self.radius * 60
            )
        )

    def pan(self, dx, dy, height):
        """Move the target in the view plane by a screen offset in pixels."""
        scale = 2 * self.distance * np.tan(np.radians(self.fov) / 2) / max(height, 1)
        view = self.view_matrix()
        right, up = view[0, :3], view[1, :3]
        self.target = self.target - right * dx * scale + up * dy * scale

    def eye(self):
        e, a = self.elevation, self.azimuth
        return self.target + self.distance * np.array(
            [np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)]
        )

    def view_matrix(self):
        return look_at(self.eye(), self.target)

    def projection_matrix(self, aspect):
        near = max(
            self.distance - 2 * self.radius, self.radius * 1e-3, self.distance * 1e-3
        )
        return perspective(self.fov, aspect, near, self.distance + 2 * self.radius)
