r"""Closest point on a triangle mesh using only NumPy and SciPy."""

import numpy as np
from scipy.spatial import cKDTree


def closest_points_on_triangles(p, a, b, c):
    r"""The closest point to each ``p[i]`` on the triangle ``(a[i], b[i], c[i])``.

    Follows Ericson, *Real-Time Collision Detection* 5.1.5, vectorised with masks
    instead of branches. All arguments are ``(n, 3)``.
    """
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4

    out = np.empty_like(p)
    done = np.zeros(len(p), bool)

    def put(mask, value):
        mask = mask & ~done
        out[mask] = value[mask]
        done[mask] = True

    put((d1 <= 0) & (d2 <= 0), a)  # vertex region a
    put((d3 >= 0) & (d4 <= d3), b)  # vertex region b
    put((d6 >= 0) & (d5 <= d6), c)  # vertex region c
    with np.errstate(divide="ignore", invalid="ignore"):
        v = d1 / (d1 - d3)
        put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v[:, None] * ab)  # edge ab
        w = d2 / (d2 - d6)
        put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w[:, None] * ac)  # edge ac
        u = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + u[:, None] * (c - b))
        denom = 1.0 / (va + vb + vc)
        v, w = vb * denom, vc * denom
    put(np.ones(len(p), bool), a + ab * v[:, None] + ac * w[:, None])  # inside
    return out


class ClosestPointLocator:
    r"""A callable that finds the closest point on a triangle mesh (exactly).

    A KD-tree over the triangle centroids provides candidates and a radius bound
    makes the search exact.

    Parameters
    ----------
    points : ``(n_points, 3)`` `ndarray`
    trilist : ``(n_tris, 3)`` `ndarray`
    """

    def __init__(self, points, trilist, n_nearest=8):
        points = np.asarray(points, dtype=np.float64)
        trilist = np.asarray(trilist)
        self.a, self.b, self.c = (points[trilist[:, i]] for i in range(3))
        centroids = (self.a + self.b + self.c) / 3
        self.radius = np.maximum.reduce(
            [np.linalg.norm(v - centroids, axis=1) for v in (self.a, self.b, self.c)]
        )
        self.max_radius = self.radius.max()
        self.n_nearest = min(n_nearest, len(trilist))
        self.tree = cKDTree(centroids)

    @classmethod
    def from_trimesh(cls, trimesh):
        if trimesh.n_dims != 3:
            raise ValueError("Closest point queries only work on 3D TriMesh instances")
        return cls(trimesh.points, trimesh.trilist)

    def __call__(self, points):
        r"""Return the nearest points on the mesh and the index of the nearest triangle.

        Parameters
        ----------
        points : ``(n_points, 3)`` `ndarray`
            Query points

        Returns
        -------
        `nearest_points`, `tri_indices` : ``(n_points, 3)`` `ndarray`, ``(n_points,)`` `ndarray`
        """
        q = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if len(q) == 0:
            return np.empty((0, 3)), np.empty(0, dtype=np.int64)
        k = self.n_nearest
        # 1. an upper bound on the distance, from the k nearest centroids
        _, near = self.tree.query(q, k=k)
        near = near.reshape(len(q), k)
        best = np.full(len(q), np.inf)
        for j in range(k):
            t = near[:, j]
            nearest = closest_points_on_triangles(q, self.a[t], self.b[t], self.c[t])
            best = np.minimum(best, np.linalg.norm(nearest - q, axis=1))
        # 2. every triangle that can be closer has its centroid within best + its radius
        ball = self.tree.query_ball_point(q, best + self.max_radius * 1.0000001)
        lens = np.fromiter((len(c) for c in ball), int, len(q))
        qi = np.repeat(np.arange(len(q)), lens)
        t = np.concatenate([np.asarray(c, dtype=int) for c in ball])
        keep = (
            np.linalg.norm(self.tree.data[t] - q[qi], axis=1)
            <= best[qi] + self.radius[t] + 1e-12
        )
        qi, t = qi[keep], t[keep]
        nearest = closest_points_on_triangles(q[qi], self.a[t], self.b[t], self.c[t])
        d = np.linalg.norm(nearest - q[qi], axis=1)
        order = np.lexsort((d, qi))  # by query, then distance: the first of each wins
        first = order[np.searchsorted(qi[order], np.arange(len(q)))]
        return nearest[first], t[first]
