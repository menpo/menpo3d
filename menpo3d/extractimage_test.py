import numpy as np
import pytest

import menpo3d.io as m3io
from menpo.shape import TriMesh
from menpo.transform import Rotation
from menpo3d.extractimage import per_vertex_occlusion_accurate


def _grid(n, rng, layers=1, gap=0.5):
    """n x n height field(s), jittered so that vertices are shared by several triangles."""
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    base = np.stack([xs.ravel(), ys.ravel(), rng.rand(n * n)], axis=1)
    index = np.arange(n * n).reshape(n, n)
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
    tris = np.concatenate(
        [
            np.stack([a.ravel(), b.ravel(), c.ravel()], 1),
            np.stack([a.ravel(), c.ravel(), d.ravel()], 1),
        ]
    )
    points, trilist = [], []
    for layer in range(layers):
        lifted = base + [0, 0, layer * (1 + gap)]
        lifted[:, :2] += rng.rand(n * n, 2) * 0.3  # layers do not share (x, y)
        points.append(lifted)
        trilist.append(tris + layer * n * n)
    return TriMesh(np.concatenate(points), np.concatenate(trilist))


def _reference(mesh):
    """Independent definition: the plane of each triangle gives z at (x, y); edge functions say whether (x, y) is inside."""
    tol = mesh.mean_edge_length() / 1000
    P, T = mesh.points, mesh.trilist
    visible = np.zeros(mesh.n_points, bool)
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    normal = np.cross(b - a, c - a)
    for i, p in enumerate(P):
        edge = lambda u, v: (
            (v[:, 0] - u[:, 0]) * (p[1] - u[:, 1])
            - (v[:, 1] - u[:, 1]) * (p[0] - u[:, 0])
        )  # noqa: E731
        e0, e1, e2 = edge(a, b), edge(b, c), edge(c, a)
        eps = 1e-9
        inside = ((e0 >= -eps) & (e1 >= -eps) & (e2 >= -eps)) | (
            (e0 <= eps) & (e1 <= eps) & (e2 <= eps)
        )
        ok = inside & (np.abs(normal[:, 2]) > 1e-12)
        z = (
            a[ok, 2]
            - (normal[ok, 0] * (p[0] - a[ok, 0]) + normal[ok, 1] * (p[1] - a[ok, 1]))
            / normal[ok, 2]
        )
        visible[i] = bool(len(z)) and p[2] - z.min() < tol
    return visible


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_occlusion_matches_an_independent_reference(layers):
    mesh = _grid(8, np.random.RandomState(layers), layers=layers)
    got = per_vertex_occlusion_accurate(mesh)
    assert np.array_equal(got, _reference(mesh))
    if layers == 1:
        assert got.all()  # a height field has nothing in front of anything
    else:
        assert (
            0 < got.sum() < mesh.n_points
        )  # the upper layers hide parts of the lower ones


def test_occlusion_follows_the_viewing_direction():
    mesh = _grid(8, np.random.RandomState(0), layers=2)
    q = Rotation(np.diag([1.0, -1.0, -1.0])).apply(mesh)  # looking from the other side
    assert not np.array_equal(
        per_vertex_occlusion_accurate(mesh), per_vertex_occlusion_accurate(q)
    )
    assert np.array_equal(per_vertex_occlusion_accurate(q), _reference(q))


@pytest.mark.parametrize("chunk_size", [1, 7, 100, 10**6])
def test_occlusion_does_not_depend_on_the_chunk_size(chunk_size):
    mesh = _grid(9, np.random.RandomState(5), layers=2)
    assert np.array_equal(
        per_vertex_occlusion_accurate(mesh, chunk_size=chunk_size),
        per_vertex_occlusion_accurate(mesh),
    )


def test_a_triangle_with_no_area_from_above_hides_nothing_and_does_not_crash():
    # a vertical wall (zero area in xy) and a sliver with collinear xy sit between the other points
    points = np.array(
        [
            [0, 0, 0],
            [4, 0, 0],
            [0, 4, 0],  # floor
            [1, 1, 1],
            [1, 1, 2],
            [2, 2, 2],  # wall: (1, 1) is a line in xy -> no area
            [3, 0.5, 5],
            [3, 0.5, 6],
            [3, 0.5, 7],
        ],  # collinear in 3D and in xy
        dtype=float,
    )
    mesh = TriMesh(points, np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    visible = per_vertex_occlusion_accurate(mesh)
    assert visible[:3].all() and visible.shape == (9,)


def test_the_bunny_visible_count_is_stable():
    mesh = m3io.import_builtin_asset.bunny_obj()
    visible = per_vertex_occlusion_accurate(mesh)
    assert visible.sum() == 1176 and mesh.n_points == 2503
    assert np.array_equal(visible, per_vertex_occlusion_accurate(mesh, chunk_size=777))
