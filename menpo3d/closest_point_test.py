import numpy as np
from numpy.testing import assert_allclose
from pytest import raises

import menpo3d
from menpo.shape import PointCloud, TriMesh
from menpo3d.closest_point import ClosestPointLocator, closest_points_on_triangles
from menpo3d.extractimage import per_vertex_occlusion_accurate


def _brute_force(points, trilist, q, n_samples=200):
    """Closest point by dense barycentric sampling of every triangle (slow, simple)."""
    a, b, c = (points[trilist[:, i]] for i in range(3))
    u, v = np.meshgrid(np.linspace(0, 1, n_samples), np.linspace(0, 1, n_samples))
    ok = (u + v) <= 1
    u, v = u[ok], v[ok]
    samples = (
        a[:, None]
        + u[None, :, None] * (b - a)[:, None]
        + v[None, :, None] * (c - a)[:, None]
    )  # (n_tris, n_samples, 3)
    d = np.linalg.norm(samples[None] - q[:, None, None], axis=-1)
    return d.reshape(len(q), -1).min(axis=1)


def test_closest_point_matches_brute_force_distance():
    rng = np.random.RandomState(0)
    points = rng.rand(30, 3)
    trilist = rng.randint(0, 30, (25, 3))
    trilist = trilist[
        (trilist[:, 0] != trilist[:, 1]) & (trilist[:, 1] != trilist[:, 2])
    ]
    q = rng.randn(40, 3)
    nearest, tri = ClosestPointLocator(points, trilist)(q)
    expected = _brute_force(points, trilist, q)
    got = np.linalg.norm(nearest - q, axis=1)
    # the dense sampling can only be farther than the exact answer, by at most its step
    assert np.all(got <= expected + 1e-9)
    assert_allclose(got, expected, atol=0.02)
    # the returned point lies on the returned triangle
    on_tri = closest_points_on_triangles(
        nearest, *(points[trilist[tri, i]] for i in range(3))
    )
    assert_allclose(on_tri, nearest, atol=1e-9)


def test_closest_point_on_a_vertex_returns_the_vertex():
    mesh = menpo3d.io.import_builtin_asset.bunny_obj()
    nearest, tri = ClosestPointLocator.from_trimesh(mesh)(mesh.points[:50])
    assert_allclose(nearest, mesh.points[:50], atol=1e-12)
    assert all(i in mesh.trilist[t] for i, t in enumerate(tri))


def test_closest_point_fails_on_2d_mesh():
    with raises(ValueError):
        ClosestPointLocator.from_trimesh(TriMesh(np.random.random((5, 2))))


def test_snap_pointcloud_to_surface():
    mesh = menpo3d.io.import_builtin_asset.bunny_obj()
    snapped, tri = mesh.snap_pointcloud_to_surface(PointCloud(mesh.points[:10] + 1e-4))
    assert snapped.n_points == 10 and tri.shape == (10,)


def test_occlusion_accurate_on_two_stacked_triangles():
    # a small triangle sits above (larger z) a big one: the vertices of the big one under
    # the small one are hidden, the rest are visible
    points = np.array(
        [
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0],  # big triangle at z=0
            [1, 1, 1],
            [2, 1, 1],
            [1, 2, 1],  # small triangle at z=1, over (1..2, 1..2)
        ],
        dtype=float,
    )
    trilist = np.array([[0, 1, 2], [3, 4, 5]])
    mesh = TriMesh(points, trilist)
    visible = per_vertex_occlusion_accurate(mesh)
    # looking along +z from below: the big triangle at z=0 is hit first for the small
    # triangle's vertices, so those are hidden; the big triangle's vertices are visible
    assert visible.tolist() == [True, True, True, False, False, False]


def test_barycentric_rebuild_returns_same_as_snapped_points():
    mesh = menpo3d.io.import_builtin_asset.bunny_obj()
    lms = mesh.landmarks[None]
    bc = mesh.barycentric_coordinates_of_pointcloud(lms)
    recon_lms = mesh.project_barycentric_coordinates(*bc)
    direct_recon_lms = mesh.snap_pointcloud_to_surface(lms)[0]
    assert_allclose(recon_lms.points, direct_recon_lms.points)
