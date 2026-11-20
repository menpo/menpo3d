"""Drawing needs an OpenGL context: skipped where there is none (a CI box without Mesa llvmpipe, say)."""

import numpy as np
import pytest

import menpo3d.io as m3io
from menpo3d.visualize.viewer.scene import Scene


@pytest.fixture(scope="module")
def worker():
    from menpo3d.visualize.viewer.worker import get_worker

    try:
        return get_worker()
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"no OpenGL: {error}")


@pytest.fixture(scope="module")
def bunny():
    return m3io.import_builtin_asset.bunny_obj()


def draw(worker, scene, w=200, h=200):
    scene.ensure_camera()
    return worker.render(scene, scene.camera, w, h)


def covered(rgb, background=(255, 255, 255)):
    return (np.abs(rgb.astype(int) - background).sum(2) > 12).mean()


def test_a_surface_fills_part_of_the_frame_in_its_colour(worker, bunny):
    scene = Scene()
    scene.add_mesh(bunny.points, bunny.trilist, colour=(1, 0, 0))
    rgb, _ = draw(worker, scene)
    assert rgb.shape == (200, 200, 3) and 0.1 < covered(rgb) < 0.5
    ink = rgb[covered_mask(rgb)]
    assert ink[:, 0].mean() > 150 and ink[:, 1].mean() < 40  # red, shaded


def covered_mask(rgb):
    return np.abs(rgb.astype(int) - 255).sum(2) > 12


def test_the_same_scene_draws_the_same_pixels_twice(worker, bunny):
    scene = Scene()
    scene.add_mesh(bunny.points, bunny.trilist)
    a, b = draw(worker, scene)[0].astype(int), draw(worker, scene)[0].astype(int)
    assert (
        np.abs(a - b).max() <= 2
    )  # GPU rasterisation may differ by a level in a few pixels, nothing more


def test_wireframe_covers_less_than_a_surface(worker, bunny):
    surface, wire = Scene(), Scene()
    surface.add_mesh(bunny.points, bunny.trilist)
    wire.add_mesh(bunny.points, bunny.trilist, mode="wireframe")
    big_wire, big_surface = (
        draw(worker, wire, 800, 800)[0],
        draw(worker, surface, 800, 800)[0],
    )
    assert (
        covered(big_wire) < covered(big_surface) * 0.9
    )  # the gaps between the lines show the background


def test_alpha_blends_with_the_background(worker, bunny):
    solid, glass = Scene(), Scene()
    solid.add_mesh(bunny.points, bunny.trilist, colour=(0, 0, 0))
    glass.add_mesh(bunny.points, bunny.trilist, colour=(0, 0, 0), alpha=0.4)
    a, b = draw(worker, solid)[0], draw(worker, glass)[0]
    mask = covered_mask(a)
    assert b[mask].mean() > a[mask].mean() + 60


def test_a_sphere_is_hidden_by_a_surface_in_front_of_it(worker):
    pts = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0.0]])
    scene = Scene()
    scene.add_mesh(pts, [[0, 1, 2]], colour=(0, 0, 1))
    scene.add_markers(
        np.array([[0, -0.2, -0.5]]), 0.4, colour=(1, 0, 0)
    )  # behind the triangle (camera is at +z)
    scene.fit_camera()
    rgb, _ = worker.render(scene, scene.camera, 200, 200)
    assert not ((rgb[..., 0] > 200) & (rgb[..., 2] < 60)).any(), (
        "the sphere shows through the triangle"
    )
    scene2 = Scene()
    scene2.add_mesh(pts, [[0, 1, 2]], colour=(0, 0, 1))
    scene2.add_markers(np.array([[0, -0.2, 0.5]]), 0.4, colour=(1, 0, 0))  # in front
    scene2.fit_camera()
    rgb2, _ = worker.render(scene2, scene2.camera, 200, 200)
    assert ((rgb2[..., 0] > 150) & (rgb2[..., 2] < 80)).any()


def test_vertex_colours_and_textures_are_used(worker, bunny):
    scene = Scene()
    scene.add_mesh(
        bunny.points,
        bunny.trilist,
        colours=np.tile([0.0, 1.0, 0.0], (bunny.n_points, 1)),
    )
    ink = draw(worker, scene)[0][covered_mask(draw(worker, scene)[0])]
    assert ink[:, 1].mean() > 150 and ink[:, 0].mean() < 40
    tex = np.zeros((4, 4, 3), np.uint8)
    tex[..., 2] = 255
    scene = Scene()
    scene.add_mesh(
        bunny.points,
        bunny.trilist,
        tcoords=np.full((bunny.n_points, 2), 0.5),
        texture=tex,
    )
    ink = draw(worker, scene)[0][covered_mask(draw(worker, scene)[0])]
    assert ink[:, 2].mean() > 150 and ink[:, 0].mean() < 40


def test_arrows_and_lines_are_drawn(worker):
    pts = np.array([[0, 0, 0], [1, 1, 0.0]])
    arrows, lines = Scene(), Scene()
    arrows.add_vectors(
        pts, np.array([[0, 0.5, 0], [0.5, 0, 0]]), colour=(0, 0, 0), scale=1.0
    )
    lines.add_lines(pts, [[0, 1]], colour=(0, 0, 0), width=3)
    assert (
        covered(draw(worker, arrows)[0]) > 0.002
        and covered(draw(worker, lines)[0]) > 0.002
    )


def test_pick_hits_the_mesh_and_the_point_reprojects_to_the_pixel(worker, bunny):
    scene = Scene()
    scene.add_mesh(bunny.points, bunny.trilist)
    _, std = draw(worker, scene)
    hit = worker.call(
        lambda r, *a: r.pick(*a), scene, scene.camera, 200, 200, 100.0, 110.0
    )
    assert hit is not None
    clip = std @ np.array([*hit["point"], 1.0])
    px, py = (clip[0] / clip[3] + 1) * 100, (1 - clip[1] / clip[3]) * 100
    assert (px, py) == pytest.approx((100.0, 110.0), abs=0.05)
    assert (
        worker.call(lambda r, *a: r.pick(*a), scene, scene.camera, 200, 200, 2.0, 2.0)
        is None
    )


def test_gpu_buffers_exist_only_while_held(worker, bunny):
    scene = Scene()
    scene.add_mesh(bunny.points, bunny.trilist)
    held = lambda r: scene.id in r.on_gpu()  # noqa: E731
    draw(worker, scene)
    assert not worker.call(held)  # a one-off render gives its buffers back
    worker.call(lambda r: r.acquire(scene))
    worker.call(lambda r: r.acquire(scene))
    assert worker.call(held)
    worker.call(lambda r: r.release(scene))
    assert worker.call(held)
    worker.call(lambda r: r.release(scene))
    assert not worker.call(held)


def test_changing_the_scene_rebuilds_what_is_on_the_gpu(worker, bunny):
    scene = Scene()
    scene.add_mesh(bunny.points, bunny.trilist, colour=(1, 0, 0))
    worker.call(lambda r: r.acquire(scene))
    try:
        before = draw(worker, scene)[0]
        scene.clear()
        scene.add_mesh(bunny.points, bunny.trilist, colour=(0, 0, 1))
        after = draw(worker, scene)[0]
        assert (
            before[covered_mask(before)][:, 0].mean() > 150
            and after[covered_mask(after)][:, 2].mean() > 150
        )
    finally:
        worker.call(lambda r: r.release(scene))


def test_different_sizes_do_not_evict_each_other_beyond_the_cache_limit(worker, bunny):
    scene = Scene()
    scene.add_mesh(bunny.points, bunny.trilist)
    for size in range(100, 100 + 2 * 12, 2):
        draw(worker, scene, size, size)
    assert worker.call(lambda r: len(r._fbos)) <= 6
    worker.call(lambda r: r.release_framebuffers())
    assert worker.call(lambda r: len(r._fbos)) == 0
