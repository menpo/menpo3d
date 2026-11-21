"""The moderngl viewers behave like the Mayavi ones from menpo's point of view: same classes, same options."""

import os
import subprocess
import sys
from collections import OrderedDict

import numpy as np
import pytest
from PIL import Image

import menpo3d.io as m3io
from menpo.shape import LabelledPointUndirectedGraph, PointCloud
from menpo3d.visualize import viewmoderngl as vm


@pytest.fixture(scope="module", autouse=True)
def gl():
    from menpo3d.visualize.viewer.worker import get_worker

    try:
        get_worker()
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"no OpenGL: {error}")


@pytest.fixture(scope="module")
def bunny():
    return m3io.import_builtin_asset.bunny_obj()


def kinds(figure):
    return [i["kind"] for i in figure.scene.items]


def test_the_viewers_are_moderngl_and_the_old_mayavi_switch_only_warns():
    import menpo3d.visualize as viz

    assert (
        viz.TriMeshViewer3d is vm.ModernGLTriMeshViewer3d
        and viz.Renderer3d is vm.ModernGLRenderer
        and viz.SurfaceViewer3d is vm.ModernGLSurfaceViewer3d
    )
    code = (
        "import warnings; warnings.simplefilter('always');"
        "import menpo3d.visualize as v; print(v.TriMeshViewer3d.__name__)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "MENPO3D_VIEWER": "mayavi"},
        capture_output=True,
        text=True,
    )
    assert out.stdout.split() == ["ModernGLTriMeshViewer3d"], out.stderr[-300:]
    assert "removed in menpo3d 0.10" in out.stderr


def test_figures_are_selected_like_mlab_figure(bunny):
    a = bunny.view(mesh_type="surface", new_figure=True).figure
    b = bunny.view(mesh_type="surface", new_figure=False).figure  # the current figure
    assert a is b and kinds(a) == ["mesh", "mesh"]
    c = bunny.view(mesh_type="surface", new_figure=True).figure
    assert c is not a and kinds(c) == ["mesh"]
    again = bunny.view(figure_id=a.name, new_figure=False).figure
    assert again is a
    with pytest.raises(ValueError, match="figure_id cannot be"):
        bunny.view(figure_id="x", new_figure=True)


def test_clear_figure_empties_the_scene_and_the_viewer_can_draw_again(bunny):
    r = bunny.view(mesh_type="surface", new_figure=True)
    r.clear_figure()
    assert r.figure.scene.items == ()
    bunny.view(figure_id=r.figure_id, new_figure=False)
    assert kinds(r.figure) == ["mesh"]


def test_every_viewer_type_adds_what_mayavi_would_draw(bunny):
    pts = bunny.points[::500]
    fig = PointCloud(pts).view(new_figure=True, render_numbering=True).figure
    assert kinds(fig) == ["markers", "labels"]
    n = len(pts)
    graph = vm.ModernGLPointGraphViewer3d(
        None, True, pts, np.array([[i, (i + 1) % n] for i in range(n)])
    ).render()
    assert kinds(graph.figure) == ["lines", "markers"]
    normals = bunny.view(
        mesh_type="surface", new_figure=True, normals=bunny.vertex_normals(), step=100
    ).figure
    assert kinds(normals) == ["vectors", "mesh"] and len(
        normals.scene.items[0]["points"]
    ) == len(range(0, bunny.n_points, 100))
    coloured = vm.ModernGLColouredTriMeshViewer3d(
        None, True, bunny.points, bunny.trilist, np.random.rand(bunny.n_points, 3)
    ).render()
    assert coloured.scene.items[0]["colours"].shape == (bunny.n_points, 3)
    textured_asset = m3io.import_builtin_asset.james_obj()
    textured = textured_asset.view(new_figure=True).figure
    assert (
        textured.scene.items[0]["texture"] is not None
        and textured.scene.items[0]["tcoords"] is not None
    )


def test_landmarks_draw_each_label_with_its_own_colour(bunny):
    pts = bunny.points[::900]
    n = len(pts)
    mask = np.zeros(n, bool)
    mask[: n // 2] = True
    group = LabelledPointUndirectedGraph.init_from_edges(
        pts, np.array([[i, i + 1] for i in range(n - 1)]), OrderedDict(a=mask, b=~mask)
    )
    fig = group.view(new_figure=True).figure
    markers = [i for i in fig.scene.items if i["kind"] == "markers"]
    assert len(markers) == 2 and markers[0]["colour"] != markers[1]["colour"]
    with pytest.raises(ValueError, match="list of marker colours"):
        group.view(new_figure=True, marker_colour=["r", "g", "b"])


def test_save_figure_writes_an_image_of_the_requested_size_and_respects_overwrite(
    bunny, tmp_path
):
    r = bunny.view(mesh_type="surface", colour="r", new_figure=True)
    path = tmp_path / "bunny.png"
    r.save_figure(str(path), size=(120, 90), overwrite=True)
    assert Image.open(path).size == (120, 90)
    with pytest.raises(ValueError, match="overwrite"):
        r.save_figure(str(path), size=(120, 90))
    r.save_figure(
        str(tmp_path / "big.jpg"),
        format="jpg",
        size=(50, 40),
        magnification=2.0,
        overwrite=True,
    )
    assert Image.open(tmp_path / "big.jpg").size == (100, 80)


def test_numbering_is_drawn_into_the_saved_image(bunny, tmp_path):
    pts = bunny.points[::700]
    plain = PointCloud(pts).view(new_figure=True, marker_size=0.002)
    numbered = PointCloud(pts).view(
        new_figure=True,
        marker_size=0.002,
        render_numbering=True,
        numbers_size=0.01,
        numbers_colour="k",
    )
    a, b = plain.figure.render((300, 300)), numbered.figure.render((300, 300))
    assert (np.abs(a.astype(int) - b.astype(int)).sum(2) > 100).sum() > 50


def test_renderer_settings_describe_the_camera(bunny):
    r = bunny.view(mesh_type="surface", new_figure=True)
    r.figure.render()
    s = r.renderer_settings
    assert (
        (s["width"], s["height"]) == r.figure.size
        and s["view_matrix"].shape == (4, 4)
        and s["projection_matrix"].shape == (4, 4)
    )
    eye_space = s["view_matrix"] @ np.append(r.figure.scene.camera.target, 1.0)
    assert eye_space[2] == pytest.approx(
        -r.figure.scene.camera.distance, rel=1e-4
    )  # the target is straight ahead


def test_a_pose_set_from_python_is_what_gets_drawn(bunny):
    r = bunny.view(mesh_type="surface", new_figure=True)
    front = r.figure.render((100, 100))
    pose = r.figure.get_pose()
    r.figure.set_pose(**{**pose, "azimuth": np.pi / 2})
    side = r.figure.render((100, 100))
    assert np.abs(front.astype(int) - side.astype(int)).sum() > 10000
    r.figure.set_pose(**pose)
    assert (
        np.abs(front.astype(int) - r.figure.render((100, 100)).astype(int)).max() <= 2
    )


def test_a_saved_pose_survives_clearing_and_showing_a_differently_sized_mesh(bunny):
    from menpo.shape import TriMesh

    big = TriMesh(bunny.points * 10 + 50, bunny.trilist)  # re-framed on its own
    r = bunny.view(mesh_type="surface", new_figure=True)
    fig = r.figure
    fig.set_pose(
        **{
            **fig.get_pose(),
            "azimuth": 1.0,
            "distance": fig.get_pose()["distance"] * 0.7,
        }
    )
    mine, pose = fig.render((100, 100)), fig.get_pose()
    fig.scene.clear()
    big.view(figure_id=fig.name, new_figure=False, mesh_type="surface")
    fig.reset_camera()
    assert fig.get_pose()["radius"] == pytest.approx(
        bunny.range().max() * 10 * 0.866, rel=0.5
    )
    assert fig.get_pose()["distance"] > pose["distance"] * 1.5
    fig.scene.clear()
    bunny.view(figure_id=fig.name, new_figure=False, mesh_type="surface")
    fig.set_pose(**pose)  # no re-framing on the next render
    assert fig.get_pose() == pose
    assert np.abs(mine.astype(int) - fig.render((100, 100)).astype(int)).max() <= 2


def test_a_figure_shown_without_the_streaming_extras_falls_back_to_a_png(
    bunny, monkeypatch
):
    fig = bunny.view(mesh_type="surface", new_figure=True).figure
    monkeypatch.setitem(
        sys.modules, "scene_streamer.widget", None
    )  # import fails like a missing anywidget
    with pytest.warns(UserWarning, match="still picture"):
        bundle = fig._repr_mimebundle_()
    assert list(bundle) == ["image/png"]


def test_show_and_the_widget_say_which_extra_to_install(bunny, monkeypatch):
    fig = bunny.view(mesh_type="surface", new_figure=True).figure
    monkeypatch.setitem(sys.modules, "scene_streamer", None)
    with pytest.raises(ImportError, match=r"menpo3d\[viz_notebook\]"):
        fig.show(
            window="browser"
        )  # (the default would open a native window and wait for it)
    with pytest.warns(UserWarning, match=r"menpo3d\[viz_notebook\]"):
        fig._repr_mimebundle_()


# ---- scalar colours, colour bar, statistics text, the height-field surface
def _bump(n=40):
    x, y = np.meshgrid(np.linspace(-2, 2, n), np.linspace(-2, 2, n), indexing="ij")
    return np.exp(-(x**2 + y**2))


def test_scalars_to_colours_maps_the_range_and_greys_out_nan():
    from menpo3d.visualize.viewer.colormap import scalars_to_colours

    rgb, lo, hi = scalars_to_colours([0.0, 0.5, 1.0, np.nan], "gray")
    assert (lo, hi) == (0.0, 1.0)
    assert rgb[0] == pytest.approx([0, 0, 0], abs=0.01)
    assert rgb[2] == pytest.approx([1, 1, 1], abs=0.01)
    assert rgb[1] == pytest.approx([0.5, 0.5, 0.5], abs=0.01)
    assert rgb[3] == pytest.approx([0.6, 0.6, 0.6])
    rgb, lo, hi = scalars_to_colours(
        [3.0, 3.0], "gray"
    )  # constant: the middle, no division by zero
    assert rgb[0] == pytest.approx([0.5, 0.5, 0.5], abs=0.01)
    with pytest.raises(ValueError, match="unknown colormap"):
        scalars_to_colours([0, 1], "nope")


def test_statistics_lines_ignore_nan():
    from menpo3d.visualize.viewer.colormap import statistics_lines

    assert statistics_lines([1.0, 3.0, np.nan]) == ["mean 2", "std 1", "max 3"]
    assert statistics_lines([np.nan]) == []


def test_the_surface_has_mayavis_auto_warp_and_drops_masked_cells():
    values = _bump(20)
    mask = np.ones_like(values, bool)
    mask[:10] = False
    r = vm.ModernGLSurfaceViewer3d(None, True, values, mask).render()
    (mesh,) = [i for i in r.figure.scene.items if i["kind"] == "mesh"]
    z = mesh["points"][mesh["trilist"].ravel(), 2]
    assert np.ptp(z) == pytest.approx(
        0.3 * (19 + 19), rel=1e-3
    )  # mlab.surf(warp_scale="auto")
    assert (
        mesh["points"][mesh["trilist"].ravel(), 0].min() >= 9
    )  # nothing drawn in the masked rows
    assert len(mesh["trilist"]) == 2 * 9 * 19
    flat = vm.ModernGLSurfaceViewer3d(None, True, values).render(warp_scale=2.0)
    (mesh,) = [i for i in flat.figure.scene.items if i["kind"] == "mesh"]
    assert np.ptp(mesh["points"][:, 2]) == pytest.approx(2.0 * np.ptp(values), rel=1e-4)


def test_a_colour_bar_and_statistics_are_in_every_render():
    base = vm.ModernGLSurfaceViewer3d(None, True, _bump()).render()
    plain = base.figure.render((400, 300))
    barred = vm.ModernGLSurfaceViewer3d(None, True, _bump()).render(
        colorbar=True, statistics=["mean", "max"]
    )
    picture = barred.figure.render((400, 300))
    changed = np.abs(picture.astype(int) - plain.astype(int)).sum(2) > 30
    ys, xs = np.nonzero(changed)
    assert xs.max() > 300  # the bar sits at the right edge
    assert (xs < 100).any() and (ys < 60).any()  # the statistics sit top left
    # the same pixels reach the saved image
    barred.figure.save("/tmp/_bar.png")
    assert np.array_equal(
        np.asarray(Image.open("/tmp/_bar.png").convert("RGB")), barred.figure.render()
    )
    os.remove("/tmp/_bar.png")


@pytest.mark.parametrize("position", ["right", "left", "top", "bottom"])
def test_a_colour_bar_ramps_through_its_colormap(position):
    r = vm.ModernGLSurfaceViewer3d(None, True, _bump()).render()
    r.figure.scene.clear()  # only the bar on a white background
    r.add_colorbar("gray", 0, 1, position=position, length=0.6, thickness=0.06)
    r.scene.add_mesh(
        np.zeros((3, 3)), np.zeros((0, 3), int)
    )  # keep a (degenerate) mesh so a frame renders
    picture = r.figure.render((400, 400)).astype(int)
    grey = (np.abs(picture[..., 0] - picture[..., 1]) < 3) & (picture[..., 0] < 250)
    assert grey.sum() > 400
    ys, xs = np.nonzero(grey)
    along = ys if position in ("right", "left") else xs
    assert np.ptp(along) > 0.4 * 400  # it spans most of the requested length


def test_a_bad_colour_bar_position_is_an_error():
    r = vm.ModernGLSurfaceViewer3d(None, True, _bump())
    with pytest.raises(ValueError, match="position"):
        r.add_colorbar("jet", 0, 1, position="middle")


# ---- menpo calls the viewers with None for "no colour chosen" (view_landmarks and others)
def _colour(item):
    return tuple(float(c) for c in item["colour"])


def test_a_colour_of_none_means_the_default_one():
    assert vm._parse_colour(None) == (1.0, 0.0, 0.0)  # red, as the viewers' own default
    assert vm._parse_colour(None, "k") == (0.0, 0.0, 0.0)
    assert vm._parse_colour("g") == (0.0, 0.5, 0.0)  # a colour that was chosen is kept
    assert vm._parse_colour((0.2, 0.4, 0.6)) == (0.2, 0.4, 0.6)


def test_the_landmarks_of_a_mesh_draw_with_the_default_colour(bunny):
    # view_landmarks passes marker_colour=None and line_colour=None: this used to raise
    # ValueError: Invalid RGBA argument: None
    mesh = bunny.copy()
    mesh.landmarks["tips"] = PointCloud(bunny.points[:4])
    fig = mesh.view_landmarks(group="tips", new_figure=True).figure
    markers = [i for i in fig.scene.items if i["kind"] == "markers"]
    assert len(markers) == 1 and _colour(markers[0]) == (1.0, 0.0, 0.0)
    assert len(markers[0]["points"]) == 4


def test_the_landmarks_of_a_mesh_still_take_a_colour_that_is_given(bunny):
    mesh = bunny.copy()
    mesh.landmarks["tips"] = PointCloud(bunny.points[:4])
    fig = mesh.view_landmarks(group="tips", new_figure=True, marker_colour="g").figure
    (markers,) = [i for i in fig.scene.items if i["kind"] == "markers"]
    assert _colour(markers) == (0.0, 0.5, 0.0)


def test_a_point_cloud_with_landmarks_draws_with_the_default_colour(bunny):
    cloud = PointCloud(bunny.points[::300])
    cloud.landmarks["some"] = PointCloud(bunny.points[:3])
    fig = cloud.view_landmarks(group="some", new_figure=True).figure
    # the cloud itself, then its landmarks
    clouds, landmarks = [i for i in fig.scene.items if i["kind"] == "markers"]
    assert len(clouds["points"]) == len(cloud.points) and len(landmarks["points"]) == 3
    assert _colour(landmarks) == (1.0, 0.0, 0.0)


def test_numbers_and_normals_with_no_colour_chosen_use_black(bunny):
    pts = bunny.points[::500]
    fig = (
        PointCloud(pts)
        .view(new_figure=True, render_numbering=True, numbers_colour=None)
        .figure
    )
    (labels,) = [i for i in fig.scene.items if i["kind"] == "labels"]
    assert _colour(labels) == (0.0, 0.0, 0.0)
    fig = bunny.view(
        mesh_type="surface",
        new_figure=True,
        normals=bunny.vertex_normals(),
        normals_colour=None,
        step=100,
    ).figure
    (vectors,) = [i for i in fig.scene.items if i["kind"] == "vectors"]
    assert _colour(vectors) == (0.0, 0.0, 0.0)


def test_a_mesh_with_no_colour_chosen_is_red(bunny):
    fig = bunny.view(mesh_type="surface", new_figure=True, colour=None).figure
    (mesh,) = [i for i in fig.scene.items if i["kind"] == "mesh"]
    assert _colour(mesh) == (1.0, 0.0, 0.0)
