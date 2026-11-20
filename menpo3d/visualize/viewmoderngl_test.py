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


def test_the_default_backend_is_moderngl_and_mayavi_stays_selectable():
    import menpo3d.visualize as viz

    assert (
        viz.TriMeshViewer3d is vm.ModernGLTriMeshViewer3d
        and viz.Renderer3d is vm.ModernGLRenderer
    )
    code = "import menpo3d.visualize as v; print(v.TriMeshViewer3d.__name__, v.Renderer3d.__name__)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "MENPO3D_VIEWER": "mayavi"},
        capture_output=True,
        text=True,
    )
    assert out.stdout.split() == ["MayaviTriMeshViewer3d", "MayaviRenderer"], (
        out.stderr[-300:]
    )


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


def test_a_figure_shown_without_the_streaming_extras_falls_back_to_a_png(
    bunny, monkeypatch
):
    fig = bunny.view(mesh_type="surface", new_figure=True).figure
    monkeypatch.setitem(
        sys.modules, "menpo3d.visualize.viewer.widget", None
    )  # import fails like a missing anywidget
    with pytest.warns(UserWarning, match="still picture"):
        bundle = fig._repr_mimebundle_()
    assert list(bundle) == ["image/png"]


def test_show_and_the_widget_say_which_extra_to_install(bunny, monkeypatch):
    fig = bunny.view(mesh_type="surface", new_figure=True).figure
    monkeypatch.setitem(sys.modules, "aiohttp", None)
    with pytest.raises(ImportError, match=r"menpo3d\[viewer\]"):
        fig.show(
            window="browser"
        )  # (the default would open a native window and wait for it)
    with pytest.warns(UserWarning, match=r"menpo3d\[viewer\]"):
        fig._repr_mimebundle_()
