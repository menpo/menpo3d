"""The moderngl viewers against Mayavi (the oracle): same meshes, same camera, silhouettes and colours must agree.

Skipped without Mayavi and a Qt binding (the ``vtk-oracle`` dependency group has both)."""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("mayavi")
pytest.importorskip("PySide6")
import menpo3d.io as m3io  # noqa: E402

SIZE = 300

MAYAVI = textwrap.dedent(
    """
    import os, sys, warnings
    warnings.filterwarnings("ignore")
    os.environ["MENPO3D_VIEWER"] = "mayavi"
    import numpy as np
    import menpo3d.io as m3io
    from menpo.shape import ColouredTriMesh
    from mayavi import mlab
    mlab.options.offscreen = True
    out, size = sys.argv[1], int(sys.argv[2])
    bunny = m3io.import_builtin_asset.bunny_obj()
    james = m3io.import_builtin_asset.james_obj()
    colours = (bunny.points - bunny.points.min(0)) / np.ptp(bunny.points, axis=0)
    cases = dict(
        surface=lambda: bunny.view(mesh_type="surface", colour="r", new_figure=True),
        wireframe=lambda: bunny.view(mesh_type="wireframe", colour="b", new_figure=True),
        coloured=lambda: ColouredTriMesh(bunny.points, bunny.trilist, colours=colours).view(new_figure=True),
        textured=lambda: james.view(new_figure=True),
    )
    for name, make in cases.items():
        r = make()
        mlab.view(azimuth=0, elevation=0, distance="auto", focalpoint="auto", figure=r.figure)  # look down -z, y up
        r.figure.scene.camera.view_up = [0, 1, 0]
        r.figure.scene.camera.orthogonalize_view_up()
        r.figure.scene.reset_zoom()
        r.save_figure(f"{out}/{name}.png", size=(size, size), overwrite=True)
    """
)


@pytest.fixture(scope="module")
def mayavi_images(tmp_path_factory):
    out = tmp_path_factory.mktemp("mayavi")
    env = {
        **os.environ,
        "ETS_TOOLKIT": "qt",
        "QT_API": "pyside6",
        "QT_QPA_PLATFORM": "offscreen",
    }
    done = subprocess.run(
        [sys.executable, "-c", MAYAVI, str(out), str(SIZE)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if done.returncode != 0:
        pytest.skip(f"Mayavi could not render offscreen here: {done.stderr[-300:]}")
    return out


@pytest.fixture(scope="module")
def ours():
    from menpo3d.visualize.viewer.worker import get_worker

    try:
        get_worker()
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"no OpenGL: {error}")
    bunny = m3io.import_builtin_asset.bunny_obj()
    james = m3io.import_builtin_asset.james_obj()
    from menpo.shape import ColouredTriMesh

    colours = (bunny.points - bunny.points.min(0)) / np.ptp(bunny.points, axis=0)
    cases = dict(
        surface=bunny.view(mesh_type="surface", colour="r", new_figure=True),
        wireframe=bunny.view(mesh_type="wireframe", colour="b", new_figure=True),
        coloured=ColouredTriMesh(bunny.points, bunny.trilist, colours=colours).view(
            new_figure=True
        ),
        textured=james.view(new_figure=True),
    )
    return {name: r.figure.render((SIZE, SIZE)) for name, r in cases.items()}


def ink(image):
    return np.abs(image.astype(int) - 255).sum(2) > 30


@pytest.mark.parametrize(
    "name,min_iou",
    [("surface", 0.95), ("coloured", 0.95), ("textured", 0.93), ("wireframe", 0.9)],
)
def test_the_silhouette_matches_mayavi(name, min_iou, mayavi_images, ours):
    theirs = np.asarray(Image.open(mayavi_images / f"{name}.png").convert("RGB"))
    a, b = ink(theirs), ink(ours[name])
    assert (a & b).sum() / (a | b).sum() > min_iou


@pytest.mark.parametrize("name", ["surface", "coloured", "textured"])
def test_the_mean_colour_is_close_to_mayavis(name, mayavi_images, ours):
    theirs = np.asarray(
        Image.open(mayavi_images / f"{name}.png").convert("RGB")
    ).astype(float)
    a, b = ink(theirs), ink(ours[name])
    assert np.abs(theirs[a].mean(0) - ours[name][b].astype(float).mean(0)).max() < 14
