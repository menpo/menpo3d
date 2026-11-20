"""The native window (moderngl-window), driven the way the window system drives it and checked on its real framebuffer.
Skipped where moderngl-window is missing or there is no display (a headless box without xvfb)."""

import os
import sys
import threading

import numpy as np
import pytest

pytest.importorskip("moderngl_window")
if sys.platform == "darwin" and os.environ.get("CI"):
    # a macOS CI runner has no GUI session: creating the Cocoa window traps (SIGTRAP) and kills the whole run
    pytest.skip("no window server on a macOS CI runner", allow_module_level=True)
import menpo3d.io as m3io  # noqa: E402
from menpo3d.visualize import viewmoderngl as vm  # noqa: E402


@pytest.fixture(scope="module")
def bunny():
    return m3io.import_builtin_asset.bunny_obj()


def make_window(bunny):
    from menpo3d.visualize.viewer.native import NativeWindow
    from menpo3d.visualize.viewer.worker import get_worker

    try:
        get_worker()
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"no OpenGL: {error}")
    fig = bunny.view(mesh_type="surface", colour="r", new_figure=True).figure
    fig.size = (240, 180)
    try:
        return NativeWindow(fig, visible=False)
    except RuntimeError as error:
        pytest.skip(f"no window here: {error}")


@pytest.fixture
def win(bunny):
    window = make_window(bunny)
    window.draw(swap=False)
    yield window
    window.close()


def test_the_window_shows_exactly_what_the_renderer_draws_the_right_way_up(win):
    assert win.shown is not None
    on_screen = win.read_screen()
    assert on_screen.shape == win.shown.shape
    assert np.abs(on_screen.astype(int) - win.shown.astype(int)).max() <= 1
    reference = win.figure.render(win.size())  # the offline picture
    assert np.abs(win.shown.astype(int) - reference.astype(int)).max() <= 2
    ink = np.abs(win.shown.astype(int) - 255).sum(2) > 30
    assert ink[: ink.shape[0] // 2].any() and ink[ink.shape[0] // 2 :].any()


def test_nothing_is_rerendered_until_something_changes(win):
    assert win.draw(swap=False) is False
    win.invalidate()
    assert win.draw(swap=False) is True


def test_dragging_rotates_and_the_next_frame_differs(win):
    before = win.shown.copy()
    azimuth = win.figure.get_pose()["azimuth"]
    win.win.mouse_press_event_func(
        50, 50, win.win.mouse.left
    )  # through the callbacks moderngl-window calls
    win.win.mouse_drag_event_func(90, 60, 40, 10)
    win.win.mouse_release_event_func(90, 60, win.win.mouse.left)
    assert win.figure.get_pose()["azimuth"] != azimuth
    assert (
        win.draw(swap=False)
        and np.abs(win.shown.astype(int) - before.astype(int)).sum() > 10000
    )


def test_dragging_right_moves_the_picture_the_way_the_mouse_moves(win):
    cam = win.figure.scene.camera
    target = cam.target.copy()
    win.press(50, 50, win.win.mouse.right)
    win.drag(
        80, 50, 30, 0
    )  # the mouse goes right, so the model follows it: the camera target goes left
    win.release(80, 50, win.win.mouse.right)
    view = cam.view_matrix()
    assert (cam.target - target) @ view[
        0, :3
    ] < 0 and cam.azimuth == 0  # moved along the view's x axis, no rotation


def test_shift_drag_pans_scrolling_zooms_and_r_resets(win):
    cam = win.figure.scene.camera
    target, distance = cam.target.copy(), cam.distance
    win.press(50, 50, win.win.mouse.left, shift=True)
    win.drag(80, 50, 30, 0)
    win.release(80, 50, win.win.mouse.left)
    assert np.linalg.norm(cam.target - target) > 0 and cam.azimuth == 0
    win.win.mouse_scroll_event_func(0.0, 2.0)
    assert cam.distance < distance
    win.win.key_event_func(win.win.keys.R, win.win.keys.ACTION_PRESS, win.win.modifiers)
    assert cam.distance == pytest.approx(distance, rel=1e-6) and np.allclose(
        cam.target, target, atol=1e-9
    )


def test_a_click_picks_and_draws_a_crosshair_but_a_drag_does_not(win):
    w, h = win.size()
    k = win.scale()
    x, y = w / 2 / k, h / 2 / k
    win.press(x, y, win.win.mouse.left)
    win.release(x, y, win.win.mouse.left)
    assert win.hit is not None and win.hit["tri"] >= 0 and win.picked is not None
    win.draw(swap=False)
    cx, cy = (int(round(v)) for v in win.picked)
    assert tuple(win.shown[cy, cx - 6]) == (255, 40, 40)
    win.press(x, y, win.win.mouse.left)
    win.drag(x + 30, y, 30, 0)
    win.release(x + 30, y, win.win.mouse.left)
    assert win.picked is None


def test_changing_the_scene_from_another_thread_redraws(win, bunny):
    win.draw(swap=False)
    t = threading.Thread(
        target=lambda: win.figure.scene.add_markers(
            bunny.points[::300], 0.01, colour=(0, 0, 1)
        )
    )
    t.start()
    t.join()
    assert win.dirty
    before = win.shown.copy()
    win.draw(swap=False)
    assert ((win.shown[..., 2] > 150) & (win.shown[..., 0] < 80)).any() and (
        win.shown != before
    ).any()


def test_s_saves_a_png_and_escape_closes(win, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    win.key(win.win.keys.S)
    assert list(tmp_path.glob("*.png"))
    win.key(win.win.keys.ESCAPE)
    assert win.closing


def test_the_loop_runs_until_closed_and_cleans_up(bunny):
    window = make_window(bunny)
    n = len(window.figure.scene.listeners)
    threading.Timer(0.5, lambda: setattr(window, "_stop", window.win.close())).start()
    window.run()  # returns once the window is closed
    assert window.win is None and len(window.figure.scene.listeners) == n - 1


def test_closing_twice_is_fine(bunny):
    window = make_window(bunny)
    window.close()
    window.close()


def test_show_picks_the_browser_when_there_is_no_native_window(bunny, monkeypatch):
    fig = bunny.view(mesh_type="surface", new_figure=True).figure
    calls = []
    import menpo3d.visualize.viewer.native as native
    import menpo3d.visualize.viewer.window as browser

    monkeypatch.setattr(
        native,
        "show",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no display")),
    )
    monkeypatch.setattr(
        browser, "show", lambda figure, block=True: calls.append("browser")
    )
    fig.show()
    assert calls == ["browser"]
    with pytest.raises(RuntimeError, match="no display"):
        fig.show(window="native")
    with pytest.raises(ValueError, match="native.*browser"):
        fig.show(window="x11")


def test_a_missing_backend_is_reported_with_the_reason(bunny):
    from menpo3d.visualize.viewer.native import open_window

    with pytest.raises(RuntimeError, match="could not open a window.*nonesuch"):
        open_window("t", (50, 50), False, backend="nonesuch")
