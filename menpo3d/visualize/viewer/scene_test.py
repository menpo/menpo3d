import numpy as np
import pytest

from menpo3d.visualize.viewer.camera import OrbitCamera
from menpo3d.visualize.viewer.scene import Scene

TRI = np.array([[0, 1, 2]])
PTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)


def test_items_are_added_and_cleared_and_listeners_hear_about_it():
    scene, heard = Scene(), []
    scene.listeners.append(lambda s: heard.append(s.version))
    scene.add_mesh(PTS, TRI)
    scene.add_lines(PTS, [[0, 1]])
    scene.add_markers(PTS, 0.1)
    scene.add_vectors(PTS, PTS)
    scene.add_labels(PTS, "abc")
    assert [i["kind"] for i in scene.items] == [
        "mesh",
        "lines",
        "markers",
        "vectors",
        "labels",
    ]
    scene.clear()
    assert scene.items == () and heard == [1, 2, 3, 4, 5, 6]


def test_the_item_list_is_replaced_not_edited_so_a_render_thread_can_keep_reading_it():
    scene = Scene()
    scene.add_mesh(PTS, TRI)
    before = scene.items
    scene.add_markers(PTS, 0.1)
    assert len(before) == 1 and len(scene.items) == 2


def test_wrong_shapes_are_rejected():
    with pytest.raises(ValueError, match="n, 3"):
        Scene().add_mesh(np.zeros((4, 2)), TRI)


def test_step_keeps_every_nth_marker_and_arrow():
    scene = Scene()
    scene.add_markers(np.zeros((10, 3)), 1, step=3)
    scene.add_vectors(np.zeros((10, 3)), np.ones((10, 3)), step=5)
    assert len(scene.items[0]["points"]) == 4 and len(scene.items[1]["points"]) == 2


def test_colours_accept_names_and_triples():
    scene = Scene()
    assert scene.add_markers(PTS, 1, colour="r")["colour"] == (1.0, 0.0, 0.0)
    assert scene.add_markers(PTS, 1, colour=(0, 0, 1))["colour"] == (0.0, 0.0, 1.0)


def test_bounds_cover_every_item():
    scene = Scene()
    assert scene.bounds() is None
    scene.add_mesh(PTS, TRI)
    scene.add_markers(PTS + 5, 1)
    lo, hi = scene.bounds()
    assert lo.tolist() == [0, 0, 0] and hi.tolist() == [6, 6, 5]


def test_the_first_content_frames_the_camera():
    scene = Scene()
    scene.add_mesh(PTS * 100, TRI)
    scene.ensure_camera()
    assert scene.camera_fitted and scene.camera.radius == pytest.approx(
        np.linalg.norm([100, 100, 0]) / 2
    )


def test_after_a_clear_similar_content_keeps_the_users_view_but_other_content_reframes():
    scene = Scene()
    scene.add_mesh(PTS, TRI)
    scene.ensure_camera()
    scene.camera.azimuth = 1.0
    scene.camera.distance = 7.0  # the user zoomed out
    scene.clear()
    scene.add_mesh(PTS * 1.2, TRI)  # same size, same place: keep the view
    scene.ensure_camera()
    assert scene.camera.distance == 7.0 and scene.camera.azimuth == 1.0
    scene.clear()
    scene.add_mesh(PTS * 500, TRI)  # a different scale: reframe (rotation stays)
    scene.ensure_camera()
    assert scene.camera.distance != 7.0 and scene.camera.azimuth == 1.0


def test_labels_carry_text_position_colour_and_a_default_size():
    scene = Scene()
    scene.add_mesh(PTS, TRI)
    scene.ensure_camera()
    scene.add_labels(PTS[:2], [7, "b"], colour=(1, 0, 0))
    labels = scene.labels()
    assert (
        [l["text"] for l in labels] == ["7", "b"]
        and labels[0]["colour"] == (1, 0, 0)
        and labels[0]["size"] > 0
    )


def test_a_camera_pose_round_trips_and_the_fit_box_is_inside_the_view():
    cam = OrbitCamera()
    lo, hi = np.array([-3.0, 0, 10]), np.array([5.0, 2, 14])
    cam.fit(lo, hi)
    pose = cam.get_pose()
    other = OrbitCamera()
    other.set_pose(**pose)
    assert other.get_pose() == pose
    std = cam.projection_matrix(1.0) @ cam.view_matrix()
    corners = np.array(
        [
            [x, y, z, 1.0]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ]
    )
    clip = corners @ std.T
    ndc = clip[:, :3] / clip[:, 3:]
    assert (np.abs(ndc) <= 1.0001).all()


def test_pan_moves_the_target_in_the_view_plane():
    cam = OrbitCamera(azimuth=0.7, elevation=0.3, distance=4)
    view_before = cam.view_matrix()
    cam.pan(100, 0, 500)
    moved = cam.target
    assert np.linalg.norm(moved) > 0
    assert (
        abs(moved @ view_before[2, :3]) < 1e-9
    )  # no movement towards or away from the camera


def test_orbit_clamps_elevation_and_zoom_is_bounded():
    cam = OrbitCamera()
    cam.orbit(0, 1e6)
    assert cam.elevation <= 1.5
    cam.zoom(1e9)
    assert cam.distance <= cam.radius * 60
