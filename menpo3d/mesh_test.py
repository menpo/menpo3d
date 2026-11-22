import warnings

import numpy as np
import pytest
from menpo.image import Image
from menpo.shape import ColouredTriMesh, PointCloud, TexturedTriMesh, TriMesh

import menpo3d.io as m3io
from menpo3d.mesh import merge_duplicate_points

# a square of two triangles, written the way a per-face OBJ is read: every corner
# is its own point (6 points for 4 positions)
CORNERS = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float
)
TRILIST = np.arange(6).reshape(2, 3)
TCOORDS = np.array([[0, 0], [1, 0], [0, 1], [1, 0], [1, 1], [0, 1]], dtype=np.float32)


def test_merging_a_trimesh_gives_the_four_positions_in_order_of_first_appearance():
    merged = merge_duplicate_points(TriMesh(CORNERS, TRILIST))
    assert type(merged) is TriMesh
    np.testing.assert_array_equal(
        merged.points, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    )
    np.testing.assert_array_equal(merged.trilist, [[0, 1, 2], [1, 3, 2]])


def test_the_surface_is_the_same_after_merging():
    mesh = TriMesh(CORNERS, TRILIST)
    merged = merge_duplicate_points(mesh)
    np.testing.assert_array_equal(
        merged.points[merged.trilist], mesh.points[mesh.trilist]
    )


def test_the_mesh_is_not_changed_and_the_result_shares_nothing_with_it():
    mesh = TriMesh(CORNERS.copy(), TRILIST.copy())
    merged = merge_duplicate_points(mesh)
    np.testing.assert_array_equal(mesh.points, CORNERS)
    np.testing.assert_array_equal(mesh.trilist, TRILIST)
    assert not np.shares_memory(merged.points, mesh.points)


def test_a_textured_mesh_keeps_its_seams():
    # the shared corner (1, 0, 0) has the same texture coordinate: merged. A
    # position with two different texture coordinates is not.
    tcoords = TCOORDS.copy()
    tcoords[5] = [0.5, 0.5]  # the second use of (0, 1, 0) differs from the first
    texture = Image.init_blank((4, 4), n_channels=3)
    mesh = TexturedTriMesh(CORNERS, tcoords, texture, TRILIST)
    merged = merge_duplicate_points(mesh)
    assert type(merged) is TexturedTriMesh
    assert merged.n_points == 5  # (1, 0, 0) merged, (0, 1, 0) kept twice
    # every corner still has the position and the texture coordinate it had
    np.testing.assert_array_equal(
        merged.points[merged.trilist], mesh.points[mesh.trilist]
    )
    np.testing.assert_array_equal(
        merged.tcoords.points[merged.trilist], mesh.tcoords.points[mesh.trilist]
    )
    assert merged.tcoords.points.dtype == np.float32
    assert merged.texture is not mesh.texture
    np.testing.assert_array_equal(merged.texture.pixels, mesh.texture.pixels)


def test_a_coloured_mesh_keeps_its_colours():
    colours = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]], float
    )
    colours[5] = [0.2, 0.2, 0.2]  # same position as corner 2, another colour
    mesh = ColouredTriMesh(CORNERS, TRILIST, colours)
    merged = merge_duplicate_points(mesh)
    assert type(merged) is ColouredTriMesh and merged.n_points == 5
    np.testing.assert_array_equal(
        merged.colours[merged.trilist], mesh.colours[mesh.trilist]
    )


def test_landmarks_and_path_are_kept():
    mesh = TriMesh(CORNERS, TRILIST)
    mesh.landmarks["LJSON"] = PointCloud(np.array([[0.5, 0.5, 0.0]]))
    mesh.path = "somewhere/mesh.obj"
    merged = merge_duplicate_points(mesh)
    np.testing.assert_array_equal(
        merged.landmarks["LJSON"].points, mesh.landmarks["LJSON"].points
    )
    assert merged.path == mesh.path
    assert merged.landmarks is not mesh.landmarks


def test_the_index_says_where_every_point_went():
    mesh = TriMesh(CORNERS, TRILIST)
    merged, index = merge_duplicate_points(mesh, return_index=True)
    np.testing.assert_array_equal(index, [0, 1, 2, 1, 3, 2])
    np.testing.assert_array_equal(merged.points[index], mesh.points)


def test_a_mesh_without_duplicates_is_unchanged_and_zero_negative_zero_are_one():
    mesh = TriMesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-0.0, 0, 0]], float),
        [[0, 1, 2], [3, 1, 2]],
    )
    merged = merge_duplicate_points(mesh)
    assert merged.n_points == 3
    clean = TriMesh(CORNERS[[0, 1, 2, 4]], [[0, 1, 2], [1, 3, 2]])
    again = merge_duplicate_points(clean)
    np.testing.assert_array_equal(again.points, clean.points)
    np.testing.assert_array_equal(again.trilist, clean.trilist)


def test_a_point_cloud_is_a_value_error():
    with pytest.raises(ValueError, match="triangle list"):
        merge_duplicate_points(PointCloud(CORNERS))


def test_points_no_triangle_uses_are_kept():
    mesh = TriMesh(
        np.vstack([CORNERS, [[5, 5, 5]]]), TRILIST
    )  # the last point is in no triangle
    merged = merge_duplicate_points(mesh)
    assert merged.n_points == 5 and np.allclose(merged.points[-1], [5, 5, 5])


# ---- the OBJ importer says when it is needed, and the helper cleans james.obj
def test_importing_a_per_face_obj_warns_and_points_at_the_helper(tmp_path):
    obj = tmp_path / "square.obj"
    obj.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\n"
        "vt 0 0\nvt 1 0\nvt 0 1\nvt 1 1\nvt 0.9 0.9\n"
        # the texture coordinate indices differ from the vertex indices
        "f 1/1 2/2 3/3\nf 2/2 4/5 3/3\n"
    )
    with pytest.warns(UserWarning, match="merge_duplicate_points") as record:
        mesh = m3io.import_mesh(obj)
    message = next(
        str(w.message) for w in record if "merge_duplicate_points" in str(w.message)
    )
    assert "6 points" in message and "4 vertices" in message
    assert mesh.n_points == 6
    assert merge_duplicate_points(mesh).n_points == 4


def test_importing_an_obj_that_needs_no_merging_does_not_warn(tmp_path):
    obj = tmp_path / "plain.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mesh = m3io.import_mesh(obj)
    assert mesh.n_points == 3


def test_the_builtin_james_mesh_goes_from_three_points_per_triangle_to_its_vertices():
    with pytest.warns(UserWarning, match="merge_duplicate_points"):
        james = m3io.import_builtin_asset.james_obj()
    assert james.n_points == 3 * james.n_tris
    merged = merge_duplicate_points(james)
    assert merged.n_points < james.n_points // 4
    assert merged.n_tris == james.n_tris
    np.testing.assert_array_equal(
        merged.points[merged.trilist], james.points[james.trilist]
    )
    np.testing.assert_array_equal(
        merged.tcoords.points[merged.trilist], james.tcoords.points[james.trilist]
    )
