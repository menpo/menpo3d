import numpy as np
import pytest
from numpy.testing import assert_array_equal

import menpo3d.io as m3io
from menpo3d.io.input.mesh import obj, ply, stl, wrl
from menpo3d.io.input.mesh.triangulate import triangulate_polygons


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_obj_keeps_unused_vertices_and_the_vertex_order(tmp_path):
    path = _write(tmp_path, "a.obj", "v 0 0 0\nv 1 0 0\nv 9 9 9\nv 0 1 0\nf 1 2 4\n")
    data = obj.read_obj(path)
    assert len(data.points) == 4 and not data.duplicated
    assert_array_equal(data.connectivity, [0, 1, 3])


def test_obj_mismatched_texture_index_gives_one_point_per_corner(tmp_path):
    text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/3 2/2 3/1\n"
    data = obj.read_obj(_write(tmp_path, "a.obj", text))
    assert data.duplicated and len(data.points) == 3
    assert_array_equal(data.connectivity, [0, 1, 2])
    assert_array_equal(data.tcoords, [[0, 1], [1, 0], [0, 0]])


def test_obj_relative_indices_and_comments(tmp_path):
    text = "# a comment\nv 0 0 0\nv 1 0 0\nv 0 1 0 # trailing\nf -3 -2 -1\n"
    assert_array_equal(
        obj.read_obj(_write(tmp_path, "a.obj", text)).connectivity, [0, 1, 2]
    )


def test_obj_face_with_two_vertices_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        obj.read_obj(_write(tmp_path, "a.obj", "v 0 0 0\nv 1 0 0\nf 1 2\n"))


def test_quads_split_along_the_shorter_diagonal():
    # a rectangle wider than tall: the diagonal 1-3 and 0-2 are equal, so VTK takes 1-3 ...
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.0]])
    assert triangulate_polygons(square, [4], [0, 1, 2, 3], warn=False).tolist() == [
        [0, 1, 3],
        [1, 2, 3],
    ]
    # ... and a quad with a short diagonal 0-2 is split along it
    kite = np.array([[0, 0, 0], [2, -1, 0], [1, 0, 0], [2, 1, 0.0]])
    assert triangulate_polygons(kite, [4], [0, 1, 2, 3], warn=False).tolist() == [
        [0, 1, 2],
        [0, 2, 3],
    ]


def test_concave_quad_is_split_through_the_reflex_vertex():
    arrow = np.array([[0, 0, 0], [2, 1, 0], [0, 2, 0], [1, 1, 0.0]])
    tris = triangulate_polygons(arrow, [4], [0, 1, 2, 3], warn=False)
    assert sorted(map(sorted, tris.tolist())) == [[0, 1, 3], [1, 2, 3]]


def test_triangles_pass_through_untouched_and_without_a_warning(recwarn):
    conn = np.array([2, 1, 0, 3, 1, 2])
    assert triangulate_polygons(np.zeros((4, 3)), [3, 3], conn).tolist() == [
        [2, 1, 0],
        [3, 1, 2],
    ]
    assert len(recwarn) == 0


def test_polygons_with_five_sides_are_split_and_warn():
    angles = np.linspace(0, 2 * np.pi, 6)[:-1]
    pentagon = np.c_[np.cos(angles), np.sin(angles), np.zeros(5)]
    with pytest.warns(UserWarning, match="Non-triangular"):
        tris = triangulate_polygons(pentagon, [5], np.arange(5))
    assert tris.shape == (3, 3) and set(tris.ravel()) == set(range(5))


def test_stl_merges_identical_vertices_in_order_of_first_appearance(tmp_path):
    text = """solid t
facet normal 0 0 0
outer loop
vertex 0 0 0
vertex 1 0 0
vertex 0 1 0
endloop
endfacet
facet normal 0 0 0
outer loop
vertex 1 0 0
vertex 1 1 0
vertex 0 1 0
endloop
endfacet
facet normal 0 0 0
outer loop
vertex 5 5 5
vertex 5 5 5
vertex 6 6 6
endloop
endfacet
endsolid t
"""
    points, trilist = stl.read_stl(_write(tmp_path, "a.stl", text))
    assert points.dtype == np.float32
    assert points.tolist() == [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [5, 5, 5],
        [6, 6, 6],
    ]
    assert trilist.tolist() == [[0, 1, 2], [1, 3, 2]]  # the collapsed facet is dropped


def test_ply_vertex_positions_are_float32(tmp_path):
    text = """ply
format ascii 1.0
element vertex 3
property double x
property double y
property double z
element face 1
property list uchar int vertex_indices
end_header
0.1 0 0
1 0 0
0 1 0
3 0 1 2
"""
    data = ply.read_ply(_write(tmp_path, "a.ply", text))
    assert data.points.dtype == np.float32 and data.points[0, 0] == np.float32(0.1)


def test_wrl_leading_dot_numbers_are_read_correctly(tmp_path):
    # VTK reads ".5" as 0; we do not copy that bug
    text = (
        "#VRML V2.0 utf8\nShape { geometry IndexedFaceSet { coord Coordinate { point "
        "[ .5 0 -.25, 1 0 0, 0 1 0.75 ] } coordIndex [ 0 1 2 -1 ] } }\n"
    )
    data = wrl.read_wrl(_write(tmp_path, "a.wrl", text))
    assert data.points[0].tolist() == [0.5, 0, -0.25]


def test_wrl_uses_the_first_indexed_face_set_and_ignores_transforms(tmp_path):
    text = """#VRML V2.0 utf8
Transform { translation 10 10 10 children [
 Shape { geometry Box { size 1 1 1 } }
 Shape { geometry IndexedFaceSet { coord Coordinate { point [0 0 0, 1 0 0, 0 1 0] } coordIndex [0 1 2 -1] } }
 Shape { geometry IndexedFaceSet { coord Coordinate { point [5 5 5, 6 5 5, 5 6 5] } coordIndex [0 1 2 -1] } }
] }
"""
    with pytest.warns(UserWarning, match="More than one"):
        data = wrl.read_wrl(_write(tmp_path, "a.wrl", text))
    assert data.points.tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0]]


def test_import_mesh_goes_through_every_registered_format(tmp_path):
    mesh = m3io.import_builtin_asset.bunny_obj()
    m3io.export_mesh(mesh, tmp_path / "b.ply", overwrite=True)
    back = m3io.import_mesh(tmp_path / "b.ply")
    assert back.n_points == mesh.n_points and back.n_tris == mesh.n_tris
