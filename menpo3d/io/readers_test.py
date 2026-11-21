import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import menpo3d.io as m3io
from menpo3d.io.input.mesh import obj, ply, stl, wrl
from menpo3d.io.input.mesh.triangulate import triangulate_polygons


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
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


def test_a_mesh_exported_to_ply_imports_back(tmp_path):
    mesh = m3io.import_builtin_asset.bunny_obj()
    m3io.export_mesh(mesh, tmp_path / "b.ply", overwrite=True)
    back = m3io.import_mesh(tmp_path / "b.ply")
    assert back.n_points == mesh.n_points and back.n_tris == mesh.n_tris


def _reference_duplicate_for_face_texture(
    points, colours, counts, connectivity, tex_counts, tex_values
):
    r"""``vtkPLYReader``'s handling of per-face (wedge) texture coordinates.

    A vertex can have only one texture coordinate in VTK, so whenever a face gives a vertex
    a different one, the vertex is duplicated (once per distinct texture coordinate).
    """
    tol = 1e-6
    n = len(points)
    positions = [tuple(p) for p in points.astype(np.float64)]
    tex = [(-1.0, -1.0)] * n
    colours = None if colours is None else list(colours)
    seen = {}  # texture coordinate -> index into point_ids
    point_ids = []
    connectivity = connectivity.copy()
    start = tex_start = 0
    for count, n_tex in zip(counts, tex_counts):
        verts = connectivity[start : start + count].copy()
        if n_tex == 2 * count:
            for k in range(count):
                new = (
                    float(np.float32(tex_values[tex_start + 2 * k])),
                    float(np.float32(tex_values[tex_start + 2 * k + 1])),
                )
                v = int(verts[k])
                current = tex[v]
                if current[0] == -1.0:
                    tex[v] = new
                    ti = seen.setdefault(new, len(seen))
                    if ti == len(point_ids):
                        point_ids.append([])
                    point_ids[ti].append(v)
                elif abs(current[0] - new[0]) >= tol or abs(current[1] - new[1]) >= tol:
                    inserted = new not in seen
                    ti = seen.setdefault(new, len(seen))
                    if ti == len(point_ids):
                        point_ids.append([])

                    def duplicate():
                        positions.append(positions[v])
                        tex.append(new)
                        if colours is not None:
                            colours.append(colours[v])
                        return len(positions) - 1

                    if inserted:
                        dp = duplicate()
                        point_ids[ti].append(dp)
                        verts[k] = dp
                    else:
                        for same in point_ids[ti]:
                            if all(
                                abs(a - b) < tol
                                for a, b in zip(positions[v], positions[same])
                            ):
                                verts[k] = same
                                break
                        else:
                            dp = duplicate()
                            point_ids[ti].append(dp)
                            verts[k] = dp
            connectivity[start : start + count] = verts
        start += count
        tex_start += n_tex
    out_points = np.array(positions, dtype=np.float32).reshape(-1, 3)
    out_colours = None if colours is None else np.array(colours, dtype=np.uint8)
    return out_points, out_colours, np.array(tex, dtype=np.float32), connectivity


# ------------------------------------------------------------------ edge cases found in review of #93
def test_a_quad_with_a_repeated_vertex_keeps_the_one_triangle_it_has():
    # VTK gives one triangle (and used to crash this reader with an IndexError); values from VTK 9.7
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.0]])
    cases = {
        (0, 1, 2, 2): [[0, 1, 2]],
        (0, 0, 1, 2): [[0, 1, 2]],
        (0, 1, 1, 3): [[0, 1, 3]],
    }
    for quad, expected in cases.items():
        tris = triangulate_polygons(square, [4], np.array(quad), warn=False)
        assert tris.tolist() == expected, quad
    pentagon = np.vstack([square, [[0.5, 1.5, 0]]])
    assert triangulate_polygons(
        pentagon, [5], [0, 1, 2, 3, 3], warn=False
    ).tolist() == [[0, 1, 3], [2, 3, 1]]
    assert triangulate_polygons(
        pentagon, [5], [0, 1, 2, 2, 4], warn=False
    ).tolist() == [[0, 1, 4], [2, 4, 1]]


def test_polygons_with_repeated_vertices_among_other_faces_keep_the_face_order():
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.0]])
    tris = triangulate_polygons(
        square,
        [3, 4, 4, 3],
        np.array([0, 1, 2, 0, 1, 2, 2, 0, 1, 2, 3, 0, 2, 3]),
        warn=False,
    )
    # triangle, the one triangle of the degenerate quad, the two of the proper quad, triangle
    assert len(tris) == 1 + 1 + 2 + 1
    assert (
        tris[0].tolist() == [0, 1, 2]
        and tris[1].tolist() == [0, 1, 2]
        and tris[-1].tolist() == [0, 2, 3]
    )


@pytest.mark.parametrize(
    "kind, k, seed, digest, n_tris",
    [  # the digest of vtkTriangleFilter's triangles (VTK 9.7) for a polygon from this seed
        ("star", 8, 1, "08599402b139e262", 6),
        ("star", 20, 2, "7f9a6ce14438ee11", 18),
        ("star", 60, 3, "093b84a6cce576b8", 58),
        ("nonplanar", 12, 4, "07b383ffb86f9ba8", 10),
        ("nonplanar", 40, 5, "9c058f2a6e697024", 32),
    ],
)
def test_ear_cutting_gives_the_triangles_vtk_gave(kind, k, seed, digest, n_tris):
    import hashlib

    rng = np.random.RandomState(seed)
    angle, radius = np.sort(rng.rand(k) * 2 * np.pi), 0.3 + rng.rand(k)
    z = 0.05 * rng.randn(k) if kind == "nonplanar" else 0 * angle
    # rounded so libm differences in the last bit (macOS, Windows) cannot change a near tie
    points = np.round(np.c_[radius * np.cos(angle), radius * np.sin(angle), z], 9)
    tris = triangulate_polygons(points, [k], np.arange(k), warn=False)
    assert len(tris) == n_tris
    assert (
        hashlib.sha256(np.ascontiguousarray(tris, np.int64).tobytes()).hexdigest()[:16]
        == digest
    )


def test_a_polygon_with_thousands_of_vertices_is_triangulated_completely():
    k = 3000
    rng = np.random.RandomState(0)
    angle, radius = np.sort(rng.rand(k) * 2 * np.pi), 0.3 + rng.rand(k)
    points = np.c_[radius * np.cos(angle), radius * np.sin(angle), 0 * angle]
    tris = triangulate_polygons(points, [k], np.arange(k), warn=False)
    assert len(tris) == k - 2
    xy = points[:, :2]

    def area(t):
        a, b, c = xy[t[:, 0]], xy[t[:, 1]], xy[t[:, 2]]
        return 0.5 * np.abs(
            (b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]
        )

    shoelace = 0.5 * abs(
        np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
    )
    assert area(tris).sum() == pytest.approx(
        shoelace, rel=1e-9
    )  # the triangles tile the polygon


def test_obj_keywords_may_be_separated_by_tabs_and_lines_joined_with_a_backslash(
    tmp_path,
):
    expected = [0, 1, 2]
    for name, text in {
        "tabs": "v\t0\t0\t0\nv\t1\t0\t0\nv\t0\t1\t0\nf\t1\t2\t3\n",
        "crlf": "v 0 0 0\r\nv 1 0 0\r\nv 0 1 0\r\nf 1 2 3\r\n",
        "continued": "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 \\\n3\n",
        "bom": "\ufeffv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
    }.items():
        data = obj.read_obj(_write(tmp_path, name + ".obj", text))
        assert len(data.points) == 3 and data.connectivity.tolist() == expected, name


def test_obj_faces_may_have_empty_texture_and_normal_fields(tmp_path):
    # VTK rejects "f 1// 2// 3//", this reader accepts it (listed in the PR as a deliberate difference)
    data = obj.read_obj(
        _write(
            tmp_path, "a.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1// 2// 3//\n"
        )
    )
    assert data.connectivity.tolist() == [0, 1, 2] and data.tcoords is None


def _binary_stl(n_header, n_facets, extra=b"", cut=0):
    import struct

    body = b"".join(
        struct.pack("<12fH", 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 + i, 0, 0)
        for i in range(n_facets)
    )
    data = b"x" * 80 + struct.pack("<I", n_header) + body + extra
    return data[: len(data) - cut] if cut else data


@pytest.mark.parametrize("header_count", [0, 1, 3, 10])
def test_binary_stl_is_read_to_the_end_whatever_the_header_count_says(
    tmp_path, header_count
):
    path = tmp_path / "a.stl"
    path.write_bytes(_binary_stl(header_count, 3))
    points, trilist = stl.read_stl(path)
    assert len(points) == 5 and len(trilist) == 3  # as VTK


@pytest.mark.parametrize(
    "data",
    [b"", b"x" * 40, _binary_stl(3, 3, extra=b"abc"), _binary_stl(3, 3, cut=20)],
    ids=["empty", "too short", "trailing bytes", "cut in a facet"],
)
def test_binary_stl_with_a_ragged_size_is_an_error(tmp_path, data):
    path = tmp_path / "a.stl"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        stl.read_stl(path)


def test_binary_stl_without_facets_is_an_empty_mesh(tmp_path):
    path = tmp_path / "a.stl"
    path.write_bytes(_binary_stl(0, 0))
    points, trilist = stl.read_stl(path)
    assert len(points) == 0 and len(trilist) == 0


_ASCII_STL = (
    "solid t\nfacet normal 0 0 1\n outer loop\n  vertex 0 0 0\n  vertex 1 0 0\n  vertex 0 1 0\n endloop\nendfacet\n"
    "facet normal 0 0 1\n outer loop\n  vertex 1 0 0\n  vertex 1 1 0\n  vertex 0 1 0\n endloop\nendfacet\nendsolid t\n"
)


@pytest.mark.parametrize("variant", ["plain", "tabs", "crlf"])
def test_ascii_stl_reads_whatever_the_whitespace(tmp_path, variant):
    text = {
        "plain": _ASCII_STL,
        "tabs": _ASCII_STL.replace(" ", "\t"),
        "crlf": _ASCII_STL.replace("\n", "\r\n"),
    }[variant]
    path = tmp_path / "a.stl"
    path.write_bytes(text.encode())
    points, trilist = stl.read_stl(path)
    assert len(points) == 4 and len(trilist) == 2


@pytest.mark.parametrize(
    "cut",
    [
        lambda t: t.replace("endsolid t\n", ""),  # VTK reads nothing from these
        lambda t: t[: t.index("endfacet", 100) + 9],  # after a whole facet, no endsolid
        lambda t: t[: t.index("vertex 1 1 0") + 8],  # in the middle of a vertex
        lambda t: t[: t.index("vertex 0 1 0", 100)],  # two vertices of the facet
        lambda t: t[: t.index("outer loop", 120) + 10],
    ],
    ids=[
        "no endsolid",
        "after a facet",
        "mid vertex",
        "two vertices",
        "after outer loop",
    ],
)
def test_truncated_ascii_stl_is_an_error_not_a_partial_mesh(tmp_path, cut):
    path = tmp_path / "a.stl"
    path.write_text(cut(_ASCII_STL))
    with pytest.raises(ValueError, match="truncated or malformed"):
        stl.read_stl(path)


def test_a_binary_stl_whose_header_starts_with_solid_is_still_binary(tmp_path):
    path = tmp_path / "a.stl"
    path.write_bytes(b"solid exported by something" + _binary_stl(3, 3)[27:])
    points, trilist = stl.read_stl(path)
    assert len(trilist) == 3


_PLY_HEADER = "ply\nformat ascii 1.0\n{extra}element vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n"


def test_ply_header_ends_at_the_end_header_line_not_at_a_comment_that_mentions_it(
    tmp_path,
):
    text = (
        _PLY_HEADER.format(extra="comment the line end_header ends this header\n")
        + "0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"
    )
    data = ply.read_ply(_write(tmp_path, "a.ply", text))
    assert len(data.points) == 3 and data.connectivity.tolist() == [0, 1, 2]


def test_ply_unknown_property_types_and_misplaced_properties_say_what_is_wrong(
    tmp_path,
):
    bad_type = _PLY_HEADER.format(extra="").replace(
        "property float z", "property float128 z"
    )
    with pytest.raises(ValueError, match="float128"):
        ply.read_ply(_write(tmp_path, "a.ply", bad_type))
    early = "ply\nformat ascii 1.0\nproperty float x\nend_header\n"
    with pytest.raises(ValueError, match="before any element"):
        ply.read_ply(_write(tmp_path, "b.ply", early))


def _wedge_ply(positions, faces, tex):
    n = len(positions)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "property list uchar float texcoord",
        "end_header",
    ]
    lines += [" ".join(f"{v:.9g}" for v in p) for p in positions]
    lines += [
        f"{len(f)} "
        + " ".join(map(str, f))
        + f" {2 * len(f)} "
        + " ".join(f"{v:.9g}" for uv in t for v in uv)
        for f, t in zip(faces, tex)
    ]
    return "\n".join(lines) + "\n"


def test_per_face_texture_coordinates_duplicate_a_vertex_once_per_distinct_coordinate(
    tmp_path,
):
    # two triangles sharing the edge 1-2; vertex 1 and 2 get different coordinates in the second face
    positions = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    faces = [[0, 1, 2], [1, 3, 2]]
    tex = [[(0, 0), (1, 0), (0, 1)], [(0.5, 0.5), (1, 1), (0.25, 0.75)]]
    data = ply.read_ply(_write(tmp_path, "a.ply", _wedge_ply(positions, faces, tex)))
    assert len(data.points) == 6  # the two shared vertices are duplicated
    assert data.connectivity.tolist() == [0, 1, 2, 4, 3, 5]
    assert data.tcoords.tolist() == [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [0.5, 0.5],
        [0.25, 0.75],
    ]
    assert data.points[[4, 5]].tolist() == [[1, 0, 0], [0, 1, 0]]


def test_a_texture_coordinate_within_a_millionth_is_the_same_coordinate(tmp_path):
    positions = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    faces = [[0, 1, 2], [1, 3, 2]]
    tex = [[(0, 0), (1, 0), (0, 1)], [(1 + 5e-7, 5e-7), (1, 1), (0, 1 - 5e-7)]]
    data = ply.read_ply(_write(tmp_path, "a.ply", _wedge_ply(positions, faces, tex)))
    assert len(data.points) == 4  # nothing duplicated: 5e-7 is below the tolerance


def test_texture_duplication_matches_the_original_quadratic_algorithm():
    # the reference is the scan over all points with the same texture coordinate that this replaced
    rng = np.random.RandomState(0)
    for trial in range(60):
        n = rng.randint(3, 40)
        base = rng.randint(0, 3, (n, 3)).astype(np.float32)
        base += (rng.rand(n, 3) < 0.4) * rng.choice([0, 5e-7, 2e-6], (n, 3)).astype(
            np.float32
        )
        if trial % 7 == 0:
            base[rng.randint(n)] = np.nan
        tex_set = np.array(
            [[0, 0], [0.5, 0.5], [1, 0], [0, 1], [0.5, 0.5000004]], np.float32
        )
        counts = rng.choice([3, 4], rng.randint(1, 30))
        connectivity = rng.randint(0, n, counts.sum())
        tex_counts = np.where(rng.rand(len(counts)) < 0.9, 2 * counts, 0)
        tex_values = np.concatenate(
            [np.zeros(0, np.float32)]
            + [
                tex_set[rng.randint(0, len(tex_set), c)].ravel()
                for c, t in zip(counts, tex_counts)
                if t
            ]
        )
        colours = rng.randint(0, 255, (n, 3)).astype(np.uint8) if trial % 2 else None
        args = (base, colours, counts, connectivity, tex_counts, tex_values)
        got = ply._duplicate_for_face_texture(*args)
        want = _reference_duplicate_for_face_texture(*args)
        for g, w in zip(got, want):
            assert (g is None) == (w is None)
            if g is not None:
                np.testing.assert_array_equal(g, w)


def test_texture_duplication_is_not_quadratic():
    # many vertices share one coordinate, as when the untextured faces all carry (0, 0): this took minutes before
    g = 90
    xs, ys = np.meshgrid(np.arange(g, dtype=np.float32), np.arange(g, dtype=np.float32))
    points = np.c_[xs.ravel(), ys.ravel(), 0 * xs.ravel()].astype(np.float32)
    idx = np.arange(g * g).reshape(g, g)
    tris = np.concatenate(
        [
            np.c_[idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()],
            np.c_[idx[:-1, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()],
        ]
    )
    tex = np.zeros((len(tris), 3, 2), np.float32)
    tex[1::2] = (np.arange(len(tris))[1::2, None, None] % 5) / 5.0
    import time

    start = time.perf_counter()
    out_points, _, _, connectivity = ply._duplicate_for_face_texture(
        points,
        None,
        np.full(len(tris), 3),
        tris.ravel(),
        np.full(len(tris), 6),
        tex.ravel(),
    )
    assert (
        time.perf_counter() - start < 20
    )  # about 1 s; the quadratic version needed over a minute here
    assert len(out_points) >= len(points) and connectivity.max() < len(out_points)


_WRL = "#VRML V2.0 utf8\nShape { geometry IndexedFaceSet {\n coord Coordinate { point [ 0 0 0, 1 0 0, 0 1 0, 1 1 0, 2 0 1 ] }\n texCoord TextureCoordinate { point [ 0 0, 1 0, 0 1, 1 1 ] }\n coordIndex [ %s ]\n%s} }\n"


def _wrl(tmp_path, coord_index, tex_index=None, extra=""):
    tex = "" if tex_index is None else " texCoordIndex [ %s ]\n" % tex_index
    return _write(tmp_path, "a.wrl", _WRL % (coord_index, tex + extra))


@pytest.mark.parametrize(
    "coord_index, tex_index, n_points, tcoords",
    [  # the faces that have a texture face of the same size are kept (values from VTK 9.7)
        (
            "0 1 2 -1 1 2 3 -1",
            "0 1 2 -1",
            3,
            [[0, 0], [1, 0], [0, 1]],
        ),  # fewer texture faces
        (
            "0 1 2 -1",
            "0 1 2 -1 1 2 3 -1",
            3,
            [[0, 0], [1, 0], [0, 1]],
        ),  # more texture faces
        (
            "0 1 2 -1 1 2 3 4 -1",
            "0 1 2 -1 1 2 3 -1",
            3,
            [[0, 0], [1, 0], [0, 1]],
        ),  # a face of another size
        (
            "0 1 2 -1 1 2 3 -1",
            "0 1 2 3 -1 1 2 3 -1",
            3,
            [[1, 0], [0, 1], [1, 1]],
        ),  # the first is dropped
    ],
)
def test_vrml_texture_faces_that_do_not_line_up_with_the_coordinate_faces(
    tmp_path, coord_index, tex_index, n_points, tcoords
):
    data = wrl.read_wrl(_wrl(tmp_path, coord_index, tex_index))
    assert len(data.points) == n_points and data.counts.tolist() == [3]
    assert data.tcoords.tolist() == tcoords


def test_vrml_normals_and_texture_coordinates_without_a_texture_index_do_not_crash(
    tmp_path,
):
    extra = " normal Normal { vector [ 0 0 1, 0 1 0, 1 0 0, 1 1 1 ] }\n normalIndex [ 0 1 2 -1 ]\n"
    data = wrl.read_wrl(_wrl(tmp_path, "0 1 2 -1", None, extra))
    assert (
        len(data.points) == 0
    )  # as VTK: without texCoordIndex there is nothing to line the corners up with
