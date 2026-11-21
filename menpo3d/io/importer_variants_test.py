r"""Importers must keep producing the topology the VTK based importers produced.

Nothing is checked in apart from the bunny that already ships with menpo3d: format
variants of it (quads, polygons, texture coordinate indices that differ from the vertex
indices, binary and ASCII PLY/STL, ...) are written at test time and the result of
importing them is compared with fingerprints recorded once from the previous VTK based
importers (``importer_fingerprints.json``). VTK is no longer used or needed: the
fingerprints are the frozen record of what those importers produced.
"""

import hashlib
import json
import struct
import warnings
from pathlib import Path

import numpy as np
import pytest

import menpo3d.io as m3io

FINGERPRINTS = Path(__file__).with_name("importer_fingerprints.json")


def _bunny():
    mesh = m3io.import_builtin_asset.bunny_obj()
    return mesh.points.astype(np.float64), mesh.trilist.astype(np.int64)


def _sha(array, dtype):
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype=dtype).tobytes()
    ).hexdigest()[:16]


def fingerprint(points, trilist, tcoords=None):
    out = {
        "n_points": int(len(points)),
        "n_tris": int(len(trilist)),
        "points": _sha(points, np.float64),
        "trilist": _sha(trilist, np.int64),
    }
    if tcoords is not None:
        out["tcoords"] = _sha(tcoords, np.float32)
    return out


# --------------------------------------------------------------------------- the variants
def _polygons(trilist, kinds):
    r"""Turn a triangulation into polygons: ``kinds`` cycles through 3, 4, 5, 6 sided faces."""
    n_points_extra = trilist.max()
    faces, i, k = [], 0, 0
    while i < len(trilist):
        size = kinds[k % len(kinds)]
        k += 1
        if size == 3 or i + 1 >= len(trilist):
            faces.append(list(trilist[i]))
            i += 1
            continue
        a, b = trilist[i], trilist[i + 1]
        face = [a[0], a[1], a[2], b[2], b[1], b[0], a[0]][:size]
        face = list(dict.fromkeys(face))  # no repeated vertices
        faces.append(face if len(face) >= 3 else list(a))
        i += 2
    assert n_points_extra >= 0
    return faces


def _write_obj(path, points, faces, tcoords=None, tindex=None, negative=False):
    lines = ["v %.9g %.9g %.9g" % tuple(p) for p in points]
    if tcoords is not None:
        lines += ["vt %.9g %.9g" % tuple(t) for t in tcoords]
    n = len(points)
    for f, face in enumerate(faces):
        tokens = []
        for k, v in enumerate(face):
            vi = (v - n) if negative else v + 1
            if tcoords is None:
                tokens.append(str(vi))
            else:
                ti = tindex[f][k] + 1
                tokens.append("%d/%d" % (vi, ti))
        lines.append("f " + " ".join(tokens))
    Path(path).write_text("\n".join(lines) + "\n")


def _write_ply(path, points, faces, kind, colours=None):
    header = [
        "ply",
        {
            "ascii": "format ascii 1.0",
            "le": "format binary_little_endian 1.0",
            "be": "format binary_big_endian 1.0",
        }[kind],
        "element vertex %d" % len(points),
    ]
    header += ["property float x", "property float y", "property float z"]
    if colours is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += [
        "element face %d" % len(faces),
        "property list uchar int vertex_indices",
        "end_header",
    ]
    head = ("\n".join(header) + "\n").encode()
    if kind == "ascii":
        rows = []
        for i, p in enumerate(points):
            row = "%.9g %.9g %.9g" % tuple(np.float32(p))
            rows.append(
                row + (" %d %d %d" % tuple(colours[i]) if colours is not None else "")
            )
        rows += ["%d %s" % (len(f), " ".join(map(str, f))) for f in faces]
        Path(path).write_bytes(head + ("\n".join(rows) + "\n").encode())
        return
    e = "<" if kind == "le" else ">"
    body = b""
    for i, p in enumerate(points):
        body += struct.pack(e + "fff", *p)
        if colours is not None:
            body += struct.pack("BBB", *colours[i])
    for f in faces:
        body += struct.pack("B", len(f)) + struct.pack(e + "%di" % len(f), *f)
    Path(path).write_bytes(head + body)


def _write_stl(path, points, trilist, binary):
    tri = points[trilist].astype(np.float32)
    if binary:
        out = b"\0" * 80 + struct.pack("<I", len(tri))
        for t in tri:
            out += struct.pack("<3f", 0, 0, 0) + t.astype("<f4").tobytes() + b"\0\0"
        Path(path).write_bytes(out)
    else:
        lines = ["solid bunny"]
        for t in tri:
            lines += ["facet normal 0 0 0", "outer loop"]
            lines += ["vertex %.9g %.9g %.9g" % tuple(v) for v in t]
            lines += ["endloop", "endfacet"]
        lines.append("endsolid bunny")
        Path(path).write_text("\n".join(lines) + "\n")


def _write_wrl(path, points, faces, tcoords=None, tindex=None):
    coord_index = " ".join(" ".join(map(str, f)) + " -1" for f in faces)
    text = [
        "#VRML V2.0 utf8",
        "Shape {",
        " geometry IndexedFaceSet {",
        "  coord Coordinate { point [",
    ]
    text += ["   %.9g %.9g %.9g," % tuple(np.float32(p)) for p in points] + ["  ] }"]
    text += ["  coordIndex [ %s ]" % coord_index]
    if tcoords is not None:
        text += ["  texCoord TextureCoordinate { point ["]
        text += ["   %.9g %.9g," % tuple(t) for t in tcoords] + ["  ] }"]
        if tindex is not None:
            text += [
                "  texCoordIndex [ %s ]"
                % " ".join(" ".join(map(str, f)) + " -1" for f in tindex)
            ]
    text += [" }", "}"]
    Path(path).write_text("\n".join(text) + "\n")


def variants(directory):
    r"""Write the variants of the bunny into ``directory``; returns ``{name: path}``."""
    points, trilist = _bunny()
    rng = np.random.RandomState(0)
    tris = [list(t) for t in trilist]
    mixed = _polygons(trilist, [3, 4, 4, 5, 3, 6, 4])
    quads = _polygons(trilist, [4])
    tcoords = rng.rand(len(points), 2).astype(np.float32)
    shuffled = rng.permutation(len(points))  # a different vt for every corner index
    d = Path(directory)
    paths = {}

    def add(name, fn, *args, **kwargs):
        paths[name] = d / name
        fn(paths[name], *args, **kwargs)

    add("tri.obj", _write_obj, points, tris)
    add("tri_negative.obj", _write_obj, points, tris, negative=True)
    add("quads.obj", _write_obj, points, quads)
    add("mixed_polygons.obj", _write_obj, points, mixed)
    add("tri_vt_matching.obj", _write_obj, points, tris, tcoords, tris)
    add(
        "tri_vt_mismatch.obj",
        _write_obj,
        points,
        tris,
        tcoords,
        [list(shuffled[t]) for t in trilist],
    )
    add(
        "mixed_vt_mismatch.obj",
        _write_obj,
        points,
        mixed,
        tcoords,
        [list(shuffled[f]) for f in mixed],
    )
    colours = rng.randint(0, 256, (len(points), 3))
    for kind in ("ascii", "le", "be"):
        add("tri_%s.ply" % kind, _write_ply, points, tris, kind, colours)
        add("mixed_%s.ply" % kind, _write_ply, points, mixed, kind)
    add("tri_ascii.stl", _write_stl, points, trilist, False)
    add("tri_binary.stl", _write_stl, points, trilist, True)
    add("tri.wrl", _write_wrl, points, tris)
    add("mixed.wrl", _write_wrl, points, mixed)
    add("tri_tex_matching.wrl", _write_wrl, points, tris, tcoords, tris)
    add(
        "tri_tex_mismatch.wrl",
        _write_wrl,
        points,
        tris,
        tcoords,
        [list(shuffled[t]) for t in trilist],
    )
    return paths


# --------------------------------------------------------------------------- reading them
def read_new(path):
    r"""(points, trilist, tcoords) as the importers return them (tcoords regardless of texture)."""
    from menpo3d.io.input.mesh import obj, ply, stl, triangulate, wrl

    suffix = Path(path).suffix
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if suffix == ".obj":
            data = obj.read_obj(path)
            points, tcoords = data.points, data.tcoords
        elif suffix == ".ply":
            data = ply.read_ply(path)
            points, tcoords = data.points.astype(np.float64), data.tcoords
        elif suffix == ".wrl":
            data = wrl.read_wrl(path)
            points, tcoords = data.points.astype(np.float64), data.tcoords
        else:
            points, trilist = stl.read_stl(path)
            return points.astype(np.float64), trilist, None
        trilist = triangulate.triangulate_polygons(
            points, data.counts, data.connectivity
        )
    return points, trilist, tcoords


# --------------------------------------------------------------------------- the tests
_NAMES = sorted(json.loads(FINGERPRINTS.read_text())) if FINGERPRINTS.exists() else []


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    return variants(tmp_path_factory.mktemp("variants"))


@pytest.mark.parametrize("name", _NAMES)
def test_topology_matches_the_vtk_importers(written, name):
    expected = json.loads(FINGERPRINTS.read_text())[name]
    points, trilist, tcoords = read_new(written[name])
    got = fingerprint(points, trilist, tcoords if "tcoords" in expected else None)
    assert got == expected


def test_fingerprints_cover_every_variant(written):
    assert sorted(written) == _NAMES


def test_texture_coordinate_mismatch_duplicates_the_points():
    # the point of the exercise: a mismatch gives one point per face corner, not a merged mesh
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        paths = variants(directory)
        points, trilist, _ = read_new(paths["tri_vt_mismatch.obj"])
        plain, plain_trilist, _ = read_new(paths["tri.obj"])
    assert len(points) == 3 * len(trilist) and len(plain) < len(points)
    assert np.array_equal(trilist, np.arange(3 * len(trilist)).reshape(-1, 3))


def test_retriangulating_warns(written):
    with pytest.warns(UserWarning, match="Non-triangular"):
        m3io.import_mesh(written["quads.obj"])


# --------------------------------------------------------------------------- through the public importers
# ``read_new`` calls the readers directly; these go through ``import_mesh`` so that the glue is pinned too:
# the mesh class, the float64 points, colours scaled to 0..1 and texture coordinates only with a texture.
@pytest.mark.parametrize("name", _NAMES)
def test_import_mesh_gives_the_recorded_topology(written, name):
    expected = json.loads(FINGERPRINTS.read_text())[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh = m3io.import_mesh(written[name])
    assert mesh.points.dtype == np.float64
    got = fingerprint(mesh.points, mesh.trilist)
    assert got == {k: v for k, v in expected.items() if k != "tcoords"}


@pytest.mark.parametrize(
    "name",
    [
        "tri_vt_matching.obj",
        "tri_vt_mismatch.obj",
        "tri_tex_matching.wrl",
        "tri_tex_mismatch.wrl",
    ],
)
def test_a_texture_next_to_the_mesh_gives_a_textured_mesh_with_the_recorded_tcoords(
    written, name, tmp_path
):
    import shutil

    from PIL import Image

    expected = json.loads(FINGERPRINTS.read_text())[name]
    path = tmp_path / name  # a copy: the written variants are shared between tests
    shutil.copy(written[name], path)
    Image.fromarray(np.full((4, 4, 3), 200, np.uint8)).save(path.with_suffix(".png"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh = m3io.import_mesh(path)
    assert type(mesh).__name__ == "TexturedTriMesh"
    got = fingerprint(mesh.points, mesh.trilist, mesh.tcoords.points)
    assert got == expected


def test_without_a_texture_the_texture_coordinates_are_dropped(written):
    mesh = m3io.import_mesh(written["tri_vt_matching.obj"])
    assert type(mesh).__name__ == "TriMesh" and not hasattr(mesh, "tcoords")


def test_ply_vertex_colours_are_scaled_to_zero_one(tmp_path):
    points, trilist = _bunny()
    colours = np.random.RandomState(1).randint(0, 256, (len(points), 3))
    for kind in ("ascii", "le", "be"):
        path = tmp_path / ("c_%s.ply" % kind)
        _write_ply(path, points, [list(t) for t in trilist], kind, colours)
        mesh = m3io.import_mesh(path)
        assert type(mesh).__name__ == "ColouredTriMesh"
        np.testing.assert_allclose(mesh.colours, colours / 255.0)
