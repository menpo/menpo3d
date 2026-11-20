r"""Importers must keep producing the topology the VTK based importers produced.

Nothing is checked in apart from the bunny that already ships with menpo3d: format
variants of it (quads, polygons, texture coordinate indices that differ from the vertex
indices, binary and ASCII PLY/STL, ...) are written at test time and the result of
importing them is compared with fingerprints recorded once from the previous VTK based
importers (``importer_fingerprints.json``). With VTK installed, check that the recorded
fingerprints are still what VTK does (or record them again, without ``--check``)::

    python -m menpo3d.io.importer_variants_test --check
"""

import hashlib
import json
import struct
import sys
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


def read_vtk(path):
    r"""What the previous importers did with VTK (needs VTK; used to record fingerprints)."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    suffix = Path(path).suffix
    reader = {
        ".obj": vtk.vtkOBJReader,
        ".ply": vtk.vtkPLYReader,
        ".stl": vtk.vtkSTLReader,
    }.get(suffix)
    if reader is not None:
        reader = reader()
        reader.SetFileName(str(path))
        reader.Update()
        polydata = reader.GetOutput()
    else:
        importer = vtk.vtkVRMLImporter()
        importer.SetFileName(str(path))
        importer.Update()
        actors = importer.GetRenderer().GetActors()
        actors.InitTraversal()
        polydata = actors.GetNextActor().GetMapper().GetInput()
    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float64)
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(polydata)
    triangles.Update()
    trilist = vtk_to_numpy(triangles.GetOutput().GetPolys().GetData()).reshape(-1, 4)[
        :, 1:
    ]
    tcoords = polydata.GetPointData().GetTCoords()
    return points, trilist, None if tcoords is None else vtk_to_numpy(tcoords)


def regenerate(check=False):
    import tempfile

    out = {}
    with tempfile.TemporaryDirectory() as directory:
        for name, path in variants(directory).items():
            out[name] = fingerprint(*read_vtk(path))
    if check:  # does the installed VTK still do what the recorded fingerprints say?
        recorded = json.loads(FINGERPRINTS.read_text())
        changed = sorted(
            k for k in set(out) | set(recorded) if out.get(k) != recorded.get(k)
        )
        for name in changed:
            print("DIFFERS", name, recorded.get(name), "->", out.get(name))
        print(
            "VTK",
            __import__("vtk").vtkVersion.GetVTKVersion(),
            "-",
            len(changed),
            "of",
            len(out),
            "differ",
        )
        return 1 if changed else 0
    FINGERPRINTS.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("recorded", len(out), "fingerprints in", FINGERPRINTS)


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


if __name__ == "__main__":
    sys.exit(regenerate(check="--check" in sys.argv))
