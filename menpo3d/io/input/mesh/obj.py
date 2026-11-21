r"""A VTK-free Wavefront OBJ reader that reproduces ``vtkOBJReader``'s topology exactly.

OBJ lets every face corner pick its own position, texture coordinate and normal
independently, VTK points carry all three together. ``vtkOBJReader`` therefore keeps the
``v`` list as the points when every ``vt``/``vn`` index equals its ``v`` index, and
otherwise gives every face corner its own point (in face order). Existing datasets are
built against those point lists, so this reader does the same. Merging duplicate vertices
is a separate step that belongs after import.
"""

import numpy as np


class ObjData:
    r"""What an OBJ file contains, after ``vtkOBJReader``'s point duplication rules."""

    def __init__(self, points, counts, connectivity, tcoords, duplicated):
        self.points = points  # (n, 3) float64
        self.counts = counts  # vertices per face
        self.connectivity = connectivity  # point index of every face corner
        self.tcoords = tcoords  # (n, 2) float32 or None; -1 where a point has none
        self.duplicated = duplicated  # True if there is one point per face corner


def _index(token, count, line_number):
    value = int(token)
    absolute = count + value if value < 0 else value - 1
    if absolute < 0:
        raise ValueError(
            "Unexpected index value {} at line {}".format(value, line_number)
        )
    return absolute


def _logical_lines(text):
    r"""``(line_number, text)`` with comments removed and lines ending in a backslash joined to the next, as VTK did."""
    lines = text.lstrip(
        "\ufeff"
    ).splitlines()  # a byte order mark must not hide the first keyword
    k = 0
    while k < len(lines):
        number, line = k + 1, lines[k].split("#", 1)[0]
        while line.rstrip().endswith("\\") and k + 1 < len(lines):
            k += 1
            line = line.rstrip()[:-1] + " " + lines[k].split("#", 1)[0]
        k += 1
        line = line.strip()
        if line:
            yield number, line


def read_obj(path):
    r"""Parse an OBJ file.

    Parameters
    ----------
    path : `str` or `pathlib.Path`

    Returns
    -------
    data : `ObjData`
    """
    vertices, vts = [], []
    counts, face_v, face_t = [], [], []  # face_t: tcoord index per corner, -1 for none
    n_normals = 0
    match = True  # every vt / vn index equals its v index

    with open(str(path), "rb") as f:
        raw = f.read()
    for line_number, line in _logical_lines(raw.decode("utf-8", "replace")):
        key, rest = (line.split(None, 1) + [""])[
            :2
        ]  # the keyword ends at any whitespace (tabs too)
        if key == "v":
            fields = rest.split()
            try:
                vertices.append((float(fields[0]), float(fields[1]), float(fields[2])))
            except (IndexError, ValueError):
                raise ValueError(
                    "Failed to parse vertex at line {}".format(line_number)
                )
        elif key == "vt":
            fields = rest.split()
            try:
                vts.append((float(fields[0]), float(fields[1])))
            except (IndexError, ValueError):
                raise ValueError(
                    "Failed to parse tcoord at line {}".format(line_number)
                )
        elif key == "vn":
            n_normals += 1
        elif key == "f":
            tokens = rest.replace("\\", " ").split()
            if len(tokens) < 3:
                raise ValueError(
                    "A face needs at least 3 vertices (line {})".format(line_number)
                )
            n_v, n_t = len(vertices), len(vts)
            corner_v, corner_t = [], []
            has_t = has_n = 0
            for token in tokens:
                parts = token.split("/")
                v = _index(parts[0], n_v, line_number)
                corner_v.append(v)
                if len(parts) > 1 and parts[1] != "":
                    t = _index(parts[1], n_t, line_number)
                    corner_t.append(t)
                    has_t += 1
                    match = match and t == v
                if len(parts) > 2 and parts[2] != "":
                    n = _index(parts[2], n_normals, line_number)
                    has_n += 1
                    match = match and n == v
            if (has_t and has_t != len(tokens)) or (has_n and has_n != len(tokens)):
                raise ValueError(
                    "A face must give vt/vn for all of its points (line {})".format(
                        line_number
                    )
                )
            counts.append(len(tokens))
            face_v.extend(corner_v)
            face_t.extend(corner_t if has_t else [-1] * len(tokens))

    points = np.array(vertices, dtype=np.float64).reshape(-1, 3)
    counts = np.array(counts, dtype=np.int64)
    connectivity = np.array(face_v, dtype=np.int64)
    corner_t = np.array(face_t, dtype=np.int64)
    if len(connectivity) and connectivity.max() >= len(points):
        raise ValueError("A face refers to a vertex that does not exist")
    if len(corner_t) and corner_t.max() >= len(vts):
        raise ValueError("A face refers to a texture coordinate that does not exist")

    vt = np.array(vts, dtype=np.float32).reshape(-1, 2)
    has_tcoords = bool(len(vts)) and bool((corner_t >= 0).any())
    if not match:
        # one new point per face corner, in face order
        tcoords = None
        if has_tcoords:
            tcoords = np.full((len(connectivity), 2), -1.0, dtype=np.float32)
            used = corner_t >= 0
            tcoords[used] = vt[corner_t[used]]
        return ObjData(
            points[connectivity],
            counts,
            np.arange(len(connectivity), dtype=np.int64),
            tcoords,
            True,
        )
    tcoords = None
    if has_tcoords:
        tcoords = np.full((len(points), 2), -1.0, dtype=np.float32)
        used = corner_t >= 0
        tcoords[connectivity[used]] = vt[corner_t[used]]
    return ObjData(points, counts, connectivity, tcoords, False)
