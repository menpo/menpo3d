r"""A VTK-free PLY reader that reproduces what ``vtkPLYReader`` returned.

Like VTK, vertex positions and texture coordinates are read as float32, faces are kept
as polygons (and later split into triangles), and the colours are the vertex
``red/green/blue`` (or ``diffuse_*``) properties.
"""

import numpy as np

_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}  # fmt: skip


class PlyData:
    def __init__(self, points, tcoords, colours, counts, connectivity):
        self.points = points  # (n, 3) float32
        self.tcoords = tcoords  # (n, 2) float32 or None
        self.colours = colours  # (n, 3 or 4) uint8 or None
        self.counts = counts
        self.connectivity = connectivity


def _parse_header(raw):
    end = raw.find(b"end_header")
    if not raw.startswith(b"ply") or end < 0:
        raise ValueError("Not a PLY file")
    newline = raw.find(b"\n", end)
    body_start = newline + 1
    fmt, elements = None, []
    for line in raw[:end].decode("ascii", "replace").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "format":
            fmt = fields[1]
        elif fields[0] == "element":
            elements.append({"name": fields[1], "count": int(fields[2]), "props": []})
        elif fields[0] == "property":
            if fields[1] == "list":
                prop = (
                    fields[4],
                    "list",
                    _TYPES[fields[3]],
                    _TYPES[fields[2]],
                )  # name, item, count
            else:
                prop = (fields[2], "scalar", _TYPES[fields[1]], None)
            elements[-1]["props"].append(prop)
    if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise ValueError("Unsupported PLY format {}".format(fmt))
    if b"\r\n" == raw[newline - 1 : newline + 1]:
        pass  # body_start already past the \n
    return fmt, elements, body_start


class _Reader:
    r"""Sequential access to the values of an ASCII or binary body."""

    def __init__(self, fmt, raw, body_start):
        self.ascii = fmt == "ascii"
        self.endian = ">" if fmt == "binary_big_endian" else "<"
        if self.ascii:
            self.tokens = np.array(raw[body_start:].split(), dtype=np.float64)
        else:
            self.raw = raw
        self.pos = body_start if not self.ascii else 0

    def read(self, count, fields):
        r"""Read ``count`` rows of a numpy structured dtype ``fields`` (ASCII: all scalars)."""
        dtype = np.dtype(
            [
                (n, self.endian + t, s) if s else (n, self.endian + t)
                for n, t, s in fields
            ]
        )
        if self.ascii:
            width = sum(int(np.prod(s)) if s else 1 for _, _, s in fields)
            block = self.tokens[self.pos : self.pos + count * width]
            self.pos += count * width
            if block.size < count * width:
                raise ValueError("The PLY file is truncated")
            block = block.reshape(count, width)
            out = np.empty(count, dtype=dtype)
            column = 0
            for name, _, shape in fields:
                w = int(np.prod(shape)) if shape else 1
                out[name] = block[:, column : column + w].reshape(
                    (count,) + (shape or ())
                )
                column += w
            return out
        size = dtype.itemsize * count
        if self.pos + size > len(self.raw):
            raise ValueError("The PLY file is truncated")
        out = np.frombuffer(self.raw, dtype=dtype, count=count, offset=self.pos)
        self.pos += size
        return out

    def peek_value(self, type_):
        if self.ascii:
            return self.tokens[self.pos]
        return np.frombuffer(
            self.raw, dtype=self.endian + type_, count=1, offset=self.pos
        )[0]


def _read_element(reader, element):
    r"""Return ``{name: array}`` for one element; list properties give ``(counts, values)``."""
    count, props = element["count"], element["props"]
    if count == 0:
        return {
            n: (np.empty(0, dtype=np.int64), np.empty(0))
            if k == "list"
            else np.empty(0)
            for n, k, _, _ in props
        }
    if not any(kind == "list" for _, kind, _, _ in props):
        fields = [(n, t, None) for n, _, t, _ in props]
        rows = reader.read(count, fields)
        return {n: rows[n] for n, _, _, _ in props}
    # lists: assume every row has the sizes of the first one, verify, else go row by row
    start = reader.pos
    row = _read_rows_slow(reader, element, 1)
    sizes = {n: len(row[n][1]) for n, kind, _, _ in props if kind == "list"}
    reader.pos = start
    fields = []
    for name, kind, type_, count_type in props:
        if kind == "list":
            if reader.ascii:
                fields.append((name + "#n", "f8", None))
                fields.append((name, "f8", (sizes[name],)))
            else:
                fields.append((name + "#n", count_type, None))
                fields.append((name, type_, (sizes[name],)))
        else:
            fields.append((name, "f8" if reader.ascii else type_, None))
    try:
        rows = reader.read(count, fields)
        if all(
            (rows[n + "#n"] == sizes[n]).all()
            for n, kind, _, _ in props
            if kind == "list"
        ):
            out = {}
            for name, kind, _, _ in props:
                if kind == "list":
                    out[name] = (
                        np.full(count, sizes[name], dtype=np.int64),
                        rows[name].reshape(-1),
                    )
                else:
                    out[name] = rows[name]
            return out
    except ValueError:
        pass
    reader.pos = start
    parsed = _read_rows_slow(reader, element, count)
    return parsed


def _read_rows_slow(reader, element, count):
    r"""Row by row, for lists with varying lengths."""
    props = element["props"]
    scalars = {n: [] for n, k, _, _ in props if k == "scalar"}
    counts = {n: [] for n, k, _, _ in props if k == "list"}
    values = {n: [] for n, k, _, _ in props if k == "list"}
    for _ in range(count):
        for name, kind, type_, count_type in props:
            if kind == "scalar":
                scalars[name].append(_take(reader, type_, 1)[0])
            else:
                n = int(_take(reader, count_type, 1)[0])
                counts[name].append(n)
                values[name].append(_take(reader, type_, n))
    out = {n: np.array(v) for n, v in scalars.items()}
    for name in counts:
        flat = np.concatenate(values[name]) if values[name] else np.empty(0)
        out[name] = (np.array(counts[name], dtype=np.int64), flat)
    return out


def _take(reader, type_, n):
    if reader.ascii:
        out = reader.tokens[reader.pos : reader.pos + n]
        reader.pos += n
        if len(out) < n:
            raise ValueError("The PLY file is truncated")
        return out
    dtype = np.dtype(reader.endian + type_)
    if reader.pos + n * dtype.itemsize > len(reader.raw):
        raise ValueError("The PLY file is truncated")
    out = np.frombuffer(reader.raw, dtype=dtype, count=n, offset=reader.pos)
    reader.pos += n * dtype.itemsize
    return out


def _duplicate_for_face_texture(
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


def read_ply(path):
    r"""Read a PLY file.

    Returns
    -------
    data : `PlyData`
    """
    with open(str(path), "rb") as f:
        raw = f.read()
    fmt, elements, body_start = _parse_header(raw)
    reader = _Reader(fmt, raw, body_start)
    vertex = faces = None
    for element in elements:
        values = _read_element(reader, element)
        if element["name"] == "vertex":
            vertex = values
        elif element["name"] == "face":
            faces = values
    if vertex is None or not all(k in vertex for k in "xyz"):
        raise ValueError("Cannot read geometry: no vertex x, y, z")

    points = np.stack([vertex[k] for k in "xyz"], axis=1).astype(np.float32)

    tcoords = None
    for u, v in (("u", "v"), ("texture_u", "texture_v"), ("s", "t")):
        if u in vertex and v in vertex:
            tcoords = np.stack([vertex[u], vertex[v]], axis=1).astype(np.float32)
            break

    colours = None
    for names in (
        ("red", "green", "blue"),
        ("diffuse_red", "diffuse_green", "diffuse_blue"),
    ):
        if all(n in vertex for n in names):
            channels = [vertex[n] for n in names]
            if "alpha" in vertex:
                channels.append(vertex["alpha"])
            colours = np.stack(channels, axis=1).astype(np.uint8)
            break

    counts = np.empty(0, dtype=np.int64)
    connectivity = np.empty(0, dtype=np.int64)
    face_tex = None
    if faces is not None:
        for name in ("vertex_indices", "vertex_index"):
            if name in faces:
                counts, connectivity = faces[name]
                connectivity = connectivity.astype(np.int64)
                break
        if tcoords is None and "texcoord" in faces:
            face_tex = faces["texcoord"]
    if len(connectivity) and (
        connectivity.min() < 0 or connectivity.max() >= len(points)
    ):
        raise ValueError("A face refers to a vertex that does not exist")
    counts = np.asarray(counts, dtype=np.int64)
    if face_tex is not None:
        points, colours, tcoords, connectivity = _duplicate_for_face_texture(
            points, colours, counts, connectivity, face_tex[0], face_tex[1]
        )
    return PlyData(points, tcoords, colours, counts, connectivity)
