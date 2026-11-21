r"""A VTK-free VRML 2.0 (``.wrl``) reader that reproduces ``vtkVRMLImporter``'s topology.

Only what menpo3d used from VTK is supported: the first ``Shape`` that has an
``IndexedFaceSet`` (``Transform`` nodes are ignored, as the importer only used the
geometry). As in VTK the ``Coordinate`` points stay the mesh points (unused ones included)
unless the texture coordinate or normal indices differ from ``coordIndex`` anywhere, in
which case every face corner becomes its own point.
"""

import re
import warnings

import numpy as np

_TOKEN = re.compile(r'#[^\n]*|"(?:[^"\\]|\\.)*"|[\[\]{}]|[^\s\[\]{},"#]+')


class Node:
    def __init__(self, type_):
        self.type = type_
        self.fields = {}

    def __repr__(self):
        return "Node({}, {})".format(self.type, list(self.fields))


def _tokenise(text):
    return [t for t in _TOKEN.findall(text) if t[0] != "#"]


class _Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.defs = {}

    def parse(self):
        nodes = []
        while self.pos < len(self.tokens):
            node = self.statement()
            if node is not None:
                nodes.append(node)
        return nodes

    def skip_group(self, open_, close):
        depth = 0
        while self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            self.pos += 1
            if token == open_:
                depth += 1
            elif token == close:
                depth -= 1
                if depth == 0:
                    return

    def statement(self):
        token = self.tokens[self.pos]
        if token == "PROTO":
            self.pos += 2
            self.skip_group("[", "]")
            self.skip_group("{", "}")
            return None
        if token == "EXTERNPROTO":
            self.pos += 2
            self.skip_group("[", "]")
            self.pos += 1  # the url (or a list of them)
            if self.pos < len(self.tokens) and self.tokens[self.pos - 1] == "[":
                self.pos -= 1
                self.skip_group("[", "]")
            return None
        if token == "ROUTE":
            self.pos += 4
            return None
        return self.node()

    def node(self):
        token = self.tokens[self.pos]
        self.pos += 1
        if token == "DEF":
            name = self.tokens[self.pos]
            self.pos += 1
            node = self.node()
            self.defs[name] = node
            return node
        if token == "USE":
            name = self.tokens[self.pos]
            self.pos += 1
            return self.defs.get(name)
        if token == "NULL":
            return None
        node = Node(token)
        if self.tokens[self.pos] != "{":
            raise ValueError("Expected '{{' after {} in the VRML file".format(token))
        self.pos += 1
        while self.tokens[self.pos] != "}":
            name = self.tokens[self.pos]
            self.pos += 1
            if name in ("eventIn", "eventOut", "field", "exposedField"):  # Script nodes
                self.pos += 2
                if name in ("field", "exposedField"):
                    self.value()
                continue
            if name == "IS":
                self.pos += 1
                continue
            node.fields[name] = self.value()
        self.pos += 1
        return node

    def value(self):
        tokens = self.tokens
        token = tokens[self.pos]
        if token == "[":
            self.pos += 1
            end = tokens.index("]", self.pos)
            chunk = tokens[self.pos : end]
            if "{" not in chunk:
                self.pos = end + 1
                try:
                    return np.array(chunk, dtype=np.float64)
                except ValueError:
                    try:
                        return np.array([int(c, 0) for c in chunk], dtype=np.float64)
                    except ValueError:
                        return [c.strip('"') for c in chunk]
            items = []
            while tokens[self.pos] != "]":
                items.append(self.node())
            self.pos += 1
            return items
        if token in ("TRUE", "FALSE"):
            self.pos += 1
            return token == "TRUE"
        if token[0] == '"':
            self.pos += 1
            return token[1:-1]
        if token in ("DEF", "USE", "NULL") or (
            self.pos + 1 < len(tokens) and tokens[self.pos + 1] == "{"
        ):
            return self.node()
        numbers = []
        while self.pos < len(tokens) and _is_number(tokens[self.pos]):
            token = tokens[self.pos]
            numbers.append(
                float(int(token, 16)) if "x" in token.lower() else float(token)
            )
            self.pos += 1
        if not numbers:
            raise ValueError("Cannot parse the VRML value {!r}".format(token))
        return numbers[0] if len(numbers) == 1 else np.array(numbers)


def _is_number(token):
    try:
        float(token)
        return True
    except ValueError:
        return token[:2].lower() == "0x"


def _shapes(nodes):
    for node in nodes:
        if node is None:
            continue
        if node.type == "Shape":
            yield node
        for value in node.fields.values():
            if isinstance(value, Node):
                yield from _shapes([value])
            elif isinstance(value, list) and value and isinstance(value[0], Node):
                yield from _shapes(value)


def _split_faces(indices):
    r"""``coordIndex``-style ``-1`` terminated lists to ``(counts, flat)``."""
    indices = np.asarray(indices, dtype=np.int64)
    ends = np.flatnonzero(indices == -1)
    counts = np.diff(np.concatenate([[-1], ends])) - 1
    flat = indices[indices != -1]
    if len(indices) and indices[-1] != -1:  # a last face without the terminating -1
        counts = np.append(counts, len(indices) - (ends[-1] + 1 if len(ends) else 0))
    return counts, flat


class WrlData:
    def __init__(self, points, counts, connectivity, tcoords, colours):
        self.points = points  # (n, 3) float32
        self.counts = counts
        self.connectivity = connectivity
        self.tcoords = tcoords  # (n, 2) float32 or None
        self.colours = colours  # (m, 4) float64 in [0, 1] or None


def read_wrl(path):
    r"""Read the first ``IndexedFaceSet`` in a VRML 2.0 file.

    Returns
    -------
    data : `WrlData`
    """
    with open(str(path), "rb") as f:
        text = f.read().decode("utf-8", "replace")
    if not text.lstrip("﻿").startswith("#VRML V2.0"):
        raise ValueError("Only VRML 2.0 files are supported")
    parser = _Parser(_tokenise(text))
    shapes = [
        s
        for s in _shapes(parser.parse())
        if isinstance(s.fields.get("geometry"), Node)
        and s.fields["geometry"].type
        in ("IndexedFaceSet", "IndexedLineSet", "PointSet")
    ]
    if not shapes:
        raise ValueError(
            "No IndexedFaceSet, IndexedLineSet or PointSet was found in the VRML file"
        )
    if len(shapes) > 1:
        warnings.warn(
            "More than one actor was detected in the scene. Only "
            "single scene actors are currently supported."
        )
    ifs = shapes[0].fields["geometry"]

    def points_of(field, name, width):
        node = ifs.fields.get(field)
        if not isinstance(node, Node):
            return None
        values = node.fields.get(name)
        if values is None:
            return None
        return (
            np.asarray(values, dtype=np.float64).reshape(-1, width).astype(np.float32)
        )

    coord = points_of("coord", "point", 3)
    if coord is None:
        raise ValueError("The IndexedFaceSet has no coordinates")
    if ifs.type == "IndexedFaceSet":
        counts, connectivity = _split_faces(ifs.fields.get("coordIndex", []))
    else:  # lines and points have no faces, the points are all there is
        counts, connectivity = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if len(connectivity) and (
        connectivity.min() < 0 or connectivity.max() >= len(coord)
    ):
        raise ValueError("A face refers to a vertex that does not exist")
    tcoords = points_of("texCoord", "point", 2)
    normals = points_of("normal", "vector", 3)
    tcoord_cells = (
        _split_faces(ifs.fields["texCoordIndex"])
        if "texCoordIndex" in ifs.fields
        else None
    )
    normal_cells = (
        _split_faces(ifs.fields["normalIndex"]) if "normalIndex" in ifs.fields else None
    )

    colours = None
    color = ifs.fields.get("color")
    if isinstance(color, Node) and "color" in color.fields:
        rgb = np.asarray(color.fields["color"], dtype=np.float64).reshape(-1, 3)
        rgba = np.floor(np.c_[rgb, np.ones(len(rgb))] * 255 + 0.5)  # vtkLookupTable
        colours = rgba / 255.0

    # --- vtkVRMLImporter: do the texture coordinates and normals line up with the points?
    def cells_match(cells, name):
        if cells[0].shape != counts.shape:
            warnings.warn(
                "Number of faces does not match {} faces, output may not be correct".format(
                    name
                )
            )
            return True
        if not np.array_equal(cells[0], counts):
            return True  # VTK reports "Face size differs" and stops checking
        return bool(np.array_equal(cells[1], connectivity))

    have_tex = tcoords is not None and tcoord_cells is not None
    have_nrm = normals is not None and normal_cells is not None
    if not have_tex and not have_nrm:
        corresponds = True
    elif tcoords is not None and len(tcoords) != len(coord):
        corresponds = False
    elif normals is not None and len(normals) != len(coord):
        corresponds = False
    else:
        corresponds = True
        if tcoord_cells is not None and tcoord_cells[0].shape != counts.shape:
            warnings.warn("Number of faces does not match texture faces")
        elif normal_cells is not None and normal_cells[0].shape != counts.shape:
            warnings.warn("Number of faces does not match normal faces")
        else:
            if have_tex and not cells_match(tcoord_cells, "texture"):
                corresponds = False
            if corresponds and have_nrm and not cells_match(normal_cells, "normal"):
                corresponds = False

    if corresponds:
        return WrlData(coord, counts, connectivity, tcoords, colours)

    # rejig: one point per face corner, polygons whose texture/normal faces differ in size are dropped
    starts = np.cumsum(counts) - counts
    tex_counts = tcoord_cells[0] if tcoord_cells is not None else np.zeros_like(counts)
    tex_starts = (
        (np.cumsum(tex_counts) - tex_counts) if tcoord_cells is not None else None
    )
    nrm_counts = normal_cells[0] if normal_cells is not None else np.zeros_like(counts)
    keep = np.ones(len(counts), dtype=bool)

    def same_size(other_counts):
        # faces without a counterpart (the other list is shorter) are dropped, like faces of another size
        n = min(len(counts), len(other_counts))
        equal = np.zeros(len(counts), dtype=bool)
        equal[:n] = counts[:n] == other_counts[:n]
        return equal

    if tcoords is not None and len(tcoords) > 0:
        keep &= same_size(tex_counts)
    if normals is not None and len(normals) > 0:
        keep &= same_size(nrm_counts)
    corner_face = np.repeat(np.arange(len(counts)), counts)
    corner_kept = keep[corner_face]
    new_connectivity = connectivity[corner_kept]
    points = coord[new_connectivity]
    new_tcoords = None
    if tcoord_cells is not None and tcoords is not None and len(tcoords) > 0:
        # the texture corners of each kept face (its texCoordIndex face, which has the same size)
        kept = np.flatnonzero(keep)
        sizes = counts[kept]
        corner_in_face = np.arange(sizes.sum()) - np.repeat(
            np.cumsum(sizes) - sizes, sizes
        )
        new_tcoords = tcoords[
            tcoord_cells[1][np.repeat(tex_starts[kept], sizes) + corner_in_face]
        ]
        if len(new_tcoords) != len(points):
            new_tcoords = None  # VTK would hand back an array of the wrong length
    return WrlData(
        points,
        counts[keep],
        np.arange(len(points), dtype=np.int64),
        new_tcoords,
        None,
    )
