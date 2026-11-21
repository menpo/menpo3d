r"""A VTK-free STL reader that reproduces ``vtkSTLReader``.

STL stores three vertices per triangle. ``vtkSTLReader`` merges identical vertices
(exact float32 equality) in order of first appearance and drops triangles that
collapse as a result, which is what existing datasets are built against.
"""

import numpy as np

_BINARY_FACET = np.dtype(
    [("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attributes", "<u2")]
)


def _read_ascii(text):
    r"""The vertices of an ASCII STL. Like VTK, which reads nothing from an incomplete file, every facet
    must be complete and the solid must end with ``endsolid``."""
    vertices = []
    seen = dict.fromkeys(("facet", "outer", "endloop", "endfacet", "endsolid"), 0)
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        word = fields[0].lower()
        if word == "vertex" and len(fields) >= 4:
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
        elif word in seen:
            seen[word] += 1
    n = seen["facet"]
    if not (
        seen["endsolid"] >= 1
        and seen["outer"] == seen["endloop"] == seen["endfacet"] == n
        and len(vertices) == 3 * n
    ):
        raise ValueError(
            "The ASCII STL file is truncated or malformed: every facet needs 'outer loop', "
            "three vertices, 'endloop' and 'endfacet', and the solid 'endsolid'"
        )
    return np.array(vertices, dtype=np.float64).reshape(-1, 3).astype(np.float32)


def _binary_facet_count(raw):
    r"""The number of facets, from the size of the file: the count in the header is ignored (many writers
    leave it 0 or wrong) and the size must be exactly the header plus whole facets, as for VTK."""
    if len(raw) < 84:
        raise ValueError("Not an STL file: it is too short to hold a header")
    n_facets, rest = divmod(len(raw) - 84, _BINARY_FACET.itemsize)
    if rest:
        raise ValueError(
            "The binary STL file is truncated: its size is not 84 bytes plus whole 50 byte facets"
        )
    return n_facets


def _read_binary(raw):
    n_facets = _binary_facet_count(raw)
    facets = np.frombuffer(raw, dtype=_BINARY_FACET, count=n_facets, offset=84)
    return facets["vertices"].reshape(-1, 3)


def merge_points(vertices):
    r"""Merge identical points, in order of first appearance.

    Parameters
    ----------
    vertices : ``(n, 3)`` `ndarray`

    Returns
    -------
    points, indices : ``(m, 3)`` `ndarray`, ``(n,)`` `ndarray`
    """
    if len(vertices) == 0:
        return vertices, np.empty(0, dtype=np.int64)
    # sort lexicographically; equal rows become adjacent (0.0 and -0.0 compare equal)
    order = np.lexsort(vertices.T[::-1])
    sorted_v = vertices[order]
    new_group = np.empty(len(order), dtype=bool)
    new_group[0] = True
    new_group[1:] = (sorted_v[1:] != sorted_v[:-1]).any(axis=1)
    group = np.cumsum(new_group) - 1
    first_seen = np.minimum.reduceat(order, np.flatnonzero(new_group))
    rank = np.empty(len(first_seen), dtype=np.int64)
    rank[np.argsort(first_seen, kind="stable")] = np.arange(len(first_seen))
    indices = np.empty(len(order), dtype=np.int64)
    indices[order] = rank[group]
    return vertices[np.sort(first_seen)], indices


def read_stl(path):
    r"""Read an STL file (ASCII or binary) into merged points and triangles.

    Returns
    -------
    points, trilist : ``(n, 3)`` float32 `ndarray`, ``(m, 3)`` int64 `ndarray`
    """
    with open(str(path), "rb") as f:
        raw = f.read()
    if raw[:5] == b"solid":
        # ASCII, unless it is a binary file whose 80 byte header happens to start with "solid"
        try:
            vertices = _read_ascii(raw.decode("utf-8", "replace"))
        except ValueError:
            if len(raw) >= 84 and len(raw) == 84 + 50 * int(
                np.frombuffer(raw, dtype="<u4", count=1, offset=80)[0]
            ):
                vertices = _read_binary(raw)
            else:
                raise
    else:
        vertices = _read_binary(raw)
    points, indices = merge_points(vertices)
    trilist = indices.reshape(-1, 3)
    keep = (
        (trilist[:, 0] != trilist[:, 1])
        & (trilist[:, 0] != trilist[:, 2])
        & (trilist[:, 1] != trilist[:, 2])
    )
    return points, trilist[keep]
