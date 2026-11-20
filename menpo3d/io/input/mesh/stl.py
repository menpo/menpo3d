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
    vertices = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[0].lower() == "vertex":
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
    if len(vertices) % 3:
        raise ValueError("The STL file has a facet with fewer than three vertices")
    return np.array(vertices, dtype=np.float64).astype(np.float32)


def _read_binary(raw):
    if len(raw) < 84:  # no room for a header and a facet count: an empty mesh
        return np.empty((0, 3), dtype=np.float32)
    n_facets = int(np.frombuffer(raw, dtype="<u4", count=1, offset=80)[0])
    if len(raw) < 84 + n_facets * _BINARY_FACET.itemsize:
        raise ValueError("The STL file is truncated")
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
    vertices = None
    if raw[:5] == b"solid":
        try:
            vertices = _read_ascii(raw.decode("utf-8", "replace"))
            if len(vertices) == 0:
                vertices = None
        except ValueError:
            vertices = None
    if vertices is None:
        vertices = _read_binary(raw)
    points, indices = merge_points(vertices)
    trilist = indices.reshape(-1, 3)
    keep = (
        (trilist[:, 0] != trilist[:, 1])
        & (trilist[:, 0] != trilist[:, 2])
        & (trilist[:, 1] != trilist[:, 2])
    )
    return points, trilist[keep]
