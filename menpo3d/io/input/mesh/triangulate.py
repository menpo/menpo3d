r"""Split polygons into triangles the way VTK's ``vtkTriangleFilter`` does.

menpo3d's importers used to hand non-triangular meshes to VTK, so existing datasets
depend on which diagonal VTK picked (and on the order and vertex order of the triangles
it produced). Quads reproduce ``vtkPolygon::SimpleTriangulation`` exactly. Polygons with
five or more sides, and degenerate quads, use an ear-clipping port of
``vtkPolygon::EarCutTriangulation`` that follows the same rules but is not guaranteed to
break every tie the same way, so a warning is always raised when anything is retriangulated.
"""

import math
import sys
import warnings

import numpy as np

_TOLERANCE = 1.0e-06  # vtkPolygon::Tolerance
_EPSILON = sys.float_info.epsilon


def _simple_quads(p):
    r"""``vtkPolygon``'s special case for quads, vectorised.

    Parameters
    ----------
    p : ``(n, 4, 3)`` `ndarray`

    Returns
    -------
    use_d1, ok : ``(n,)`` boolean `ndarray`
        ``use_d1`` selects the triangles (0 1 2)(0 2 3), otherwise (0 1 3)(1 2 3).
        ``ok`` is False where VTK's quad special case gives up (degenerate quads).
    """
    x0, x1, x2, x3 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    d1, d2 = x2 - x0, x3 - x1
    d1_n2, d2_n2 = (d1 * d1).sum(1), (d2 * d2).sum(1)
    mins, maxs = p.min(axis=1), p.max(axis=1)
    tol = _TOLERANCE * np.linalg.norm(maxs - mins, axis=1)
    tol2 = tol * tol

    use_d1 = d1_n2 < d2_n2
    ok = np.where(use_d1, d1_n2, d2_n2) >= tol2

    def fan(use_d1):
        d = np.where(use_d1[:, None], d1, d2)
        v1 = np.where(use_d1[:, None], x1 - x0, x2 - x1)
        v3 = np.where(use_d1[:, None], x3 - x0, x0 - x1)
        return v1, v3, np.cross(v1, d), np.cross(d, v3)

    v1, v3, n1, n2 = fan(use_d1)
    sq = lambda v: (v * v).sum(1)  # noqa: E731
    ok &= (sq(v1) >= tol2) & (sq(v3) >= tol2) & (sq(n1) >= tol2) & (sq(n2) >= tol2)
    normal = n1 + n2
    norm = np.linalg.norm(normal, axis=1)
    ok &= norm != 0
    with np.errstate(invalid="ignore", divide="ignore"):
        normal = normal / norm[:, None]
    a1, a2 = (n1 * normal).sum(1), (n2 * normal).sum(1)
    ok &= (a1 != 0) & ((a1 < 0) | (a2 != 0))
    concave = (a1 < 0) | (a2 < 0)

    # concave: try the other diagonal; give up if that is not a valid arrowhead.
    # (VTK builds these vectors with the fan of one diagonal crossed with the other.)
    flipped = use_d1 != concave
    v1b = np.where(flipped[:, None], x1 - x0, x2 - x1)
    v3b = np.where(flipped[:, None], x3 - x0, x0 - x1)
    db = np.where(flipped[:, None], d2, d1)
    n1b, n2b = np.cross(v1b, db), np.cross(db, v3b)
    other_ok = (
        (sq(v1b) >= tol2)
        & (sq(v3b) >= tol2)
        & ((n1b * normal).sum(1) > 0)
        & ((n2b * normal).sum(1) > 0)
    )
    ok &= ~concave | other_ok
    return np.where(concave, flipped, use_d1), ok


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


class _PriorityQueue:
    r"""``vtkPriorityQueue``: a binary min-heap whose tie behaviour decides which ear is cut."""

    def __init__(self):
        self.items = []  # [priority, id]
        self.location = {}

    def __len__(self):
        return len(self.items)

    def insert(self, priority, id_):
        if self.location.get(id_, -1) != -1:
            return
        self.items.append([priority, id_])
        i = len(self.items) - 1
        self.location[id_] = i
        while i > 0:
            parent = (i - 1) // 2
            if not self.items[i][0] < self.items[parent][0]:
                break
            self._swap(i, parent)
            i = parent

    def _swap(self, i, j):
        a, b = self.items[i], self.items[j]
        self.items[i], self.items[j] = b, a
        self.location[b[1]], self.location[a[1]] = i, j

    def pop(self, location=0):
        items = self.items
        id_ = items[location][1]
        last = len(items) - 1
        if location == last:  # removing the last item: nothing to move
            items.pop()
            self.location[id_] = -1
            return id_
        items[location] = items[last]
        self.location[items[location][1]] = location
        self.location[id_] = -1
        items.pop()
        last -= 1  # index of the last item, ``MaxId`` in VTK
        if last <= 0:
            return id_
        last_node_to_check = (last - 1) // 2
        i = location
        while i <= last_node_to_check:
            idx = 2 * i + 1
            j = idx if (idx == last or items[idx][0] < items[idx + 1][0]) else idx + 1
            if items[i][0] > items[j][0]:
                self._swap(i, j)
                i = j
            else:
                break
        i = location
        while i > 0:
            parent = (i - 1) // 2
            if items[i][0] < items[parent][0]:
                self._swap(i, parent)
                i = parent
            else:
                break
        return id_

    def delete(self, id_):
        location = self.location.get(id_, -1)
        if location != -1:
            self.pop(location)


def _lines_intersect(a1, a2, b1, b2, tolerance):
    r"""``vtkLine::Intersection`` with ``AbsoluteFuzzy`` tolerance, as a boolean."""
    a21, b21, b1a1 = _sub(a2, a1), _sub(b2, b1), _sub(b1, a1)
    a00, a01 = _dot(a21, a21), -_dot(a21, b21)
    a10, a11 = a01, _dot(b21, b21)
    c0, c1 = _dot(a21, b1a1), -_dot(b21, b1a1)
    # vtkMath::SolveLinearSystemGEPP2x2
    b0, b1_ = c0, c1
    swapped = False
    if a00 == 0 or a01 == 0 or a10 == 0 or a11 == 0:
        if a01 == 0 or a11 == 0:
            a00, a01 = a01, a00
            a10, a11 = a11, a10
            swapped = True
        if a00 == 0:
            a00, a10 = a10, a00
            a01, a11 = a11, a01
            b0, b1_ = b1_, b0
    else:
        if abs(a00) < abs(a10):
            a00, a10 = a10, a00
            a01, a11 = a11, a01
            b0, b1_ = b1_, b0
        f = -a10 / a00
        a11 += a01 * f
        b1_ += b0 * f
    if abs(a11) < 256 * _EPSILON or a11 == 0 or a00 == 0:
        return True  # collinear: VTK reports OnLine, which also blocks the ear
    x1 = b1_ / a11
    x0 = (b0 - a01 * x1) / a00
    if not (math.isfinite(x0) and math.isfinite(x1)):
        return True
    if swapped:
        x0, x1 = x1, x0
    u, v = x0, x1
    ptu = tuple(a21[k] * u + a1[k] for k in range(3))
    ptv = tuple(b21[k] * v + b1[k] for k in range(3))
    diff = _sub(ptu, ptv)
    tol2 = tolerance * tolerance
    if _dot(diff, diff) > tol2:
        return False
    if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
        return True
    if tol2 > 0.0:
        u_tol = math.sqrt(tol2 / _dot(a21, a21))
        v_tol = math.sqrt(tol2 / _dot(b21, b21))
        return -u_tol <= u <= 1.0 + u_tol and -v_tol <= v <= 1.0 + v_tol
    return False


def _earcut(points):
    r"""Ear clipping of one polygon, ported from ``vtkPolygon::EarCutTriangulation``.

    Returns ``(triangles, success)``: a list of local index triples (VTK keeps the ones
    found before giving up) and whether the whole polygon was triangulated.
    """
    n = len(points)
    if n < 3:
        return [], False
    pts = [tuple(float(c) for c in p) for p in points]
    lo = [min(p[k] for p in pts) for k in range(3)]
    hi = [max(p[k] for p in pts) for k in range(3)]
    tol = _TOLERANCE * _norm(_sub(hi, lo))
    tol2 = tol * tol
    tol = math.sqrt(tol2) if tol2 > 0.0 else 0.0
    nxt = [(i + 1) % n for i in range(n)]
    prv = [(i - 1) % n for i in range(n)]
    head, count = 0, n
    # drop coincident neighbours
    v = head
    for _ in range(n):
        w = nxt[v]
        d = _sub(pts[v], pts[w])
        if _dot(d, d) < tol2:
            prv[nxt[w]] = v
            nxt[v] = nxt[w]
            if w == head:
                head = v
            count -= 1
        else:
            v = w
    # polygon normal
    normal = (0.0, 0.0, 0.0)
    anchor = pts[head]
    v = nxt[head]
    while nxt[v] != head:
        c = _cross(_sub(pts[v], anchor), _sub(pts[nxt[v]], anchor))
        normal = (normal[0] + c[0], normal[1] + c[1], normal[2] + c[2])
        v = nxt[v]
    length = _norm(normal)
    if length == 0.0:
        return [], False
    normal = (normal[0] / length, normal[1] / length, normal[2] / length)

    measures = {}

    def measure(v):
        x, a, b = pts[v], pts[prv[v]], pts[nxt[v]]
        v1, v2, v3 = _sub(x, a), _sub(b, x), _sub(a, b)
        area = _dot(_cross(v1, v2), normal)
        if area < 0.0:
            return -1.0
        if area == 0.0:
            return -sys.float_info.max
        perimeter = _norm(v1) + _norm(v2) + _norm(v3)
        measures[v] = perimeter * perimeter / area
        return measures[v]

    def can_remove(v):
        if count <= 3:
            return True
        p, q = prv[v], nxt[v]
        s_n = _cross(_sub(pts[q], pts[p]), normal)
        den = _norm(s_n)
        if den == 0.0:
            return False
        s_n = (s_n[0] / den, s_n[1] / den, s_n[2] / den)

        def sign(w):
            val = _dot(s_n, _sub(pts[w], pts[p]))
            return 1 if val > tol else (-1 if val < -tol else 0)

        current = sign(nxt[q])
        one_negative = current < 0
        w = nxt[nxt[q]]
        while w != p:
            s = sign(w)
            if s != current:
                if not one_negative:
                    one_negative = s < 0
                if _lines_intersect(pts[p], pts[q], pts[w], pts[prv[w]], tol):
                    return False
                current = s
            w = nxt[w]
        return one_negative

    queue = _PriorityQueue()
    v = head
    for _ in range(count):
        m = measure(v)
        if m > 0.0:
            queue.insert(m, v)
        v = nxt[v]
    triangles = []
    while count > 2 and len(queue) > 0:
        v = queue.pop()
        if not can_remove(v):
            continue
        triangles.append((v, nxt[v], prv[v]))
        count -= 1
        if count < 3:
            break
        if v == head:
            head = nxt[v]
        p, q = prv[v], nxt[v]
        nxt[p], prv[q] = q, p
        queue.delete(p)
        queue.delete(q)
        for w in (p, q):
            m = measure(w)
            if m > 0.0:
                queue.insert(m, w)
    return triangles, count <= 2


def _quad_fallback(points):
    r"""What VTK ends up with when the ear cut of a quad fails.

    ``vtkPolygon`` hands the quad to ``vtkQuad``, which should pick the shorter diagonal, but
    the copy it works on has no points, so (as measured on random quads) it always returns
    the triangles (0 1 2)(0 2 3).
    """
    return [(0, 1, 2), (0, 2, 3)]


def _kept(n_out, n_found):
    r"""Rows of the output to keep when some polygons produced fewer triangles than expected."""
    if np.array_equal(n_out, n_found):
        return None
    starts = np.cumsum(n_out) - n_out
    keep = np.zeros(n_out.sum(), dtype=bool)
    for start, found in zip(starts, n_found):
        keep[start : start + found] = True
    return keep


def triangulate_polygons(points, counts, connectivity, warn=True):
    r"""Triangulate polygons exactly as VTK would have.

    Parameters
    ----------
    points : ``(n_points, 3)`` `ndarray`
    counts : ``(n_polys,)`` `ndarray`
        Number of vertices of each polygon.
    connectivity : ``(sum(counts),)`` `ndarray`
        The vertex indices of the polygons, one after the other.
    warn : `bool`, optional
        Warn when any polygon had to be split.

    Returns
    -------
    trilist : ``(n_tris, 3)`` `ndarray`
        Triangles in polygon order (triangles pass through unchanged).
    """
    counts = np.asarray(counts)
    connectivity = np.asarray(connectivity, dtype=np.int64)
    if len(counts) == 0:
        return np.empty((0, 3), dtype=np.int64)
    if np.all(counts == 3):
        return connectivity.reshape(-1, 3)

    if warn:
        warnings.warn(
            "Non-triangular mesh connectivity was detected - it is being coerced into a "
            "triangular mesh the way VTK did (quads along their shorter diagonal). "
            "Polygons with more than four sides may be split differently to VTK."
        )

    starts = np.cumsum(counts) - counts
    n_out = np.maximum(counts - 2, 0)
    out_start = np.cumsum(n_out) - n_out
    trilist = np.empty((n_out.sum(), 3), dtype=np.int64)
    n_found = n_out.copy()

    tris = np.flatnonzero(counts == 3)
    trilist[out_start[tris]] = connectivity[starts[tris, None] + np.arange(3)]

    quads = np.flatnonzero(counts == 4)
    if len(quads):
        ids = connectivity[starts[quads, None] + np.arange(4)]  # (q, 4)
        use_d1, ok = _simple_quads(points[ids])
        first = np.where(use_d1[:, None], ids[:, [0, 1, 2]], ids[:, [0, 1, 3]])
        second = np.where(use_d1[:, None], ids[:, [0, 2, 3]], ids[:, [1, 2, 3]])
        for k in np.flatnonzero(~ok):  # degenerate quads: the general algorithm
            local, success = _earcut(points[ids[k]])
            if not success:
                local = _quad_fallback(points[ids[k]])
            first[k], second[k] = ids[k][list(local[0])], ids[k][list(local[1])]
        trilist[out_start[quads]] = first
        trilist[out_start[quads] + 1] = second

    for poly in np.flatnonzero(counts > 4):
        ids = connectivity[starts[poly] : starts[poly] + counts[poly]]
        local, _ = _earcut(
            points[ids]
        )  # on failure VTK keeps the triangles found so far
        n_found[poly] = len(local)
        if local:
            trilist[out_start[poly] : out_start[poly] + len(local)] = ids[
                np.array(local)
            ]
    return trilist[keep] if (keep := _kept(n_out, n_found)) is not None else trilist
