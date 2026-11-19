import numpy as np
from menpo3d.rasterize import model_to_clip_transform
from menpo.shape import PointCloud


def render_hi_res_shape_image(mesh, render_width=3000):
    from .rasterize import GLRasterizer

    h, w = mesh.range()[:2]
    aspect_ratio = w / h
    height = render_width * aspect_ratio

    r = GLRasterizer(
        projection_matrix=model_to_clip_transform(mesh).h_matrix,
        width=render_width,
        height=height,
    )
    return r.model_to_image_transform, r.rasterize_mesh_with_shape_image(mesh)[1]


def per_vertex_occlusion(mesh, err_proportion=0.0001, err_norm="z", render_width=3000):
    # Render a high-resolution shape image for visibility testing
    model_to_image_transform, shape_image = render_hi_res_shape_image(
        mesh, render_width=render_width
    )
    # err_proportion=0.01 is 1% deviation of total range of 3D shape
    err_scale = (
        mesh.range()[2].sum() if err_norm == "z" else np.sqrt((mesh.range() ** 2).sum())
    )
    threshold = err_scale * err_proportion

    sample_points_3d = mesh
    sample_points_2d = model_to_image_transform.apply(sample_points_3d)

    xyz_found = shape_image.as_unmasked().sample(sample_points_2d, order=1).T
    err = np.sum((xyz_found - sample_points_3d.points) ** 2, axis=1)
    return err < threshold


def per_vertex_occlusion_accurate(mesh, chunk_size=20000):
    r"""Exact per-vertex visibility looking down the z axis (from -z towards +z).

    A vertex is visible if the first surface hit by a ray cast along +z from below
    the mesh is that vertex (to within a tolerance of a thousandth of the mean edge
    length). Uses NumPy and SciPy only.
    """
    from scipy.spatial import cKDTree

    tol = mesh.mean_edge_length() / 1000
    points, trilist = mesh.points, mesh.trilist
    a, b, c = (points[trilist[:, i], :] for i in range(3))
    # work in the xy plane: where a vertex's (x, y) falls inside a triangle, what is its z?
    xy_a, v0, v1 = a[:, :2], (b - a)[:, :2], (c - a)[:, :2]
    det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    nondegenerate = np.abs(det) > 0
    centroids = (a + b + c)[:, :2] / 3
    radius = np.maximum.reduce(
        [np.linalg.norm(p - centroids, axis=1) for p in (xy_a, b[:, :2], c[:, :2])]
    )
    tree = cKDTree(centroids)
    eps = 1e-9

    first_z = np.full(mesh.n_points, np.inf)
    for start in range(0, mesh.n_points, chunk_size):
        q = points[start : start + chunk_size]
        ball = tree.query_ball_point(q[:, :2], radius.max() * 1.0000001)
        lens = np.fromiter((len(x) for x in ball), int, len(q))
        qi = np.repeat(np.arange(len(q)), lens)
        t = np.concatenate([np.asarray(x, dtype=int) for x in ball])
        keep = nondegenerate[t] & (
            np.linalg.norm(centroids[t] - q[qi, :2], axis=1) <= radius[t] + 1e-12
        )
        qi, t = qi[keep], t[keep]
        d = q[qi, :2] - xy_a[t]
        u = (d[:, 0] * v1[t, 1] - d[:, 1] * v1[t, 0]) / det[t]
        v = (v0[t, 0] * d[:, 1] - v0[t, 1] * d[:, 0]) / det[t]
        inside = (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps)
        qi, t, u, v = qi[inside], t[inside], u[inside], v[inside]
        z = a[t, 2] + u * (b[t, 2] - a[t, 2]) + v * (c[t, 2] - a[t, 2])
        chunk_first = np.full(len(q), np.inf)
        np.minimum.at(chunk_first, qi, z)
        first_z[start : start + chunk_size] = chunk_first

    return points[:, 2] - first_z < tol


def extract_per_vertex_colour(mesh, image):
    return image.sample(PointCloud(mesh.points[:, :2])).T


def extract_per_vertex_colour_with_occlusion(mesh, image, render_width=3000):
    colours = extract_per_vertex_colour(mesh, image)
    mask = per_vertex_occlusion(mesh, render_width=render_width)
    return colours, mask


def extract_per_vertex_features(mesh, image, feature_f, diagonal_range=None):
    image = image.copy()
    image.landmarks["mesh_2d"] = mesh.with_dims([0, 1])
    if diagonal_range is not None:
        image = image.rescale_landmarks_to_diagonal_range(
            diagonal_range, group="mesh_2d"
        )
    feature_image = feature_f(image)
    return extract_per_vertex_colour(feature_image.landmarks["mesh_2d"], feature_image)
