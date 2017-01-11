import numpy as np
from menpo.transform.piecewiseaffine.base import barycentric_vectors
from menpo.image import BooleanImage, MaskedImage


def _pixels_to_check_python(start, end, _):
    pixel_locations = []
    tri_indices = []

    for i, ((s_x, s_y), (e_x, e_y)) in enumerate(zip(start, end)):
        for x in range(s_x, e_x):
            for y in range(s_y, e_y):
                pixel_locations.append((x, y))
                tri_indices.append(i)

    pixel_locations = np.array(pixel_locations)
    tri_indices = np.array(tri_indices)
    return pixel_locations, tri_indices


try:
    from .tripixel import pixels_to_check
except IOError:
    print('Falling back to CPU pixel checking')
    pixels_to_check = _pixels_to_check_python


def pixel_locations_and_tri_indices(mesh):
    vertex_trilist = mesh.points[mesh.trilist]
    start = np.floor(vertex_trilist.min(axis=1)[:, :2])
    end = np.ceil(vertex_trilist.max(axis=1)[:, :2])
    start = start.astype(int)
    end = end.astype(int)
    n_sites = np.product((end - start), axis=1).sum()
    return pixels_to_check(start, end, n_sites)


def alpha_beta(i, ij, ik, points):
    ip = points - i
    dot_jj = np.einsum('dt, dt -> t', ij, ij)
    dot_kk = np.einsum('dt, dt -> t', ik, ik)
    dot_jk = np.einsum('dt, dt -> t', ij, ik)
    dot_pj = np.einsum('dt, dt -> t', ip, ij)
    dot_pk = np.einsum('dt, dt -> t', ip, ik)

    d = 1.0/(dot_jj * dot_kk - dot_jk * dot_jk)
    alpha = (dot_kk * dot_pj - dot_jk * dot_pk) * d
    beta = (dot_jj * dot_pk - dot_jk * dot_pj) * d
    return alpha, beta


def xy_bcoords(mesh, tri_indices, pixel_locations):
    i, ij, ik = barycentric_vectors(mesh.points[:, :2], mesh.trilist)
    i = i[:, tri_indices]
    ij = ij[:, tri_indices]
    ik = ik[:, tri_indices]
    a, b = alpha_beta(i, ij, ik, pixel_locations.T)
    c = 1 - a - b
    bcoords = np.array([c, a, b]).T
    return bcoords


def tri_containment(bcoords):
    alpha, beta, _ = bcoords.T
    return np.logical_and(np.logical_and(
        alpha >= 0, beta >= 0),
        alpha + beta <= 1)


def z_values_for_bcoords(mesh, bcoords, tri_indices):
    return mesh.barycentric_coordinate_interpolation(
        mesh.points[:, -1][..., None], bcoords, tri_indices)[:, 0]


def pixel_sample_uniform(xy, n_samples):
    chosen_mask = np.random.permutation(np.arange(xy.shape[0]))[:n_samples]
    return xy[chosen_mask]


def unique_locations(xy, width, height):
    mask = np.zeros([width, height], dtype=np.bool)
    mask[xy[:, 0], xy[:, 1]] = True
    return np.vstack(np.nonzero(mask)).T


def location_to_index(xy, width):
    return xy[:, 0] * width + xy[:, 1]


def rasterize_barycentric_coordinates(mesh, image_shape):
    height, width = int(image_shape[0]), int(image_shape[1])
    # 1. Find all pixel-sites that may need to be rendered to
    #    + the triangle that may partake in rendering
    yx, tri_indices = pixel_locations_and_tri_indices(mesh)

    # 2. Limit to only pixel sites in the image
    out_of_bounds = np.logical_or(
        np.any(yx < 0, axis=1),
        np.any((np.array([height, width]) - yx) <= 0, axis=1))
    in_image = ~out_of_bounds
    yx = yx[in_image]
    tri_indices = tri_indices[in_image]

    # # Optionally limit to subset of pixels
    # if n_random_samples is not None:
    #     # 2. Find the unique pixel sites
    #     xy_u = unique_locations(yx, width, height)
    #
    #     xy_u = pixel_sample_uniform(xy_u, n_random_samples)
    #     to_keep = np.in1d(location_to_index(yx, width),
    #                       location_to_index(xy_u, width))
    #     yx = yx[to_keep]
    #     tri_indices = tri_indices[to_keep]

    bcoords = xy_bcoords(mesh, tri_indices, yx)

    # check the mask based on triangle containment
    in_tri_mask = tri_containment(bcoords)

    # use this mask on the pixels
    yx = yx[in_tri_mask]
    bcoords = bcoords[in_tri_mask]
    tri_indices = tri_indices[in_tri_mask]

    # Find the z values for all pixels and calculate the mask
    z_values = z_values_for_bcoords(mesh, bcoords, tri_indices)

    # argsort z from smallest to biggest - use this to sort all data
    sort = np.argsort(z_values)
    yx = yx[sort]
    bcoords = bcoords[sort]
    tri_indices = tri_indices[sort]

    # make a unique id per-pixel location
    pixel_index = yx[:, 0] * width + yx[:, 1]
    # find the first instance of each pixel site by depth
    _, z_buffer_mask = np.unique(pixel_index, return_index=True)

    # mask the locations one last time
    yx = yx[z_buffer_mask]
    bcoords = bcoords[z_buffer_mask]
    tri_indices = tri_indices[z_buffer_mask]
    return yx, bcoords, tri_indices


def rasterize_barycentric_coordinate_images(mesh, image_shape):
    h, w = image_shape
    yx, bcoords, tri_indices = rasterize_barycentric_coordinates(mesh,
                                                                 image_shape)

    tri_index_img = np.zeros((1, h, w), dtype=int)
    bcoord_img = np.zeros((3, h, w))
    mask = np.zeros((h, w), dtype=np.bool)
    mask[yx[:, 0], yx[:, 1]] = True
    tri_index_img[:, yx[:, 0], yx[:, 1]] = tri_indices
    bcoord_img[:, yx[:, 0], yx[:, 1]] = bcoords.T

    mask = BooleanImage(mask)
    return (MaskedImage(tri_index_img, mask=mask.copy(), copy=False),
            MaskedImage(bcoord_img, mask=mask.copy(), copy=False))
