#cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np


def pixels_to_check(long long[:, :] start, long long[:, :] end, long long n_pixels):

    cdef long long[:, :] pixel_locations = np.empty((n_pixels, 2), dtype=np.int64)
    cdef long long[:] tri_indices = np.empty(n_pixels, dtype=np.int64)

    cdef long long n_tris = start.shape[0]
    cdef long long i, j, k, sx, sy, ex, ey
    cdef long long n = 0

    for i in range(n_tris):
        sx = start[i, 0]
        sy = start[i, 1]
        ex = end[i, 0]
        ey = end[i, 1]
        for j in range(sx, ex):
            for k in range(sy, ey):
                pixel_locations[n, 0] = j
                pixel_locations[n, 1] = k
                tri_indices[n] = i
                n += 1

    return np.asarray(pixel_locations), np.asarray(tri_indices)
