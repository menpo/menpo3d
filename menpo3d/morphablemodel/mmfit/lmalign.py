from __future__ import division
import numpy as np
import cv2
from menpo.transform import Translation, Rotation, Homogeneous


flip_xy_yx = Homogeneous(np.array([[0, 1, 0],
                                   [1, 0, 0],
                                   [0, 0, 1]]))
drop_h = Homogeneous(np.eye(4)[:3])


# equivalent to rotating y by 180, then rotating z by 180
# This is also equivalent to glulookat to lookat the origin
#   gluLookAt(0,0,0,0,0,1,0,-1,0);
axes_flip_matrix = np.eye(4)
axes_flip_matrix[1, 1] = -1
axes_flip_matrix[2, 2] = -1
axes_flip_t = Homogeneous(axes_flip_matrix)


def weak_projection_matrix(width, height, mesh_camera_space):

    # Identify how far and near the mesh is in camera space.
    # we want to ensure that the near and far planes are
    # set so that all the mesh is displayed.
    near_bounds, far_bounds = mesh_camera_space.bounds()

    # Rather than just use the bounds, we add 10% in each direction
    # just to avoid any numerical errors.
    average_plane = (near_bounds[-1] + far_bounds[-1]) * 0.5
    padded_range = mesh_camera_space.range()[-1] * 1.1
    near_plane = average_plane - padded_range
    far_plane = average_plane + padded_range

    plane_sum = far_plane + near_plane
    plane_prod = far_plane * near_plane
    denom = far_plane - near_plane
    max_d = max(width, height)

    return np.array([[2.0 * max_d / width, 0,                    0,                    0],
                     [0,                   2.0 * max_d / height, 0,                    0],
                     [0,                   0,                    (-plane_sum) / denom, (-2.0 * plane_prod) / denom],
                     [0,                   0,                    -1,                   0]])


def retrieve_view_projection_transforms(image, mesh, group=None):

    rows = image.shape[0]
    cols = image.shape[1]
    max_d = max(rows, cols)
    camera_matrix = np.array([[max_d, 0,     cols / 2.0],
                              [0,     max_d, rows / 2.0],
                              [0,     0,     1.0]])
    distortion_coeffs = np.zeros(4)

    converged, r_vec, t_vec = cv2.solvePnP(mesh.landmarks[group].lms.points, 
                                           image.landmarks[group].lms.points[:, ::-1], 
                                           camera_matrix, 
                                           distortion_coeffs)

    rotation_matrix = cv2.Rodrigues(r_vec)[0]
    
    h_camera_matrix = np.eye(4)
    h_camera_matrix[:3, :3] = camera_matrix

    c = Homogeneous(h_camera_matrix)
    t = Translation(t_vec.ravel())
    r = Rotation(rotation_matrix)

    view_t_flipped = r.compose_before(t)
    view_t = view_t_flipped.compose_before(axes_flip_t)
    proj_t = Homogeneous(weak_projection_matrix(image.width, image.height, view_t_flipped.apply(mesh)))
    return view_t, proj_t, r
