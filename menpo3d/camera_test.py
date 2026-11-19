import numpy as np
from numpy.testing import assert_allclose

from menpo.shape import PointCloud
from menpo3d.camera import PerspectiveProjection, pinhole_intrinsic_matrix


def test_perspective_projection_of_points():
    projection = PerspectiveProjection(focal_length=500, image_shape=(480, 640))
    points = PointCloud(np.array([[0, 0, 2.0], [0.1, 0.2, 2.0]]))
    # (y, x) order, the principal point at the middle of the image
    assert_allclose(
        projection.apply(points).points, [[240, 320, 2], [240 + 50, 320 + 25, 2]]
    )


def test_pinhole_intrinsic_matrix_has_the_principal_point_in_the_middle():
    intrinsics = pinhole_intrinsic_matrix(480, 640).h_matrix
    assert_allclose(intrinsics[:2, 2], [320, 240])
    assert_allclose(intrinsics[0, 0], intrinsics[1, 1])
