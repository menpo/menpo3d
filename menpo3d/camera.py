from __future__ import division
import numpy as np
import warnings

from menpo.base import Vectorizable, MenpoMissingDependencyError
from menpo.transform import Transform, Translation, Rotation


def align_2d_3d(points_3d, points_image, image_shape, focal_length=None,
                distortion_coeffs=None):
    try:
        import cv2
    except ImportError:
        raise MenpoMissingDependencyError('opencv3')
    height, width = image_shape
    # Create camera matrix
    focal_length = (max(height, width)
                    if focal_length is None else focal_length)
    c_x = width / 2.0
    c_y = height / 2.0
    camera_matrix = np.array([[focal_length, 0, c_x],
                              [0, focal_length, c_y],
                              [0, 0, 1.0]])

    # If distortion coefficients are None, set them to zero
    if distortion_coeffs is None:
        distortion_coeffs = np.zeros(4)
    # Estimate the camera pose given the 3D sparse pointcloud on the
    # mesh, its 2D projection on the image, the camera matrix and the
    # distortion coefficients
    lm2d = points_image.points[:, ::-1].copy()
    converged, r_vec, t_vec = cv2.solvePnP(points_3d.points,
                                           lm2d,
                                           camera_matrix, distortion_coeffs)

    if not converged:
        warnings.warn('cv2.SolvePnP did not converge to a solution')

    # Create rotation and translation transform objects from the vectors
    # acquired at the previous step
    rotation_matrix = cv2.Rodrigues(r_vec)[0]
    r = Rotation(rotation_matrix)
    t = Translation(t_vec.ravel())

    return r, t, focal_length


class OrthographicProjection(Transform, Vectorizable):

    def __init__(self, focal_length, image_shape):
        self.focal_length = focal_length
        self.height = image_shape[0]
        self.width = image_shape[1]

    @property
    def n_dims(self):
        return 3

    @property
    def n_parameters(self):
        return 1

    def as_vector(self):
        return np.array([self.focal_length])

    def _from_vector_inplace(self, vector):
        self.focal_length = vector[0]

    def _apply(self, x, **kwargs):
        f = self.focal_length
        c_x = self.width / 2
        c_y = self.height / 2

        output = np.empty_like(x)
        output[:, 0] = (f * x[:, 1]) + c_y
        output[:, 1] = (f * x[:, 0]) + c_x
        output[:, 2] = x[:, 2]

        return output


class PerspectiveProjection(OrthographicProjection):
    def _apply(self, x, **kwargs):
        f = self.focal_length
        c_x = self.width / 2
        c_y = self.height / 2

        output = np.empty_like(x)
        output[:, 0] = (f * x[:, 1]) / x[:, 2] + c_y
        output[:, 1] = (f * x[:, 0]) / x[:, 2] + c_x
        output[:, 2] = x[:, 2]

        return output


class OrthographicCamera(Vectorizable):
    def __init__(self, rotation, translation, projection):
        self.rotation_transform = rotation
        self.translation_transform = translation
        self.projection_transform = projection

    @property
    def focal_length(self):
        return self.projection_transform.focal_length

    @classmethod
    def init_from_image_shape_and_vector(cls, image_shape, vector):
        r = Rotation.init_identity(n_dims=3)
        t = Translation.init_identity(n_dims=3)
        p = OrthographicProjection(focal_length=1, image_shape=image_shape)
        return cls(r, t, p).from_vector(vector)

    @classmethod
    def init_from_2d_projected_shape(cls, points_3d, points_image,
                                     image_shape, focal_length=None,
                                     distortion_coeffs=None):
        r, t, focal_length = align_2d_3d(
            points_3d, points_image, image_shape, focal_length=focal_length,
            distortion_coeffs=distortion_coeffs)
        return OrthographicCamera(r, t, OrthographicProjection(focal_length,
                                                               image_shape))

    def apply(self, instance, **kwargs):
        return self.camera_transform.apply(instance)

    @property
    def n_parameters(self):
        return (self.projection_transform.n_parameters +
                self.rotation_transform.n_parameters +
                self.translation_transform.n_parameters)

    def as_vector(self):
        # focal_length, q_w, q_x, q_y, q_z, t_x, t_y, t_z
        params = np.zeros(self.n_parameters)

        # focal length
        params[:1] = self.projection_transform.as_vector()

        # 4 parameters: q_w, q_x, q_y, q_z
        params[1:5] = self.rotation_transform.as_vector()

        # 3 parameters: t_x, t_y, t_z
        params[5:] = self.translation_transform.as_vector()
        return params

    def _from_vector_inplace(self, vector):
        self.projection_transform._from_vector_inplace(vector[:1])
        self.rotation_transform._from_vector_inplace(vector[1:5])
        self.translation_transform._from_vector_inplace(vector[5:])

    @property
    def view_transform(self):
        return self.rotation_transform.compose_before(self.translation_transform)

    @property
    def camera_transform(self):
        return self.view_transform.compose_before(self.projection_transform)


class PerspectiveCamera(OrthographicCamera):
    @classmethod
    def init_from_image_shape_and_vector(cls, image_shape, vector):
        r = Rotation.init_identity(n_dims=3)
        t = Translation.init_identity(n_dims=3)
        p = PerspectiveProjection(focal_length=1, image_shape=image_shape)
        return cls(r, t, p).from_vector(vector)

    @classmethod
    def init_from_2d_projected_shape(cls, points_3d, points_image,
                                     image_shape, focal_length=None,
                                     distortion_coeffs=None):
        r, t, focal_length = align_2d_3d(
            points_3d, points_image, image_shape, focal_length=focal_length,
            distortion_coeffs=distortion_coeffs)
        return PerspectiveCamera(r, t, PerspectiveProjection(focal_length,
                                                             image_shape))
