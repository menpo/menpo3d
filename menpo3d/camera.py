import numpy as np
from menpo.base import Vectorizable
from menpo.transform import Homogeneous, Rotation, Transform, Translation


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

    @classmethod
    def init_from_2d_projected_shape(
        cls,
        points_3d,
        points_image,
        image_shape,
        focal_length=None,
        distortion_coeffs=None,
    ):
        from menpo3d.correspond import solve_pnp  # Avoid circular import

        model_view_t, _ = solve_pnp(
            points_image,
            points_3d,
            pinhole_intrinsic_matrix(
                image_shape[0], image_shape[1], focal_length=focal_length
            ),
            distortion_coefficients=distortion_coeffs,
        )
        rotation = Rotation(model_view_t.h_matrix[:3, :3])
        translation = Translation(model_view_t.h_matrix[:3, -1])
        return OrthographicCamera(
            rotation, translation, OrthographicProjection(focal_length, image_shape)
        )


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
    def init_from_2d_projected_shape(
        cls,
        points_3d,
        points_image,
        image_shape,
        focal_length=None,
        distortion_coeffs=None,
    ):
        raise NotImplementedError(
            "Orthographic camera pose estimation not " "implemented."
        )

    def apply(self, instance, **kwargs):
        return self.camera_transform.apply(instance)

    @property
    def n_parameters(self):
        return (
            self.projection_transform.n_parameters
            + self.rotation_transform.n_parameters
            + self.translation_transform.n_parameters
        )

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
    def init_from_2d_projected_shape(
        cls,
        points_3d,
        points_image,
        image_shape,
        focal_length=None,
        distortion_coeffs=None,
    ):
        from menpo3d.correspond import solve_pnp  # Avoid circular import

        model_view_t, _ = solve_pnp(
            points_image,
            points_3d,
            pinhole_intrinsic_matrix(
                image_shape[0], image_shape[1], focal_length=focal_length
            ),
            distortion_coefficients=distortion_coeffs,
        )
        rotation = Rotation(model_view_t.h_matrix[:3, :3])
        translation = Translation(model_view_t.h_matrix[:3, -1])
        return PerspectiveCamera(
            rotation, translation, PerspectiveProjection(focal_length, image_shape)
        )


def pinhole_intrinsic_matrix(image_height, image_width, focal_length=None):
    r"""
    Create a basic "pinhole" type camera intrinsic matrix. Focal length is in pixels
    and principal point is in the image center. Note this follows OpenCV image
    conventions and thus the "first" axis is the x-axis rather than the typical
    menpo convention of the "first" axis being the y-axis.

        [fx,  0, cx, 0]
        [ 0, fy, cy, 0]
        [ 0,  0,  1, 0]
        [ 0,  0,  0, 1]

    Parameters
    ----------
    image_height : int
        Image height
    image_width : int
        Image width
    focal_length : float, optional
        If given, the focal length (fx=fy) in pixels. If not given, the max
        of the width and height is used.

    Returns
    -------
    :map`Homogeneous`
        3D camera intrinsics matrix as a Homogeneous matrix
    """
    if focal_length is None:
        focal_length = max(image_height, image_width)
    return Homogeneous(
        np.array(
            [
                [focal_length, 0, image_width / 2, 0],
                [0, focal_length, image_height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
    )
