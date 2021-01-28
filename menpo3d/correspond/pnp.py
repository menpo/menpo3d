import numpy as np
from menpo.transform import Homogeneous, Rotation, Translation
from menpo3d.camera import pinhole_intrinsic_matrix
from menpo3d.rasterize.transform import flip_xy_yx

# Note this is copied from OpenCV to avoid importing OpenCV at the module level
SOLVEPNP_ITERATIVE = 0

_drop_h = Homogeneous(np.eye(4)[:3])


def solve_pnp(
    points_2d,
    points_3d,
    intrinsic_matrix,
    distortion_coefficients=None,
    pnp_method=SOLVEPNP_ITERATIVE,
    n_iterations=100,
    reprojection_error=8.0,
    initial_transform=None,
):
    """
    Use OpenCV to solve the Perspective-N-Point problem (PnP). Uses Ransac PnP as this
    typically provides better results. The image and mesh must both have the same
    landmark group name attached.

    Note the intrinsic matrix (if given) must be in "OpenCV" space and thus
    has the "x" and "y" axes flipped w.r.t the menpo norm. E.g. the intrinsic matrix
    is defined as follows:

        [fx,  0, cx, 0]
        [ 0, fy, cy, 0]
        [ 0,  0,  1, 0]
        [ 0,  0,  0, 1]

    Parameters
    ----------
    points_2d : :map`Pointcloud` or subclass
        The 2D points in the image to solve the PnP problem with.
    points_3d : :map`Pointcloud` or subclass
        The 3D points to solve the PnP problem with
    group : str, optional
        The name of the landmark group
    intrinsic_matrix : :map`Homogeneous`
        The intrinsic matrix - if the intrinsic matrix is unknow please see
        usage of pinhole_intrinsic_matrix()
    distortion_coefficients : ``(D,)`` `ndarray`
        The distortion coefficients (if not given assumes 0 coefficients). See the
        OpenCV documentation for the distortion coefficient types that are supported.
    pnp_method : int
        The OpenCV PNP method e.g. cv2.SOLVEPNP_ITERATIVE or otherwise
    n_iterations : int
        The number of iterations to perform
    reprojection_error : float
        The maximum reprojection error to allow for a point to be considered an
        inlier.
    initial_transform : :map`Homogeneous`
        The initialization for the cv2.SOLVEPNP_ITERATIVE method. Compatible
        with the returned model transformation returned by this method.

    Returns
    -------
    model_view_t : :map`Homogeneous`
        The combined ModelView transform. Can be used to place the 3D points
        in "eye space".
    proj_t : :map`Homogeneous`
        A transform that can be used to project the input 3D points
        back into the image
    """
    import cv2

    if distortion_coefficients is None:
        distortion_coefficients = np.zeros(4)

    r_vec = t_vec = None
    if initial_transform is not None:
        if pnp_method != cv2.SOLVEPNP_ITERATIVE:
            raise ValueError(
                "Initial estimates can only be given to SOLVEPNP_ITERATIVE"
            )
        else:
            r_vec = cv2.Rodrigues(initial_transform.h_matrix[:3, :3])[0]
            t_vec = initial_transform.h_matrix[:3, -1].ravel()

    converged, r_vec, t_vec, _ = cv2.solvePnPRansac(
        points_3d.points,
        points_2d.points[:, ::-1],
        intrinsic_matrix.h_matrix[:3, :3],
        distortion_coefficients,
        flags=pnp_method,
        iterationsCount=n_iterations,
        reprojectionError=reprojection_error,
        useExtrinsicGuess=r_vec is not None,
        rvec=r_vec,
        tvec=t_vec,
    )
    if not converged:
        raise ValueError("cv2.solvePnPRansac failed to converge")

    rotation = Rotation(cv2.Rodrigues(r_vec)[0])
    translation = Translation(t_vec.ravel())

    model_view_t = rotation.compose_before(translation)
    proj_t = intrinsic_matrix.compose_before(flip_xy_yx()).compose_before(_drop_h)
    return model_view_t, proj_t


def solve_pnp_landmarks(
    image,
    mesh,
    group=None,
    intrinsic_matrix=None,
    distortion_coefficients=None,
    pnp_method=SOLVEPNP_ITERATIVE,
    n_iterations=100,
    reprojection_error=8.0,
    initial_transform=None,
):
    """
    Use OpenCV to solve the Perspective-N-Point problem (PnP). Uses Ransac PnP as this
    typically provides better results. The image and mesh must both have the same
    landmark group name attached.

    Note the intrinsic matrix (if given) must be in "OpenCV" space and thus
    has the "x" and "y" axes flipped w.r.t the menpo norm. E.g. the intrinsic matrix
    is defined as follows:

        [fx,  0, cx, 0]
        [ 0, fy, cy, 0]
        [ 0,  0,  1, 0]
        [ 0,  0,  0, 1]

    Parameters
    ----------
    image : :map`Image` or subclass
        The image to solve the PnP problem in. Must have landmarks attached.
    mesh : :map`Pointcloud` or subclass
        The mesh to solve the PnP problem with.  Must have landmarks attached.
    group : str, optional
        The name of the landmark group
    intrinsic_matrix : :map`Homogeneous`
        The intrinsic matrix - if not provided a basic pinhole camera is
        assumed.
    distortion_coefficients : ``(D,)`` `ndarray`
        The distortion coefficients (if not given assumes 0 coefficients). See the
        OpenCV documentation for the distortion coefficient types that are supported.
    pnp_method : int
        The OpenCV PNP method e.g. cv2.SOLVEPNP_ITERATIVE or otherwise
    n_iterations : int
        The number of iterations to perform
    reprojection_error : float
        The maximum reprojection error to allow for a point to be considered an
        inlier.
    initial_transform : :map`Homogeneous`
        The initialization for the cv2.SOLVEPNP_ITERATIVE method. Compatible
        with the returned model transformation returned by this method.

    Returns
    -------
    model_view_t : :map`Homogeneous`
        The combined ModelView transform. Can be used to place the mesh
        in "eye space".
    proj_t : :map`Homogeneous`
        A transform that can be used to project the input mesh landmarks
        back into the image

    Examples
    --------
    >>> from camera import pinhole_intrinsic_matrix    >>> from menpo3d.correspond import solve_pnp
    >>> from menpo3d.rasterize.transform import (
    >>>      opengl_clip_matrix_from_mesh
    >>> )
    >>> template = m3dio.import_builtin_asset.template_ply()
    >>> image = mio.import_builtin_asset.lenna_png()

    >>> # Project the mesh back into the image
    >>> model_view_t, proj_t = solve_pnp(image, template, group="LJSON")
    >>> mesh_to_image = model_view_t.compose_before(proj_t)
    >>> image.landmarks['projected'] = mesh_to_image.apply(template.landmarks['LJSON'])

    >>> # Rasterize the mesh into the image
    >>> opengl_proj_t = opengl_clip_matrix_from_mesh(
    >>>     pinhole_intrinsic_matrix(image_height=image.height, image_width=image.width),
    >>>     image.width,
    >>>     image.height,
    >>>     view_t.apply(template),
    >>> )
    >>> rasterizer = GLRasterizer(height=image.height, width=image.width,
    >>>                           model_matrix=model_view_t.h_matrix,
    >>>                           projection_matrix=opengl_proj_t.h_matrix)
    >>> rasterized_image = rasterizer.rasterize_mesh(template)
    """
    if intrinsic_matrix is None:
        intrinsic_matrix = pinhole_intrinsic_matrix(image.shape[0], image.shape[1])

    return solve_pnp(
        image.landmarks[group],
        mesh.landmarks[group],
        intrinsic_matrix=intrinsic_matrix,
        distortion_coefficients=distortion_coefficients,
        pnp_method=pnp_method,
        n_iterations=n_iterations,
        reprojection_error=reprojection_error,
        initial_transform=initial_transform,
    )
