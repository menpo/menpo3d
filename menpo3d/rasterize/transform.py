from functools import reduce

import numpy as np
from menpo.transform import Homogeneous, NonUniformScale, Scale, Translation


def rotate_y_180_rotate_z_180():
    """
    Equivalent to rotating y by 180, then rotating z by 180.
    This simulates calling gluLookAt to lookat the origin

        gluLookAt(0,0,0,0,0,1,0,-1,0)

    Returns
    -------
    :map`Homogeneous`
        The matrix for applying the transformation
    """
    axes_flip_matrix = np.eye(4)
    axes_flip_matrix[1, 1] = -1
    axes_flip_matrix[2, 2] = -1
    return Homogeneous(axes_flip_matrix)


def flip_xy_yx():
    r"""
    Return a :map`Homogeneous` that flips the x and y axes.

    Returns
    -------
    :map`Homogeneous`
        The matrix for applying the transformation
    """
    return Homogeneous(
        np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    )


def dims_3to2():
    r"""
    Return a :map`Homogeneous` that strips off the 3D axis of a 3D shape
    leaving just the first two axes.

    Returns
     ------
    :map`Homogeneous`
        The matrix for applying the transformation
    """
    return Homogeneous(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))


def dims_2to3(x=0):
    r"""
    Return a :map`Homogeneous` that adds on a 3rd axis to a 2D shape.

    Parameters
    ----------
    x : `float`, optional
        The value that will be assigned to the new third dimension

    Returns
     ------
    :map`Homogeneous`
        The matrix for applying the transformation
    """
    return Homogeneous(np.array([[1, 0, 0], [0, 1, 0], [0, 0, x], [0, 0, 1]]))


def model_to_clip_transform(points, xy_scale=0.9, z_scale=0.3):
    r"""
    Produces an Affine Transform which centres and scales 3D points to fit
    into the OpenGL clipping space ([-1, 1], [-1, 1], [1, 1-]). This can be
    used to construct an appropriate projection matrix for use in an
    orthographic Rasterizer. Note that the z-axis is flipped as is default in
    OpenGL - as a result this transform converts the right handed coordinate
    input into a left hand one.

    Parameters
    ----------
    points: :map:`PointCloud`
        The points that should be adjusted.
    xy_scale: `float` 0-1, optional
        Amount by which the boundary is relaxed so the points are not
        right against the edge. A value of 1 means the extremities of the
        point cloud will be mapped onto [-1, 1] [-1, 1] exactly (no boarder)
        A value of 0.5 means the points will be mapped into the range
        [-0.5, 0.5].

        Default: 0.9 (map to [-0.9, 0.9])

    z_scale: float 0-1, optional
        Scale factor by which the z-dimension is squeezed. A value of 1
        means the z-range of the points will be mapped to exactly fit in
        [1, -1]. A scale of 0.1 means the z-range is compressed to fit in the
        range [0.1, -0.1].

    Returns
    -------
    :map:`Affine`
        The affine transform that creates this mapping
    """
    # 1. Centre the points on the origin
    center = Translation(points.centre_of_bounds()).pseudoinverse()
    # 2. Scale the points to exactly fit the boundaries
    scale = Scale(points.range() / 2.0)
    # 3. Apply the relaxations requested - note the flip in the z axis!!
    # This is because OpenGL by default evaluates depth as bigger number ==
    # further away. Thus not only do we need to get to clip space [-1, 1] in
    # all dims) but we must invert the z axis so depth buffering is correctly
    # applied.
    b_scale = NonUniformScale([xy_scale, xy_scale, -z_scale])
    return center.compose_before(scale.pseudoinverse()).compose_before(b_scale)


def clip_to_image_transform(width, height):
    r"""
    Affine transform that converts 3D clip space coordinates into 2D image
    space coordinates. Note that the z axis of the clip space coordinates is
    ignored.

    Parameters
    ----------
    width: int
        The width of the image
    height: int
        The height of the image

    Returns
    -------
    :map`Homogeneous`
        A homogeneous transform that moves clip space coordinates into image
        space.
    """
    # 1. Remove the z axis from the clip space
    rem_z = dims_3to2()
    # 2. invert the y direction (up becomes down)
    invert_y = Scale([1, -1])
    # 3. [-1, 1] [-1, 1] -> [0, 2] [0, 2]
    t = Translation([1, 1])
    # 4. [0, 2] [0, 2] -> [0, 1] [0, 1]
    unit_scale = Scale(0.5, n_dims=2)
    # 5. [0, 1] [0, 1] -> [0, w - 1] [0, h - 1]
    im_scale = Scale([width - 1, height - 1])
    # 6. [0, w] [0, h] -> [0, h] [0, w]
    xy_yx = Homogeneous(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float))
    # reduce the full transform chain to a single affine matrix
    transforms = [rem_z, invert_y, t, unit_scale, im_scale, xy_yx]
    return reduce(lambda a, b: a.compose_before(b), transforms)


def opengl_clip_matrix(
    intrinsic_matrix,
    image_width,
    image_height,
    near_plane,
    far_plane,
    include_axes_transformation=True,
):
    """
    Build the transformation matrix that can project points into the OpenGL
    "clip space". Note that we take the convention following OpenGL and thus
    the given intrinsic matrix should be defined as follows:

            [fx,  0, cx, 0]
            [ 0, fy, cy, 0]
            [ 0,  0,  1, 0]
            [ 0,  0,  0, 1]

    Which is different than the convention normally followed in menpo where the axes
    always match the matrices e.g. this is likely "flipped" w.r.t the x and y axis
    in comparison to normal menpo conventions.

    Parameters
    ----------
    intrinsic_matrix : :map`Homogeneous`
        The intrinsic camera matrix
    image_width : int
        Image width
    image_height : int
        Image height
    near_plane : float
        Near plane (e.g. any vertices closer than this to the camera are culled from
        the view frustrum)
    far_plane :
        Far plane (e.g. any vertices further than this from the camera are culled from
        the view frustrum)
    include_axes_transformation : bool
        If True, apply the standard OpenGL axes transformation equivalent to calling
        gluLookAt(0,0,0,0,0,1,0,-1,0)

    Returns
    -------
    :map`Homogeneous`
        A homogeneous transform that can be used as a projection matrix in the
        rasterizer for transforming the vertices into OpenGL "clip space".
    """
    plane_sum = far_plane + near_plane
    plane_prod = far_plane * near_plane
    denom = far_plane - near_plane

    fx = intrinsic_matrix.h_matrix[0, 0]
    fy = intrinsic_matrix.h_matrix[1, 1]
    cx = intrinsic_matrix.h_matrix[0, 2]
    cy = intrinsic_matrix.h_matrix[1, 2]

    clip_fx = 2 * fx / image_width
    clip_fy = 2 * fy / image_height
    clip_cx = (image_width - 2 * cx) / image_width
    clip_cy = (image_height - 2 * cy) / image_height

    clip_matrix = Homogeneous(
        np.array(
            [
                [clip_fx, 0, clip_cx, 0],
                [0, clip_fy, clip_cy, 0],
                [0, 0, (-plane_sum) / denom, (-2.0 * plane_prod) / denom],
                [0, 0, -1, 0],
            ]
        )
    )

    if include_axes_transformation:
        clip_matrix = rotate_y_180_rotate_z_180().compose_before(clip_matrix)

    return clip_matrix


def opengl_clip_matrix_from_mesh(
    intrinsic_matrix,
    image_width,
    image_height,
    mesh,
    plane_scale=1.1,
    include_axes_transformation=True,
):
    r"""
    See documentation for ``opengl_clip_matrix`` for more information. This uses
    the mesh to define the near and far planes.

    Parameters
    ----------
    intrinsic_matrix : :map`Homogeneous`
        The intrinsic camera matrix
    image_width : int
        Image width
    image_height : int
        Image height
    mesh : :map`Pointcloud`
        Mesh to use for computing the near and far planes. Must be in "eye space" to
        ensure the planes are computed correctly. See
        `estimate_near_and_far_planes` for more information about "eye space".
    plane_scale : float
        How much to scale the mesh range by to ensure that the mesh is within the
        near and far planes without worrying about numerical issues.
    include_axes_transformation : bool
        If True, apply the standard OpenGL axes transformation equivalent to calling
        gluLookAt(0,0,0,0,0,1,0,-1,0)

    Returns
    -------
    :map`Homogeneous`
        A homogeneous transform that can be used as a projection matrix in the
        rasterizer for transforming the vertices into OpenGL "clip space".
    """
    near_plane, far_plane = estimate_near_and_far_planes(mesh)
    return opengl_clip_matrix(
        intrinsic_matrix,
        image_width=image_width,
        image_height=image_height,
        near_plane=near_plane,
        far_plane=far_plane,
        include_axes_transformation=include_axes_transformation,
    )


def estimate_near_and_far_planes(mesh, scale=1.1):
    """
    Identify how far and near the mesh is in "eye space". We want to ensure that the
    near and far planes are set such that all of the mesh is displayed.

    We define the set of transformations as follows:

        1) The mesh initially exists in "object space"
        2) The Model transform converts from "object space" to "world space"
        3) The View transform converts from "world space" to "eye space"
        4) The Projection transform converts from "eye space" to "clip space"

    See http://www.songho.ca/opengl/gl_transform.html for more information.

    Parameters
    ----------
    mesh : :map:`PointCloud` or subclass
        The mesh in eye space.
    scale : float
        How much to scale the mesh range by to ensure that the mesh is within the
        near and far planes without worrying about numerical issues.

    Returns
    -------
    near_plane : float
        The near plane
    far_plane : float
        The far plane
    """
    near_bounds, far_bounds = mesh.bounds()
    average_plane = (near_bounds[-1] + far_bounds[-1]) * 0.5
    padded_range = mesh.range()[-1] * scale
    near_plane = average_plane - padded_range
    far_plane = average_plane + padded_range
    return near_plane, far_plane


def pinhole_intrinsic_matrix(image_height, image_width):
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

    Returns
    -------
    :map`Homogeneous`
        3D camera intrinsics matrix as a Homogeneous matrix
    """
    max_d = max(image_height, image_width)
    return Homogeneous(
        np.array(
            [
                [max_d, 0, image_width / 2, 0],
                [0, max_d, image_height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
    )
