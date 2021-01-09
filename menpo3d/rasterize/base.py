def rasterize_mesh_from_barycentric_coordinate_images(
    mesh, bcoords_image, tri_indices_image
):
    r"""
    Renders an image of a `menpo.shape.TexturedTriMesh` or
    `menpo.shape.ColouredTriMesh` from a barycentric coordinate image pair.

    Note that the texture is rendered without any lighting model - think of
    this as a piecewise affine warp of the mesh's texture into the image (
    with z-buffering). As there is no lighting model, only meshes with
    colour/texture can be used with this method (a single color for the whole
    mesh would render flat with no shading).

    Parameters
    ----------
    mesh : `menpo.shape.TexturedTriMesh` or `menpo.shape.ColouredTriMesh`
        The 3D mesh who's texture will be rendered to the image.
    bcoords_image : `menpo.image.MaskedImage`
        The per-triangle barycentric coordinates for what should be rendered
        into each pixel. See :map:`rasterize_barycentric_coordinate_images`.
    tri_indices_image : `menpo.image.MaskedImage`
        The triangle index identifying the triangle that is visable at a pixel
        after z-buffering. See :map:`rasterize_barycentric_coordinate_images`.

    Returns
    -------
    `menpo.image.MaskedImage`
        A rasterized image of the mesh.
    """
    # Sample the mesh texture space to find the colors-per pixel
    colours = mesh.sample_texture_with_barycentric_coordinates(
        bcoords_image.as_vector(keep_channels=True).T, tri_indices_image.as_vector()
    )
    # Rebuild the image using the usual from_vector machinery
    return tri_indices_image.from_vector(colours.T, n_channels=mesh.n_channels)


def rasterize_shape_image_from_barycentric_coordinate_images(
    mesh, bcoords_image, tri_indices_image
):
    r"""
    Renders an XYZ shape image of a `menpo.shape.TexturedTriMesh` or
    `menpo.shape.ColouredTriMesh` from a barycentric coordinate image pair.


    Parameters
    ----------
    mesh : `menpo.shape.TexturedTriMesh` or `menpo.shape.ColouredTriMesh`
        The 3D mesh who's texture will be rendered to the image.
    bcoords_image : `menpo.image.MaskedImage`
        The per-triangle barycentric coordinates for what should be rendered
        into each pixel. See :map:`rasterize_barycentric_coordinate_images`.
    tri_indices_image : `menpo.image.MaskedImage`
        The triangle index identifying the triangle that is visable at a pixel
        after z-buffering. See :map:`rasterize_barycentric_coordinate_images`.

    Returns
    -------
    `menpo.image.MaskedImage`
        A rasterized shape image image of the mesh.
    """
    # Sample the mesh texture space to find the colors-per pixel
    shape_per_pixel = mesh.project_barycentric_coordinates(
        bcoords_image.as_vector(keep_channels=True).T, tri_indices_image.as_vector()
    )
    # Rebuild the image using the usual from_vector machinery
    return tri_indices_image.from_vector(
        shape_per_pixel.points.T, n_channels=mesh.n_channels
    )


def rasterize_mesh(mesh_in_img, image_shape):
    from .cpu import rasterize_barycentric_coordinate_images

    bcs = rasterize_barycentric_coordinate_images(mesh_in_img, image_shape)
    return rasterize_mesh_from_barycentric_coordinate_images(mesh_in_img, *bcs)
