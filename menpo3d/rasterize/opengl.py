import numpy as np
from cyrasterize.base import CyRasterizerBase

from menpo.image import MaskedImage
from menpo.shape import TriMesh
from menpo.transform import Homogeneous

from .transform import clip_to_image_transform


def tri_bcoords_for_mesh(mesh):
    bc_per_tri = np.array([[1, 0],
                           [0, 1],
                           [0, 0]])
    bc = np.tile(bc_per_tri.T, mesh.n_tris).T

    index = np.repeat(np.arange(mesh.n_tris), 3, axis=0)

    return np.hstack((bc, index[:, None]))


def dedup_vertices(mesh):
    old_to_new = mesh.trilist.ravel()
    new_trilist = np.arange(old_to_new.shape[0]).reshape([-1, 3])
    new_points = mesh.points[old_to_new]
    return TriMesh(new_points, trilist=new_trilist), old_to_new


# Subclass the CyRasterizerBase class to add Menpo-specific features
# noinspection PyProtectedMember
class GLRasterizer(CyRasterizerBase):

    def __reduce__(self):
        return (GLRasterizer, (self.width, self.height,
                               self.model_matrix, self.view_matrix,
                               self.projection_matrix))

    @property
    def model_to_clip_matrix(self):
        return np.dot(self.projection_matrix,
                      np.dot(self.view_matrix, self.model_matrix))

    @property
    def model_transform(self):
        return Homogeneous(self.model_matrix)

    @property
    def view_transform(self):
        return Homogeneous(self.view_matrix)

    @property
    def projection_transform(self):
        return Homogeneous(self.projection_matrix)

    @property
    def model_to_clip_transform(self):
        r"""
        Transform that takes 3D points from model space to 3D clip space
        """
        return Homogeneous(self.model_to_clip_matrix)

    @property
    def clip_to_image_transform(self):
        r"""
        Affine transform that converts 3D clip space coordinates into 2D image
        space coordinates
        """
        return clip_to_image_transform(self.width, self.height)

    @property
    def model_to_image_transform(self):
        r"""
        TransformChain from 3D model space to 2D image space.
        """
        return self.model_to_clip_transform.compose_before(
            self.clip_to_image_transform)

    def rasterize_mesh_with_f3v_interpolant(self, mesh, per_vertex_f3v=None,
                                            normals=None):
        r"""
        Rasterize the object to an image and generate an interpolated
        3-float image from a per vertex float 3 vector.

        Parameters
        ----------
        mesh : object implementing the Rasterizable interface.
        per_vertex_f3v : optional, ndarray (n_points, 3)
            A per-vertex 3 vector of floats that will be interpolated across
            the image.
            If None, the model's shape is used (making
            this method equivalent to rasterize_mesh_with_shape_image)
        normals : ndarray, shape (n_points, 3)
            A matrix specifying custom per-vertex normals to be used. If omitted,
            the normals will be calculated from the triangulation of triangle normals.

        Returns
        -------
        rgb_image : 3-channel MaskedImage of shape (width, height)
            The result of the rasterization. Mask is true iff the pixel was
            rendered to by OpenGL.

        interp_image: 3 channel MaskedImage of shape (width, height)
            The result of interpolating the per_vertex_f3v across the
            visible primitives.

        """
        if not (hasattr(mesh, 'points') and
                hasattr(mesh, 'trilist')):
            raise ValueError('Rasterizable types have to have points and '
                             'trilist properties.')
        if hasattr(mesh, 'tcoords'):
            images = self._rasterize_texture_with_interp(
                mesh.points, mesh.trilist, mesh.texture.pixels, mesh.tcoords.points,
                normals=normals, per_vertex_f3v=per_vertex_f3v)
        else:
            #TODO: This should use a different shader!
            # TODO This should actually use the colour provided.
            # But I'm hacking it here to work quickly.
            if hasattr(mesh, 'colours'):
                colours = mesh.colours
            else:
                # just make a grey colour
                colours = np.ones((mesh.n_points, 3)) * 0.5
            # Fake some texture coordinates and a texture as required by the
            # shader
            fake_tcoords = np.random.randn(mesh.n_points, 2)
            fake_texture = np.zeros([2, 2, 3])

            # The RGB image is going to be broken due to the fake texture
            # information we passed in
            _, rgb_image = self._rasterize_texture_with_interp(
                mesh.points, mesh.trilist, fake_texture, fake_tcoords,
                per_vertex_f3v=colours)
            _, f3v_image = self._rasterize_texture_with_interp(
                mesh.points, mesh.trilist, fake_texture, fake_tcoords,
                per_vertex_f3v=per_vertex_f3v)

            images = rgb_image, f3v_image

        from menpo.landmark import Landmarkable
        if isinstance(mesh, Landmarkable):
            # Transform all landmarks and set them on the image
            image_lms = self.model_to_image_transform.apply(
                mesh.landmarks)
            for image in images:
                image.landmarks = image_lms
        return images

    def rasterize_mesh_with_shape_image(self, mesh):
        r"""Rasterize a mesh and additionally generate an interpolated
        3-float image from the shape information on the mesh.

        Parameters
        ----------
        mesh : object implementing the Rasterizable interface.

        Returns
        -------
        rgb_image : 3 channel MaskedImage of shape (width, height)
            The result of the rasterization. Mask is true iff the pixel was
            rendered to by OpenGL.
        shape_image: 3 channel MaskedImage of shape (width, height)
            The result of interpolating the spatial information of each vertex
            across the visible primitives. Note that the shape information
            is *NOT* adjusted by the P,V,M matrices, and so the resulting
            shape image is always in the original objects reference shape
            (i.e. the z value will not necessarily correspond to a depth
            buffer).
        """
        return self.rasterize_mesh_with_f3v_interpolant(mesh, per_vertex_f3v=None)

    def rasterize_mesh(self, mesh):
        r"""Rasterize a mesh to an image.

        Parameters
        ----------
        mesh : object implementing the Rasterizable interface.

        Returns
        -------
        rgb_image : 3 channel MaskedImage of shape (width, height)
            The result of the rasterization. Mask is true iff the pixel was
            rendered to by OpenGL.
        """
        return self.rasterize_mesh_with_shape_image(mesh)[0]

    def rasterize_barycentric_coordinate_image(self, mesh):

        # Convert the mesh into a version with one vertex per triangle
        # (Carefully looking after the normals)
        normals = mesh.vertex_normals()
        mesh, dedup_map = dedup_vertices(mesh)
        normals = normals[dedup_map]

        per_vertex_f3v = tri_bcoords_for_mesh(mesh)

        images = self.rasterize_mesh_with_f3v_interpolant(mesh, normals=normals,
                                                          per_vertex_f3v=per_vertex_f3v)

        # the interpolated image is [tri_index, alpha, beta]
        # -> split this into two images, one tri_index, one bc
        inverse_image = images[1]

        vectors = inverse_image.as_vector(keep_channels=True)
        tri_indices = vectors[2].astype(np.uint32)

        a, b = vectors[:2]
        g = 1 - a - b
        bcoords = np.vstack([a, b, g])

        tri_index_image = inverse_image.from_vector(tri_indices, n_channels=1)
        bcoords_image = inverse_image.from_vector(bcoords, n_channels=3)

        return tri_index_image, bcoords_image

    def _rasterize_texture_with_interp(self, points, trilist, texture, tcoords,
                                       normals=None, per_vertex_f3v=None):
        r"""Rasterizes a textured mesh along with it's interpolant data
        through OpenGL.

        Parameters
        ----------
        r : object
            Any object with fields named 'points', 'trilist', 'texture' and
            'tcoords' specifying the data that will be used to render. Such
            objects are handed out by the
            _rasterize_generate_textured_mesh method on Rasterizable
            subclasses
        normals : ndarray, shape (n_points, 3)
            A matrix specifying custom per-vertex normals to be used. If omitted,
            the normals will be calculated from the triangulation of triangle normals.
        per_vertex_f3v : ndarray, shape (n_points, 3), optional
            A matrix specifying arbitrary 3 floating point numbers per
            vertex. This data will be linearly interpolated across triangles
            and returned in the f3v image. If none, the shape information is
            used

        Returns
        -------
        image : MaskedImage
            The rasterized image returned from OpenGL. Note that the
            behavior of the rasterization is governed by the projection,
            rotation and view matrices that may be set on this class,
            as well as the width and height of the rasterization, which is
            determined on the creation of this class. The mask is True if a
            triangle is visible at that pixel in the output, and False if not.

        f3v_image : MaskedImage
            The rasterized image returned from OpenGL. Note that the
            behavior of the rasterization is governed by the projection,
            rotation and view matrices that may be set on this class,
            as well as the width and height of the rasterization, which is
            determined on the creation of this class.

        """
        # make a call out to the CyRasterizer _rasterize method
        # first, roll the axes to get things to the way OpenGL expects them
        texture = np.rollaxis(texture, 0, len(texture.shape))
        rgb_pixels, f3v_pixels, mask = self._rasterize(
            points, trilist, texture, tcoords, normals=normals, per_vertex_f3v=per_vertex_f3v)
        # roll back the results so things are as Menpo expects
        return (MaskedImage(np.array(np.rollaxis(rgb_pixels, -1), dtype=np.float), mask=mask),
                MaskedImage(np.array(np.rollaxis(f3v_pixels, -1), dtype=np.float), mask=mask))
