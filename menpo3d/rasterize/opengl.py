import numpy as np
from cyrasterize.base import CyRasterizerBase

from menpo.image import MaskedImage
from menpo.transform import Homogeneous

from .transform import clip_to_image_transform


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

    def rasterize_mesh_with_f3v_interpolant(self, m, per_vertex_f3v=None):
        r"""
        Rasterize the object to an image and generate an interpolated
        3-float image from a per vertex float 3 vector.

        If no per_vertex_f3v is provided, the model's shape is used (making
        this method equivalent to rasterize_mesh_with_shape_image)

        Parameters
        ----------
        rasterizable : object implementing the Rasterizable interface.
            Will be queried for some state to rasterize via the Rasterizable
            interface. Note that currently, color mesh rasterizations are
            not supported.

        per_vertex_f3v : optional, ndarray (n_points, 3)
            A per-vertex 3 vector of floats that will be interpolated across
            the image.


        Returns
        -------
        rgb_image : 3 channel MaskedImage of shape (width, height)
            The result of the rasterization. Mask is true iff the pixel was
            rendered to by OpenGL.

        interp_image: 3 channel MaskedImage of shape (width, height)
            The result of interpolating the per_vertex_f3v across the
            visible primitives.

        """
        if not (hasattr(m, 'points') and
                hasattr(m, 'trilist')):
            raise ValueError('Rasterizable types have to have points and '
                             'trilist properties.')
        if hasattr(m, 'tcoords'):
            images = self._rasterize_texture_with_interp(
                m.points, m.trilist, m.texture.pixels, m.tcoords.points,
                per_vertex_f3v=per_vertex_f3v)
        else:
            #TODO: This should use a different shader!
            # TODO This should actually use the colour provided.
            # But I'm hacking it here to work quickly.
            if hasattr(m, 'colours'):
                colours = m.colours
            else:
                # just make a grey colour
                colours = np.ones((m.n_points, 3)) * 0.5
            # Fake some texture coordinates and a texture as required by the
            # shader
            fake_tcoords = np.random.randn(m.n_points, 2)
            fake_texture = np.zeros([2, 2, 3])

            # The RGB image is going to be broken due to the fake texture
            # information we passed in
            _, rgb_image = self._rasterize_texture_with_interp(
                m.points, m.trilist, fake_texture, fake_tcoords,
                per_vertex_f3v=colours)
            _, f3v_image = self._rasterize_texture_with_interp(
                m.points, m.trilist, fake_texture, fake_tcoords,
                per_vertex_f3v=per_vertex_f3v)

            images = rgb_image, f3v_image

        from menpo.landmark import Landmarkable
        if isinstance(m, Landmarkable):
            # Transform all landmarks and set them on the image
            image_lms = self.model_to_image_transform.apply(
                m.landmarks)
            for image in images:
                image.landmarks = image_lms
        return images

    def rasterize_mesh_with_shape_image(self, rasterizable):
        r"""Rasterize the object to an image and generate an interpolated
        3-float image from the shape information on the rasterizable object.

        Parameters
        ----------
        rasterizable : object implementing the Rasterizable interface.
            Will be queried for some state to rasterize via the Rasterizable
            interface. Note that currently, color mesh rasterizations are
            not supported.


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
        return self.rasterize_mesh_with_f3v_interpolant(rasterizable)

    def rasterize_mesh(self, rasterizable):
        r"""Rasterize the object to an image and generate an interpolated
        3-float image from the shape information on the rasterizable object.

        Parameters
        ----------
        rasterizable : object implementing the Rasterizable interface.
            Will be queried for some state to rasterize via the Rasterizable
            interface. Note that currently, color mesh rasterizations are
            not supported.


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
        return self.rasterize_mesh_with_shape_image(rasterizable)[0]

    def _rasterize_texture_with_interp(self, points, trilist, texture, tcoords,
                                       per_vertex_f3v=None):
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
        per_vertex_f3v: ndarray, shape (n_points, 3)
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
        rgb_pixels, f3v_pixels, mask = self._rasterize(
            points, trilist, texture, tcoords, per_vertex_f3v=per_vertex_f3v)
        return (MaskedImage(np.array(rgb_pixels, dtype=np.float), mask=mask),
                MaskedImage(np.array(f3v_pixels, dtype=np.float), mask=mask))
