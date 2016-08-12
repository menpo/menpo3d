from menpo3d.rasterize import GLRasterizer
from menpo.image import MaskedImage
from menpo.shape import TriMesh
import numpy as np


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


def params_from_ir(inverse_image):   
    vectors = inverse_image.as_vector(keep_channels=True)
    tri_indices = vectors[2].astype(np.uint32)

    a, b = vectors[:2]
    g = 1 - a - b
    b_coords = np.vstack([a, b, g])
    
    return b_coords, tri_indices


class InverseRenderer(GLRasterizer):
    def _rasterize_norms(self, points, normals, trilist, texture, tcoords,
                         per_vertex_f3v=None):
        points = np.require(points, dtype=np.float32, requirements='c')
        normals = np.require(normals, dtype=np.float32, requirements='c')
        trilist = np.require(trilist, dtype=np.uint32, requirements='c')
        texture = np.require(np.flipud(texture), dtype=np.float32, requirements='c')
        tcoords = np.require(tcoords, dtype=np.float32, requirements='c')
        
        
        if per_vertex_f3v is None:
            per_vertex_f3v = points
        interp = np.require(per_vertex_f3v, dtype=np.float32, requirements='c')

        rgb_fb, f3v_fb = self._opengl.render_offscreen_rgb_custom_vertex_normals(
            points, normals, interp, trilist, tcoords, texture)
        
        mask = rgb_fb[..., 3].astype(np.bool)

        return np.flipud(rgb_fb[..., :3]).copy(), np.flipud(f3v_fb), np.flipud(mask)

    def _rasterize_texture_with_interp(self, points, trilist, texture, tcoords,
                                       per_vertex_f3v=None, normals=None):
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
        # first, roll the axes to get things to the way OpenGL expects them
        texture = np.rollaxis(texture, 0, len(texture.shape))
        if normals is None:
            rgb_pixels, f3v_pixels, mask = self._rasterize(
                points, trilist, texture, tcoords, per_vertex_f3v=per_vertex_f3v)
        else:
            rgb_pixels, f3v_pixels, mask = self._rasterize_norms(
                points, normals, trilist, texture, tcoords, per_vertex_f3v=per_vertex_f3v)
            # roll back the results so things are as Menpo expects
        return (MaskedImage(np.array(np.rollaxis(rgb_pixels, -1), dtype=np.float), mask=mask),
                MaskedImage(np.array(np.rollaxis(f3v_pixels, -1), dtype=np.float), mask=mask))

    def inverse_render(self, m):

        # Convert the mesh into a version with one vertex per triangle
        # (Carefully looking after the normals)
        normals = m.vertex_normals()
        m, dedup_map = dedup_vertices(m)
        normals = normals[dedup_map]

        per_vertex_f3v = tri_bcoords_for_mesh(m)

        if hasattr(m, 'tcoords'):
            images = self._rasterize_texture_with_interp(
                m.points, m.trilist, m.texture.pixels, m.tcoords.points,
                per_vertex_f3v=per_vertex_f3v, normals=normals)
        else:
            # TODO: This should use a different shader!
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
                per_vertex_f3v=colours, normals=normals)
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