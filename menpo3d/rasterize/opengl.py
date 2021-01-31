from contextlib import contextmanager
from enum import IntEnum
from functools import wraps
from pathlib import Path

import moderngl
import numpy as np

from menpo.image import MaskedImage
from menpo.shape import TriMesh
from menpo.transform import Homogeneous
from .transform import clip_to_image_transform

CONTAINING_DIR = Path(__file__).parent
PASSTHROUGH_TEXTURE_VERT_SHADER = CONTAINING_DIR / "shaders/texture/passthrough.vert"
PASSTHROUGH_TEXTURE_FRAG_SHADER = CONTAINING_DIR / "shaders/texture/passthrough.frag"
PASSTHROUGH_PER_VERTEX_VERT_SHADER = (
    CONTAINING_DIR / "shaders/per_vertex/passthrough.vert"
)
PASSTHROUGH_PER_VERTEX_FRAG_SHADER = (
    CONTAINING_DIR / "shaders/per_vertex/passthrough.frag"
)


class _LOADED_SHADER_TYPE(IntEnum):
    TEXTURE = 0
    PER_VERTEX = 1


def tri_bcoords_for_mesh(mesh):
    bc_per_tri = np.array([[1, 0], [0, 1], [0, 0]])
    bc = np.tile(bc_per_tri.T, mesh.n_tris).T

    index = np.repeat(np.arange(mesh.n_tris), 3, axis=0)

    return np.hstack((bc, index[:, None]))


def dedup_vertices(mesh):
    old_to_new = mesh.trilist.ravel()
    new_trilist = np.arange(old_to_new.shape[0]).reshape([-1, 3])
    new_points = mesh.points[old_to_new]
    return TriMesh(new_points, trilist=new_trilist), old_to_new


def _verify_opengl_homogeneous_matrix(matrix):
    if matrix.shape != (4, 4):
        raise ValueError("OpenGL matrices must have shape (4,4)")
    return np.require(matrix, dtype=np.float32, requirements="C")


@contextmanager
def safe_release(obj):
    try:
        yield obj
    except:
        obj.release()


def with_context(method):
    """
    Because it's possible to make multiple GLRasterizers in the same Python
    process it's important we ensure that the correct OpenGL context is activated
    when rasterization is performed. This is particularly important given that
    the rasterizers may be created inside Python processes that have OpenGL
    contexts not even owned by menpo3d (e.g. VTK or otherwise).
    """

    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        with self.opengl_ctx:
            return method(self, *method_args, **method_kwargs)

    return _impl


class BasePassthroughProgram:
    def __init__(self, context):
        """
        Base program for a "passthrough" shader which just rasterizes the mesh
        without considering any lighting or mesh normals.

        Parameters
        ----------
        context : moderngl context
            The context used for rendering
        """
        self.context = context
        self.program = None
        self.packed_vbo = None
        self.index_buffer_vbo = None

    def __del__(self):
        # Make sure to always release the VBOs when the object is destroyed
        if self.packed_vbo is not None:
            self.packed_vbo.release()
        if self.index_buffer_vbo is not None:
            self.index_buffer_vbo.release()

    def _get_packed_vbo(self, buffer_bytes):
        # Handle creating the "packed" vbo which contains the data required
        # for rendering with the shader. buffer_bytes is a bytes array. This
        # tries to "cache" the VBO in the case that repeated rendering of the
        # same mesh is performed
        n_bytes = len(buffer_bytes)
        if self.packed_vbo is None or n_bytes != self.packed_vbo._size:
            if self.packed_vbo is not None:
                self.packed_vbo.release()
            self.packed_vbo = self.context.buffer(buffer_bytes)
        elif self.packed_vbo is not None:
            self.packed_vbo.write(buffer_bytes)

    def _get_index_buffer_vbo(self, buffer_bytes):
        # Handle creating the index buffer vbo which contains the data required
        # for rendering with the shader. buffer_bytes is a bytes array. This
        # tries to "cache" the VBO in the case that repeated rendering of the
        # same mesh is performed
        n_bytes = len(buffer_bytes)
        if self.index_buffer_vbo is None or n_bytes != self.index_buffer_vbo._size:
            if self.index_buffer_vbo is not None:
                self.index_buffer_vbo.release()
            self.index_buffer_vbo = self.context.buffer(buffer_bytes)
        elif self.index_buffer_vbo is not None:
            self.index_buffer_vbo.write(buffer_bytes)

    def set_mvp(self, mvp_matrix):
        """
        Set the MVP (Model, View, Projection) matrix for the shader. This is
        modelled as a uniform called "MVP".

        Parameters
        ----------
        mvp_matrix : ndarray (4, 4)
            The combined (Model, View, Projection) matrix to set
        """
        mvp_matrix = np.require(mvp_matrix, dtype=np.float32, requirements=["C"])
        self.program["MVP"].write(mvp_matrix.tobytes())


class TexturePassthroughProgram(BasePassthroughProgram):
    def __init__(self, context):
        """
        Implements a "passthrough" shader for textured meshes. This just
        rasterizes the mesh without considering any lighting or mesh normals.

        Parameters
        ----------
        context : moderngl context
            The context used for rendering
        """
        super().__init__(context)
        self.program = self.context.program(
            vertex_shader=PASSTHROUGH_TEXTURE_VERT_SHADER.read_text(encoding="ascii"),
            fragment_shader=PASSTHROUGH_TEXTURE_FRAG_SHADER.read_text(encoding="ascii"),
        )
        self.texture = None
        self._texture_shape = None
        self._pixels_ref = None

    def __del__(self):
        super().__del__()
        if self.texture is not None:
            self.texture.release()
        # Release program last
        if self.program is not None:
            self.program.release()

    def _build_texture(self, pixels):
        assert pixels.shape[-1] == 3, "Only RGB textures supported"

        # Try caching the texture if repeated rendering of the same mesh is
        # being performed
        shape = pixels.shape[:2][::-1]
        if self._texture_shape is None or self._texture_shape != shape:
            self._texture_shape = shape
            # Note this keeps the texture alive and so uses lots of memory for large textures
            self._pixels_ref = pixels
            self.texture = self.context.texture(
                size=shape, components=3, data=pixels.tobytes(), dtype="f4"
            )
        elif not np.allclose(self._pixels_ref, pixels):
            assert self.texture is not None
            self.texture.write(pixels.tobytes())

    def create_vao(self, mesh, per_vertex_f3v):
        """
        Create a simple Vertex Array Object (VAO) that contains the per vertex
        array data used by the shader. Ensures that the correct texture
        is also allocated.

        Parameters
        ----------
        mesh :  `menpo.shape.TexturedTriMesh`
        per_vertex_f3v : ndarray, (n_points, 3)

        Returns
        -------
        `moderngl.vertex_array.VertexArray`
            VAO to be used by the moderngl rendering context
        """
        # Grab the texture in [H, W, 3] format - note the pixel values need to
        # be [0, 1] for moderngl
        pixels = mesh.texture.pixels_with_channels_at_back(out_dtype=np.float32)
        # Note that we assume the texture coordinates are flipped inside the
        # shader
        self._build_texture(pixels)
        # Create the packed VBO where the entries here are in the same order as
        # the shader inputs
        packed = np.concatenate(
            [mesh.points, mesh.tcoords.points, per_vertex_f3v], axis=1
        )

        packed_buffer = np.require(packed, dtype=np.float32, requirements=["C"])
        index_buffer = np.require(mesh.trilist, dtype=np.uint32, requirements=["C"])
        self._get_packed_vbo(packed_buffer.tobytes())
        self._get_index_buffer_vbo(index_buffer.tobytes())
        # Note the ordering of the inputs list matches the ordering of the
        # packed vector above
        return self.context.simple_vertex_array(
            self.program,
            self.packed_vbo,
            "in_vert",
            "in_text",
            "in_f3v",
            index_buffer=self.index_buffer_vbo,
        )


class PerVertexPassthroughProgram(BasePassthroughProgram):
    def __init__(self, context):
        """
        Implements a "passthrough" shader for coloured meshes. This just
        rasterizes the mesh without considering any lighting or mesh normals.

        Parameters
        ----------
        context : moderngl context
            The context used for rendering
        """
        super().__init__(context)
        self.program = self.context.program(
            vertex_shader=PASSTHROUGH_PER_VERTEX_VERT_SHADER.read_text(
                encoding="ascii"
            ),
            fragment_shader=PASSTHROUGH_PER_VERTEX_FRAG_SHADER.read_text(
                encoding="ascii"
            ),
        )

    def __del__(self):
        super().__del__()
        # Release program last
        if self.program is not None:
            self.program.release()

    def create_vao(self, mesh, per_vertex_f3v):
        """
        Create a simple Vertex Array Object (VAO) that contains the per vertex
        array data used by the shader. Ensures that the per-vertex colour has
        the correct format (n_points, 3). Note that if no per-vertex colour
        is passed then a uniform gray colour is used (0.5, 0.5, 0.5).

        Parameters
        ----------
        mesh :  `menpo.shape.ColouredTriMesh`
        per_vertex_f3v : ndarray, (n_points, 3)

        Returns
        -------
        `moderngl.vertex_array.VertexArray`
            VAO to be used by the moderngl rendering context
        """
        colours = mesh.colours
        if colours is None:
            # Default to gray per vertex colouring
            colours = np.full(mesh.points.shape, 0.5, dtype=np.float32)
        if colours.shape[-1] == 1:
            # Force from grayscale to colour by repeating
            colours = np.repeat(colours, 3, axis=-1)
        assert colours.shape[-1] == 3, "Only RGB colours are supported"

        packed = np.concatenate([mesh.points, colours, per_vertex_f3v], axis=1)

        packed_buffer = np.require(packed, dtype=np.float32, requirements=["C"])
        index_buffer = np.require(mesh.trilist, dtype=np.uint32, requirements=["C"])
        self._get_packed_vbo(packed_buffer.tobytes())
        self._get_index_buffer_vbo(index_buffer.tobytes())
        return self.context.simple_vertex_array(
            self.program,
            self.packed_vbo,
            "in_vert",
            "in_color",
            "in_f3v",
            index_buffer=self.index_buffer_vbo,
        )


class GLRasterizer:
    def __init__(
        self,
        width=1024,
        height=768,
        model_matrix=None,
        view_matrix=None,
        projection_matrix=None,
    ):
        # Make a single OpenGL context that will be managed by the lifetime of
        # this class. We will dynamically create two default "pass through"
        # programs based on the type of the input mesh
        self.opengl_ctx = moderngl.create_standalone_context()
        self.width = width
        self.height = height

        self._model_matrix = model_matrix if model_matrix is not None else np.eye(4)
        self._view_matrix = view_matrix if view_matrix is not None else np.eye(4)
        self._projection_matrix = (
            projection_matrix if projection_matrix is not None else np.eye(4)
        )
        self._vertex_shader = None
        self._fragment_shader = None
        self._shader_type = None
        # We will dynamically build the program based on the mesh type
        self._active_program = None
        self._texture_program = None
        self._per_vertex_program = None

        self._f3v_renderbuffer = self.opengl_ctx.renderbuffer(
            self.size, components=3, dtype="f4"
        )
        self._rgba_renderbuffer = self.opengl_ctx.renderbuffer(
            self.size, components=4, dtype="f4"
        )
        self._depth_renderbuffer = self.opengl_ctx.depth_renderbuffer(self.size)
        self._fbo = self.opengl_ctx.framebuffer(
            [self._rgba_renderbuffer, self._f3v_renderbuffer], self._depth_renderbuffer
        )

    def __del__(self):
        # Ensure we avoid memory leaks by manually releasing all moderngl
        # objects
        self._rgba_renderbuffer.release()
        self._f3v_renderbuffer.release()
        self._depth_renderbuffer.release()

    @property
    def size(self):
        return self.width, self.height

    @property
    def model_matrix(self):
        return self._model_matrix

    @model_matrix.setter
    def model_matrix(self, value):
        self._model_matrix = _verify_opengl_homogeneous_matrix(value)

    @property
    def view_matrix(self):
        return self._view_matrix

    @view_matrix.setter
    def view_matrix(self, value):
        self._view_matrix = _verify_opengl_homogeneous_matrix(value)

    @property
    def projection_matrix(self):
        return self._projection_matrix

    @projection_matrix.setter
    def projection_matrix(self, value):
        self._projection_matrix = _verify_opengl_homogeneous_matrix(value)

    @property
    def mvp_matrix(self):
        return np.linalg.multi_dot(
            [self.projection_matrix, self.view_matrix, self.model_matrix]
        )

    @property
    def model_to_clip_matrix(self):
        return np.dot(
            self.projection_matrix, np.dot(self.view_matrix, self.model_matrix)
        )

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
    def mvp_transform(self):
        return Homogeneous(self.mvp_matrix)

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
        return self.model_to_clip_transform.compose_before(self.clip_to_image_transform)

    def _set_active_program_to_texture(self):
        # Ensure that we only build a single texture program per rasterizer
        if self._texture_program is None:
            self._texture_program = TexturePassthroughProgram(self.opengl_ctx)
        self._active_program = self._texture_program
        self._shader_type = _LOADED_SHADER_TYPE.TEXTURE

    def _set_active_program_to_per_vertex(self):
        # Ensure that we only build a single per vertex program per rasterizer
        if self._per_vertex_program is None:
            self._per_vertex_program = PerVertexPassthroughProgram(self.opengl_ctx)
        self._active_program = self._per_vertex_program
        self._shader_type = _LOADED_SHADER_TYPE.PER_VERTEX

    def _set_active_program_by_mesh_type(self, mesh):
        # Set the active program based on the input mesh type - this allows
        # lazy instantiation of the program
        if hasattr(mesh, "tcoords"):
            self._set_active_program_to_texture()
        elif hasattr(mesh, "colours"):
            self._set_active_program_to_per_vertex()
        else:
            raise ValueError(
                "Unknown mesh type, only textured (tcoords) or "
                "coloured (colours) TriMeshes are supported"
            )

    @with_context
    def _rasterize(self, mesh, per_vertex_f3v, fetch_f3v=True):
        """
        This defines the main rasterize method with moderngl. Ensures that
        the program is setup correctly and that the outputs are also parsed
        correctly.

        Parameters
        ----------
        mesh : `menpo.shape.ColouredTriMesh` or `menpo.shape.TexturedTriMesh`
            Mesh to render
        per_vertex_f3v : ndarray, (n_points, 3)
            Per vertex floats to render
        fetch_f3v : bool
            If True, then fetch the per-vertex floats, otherwise skip fetching them.
            Note that at the moment they are always rendered but we can skip
            fetching them if we don't use them.

        Returns
        -------
        rgb_image : ndarray float32, [H, W, 3]
            RGB image representing the rasterized image (with either the
            texture rasterized or the per vertex colours)
        f3v_image : ndarray float32, [H, W, 3] or None if fetch_f3v is False
            RGB image representing the rasterized per-vertex arbitrary floating
            point values
        mask : ndarray bool, [H, W, 1]
            True where pixels were written to, False otherwise
        """
        self._active_program.set_mvp(self.mvp_matrix)
        vao = self._active_program.create_vao(mesh, per_vertex_f3v)

        # Rendering
        self._fbo.use()
        self.opengl_ctx.clear(1.0, 1.0, 1.0, 0.0)
        self.opengl_ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        if self._shader_type == _LOADED_SHADER_TYPE.TEXTURE:
            self._active_program.texture.use()

        with safe_release(vao):
            vao.render()

        # Note that this is dependent on the shader output locations and
        # how the renderbuffers are wired up
        rgba_data = self._fbo.read(components=4, attachment=0, dtype="f4")
        f3v_data = None
        if fetch_f3v:
            f3v_data = self._fbo.read(components=3, attachment=1, dtype="f4")

        rgba_image = np.frombuffer(rgba_data, dtype=np.float32)
        rgba_image = rgba_image.reshape(self.height, self.width, 4)[::-1]
        rgba_image = np.require(rgba_image, dtype=np.float32, requirements=["C"])

        f3v_image = None
        if f3v_data is not None:
            f3v_image = np.frombuffer(f3v_data, dtype=np.float32)
            f3v_image = f3v_image.reshape(self.height, self.width, 3)[::-1]
            f3v_image = np.require(f3v_image, dtype=np.float32, requirements=["C"])

        return rgba_image[..., :3], f3v_image, rgba_image[..., -1].astype(np.bool)

    def rasterize_mesh_with_f3v_interpolant(self, mesh, per_vertex_f3v=None):
        r"""
        Rasterize the object to an image and generate an interpolated
        3-float image from a per vertex float 3 vector.

        Parameters
        ----------
        mesh : object implementing the Rasterizable interface.
        per_vertex_f3v : optional, ndarray (n_points, 3)
            A per-vertex 3 vector of floats that will be interpolated across
            the image.
            If None, a zero is passed for every vertex.

        Returns
        -------
        rgb_image : 3-channel MaskedImage of shape (width, height)
            The result of the rasterization. Mask is true iff the pixel was
            rendered to by OpenGL.

        interp_image: 3 channel MaskedImage of shape (width, height)
            The result of interpolating the per_vertex_f3v across the
            visible primitives.

        """
        if not (hasattr(mesh, "points") and hasattr(mesh, "trilist")):
            raise ValueError(
                "Rasterizable types must have points and trilist properties."
            )

        if per_vertex_f3v is None:
            per_vertex_f3v = np.zeros(mesh.points.shape, dtype=np.float32)

        self._set_active_program_by_mesh_type(mesh)

        rgb_pixels, f3v_pixels, mask = self._rasterize(mesh, per_vertex_f3v)
        images = (
            MaskedImage.init_from_channels_at_back(rgb_pixels, mask=mask),
            MaskedImage.init_from_channels_at_back(f3v_pixels, mask=mask),
        )

        # Transform all landmarks and set them on the image
        image_lms = self.model_to_image_transform.apply(mesh.landmarks)
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
        return self.rasterize_mesh_with_f3v_interpolant(
            mesh, per_vertex_f3v=mesh.points
        )

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
        self._set_active_program_by_mesh_type(mesh)

        per_vertex_f3v = np.zeros(mesh.points.shape, dtype=np.float32)
        rgb_pixels, _, mask = self._rasterize(mesh, per_vertex_f3v, fetch_f3v=False)
        image = MaskedImage.init_from_channels_at_back(rgb_pixels, mask=mask)
        image.landmarks = self.model_to_image_transform.apply(mesh.landmarks)

        return image

    def rasterize_barycentric_coordinate_image(self, mesh):
        # Convert the mesh into a version with one vertex per triangle
        mesh, _ = dedup_vertices(mesh)

        per_vertex_f3v = tri_bcoords_for_mesh(mesh)

        images = self.rasterize_mesh_with_f3v_interpolant(
            mesh, per_vertex_f3v=per_vertex_f3v
        )

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
