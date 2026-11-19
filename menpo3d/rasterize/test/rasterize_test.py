import numpy as np
import pytest

import menpo3d.io as m3io
from menpo.shape import ColouredTriMesh, TriMesh
from menpo.transform import Translation, UniformScale
from menpo3d.rasterize import GLRasterizer, rasterize_barycentric_coordinate_images
from menpo3d.rasterize import cpu


def a_triangle():
    return TriMesh(
        np.array([[10, 10, 1], [40, 10, 1], [10, 40, 1.0]]), np.array([[0, 1, 2]])
    )


def test_cpu_rasterizes_a_triangle():
    # np.product, used to count the pixels to check, was removed in NumPy 2
    # and tripixel used the Python 2 `long`
    bcoords, tri_indices = rasterize_barycentric_coordinate_images(
        a_triangle(), (50, 50)
    )
    mask = bcoords.mask.pixels[0]
    area = 30 * 30 / 2
    assert area <= mask.sum() <= area + 4 * 30  # the area plus the edge pixels
    assert np.allclose(bcoords.pixels.sum(axis=0)[mask], 1)
    assert np.all(tri_indices.pixels[0][mask] == 0)


def test_cython_pixel_checking_matches_the_python_fallback():
    rng = np.random.RandomState(0)
    start = rng.randint(0, 20, size=(30, 2)).astype(np.int64)
    end = start + rng.randint(0, 6, size=(30, 2)).astype(np.int64)
    n_pixels = int(np.prod(end - start, axis=1).sum())
    expected = cpu._pixels_to_check_python(start, end, n_pixels)
    from menpo3d.rasterize.tripixel import pixels_to_check

    for got, want in zip(pixels_to_check(start, end, n_pixels), expected):
        assert np.array_equal(got, want)


def a_coloured_bunny():
    bunny = m3io.import_builtin_asset.bunny_obj()
    centred = Translation(-bunny.centre()).apply(bunny)
    # into the [-1, 1] clip volume of the default (identity) camera
    scaled = UniformScale(0.9 / np.abs(centred.points).max(), 3).apply(centred)
    colours = np.random.RandomState(0).rand(scaled.n_points, 3)
    return ColouredTriMesh(scaled.points, scaled.trilist, colours=colours)


@pytest.fixture(scope="module")
def gl_rasterizer():
    moderngl = pytest.importorskip("moderngl")
    try:
        return GLRasterizer(width=96, height=96)
    except Exception as error:  # no OpenGL on this machine
        pytest.skip(f"no OpenGL context: {error}")


def test_gl_rasterizes_a_coloured_mesh(gl_rasterizer):
    image = gl_rasterizer.rasterize_mesh(a_coloured_bunny())
    assert image.shape == (96, 96)
    assert image.n_channels == 3
    assert 0.1 < image.mask.pixels.mean() < 0.9


def test_gl_barycentric_coordinate_image(gl_rasterizer):
    # this raised "Unknown mesh type": the mesh was converted to a plain TriMesh
    mesh = a_coloured_bunny()
    tri_indices, bcoords = gl_rasterizer.rasterize_barycentric_coordinate_image(mesh)
    mask = tri_indices.mask.pixels[0]
    assert mask.sum() > 1000
    assert np.all(tri_indices.pixels[0][mask] < mesh.n_tris)
    assert np.allclose(bcoords.pixels.sum(axis=0)[mask], 1, atol=1e-4)


def test_gl_and_cpu_cover_the_same_pixels(gl_rasterizer):
    mesh = a_coloured_bunny()
    gl_mask = gl_rasterizer.rasterize_mesh(mesh).mask.pixels[0]
    in_image = gl_rasterizer.model_to_image_transform.apply(mesh)
    cpu_mask = rasterize_barycentric_coordinate_images(in_image, (96, 96))[
        0
    ].mask.pixels[0]
    intersection = (gl_mask & cpu_mask).sum()
    union = (gl_mask | cpu_mask).sum()
    assert intersection / union > 0.95
