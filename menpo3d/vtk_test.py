import menpo3d
from menpo3d.vtkutils import trimesh_to_vtk, trimesh_from_vtk
from numpy.testing import assert_allclose
import numpy as np
from nose.tools import raises
from menpo.shape import TriMesh

def test_trimesh_to_vtk_and_back_is_same():
    bunny = menpo3d.io.import_builtin_asset.bunny_obj()
    bunny_vtk = trimesh_to_vtk(bunny)
    bunny_back = trimesh_from_vtk(bunny_vtk)
    assert_allclose(bunny.points, bunny_back.points)
    assert np.all(bunny.trilist == bunny_back.trilist)


@raises(ValueError)
def test_trimesh_to_vtk_fails_on_2d_mesh():
    points = np.random.random((5, 2))
    test_mesh = TriMesh(points)
    trimesh_to_vtk(test_mesh)


def test_barycentric_rebuild_returns_same_as_snapped_points():
    mesh = menpo3d.io.import_builtin_asset.bunny_obj()
    lms = mesh.landmarks[None].lms
    bc = mesh.barycentric_coordinates_of_pointcloud(lms)
    recon_lms = mesh.project_barycentric_coordinates(*bc)
    direct_recon_lms = mesh.snap_pointcloud_to_surface(lms)[0]
    assert_allclose(recon_lms.points, direct_recon_lms.points)
