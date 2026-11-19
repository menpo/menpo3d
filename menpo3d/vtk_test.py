import menpo3d
import numpy as np
import pytest

pytest.importorskip("vtk")
from menpo.shape import TriMesh
from menpo3d.vtkutils import trimesh_to_vtk, trimesh_from_vtk
from numpy.testing import assert_allclose
from pytest import raises


def test_trimesh_to_vtk_and_back_is_same():
    bunny = menpo3d.io.import_builtin_asset.bunny_obj()
    bunny_vtk = trimesh_to_vtk(bunny)
    bunny_back = trimesh_from_vtk(bunny_vtk)
    assert_allclose(bunny.points, bunny_back.points)
    assert np.all(bunny.trilist == bunny_back.trilist)


def test_trimesh_to_vtk_fails_on_2d_mesh():
    with raises(ValueError):
        points = np.random.random((5, 2))
        test_mesh = TriMesh(points)
        trimesh_to_vtk(test_mesh)
