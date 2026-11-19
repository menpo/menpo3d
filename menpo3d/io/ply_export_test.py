import numpy as np
import pytest
from numpy.testing import assert_allclose

import menpo3d.io as m3io
from menpo3d.io.output.mesh import ply_exporter


@pytest.mark.parametrize("binary", [False, True])
def test_ply_export_round_trip(tmp_path, binary):
    mesh = m3io.import_builtin_asset.bunny_obj()
    path = tmp_path / "bunny.ply"
    ply_exporter(mesh, path, binary=binary)
    back = m3io.import_mesh(path)
    assert_allclose(back.points, mesh.points, atol=1e-6)
    assert np.all(back.trilist == mesh.trilist)
