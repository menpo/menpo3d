import numpy as np

import menpo3d.io as m3io
from menpo.transform import Translation
from menpo3d.correspond import non_rigid_icp


def test_non_rigid_icp_moves_a_mesh_onto_a_translated_copy():
    # np.in1d, used to mask the landmarks and edges, is deprecated by NumPy 2
    source = m3io.import_builtin_asset.bunny_obj()
    target = Translation([0.01, 0.0, 0.0]).apply(source)
    result = non_rigid_icp(source, target, eps=1e-3)
    extent = np.ptp(target.points, axis=0).max()
    assert np.abs(result.points - target.points).max() < 0.05 * extent
    assert result.n_points == source.n_points
