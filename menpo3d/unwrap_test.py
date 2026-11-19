import numpy as np
from numpy.testing import assert_allclose

import menpo3d.io as m3io
from menpo3d.unwrap import optimal_cylindrical_unwrap


def test_cylindrical_unwrap_keeps_the_height():
    bunny = m3io.import_builtin_asset.bunny_obj()
    unwrap = optimal_cylindrical_unwrap(bunny)
    unwrapped = unwrap.apply(bunny)
    assert unwrapped.points.shape == bunny.points.shape
    assert np.all(np.isfinite(unwrapped.points))
    # the cylinder's axis is y: the height is unchanged
    assert_allclose(unwrapped.points[:, 1], bunny.points[:, 1], atol=1e-9)
