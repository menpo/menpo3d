import menpo.io as mio
import menpo3d.io as m3dio
import numpy as np
from menpo3d.correspond import solve_pnp


def test_solvepnp():
    template = m3dio.import_builtin_asset.template_ply()
    image = mio.import_builtin_asset.lenna_png()

    # Comes from running 100 times and taking the mean
    expected = np.array(
        [
            [0.9188, 0.1053, 0.3804, -622.7377],
            [0.0395, -0.9834, 0.1769, 28059.479],
            [0.3927, -0.1475, -0.9077, 512972.2382],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # Ransac is random and we can't control the seed so we do a fuzzy test
    model_view_t, _ = solve_pnp(image, template, group="LJSON")
    np.testing.assert_allclose(model_view_t.h_matrix, expected, atol=1e-2)
