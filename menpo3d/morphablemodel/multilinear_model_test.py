import numpy as np

from menpo.model import PCAModel
from menpo.shape import PointCloud
from menpo3d.morphablemodel.multilinear_model import MultilinearModel


def a_model(seed):
    rng = np.random.RandomState(seed)
    return PCAModel([PointCloud(rng.rand(10, 3)) for _ in range(6)])


def test_active_components_can_be_set_per_model():
    # the setter used an undefined name (`value`) instead of the loop variable
    model = MultilinearModel(a_model(0), a_model(1))
    model.n_active_components = [2, 3]
    assert [m.n_active_components for m in model.models] == [2, 3]
    assert model.n_active_components == 5
