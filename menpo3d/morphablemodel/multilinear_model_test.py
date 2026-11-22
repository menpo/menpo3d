import numpy as np
import pytest
from menpo.base import MenpoDeprecationWarning
from menpo.model import PCAModel
from menpo.shape import PointCloud

from menpo3d.morphablemodel.multilinear_model import MultilinearModel


def a_model(seed):
    rng = np.random.RandomState(seed)
    return PCAModel([PointCloud(rng.rand(10, 3)) for _ in range(6)])


def n_comps(model):
    return [m.n_components for m in model.models]


def test_trimmed_per_model_and_one_number_for_the_first():
    model = MultilinearModel(a_model(0), a_model(1))
    assert n_comps(model) == [5, 5] and model.n_components == 10
    assert n_comps(model.trim_components([2, 3])) == [2, 3]
    assert model.trim_components([2, 3]).n_components == 5
    assert n_comps(model.trim_components(3)) == [3, 5]  # the first model only
    assert n_comps(model) == [5, 5]  # left as it was


def test_trimmed_gives_new_models_that_share_the_memory():
    model = MultilinearModel(a_model(0), a_model(1))
    small = model.trim_components(3)
    for old, new in zip(model.models, small.models):
        assert new is not old
        assert np.shares_memory(new.components, old.components)


def test_an_instance_of_a_trimmed_model_uses_its_numbers_of_components():
    model = MultilinearModel(a_model(0), a_model(1)).trim_components([2, 3])
    instance = model.instance(np.zeros(5))
    assert isinstance(instance, PointCloud) and instance.n_points == 10
