from types import SimpleNamespace

import numpy as np
import pytest
from menpo.model import PCAModel
from menpo.shape import PointCloud

from menpo3d.morphablemodel.fitter import trimmed_copy


def a_model(seed):
    rng = np.random.RandomState(seed)
    return PCAModel([PointCloud(rng.rand(10, 3)) for _ in range(8)])


def a_morphable_model():
    # what LucasKanadeMMFitter takes per scale: only the shape and texture models matter here
    return SimpleNamespace(
        shape_model=a_model(0), texture_model=a_model(1), landmarks="shared"
    )


def test_trimmed_copy_trims_the_shape_and_texture_models_and_leaves_the_model_alone():
    mm = a_morphable_model()
    small = trimmed_copy(mm, n_shape=3, n_texture=2)
    assert small.shape_model.n_components == 3 and small.texture_model.n_components == 2
    assert mm.shape_model.n_components == 7 and mm.texture_model.n_components == 7
    assert small.landmarks is mm.landmarks
    assert np.shares_memory(small.shape_model.components, mm.shape_model.components)


def test_trimmed_copy_none_gives_new_models_with_all_the_components():
    mm = a_morphable_model()
    small = trimmed_copy(mm)
    assert (
        small.shape_model is not mm.shape_model
        and small.texture_model is not mm.texture_model
    )
    assert small.shape_model.n_components == 7


def test_a_copy_per_scale_with_different_numbers():
    mm = a_morphable_model()
    scales = [trimmed_copy(mm, n_shape=n, n_texture=n) for n in (2, 5)]
    assert [s.shape_model.n_components for s in scales] == [2, 5]
    assert mm.shape_model.n_components == 7


def test_trimmed_copy_refuses_what_is_not_a_number():
    with pytest.raises(ValueError, match="integer or a float or None"):
        trimmed_copy(a_morphable_model(), n_shape="3")
