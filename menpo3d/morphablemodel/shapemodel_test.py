import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from menpo.model import PCAModel
from menpo.shape import PointCloud
from menpo3d.morphablemodel.shapemodel import ShapeModel


def a_shape_model(seed=0):
    rng = np.random.RandomState(seed)
    return ShapeModel(PCAModel([PointCloud(rng.rand(10, 3)) for _ in range(8)]))


def test_project_onto_the_first_components():
    sm = a_shape_model()
    instance = PointCloud(np.random.RandomState(5).rand(10, 3))
    k = 3
    expected = np.linalg.lstsq(
        sm.model.components[:k].T,
        instance.as_vector() - sm.model.mean_vector,
        rcond=None,
    )[0]
    weights = sm.project(instance, n_components=k)
    assert weights.shape == (k,)
    assert_allclose(weights, expected, atol=1e-10)
    # all of them, by default and by asking for more than there are
    assert sm.project(instance).shape == (sm.n_components,)
    assert_allclose(sm.project(instance, n_components=100), sm.project(instance))


def test_project_with_n_components_leaves_the_model_as_it_was():
    sm = a_shape_model()
    instance = PointCloud(np.random.RandomState(5).rand(10, 3))
    components, n_active = sm.model.components.copy(), sm.model.n_components
    sm.project(instance, n_components=3)
    assert sm.model.n_components == n_active == sm.n_components
    assert_array_equal(sm.model.components, components)


def test_project_with_a_bad_n_components_raises_and_leaves_the_model_as_it_was():
    sm = a_shape_model()
    instance = PointCloud(np.random.RandomState(5).rand(10, 3))
    with pytest.raises(ValueError):
        sm.project(instance, n_components=-1)
    assert sm.model.n_components == sm.n_components
