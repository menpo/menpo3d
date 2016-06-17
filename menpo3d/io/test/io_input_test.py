from nose.tools import raises
import numpy as np
from numpy.testing import assert_allclose
import menpo3d.io as mio
from menpo.shape import TriMesh, TexturedTriMesh, PointCloud
from menpo.image import Image


# ground truth bunny landmarks
bunny_leye = np.array([[-0.0907334573249135,
                        0.13944519477304135,
                        0.016432549244098166]])
bunny_reye = np.array([[-0.06188841143305087,
                        0.1404748336910087,
                        0.03628544974441027]])
bunny_nose = np.array([[-0.0824712814601403,
                        0.12724167358964233,
                        0.051332619501298894]])
bunny_mouth = np.array([[-0.08719271323528464,
                         0.11440556680892036,
                         0.044485263772198975],
                        [-0.0733831000935048,
                         0.11610087742529278,
                         0.051836298703986136]])


def test_import_asset_bunny():
    mesh = mio.import_builtin_asset('bunny.obj')
    assert(isinstance(mesh, TriMesh))
    assert(isinstance(mesh.points, np.ndarray))
    assert(mesh.points.shape[1] == 3)
    assert(isinstance(mesh.trilist, np.ndarray))
    assert(mesh.trilist.shape[1] == 3)


def test_import_asset_james():
    mesh = mio.import_builtin_asset('james.obj')
    assert(isinstance(mesh, TexturedTriMesh))
    assert(isinstance(mesh.points, np.ndarray))
    assert(mesh.points.shape[1] == 3)
    assert(isinstance(mesh.trilist, np.ndarray))
    assert(mesh.trilist.shape[1] == 3)
    assert(isinstance(mesh.texture, Image))
    assert(isinstance(mesh.tcoords, PointCloud))
    assert(mesh.tcoords.points.shape[1] == 2)


@raises(ValueError)
def test_import_incorrect_built_in():
    mio.import_builtin_asset('adskljasdlkajd.obj')


def test_json_landmarks_bunny():
    mesh = mio.import_builtin_asset('bunny.obj')
    assert('LJSON' in mesh.landmarks.group_labels)
    lms = mesh.landmarks['LJSON']
    labels = {'reye', 'mouth', 'nose', 'leye'}
    assert(len(labels - set(lms.labels)) == 0)
    assert_allclose(lms['leye'].points, bunny_leye, atol=1e-7)
    assert_allclose(lms['reye'].points, bunny_reye, atol=1e-7)
    assert_allclose(lms['nose'].points, bunny_nose, atol=1e-7)
    assert_allclose(lms['mouth'].points, bunny_mouth, atol=1e-7)


def test_custom_landmark_logic_bunny():
    def f(path):
        return {
            'no_nose': path.with_name('bunny_no_nose.ljson'),
            'full_set': path.with_name('bunny.ljson')
        }
    mesh = mio.import_mesh(mio.data_path_to('bunny.obj'), landmark_resolver=f)
    assert('no_nose' in mesh.landmarks.group_labels)
    lms = mesh.landmarks['no_nose']
    labels = {'reye', 'mouth', 'leye'}
    assert(len(set(lms.labels) - labels) == 0)
    assert_allclose(lms['leye'].points, bunny_leye, atol=1e-7)
    assert_allclose(lms['reye'].points, bunny_reye, atol=1e-7)
    assert_allclose(lms['mouth'].points, bunny_mouth, atol=1e-7)

    assert('full_set' in mesh.landmarks.group_labels)
    lms = mesh.landmarks['full_set']
    labels = {'reye', 'mouth', 'nose', 'leye'}
    assert(len(set(lms.labels) - labels) == 0)
    assert_allclose(lms['leye'].points, bunny_leye, atol=1e-7)
    assert_allclose(lms['reye'].points, bunny_reye, atol=1e-7)
    assert_allclose(lms['nose'].points, bunny_nose, atol=1e-7)
    assert_allclose(lms['mouth'].points, bunny_mouth, atol=1e-7)


def test_custom_landmark_logic_None_bunny():
    def f(mesh):
        return None
    mesh = mio.import_mesh(mio.data_path_to('bunny.obj'), landmark_resolver=f)
    assert(mesh.landmarks.n_groups == 0)


def test_json_landmarks_bunny_direct():
    lms = mio.import_landmark_file(mio.data_path_to('bunny.ljson'))
    labels = {'reye', 'mouth', 'nose', 'leye'}
    assert(len(labels - set(lms.labels)) == 0)
    assert_allclose(lms['leye'].points, bunny_leye, atol=1e-7)
    assert_allclose(lms['reye'].points, bunny_reye, atol=1e-7)
    assert_allclose(lms['nose'].points, bunny_nose, atol=1e-7)
    assert_allclose(lms['mouth'].points, bunny_mouth, atol=1e-7)


def test_ls_builtin_assets():
    assert(set(mio.ls_builtin_assets()) == {'bunny.ljson', 'bunny.obj',
                                            'bunny_no_nose.ljson',
                                            'james.jpg', 'james.mtl',
                                            'james.obj'})


def test_mesh_paths():
    ls = mio.mesh_paths(mio.data_dir_path())
    assert(len(list(ls)) == 2)


@raises(ValueError)
def test_import_landmark_files_wrong_path_raises_value_error():
    list(mio.import_landmark_files('asldfjalkgjlaknglkajlekjaltknlaekstjlakj'))


@raises(ValueError)
def test_import_meshes_wrong_path_raises_value_error():
    list(mio.import_meshes('asldfjalkgjlaknglkajlekjaltknlaekstjlakj'))
