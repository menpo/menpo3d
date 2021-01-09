import contextlib
import os
import tempfile
from unittest.mock import MagicMock, PropertyMock, patch

import menpo3d.io as mio

test_obj = mio.import_builtin_asset("james.obj")
test_lg = mio.import_landmark_file(mio.data_path_to("bunny.ljson"))


@contextlib.contextmanager
def _temporary_path(extension):
    # Create a temporary file and remove it
    fake_path = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
    fake_path.close()
    fake_path = fake_path.name
    os.unlink(fake_path)
    yield fake_path
    if os.path.exists(fake_path):
        os.unlink(fake_path)


@patch("menpo3d.io.output.base.Path.exists")
@patch("{}.open".format(__name__), create=True)
def test_export_mesh_obj(mock_open, exists):
    exists.return_value = False
    fake_path = "/fake/fake.obj"
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_mesh(test_obj, f, extension="obj")


def test_export_mesh_ply_ascii():
    with _temporary_path(".ply") as f:
        mio.export_mesh(test_obj, f)
        assert os.path.exists(f)


def test_export_mesh_ply_binary():
    with _temporary_path(".ply") as f:
        mio.export_mesh(test_obj, f, binary=True)
        assert os.path.exists(f)


@patch("PIL.Image.EXTENSION")
@patch("menpo.image.base.PILImage")
@patch("menpo3d.io.output.base.Path.exists")
@patch("menpo.io.output.base.Path.open")
def test_export_mesh_obj_textured(mock_open, exists, PILImage, PIL):
    PIL.return_value.Image.EXTENSION = {".jpg": None}
    exists.return_value = False
    mock_open.return_value = MagicMock()
    fake_path = "/fake/fake.obj"
    mio.export_textured_mesh(test_obj, fake_path, extension="obj")
    assert PILImage.fromarray.called


@patch("PIL.Image.EXTENSION")
@patch("menpo.image.base.PILImage")
@patch("menpo3d.io.output.base.Path.exists")
@patch("menpo.io.output.base.Path.open")
def test_export_mesh_ply_textured(mock_open, exists, PILImage, PIL):
    PIL.return_value.Image.EXTENSION = {".jpg": None}
    exists.return_value = False
    mock_open.return_value = MagicMock()
    with _temporary_path(".ply") as f:
        mio.export_textured_mesh(test_obj, f)
        assert PILImage.fromarray.called


@patch("menpo.io.output.landmark.json.dump")
@patch("menpo3d.io.output.base.Path.exists")
@patch("{}.open".format(__name__), create=True)
def test_export_landmark_ljson(mock_open, exists, json_dump):
    exists.return_value = False
    fake_path = "/fake/fake.ljson"
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_landmark_file(test_lg, f, extension="ljson")
    assert json_dump.called
