import sys

import pytest

from menpo3d.visualize._optional import import_mlab, import_tvtk


@pytest.mark.parametrize("importer", [import_mlab, import_tvtk])
def test_viewers_explain_how_to_install_mayavi(monkeypatch, importer):
    for name in ("mayavi", "mayavi.mlab", "tvtk", "tvtk.api"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match=r"menpo3d\[mayavi\]"):
        importer()


def test_a_viewer_without_mayavi_raises_an_error_that_names_the_package(monkeypatch):
    # menpo wraps this in Menpo3dMissingError, which reads the package name from ``name``
    from menpo3d.visualize.viewmayavi import MayaviRenderer

    for name in ("mayavi", "mayavi.mlab", "tvtk", "tvtk.api"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match=r"menpo3d\[mayavi\]") as error:
        MayaviRenderer(None, False)
    assert error.value.name == "mayavi"
