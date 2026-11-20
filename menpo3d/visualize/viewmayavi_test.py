import sys

import pytest

from menpo3d.visualize._optional import import_mlab, import_tvtk


@pytest.mark.parametrize("importer", [import_mlab, import_tvtk])
def test_viewers_explain_how_to_install_mayavi(monkeypatch, importer):
    for name in ("mayavi", "mayavi.mlab", "tvtk", "tvtk.api"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match=r"menpo3d\[mayavi\]"):
        importer()
