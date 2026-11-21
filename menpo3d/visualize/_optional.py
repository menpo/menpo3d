r"""Mayavi (and the VTK it brings) is an optional dependency of menpo3d."""

_MESSAGE = (
    "The 3D viewers need Mayavi, which is not installed. "
    'Install it with `pip install "menpo3d[mayavi]"` '
    "(or `conda install -c conda-forge mayavi`)."
)


def import_mlab():
    try:
        import mayavi.mlab as mlab
    except ImportError as error:
        raise ImportError(_MESSAGE, name="mayavi") from error
    return mlab


def import_tvtk():
    try:
        from tvtk.api import tvtk
    except ImportError as error:
        raise ImportError(_MESSAGE, name="mayavi") from error
    return tvtk
