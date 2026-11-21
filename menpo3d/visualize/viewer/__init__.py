"""menpo3d's moderngl viewer: a scene (NumPy data), one OpenGL renderer, and an optional streaming hub that shows what the
renderer draws in a notebook widget or a standalone window."""

MESSAGE = (
    "The interactive 3D viewer needs a few extra packages. "
    'Install them all, H.264 video streaming included, with `pip install "menpo3d[viz]"`. '
    "Or only what you need: "
    '`"menpo3d[viz_notebook]"` (notebook widget and browser window, JPEG frames), '
    '`"menpo3d[viz_h264]"` (adds H.264 video; PyAV), `"menpo3d[viz_native_window]"` (the native window). '
    "Offline rendering and `save_figure` need none of them."
)


def require_viewer():
    """Raise a helpful ImportError if the streaming extras are missing."""
    try:
        import aiohttp  # noqa: F401
    except ImportError as error:
        raise ImportError(MESSAGE) from error
