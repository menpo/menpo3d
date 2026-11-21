"""menpo3d's moderngl viewer: a scene (NumPy data), one OpenGL renderer and its GL thread. The renderer's frames can be shown
by a native window (``native.py``) or streamed to a notebook widget and a browser window by the scene-streamer package."""

MESSAGE = (
    "The interactive 3D viewer needs a few extra packages. "
    'Install them all, H.264 video streaming included, with `pip install "menpo3d[viz]"`. '
    "Or only what you need: "
    '`"menpo3d[viz_notebook]"` (notebook widget and browser window, JPEG frames), '
    '`"menpo3d[viz_h264]"` (adds H.264 video; PyAV), `"menpo3d[viz_native_window]"` (the native window). '
    "Offline rendering and `save_figure` need none of them."
)


def require_viewer():
    """Raise a helpful ImportError if scene-streamer is missing, and tell it how to reach the renderer."""
    try:
        import scene_streamer
    except ImportError as error:
        raise ImportError(MESSAGE) from error
    from .worker import get_worker

    scene_streamer.configure(worker=get_worker)
