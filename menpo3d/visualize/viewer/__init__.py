"""menpo3d's moderngl viewer: a scene (NumPy data), one OpenGL renderer, and an optional streaming hub that shows what the
renderer draws in a notebook widget or a standalone window."""

MESSAGE = (
    "The interactive 3D viewer needs a few extra packages. "
    'Install them with `pip install "menpo3d[viewer]"` (offline rendering, `save_figure`, needs none of them).'
)


def require_viewer():
    """Raise a helpful ImportError if the streaming extras are missing."""
    try:
        import aiohttp  # noqa: F401
        import av  # noqa: F401
        import simplejpeg  # noqa: F401
    except ImportError as error:
        raise ImportError(MESSAGE) from error
