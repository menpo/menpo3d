"""One way to make an OpenGL context for everything in menpo3d (the rasterizer and the viewer).

EGL first: it needs no X server, so it works headless and over SSH. If EGL has no device the platform default is used.
Mixing backends in one process (an X11/GLX context for one user, EGL for another) makes the process exit with an X error,
which is why every context here comes from this function. ``MENPO3D_GL_BACKEND`` (``egl``, ``x11``, ...) overrides.
"""

import os


def create_context(backend=None):
    import moderngl

    backend = backend or os.environ.get("MENPO3D_GL_BACKEND")
    failures = []
    for choice in [backend] if backend else ["egl", None]:
        try:
            if choice:
                return moderngl.create_standalone_context(backend=choice)
            return moderngl.create_standalone_context()
        except Exception as error:  # noqa: BLE001 - no EGL device, no display, ...
            failures.append(f"{choice or 'default'}: {error}")
    raise RuntimeError(
        "could not create an OpenGL context ("
        + "; ".join(failures)
        + "). A GPU or software GL such as Mesa llvmpipe is needed."
    )
