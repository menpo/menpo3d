"""One thread owns the OpenGL context (EGL contexts belong to the thread that made them). Offline renders, the
notebook widget and the standalone window all go through it."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

_worker = None
_lock = threading.Lock()


class Worker:
    def __init__(self, backend=None, samples=4):
        self.renderer = None
        self._backend, self._samples = backend, samples
        self.pool = ThreadPoolExecutor(
            1, thread_name_prefix="menpo3d-gl", initializer=self._init
        )
        self.pool.submit(
            lambda: None
        ).result()  # start the thread now; raises here if there is no GL
        if self.renderer is None:
            raise RuntimeError(self._error)

    def _init(self):
        from .gl import GLRenderer

        try:
            self.renderer = GLRenderer(self._backend, self._samples)
        except Exception as error:  # noqa: BLE001
            self._error = f"could not start the OpenGL renderer: {error}"

    @property
    def gl_info(self):
        return self.renderer.gl_info

    def call(self, fn, *args):
        """Run fn(renderer, *args) on the GL thread and wait for the result."""
        return self.pool.submit(fn, self.renderer, *args).result()

    def run(self, fn, *args):
        """Awaitable version for asyncio code."""
        return asyncio.get_running_loop().run_in_executor(
            self.pool, fn, self.renderer, *args
        )

    def render(self, scene, camera, width, height, style=None):
        return self.call(
            lambda r, *a: _copy(r.render(*a)), scene, camera, width, height, style
        )


def _copy(result):
    rgb, mvp = result
    return rgb.copy(), mvp


def get_worker():
    global _worker
    with _lock:
        if _worker is None:
            _worker = Worker()
        return _worker
