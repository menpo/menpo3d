"""A native viewer window, built on moderngl-window: no browser, no network, no encoding.

It shows what the renderer draws, exactly like the browser viewer does, but the frames never leave the process: each frame
is rendered by the shared renderer (``Figure.render``, labels included), uploaded to a texture and drawn over the window.

Backends are tried in order: GLFW (a small wheel that bundles the library; X11, macOS, Windows, and Wayland desktops
through XWayland), then pyglet if it happens to be installed. It needs a display: over SSH use the browser viewer
instead (``figure.show(window="browser")``).
"""

import os
import time

import numpy as np

_QUAD_VS = """#version 330
out vec2 uv;
void main(){ vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); uv = vec2(p.x, 1.0 - p.y); gl_Position = vec4(p*2.0 - 1.0, 0.0, 1.0); }"""
_QUAD_FS = """#version 330
uniform sampler2D tex; in vec2 uv; out vec4 frag;
void main(){ frag = vec4(texture(tex, uv).rgb, 1.0); }"""

BACKENDS = ("glfw", "pyglet")
MESSAGE = 'The native window needs moderngl-window and glfw: pip install "menpo3d[viz_native_window]" (or use the browser viewer: figure.show(window="browser")).'


def require_window():
    try:
        import moderngl_window  # noqa: F401
    except ImportError as error:
        raise ImportError(MESSAGE) from error


def _prefer_x11_on_wayland():
    """On a Wayland desktop GLFW gives an EGL context, which moderngl cannot attach to; XWayland gives a GLX one it can."""
    if not (os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY")):
        return
    try:
        import glfw

        glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
    except Exception:  # noqa: BLE001 - GLFW older than 3.4 has no platform hint
        pass


def open_window(title, size, visible, backend=None):
    """The first backend that can open an OpenGL 3.3 window; RuntimeError listing why each failed otherwise."""
    import moderngl_window as mglw

    names = (
        [backend]
        if backend
        else (
            [os.environ["MENPO3D_WINDOW_BACKEND"]]
            if os.environ.get("MENPO3D_WINDOW_BACKEND")
            else BACKENDS
        )
    )
    failures = []
    for name in names:
        try:
            if name == "glfw":
                _prefer_x11_on_wayland()
            cls = mglw.get_window_cls(f"moderngl_window.context.{name}.Window")
            return cls(
                title=title,
                size=tuple(int(v) for v in size),
                gl_version=(3, 3),
                visible=visible,
                vsync=True,
                resizable=True,
                samples=0,
            )
        except Exception as error:  # noqa: BLE001 - backend not installed, no display, no 3.3 context...
            failures.append(f"{name}: {type(error).__name__}: {str(error)[:100]}")
    raise RuntimeError(
        "could not open a window ("
        + "; ".join(failures)
        + "). Over SSH use figure.show(window='browser')."
    )


class NativeWindow:
    """Drag: rotate. Shift- or right/middle-drag: pan. Wheel: zoom. Click: pick a triangle. r: reset view. s: save a PNG. q/Esc: close."""

    def __init__(self, figure, title=None, visible=True, backend=None):
        require_window()
        self.figure = figure
        self.win = open_window(
            title or f"menpo3d - {figure.name}", figure.size, visible, backend
        )
        self.ctx = self.win.ctx
        self.prog = self.ctx.program(vertex_shader=_QUAD_VS, fragment_shader=_QUAD_FS)
        self.vao = self.ctx.vertex_array(self.prog, [])
        self.texture = None
        self.dirty = True
        self.picked = None  # (x, y) in framebuffer pixels
        self.hit = None
        self.shown = None  # the last frame uploaded, (h, w, 3)
        self._down = self._pan = False
        self._moved = 0.0
        self._last = (0.0, 0.0)
        w = self.win
        w.resize_func = lambda *_: self.invalidate()
        w.mouse_press_event_func = lambda x, y, button: self.press(
            x, y, button, w.modifiers.shift
        )
        w.mouse_drag_event_func = lambda x, y, dx, dy: self.drag(x, y, dx, dy)
        w.mouse_release_event_func = lambda x, y, button: self.release(x, y, button)
        w.mouse_scroll_event_func = lambda dx, dy: self.scroll(dy)
        w.key_event_func = lambda key, action, modifiers: (
            self.key(key) if action == w.keys.ACTION_PRESS else None
        )
        figure.scene.listeners.append(self._scene_changed)

    # ---- state
    def invalidate(self):
        self.dirty = True

    def _scene_changed(self, scene):
        self.dirty = True  # picked up by the loop within a frame, from whichever thread changed the scene

    def size(self):
        w, h = self.win.buffer_size
        return max(int(w), 2) // 2 * 2, max(int(h), 2) // 2 * 2

    def scale(self):
        """Framebuffer pixels per window coordinate (2 on a HiDPI screen)."""
        return float(self.win.pixel_ratio)

    # ---- input (called by moderngl-window; tests call these directly)
    def press(self, x, y, button, shift=False):
        self._down = True
        self._moved = 0.0
        self._pan = button in (self.win.mouse.right, self.win.mouse.middle) or bool(
            shift
        )
        self._last = (x, y)

    def drag(self, x, y, dx, dy):
        if not self._down:
            return
        k = self.scale()
        self._last = (x, y)
        self._moved += abs(dx) + abs(dy)
        self.picked = None
        cam = self.figure.scene.camera
        if self._pan:
            cam.pan(dx * k, dy * k, self.size()[1])
        else:
            cam.orbit(dx * k, dy * k)
        self.dirty = True

    def release(self, x, y, button):
        if not self._down:
            return
        self._down = False
        if self._moved < 4 and not self._pan and button == self.win.mouse.left:
            k = self.scale()
            self.pick(x * k, y * k)

    def scroll(self, dy):
        self.figure.scene.camera.zoom(-dy * 100.0)
        self.picked = None
        self.dirty = True

    def key(self, key):
        keys = self.win.keys
        if key in (keys.Q, keys.ESCAPE):
            self.win.close()
        elif key == keys.R:
            cam = self.figure.scene.camera
            cam.azimuth, cam.elevation = 0.0, 0.0
            self.figure.scene.fit_camera()
            self.picked = None
            self.dirty = True
        elif key == keys.S:
            path = os.path.abspath(f"{self.figure.name.replace(' ', '_')}.png")
            self.figure.save(path, size=self.size())
            self.win.title = f"menpo3d - saved {path}"

    def pick(self, x, y):
        from .worker import get_worker

        w, h = self.size()
        scene = self.figure.scene
        hit = get_worker().call(
            lambda r, *a: r.pick(*a),
            scene,
            scene.camera.copy(),
            w,
            h,
            float(x),
            float(y),
        )
        self.picked = (x, y) if hit else None
        self.hit = hit
        self.win.title = "menpo3d - " + (
            f"triangle {hit['tri']}, vertex {hit['vertex']}"
            if hit
            else self.figure.name
        )
        self.dirty = True
        return hit

    # ---- drawing
    def render_frame(self):
        w, h = self.size()
        rgb = self.figure.render((w, h))
        if self.picked is not None:
            rgb = rgb.copy()
            x, y = (int(round(v)) for v in self.picked)
            rgb[max(y - 1, 0) : y + 2, max(x - 9, 0) : x + 10] = (255, 40, 40)
            rgb[max(y - 9, 0) : y + 10, max(x - 1, 0) : x + 2] = (255, 40, 40)
        return rgb

    def draw(self, swap=True):
        """Render and upload if anything changed, then draw the picture over the window. Returns whether it re-rendered."""
        rendered = False
        if self.dirty:
            self.dirty = False
            rgb = self.render_frame()
            h, w, _ = rgb.shape
            if self.texture is None or self.texture.size != (w, h):
                if self.texture is not None:
                    self.texture.release()
                self.texture = self.ctx.texture((w, h), 3)
                self.texture.filter = (self.ctx.NEAREST, self.ctx.NEAREST)
            self.texture.write(np.ascontiguousarray(rgb).tobytes())
            self.shown = rgb
            rendered = True
        if self.texture is not None:
            self.win.use()
            self.ctx.disable(self.ctx.DEPTH_TEST)
            self.texture.use(0)
            self.vao.render(vertices=3)
        if swap:
            self.win.swap_buffers()  # also pulls the window system's events
        return rendered

    def read_screen(self):
        """What the window's back buffer holds now (for tests, call after ``draw(swap=False)``): (h, w, 3), top row first."""
        w, h = self.win.buffer_size
        data = self.win.fbo.read(viewport=(0, 0, w, h), components=3)
        return np.frombuffer(data, np.uint8).reshape(h, w, 3)[::-1].copy()

    @property
    def closing(self):
        return self.win.is_closing

    def run(self):
        """Block until the window is closed."""
        try:
            while not self.win.is_closing:
                self.draw()
                if not self.dirty:
                    time.sleep(
                        0.005
                    )  # vsync paces the loop; this just keeps an idle window cheap
        finally:
            self.close()

    def close(self):
        if self.win is None:
            return
        try:
            self.figure.scene.listeners.remove(self._scene_changed)
        except ValueError:
            pass
        if self.texture is not None:
            self.texture.release()
        self.win.destroy()
        self.win = None


def show(figure, block=True):
    """Open the native window on ``figure``; with ``block`` return when it is closed."""
    window = NativeWindow(figure)
    if block:
        window.run()
        return None
    return window
