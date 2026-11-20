"""The standalone viewer window: a browser window in "app" mode (no tabs, no address bar) pointed at the hub.

It is the same page and the same renderer as the notebook widget, so it also works over SSH (forward the port, or set
``configure(public_url=...)``). ``show(figure)`` blocks until the window is closed, like ``mlab.show()``.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser

from .hub import get_hub

_CANDIDATES = (
    "google-chrome-stable",
    "google-chrome",
    "chromium",
    "chromium-browser",
    "chrome",
    "microsoft-edge",
    "brave-browser",
)
_MAC = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)


def find_browser():
    """A Chromium-family browser (they have --app windows and WebCodecs), or None."""
    env = os.environ.get("MENPO3D_BROWSER")
    if env:
        return env
    for name in _CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    if sys.platform == "darwin":
        for path in _MAC:
            if os.path.exists(path):
                return path
    if sys.platform == "win32":
        for base in filter(
            None,
            (
                os.environ.get("PROGRAMFILES"),
                os.environ.get("PROGRAMFILES(X86)"),
                os.environ.get("LOCALAPPDATA"),
            ),
        ):
            for rel in (
                r"Google\Chrome\Application\chrome.exe",
                r"Microsoft\Edge\Application\msedge.exe",
            ):
                path = os.path.join(base, rel)
                if os.path.exists(path):
                    return path
    return None


def open_window(url, size=(900, 700)):
    """Open ``url`` in an app-mode window. Returns the browser process, or None if a normal tab had to do."""
    browser = find_browser()
    if browser is None:
        webbrowser.open(url)
        return None
    profile = os.path.join(
        tempfile.gettempdir(), "menpo3d-viewer-profile"
    )  # own profile: never joins (or kills) the user's session
    command = [
        browser,
        f"--app={url}",
        f"--user-data-dir={profile}",
        f"--window-size={size[0]},{size[1]}",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    return subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def show(figure, block=True, timeout=30.0):
    """Open the window on ``figure``. With ``block`` wait until it is closed (Ctrl-C closes it)."""
    hub = get_hub()
    hub.register(figure.scene)
    figure.scene.ensure_camera()
    process = open_window(hub.view_url(figure.scene), figure.size)
    if not block:
        return process
    start, seen, gone_since = time.monotonic(), False, None
    try:
        while True:
            time.sleep(0.25)
            viewers = hub.n_viewers(figure.scene)
            if viewers:
                seen, gone_since = True, None
            elif seen:
                gone_since = gone_since or time.monotonic()
                if (
                    time.monotonic() - gone_since > 3.0
                ):  # the window (and so its socket) is gone
                    break
            elif time.monotonic() - start > timeout:
                print(
                    f"no viewer connected within {timeout:.0f} s; open {hub.view_url(figure.scene)} in a browser",
                    file=sys.stderr,
                )
                break
            if (
                process is not None
                and process.poll() is not None
                and not viewers
                and seen
            ):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
    return None
