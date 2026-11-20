"""The streaming hub: an asyncio/aiohttp server running in a thread of the *same process* that owns the scenes.

Viewers (a notebook widget, a standalone window) connect with a websocket and receive frames the shared GL thread renders
from the scene's camera: H.264 for interactive use, a lossless-ish still when the input stops. Because the scene lives in
the process, changing it from Python redraws every attached viewer, and a figure's camera is shared by all its viewers.

One websocket carries any number of viewers (tagged with a viewer id), so ten widgets in a notebook cost one connection.
Frames are paused for viewers that are scrolled out of view or frozen, and everything a viewer held is released when its
page goes away.
"""

import asyncio
import collections
import copy
import gc
import io
import json
import secrets
import struct
import threading
import time
import warnings
import weakref
from pathlib import Path

HEADER = struct.Struct(
    "<BHIdff16f"
)  # kind, viewer id, seq, echoed client ms, render ms, encode ms, 4x4 view-projection (row-major)
DELTA, KEY, JPEG, STILL = 0, 1, 2, 3
STILL_AFTER = 0.2
MAX_INFLIGHT = 2
LEVEL_STEP = 6  # quality ladder: crf/qp + 6 per level (JPEG: quality x (1 - 0.3 level))
STATIC = Path(__file__).with_name("static")


class Session:
    """One viewer on one scene."""

    def __init__(self, conn, vid, hub):
        self.conn, self.vid, self.hub = conn, vid, hub
        self.scene = None
        self.held = None
        self.w, self.h = 800, 600
        self.codec, self.quality, self.bw, self.rtt = "h264", 26, 0.0, 0.0
        self.adapt, self.level, self.level_changed = True, 0, 0.0
        self.sent, self.lat_ewma, self.lat_recent = (
            collections.deque(),
            None,
            collections.deque(maxlen=90),
        )
        self.encoder = self.encoder_key = None
        self.need_key = True
        self.seq = self.inflight = 0
        self.window = MAX_INFLIGHT
        self.visible, self.paused = True, False
        self.last_msg = time.monotonic()
        self.dirty, self.want_still = True, False
        self.last_input_ts = 0.0
        self.wake = asyncio.Event()
        self.link_free = 0.0
        self.still_timer = None
        self.closed = False

    # ---- rate control: the quality ladder moves on ack latency relative to the best latency seen lately
    def effective_quality(self):
        if not self.adapt or self.level == 0:
            return self.quality
        if self.codec == "jpeg":
            return max(15, int(self.quality * (1 - 0.3 * self.level)))
        return min(51, self.quality + LEVEL_STEP * self.level)

    def note_ack(self):
        if not self.sent:
            return
        loop = asyncio.get_running_loop()
        sent_at, _ = self.sent.popleft()
        lat = 1000 * (loop.time() - sent_at)
        self.lat_recent.append(lat)
        self.lat_ewma = (
            lat if self.lat_ewma is None else 0.8 * self.lat_ewma + 0.2 * lat
        )
        if not self.adapt or len(self.lat_recent) < 8:
            return
        floor, now = min(self.lat_recent), loop.time()
        if now - self.level_changed < 1.5:
            return
        if self.lat_ewma > floor + 70 and self.level < 3:
            self.level += 1
        elif (
            self.lat_ewma < floor + 25
            and self.level > 0
            and now - self.level_changed > 4
        ):
            self.level -= 1
        else:
            return
        self.level_changed = now
        self.lat_recent.clear()
        self.lat_ewma = None

    # ---- messages from the page
    def send_meta(self):
        sc = self.scene
        self.conn.send_json(
            dict(
                t="meta",
                v=self.vid,
                labels=sc.labels(),
                fov=sc.camera.fov,
                n_tris=sc.n_triangles(),
                version=sc.version,
            )
        )

    def scene_changed(self):
        self.send_meta()
        self.dirty = True
        self.wake.set()

    def handle(self, msg):
        t = msg["t"]
        self.last_msg = time.monotonic()
        if self.paused:  # the page is alive again (a frozen page was thawed): resume with a fresh keyframe
            self.paused, self.need_key, self.dirty = False, True, True
            self.wake.set()
        loop = asyncio.get_running_loop()
        if t == "scene":
            scene = self.hub.scenes.get(msg["id"])
            if scene is None:
                self.conn.send_json(
                    dict(t="error", v=self.vid, message=f"unknown scene {msg['id']}")
                )
                return
            self.scene = scene
            self.send_meta()
        elif self.scene is None:
            return
        elif t == "cfg":
            self.codec = msg.get("codec", self.codec)
            self.quality = int(
                msg.get("quality") or (80 if self.codec == "jpeg" else 26)
            )
            self.bw, self.rtt = float(msg.get("bw", 0)), float(msg.get("rtt", 0))
            self.window = int(msg.get("win") or MAX_INFLIGHT)
            self.adapt = bool(msg.get("adapt", True))
            self.level = 0
            self.need_key = True
        elif t == "resize":
            self.w, self.h = (
                max(16, min(3840, int(msg["w"]) // 2 * 2)),
                max(16, min(2160, int(msg["h"]) // 2 * 2)),
            )
        elif t == "drag":
            self.scene.camera.orbit(msg["dx"], msg["dy"])
        elif t == "pan":
            self.scene.camera.pan(msg["dx"], msg["dy"], self.h)
        elif t == "wheel":
            self.scene.camera.zoom(msg["dy"])
        elif t == "pose":
            self.scene.camera.set_pose(**msg["pose"])
        elif t == "reset":
            cam = self.scene.camera
            cam.azimuth, cam.elevation = 0.0, 0.0
            self.scene.fit_camera()
        elif t == "pick":
            camera, scene, w, h = self.scene.camera.copy(), self.scene, self.w, self.h

            async def do_pick(req=msg.get("req")):
                hit = await self.hub.worker.run(
                    lambda r: r.pick(
                        scene, camera, w, h, float(msg["x"]), float(msg["y"])
                    )
                )
                self.conn.send_json(dict(t="picked", v=self.vid, req=req, hit=hit))

            loop.create_task(do_pick())
            return
        elif t == "key":
            self.need_key = True
        elif t == "visible":
            self.visible = bool(msg["on"])
            if self.visible:
                self.need_key, self.dirty = (
                    True,
                    True,
                )  # a fresh encoder starts with a keyframe of the current pose
                self.wake.set()
            else:
                if self.still_timer:
                    self.still_timer.cancel()
                self.dirty = self.want_still = False
                loop.create_task(self.hub.worker.run(lambda r: self.drop_gpu(r)))
            return
        elif t == "ack":
            self.inflight = max(0, self.inflight - 1)
            self.note_ack()
            self.wake.set()
            return
        elif t == "getpose":
            self.conn.send_json(
                {"t": "pose", "v": self.vid, "pose": self.scene.camera.get_pose()}
            )
            return
        if "ts" in msg:
            self.last_input_ts = msg["ts"]
        self.dirty, self.want_still = True, False
        self.wake.set()
        if self.still_timer:
            self.still_timer.cancel()
        self.still_timer = loop.call_later(
            STILL_AFTER + self.rtt / 2000, self._still_due
        )

    def _still_due(self):
        self.want_still = True
        self.wake.set()

    def check_stalled(self, after=5.0):
        """Frames in flight but nothing heard for `after` seconds: the page is frozen or gone. Stop streaming to it."""
        if (
            not self.paused
            and self.inflight >= self.window
            and time.monotonic() - self.last_msg > after
        ):
            self.paused, self.inflight = True, 0
            self.sent.clear()
            asyncio.get_running_loop().create_task(
                self.hub.worker.run(lambda r: self.drop_gpu(r))
            )

    # ---- everything below runs on the GL thread
    def jpeg_quality(self, quality):
        """JPEG quality for this frame. A viewer that asked for video but gets JPEG follows the same ladder from quality 80."""
        return (
            quality
            if self.codec == "jpeg" and self.quality > 30
            else max(15, int(80 * (1 - 0.3 * self.level)))
        )

    def use_h264(self, w, h, quality):
        """Make (or reuse) the video encoder. False means: send JPEG instead (no PyAV, or this ffmpeg lacks the encoder)."""
        from .codecs import H264Encoder, have_h264

        if not have_h264():
            self.hub.note_fallback(
                "PyAV is not installed (pip install av): streaming JPEG instead of H.264"
            )
            return False
        backend = "h264_nvenc" if self.codec == "nvenc" else "libx264"
        key = (backend, w, h, quality)
        if self.encoder_key != key:
            try:
                self.encoder, self.encoder_key, self.need_key = (
                    H264Encoder(w, h, backend, quality=quality),
                    key,
                    True,
                )
            except Exception as error:  # noqa: BLE001 - e.g. an ffmpeg build without libx264
                self.hub.note_fallback(
                    f"the {backend} encoder is unavailable ({error}): streaming JPEG instead"
                )
                self.codec = "jpeg"
                return False
        return True

    def drop_gpu(self, renderer):
        self.encoder, self.encoder_key = None, None
        if self.held is not None:
            renderer.release(self.held)
            self.held = None
        gc.collect()

    def frame(self, renderer, camera, w, h, still, quality):
        t0 = time.perf_counter()
        if self.held is not self.scene:
            if self.held is not None:
                renderer.release(self.held)
            renderer.acquire(self.scene)
            self.held = self.scene
        rgb, std = renderer.render(self.scene, camera, w, h)
        t1 = time.perf_counter()
        if still:
            payload, kind = self.hub.jpeg().encode(rgb, quality=95)[0], STILL
        elif self.codec == "jpeg" or not self.use_h264(w, h, quality):
            payload, kind = (
                self.hub.jpeg().encode(rgb, quality=self.jpeg_quality(quality))[0],
                JPEG,
            )
        else:
            payload, is_key = self.encoder.encode(rgb, keyframe=self.need_key)
            kind, self.need_key = (KEY if is_key else DELTA), False
        t2 = time.perf_counter()
        return kind, payload, 1000 * (t1 - t0), 1000 * (t2 - t1), std

    async def close(self):
        """Release timers, the encoder (its threads) and the scene's GPU buffers. Safe to call twice."""
        self.closed = True
        if self.still_timer:
            self.still_timer.cancel()
        self.wake.set()
        await self.hub.worker.run(lambda r: self.drop_gpu(r))

    async def pump(self):
        loop = asyncio.get_running_loop()
        worker = self.hub.worker
        while not self.closed:
            await self.wake.wait()
            self.wake.clear()
            if (
                self.closed
                or self.scene is None
                or self.inflight >= self.window
                or not self.visible
                or self.paused
                or not (self.dirty or self.want_still)
            ):
                continue
            still = self.want_still and not self.dirty
            self.dirty = self.want_still = False
            scene = self.scene
            scene.ensure_camera()
            camera, echo, w, h = scene.camera.copy(), self.last_input_ts, self.w, self.h
            try:
                kind, payload, t_render, t_encode, std = await worker.run(
                    lambda r: self.frame(
                        r, camera, w, h, still, self.effective_quality()
                    )
                )
            except Exception as error:  # noqa: BLE001 - tell the page, keep the server alive
                self.conn.send_json(
                    dict(
                        t="error",
                        v=self.vid,
                        message=f"{type(error).__name__}: {error}",
                    )
                )
                continue
            if not payload:  # the encoder is buffering (NVENC start-up)
                self.dirty = True
                self.wake.set()
                continue
            self.seq += 1
            self.inflight += 1
            data = (
                HEADER.pack(
                    kind, self.vid, self.seq, echo, t_render, t_encode, *std.reshape(-1)
                )
                + payload
            )
            now = loop.time()
            self.sent.append((now, len(data)))
            if self.bw > 0:  # emulated link, for testing slow connections
                self.link_free = max(now, self.link_free) + len(data) * 8 / (
                    self.bw * 1e6
                )
                due = self.link_free + self.rtt / 2000
            else:
                due = now + self.rtt / 2000
            loop.call_later(max(0, due - now), self.conn.send_bytes, data)
            self.wake.set()


class Connection:
    """One websocket, many viewers. All writes go through one queue so frames never interleave mid-message."""

    def __init__(self, ws, hub):
        self.ws, self.hub = ws, hub
        self.sessions = {}
        self.last_rx = time.monotonic()
        self.out = asyncio.Queue()
        self.sender = asyncio.ensure_future(self._send_loop())

    async def _send_loop(self):
        while True:
            item = await self.out.get()
            if self.ws.closed:
                continue
            try:
                await (
                    self.ws.send_bytes(item)
                    if isinstance(item, bytes)
                    else self.ws.send_json(item)
                )
            except ConnectionError:
                pass

    def send_json(self, obj):
        self.out.put_nowait(obj)

    def send_bytes(self, data):
        self.out.put_nowait(data)

    def route(self, msg):
        t = msg["t"]
        if t == "ping":
            self.send_json({"t": "pong", "ts": msg.get("ts")})
            return
        vid = msg.get("v", 0)
        if t == "open":
            if vid not in self.sessions:
                sess = self.sessions[vid] = Session(self, vid, self.hub)
                self.hub.sessions.add(sess)
                asyncio.ensure_future(sess.pump())
            return
        if t == "close":
            sess = self.sessions.pop(vid, None)
            if sess:
                self.hub.sessions.discard(sess)
                asyncio.ensure_future(sess.close())
            return
        if vid in self.sessions:
            self.sessions[vid].handle(msg)

    async def close(self):
        self.sender.cancel()
        sessions, self.sessions = list(self.sessions.values()), {}
        for sess in sessions:
            self.hub.sessions.discard(sess)
            await sess.close()


class Hub:
    """Owns the scenes registry, the web server thread and (through get_worker) the GL thread."""

    def __init__(self, host="127.0.0.1", port=0, public_url=None, stale=90.0):
        self.host, self.port, self.public_url, self.stale = (
            host,
            port,
            public_url,
            stale,
        )
        self.token = secrets.token_urlsafe(16)
        self.scenes = weakref.WeakValueDictionary()
        self.sessions = set()
        self.clients = set()
        self.loop = None
        self.thread = None
        self._jpeg = None
        self._ready = threading.Event()
        self._error = None
        self._noted = set()

    @property
    def worker(self):
        from .worker import get_worker

        return get_worker()

    def jpeg(self):
        from .codecs import JpegEncoder

        if self._jpeg is None:
            self._jpeg = JpegEncoder()
        return self._jpeg

    def note_fallback(self, message):
        """Say once, in Python, that a viewer is not getting the best codec (the picture still arrives)."""
        if message not in self._noted:
            self._noted.add(message)
            warnings.warn(message, stacklevel=2)

    # ---- scenes
    def register(self, scene):
        if scene.id not in self.scenes:
            self.scenes[scene.id] = scene
            scene.listeners.append(self._scene_changed)

    def _scene_changed(self, scene):
        """Called from whatever thread changed the scene: wake its viewers on the event loop."""
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._notify, scene.id)

    def _notify(self, scene_id):
        for sess in list(self.sessions):
            if sess.scene is not None and sess.scene.id == scene_id and not sess.closed:
                sess.scene_changed()

    def n_viewers(self, scene=None):
        return sum(
            1
            for s in list(self.sessions)
            if scene is None or (s.scene is not None and s.scene.id == scene.id)
        )

    # ---- server thread
    @property
    def url(self):
        self.start()
        return (
            self.public_url
            or f"http://{self.host if self.host != '0.0.0.0' else '127.0.0.1'}:{self.port}"
        )

    def view_url(self, scene):
        return f"{self.url}/view?scene={scene.id}&token={self.token}"

    def start(self):
        if self.thread is not None:
            self._ready.wait()
            if self._error:
                raise self._error
            return
        self.thread = threading.Thread(
            target=self._run, name="menpo3d-hub", daemon=True
        )
        self.thread.start()
        self._ready.wait()
        if self._error:
            self.thread = None
            raise self._error

    def _run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._serve())
        except Exception as error:  # noqa: BLE001
            self._error = error
            self._ready.set()
            return
        self.loop.run_forever()

    async def _serve(self):
        from aiohttp import web

        app = web.Application()
        app.add_routes(
            [
                web.get("/viewer.js", self._static("viewer.js", "text/javascript")),
                web.get("/view", self._guard(self._view_page)),
                web.get("/ws", self._guard(self._ws)),
                web.get("/snapshot.png", self._guard(self._snapshot)),
                web.get("/stats", self._guard(self._stats)),
            ]
        )
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        self._runner = runner
        self._ready.set()

    def stop(self):
        if self.loop is not None:
            fut = asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self.loop)
            fut.result(5)
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread = None

    # ---- routes
    def _guard(self, handler):
        from aiohttp import web

        async def wrapped(request):
            if not secrets.compare_digest(request.query.get("token", ""), self.token):
                raise web.HTTPForbidden(text="missing or wrong token")
            return await handler(request)

        return wrapped

    def _static(self, name, content_type):
        from aiohttp import web

        async def handler(request):
            # modules loaded from another origin (a notebook page) need CORS; the file is public code, not data
            return web.Response(
                body=(STATIC / name).read_bytes(),
                content_type=content_type,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Cache-Control": "no-store",
                },
            )

        return handler

    async def _view_page(self, request):
        from aiohttp import web

        return web.Response(
            body=(STATIC / "view.html").read_bytes(),
            content_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    async def _stats(self, request):
        from aiohttp import web

        return web.json_response(
            dict(
                clients=len(self.clients),
                viewers=self.n_viewers(),
                scenes=len(self.scenes),
                gpu=await self.worker.run(lambda r: r.on_gpu()),
                fbos=await self.worker.run(lambda r: len(r._fbos)),
            )
        )

    async def _snapshot(self, request):
        from aiohttp import web
        from PIL import Image

        scene = self.scenes.get(request.query.get("scene", ""))
        if scene is None:
            raise web.HTTPNotFound(text="no such scene")
        w, h = (
            int(request.query.get("w", 640)) // 2 * 2,
            int(request.query.get("h", 480)) // 2 * 2,
        )
        scene.ensure_camera()
        camera = scene.camera.copy()
        rgb = await self.worker.run(lambda r: r.render(scene, camera, w, h)[0].copy())
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "PNG")
        return web.Response(body=buf.getvalue(), content_type="image/png")

    async def _ws(self, request):
        from aiohttp import WSMsgType, web

        ws = web.WebSocketResponse(
            max_msg_size=0, heartbeat=10
        )  # protocol pings; a peer that doesn't answer is closed
        await ws.prepare(request)
        conn = Connection(ws, self)
        self.clients.add(conn)

        async def watchdog():
            while True:
                await asyncio.sleep(1)
                for sess in list(conn.sessions.values()):
                    sess.check_stalled()
                if (
                    time.monotonic() - conn.last_rx > self.stale
                ):  # live pages ping every 5 s; frozen/back-forward-cache pages go silent
                    await (
                        ws.close()
                    )  # a restored page reconnects and reopens its viewers
                    return

        dog = asyncio.ensure_future(watchdog())
        loop = asyncio.get_running_loop()
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                m = json.loads(msg.data)
                conn.last_rx = time.monotonic()
                sess = conn.sessions.get(m.get("v", 0))
                loop.call_later(sess.rtt / 2000 if sess else 0, conn.route, m)
        dog.cancel()
        await conn.close()
        self.clients.discard(conn)
        if not self.clients:  # nobody left: framebuffers are not needed either
            await self.worker.run(lambda r: r.release_framebuffers())
        return ws


_hub = None
_hub_lock = threading.Lock()
_config = {}


def configure(host=None, port=None, public_url=None):
    """Set how the hub listens before it starts, e.g. ``configure(host="0.0.0.0", public_url="https://box/proxy/8765")``
    for a remote machine reached through a port forward or jupyter-server-proxy."""
    if _hub is not None and _hub.thread is not None:
        raise RuntimeError(
            "the viewer server is already running; call configure() before the first viewer"
        )
    for key, value in dict(host=host, port=port, public_url=public_url).items():
        if value is not None:
            _config[key] = value


def get_hub():
    global _hub
    with _hub_lock:
        if _hub is None:
            _hub = Hub(**_config)
        return _hub
