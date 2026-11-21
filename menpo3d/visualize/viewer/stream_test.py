"""menpo3d's scene, camera and renderer driven through scene-streamer's hub, the way a browser drives it."""

import asyncio
import json
from io import BytesIO

import numpy as np
import pytest

pytest.importorskip("scene_streamer")
import aiohttp  # noqa: E402
import scene_streamer  # noqa: E402
from PIL import Image  # noqa: E402
from scene_streamer import protocols  # noqa: E402
from scene_streamer.hub import HEADER, JPEG, STILL  # noqa: E402

import menpo3d.io as m3io  # noqa: E402
from menpo3d.visualize.viewer.scene import Scene  # noqa: E402
from menpo3d.visualize.viewer.worker import get_worker  # noqa: E402


@pytest.fixture(scope="module")
def hub():
    try:
        worker = get_worker()
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"no OpenGL: {error}")
    h = scene_streamer.Hub(worker=lambda: worker)
    h.start()
    yield h
    h.stop()


@pytest.fixture
def scene(hub):
    bunny = m3io.import_builtin_asset.bunny_obj()
    s = Scene()
    s.add_mesh(bunny.points, bunny.trilist, colour=(1, 0, 0))
    s.add_labels(bunny.points[:2], "ab")
    hub.register(s)
    return s


def test_the_scene_the_camera_and_the_renderer_satisfy_the_protocols(scene):
    from menpo3d.visualize.viewer.gl import GLRenderer

    assert isinstance(scene, protocols.Scene)
    assert isinstance(scene.camera, protocols.Camera)
    assert isinstance(get_worker(), protocols.Worker)
    assert issubclass(GLRenderer, protocols.Renderer)


def test_a_page_gets_real_frames_and_picks_a_triangle(hub, scene):
    async def go():
        got = dict(frames=[], json=[])
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                f"{hub.url}/ws?token={hub.token}", max_msg_size=0
            ) as ws:
                for msg in (
                    dict(t="open", v=1),
                    dict(t="scene", v=1, id=scene.id),
                    dict(t="cfg", v=1, codec="jpeg", adapt=False),
                    dict(t="resize", v=1, w=160, h=120),
                    dict(t="drag", v=1, dx=1, dy=0, ts=1),
                ):
                    await ws.send_str(json.dumps(msg))
                end = asyncio.get_event_loop().time() + 1.5
                asked = False
                while asyncio.get_event_loop().time() < end:
                    try:
                        m = await asyncio.wait_for(ws.receive(), 0.1)
                    except asyncio.TimeoutError:
                        continue
                    if m.type == aiohttp.WSMsgType.BINARY:
                        got["frames"].append(m.data)
                        await ws.send_str(json.dumps(dict(t="ack", v=1)))
                        if not asked:
                            asked = True
                            for req, x, y in ((7, 80.0, 60.0), (8, 1.0, 1.0)):
                                await ws.send_str(
                                    json.dumps(dict(t="pick", v=1, x=x, y=y, req=req))
                                )
                    elif m.type == aiohttp.WSMsgType.TEXT:
                        got["json"].append(json.loads(m.data))
            async with session.get(f"{hub.url}/stats?token={hub.token}") as r:
                got["stats"] = await r.json()
        return got

    got = asyncio.run(go())
    kinds = [HEADER.unpack_from(f)[0] for f in got["frames"]]
    assert kinds and set(kinds) <= {JPEG, STILL}
    image = Image.open(BytesIO(got["frames"][-1][HEADER.size :]))
    assert image.size == (160, 120)
    assert np.asarray(image).std() > 5  # the bunny, not a blank frame
    meta = [m for m in got["json"] if m["t"] == "meta"][0]
    assert meta["n_tris"] == scene.n_triangles() and len(meta["labels"]) == 2
    hits = {m["req"]: m["hit"] for m in got["json"] if m["t"] == "picked"}
    assert hits[7]["tri"] >= 0 and hits[8] is None
    assert set(got["stats"]) >= {"gpu", "fbos", "clients", "viewers", "scenes"}
