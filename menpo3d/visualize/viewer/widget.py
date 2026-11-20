"""The notebook widget: an anywidget that embeds viewer.js. Every widget on a page shares one websocket to the hub.

The browser also sends a lossless still once the camera stops; it is mirrored into an ordinary image output (created with
a display id at display time) so the picture stays in the saved notebook after the kernel is gone. While the widget is
live that picture is hidden.
"""

import base64
import weakref

import anywidget
import traitlets
from IPython.display import Image, display

from .hub import get_hub

_ESM = """
export default {
  async render({model, el}) {
    const url = model.get('url');
    const {Viewer} = await import(`${url}/viewer.js`);   // one module instance per page: widgets share one websocket per server
    const viewer = new Viewer(el, {url, token: model.get('token'), scene: model.get('scene'), w: model.get('width'), h: model.get('height')});
    const hidden = new Map();
    const hideStill = () => {   // the saved still below the widget is for dead kernels: hide it while the viewer is live
      const child = el.closest('.jp-OutputArea-child');
      if (!child) return;
      for (const n of child.parentElement.children)
        if (n !== child && n.querySelector('img') && !hidden.has(n)) { hidden.set(n, n.style.display); n.style.display = 'none'; }
    };
    setTimeout(hideStill, 0); setTimeout(hideStill, 500);
    viewer.onstillframe = png => { model.set('still', png); model.save_changes(); setTimeout(hideStill, 100); };
    return () => { hidden.forEach((d, n) => { n.style.display = d; }); viewer.destroy(); };
  }
};
"""

_BLANK = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


class FigureWidget(anywidget.AnyWidget):
    _esm = _ESM
    url = traitlets.Unicode().tag(sync=True)
    token = traitlets.Unicode().tag(sync=True)
    scene = traitlets.Unicode().tag(sync=True)
    width = traitlets.Int(640).tag(sync=True)
    height = traitlets.Int(480).tag(sync=True)
    still = traitlets.Unicode("").tag(
        sync=True
    )  # data URL of the last lossless frame, written by the browser

    def __init__(self, figure, **kw):
        hub = get_hub()
        hub.register(figure.scene)
        super().__init__(
            url=hub.url,
            token=hub.token,
            scene=figure.scene.id,
            width=figure.size[0],
            height=figure.size[1],
            **kw,
        )
        self._figure = weakref.ref(figure)
        self._handle = None
        self.observe(self._save_still, "still")

    def _repr_mimebundle_(self, **kwargs):
        """Called when displayed, inside the cell's execution: emit a placeholder image with a display id. A later
        update_display (from a widget message, outside any cell execution) still finds it, and it is saved in the .ipynb."""
        figure = self._figure()
        if self._handle is None and (figure is None or figure.persist_still):
            self._handle = display(Image(data=_BLANK, format="png"), display_id=True)
        return super()._repr_mimebundle_(**kwargs)

    def _save_still(self, change):
        if self._handle is not None and change["new"]:
            self._handle.update(
                Image(
                    data=base64.b64decode(change["new"].split(",", 1)[1]), format="png"
                )
            )


def figure_widget(figure):
    """One widget per figure, so re-displaying a figure (the menpowidgets pattern) keeps its viewer and camera."""
    widget = getattr(figure, "_widget", None)
    if widget is None:
        widget = figure._widget = FigureWidget(figure)
    return widget
