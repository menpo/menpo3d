"""Screen-space overlays (a colour bar, text such as statistics), drawn into the rendered frame.

They are part of the picture, so the live stream, the saved still, the native window and ``save_figure`` all show the same.
"""

import numpy as np

_FONT = None


def _font(px):
    global _FONT
    from matplotlib import font_manager
    from PIL import ImageFont

    if _FONT is None:
        _FONT = font_manager.findfont("DejaVu Sans")
    return ImageFont.truetype(_FONT, max(8, int(px)))


def _ticks(vmin, vmax, n):
    return [vmin + (vmax - vmin) * k / (n - 1) for k in range(n)]


def _text_size(draw, text, font):
    left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _draw_text(draw, item, w, h):
    font = _font(item["size"] * h)
    tw, th = _text_size(draw, item["text"], font)
    pad = int(0.012 * h)
    where = item["position"]
    if isinstance(where, str):
        top, left = where.startswith("top"), where.endswith("left")
        x = pad if left else w - tw - pad
        y = pad if top else h - th - pad
    else:
        x, y = where[0] * w, where[1] * h
    if item["background"] is not None:
        draw.rectangle(
            (x - pad / 2, y - pad / 2, x + tw + pad / 2, y + th + pad / 2),
            fill=tuple(int(255 * c) for c in item["background"]),
        )
    draw.multiline_text(
        (x, y),
        item["text"],
        fill=tuple(int(255 * c) for c in item["colour"]),
        font=font,
    )


def _draw_colorbar(image, draw, item, w, h):
    from PIL import Image

    font = _font(item["label_size"] * h)
    vertical = item["position"] in ("right", "left")
    thick, length = int(item["thickness"] * h), int(item["length"] * h)
    pad = int(0.03 * h)
    ticks = _ticks(item["vmin"], item["vmax"], item["n_ticks"])
    labels = [f"{t:.3g}" for t in ticks]
    label_w = max(_text_size(draw, s, font)[0] for s in labels)
    title_h = _text_size(draw, item["title"], font)[1] + 4 if item["title"] else 0
    if vertical:
        x0 = w - pad - thick - label_w - 6 if item["position"] == "right" else pad
        y0 = (h - length) // 2 + title_h // 2
        ramp = item["lut"][np.linspace(255, 0, length).astype(int)][:, None, :]
        ramp = np.repeat(ramp, thick, axis=1)
        image.paste(Image.fromarray(ramp), (x0, y0))
        box = (x0, y0, x0 + thick - 1, y0 + length - 1)
    else:
        x0 = (w - length) // 2
        y0 = (
            h - pad - thick - int(item["label_size"] * h * 1.6)
            if item["position"] == "bottom"
            else pad + title_h
        )
        ramp = item["lut"][np.linspace(0, 255, length).astype(int)][None, :, :]
        ramp = np.repeat(ramp, thick, axis=0)
        image.paste(Image.fromarray(ramp), (x0, y0))
        box = (x0, y0, x0 + length - 1, y0 + thick - 1)
    colour = tuple(int(255 * c) for c in item["colour"])
    draw.rectangle(box, outline=colour)
    for t, s in zip(ticks, labels):
        f = (
            (t - item["vmin"]) / (item["vmax"] - item["vmin"])
            if item["vmax"] != item["vmin"]
            else 0.5
        )
        tw, th = _text_size(draw, s, font)
        if vertical:
            y = y0 + (1 - f) * (length - 1)
            tx = x0 + thick + 4 if item["position"] == "right" else x0 + thick + 4
            draw.line((x0 + thick - 1, y, x0 + thick + 2, y), fill=colour)
            draw.text((tx, y - th / 2 - 1), s, fill=colour, font=font)
        else:
            x = x0 + f * (length - 1)
            ty = y0 + thick + 3
            draw.line((x, y0 + thick - 1, x, y0 + thick + 2), fill=colour)
            draw.text((x - tw / 2, ty), s, fill=colour, font=font)
    if item["title"]:
        tw, _ = _text_size(draw, item["title"], font)
        tx = box[0] if vertical else (w - tw) / 2
        ty = y0 - title_h if vertical else y0 - title_h
        draw.text(
            (max(2, min(tx, w - tw - 2)), ty), item["title"], fill=colour, font=font
        )


def draw_overlays(rgb, items):
    """A copy of ``rgb`` (h, w, 3 uint8) with the overlay items drawn on it; ``rgb`` itself when there are none."""
    from PIL import Image, ImageDraw

    overlays = [i for i in items if i["kind"] in ("colorbar", "text")]
    if not overlays:
        return rgb
    h, w, _ = rgb.shape
    image = Image.fromarray(np.ascontiguousarray(rgb))
    draw = ImageDraw.Draw(image)
    for item in overlays:
        if item["kind"] == "colorbar":
            _draw_colorbar(image, draw, item, w, h)
        else:
            _draw_text(draw, item, w, h)
    return np.asarray(image)
