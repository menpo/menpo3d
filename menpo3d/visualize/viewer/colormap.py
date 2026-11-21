"""Scalars to colours: the baseline for anything coloured by a value (heat maps, errors, curvature)."""

import numpy as np

NAN_COLOUR = (0.6, 0.6, 0.6)


def get_cmap(cmap):
    """A matplotlib colormap from its name (or the colormap itself)."""
    from matplotlib import colormaps

    if not isinstance(cmap, str):
        return cmap
    try:
        return colormaps[cmap]
    except KeyError:
        raise ValueError(f"unknown colormap {cmap!r}") from None


def scalars_to_colours(values, cmap="jet", vmin=None, vmax=None):
    """Per-value RGB (n, 3) float32 in 0..1, plus the range used. NaN values are grey; a constant array maps to the middle."""
    values = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(values)
    lo = float(values[finite].min()) if vmin is None and finite.any() else vmin
    hi = float(values[finite].max()) if vmax is None and finite.any() else vmax
    lo, hi = (0.0 if lo is None else lo), (1.0 if hi is None else hi)
    span = hi - lo
    t = np.full(len(values), 0.5) if span <= 0 else (values - lo) / span
    rgb = get_cmap(cmap)(np.clip(np.nan_to_num(t, nan=0.0), 0, 1))[:, :3]
    rgb[~finite] = NAN_COLOUR
    return rgb.astype("f4"), lo, hi


def statistics_lines(values, names=("mean", "std", "max")):
    """['mean 0.12', 'std 0.03', 'max 0.40'] over the finite values: the text that goes with a colour bar."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return []
    fns = dict(
        mean=np.mean,
        std=np.std,
        min=np.min,
        max=np.max,
        median=np.median,
        rms=lambda v: np.sqrt(np.mean(v**2)),
    )
    return [f"{n} {fns[n](values):.3g}" for n in names]
