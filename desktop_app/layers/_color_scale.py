from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib.colors import Normalize


def symmetric_max_abs(matrix: np.ndarray) -> float:
    """
    Shared by every layer that colors a signed field on a symmetric
    White/Black scale (Attack Influence, Equipotential) -- matches
    `visualization/attack_influence_plot.py`/`equipotential_plot.py`'s own
    `max(1.0, float(np.max(np.abs(matrix))))` exactly, extracted once instead
    of duplicated per layer.
    """
    return max(1.0, float(np.max(np.abs(matrix))))


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Shared by every layer reusing a `visualization/*_plot.py` hex color constant as GL RGBA."""
    hex_color = hex_color.lstrip("#")
    red = int(hex_color[0:2], 16) / 255.0
    green = int(hex_color[2:4], 16) / 255.0
    blue = int(hex_color[4:6], 16) / 255.0
    return (red, green, blue, alpha)


def colorize_grid(matrix: np.ndarray, colormap_name: str, alpha: float) -> np.ndarray:
    """
    Symmetric-normalize -> named colormap -> alpha-scale pipeline for an 8x8
    signed scalar grid (Milestone F, Source Potential). Matches
    `attack_influence_layer.py`'s own inline `colorize_matrix` math exactly
    (`Normalize(vmin=-symmetric_max_abs(matrix), vmax=+that)`, then an alpha
    multiply), generalized to a caller-supplied colormap name/alpha so a
    second grid-colored field layer doesn't need to duplicate it.
    `attack_influence_layer.py` itself is left calling its own inline version
    unchanged -- not routed through this helper -- so this addition carries
    zero risk to Attack Influence's tested behavior.

    Each caller's own `symmetric_max_abs(matrix)` call happens independently
    here, per `matrix` -- two different fields (e.g. Attack Influence and
    Source Potential) never share a color scale, matching
    `visualization/source_potential_plot.py`'s own documented requirement
    that each panel use "собствена, независима нормализация на цвета."
    """
    max_absolute_value = symmetric_max_abs(matrix)
    normalization = Normalize(vmin=-max_absolute_value, vmax=max_absolute_value)
    colormap = matplotlib.colormaps[colormap_name]

    colors = colormap(normalization(matrix)).astype(np.float32)
    colors[..., 3] *= alpha
    return colors
