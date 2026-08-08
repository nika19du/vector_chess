from __future__ import annotations

import numpy as np


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
