from __future__ import annotations

import numpy as np

# Board extent in plot-space (analysis/geometry.py's convention: x in
# [0, 8] = file a..h, y in [0, 8] with y=0 = rank 8 at the top). Matches
# desktop_app.gl_canvas.build_grid_positions's own NDC mapping of the same
# 8x8 extent exactly, so plot-space geometry lines up with the board grid
# it's drawn over.
BOARD_EXTENT = 8.0


def plot_to_ndc(xy: np.ndarray) -> np.ndarray:
    """
    Converts an (N, 2) array of plot-space coordinates (the convention every
    `visualization/*_plot.py` module and `AttackInfluenceSurface`/critical
    points/ridge-valley/Morse-Smale data already use -- see
    analysis/geometry.py's module docstring) into normalized device
    coordinates for `desktop_app.gl_canvas`. Not a duplicate of
    analysis/geometry.py's transforms -- those convert between chess
    squares/matrix indices/plot space; this is the one further step from
    plot space to this specific GL canvas's NDC space.
    """
    xy = np.asarray(xy, dtype=np.float32)
    ndc_x = -1.0 + (xy[..., 0] / BOARD_EXTENT) * 2.0
    ndc_y = 1.0 - (xy[..., 1] / BOARD_EXTENT) * 2.0
    return np.stack([ndc_x, ndc_y], axis=-1).astype(np.float32)
