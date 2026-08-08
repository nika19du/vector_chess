from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import numpy as np
from matplotlib.colors import Normalize

from desktop_app.layer_registry import LayerDefinition
from desktop_app.layers._color_scale import symmetric_max_abs
from desktop_app.layers._lerp import lerp_array
from desktop_app.position_cache import CacheEntry

GRID_SIZE = 8
VERTICES_PER_QUAD = 6

# Matches visualization/attack_influence_plot.py::draw_attack_influence_overlay
# exactly (imshow(matrix, cmap="RdBu_r", norm=Normalize(-m, m), alpha=0.72)),
# so the interactive canvas and the existing static matplotlib plot render the
# identical color for the identical position -- no colormap logic is
# reimplemented, only reused (matplotlib is already a pinned dependency).
COLORMAP_NAME = "RdBu_r"
OVERLAY_ALPHA = 0.72


@dataclass(frozen=True)
class GridColorFrame:
    """An 8x8 RGBA grid, row 0 = rank 8, column 0 = file a (analysis/geometry.py)."""

    colors: np.ndarray


def colorize_matrix(matrix: np.ndarray) -> GridColorFrame:
    """
    The colorization half of `build_attack_influence_frame`, extracted so
    `interpolate_attack_influence_frame` (this milestone's animation) can
    run the exact same colormap/normalization pipeline over an
    already-interpolated matrix, instead of a fresh per-position one. Not a
    behavior change for `build_attack_influence_frame` below, which now
    just supplies the matrix this expects.
    """
    max_absolute_value = symmetric_max_abs(matrix)
    normalization = Normalize(vmin=-max_absolute_value, vmax=max_absolute_value)
    colormap = matplotlib.colormaps[COLORMAP_NAME]

    colors = colormap(normalization(matrix)).astype(np.float32)
    colors[..., 3] *= OVERLAY_ALPHA
    return GridColorFrame(colors=colors)


def build_attack_influence_frame(entry: CacheEntry) -> GridColorFrame:
    if entry.analysis is None:
        raise ValueError("cache entry is not READY: no analysis")

    matrix = np.array(entry.analysis.attack_influence_field.matrix, dtype=float)
    return colorize_matrix(matrix)


def interpolate_attack_influence_frame(entry_a: CacheEntry, entry_b: CacheEntry, t: float) -> GridColorFrame:
    """
    Move-to-move animation (docs/interactive_ui.md Part 6: "Surface | Linear
    interpolation of the cached z grid between two positions"). Both grids
    share the exact same 8x8 shape and domain regardless of position, so a
    plain elementwise lerp is valid -- no resampling/correspondence needed,
    unlike the discrete-object layers. `lerp_array`'s exact-endpoint
    branching guarantees `t=0`/`t=1` reproduce `build_attack_influence_frame`
    on `entry_a`/`entry_b` exactly, not merely approximately.
    """
    if entry_a.analysis is None or entry_b.analysis is None:
        raise ValueError("both cache entries must be READY to interpolate")

    matrix_a = np.array(entry_a.analysis.attack_influence_field.matrix, dtype=float)
    matrix_b = np.array(entry_b.analysis.attack_influence_field.matrix, dtype=float)
    interpolated = lerp_array(matrix_a, matrix_b, t)
    return colorize_matrix(interpolated)


def render_attack_influence_frame(frame: GridColorFrame) -> np.ndarray:
    """
    Expands the 8x8 RGBA grid into per-vertex colors for the canvas's static
    64-quad VBO layout (row-major, matching
    desktop_app.gl_canvas.build_grid_positions's vertex order exactly).
    """
    flat = np.asarray(frame.colors, dtype=np.float32).reshape(GRID_SIZE * GRID_SIZE, 4)
    return np.repeat(flat, VERTICES_PER_QUAD, axis=0)


ATTACK_INFLUENCE_LAYER = LayerDefinition(
    id="attack_influence",
    display_name="Attack Influence",
    data_source=build_attack_influence_frame,
    renderer=render_attack_influence_frame,
)
