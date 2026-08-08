from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OpenGL import GL

from analysis.geometry import math_to_plot_coords
from desktop_app.gl_canvas import LayerGeometry
from desktop_app.layer_registry import LayerDefinition
from desktop_app.layers._color_scale import hex_to_rgba
from desktop_app.layers._ndc import plot_to_ndc
from desktop_app.position_cache import CacheEntry
from visualization.gradient_plot import (
    MAX_ARROW_LENGTH,
    MIN_ARROW_LENGTH,
    get_contrasting_arrow_color,
    should_draw_vector,
)

# The one geometric cap `draw_single_gradient_vector` applies that isn't a
# named module constant (it's an inline literal there) -- kept as one, here.
_MAX_ARROW_LENGTH_CAP = 0.58


@dataclass(frozen=True)
class GradientSegment:
    start_xy: tuple[float, float]  # plot space
    end_xy: tuple[float, float]
    color: tuple[float, float, float, float]


@dataclass(frozen=True)
class GradientFrame:
    segments: list[GradientSegment]


def build_gradient_frame(entry: CacheEntry) -> GradientFrame:
    """
    Reuses `should_draw_vector` (magnitude/relative-strength/on-piece
    filtering) and `get_contrasting_arrow_color` (background-dependent
    color) from `visualization/gradient_plot.py` directly, unchanged.
    `get_contrasting_arrow_color` only ever reads `analysis.
    attack_influence_field.matrix` -- `FullPositionAnalysis` carries that
    exact attribute path, so it is passed directly rather than constructed
    into an unrelated `MoveAnalysis` just to satisfy a type name.

    The arrow start/end geometry itself is reproduced from
    `draw_single_gradient_vector`'s inline arithmetic (not a separate
    importable function there) -- same formula, same constants, adapted to
    return plot-space endpoints instead of a matplotlib annotation.
    """
    if entry.analysis is None:
        raise ValueError("cache entry is not READY: no analysis")

    analysis = entry.analysis
    gradient_field = analysis.gradient_field
    max_magnitude = gradient_field.max_magnitude

    segments: list[GradientSegment] = []
    if max_magnitude <= 0:
        return GradientFrame(segments=segments)

    for vector in gradient_field.vectors:
        if vector.magnitude <= 0:
            continue
        if not should_draw_vector(board=analysis.board, vector=vector, max_magnitude=max_magnitude):
            continue

        plot_x, plot_y = math_to_plot_coords(vector.x, vector.y)
        center_x = plot_x + 0.5
        center_y = plot_y + 0.5

        unit_x = vector.delta_x / vector.magnitude
        unit_y = -(vector.delta_y / vector.magnitude)  # math Y grows up, plot Y grows down

        relative_strength = vector.magnitude / max_magnitude
        arrow_length = MIN_ARROW_LENGTH + relative_strength * (MAX_ARROW_LENGTH - MIN_ARROW_LENGTH)
        arrow_length = min(arrow_length, _MAX_ARROW_LENGTH_CAP)

        delta_x = unit_x * arrow_length
        delta_y = unit_y * arrow_length

        start_xy = (center_x - delta_x / 2, center_y - delta_y / 2)
        end_xy = (center_x + delta_x / 2, center_y + delta_y / 2)

        color_hex = get_contrasting_arrow_color(analysis=analysis, vector=vector)
        alpha = 0.72 + relative_strength * 0.25
        color = hex_to_rgba(color_hex, alpha=alpha)

        segments.append(GradientSegment(start_xy=start_xy, end_xy=end_xy, color=color))

    return GradientFrame(segments=segments)


def render_gradient_frame(frame: GradientFrame) -> list[LayerGeometry]:
    """No arrowhead geometry -- a plain line segment per vector (direction,
    length, and contrast color are preserved; the arrowhead is decoration)."""
    if not frame.segments:
        positions = np.zeros((0, 2), dtype=np.float32)
        colors = np.zeros((0, 4), dtype=np.float32)
    else:
        plot_positions = np.array(
            [xy for segment in frame.segments for xy in (segment.start_xy, segment.end_xy)],
            dtype=np.float32,
        )
        positions = plot_to_ndc(plot_positions)
        colors = np.array(
            [segment.color for segment in frame.segments for _ in range(2)],
            dtype=np.float32,
        )

    return [LayerGeometry(positions=positions, colors=colors, primitive=GL.GL_LINES)]


GRADIENT_LAYER = LayerDefinition(
    id="gradient",
    display_name="Gradient",
    data_source=build_gradient_frame,
    renderer=render_gradient_frame,
)
