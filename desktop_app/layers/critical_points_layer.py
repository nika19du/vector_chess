from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OpenGL import GL

from desktop_app.correspondence import CriticalPointCorrespondence
from desktop_app.gl_canvas import LayerGeometry
from desktop_app.layer_registry import LayerDefinition
from desktop_app.layers._color_scale import hex_to_rgba
from desktop_app.layers._lerp import lerp
from desktop_app.layers._ndc import plot_to_ndc
from desktop_app.position_cache import CacheEntry
from visualization.critical_points_plot import MARKER_SPECS

# Half-width, in plot-space units (one board cell == 1.0), of the small
# colored quad standing in for matplotlib's per-classification glyph
# (triangle/inverted-triangle/x/circle). Classification stays fully legible
# by color -- the load-bearing channel; exact glyph shape is decoration this
# phase doesn't build (see the plan's rendering-simplifications note).
MARKER_HALF_SIZE = 0.12


@dataclass(frozen=True)
class CriticalPointMarker:
    xy: tuple[float, float]  # plot space
    color: tuple[float, float, float, float]


@dataclass(frozen=True)
class CriticalPointsFrame:
    markers: list[CriticalPointMarker]


def build_critical_points_frame(entry: CacheEntry) -> CriticalPointsFrame:
    """
    Only quality-accepted points are drawn -- matches
    `visualization/critical_points_plot.py`'s own `show_only_accepted=True`
    default exactly (`assessment.is_accepted`, nothing else). Colors reuse
    `MARKER_SPECS`'s exact hex values; "degenerate"'s hollow-marker
    `facecolor: "none"` maps to its `edgecolor` instead, since this layer's
    markers are always filled quads.
    """
    if entry.analysis is None:
        raise ValueError("cache entry is not READY: no analysis")

    markers: list[CriticalPointMarker] = []
    for assessment in entry.analysis.critical_point_assessments:
        if not assessment.is_accepted:
            continue

        point = assessment.point
        color = _marker_color(point)
        if color is None:
            continue  # "unclassified" never appears in an accepted assessment

        markers.append(CriticalPointMarker(xy=(point.x, point.y), color=color))

    return CriticalPointsFrame(markers=markers)


def _marker_color(point) -> tuple[float, float, float, float] | None:
    spec = MARKER_SPECS.get(point.classification)
    if spec is None:
        return None
    color_hex = spec["facecolor"] if spec["facecolor"] != "none" else spec["edgecolor"]
    return hex_to_rgba(color_hex)


def interpolate_critical_points_frame(correspondence: CriticalPointCorrespondence, t: float) -> CriticalPointsFrame:
    """
    Move-to-move animation (docs/interactive_ui.md Part 6: "Critical Points
    | Position/confidence interpolate between matched pairs; unmatched
    points fade in/out"). Matched pairs are guaranteed the same
    classification (desktop_app/correspondence.py's own matching
    constraint), so their color never changes mid-transition, only position.

    Endpoint exactness: at `t<=0.0`, `appeared` points are omitted entirely
    (not merely drawn at alpha 0) and `disappeared` points are drawn at full
    alpha, so the marker set exactly reproduces
    `build_critical_points_frame(entry_a)`'s accepted set -- and
    symmetrically at `t>=1.0` for `entry_b`. Strictly between, both fade.
    """
    markers: list[CriticalPointMarker] = []

    for match in correspondence.matched:
        color = _marker_color(match.current)
        if color is None:
            continue
        x = lerp(match.previous.x, match.current.x, t)
        y = lerp(match.previous.y, match.current.y, t)
        markers.append(CriticalPointMarker(xy=(x, y), color=color))

    if t < 1.0:
        fade_out = lerp(1.0, 0.0, t)
        for point in correspondence.disappeared:
            base_color = _marker_color(point)
            if base_color is None:
                continue
            color = (base_color[0], base_color[1], base_color[2], base_color[3] * fade_out)
            markers.append(CriticalPointMarker(xy=(point.x, point.y), color=color))

    if t > 0.0:
        fade_in = lerp(0.0, 1.0, t)
        for point in correspondence.appeared:
            base_color = _marker_color(point)
            if base_color is None:
                continue
            color = (base_color[0], base_color[1], base_color[2], base_color[3] * fade_in)
            markers.append(CriticalPointMarker(xy=(point.x, point.y), color=color))

    return CriticalPointsFrame(markers=markers)


def render_critical_points_frame(frame: CriticalPointsFrame) -> list[LayerGeometry]:
    if not frame.markers:
        positions = np.zeros((0, 2), dtype=np.float32)
        colors = np.zeros((0, 4), dtype=np.float32)
        return [LayerGeometry(positions=positions, colors=colors, primitive=GL.GL_TRIANGLES)]

    plot_positions: list[tuple[float, float]] = []
    vertex_colors: list[tuple[float, float, float, float]] = []
    for marker in frame.markers:
        x, y = marker.xy
        half = MARKER_HALF_SIZE
        quad = (
            (x - half, y - half),
            (x + half, y - half),
            (x + half, y + half),
            (x - half, y - half),
            (x + half, y + half),
            (x - half, y + half),
        )
        plot_positions.extend(quad)
        vertex_colors.extend([marker.color] * 6)

    positions = plot_to_ndc(np.array(plot_positions, dtype=np.float32))
    colors = np.array(vertex_colors, dtype=np.float32)
    return [LayerGeometry(positions=positions, colors=colors, primitive=GL.GL_TRIANGLES)]


CRITICAL_POINTS_LAYER = LayerDefinition(
    id="critical_points",
    display_name="Critical Points",
    data_source=build_critical_points_frame,
    renderer=render_critical_points_frame,
)
