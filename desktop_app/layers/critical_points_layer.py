from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OpenGL import GL

from desktop_app.correspondence import CriticalPointCorrespondence
from desktop_app.gl_canvas import LayerGeometry
from desktop_app.layer_registry import LayerDefinition
from desktop_app.layers._color_scale import hex_to_rgba
from desktop_app.layers._critical_point_glyphs import (
    cross_line_segments,
    ring_line_segments,
    triangle_down_vertices,
    triangle_up_vertices,
)
from desktop_app.layers._lerp import lerp
from desktop_app.layers._ndc import plot_to_ndc
from desktop_app.position_cache import CacheEntry
from visualization.critical_points_plot import MARKER_LINEWIDTH, MARKER_SPECS

# Half-width (or radius, for the degenerate ring), in plot-space units (one
# board cell == 1.0), of each classification's glyph.
#
# Compared against the reference's actual rendered footprint rather than
# picked arbitrarily: critical_points_plot.py's figure is figsize=(10, 10)
# at matplotlib's default 100 DPI (no dpi= override anywhere in
# visualization/*.py) -- a 1000x1000px figure over an 8-cell plot domain,
# so roughly 100-125px/cell once axis margins are accounted for.
# MARKER_SIZE=220 is scatter's area in points^2, so the marker's linear
# size is sqrt(220) ~= 14.8pt * (100/72)px/pt ~= 20.6px -- about 17-21% of
# one cell. This value's full extent (2 * MARKER_HALF_SIZE = 0.24 cells,
# i.e. 24% of a cell) is the same order of magnitude, not an arbitrary
# choice -- kept unchanged from the pre-V3 quad marker's own size rather
# than enlarged, since the comparison shows the existing footprint is
# already visually equivalent.
MARKER_HALF_SIZE = 0.12


@dataclass(frozen=True)
class CriticalPointMarker:
    xy: tuple[float, float]  # plot space
    color: tuple[float, float, float, float]
    # V3: which glyph shape to draw -- one of MARKER_SPECS's four keys
    # ("maximum"/"minimum"/"saddle"/"degenerate"). Classification is never
    # inferred from color; it's carried through explicitly from
    # ClassifiedCriticalPoint.classification at every point this dataclass
    # is constructed, the same source of truth the color already comes from.
    classification: str


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

        markers.append(CriticalPointMarker(xy=(point.x, point.y), color=color, classification=point.classification))

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
        # Matched pairs are guaranteed the same classification
        # (desktop_app/correspondence.py's own matching constraint) -- using
        # match.current's here is consistent with color already doing the
        # same, not a new assumption.
        color = _marker_color(match.current)
        if color is None:
            continue
        x = lerp(match.previous.x, match.current.x, t)
        y = lerp(match.previous.y, match.current.y, t)
        markers.append(CriticalPointMarker(xy=(x, y), color=color, classification=match.current.classification))

    if t < 1.0:
        fade_out = lerp(1.0, 0.0, t)
        for point in correspondence.disappeared:
            base_color = _marker_color(point)
            if base_color is None:
                continue
            color = (base_color[0], base_color[1], base_color[2], base_color[3] * fade_out)
            markers.append(CriticalPointMarker(xy=(point.x, point.y), color=color, classification=point.classification))

    if t > 0.0:
        fade_in = lerp(0.0, 1.0, t)
        for point in correspondence.appeared:
            base_color = _marker_color(point)
            if base_color is None:
                continue
            color = (base_color[0], base_color[1], base_color[2], base_color[3] * fade_in)
            markers.append(CriticalPointMarker(xy=(point.x, point.y), color=color, classification=point.classification))

    return CriticalPointsFrame(markers=markers)


def render_critical_points_frame(frame: CriticalPointsFrame) -> list[LayerGeometry]:
    """
    Two geometries, two primitives -- the same "fill + line" pattern
    morse_smale_layer.py already established, not a new renderer mechanism:
    maximum/minimum are solid filled triangles (GL_TRIANGLES), saddle's X
    and degenerate's hollow ring are outline-only (GL_LINES), matching the
    reference's own filled-marker vs hollow-marker distinction
    (MARKER_SPECS's facecolor "none" for degenerate, and 'x' being an
    unfilled line marker in matplotlib too -- see UNFILLED_LINE_MARKERS in
    visualization/critical_points_plot.py).
    """
    triangle_positions: list[tuple[float, float]] = []
    triangle_colors: list[tuple[float, float, float, float]] = []
    line_positions: list[tuple[float, float]] = []
    line_colors: list[tuple[float, float, float, float]] = []

    for marker in frame.markers:
        x, y = marker.xy
        half = MARKER_HALF_SIZE

        if marker.classification == "maximum":
            triangle_positions.extend(triangle_up_vertices(x, y, half))
            triangle_colors.extend([marker.color] * 3)
        elif marker.classification == "minimum":
            triangle_positions.extend(triangle_down_vertices(x, y, half))
            triangle_colors.extend([marker.color] * 3)
        elif marker.classification == "saddle":
            segments = cross_line_segments(x, y, half)
            line_positions.extend(segments)
            line_colors.extend([marker.color] * len(segments))
        elif marker.classification == "degenerate":
            segments = ring_line_segments(x, y, half)
            line_positions.extend(segments)
            line_colors.extend([marker.color] * len(segments))

    if triangle_positions:
        triangle_geometry = LayerGeometry(
            positions=plot_to_ndc(np.array(triangle_positions, dtype=np.float32)),
            colors=np.array(triangle_colors, dtype=np.float32),
            primitive=GL.GL_TRIANGLES,
        )
    else:
        triangle_geometry = LayerGeometry(
            positions=np.zeros((0, 2), dtype=np.float32),
            colors=np.zeros((0, 4), dtype=np.float32),
            primitive=GL.GL_TRIANGLES,
        )

    if line_positions:
        line_geometry = LayerGeometry(
            positions=plot_to_ndc(np.array(line_positions, dtype=np.float32)),
            colors=np.array(line_colors, dtype=np.float32),
            primitive=GL.GL_LINES,
            line_width=MARKER_LINEWIDTH,
        )
    else:
        line_geometry = LayerGeometry(
            positions=np.zeros((0, 2), dtype=np.float32),
            colors=np.zeros((0, 4), dtype=np.float32),
            primitive=GL.GL_LINES,
            line_width=MARKER_LINEWIDTH,
        )

    return [triangle_geometry, line_geometry]


CRITICAL_POINTS_LAYER = LayerDefinition(
    id="critical_points",
    display_name="Critical Points",
    data_source=build_critical_points_frame,
    renderer=render_critical_points_frame,
)
