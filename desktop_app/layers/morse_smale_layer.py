from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OpenGL import GL

from analysis.morse_smale import compute_cell_geometry
from desktop_app.correspondence import CellCorrespondence
from desktop_app.gl_canvas import LayerGeometry
from desktop_app.layer_registry import LayerDefinition
from desktop_app.layers._arc_length import resample_by_arc_length
from desktop_app.layers._color_scale import hex_to_rgba
from desktop_app.layers._lerp import lerp, lerp_point
from desktop_app.layers._ndc import plot_to_ndc
from desktop_app.position_cache import CacheEntry
from visualization.morse_smale_plot import (
    ACCEPTED_CELL_FILL_ALPHA,
    ACCEPTED_CELL_FILL_COLOR,
    ACCEPTED_CELL_LINE_COLOR,
)

MIN_POINTS_TO_DRAW = 2  # matches visualization/morse_smale_plot.py's own constant

# Boundary resample count before lerping a matched cell's polygon along
# normalized arc length (docs/interactive_ui.md Part 6) -- mirrors
# desktop_app/layers/ridge_valley_layer.py's CHAIN_RESAMPLE_COUNT exactly;
# kept as its own constant since the two layers' animation smoothness are
# independent rendering choices, not one shared mathematical value.
CELL_RESAMPLE_COUNT = 24


@dataclass(frozen=True)
class MorseSmaleCellGeometryFrame:
    """
    Each accepted, closed cell's polygon, already the actual traced boundary
    (not a straight-line approximation) via compute_cell_geometry, plus its
    centroid for fan triangulation.

    `fill_colors`/`line_colors` are per-cell (parallel to `polygons`/
    `centroids`), not one shared color for every cell -- needed so
    `interpolate_morse_smale_frame` (this milestone) can fade an individual
    birthing/dying cell's alpha independently of matched cells' constant
    alpha. `build_morse_smale_frame` below fills both lists with the same
    constant color per cell, so its own rendered output is unchanged.
    """

    polygons: list[list[tuple[float, float]]]
    centroids: list[tuple[float, float]]
    fill_colors: list[tuple[float, float, float, float]]
    line_colors: list[tuple[float, float, float, float]]


def build_morse_smale_frame(entry: CacheEntry) -> MorseSmaleCellGeometryFrame:
    """
    Only closed cells with a matched, accepted quality assessment are drawn
    -- matches `visualization/morse_smale_plot.py`'s own `show_only_accepted
    =True` default and its exact "no assessment found => not accepted"
    conservative rule (`assessment_by_cell_id.get(cell.cell_id)`, matched by
    `cell_id`, not list position). Draws cell fills only -- critical-point
    markers are the separately toggleable Critical Points layer's
    responsibility (see the plan's note on why Morse-Smale doesn't also
    draw them here, unlike the combined static plot).
    """
    if entry.analysis is None:
        raise ValueError("cache entry is not READY: no analysis")

    complex_ = entry.analysis.morse_smale_complex
    assessment_by_cell_id = {
        assessment.cell.cell_id: assessment for assessment in entry.analysis.cell_assessments
    }

    polygons: list[list[tuple[float, float]]] = []
    centroids: list[tuple[float, float]] = []
    fill_color = hex_to_rgba(ACCEPTED_CELL_FILL_COLOR, alpha=ACCEPTED_CELL_FILL_ALPHA)
    line_color = hex_to_rgba(ACCEPTED_CELL_LINE_COLOR, alpha=1.0)
    fill_colors: list[tuple[float, float, float, float]] = []
    line_colors: list[tuple[float, float, float, float]] = []

    for cell in complex_.cells:
        if not cell.is_closed:
            continue

        assessment = assessment_by_cell_id.get(cell.cell_id)
        is_accepted = assessment is not None and assessment.is_accepted
        if not is_accepted:
            continue

        geometry = compute_cell_geometry(cell)
        if geometry is None or len(geometry.polygon) < MIN_POINTS_TO_DRAW:
            continue

        polygons.append([(float(x), float(y)) for x, y in geometry.polygon])
        centroids.append((float(geometry.centroid[0]), float(geometry.centroid[1])))
        fill_colors.append(fill_color)
        line_colors.append(line_color)

    return MorseSmaleCellGeometryFrame(
        polygons=polygons,
        centroids=centroids,
        fill_colors=fill_colors,
        line_colors=line_colors,
    )


def render_morse_smale_frame(frame: MorseSmaleCellGeometryFrame) -> list[LayerGeometry]:
    """
    Two geometries, two primitives: a centroid-fan triangulation for the
    fill (correct for the star-convex basin shapes Morse-Smale cells are; a
    rendering-technique choice, not a math change -- `compute_cell_geometry`
    already computes the centroid this fans from) and a line loop for the
    boundary, reusing the exact same traced polygon points for both.
    """
    fill_positions: list[tuple[float, float]] = []
    fill_vertex_colors: list[tuple[float, float, float, float]] = []
    line_positions: list[tuple[float, float]] = []
    line_vertex_colors: list[tuple[float, float, float, float]] = []

    for polygon, centroid, fill_color, line_color in zip(
        frame.polygons, frame.centroids, frame.fill_colors, frame.line_colors
    ):
        # Fan triangulation from the centroid: (centroid, p[i], p[i+1]) per edge.
        for index in range(len(polygon) - 1):
            fill_positions.extend([centroid, polygon[index], polygon[index + 1]])
            fill_vertex_colors.extend([fill_color] * 3)

        # Boundary as consecutive line segments over the same traced points.
        for index in range(len(polygon) - 1):
            line_positions.append(polygon[index])
            line_positions.append(polygon[index + 1])
            line_vertex_colors.append(line_color)
            line_vertex_colors.append(line_color)

    if fill_positions:
        fill_geometry = LayerGeometry(
            positions=plot_to_ndc(np.array(fill_positions, dtype=np.float32)),
            colors=np.array(fill_vertex_colors, dtype=np.float32),
            primitive=GL.GL_TRIANGLES,
        )
    else:
        fill_geometry = LayerGeometry(
            positions=np.zeros((0, 2), dtype=np.float32),
            colors=np.zeros((0, 4), dtype=np.float32),
            primitive=GL.GL_TRIANGLES,
        )

    if line_positions:
        line_geometry = LayerGeometry(
            positions=plot_to_ndc(np.array(line_positions, dtype=np.float32)),
            colors=np.array(line_vertex_colors, dtype=np.float32),
            primitive=GL.GL_LINES,
        )
    else:
        line_geometry = LayerGeometry(
            positions=np.zeros((0, 2), dtype=np.float32),
            colors=np.zeros((0, 4), dtype=np.float32),
            primitive=GL.GL_LINES,
        )

    return [fill_geometry, line_geometry]


def interpolate_morse_smale_frame(correspondence: CellCorrespondence, t: float) -> MorseSmaleCellGeometryFrame:
    """
    Move-to-move animation (docs/interactive_ui.md Part 6: "Morse-Smale
    cells | Fill/boundary cross-fade on birth/death of matched cells").

    Matched cells: boundary resampled to `CELL_RESAMPLE_COUNT` points by
    normalized arc length (same technique as ridge/valley chains -- a closed
    polygon is just a curve whose first and last points coincide) and lerped
    point-for-point, with the centroid lerped alongside; full, constant
    alpha throughout, per "interpolate matched boundaries when valid."

    Unmatched cells (appeared/disappeared): drawn at their own single-
    position polygon/centroid, never morphed toward a guessed shape, only
    faded -- "otherwise use a controlled fade for birth/death instead of
    fabricating topology," exactly as instructed.

    Endpoint exactness mirrors the other three interpolated layers: at
    `t<=0.0`/`t>=1.0`, matched cells use the original previous/current
    polygon verbatim (no resampling), and appeared/disappeared cells are
    included or omitted exactly as `build_morse_smale_frame` would for the
    respective single position.
    """
    polygons: list[list[tuple[float, float]]] = []
    centroids: list[tuple[float, float]] = []
    fill_colors: list[tuple[float, float, float, float]] = []
    line_colors: list[tuple[float, float, float, float]] = []

    base_fill_color = hex_to_rgba(ACCEPTED_CELL_FILL_COLOR, alpha=ACCEPTED_CELL_FILL_ALPHA)
    base_line_color = hex_to_rgba(ACCEPTED_CELL_LINE_COLOR, alpha=1.0)

    def _scaled(color: tuple[float, float, float, float], alpha_scale: float) -> tuple[float, float, float, float]:
        return (color[0], color[1], color[2], color[3] * alpha_scale)

    for match in correspondence.matched:
        if t <= 0.0:
            polygon = list(match.previous_geometry.polygon)
            centroid = match.previous_geometry.centroid
        elif t >= 1.0:
            polygon = list(match.current_geometry.polygon)
            centroid = match.current_geometry.centroid
        else:
            previous_samples = resample_by_arc_length(list(match.previous_geometry.polygon), CELL_RESAMPLE_COUNT)
            current_samples = resample_by_arc_length(list(match.current_geometry.polygon), CELL_RESAMPLE_COUNT)
            polygon = [lerp_point(prev, cur, t) for prev, cur in zip(previous_samples, current_samples)]
            centroid = lerp_point(match.previous_geometry.centroid, match.current_geometry.centroid, t)

        polygons.append([(float(x), float(y)) for x, y in polygon])
        centroids.append((float(centroid[0]), float(centroid[1])))
        fill_colors.append(base_fill_color)
        line_colors.append(base_line_color)

    if t < 1.0:
        fade_out = lerp(1.0, 0.0, t)
        for unmatched in correspondence.disappeared:
            polygons.append([(float(x), float(y)) for x, y in unmatched.geometry.polygon])
            centroids.append((float(unmatched.geometry.centroid[0]), float(unmatched.geometry.centroid[1])))
            fill_colors.append(_scaled(base_fill_color, fade_out))
            line_colors.append(_scaled(base_line_color, fade_out))

    if t > 0.0:
        fade_in = lerp(0.0, 1.0, t)
        for unmatched in correspondence.appeared:
            polygons.append([(float(x), float(y)) for x, y in unmatched.geometry.polygon])
            centroids.append((float(unmatched.geometry.centroid[0]), float(unmatched.geometry.centroid[1])))
            fill_colors.append(_scaled(base_fill_color, fade_in))
            line_colors.append(_scaled(base_line_color, fade_in))

    return MorseSmaleCellGeometryFrame(
        polygons=polygons, centroids=centroids, fill_colors=fill_colors, line_colors=line_colors
    )


MORSE_SMALE_LAYER = LayerDefinition(
    id="morse_smale",
    display_name="Morse-Smale",
    data_source=build_morse_smale_frame,
    renderer=render_morse_smale_frame,
)
