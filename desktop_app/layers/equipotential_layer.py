from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure
from OpenGL import GL

from desktop_app.gl_canvas import LayerGeometry
from desktop_app.layer_registry import LayerDefinition
from desktop_app.layers._color_scale import hex_to_rgba, symmetric_max_abs
from desktop_app.layers._ndc import plot_to_ndc
from desktop_app.position_cache import CacheEntry
from visualization.equipotential_plot import CONTOUR_LEVEL_COUNT, CONTOUR_LINE_ALPHA, CONTOUR_LINE_COLOR


@dataclass(frozen=True)
class EquipotentialFrame:
    # Plot-space line segments, already flattened to consecutive pairs.
    segments_xy: list[tuple[tuple[float, float], tuple[float, float]]]
    color: tuple[float, float, float, float]


def build_equipotential_frame(entry: CacheEntry) -> EquipotentialFrame:
    """
    Reuses matplotlib's own contour-tracing algorithm to get segment
    coordinates, rather than reimplementing marching squares: this *is*
    "reuse the existing mathematical pipeline exactly," since matplotlib's
    `contour()` is how `visualization/equipotential_plot.py` already derives
    equipotential lines today -- called here for its geometry, not to
    produce a rendered plot. `CONTOUR_LEVEL_COUNT`/`CONTOUR_LINE_COLOR`/
    `CONTOUR_LINE_ALPHA` are the exact constants that module already uses.
    """
    if entry.analysis is None:
        raise ValueError("cache entry is not READY: no analysis")

    surface = entry.analysis.surface
    max_absolute_value = symmetric_max_abs(np.asarray(surface.z))
    levels = np.linspace(-max_absolute_value, max_absolute_value, CONTOUR_LEVEL_COUNT)

    # A throwaway, never-shown Figure -- used only to run the contour
    # algorithm and read back its segment coordinates, never rendered.
    figure = Figure()
    axes = figure.add_subplot()
    contour_set = axes.contour(surface.x, surface.y, surface.z, levels=levels)

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for level_segments in contour_set.allsegs:
        for polyline in level_segments:
            for index in range(len(polyline) - 1):
                start = (float(polyline[index][0]), float(polyline[index][1]))
                end = (float(polyline[index + 1][0]), float(polyline[index + 1][1]))
                segments.append((start, end))

    color = hex_to_rgba(CONTOUR_LINE_COLOR, alpha=CONTOUR_LINE_ALPHA)
    return EquipotentialFrame(segments_xy=segments, color=color)


def render_equipotential_frame(frame: EquipotentialFrame) -> list[LayerGeometry]:
    if not frame.segments_xy:
        positions = np.zeros((0, 2), dtype=np.float32)
        colors = np.zeros((0, 4), dtype=np.float32)
    else:
        plot_positions = np.array(
            [xy for segment in frame.segments_xy for xy in segment],
            dtype=np.float32,
        )
        positions = plot_to_ndc(plot_positions)
        colors = np.array([frame.color] * len(plot_positions), dtype=np.float32)

    return [LayerGeometry(positions=positions, colors=colors, primitive=GL.GL_LINES)]


EQUIPOTENTIAL_LAYER = LayerDefinition(
    id="equipotential",
    display_name="Equipotential",
    data_source=build_equipotential_frame,
    renderer=render_equipotential_frame,
)
