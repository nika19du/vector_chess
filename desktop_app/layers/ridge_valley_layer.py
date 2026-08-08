from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OpenGL import GL

from desktop_app.correspondence import ChainCorrespondence
from desktop_app.gl_canvas import LayerGeometry
from desktop_app.layer_registry import LayerDefinition
from desktop_app.layers._arc_length import resample_by_arc_length
from desktop_app.layers._color_scale import hex_to_rgba
from desktop_app.layers._lerp import lerp
from desktop_app.layers._ndc import plot_to_ndc
from desktop_app.position_cache import CacheEntry
from visualization.ridge_valley_plot import (
    ACCEPTED_ALPHA,
    MIN_POINTS_TO_DRAW,
    RIDGE_LINE_COLOR,
    VALLEY_LINE_COLOR,
)

# How many points a matched chain is resampled to before interpolating along
# normalized arc length (docs/interactive_ui.md Part 6: "interpolate matched
# chains along normalized arc length") -- a rendering constant controlling
# animation smoothness only, unrelated to MIN_POINTS_TO_DRAW's display
# threshold above.
CHAIN_RESAMPLE_COUNT = 24


@dataclass(frozen=True)
class RidgeValleyChainGeometry:
    points_xy: list[tuple[float, float]]  # plot space, chain order
    color: tuple[float, float, float, float]


@dataclass(frozen=True)
class RidgeValleyFrame:
    chains: list[RidgeValleyChainGeometry]


def build_ridge_valley_frame(entry: CacheEntry) -> RidgeValleyFrame:
    """
    Only quality-accepted chains meeting `MIN_POINTS_TO_DRAW` are drawn --
    matches `visualization/ridge_valley_plot.py`'s own `show_only_accepted
    =True` default exactly. Ridge/valley chains are traced from already
    quality-accepted critical points regardless of display filtering
    (`docs/mathematics.md` Section 10's own math, unchanged by
    `build_full_position_analysis`); this layer only controls what's shown.
    """
    if entry.analysis is None:
        raise ValueError("cache entry is not READY: no analysis")

    chains: list[RidgeValleyChainGeometry] = []
    for assessment in entry.analysis.ridge_assessments + entry.analysis.valley_assessments:
        if not assessment.is_accepted:
            continue

        chain = assessment.chain
        if len(chain.points) < MIN_POINTS_TO_DRAW:
            continue

        color_hex = RIDGE_LINE_COLOR if chain.kind == "ridge" else VALLEY_LINE_COLOR
        color = hex_to_rgba(color_hex, alpha=ACCEPTED_ALPHA)
        chains.append(
            RidgeValleyChainGeometry(
                points_xy=[(point.x, point.y) for point in chain.points],
                color=color,
            )
        )

    return RidgeValleyFrame(chains=chains)


def _chain_color(kind: str, alpha: float) -> tuple[float, float, float, float]:
    color_hex = RIDGE_LINE_COLOR if kind == "ridge" else VALLEY_LINE_COLOR
    return hex_to_rgba(color_hex, alpha=alpha)


def interpolate_ridge_valley_frame(
    ridge_correspondence: ChainCorrespondence, valley_correspondence: ChainCorrespondence, t: float
) -> RidgeValleyFrame:
    """
    Move-to-move animation (docs/interactive_ui.md Part 6: "Ridge / Valley |
    Chain points interpolate along matched chains; a chain whose match
    breaks animates a visible split"). Matched chains are independently
    traced on each side, so they rarely share a point count -- both are
    resampled to `CHAIN_RESAMPLE_COUNT` points by normalized arc length
    (`_arc_length.resample_by_arc_length`) before lerping point-for-point,
    rather than pairing raw points by index.

    Endpoint exactness: at `t<=0.0`/`t>=1.0`, a matched chain draws its
    original (unresampled) previous/current points verbatim -- resampling
    only runs strictly between the endpoints, so `t=0`/`t=1` reproduce
    `build_ridge_valley_frame` on the respective entry exactly. `appeared`
    chains are omitted entirely at `t<=0.0`, `disappeared` at `t>=1.0`, for
    the same reason as the critical points layer.
    """
    chains: list[RidgeValleyChainGeometry] = []

    for correspondence in (ridge_correspondence, valley_correspondence):
        for match in correspondence.matched:
            kind = match.current.kind
            if t <= 0.0:
                points_xy = [(point.x, point.y) for point in match.previous.points]
            elif t >= 1.0:
                points_xy = [(point.x, point.y) for point in match.current.points]
            else:
                previous_samples = resample_by_arc_length(
                    [(point.x, point.y) for point in match.previous.points], CHAIN_RESAMPLE_COUNT
                )
                current_samples = resample_by_arc_length(
                    [(point.x, point.y) for point in match.current.points], CHAIN_RESAMPLE_COUNT
                )
                points_xy = [
                    (lerp(prev[0], cur[0], t), lerp(prev[1], cur[1], t))
                    for prev, cur in zip(previous_samples, current_samples)
                ]
            chains.append(RidgeValleyChainGeometry(points_xy=points_xy, color=_chain_color(kind, ACCEPTED_ALPHA)))

        if t < 1.0:
            fade_out = lerp(1.0, 0.0, t)
            for chain in correspondence.disappeared:
                chains.append(
                    RidgeValleyChainGeometry(
                        points_xy=[(point.x, point.y) for point in chain.points],
                        color=_chain_color(chain.kind, ACCEPTED_ALPHA * fade_out),
                    )
                )

        if t > 0.0:
            fade_in = lerp(0.0, 1.0, t)
            for chain in correspondence.appeared:
                chains.append(
                    RidgeValleyChainGeometry(
                        points_xy=[(point.x, point.y) for point in chain.points],
                        color=_chain_color(chain.kind, ACCEPTED_ALPHA * fade_in),
                    )
                )

    return RidgeValleyFrame(chains=chains)


def render_ridge_valley_frame(frame: RidgeValleyFrame) -> list[LayerGeometry]:
    plot_positions: list[tuple[float, float]] = []
    vertex_colors: list[tuple[float, float, float, float]] = []

    for chain in frame.chains:
        points = chain.points_xy
        for index in range(len(points) - 1):
            plot_positions.append(points[index])
            plot_positions.append(points[index + 1])
            vertex_colors.append(chain.color)
            vertex_colors.append(chain.color)

    if not plot_positions:
        positions = np.zeros((0, 2), dtype=np.float32)
        colors = np.zeros((0, 4), dtype=np.float32)
    else:
        positions = plot_to_ndc(np.array(plot_positions, dtype=np.float32))
        colors = np.array(vertex_colors, dtype=np.float32)

    return [LayerGeometry(positions=positions, colors=colors, primitive=GL.GL_LINES)]


RIDGE_VALLEY_LAYER = LayerDefinition(
    id="ridge_valley",
    display_name="Ridge / Valley",
    data_source=build_ridge_valley_frame,
    renderer=render_ridge_valley_frame,
)
