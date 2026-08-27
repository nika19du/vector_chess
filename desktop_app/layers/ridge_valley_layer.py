from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OpenGL import GL

from desktop_app.correspondence import ChainCorrespondence, CriticalPointCorrespondence
from desktop_app.gl_canvas import LayerGeometry
from desktop_app.layer_registry import LayerDefinition
from desktop_app.layers._arc_length import resample_by_arc_length
from desktop_app.layers._color_scale import hex_to_rgba
from desktop_app.layers._lerp import lerp
from desktop_app.layers._ndc import plot_to_ndc
from desktop_app.position_cache import CacheEntry
from visualization.critical_points_plot import strongest_critical_point
from visualization.ridge_valley_plot import (
    ACCEPTED_ALPHA,
    MIN_POINTS_TO_DRAW,
    RIDGE_LINE_COLOR,
    RIDGE_VALLEY_LINEWIDTH,
    VALLEY_LINE_COLOR,
)

# How many points a matched chain is resampled to before interpolating along
# normalized arc length (docs/interactive_ui.md Part 6: "interpolate matched
# chains along normalized arc length") -- a rendering constant controlling
# animation smoothness only, unrelated to MIN_POINTS_TO_DRAW's display
# threshold above.
CHAIN_RESAMPLE_COUNT = 24

# Milestone C (VECTORCHESS_MODEL_V2_INTEGRATION_AUDIT.md Sec. 10;
# Experiment 007 REPORT.md): a chain anchored at the position's strongest
# critical point (desktop_app/layers/critical_points_layer.py's own
# PROMINENT_MARKER_SCALE) inherits that point's visual cue -- a thicker
# line, not a separate confidence score. Kept modest for the same
# readability reason as the marker cue.
PROMINENT_CHAIN_LINEWIDTH_SCALE = 1.35


@dataclass(frozen=True)
class RidgeValleyChainGeometry:
    points_xy: list[tuple[float, float]]  # plot space, chain order
    color: tuple[float, float, float, float]
    # Milestone C: 0.0..1.0, same continuous/lerpable convention as
    # CriticalPointMarker.prominence in critical_points_layer.py -- 1.0
    # means "this chain's anchor is the position's strongest critical
    # point". Never independently scored; always inherited (see
    # build_ridge_valley_frame / interpolate_ridge_valley_frame below).
    prominence: float = 0.0


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

    # Milestone C: the same accepted set + ranking helper
    # desktop_app/layers/critical_points_layer.py uses -- a chain's
    # prominence is never independently scored, only compared (by object
    # identity, per ClassifiedCriticalPoint's own identity contract) against
    # this position's one strongest point.
    accepted_points = [
        assessment.point for assessment in entry.analysis.critical_point_assessments if assessment.is_accepted
    ]
    strongest = strongest_critical_point(accepted_points)

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
                prominence=1.0 if chain.anchor is not None and chain.anchor is strongest else 0.0,
            )
        )

    return RidgeValleyFrame(chains=chains)


def _chain_color(kind: str, alpha: float) -> tuple[float, float, float, float]:
    color_hex = RIDGE_LINE_COLOR if kind == "ridge" else VALLEY_LINE_COLOR
    return hex_to_rgba(color_hex, alpha=alpha)


def _chain_prominence(chain, strongest) -> float:
    return 1.0 if chain.anchor is not None and chain.anchor is strongest else 0.0


# Default for `critical_point_correspondence` below: `strongest_previous`/
# `strongest_current` both default to None on CriticalPointCorrespondence,
# so every chain's prominence resolves to 0.0 (no anchor can `is None`) --
# i.e. exactly today's pre-Milestone-C behavior, for any caller that
# doesn't (yet) have a real one to pass. Safe as a default value: the
# dataclass is frozen/immutable.
_NO_PROMINENCE_CORRESPONDENCE = CriticalPointCorrespondence(matched=[], appeared=[], disappeared=[])


def interpolate_ridge_valley_frame(
    ridge_correspondence: ChainCorrespondence,
    valley_correspondence: ChainCorrespondence,
    t: float,
    critical_point_correspondence: CriticalPointCorrespondence = _NO_PROMINENCE_CORRESPONDENCE,
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

    Milestone C: `critical_point_correspondence` supplies the same
    `strongest_previous`/`strongest_current` (real endpoints, computed once
    in desktop_app/correspondence.py) `interpolate_critical_points_frame`
    already uses -- a matched chain's prominence is looked up against its
    own `anchor` on each side and lerped by `t`, exactly like `xy`; never
    re-derived from this frame's own interpolated geometry.
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
            prominence_previous = _chain_prominence(match.previous, critical_point_correspondence.strongest_previous)
            prominence_current = _chain_prominence(match.current, critical_point_correspondence.strongest_current)
            prominence = lerp(prominence_previous, prominence_current, t)
            chains.append(
                RidgeValleyChainGeometry(
                    points_xy=points_xy, color=_chain_color(kind, ACCEPTED_ALPHA), prominence=prominence
                )
            )

        if t < 1.0:
            fade_out = lerp(1.0, 0.0, t)
            for chain in correspondence.disappeared:
                chains.append(
                    RidgeValleyChainGeometry(
                        points_xy=[(point.x, point.y) for point in chain.points],
                        color=_chain_color(chain.kind, ACCEPTED_ALPHA * fade_out),
                        prominence=_chain_prominence(chain, critical_point_correspondence.strongest_previous),
                    )
                )

        if t > 0.0:
            fade_in = lerp(0.0, 1.0, t)
            for chain in correspondence.appeared:
                chains.append(
                    RidgeValleyChainGeometry(
                        points_xy=[(point.x, point.y) for point in chain.points],
                        color=_chain_color(chain.kind, ACCEPTED_ALPHA * fade_in),
                        prominence=_chain_prominence(chain, critical_point_correspondence.strongest_current),
                    )
                )

    return RidgeValleyFrame(chains=chains)


def _chain_segments(
    chains: list[RidgeValleyChainGeometry],
) -> tuple[np.ndarray, np.ndarray]:
    plot_positions: list[tuple[float, float]] = []
    vertex_colors: list[tuple[float, float, float, float]] = []

    for chain in chains:
        points = chain.points_xy
        for index in range(len(points) - 1):
            plot_positions.append(points[index])
            plot_positions.append(points[index + 1])
            vertex_colors.append(chain.color)
            vertex_colors.append(chain.color)

    if not plot_positions:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    return plot_to_ndc(np.array(plot_positions, dtype=np.float32)), np.array(vertex_colors, dtype=np.float32)


def render_ridge_valley_frame(frame: RidgeValleyFrame) -> list[LayerGeometry]:
    """
    Two geometries, not one, whenever at least one chain is prominent
    (Milestone C): matplotlib-style per-object line width has no direct GL
    equivalent within a single batched GL_LINES call (`line_width` is one
    scalar per `LayerGeometry`, not per-vertex), so chains are split into a
    normal-weight batch and a prominent-weight batch -- the same
    "two geometries, two primitives" pattern morse_smale_layer.py and
    critical_points_layer.py already use, not a new rendering mechanism.
    A position with no prominent chain (e.g. no accepted ridge/valley chain
    anchored at the strongest point) renders exactly as before: one
    GL_LINES call at RIDGE_VALLEY_LINEWIDTH.
    """
    normal_chains = [chain for chain in frame.chains if chain.prominence < 1.0]
    prominent_chains = [chain for chain in frame.chains if chain.prominence >= 1.0]

    geometries: list[LayerGeometry] = []

    positions, colors = _chain_segments(normal_chains)
    # V2 (visual hierarchy): matches visualization/ridge_valley_plot.py's own
    # RIDGE_VALLEY_LINEWIDTH -- the reference's "strong structural lines"
    # weight, only chains accepted at quality-filter time ever reach here
    # (see build_ridge_valley_frame's docstring), so REJECTED_LINEWIDTH is
    # never relevant to what actually gets drawn.
    geometries.append(
        LayerGeometry(positions=positions, colors=colors, primitive=GL.GL_LINES, line_width=RIDGE_VALLEY_LINEWIDTH)
    )

    if prominent_chains:
        positions, colors = _chain_segments(prominent_chains)
        geometries.append(
            LayerGeometry(
                positions=positions,
                colors=colors,
                primitive=GL.GL_LINES,
                line_width=RIDGE_VALLEY_LINEWIDTH * PROMINENT_CHAIN_LINEWIDTH_SCALE,
            )
        )

    return geometries


RIDGE_VALLEY_LAYER = LayerDefinition(
    id="ridge_valley",
    display_name="Ridge / Valley",
    data_source=build_ridge_valley_frame,
    renderer=render_ridge_valley_frame,
    category="Topology",
    short_caption="Ridge and valley lines connecting the surface's peaks and dips.",
)
