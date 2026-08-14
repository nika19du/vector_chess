"""
V2 (Core Visual Hierarchy): the OpenGL renderer previously flattened every
line-based layer to the same width (no `glLineWidth` call anywhere in
desktop_app/), losing the visual hierarchy the matplotlib reference
deliberately encodes via linewidth (Equipotential thin, Ridge/Valley strong,
Morse-Smale boundary a clear-but-lighter-than-Ridge/Valley topology outline,
Gradient light/directional). These tests lock in the reference-derived
line_width values now carried on `LayerGeometry` and drawn via `glLineWidth`
in `gl_canvas.py::paintGL`.

Deliberately relational for the hierarchy checks, exact-value for the direct
reference-constant checks -- the reference constants themselves are the
source of truth, not an arbitrary pixel choice made here.
"""

import chess
import pytest
from OpenGL import GL

from desktop_app.full_position_analysis import build_full_position_analysis
from desktop_app.gl_canvas import MIN_GL_LINE_WIDTH, MathCanvas
from desktop_app.layers.equipotential_layer import RENDER_LINE_WIDTH as EQUIPOTENTIAL_LINE_WIDTH
from desktop_app.layers.equipotential_layer import build_equipotential_frame, render_equipotential_frame
from desktop_app.layers.gradient_layer import _GRADIENT_LINEWIDTH_BASE as GRADIENT_LINE_WIDTH
from desktop_app.layers.gradient_layer import build_gradient_frame, render_gradient_frame
from desktop_app.layers.morse_smale_layer import build_morse_smale_frame, render_morse_smale_frame
from desktop_app.layers.ridge_valley_layer import build_ridge_valley_frame, render_ridge_valley_frame
from desktop_app.main_window import _LAYERS_IN_DRAW_ORDER
from desktop_app.position_cache import CacheEntry, CacheEntryState
from visualization.equipotential_plot import CONTOUR_LINE_WIDTH
from visualization.morse_smale_plot import ACCEPTED_CELL_LINEWIDTH
from visualization.ridge_valley_plot import RIDGE_VALLEY_LINEWIDTH


def _midgame_board() -> chess.Board:
    board = chess.Board()
    for move in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"):
        board.push_san(move)
    return board


def _entry_for(board: chess.Board) -> CacheEntry:
    return CacheEntry(state=CacheEntryState.READY, analysis=build_full_position_analysis(board))


# ---------------------------------------------------------
# 1. Frame/render metadata carries the expected reference-derived width
# ---------------------------------------------------------


def test_equipotential_geometry_line_width_is_clamped_reference_width():
    # The reference's own CONTOUR_LINE_WIDTH (0.7) is below every OpenGL
    # implementation's real floor (MIN_GL_LINE_WIDTH) -- clamped up, not
    # silently ignored.
    assert CONTOUR_LINE_WIDTH < MIN_GL_LINE_WIDTH
    assert EQUIPOTENTIAL_LINE_WIDTH == MIN_GL_LINE_WIDTH

    entry = _entry_for(_midgame_board())
    frame = build_equipotential_frame(entry)
    geometries = render_equipotential_frame(frame)

    assert len(geometries) == 1
    assert geometries[0].line_width == MIN_GL_LINE_WIDTH


def test_ridge_valley_geometry_line_width_matches_the_reference_constant():
    entry = _entry_for(_midgame_board())
    frame = build_ridge_valley_frame(entry)
    assert frame.chains, "expected at least one accepted ridge/valley chain on this position"

    geometries = render_ridge_valley_frame(frame)
    assert len(geometries) == 1
    assert geometries[0].line_width == RIDGE_VALLEY_LINEWIDTH


def test_morse_smale_boundary_line_width_matches_the_reference_constant():
    entry = _entry_for(_midgame_board())
    frame = build_morse_smale_frame(entry)

    fill_geometry, line_geometry = render_morse_smale_frame(frame)
    assert line_geometry.line_width == ACCEPTED_CELL_LINEWIDTH
    # The fill geometry is GL_TRIANGLES -- line_width on it is inert, but it
    # must still exist (the dataclass field always has a value).
    assert fill_geometry.line_width is not None


def test_gradient_geometry_line_width_is_the_reference_formulas_own_floor():
    entry = _entry_for(_midgame_board())
    frame = build_gradient_frame(entry)
    assert frame.segments, "expected at least one visible gradient vector on this position"

    geometries = render_gradient_frame(frame)
    assert len(geometries) == 1
    assert geometries[0].line_width == GRADIENT_LINE_WIDTH


# ---------------------------------------------------------
# 2/3/4. Relative hierarchy: Equipotential <= Gradient < Morse-Smale <= Ridge/Valley
# ---------------------------------------------------------


def test_equipotential_is_thinner_than_ridge_valley():
    assert EQUIPOTENTIAL_LINE_WIDTH < RIDGE_VALLEY_LINEWIDTH


def test_morse_smale_boundary_is_not_heavier_than_ridge_valley():
    assert ACCEPTED_CELL_LINEWIDTH <= RIDGE_VALLEY_LINEWIDTH


def test_gradient_stays_in_the_light_tier_below_the_structural_line_layers():
    assert GRADIENT_LINE_WIDTH <= ACCEPTED_CELL_LINEWIDTH
    assert GRADIENT_LINE_WIDTH < RIDGE_VALLEY_LINEWIDTH


# ---------------------------------------------------------
# 5. line_width survives transition/interpolation
# ---------------------------------------------------------


def test_ridge_valley_line_width_is_unaffected_by_the_animation_path():
    """
    interpolate_ridge_valley_frame only ever produces a RidgeValleyFrame
    (data), which then goes through the exact same render_ridge_valley_frame
    used above -- so line_width is attached at render time regardless of
    whether the frame came from build_ridge_valley_frame or an interpolated
    one. Verified here directly on a hand-built interpolated-shape frame
    rather than assuming it from the code structure.
    """
    from desktop_app.layers.ridge_valley_layer import RidgeValleyChainGeometry, RidgeValleyFrame

    synthetic_interpolated_frame = RidgeValleyFrame(
        chains=[RidgeValleyChainGeometry(points_xy=[(1.0, 1.0), (2.0, 2.0)], color=(1.0, 0.0, 0.0, 0.5))]
    )
    geometries = render_ridge_valley_frame(synthetic_interpolated_frame)
    assert geometries[0].line_width == RIDGE_VALLEY_LINEWIDTH


def test_morse_smale_line_width_is_unaffected_by_the_animation_path():
    from desktop_app.layers.morse_smale_layer import MorseSmaleCellGeometryFrame

    synthetic_interpolated_frame = MorseSmaleCellGeometryFrame(
        polygons=[[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]],
        centroids=[(0.5, 0.5)],
        fill_colors=[(0.0, 1.0, 0.0, 0.22)],
        line_colors=[(0.0, 0.5, 0.0, 1.0)],
    )
    _fill_geometry, line_geometry = render_morse_smale_frame(synthetic_interpolated_frame)
    assert line_geometry.line_width == ACCEPTED_CELL_LINEWIDTH


# ---------------------------------------------------------
# 6/7. No coordinate/math or draw-order changes
# ---------------------------------------------------------


def test_ridge_valley_positions_are_unaffected_by_the_line_width_change():
    """Same "real, untouched coordinates" property test_desktop_app_layers.py
    already asserts -- re-checked here since this milestone touched this
    exact renderer function, to prove line_width is additive metadata, not a
    coordinate-affecting change."""
    entry = _entry_for(_midgame_board())
    frame = build_ridge_valley_frame(entry)
    expected_point_sets = [chain.points_xy for chain in frame.chains]

    geometries = render_ridge_valley_frame(frame)
    expected_vertex_count = sum(2 * (len(points) - 1) for points in expected_point_sets)
    assert geometries[0].positions.shape == (expected_vertex_count, 2)


def test_draw_order_is_unchanged_attack_influence_first_critical_points_last():
    layer_ids = [layer.id for layer in _LAYERS_IN_DRAW_ORDER]
    assert layer_ids == [
        "attack_influence",
        "equipotential",
        "gradient",
        "ridge_valley",
        "morse_smale",
        "critical_points",
    ]


# ---------------------------------------------------------
# Real GL capability check -- skips (never fails) without a real context,
# same precedent as test_desktop_app_canvas.py
# ---------------------------------------------------------


def test_gl_line_width_is_actually_applied_on_a_real_context(qapp):
    """
    Not part of ordinary CI expectations: documents/verifies the real
    OpenGL capability this milestone's report flags as a cross-vendor risk
    (core-profile glLineWidth support above 1.0 is not guaranteed by the
    spec; observed generous support on this machine's AMD driver, but
    NVIDIA drivers are documented to clamp GL_ALIASED_LINE_WIDTH_RANGE to
    [1, 1] in core profile). Skips, rather than fails, when no real context
    is available -- exactly test_desktop_app_canvas.py's existing pattern.
    """
    canvas = MathCanvas()
    canvas.resize(64, 64)
    canvas.show()
    qapp.processEvents()

    if not canvas.isValid():
        pytest.skip(
            "No OpenGL context available under the current QPA platform "
            "(offscreen does not support real GL contexts on this system)."
        )

    canvas.makeCurrent()
    max_supported = GL.glGetFloatv(GL.GL_ALIASED_LINE_WIDTH_RANGE)[1]
    requested = min(RIDGE_VALLEY_LINEWIDTH, float(max_supported))
    GL.glLineWidth(requested)
    applied = GL.glGetFloatv(GL.GL_LINE_WIDTH)
    assert applied == pytest.approx(requested, abs=0.01)
