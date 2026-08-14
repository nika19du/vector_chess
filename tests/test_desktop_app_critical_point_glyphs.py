"""
V3 (Critical Point Visual Language): maximum/minimum/saddle/degenerate used
to render as identical filled quads, distinguished only by color. This adds
real glyph shapes (triangle-up/triangle-down/X/hollow-ring) matching the
matplotlib reference's own marker shapes -- restoring the classification
distinction without relying on color alone.

Structured in two layers, matching the module split: pure geometry-generator
tests (no Qt/GL/analysis needed at all) first, then the classification ->
glyph dispatch and transition-stability tests against real/synthetic frames.
"""

import math

import chess
import pytest
from OpenGL import GL

from desktop_app.correspondence import CriticalPointCorrespondence, CriticalPointMatch
from desktop_app.full_position_analysis import build_full_position_analysis
from desktop_app.layers._critical_point_glyphs import (
    RING_SEGMENT_COUNT,
    cross_line_segments,
    ring_line_segments,
    triangle_down_vertices,
    triangle_up_vertices,
)
from desktop_app.layers.critical_points_layer import (
    MARKER_HALF_SIZE,
    build_critical_points_frame,
    interpolate_critical_points_frame,
    render_critical_points_frame,
)
from desktop_app.main_window import _LAYERS_IN_DRAW_ORDER, MainWindow
from desktop_app.position_cache import CacheEntry, CacheEntryState
from tests.conftest import make_critical_point


def _midgame_board() -> chess.Board:
    board = chess.Board()
    for move in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"):
        board.push_san(move)
    return board


def _entry_for(board: chess.Board) -> CacheEntry:
    return CacheEntry(state=CacheEntryState.READY, analysis=build_full_position_analysis(board))


# ---------------------------------------------------------
# Pure geometry generators -- no Qt/GL/analysis needed
# ---------------------------------------------------------


def test_triangle_up_apex_points_up_and_is_centered_at_xy():
    """
    Plot-space y grows downward (screen convention, matching
    analysis/geometry.py and gradient_layer.py's own "plot Y grows down")
    -- the visually-higher apex is therefore at the *smaller* y value.
    """
    vertices = triangle_up_vertices(2.0, 3.0, 0.12)
    apex = vertices[0]
    base_left, base_right = vertices[1], vertices[2]

    assert apex == (2.0, 3.0 - 0.12)
    assert base_left[1] == base_right[1] == 3.0 + 0.12
    # Centered: apex visually above center, base below, base symmetric around x.
    assert base_left[0] < 2.0 < base_right[0]
    assert base_right[0] - 2.0 == pytest.approx(2.0 - base_left[0])


def test_triangle_down_is_triangle_up_mirrored_across_y():
    up = triangle_up_vertices(1.0, 1.0, 0.2)
    down = triangle_down_vertices(1.0, 1.0, 0.2)

    mirrored_up = [(x, 2.0 - y) for x, y in up]  # reflect across y=1.0
    assert sorted(down) == sorted(mirrored_up)


def test_cross_line_segments_form_two_diagonals_of_the_bounding_square():
    half = 0.15
    segments = cross_line_segments(0.0, 0.0, half)

    assert len(segments) == 4  # two segments, 2 endpoints each
    diagonal_a = (segments[0], segments[1])
    diagonal_b = (segments[2], segments[3])

    # Each diagonal spans opposite corners of the half x half bounding box.
    assert {tuple(sorted(point)) for point in diagonal_a} != {tuple(sorted(point)) for point in diagonal_b}
    for point in segments:
        assert abs(point[0]) == pytest.approx(half)
        assert abs(point[1]) == pytest.approx(half)


def test_ring_line_segments_form_a_closed_circle_at_the_given_radius():
    x, y, radius = 5.0, 5.0, 0.12
    segments = ring_line_segments(x, y, radius)

    assert len(segments) == RING_SEGMENT_COUNT * 2
    for point_x, point_y in segments:
        distance = math.hypot(point_x - x, point_y - y)
        assert distance == pytest.approx(radius, abs=1e-9)
    # Closed loop: the last segment's endpoint is the first segment's start.
    assert segments[-1] == segments[0]


def test_ring_line_segments_respects_a_custom_segment_count():
    segments = ring_line_segments(0.0, 0.0, 1.0, segment_count=8)
    assert len(segments) == 16


def test_all_glyphs_stay_within_their_declared_half_size_bounds():
    x, y, half = 3.0, 4.0, 0.12
    all_points = (
        triangle_up_vertices(x, y, half)
        + triangle_down_vertices(x, y, half)
        + cross_line_segments(x, y, half)
        + ring_line_segments(x, y, half)
    )
    for point_x, point_y in all_points:
        assert x - half - 1e-9 <= point_x <= x + half + 1e-9
        assert y - half - 1e-9 <= point_y <= y + half + 1e-9


def test_shapes_are_structurally_distinct_from_each_other():
    """
    Not merely different colors -- genuinely different vertex sequences.
    Bounding-box corners are legitimately shared between shapes (the
    triangles' base corners and the cross's endpoints both sit at the same
    (x+-half, y+-half) points) -- that's expected geometry, not a collision,
    so this compares full ordered vertex sequences and counts instead of
    requiring disjoint point sets.
    """
    x, y, half = 0.0, 0.0, 0.1
    up = triangle_up_vertices(x, y, half)
    down = triangle_down_vertices(x, y, half)
    cross = cross_line_segments(x, y, half)
    ring = ring_line_segments(x, y, half)

    assert up != down
    assert len(up) == len(down) == 3
    assert len(cross) == 4
    assert len(ring) == RING_SEGMENT_COUNT * 2
    # Every ring point sits at exactly distance `half` from center -- true
    # of no triangle *base* corner (those sit at the bounding-box corners,
    # distance half*sqrt(2)) -- confirming the ring is a genuinely different
    # shape family (a circle), not just a triangle/cross with extra points.
    assert all(math.hypot(px - x, py - y) == pytest.approx(half) for px, py in ring)
    base_corners = up[1:] + down[1:]  # excludes each triangle's apex
    assert not any(math.hypot(px - x, py - y) == pytest.approx(half) for px, py in base_corners)


# ---------------------------------------------------------
# Classification -> glyph dispatch, on real frames
# ---------------------------------------------------------


def test_every_accepted_classification_present_on_this_position_gets_a_glyph():
    entry = _entry_for(_midgame_board())
    frame = build_critical_points_frame(entry)
    present_classifications = {marker.classification for marker in frame.markers}
    assert present_classifications, "expected at least one accepted critical point on this position"
    assert present_classifications <= {"maximum", "minimum", "saddle", "degenerate"}


def test_maximum_and_minimum_only_contribute_triangle_geometry():
    entry = _entry_for(_midgame_board())
    frame = build_critical_points_frame(entry)
    triangle_geometry, _line_geometry = render_critical_points_frame(frame)

    filled_markers = [m for m in frame.markers if m.classification in ("maximum", "minimum")]
    assert triangle_geometry.positions.shape[0] == len(filled_markers) * 3


def test_saddle_and_degenerate_only_contribute_line_geometry():
    entry = _entry_for(_midgame_board())
    frame = build_critical_points_frame(entry)
    _triangle_geometry, line_geometry = render_critical_points_frame(frame)

    expected_vertex_count = sum(
        4 if m.classification == "saddle" else RING_SEGMENT_COUNT * 2
        for m in frame.markers
        if m.classification in ("saddle", "degenerate")
    )
    assert line_geometry.positions.shape[0] == expected_vertex_count


def test_marker_center_is_exactly_the_original_critical_point_coordinate():
    """
    No coordinate change: each glyph's bounding-box center must land
    exactly on the accepted point's own (x, y), the same coordinate the
    pre-V3 quad marker was centered on.

    Bounding-box center, not vertex average: a triangle's apex (1 vertex)
    and base (2 vertices) don't average to the shape's visual center --
    only min/max of each axis does, and it does so uniformly across all
    four glyph shapes, which a vertex average does not.
    """
    entry = _entry_for(_midgame_board())
    frame = build_critical_points_frame(entry)

    for marker in frame.markers:
        x, y = marker.xy
        if marker.classification == "maximum":
            vertices = triangle_up_vertices(x, y, MARKER_HALF_SIZE)
        elif marker.classification == "minimum":
            vertices = triangle_down_vertices(x, y, MARKER_HALF_SIZE)
        elif marker.classification == "saddle":
            vertices = cross_line_segments(x, y, MARKER_HALF_SIZE)
        else:
            vertices = ring_line_segments(x, y, MARKER_HALF_SIZE)

        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        bounding_center_x = (min(xs) + max(xs)) / 2
        bounding_center_y = (min(ys) + max(ys)) / 2
        assert bounding_center_x == pytest.approx(x, abs=1e-6)
        assert bounding_center_y == pytest.approx(y, abs=1e-6)


# ---------------------------------------------------------
# Transition/interpolation stability
# ---------------------------------------------------------


def test_matched_points_keep_their_classification_throughout_interpolation():
    match = CriticalPointMatch(
        previous=make_critical_point(0.0, 0.0, classification="maximum"),
        current=make_critical_point(1.0, 1.0, classification="maximum"),
    )
    correspondence = CriticalPointCorrespondence(matched=[match], appeared=[], disappeared=[])

    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        frame = interpolate_critical_points_frame(correspondence, t)
        assert len(frame.markers) == 1
        assert frame.markers[0].classification == "maximum"


def test_a_maximum_never_renders_as_a_minimum_or_saddle_mid_transition():
    """Direct regression for the plan's explicit worry: transition-frame
    plumbing must never let one classification's glyph leak into another's
    slot mid-animation."""
    match = CriticalPointMatch(
        previous=make_critical_point(0.0, 0.0, classification="maximum"),
        current=make_critical_point(2.0, 2.0, classification="maximum"),
    )
    correspondence = CriticalPointCorrespondence(matched=[match], appeared=[], disappeared=[])

    frame = interpolate_critical_points_frame(correspondence, 0.5)
    triangle_geometry, line_geometry = render_critical_points_frame(frame)

    assert triangle_geometry.positions.shape[0] == 3  # one filled triangle
    assert line_geometry.positions.shape[0] == 0  # no X or ring drawn


def test_birth_and_death_points_keep_their_own_classification():
    appeared = make_critical_point(3.0, 3.0, classification="saddle")
    disappeared = make_critical_point(4.0, 4.0, classification="degenerate")
    correspondence = CriticalPointCorrespondence(matched=[], appeared=[appeared], disappeared=[disappeared])

    frame_mid = interpolate_critical_points_frame(correspondence, 0.5)
    classifications = {marker.classification for marker in frame_mid.markers}
    assert classifications == {"saddle", "degenerate"}


def test_interpolated_position_is_the_exact_linear_lerp_not_a_glyph_reassignment():
    match = CriticalPointMatch(
        previous=make_critical_point(0.0, 0.0, classification="minimum"),
        current=make_critical_point(4.0, 2.0, classification="minimum"),
    )
    correspondence = CriticalPointCorrespondence(matched=[match], appeared=[], disappeared=[])

    frame = interpolate_critical_points_frame(correspondence, 0.25)
    assert frame.markers[0].xy == pytest.approx((1.0, 0.5))
    assert frame.markers[0].classification == "minimum"


# ---------------------------------------------------------
# Draw order unchanged
# ---------------------------------------------------------


def test_draw_order_still_ends_with_critical_points_on_top():
    layer_ids = [layer.id for layer in _LAYERS_IN_DRAW_ORDER]
    assert layer_ids[-1] == "critical_points"
    assert layer_ids[0] == "attack_influence"


# ---------------------------------------------------------
# Branch-switch classification freshness (V6)
# ---------------------------------------------------------


def test_critical_point_classification_is_freshly_computed_after_a_branch_switch(qtbot):
    """
    Every existing classification-stability test (above) exercises
    continuous scrub interpolation between two *matched* frames. None of
    them exercise a discrete branch switch: undo() back to a shared
    ancestor, then a *different* move than the abandoned line. PositionCache
    is FEN-keyed, so a genuinely different position is always computed from
    scratch by `build_full_position_analysis`, never interpolated from the
    abandoned branch -- this test drives the real MainWindow/SessionState/
    TransitionController pipeline through exactly that sequence and asserts
    the rendered critical-points geometry after the branch switch is
    bit-for-bit identical to an independent, from-scratch computation for
    the new position, not anything derived from the abandoned line.
    """
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)

    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5"):
        board = window.session_state.current_node.board()
        window.session_state.make_move(board.parse_san(san))

    branch_point = window.session_state.current_node
    abandoned_board = branch_point.board()
    abandoned_move = abandoned_board.parse_san("a6")
    window.session_state.make_move(abandoned_move)
    abandoned_board.push(abandoned_move)
    qtbot.waitUntil(lambda: window.transition_controller._to_fen == abandoned_board.board_fen(), timeout=5000)
    qtbot.waitUntil(lambda: not window.transition_controller.is_animating, timeout=5000)

    window.session_state.undo()
    qtbot.waitUntil(lambda: window.session_state.current_node is branch_point, timeout=5000)
    qtbot.waitUntil(lambda: not window.transition_controller.is_animating, timeout=5000)

    new_board = branch_point.board()
    new_move = new_board.parse_san("Nf6")
    window.session_state.make_move(new_move)
    new_board.push(new_move)
    new_fen = new_board.board_fen()
    assert new_fen != abandoned_board.board_fen(), "test setup: the two branches must reach different positions"

    qtbot.waitUntil(lambda: window.transition_controller._to_fen == new_fen, timeout=5000)
    qtbot.waitUntil(lambda: not window.transition_controller.is_animating, timeout=5000)

    actual_geometries = window.canvas._layer_geometries["critical_points"]

    expected_entry = _entry_for(new_board)
    expected_frame = build_critical_points_frame(expected_entry)
    expected_geometries = render_critical_points_frame(expected_frame)

    assert len(actual_geometries) == len(expected_geometries)
    for actual, expected in zip(actual_geometries, expected_geometries):
        assert actual.positions == pytest.approx(expected.positions)
        assert actual.colors == pytest.approx(expected.colors)
        assert actual.primitive == expected.primitive
