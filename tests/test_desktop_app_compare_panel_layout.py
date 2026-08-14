"""
Branch Comparison V1 -- layout refinement pass (post-approval, following a
real-screenshot review). Covers ONLY layout/geometry invariants; the
mathematics/state tested elsewhere (tests/test_desktop_app_compare_state.py,
_compare_math.py, _compare_panel.py, _timeline_compare.py, _compare_scrub.py)
is unchanged and not re-tested here.

Deliberately relational, not exact-pixel -- the same established pattern
tests/test_desktop_app_responsive_layout.py already uses for the normal V7
workspace.
"""

import chess

from desktop_app.compare_panel import (
    BLOCK_GAP,
    MAX_GRID_PIXELS,
    MIN_GRID_PIXELS,
    _square_pixels_for_size,
)
from desktop_app.main_window import MainWindow

RESOLUTIONS = [(1280, 720), (1600, 900), (1920, 1080)]


def _siblings(session_state):
    session_state.make_move(chess.Move.from_uci("e2e4"))
    e4_node = session_state.current_node
    session_state.undo()
    session_state.make_move(chess.Move.from_uci("d2d4"))
    d4_node = session_state.current_node
    return e4_node, d4_node


def _resize_and_settle(app, window, width, height):
    window.resize(width, height)
    for _ in range(5):
        app.processEvents()


def _open_compare_at(qapp, window, width, height):
    _resize_and_settle(qapp, window, width, height)
    for _ in range(3):
        qapp.processEvents()


# ---------------------------------------------------------
# _square_pixels_for_size: pure function, no Qt layout needed
# ---------------------------------------------------------


def test_square_size_clamps_to_the_minimum_when_space_is_very_tight():
    assert _square_pixels_for_size(300, 300) == MIN_GRID_PIXELS


def test_square_size_clamps_to_the_maximum_when_space_is_abundant():
    assert _square_pixels_for_size(5000, 5000) == MAX_GRID_PIXELS


def test_square_size_is_monotonic_non_decreasing_in_width():
    small = _square_pixels_for_size(1000, 900)
    large = _square_pixels_for_size(1800, 900)
    assert large >= small


def test_square_size_is_monotonic_non_decreasing_in_height():
    small = _square_pixels_for_size(1600, 500)
    large = _square_pixels_for_size(1600, 900)
    assert large >= small


def test_square_size_is_bounded_by_the_tighter_of_width_or_height():
    # A wide-but-short panel must be governed by height, not width -- this
    # is the fix for "Δ pushed below the fold on a short window": growing
    # squares to fill width alone, ignoring height, would reintroduce it.
    wide_short = _square_pixels_for_size(3000, 500)
    wide_tall = _square_pixels_for_size(3000, 1200)
    assert wide_short < wide_tall


# ---------------------------------------------------------
# real geometry at target resolutions
# ---------------------------------------------------------


def test_board_a_and_board_b_are_always_the_same_size(qapp, qtbot):
    window = MainWindow()
    window.show()
    e4_node, d4_node = _siblings(window.session_state)
    window.session_state.enter_compare(d4_node, e4_node)

    for width, height in RESOLUTIONS:
        _open_compare_at(qapp, window, width, height)
        panel = window.compare_panel
        assert panel._board_view_a.size() == panel._board_view_b.size()
        assert panel._influence_view_a.size() == panel._influence_view_b.size()
        assert panel._board_view_a.size() == panel._influence_view_a.size()


def test_gap_between_a_and_b_is_small_and_constant_across_resolutions(qapp, qtbot):
    # The original bug: A and B each centered independently within their own
    # half of the full window width, producing a gap that GREW with window
    # width (approaching half the window). The fix uses a fixed gap
    # (BLOCK_GAP) regardless of how much total width is available.
    window = MainWindow()
    window.show()
    e4_node, d4_node = _siblings(window.session_state)
    window.session_state.enter_compare(d4_node, e4_node)

    gaps = {}
    for width, height in RESOLUTIONS:
        _open_compare_at(qapp, window, width, height)
        panel = window.compare_panel
        influence_a_right = panel._influence_view_a.mapTo(panel, panel._influence_view_a.rect().topRight()).x()
        board_b_left = panel._board_view_b.mapTo(panel, panel._board_view_b.rect().topLeft()).x()
        gap = board_b_left - influence_a_right
        gaps[width] = gap
        # Bounded and small -- nowhere near "opposite edges of the window".
        assert 0 < gap <= BLOCK_GAP + 4

    # And constant: the gap must not grow as the window gets wider (the
    # exact symptom from the real-screenshot review).
    assert gaps[1280] == gaps[1600] == gaps[1920]


def test_a_and_b_never_overlap(qapp, qtbot):
    window = MainWindow()
    window.show()
    e4_node, d4_node = _siblings(window.session_state)
    window.session_state.enter_compare(d4_node, e4_node)

    for width, height in RESOLUTIONS:
        _open_compare_at(qapp, window, width, height)
        panel = window.compare_panel
        influence_a_right = panel._influence_view_a.mapTo(panel, panel._influence_view_a.rect().topRight()).x()
        board_b_left = panel._board_view_b.mapTo(panel, panel._board_view_b.rect().topLeft()).x()
        assert influence_a_right <= board_b_left


def test_squares_scale_upward_with_more_available_space(qapp, qtbot):
    window = MainWindow()
    window.show()
    e4_node, d4_node = _siblings(window.session_state)
    window.session_state.enter_compare(d4_node, e4_node)

    sizes = {}
    for width, height in RESOLUTIONS:
        _open_compare_at(qapp, window, width, height)
        sizes[width] = window.compare_panel._board_view_a.width()

    # Strictly increasing across the three target resolutions -- not stuck
    # at a single fixed size regardless of how much room is available (the
    # original ~200px-always complaint).
    assert sizes[1280] < sizes[1600] < sizes[1920]
    # And no longer capped at the old fixed constant on a large display.
    assert sizes[1920] > 200


def test_squares_never_exceed_the_declared_maximum(qapp, qtbot):
    window = MainWindow()
    window.show()
    e4_node, d4_node = _siblings(window.session_state)
    window.session_state.enter_compare(d4_node, e4_node)

    for width, height in RESOLUTIONS:
        _open_compare_at(qapp, window, width, height)
        assert window.compare_panel._board_view_a.width() <= MAX_GRID_PIXELS


# ---------------------------------------------------------
# Δ and summary visible in the initial viewport where resolution permits
# ---------------------------------------------------------


def test_no_horizontal_scrolling_needed_at_any_target_resolution(qapp, qtbot):
    window = MainWindow()
    window.show()
    e4_node, d4_node = _siblings(window.session_state)
    window.session_state.enter_compare(d4_node, e4_node)

    for width, height in RESOLUTIONS:
        _open_compare_at(qapp, window, width, height)
        assert window.compare_panel._scroll_area.horizontalScrollBar().maximum() == 0


def test_no_vertical_scrolling_needed_at_1600x900_or_1920x1080(qapp, qtbot):
    window = MainWindow()
    window.show()
    e4_node, d4_node = _siblings(window.session_state)
    window.session_state.enter_compare(d4_node, e4_node)

    for width, height in [(1600, 900), (1920, 1080)]:
        _open_compare_at(qapp, window, width, height)
        assert window.compare_panel._scroll_area.verticalScrollBar().maximum() == 0


def test_1280x720_needs_no_more_than_graceful_minor_scrolling(qapp, qtbot):
    # The approved refinement explicitly allows scrolling at 1280x720 "if
    # genuinely necessary" -- this only guards against a regression back to
    # the original "half the comparison result is below the fold" problem,
    # not a strict zero-scroll requirement at the smallest target size.
    window = MainWindow()
    window.show()
    e4_node, d4_node = _siblings(window.session_state)
    window.session_state.enter_compare(d4_node, e4_node)
    _open_compare_at(qapp, window, 1280, 720)

    scroll_needed = window.compare_panel._scroll_area.verticalScrollBar().maximum()
    viewport_height = window.compare_panel._scroll_area.viewport().height()
    assert scroll_needed <= viewport_height * 0.25


def test_difference_view_and_metrics_are_both_inside_the_same_result_card(qapp, qtbot):
    # V1.1 polish: Δ + legend + metrics live inside one bordered "result
    # card" (self._result_frame) -- their vertical positions must be close
    # together (same visual group), not scattered across the page.
    window = MainWindow()
    window.show()
    e4_node, d4_node = _siblings(window.session_state)
    window.session_state.enter_compare(d4_node, e4_node)
    _open_compare_at(qapp, window, 1920, 1080)
    panel = window.compare_panel

    diff_top = panel._diff_view.mapTo(panel, panel._diff_view.rect().topLeft()).y()
    metrics_row = panel._metrics_grid.itemAt(0).layout()
    first_metric_widget = metrics_row.itemAt(0).widget()
    metrics_top = first_metric_widget.mapTo(panel, first_metric_widget.rect().topLeft()).y()
    assert abs(diff_top - metrics_top) < panel._diff_view.height()

    # Both must be visually inside the same bordered result frame -- not
    # merely nearby, but actual descendants of it.
    assert panel._result_frame.isAncestorOf(panel._diff_view)
    assert panel._result_frame.isAncestorOf(first_metric_widget)


def test_difference_view_is_visible_by_default_once_ready(qapp, qtbot):
    window = MainWindow()
    window.show()
    e4_node, d4_node = _siblings(window.session_state)
    window.session_state.enter_compare(d4_node, e4_node)
    _open_compare_at(qapp, window, 1920, 1080)
    panel = window.compare_panel

    # No checkbox exists anymore (V1.1 polish item 4) -- Δ is unconditional.
    assert not hasattr(panel, "_diff_checkbox")
    assert panel._diff_view.isVisible() is True
    assert panel._metrics_grid.count() == 3


# ---------------------------------------------------------
# Close restores the normal workspace exactly, including the audio mixer
# ---------------------------------------------------------


def test_close_restores_the_normal_workspace_including_the_audio_mixer(qapp, qtbot):
    window = MainWindow()
    e4_node, d4_node = _siblings(window.session_state)
    qtbot.wait(10)

    window.session_state.enter_compare(d4_node, e4_node)
    assert window.audio_mixer_panel.isHidden() is True

    window.session_state.exit_compare()

    assert window.audio_mixer_panel.isHidden() is False
    assert window.board_panel.isHidden() is False
    assert window.canvas.isHidden() is False
    assert window.layer_panel.isHidden() is False
    assert window.compare_panel.isHidden() is True


def test_timeline_panel_remains_visible_and_usable_during_compare(qapp, qtbot):
    # Timeline must stay usable (not just present) -- it's the "any real
    # navigation exits compare mode" affordance.
    window = MainWindow()
    e4_node, d4_node = _siblings(window.session_state)
    qtbot.wait(10)

    window.session_state.enter_compare(d4_node, e4_node)

    assert window.timeline_panel.isHidden() is False
    assert window.timeline_panel._first_button.isEnabled() or window.timeline_panel._previous_button.isEnabled()
