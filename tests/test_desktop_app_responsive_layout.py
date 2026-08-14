"""
V5 (Responsive Layout Refinement): closes the remaining weak point after V0-
V4 -- MainWindow's true minimum height (863px post-V4) overflowed the
requested window size by 143px at 1280x720 and 95px at 1366x768. Fixed via
pure spacing/margin trims (main_window.py's grid, TimelinePanel,
BoardPanel, LayerPanel, AudioMixerPanel -- ~87px, zero content change) plus
bounding BoardPanel between MIN_BOARD_PIXELS and BOARD_PIXELS instead of a
fixed 480x480 (the remaining, measured blocker).

Deliberately relational, not exact-pixel, per the established V0 pattern.
"""

import chess
import pytest

from desktop_app.board_panel import BOARD_PIXELS, MIN_BOARD_PIXELS, BoardPanel
from desktop_app.gl_canvas import compute_square_viewport
from desktop_app.main_window import MainWindow
from desktop_app.session_state import SessionState

RESOLUTIONS = [(1280, 720), (1366, 768), (1600, 900), (1920, 1080)]


def _resize_and_settle(app, window, width, height):
    window.resize(width, height)
    for _ in range(5):
        app.processEvents()


# ---------------------------------------------------------
# No overflow at any of the four target resolutions
# ---------------------------------------------------------


def test_main_window_never_exceeds_the_requested_size_at_any_target_resolution(qapp, qtbot):
    window = MainWindow()
    window.show()
    for width, height in RESOLUTIONS:
        _resize_and_settle(qapp, window, width, height)
        central = window.centralWidget()
        assert central.width() <= width
        assert central.height() <= height


def test_1280x720_fits_exactly_the_originally_reported_weak_point(qapp, qtbot):
    window = MainWindow()
    window.show()
    _resize_and_settle(qapp, window, 1280, 720)
    central = window.centralWidget()
    assert (central.width(), central.height()) == (1280, 720)


def test_1366x768_fits_exactly(qapp, qtbot):
    window = MainWindow()
    window.show()
    _resize_and_settle(qapp, window, 1366, 768)
    central = window.centralWidget()
    assert (central.width(), central.height()) == (1366, 768)


# ---------------------------------------------------------
# Board: square, bounded, still dedicated primary content
# ---------------------------------------------------------


def test_board_is_square_and_within_bounds_at_every_target_resolution(qapp, qtbot):
    window = MainWindow()
    window.show()
    for width, height in RESOLUTIONS:
        _resize_and_settle(qapp, window, width, height)
        board_view = window.board_panel.board_view
        assert board_view.width() == board_view.height()
        assert MIN_BOARD_PIXELS <= board_view.width() <= BOARD_PIXELS


def test_board_shrinks_at_the_tightest_resolution_and_stays_at_default_size_where_room_allows(qapp, qtbot):
    """
    Originally written to distinguish 'always pinned at the floor' from
    'genuinely responsive'. V7 (desktop workspace layout) investigated
    letting the board regrow past MIN_BOARD_PIXELS on roomy windows
    (removing board_panel's old rowSpan does unblock WIDTH growth) but
    found no reliable, non-hacky way to keep WIDTH and HEIGHT growing in
    lockstep (see board_panel.py's sizeHint docstring), so the board is
    deliberately kept pinned at MIN_BOARD_PIXELS at every resolution,
    matching pre-V7 behavior exactly. `roomiest_side >= tightest_side`
    still holds (trivially, both equal MIN_BOARD_PIXELS) -- this test no
    longer distinguishes the two cases it was named for, but still
    correctly asserts the board never shrinks below its floor or exceeds
    its ceiling.
    """
    window = MainWindow()
    window.show()

    _resize_and_settle(qapp, window, 1280, 720)
    tightest_side = window.board_panel.board_view.width()

    _resize_and_settle(qapp, window, 1920, 1080)
    roomiest_side = window.board_panel.board_view.width()

    assert tightest_side == MIN_BOARD_PIXELS
    assert roomiest_side >= tightest_side


# ---------------------------------------------------------
# Square GL viewport still correct (V0 invariant, re-checked post-V5)
# ---------------------------------------------------------


def test_math_canvas_viewport_remains_square_at_every_target_resolution(qapp, qtbot):
    window = MainWindow()
    window.show()
    for width, height in RESOLUTIONS:
        _resize_and_settle(qapp, window, width, height)
        canvas = window.canvas
        _, _, viewport_w, viewport_h = compute_square_viewport(canvas.width(), canvas.height())
        assert viewport_w == viewport_h
        assert viewport_w > 0


# ---------------------------------------------------------
# Primary content still dominant; secondary/tertiary still compact
# ---------------------------------------------------------


def test_primary_content_still_dominates_tertiary_strips_at_every_resolution(qapp, qtbot):
    window = MainWindow()
    window.show()
    for width, height in RESOLUTIONS:
        _resize_and_settle(qapp, window, width, height)
        primary_height = window.board_panel.height()
        tertiary_height = window.timeline_panel.height() + window.audio_mixer_panel.height()
        assert primary_height > tertiary_height


def test_layer_panel_stays_usable_at_every_resolution(qapp, qtbot):
    """Usable == every checkbox/slider/the preset combo remain real,
    interactive widgets with nonzero size -- not clipped to zero."""
    window = MainWindow()
    window.show()
    for width, height in RESOLUTIONS:
        _resize_and_settle(qapp, window, width, height)
        assert window.layer_panel.width() > 0
        assert window.layer_panel.height() > 0
        assert window.layer_panel._preset_combo.height() > 0
        for checkbox in window.layer_panel._checkboxes.values():
            assert checkbox.height() > 0


def test_timeline_and_mixer_remain_compact_across_all_four_resolutions(qapp, qtbot):
    window = MainWindow()
    window.show()
    _resize_and_settle(qapp, window, *RESOLUTIONS[0])
    baseline_timeline = window.timeline_panel.height()
    baseline_mixer = window.audio_mixer_panel.height()

    for width, height in RESOLUTIONS[1:]:
        _resize_and_settle(qapp, window, width, height)
        assert window.timeline_panel.height() == baseline_timeline
        assert window.audio_mixer_panel.height() == baseline_mixer


# ---------------------------------------------------------
# Stability: repeated resize cycles across all four resolutions
# ---------------------------------------------------------


def test_repeated_resize_cycles_across_all_four_resolutions_produce_stable_geometry(qapp, qtbot):
    window = MainWindow()
    window.show()

    def _snapshot():
        return {
            "board": (window.board_panel.width(), window.board_panel.height()),
            "board_view": (window.board_panel.board_view.width(), window.board_panel.board_view.height()),
            "canvas": (window.canvas.width(), window.canvas.height()),
            "layer_panel": (window.layer_panel.width(), window.layer_panel.height()),
            "timeline": (window.timeline_panel.width(), window.timeline_panel.height()),
            "mixer": (window.audio_mixer_panel.width(), window.audio_mixer_panel.height()),
        }

    first_pass = []
    for width, height in RESOLUTIONS:
        _resize_and_settle(qapp, window, width, height)
        first_pass.append(_snapshot())

    second_pass = []
    for width, height in RESOLUTIONS:
        _resize_and_settle(qapp, window, width, height)
        second_pass.append(_snapshot())

    assert first_pass == second_pass


# ---------------------------------------------------------
# Responsive board: dedicated hit-testing/painting correctness at a
# non-default size (new behavior this milestone introduces)
# ---------------------------------------------------------


def test_board_hit_testing_is_correct_at_a_shrunk_non_default_size(qtbot):
    """
    Every pre-existing board_panel interaction test only ever exercises the
    board at its default (unresized, BOARD_PIXELS) size. This is the one
    dedicated to the genuinely new behavior: hit-testing must still resolve
    to the right square after the board has actually been resized smaller
    -- not merely bounded in theory.
    """
    state = SessionState(chess.pgn.Game())
    panel = BoardPanel(state)
    qtbot.addWidget(panel)

    panel.board_view.resize(MIN_BOARD_PIXELS, MIN_BOARD_PIXELS)

    offset_x, offset_y, square_pixels = panel.board_view._board_geometry()
    assert square_pixels == pytest.approx(MIN_BOARD_PIXELS / 8)

    # e2's center in the resized board.
    e2_x, e2_y = panel.board_view._square_top_left(chess.E2)
    resolved = panel.board_view._pixel_to_square(e2_x + square_pixels / 2, e2_y + square_pixels / 2)
    assert resolved == chess.E2


def test_board_click_to_move_still_works_at_a_shrunk_size(qtbot):
    """End-to-end: an actual mouse click at the resized board's computed
    square center still produces a legal move -- not just the pure
    coordinate math in isolation above."""
    from PySide6.QtCore import QPoint, Qt

    state = SessionState(chess.pgn.Game())
    panel = BoardPanel(state)
    qtbot.addWidget(panel)
    panel.board_view.resize(MIN_BOARD_PIXELS, MIN_BOARD_PIXELS)

    def _center(square: chess.Square) -> QPoint:
        x, y = panel.board_view._square_top_left(square)
        _offset_x, _offset_y, square_pixels = panel.board_view._board_geometry()
        return QPoint(int(x + square_pixels / 2), int(y + square_pixels / 2))

    qtbot.mouseClick(panel.board_view, Qt.MouseButton.LeftButton, pos=_center(chess.E2))
    assert panel.board_view._selected_square == chess.E2

    with qtbot.waitSignal(panel._session_state.current_node_changed, timeout=1000):
        qtbot.mouseClick(panel.board_view, Qt.MouseButton.LeftButton, pos=_center(chess.E4))

    board = panel._session_state.current_node.board()
    assert board.piece_at(chess.E4) is not None
    assert board.piece_at(chess.E2) is None


def test_board_geometry_letterboxes_within_a_non_square_widget_rect():
    """
    Direct proof the board never distorts even if Qt ever hands it a
    non-square rect within its bounds (compute_square_viewport's own
    contract, reused here) -- mirrors MathCanvas's V0 letterboxing
    guarantee, now shared by the board too.
    """
    state = SessionState(chess.pgn.Game())
    panel = BoardPanel(state)

    panel.board_view.resize(480, 400)  # a non-square rect within [min,max] bounds independently per axis
    offset_x, offset_y, square_pixels = panel.board_view._board_geometry()
    side = square_pixels * 8
    assert side == pytest.approx(400)  # the smaller dimension
    assert offset_x == pytest.approx((480 - 400) / 2)
    assert offset_y == pytest.approx(0)


# ---------------------------------------------------------
# V6a (canvas-collapse fix): real-rendering-only regression coverage
#
# Every test above (and every other test in this suite) runs under the
# project's default QT_QPA_PLATFORM=offscreen -- offscreen's fallback font
# produces compact ~17px checkbox rows and NEVER reproduced the real bug
# this section protects against: real on-screen Windows rendering (Segoe
# UI, ~24px rows, +41%) starved MathCanvas's stretch=1 leftover grid space
# down to as little as 55px tall at 1280x720/1366x768, on code that passed
# every offscreen-based test in this file. Gated on QApplication.
# platformName() rather than canvas.isValid() -- a different concern (GL
# context availability) that happens to correlate with real rendering on
# this project's tested machines but isn't the actual mechanism at fault
# here (font metrics, not GL). Skips with an honest, explicit reason under
# offscreen rather than silently passing on data that structurally cannot
# exercise this bug -- see the class's own module docstring convention in
# tests/test_desktop_app_canvas.py for the same pattern applied to GL.
# ---------------------------------------------------------


def _skip_unless_real_rendering(qapp):
    if qapp.platformName() == "offscreen":
        pytest.skip(
            "Real font-metrics regression test needs a real (non-offscreen) "
            "QPA platform -- offscreen's fallback font never reproduces the "
            "canvas-collapse bug this test protects against. Re-run with "
            "QT_QPA_PLATFORM unset (or set to the platform default) on a "
            "machine with a display to exercise it for real."
        )


def test_canvas_stays_usable_under_real_font_metrics_at_every_target_resolution(qapp, qtbot):
    """
    The core V6a regression test: on the pre-fix implementation, this
    fails at 1280x720 and 1366x768 under real rendering (canvas measured
    as short as 55px tall) while every offscreen-based test in this file
    passed unchanged on the exact same pre-fix code. Also confirms the
    board stays square under real rendering at every resolution -- V7
    (desktop workspace layout) removed board_panel's old rowSpan and
    investigated letting board_view regrow past MIN_BOARD_PIXELS, but
    found no reliable, non-hacky way to keep WIDTH (governed by
    workspace_row's QHBoxLayout) and HEIGHT (governed by BoardPanel's own
    QVBoxLayout) growing in lockstep, so board_view.setMaximumWidth is
    pinned to MIN_BOARD_PIXELS in main_window.py, keeping both axes
    governed identically -- see board_panel.py's sizeHint docstring for
    the full investigation.
    """
    _skip_unless_real_rendering(qapp)

    window = MainWindow()
    window.show()
    for width, height in RESOLUTIONS:
        _resize_and_settle(qapp, window, width, height)
        assert window.canvas.height() >= MIN_BOARD_PIXELS, (
            f"canvas collapsed to {window.canvas.height()}px tall at {width}x{height} "
            "under real rendering"
        )
        board_view = window.board_panel.board_view
        assert board_view.width() == board_view.height(), (
            f"board_view went non-square ({board_view.width()}x{board_view.height()}) "
            f"at {width}x{height} under real rendering"
        )
        assert MIN_BOARD_PIXELS <= board_view.width() <= BOARD_PIXELS


def test_canvas_recovers_after_shrinking_and_regrowing_under_real_rendering(qapp, qtbot):
    """Shrink to the tightest target resolution, then regrow to the
    roomiest -- the canvas must still be at least MIN_BOARD_PIXELS tall
    immediately after shrinking, and grow back to reclaim the newly
    available room, not get stuck at a stale small size."""
    _skip_unless_real_rendering(qapp)

    window = MainWindow()
    window.show()

    _resize_and_settle(qapp, window, 1920, 1080)
    roomy_height = window.canvas.height()
    assert roomy_height >= MIN_BOARD_PIXELS

    _resize_and_settle(qapp, window, 1280, 720)
    shrunk_height = window.canvas.height()
    assert shrunk_height >= MIN_BOARD_PIXELS

    _resize_and_settle(qapp, window, 1920, 1080)
    regrown_height = window.canvas.height()
    assert regrown_height >= MIN_BOARD_PIXELS
    assert regrown_height >= shrunk_height


def test_layer_panel_controls_are_directly_visible_without_any_scroll_gesture(qapp, qtbot):
    """
    V7 (desktop workspace layout) supersedes V6a's fixed-height scroll
    budget for LayerPanel's controls: the preset combo and every layer
    checkbox now live directly in the panel's own layout, never inside a
    QScrollArea, so there is no scroll position that can hide them --
    closing the exact bug a live screenshot exposed (V6a's shared
    controls+legend scroll budget could land on a position showing only
    the legend, with every checkbox/slider/preset combo scrolled out of
    view). This replaces the old
    test_layer_panel_controls_beyond_the_scroll_budget_remain_reachable_and_functional,
    whose entire premise (controls needing ensureWidgetVisible to reach)
    no longer applies.
    """
    from PySide6.QtWidgets import QScrollArea

    window = MainWindow()
    window.show()
    _resize_and_settle(qapp, window, 1280, 720)

    from desktop_app.layers.critical_points_layer import CRITICAL_POINTS_LAYER

    last_checkbox = window.layer_panel._checkboxes[CRITICAL_POINTS_LAYER.id]

    def _is_descendant_of_scroll_area(widget):
        ancestor = widget.parentWidget()
        while ancestor is not None and ancestor is not window.layer_panel:
            if isinstance(ancestor, QScrollArea):
                return True
            ancestor = ancestor.parentWidget()
        return False

    assert not _is_descendant_of_scroll_area(last_checkbox)
    assert not _is_descendant_of_scroll_area(window.layer_panel._preset_combo)
    assert last_checkbox.isVisibleTo(window)

    was_checked = last_checkbox.isChecked()
    last_checkbox.setChecked(not was_checked)
    assert window.session_state.layer_visible(CRITICAL_POINTS_LAYER.id) == (not was_checked)
