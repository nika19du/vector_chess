import chess
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from desktop_app.layers.attack_influence_layer import ATTACK_INFLUENCE_LAYER
from desktop_app.main_window import MainWindow
from tests.conftest import ready_cache_entry

STRIP_WIDTH = 400


def _expected_vertex_colors(board: chess.Board):
    entry = ready_cache_entry(board)
    frame = ATTACK_INFLUENCE_LAYER.data_source(entry)
    return ATTACK_INFLUENCE_LAYER.renderer(frame)


def _mouse_event(kind, x: float, y: float, button, buttons) -> QMouseEvent:
    local = QPointF(x, y)
    return QMouseEvent(kind, local, local, button, buttons, Qt.KeyboardModifier.NoModifier)


def _press(strip, x: float, y: float = 5.0) -> None:
    strip.mousePressEvent(
        _mouse_event(QEvent.Type.MouseButtonPress, x, y, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    )


def _move(strip, x: float, y: float = 5.0) -> None:
    strip.mouseMoveEvent(
        _mouse_event(QEvent.Type.MouseMove, x, y, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    )


def _release(strip, x: float, y: float = 5.0) -> None:
    strip.mouseReleaseEvent(
        _mouse_event(QEvent.Type.MouseButtonRelease, x, y, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    )


def _ready_window(qtbot, moves: list[str]) -> MainWindow:
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    for uci in moves:
        window.session_state.make_move(chess.Move.from_uci(uci))
        node = window.session_state.current_node
        expected = _expected_vertex_colors(node.board())
        qtbot.waitUntil(
            lambda expected=expected: window.canvas._overlay_colors is not None
            and (window.canvas._overlay_colors == expected).all(),
            timeout=5000,
        )
        # Rule 6 (a scrub press is ignored while an ordinary move animation
        # is running) needs the animation fully settled, not merely visually
        # converged -- overlay colors can already match near the tail of the
        # ~500ms animation while transition_controller.is_animating is still
        # True.
        qtbot.waitUntil(lambda: not window.transition_controller.is_animating, timeout=5000)
    window.timeline_panel._scrub_strip.resize(STRIP_WIDTH, window.timeline_panel._scrub_strip.FIXED_HEIGHT)
    qtbot.wait(10)
    return window


# ---------------------------------------------------------
# full MainWindow scrub: canvas updates, current_node stays put mid-drag
# ---------------------------------------------------------


def test_scrub_updates_canvas_without_committing_current_node(qapp, qtbot):
    window = _ready_window(qtbot, ["e2e4", "e7e5", "g1f3"])
    strip = window.timeline_panel._scrub_strip
    original_node = window.session_state.current_node

    _press(strip, 0)
    _move(strip, STRIP_WIDTH // 2)
    qtbot.wait(10)  # drain the scrub strip's coalesced frame

    assert window.session_state.current_node is original_node
    assert window.board_panel.board_view._preview_node is not None

    _release(strip, STRIP_WIDTH // 2)
    qtbot.wait(10)


def test_scrub_release_commits_and_final_frame_matches_a_static_render(qapp, qtbot):
    window = _ready_window(qtbot, ["e2e4", "e7e5", "g1f3", "b8c6"])
    strip = window.timeline_panel._scrub_strip
    target_node = window.session_state.mainline_path()[2]  # e7e5

    _press(strip, 0)
    with qtbot.waitSignal(window.session_state.current_node_changed, timeout=1000):
        _release(strip, STRIP_WIDTH * 2 // 4)  # segment (e4, e5), lands past the midpoint -> e5

    assert window.session_state.current_node is target_node

    expected = _expected_vertex_colors(target_node.board())
    qtbot.waitUntil(
        lambda: window.canvas._overlay_colors is not None and (window.canvas._overlay_colors == expected).all(),
        timeout=5000,
    )
    assert window.board_panel.board_view._preview_node is None
    qtbot.wait(10)


# ---------------------------------------------------------
# branch switch mid-drag cancels the scrub
# ---------------------------------------------------------


def test_branch_switch_during_scrub_cancels_it(qapp, qtbot):
    window = _ready_window(qtbot, ["e2e4"])
    strip = window.timeline_panel._scrub_strip

    old_branch_node = window.session_state.current_node
    window.timeline_panel._previous_button.click()
    qtbot.wait(10)
    with qtbot.waitSignal(window.session_state.current_node_changed, timeout=1000):
        window.session_state.make_move(chess.Move.from_uci("d2d4"))  # a new branch
    qtbot.wait(10)
    # Rule 6: a scrub press is ignored while an ordinary move animation is
    # still running -- wait for the move's own transition to settle first,
    # exactly as a real drag beginning shortly after a move would need to.
    qtbot.waitUntil(lambda: not window.transition_controller.is_animating, timeout=5000)

    _press(strip, 0)
    assert window.scrub_controller.is_scrubbing is True
    _move(strip, 50)
    qtbot.wait(10)

    # Switch back into the old branch via SessionState directly (equivalent
    # to a Timeline branch-badge click) while the scrub is still in progress.
    with qtbot.waitSignal(window.session_state.current_node_changed, timeout=1000):
        window.session_state.set_current_node(old_branch_node)

    assert window.scrub_controller.is_scrubbing is False
    assert window.board_panel.board_view._preview_node is None
    qtbot.wait(10)


# ---------------------------------------------------------
# clean shutdown with a scrub mid-drag
# ---------------------------------------------------------


def test_close_event_mid_drag_shuts_down_cleanly(qapp, qtbot):
    window = _ready_window(qtbot, ["e2e4", "e7e5"])
    strip = window.timeline_panel._scrub_strip

    _press(strip, 0)
    _move(strip, STRIP_WIDTH // 2)  # a scrub frame + a rebuild-cancel path is now pending

    window.close()  # must not crash and must fully shut down both controllers

    assert window.position_cache._shutdown_requested.is_set()
    assert window.scrub_controller.is_scrubbing is False
    qtbot.wait(10)


# ---------------------------------------------------------
# stress: rapid scrubbing across the whole path, back and forth, many times
# ---------------------------------------------------------


def test_repeated_rapid_scrub_stress_does_not_crash_or_duplicate_work(qapp, qtbot):
    window = _ready_window(qtbot, ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"])
    strip = window.timeline_panel._scrub_strip

    request_count = {"n": 0}
    real_request = window.position_cache.request

    def counting_request(board):
        request_count["n"] += 1
        return real_request(board)

    window.position_cache.request = counting_request

    _press(strip, 0)
    for _ in range(3):
        for x in range(0, STRIP_WIDTH + 1, 17):  # sweep forward
            _move(strip, x)
        for x in range(STRIP_WIDTH, -1, -17):  # sweep backward
            _move(strip, x)
    qtbot.wait(20)
    _release(strip, STRIP_WIDTH // 2)
    qtbot.wait(20)

    # request() is idempotent per fen -- every node in the 6-node path was
    # already visited (and thus already READY) before scrubbing began, so
    # this sweep must not trigger unbounded/duplicate background work.
    all_path_fens = {node.board().board_fen() for node in window.session_state.mainline_path()}
    assert request_count["n"] > 0  # prefetch did run
    for fen in all_path_fens:
        assert window.position_cache.get(fen).state.name == "READY"

    assert window.scrub_controller.is_scrubbing is False


# ---------------------------------------------------------
# existing discrete Timeline behavior is unaffected
# ---------------------------------------------------------


def test_discrete_navigation_still_works_unchanged_alongside_scrubbing(qapp, qtbot):
    window = _ready_window(qtbot, ["e2e4", "e7e5"])
    root = window.session_state.mainline_path()[0]

    window.timeline_panel._first_button.click()
    qtbot.wait(10)

    assert window.session_state.current_node is root
    qtbot.wait(10)
