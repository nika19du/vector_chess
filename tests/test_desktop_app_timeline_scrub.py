import chess
import chess.pgn
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from desktop_app.board_panel import BoardPanel
from desktop_app.gl_canvas import MathCanvas
from desktop_app.position_cache import PositionCache
from desktop_app.scrub_controller import ScrubController
from desktop_app.session_state import SessionState
from desktop_app.timeline_panel import TimelinePanel
from desktop_app.transition_controller import TransitionController

STRIP_WIDTH = 400  # divisible cleanly by common segment counts used below

# Every test in this file drains pending Qt timers with qtbot.wait(10)
# before returning -- see test_desktop_app_timeline_panel.py's own
# convention/comment: none of the objects built here (panel, scrub_controller,
# transition_controller, ...) have a Qt parent (unlike production's
# MainWindow-owned instances), so a still-pending QTimer.singleShot (the
# TimelinePanel deferred-rebuild guard, or ScrubController's own coalesced-
# frame guard) left scheduled when the test function returns would fire
# against an already-Python-GC'd object during pytest-qt's teardown event
# pump -- a real, reproducible native crash in this codebase (see
# tests/conftest.py), not merely test hygiene.


def _wired_panel(moves: list[str]):
    """
    A TimelinePanel with a real, fully wired ScrubController/BoardPanel/
    TransitionController -- not the full MainWindow (that's Phase 6's own
    integration test scope) -- exercising the same construction shape
    MainWindow will use.
    """
    session_state = SessionState(chess.pgn.Game())
    for uci in moves:
        session_state.make_move(chess.Move.from_uci(uci))

    position_cache = PositionCache()
    canvas = MathCanvas()
    board_panel = BoardPanel(session_state)
    transition_controller = TransitionController(
        canvas=canvas,
        session_state=session_state,
        position_cache=position_cache,
        on_settled=lambda fen: None,
    )
    scrub_controller = ScrubController(canvas=canvas, position_cache=position_cache)

    panel = TimelinePanel(
        session_state,
        None,
        board_panel=board_panel,
        transition_controller=transition_controller,
        scrub_controller=scrub_controller,
    )
    panel._scrub_strip.resize(STRIP_WIDTH, panel._scrub_strip.FIXED_HEIGHT)

    return panel, session_state, board_panel, transition_controller, scrub_controller


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


# ---------------------------------------------------------
# inert without a ScrubController (pre-5e.2 discrete-only construction)
# ---------------------------------------------------------


def test_scrub_strip_is_inert_without_a_scrub_controller(qapp, qtbot):
    session_state = SessionState(chess.pgn.Game())
    session_state.make_move(chess.Move.from_uci("e2e4"))
    panel = TimelinePanel(session_state, None)  # unchanged discrete-only shape
    panel._scrub_strip.resize(STRIP_WIDTH, panel._scrub_strip.FIXED_HEIGHT)
    original_node = session_state.current_node

    _press(panel._scrub_strip, 0)
    _move(panel._scrub_strip, 200)
    _release(panel._scrub_strip, 200)

    assert session_state.current_node is original_node  # nothing happened, no crash
    qtbot.wait(10)


# ---------------------------------------------------------
# rule 6: press while an ordinary move animation is running is ignored
# ---------------------------------------------------------


def test_press_while_animating_is_ignored(qapp, qtbot):
    panel, session_state, board_panel, transition_controller, scrub_controller = _wired_panel(["e2e4", "e7e5"])
    original_node = session_state.current_node

    transition_controller._timer.start()  # simulate an in-flight animation
    try:
        _press(panel._scrub_strip, 0)

        assert scrub_controller.is_scrubbing is False
        assert session_state.current_node is original_node
    finally:
        transition_controller._timer.stop()
    qtbot.wait(10)


# ---------------------------------------------------------
# drag updates the board preview without touching current_node
# ---------------------------------------------------------


def test_drag_updates_preview_without_committing_current_node(qapp, qtbot):
    panel, session_state, board_panel, transition_controller, scrub_controller = _wired_panel(
        ["e2e4", "e7e5", "g1f3"]
    )
    original_node = session_state.current_node  # active_path has 4 nodes -> 3 segments

    _press(panel._scrub_strip, 0)
    assert scrub_controller.is_scrubbing is True

    _move(panel._scrub_strip, STRIP_WIDTH // 2)  # partway through the path

    assert session_state.current_node is original_node  # never mutated mid-drag
    assert board_panel.board_view._preview_node is not None  # preview reflects the drag

    # End the drag cleanly (a real release) rather than leaving it mid-flight
    # -- both drains any scheduled scrub frame and returns everything to
    # idle before local references go out of scope.
    _release(panel._scrub_strip, STRIP_WIDTH // 2)
    qtbot.wait(10)


def test_pressing_with_fewer_than_two_path_nodes_is_a_no_op(qapp, qtbot):
    panel, session_state, board_panel, transition_controller, scrub_controller = _wired_panel([])  # root only

    _press(panel._scrub_strip, 0)

    assert scrub_controller.is_scrubbing is False
    qtbot.wait(10)


# ---------------------------------------------------------
# release commits exactly once, via set_current_node
# ---------------------------------------------------------


def test_release_commits_the_snapped_node_via_a_single_set_current_node_call(qapp, qtbot):
    panel, session_state, board_panel, transition_controller, scrub_controller = _wired_panel(
        ["e2e4", "e7e5", "g1f3", "b8c6"]
    )
    # active_path(): root, e4, e5, Nf3, Nc6 -- 5 nodes, 4 segments of 100px each.
    received = []
    session_state.current_node_changed.connect(lambda node: received.append(node))

    _press(panel._scrub_strip, 0)
    _move(panel._scrub_strip, 150)  # segment 1 (e4..e5), t=0.5 -> snaps to e5 (t>=0.5 rule)

    with qtbot.waitSignal(session_state.current_node_changed, timeout=1000):
        _release(panel._scrub_strip, 150)

    assert len(received) == 1  # exactly one commit, not one per mousemove
    assert scrub_controller.is_scrubbing is False
    assert board_panel.board_view._preview_node is None
    assert session_state.current_node.move == chess.Move.from_uci("e7e5")
    qtbot.wait(10)


def test_release_at_the_very_start_commits_the_root(qapp, qtbot):
    panel, session_state, board_panel, transition_controller, scrub_controller = _wired_panel(["e2e4", "e7e5"])
    root = session_state.mainline_path()[0]  # current_node is the tip (e7e5); root differs -> a real commit

    _press(panel._scrub_strip, 0)
    with qtbot.waitSignal(session_state.current_node_changed, timeout=1000):
        _release(panel._scrub_strip, 0)  # x=0 -> path_index=0, t=0.0 -> root

    assert session_state.current_node is root
    qtbot.wait(10)


# ---------------------------------------------------------
# branch switch (or any external navigation) mid-drag cancels the scrub
# ---------------------------------------------------------


def test_external_navigation_during_drag_cancels_the_scrub(qapp, qtbot):
    panel, session_state, board_panel, transition_controller, scrub_controller = _wired_panel(
        ["e2e4", "e7e5", "g1f3"]
    )

    _press(panel._scrub_strip, 0)
    assert scrub_controller.is_scrubbing is True
    _move(panel._scrub_strip, 100)
    assert board_panel.board_view._preview_node is not None
    qtbot.wait(10)  # drain the scrub frame this move scheduled before navigating externally

    # Externally-driven navigation mid-drag (e.g. a branch switch elsewhere).
    with qtbot.waitSignal(session_state.current_node_changed, timeout=1000):
        session_state.undo()

    assert scrub_controller.is_scrubbing is False
    assert board_panel.board_view._preview_node is None
    qtbot.wait(10)


def test_full_active_branch_is_scrubbable_past_current_node(qapp, qtbot):
    # Resolved scope: scrubbing covers the full active branch (root..tip),
    # not just root..current_node -- undo once, then confirm the strip's
    # path still reaches the tip via _active_child.
    panel, session_state, board_panel, transition_controller, scrub_controller = _wired_panel(
        ["e2e4", "e7e5", "g1f3"]
    )
    tip = session_state.current_node
    with qtbot.waitSignal(session_state.current_node_changed, timeout=1000):
        session_state.undo()  # current_node is now e7e5; g1f3 (tip) is still active-forward

    _press(panel._scrub_strip, 0)

    assert scrub_controller.path_length == len(session_state.active_path())
    assert scrub_controller.path_length == 4  # root, e4, e5, Nf3(tip)

    with qtbot.waitSignal(session_state.current_node_changed, timeout=1000):
        _release(panel._scrub_strip, STRIP_WIDTH)  # drag all the way to the far end -> the tip

    assert session_state.current_node is tip
    qtbot.wait(10)
