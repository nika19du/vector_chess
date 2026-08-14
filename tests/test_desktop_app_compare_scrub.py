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

STRIP_WIDTH = 400


def _wired_panel():
    """Same construction shape as tests/test_desktop_app_timeline_scrub.py's own helper."""
    session_state = SessionState(chess.pgn.Game())
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

    return panel, session_state, scrub_controller


def _siblings(session_state):
    session_state.make_move(chess.Move.from_uci("e2e4"))
    e4_node = session_state.current_node
    session_state.undo()
    session_state.make_move(chess.Move.from_uci("d2d4"))
    d4_node = session_state.current_node
    return e4_node, d4_node


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
# Branch Comparison V1: scrub is disabled outright while comparing
# ---------------------------------------------------------


def test_scrub_press_is_a_no_op_while_comparing(qapp, qtbot):
    panel, session_state, scrub_controller = _wired_panel()
    node_a, node_b = _siblings(session_state)
    session_state.enter_compare(node_a, node_b)
    original_node = session_state.current_node

    _press(panel._scrub_strip, 0)
    _move(panel._scrub_strip, 200)
    _release(panel._scrub_strip, 200)

    assert scrub_controller.is_scrubbing is False
    assert session_state.current_node is original_node
    # And the comparison itself must survive an inert scrub attempt --
    # scrub being disabled must not itself have exited compare mode.
    assert session_state.compare_state is not None
    qtbot.wait(10)


def test_scrub_works_again_immediately_after_exiting_compare(qapp, qtbot):
    panel, session_state, scrub_controller = _wired_panel()
    node_a, node_b = _siblings(session_state)
    session_state.enter_compare(node_a, node_b)
    session_state.exit_compare()

    _press(panel._scrub_strip, 0)

    assert scrub_controller.is_scrubbing is True
    _release(panel._scrub_strip, 0)
    qtbot.wait(10)
