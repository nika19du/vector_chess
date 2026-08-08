import chess
import chess.pgn
from PySide6.QtCore import QPoint, Qt

from desktop_app.board_panel import SQUARE_PIXELS, BoardPanel
from desktop_app.session_state import SessionState


def _make_panel(qtbot) -> BoardPanel:
    state = SessionState(chess.pgn.Game())
    panel = BoardPanel(state)
    qtbot.addWidget(panel)
    return panel


def _center(panel: BoardPanel, square: chess.Square) -> QPoint:
    x, y = panel.board_view._square_top_left(square)
    return QPoint(x + SQUARE_PIXELS // 2, y + SQUARE_PIXELS // 2)


# ---------------------------------------------------------
# click-to-move
# ---------------------------------------------------------


def test_click_to_move_plays_a_legal_move_via_two_separate_clicks(qtbot):
    panel = _make_panel(qtbot)
    view = panel.board_view

    qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=_center(panel, chess.E2))
    assert view._selected_square == chess.E2

    with qtbot.waitSignal(panel._session_state.current_node_changed, timeout=1000):
        qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=_center(panel, chess.E4))

    board = panel._session_state.current_node.board()
    assert board.piece_at(chess.E4) is not None
    assert board.piece_at(chess.E2) is None
    assert view._selected_square is None  # cleared after a successful move


def test_clicking_a_different_own_piece_reselects_instead_of_attempting_a_move(qtbot):
    panel = _make_panel(qtbot)
    view = panel.board_view

    qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=_center(panel, chess.E2))
    assert view._selected_square == chess.E2

    qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=_center(panel, chess.D2))

    assert view._selected_square == chess.D2
    assert chess.D4 in view._legal_destinations


# ---------------------------------------------------------
# drag-and-drop
# ---------------------------------------------------------


def test_drag_and_drop_plays_a_legal_move_via_press_and_release(qtbot):
    panel = _make_panel(qtbot)
    view = panel.board_view

    with qtbot.waitSignal(panel._session_state.current_node_changed, timeout=1000):
        qtbot.mousePress(view, Qt.MouseButton.LeftButton, pos=_center(panel, chess.G1))
        qtbot.mouseRelease(view, Qt.MouseButton.LeftButton, pos=_center(panel, chess.F3))

    board = panel._session_state.current_node.board()
    assert board.piece_at(chess.F3) is not None
    assert board.piece_at(chess.F3).piece_type == chess.KNIGHT
    assert board.piece_at(chess.G1) is None


def test_dragging_to_a_non_legal_square_does_not_move_the_piece(qtbot):
    panel = _make_panel(qtbot)
    view = panel.board_view
    original_fen = panel._session_state.current_node.board().board_fen()

    received = []
    panel._session_state.current_node_changed.connect(lambda node: received.append(node))

    qtbot.mousePress(view, Qt.MouseButton.LeftButton, pos=_center(panel, chess.G1))
    qtbot.mouseRelease(view, Qt.MouseButton.LeftButton, pos=_center(panel, chess.G3))
    qtbot.wait(50)

    assert received == []
    assert panel._session_state.current_node.board().board_fen() == original_fen
    assert view._selected_square is None


# ---------------------------------------------------------
# legal-square highlighting
# ---------------------------------------------------------


def test_selecting_a_pawn_highlights_both_of_its_opening_destinations(qtbot):
    panel = _make_panel(qtbot)
    view = panel.board_view

    qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=_center(panel, chess.E2))

    assert view._legal_destinations == {chess.E3, chess.E4}


def test_selecting_a_piece_with_no_legal_moves_highlights_nothing(qtbot):
    panel = _make_panel(qtbot)
    view = panel.board_view

    # The a1 rook is boxed in at the start of the game -- pressing it still
    # selects it (selection only checks whose piece it is), but there is
    # nothing to highlight since it has no legal destinations.
    qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=_center(panel, chess.A1))

    assert view._selected_square == chess.A1
    assert view._legal_destinations == set()


# ---------------------------------------------------------
# side-to-move display
# ---------------------------------------------------------


def test_side_to_move_label_starts_as_white(qtbot):
    panel = _make_panel(qtbot)
    assert panel.side_to_move_label.text() == "White to move"


def test_side_to_move_label_updates_after_a_move(qtbot):
    panel = _make_panel(qtbot)
    view = panel.board_view

    qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=_center(panel, chess.E2))
    with qtbot.waitSignal(panel._session_state.current_node_changed, timeout=1000):
        qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=_center(panel, chess.E4))

    assert panel.side_to_move_label.text() == "Black to move"


# ---------------------------------------------------------
# illegal-move rejection
# ---------------------------------------------------------


def test_clicking_a_non_legal_empty_square_deselects_without_moving(qtbot):
    panel = _make_panel(qtbot)
    view = panel.board_view
    original_fen = panel._session_state.current_node.board().board_fen()

    received = []
    panel._session_state.current_node_changed.connect(lambda node: received.append(node))

    qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=_center(panel, chess.E2))
    qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=_center(panel, chess.E5))  # not e3/e4
    qtbot.wait(50)

    assert received == []
    assert panel._session_state.current_node.board().board_fen() == original_fen
    assert view._selected_square is None


def test_the_board_repaints_to_reflect_undo(qtbot):
    panel = _make_panel(qtbot)
    view = panel.board_view
    state = panel._session_state

    qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=_center(panel, chess.E2))
    with qtbot.waitSignal(state.current_node_changed, timeout=1000):
        qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=_center(panel, chess.E4))

    with qtbot.waitSignal(state.current_node_changed, timeout=1000):
        state.undo()

    board = view._board_node.board()
    assert board.piece_at(chess.E2) is not None
    assert board.piece_at(chess.E4) is None
