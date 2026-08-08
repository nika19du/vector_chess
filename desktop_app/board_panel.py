from __future__ import annotations

import chess
import chess.pgn
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from desktop_app.session_state import SessionState
from visualization.board_plot import UNICODE_PIECES


# docs/interactive_ui.md Part 9: the mockup's neutral slate board, not
# visualization/board_plot.py's wood tones -- a deliberate signal this is an
# instrument, not a photorealistic chess set. Legal-destination/selection
# marks reuse the tension-gold accent, whose role Part 9 already reserves for
# selection/active-highlight, not board chrome.
LIGHT_SQUARE_COLOR = QColor("#cbd5e1")
DARK_SQUARE_COLOR = QColor("#475569")
SELECTED_SQUARE_OVERLAY = QColor(234, 179, 8, 90)
LEGAL_DESTINATION_DOT = QColor(234, 179, 8, 170)

BOARD_PIXELS = 480
SQUARE_PIXELS = BOARD_PIXELS // 8


class _BoardView(QWidget):
    """
    The interactive 8x8 grid: painting + mouse handling.

    Owns a small pixel<->square mapping local to this widget's own pixel
    space. This is deliberately NOT a reuse of analysis/geometry.py's
    transforms -- those convert to/from matplotlib plot-unit space for the
    math layer's own renderer, a different coordinate space entirely. This
    widget keeps the same rank-8-at-top visual convention for consistency,
    but the concrete transform is necessarily its own.

    Click-to-move and drag-and-drop are the same code path on purpose: a
    press selects a piece (if nothing is already selected); a release
    decides what to do with the currently selected square. Whether the press
    and release happen as two separate clicks or one continuous drag makes
    no difference to this logic -- there is no separate "drag" state machine
    to keep in sync with the click one, and no floating drag-follow visual
    (that is animation, out of Phase 5b's scope).
    """

    def __init__(self, session_state: SessionState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session_state = session_state
        self._board_node = session_state.current_node
        self._selected_square: chess.Square | None = None
        self._legal_destinations: set[chess.Square] = set()

        self.setFixedSize(BOARD_PIXELS, BOARD_PIXELS)
        session_state.current_node_changed.connect(self._on_current_node_changed)

    def _on_current_node_changed(self, node: chess.pgn.GameNode) -> None:
        self._board_node = node
        self._clear_selection()
        self.update()

    def _pixel_to_square(self, x: float, y: float) -> chess.Square | None:
        if not (0 <= x < BOARD_PIXELS and 0 <= y < BOARD_PIXELS):
            return None
        file_index = int(x // SQUARE_PIXELS)
        visual_row = int(y // SQUARE_PIXELS)  # 0 = top of widget = rank 8
        rank_index = 7 - visual_row
        return chess.square(file_index, rank_index)

    def _square_top_left(self, square: chess.Square) -> tuple[int, int]:
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        visual_row = 7 - rank_index
        return file_index * SQUARE_PIXELS, visual_row * SQUARE_PIXELS

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        board = self._board_node.board()

        for rank_index in range(8):
            for file_index in range(8):
                x, y = self._square_top_left(chess.square(file_index, rank_index))
                is_light = (rank_index + file_index) % 2 == 1
                color = LIGHT_SQUARE_COLOR if is_light else DARK_SQUARE_COLOR
                painter.fillRect(x, y, SQUARE_PIXELS, SQUARE_PIXELS, color)

        if self._selected_square is not None:
            x, y = self._square_top_left(self._selected_square)
            painter.fillRect(x, y, SQUARE_PIXELS, SQUARE_PIXELS, SELECTED_SQUARE_OVERLAY)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(LEGAL_DESTINATION_DOT)
        radius = SQUARE_PIXELS * 0.12
        for square in self._legal_destinations:
            x, y = self._square_top_left(square)
            center_x, center_y = x + SQUARE_PIXELS / 2, y + SQUARE_PIXELS / 2
            painter.drawEllipse(QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2))

        font = QFont()
        font.setPointSize(int(SQUARE_PIXELS * 0.5))
        painter.setFont(font)
        painter.setPen(Qt.GlobalColor.black)
        for square, piece in board.piece_map().items():
            x, y = self._square_top_left(square)
            symbol = UNICODE_PIECES[piece.symbol()]
            painter.drawText(QRectF(x, y, SQUARE_PIXELS, SQUARE_PIXELS), Qt.AlignmentFlag.AlignCenter, symbol)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        square = self._pixel_to_square(event.position().x(), event.position().y())
        if square is None or self._selected_square is not None:
            return

        board = self._board_node.board()
        piece = board.piece_at(square)
        if piece is not None and piece.color == board.turn:
            self._select(square, board)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        square = self._pixel_to_square(event.position().x(), event.position().y())
        if square is None or self._selected_square is None:
            return

        if square == self._selected_square:
            return  # a plain click on the square that was just selected

        board = self._board_node.board()

        if square in self._legal_destinations:
            move = self._build_move(self._selected_square, square, board)
            details = self._session_state.make_move(move)
            if details is None:
                # Rejected by execute_move despite passing our own
                # highlight computation -- should not happen since both use
                # the same board.legal_moves, but fail safe rather than
                # leave a stale selection.
                self._clear_selection()
            # On success, _on_current_node_changed already clears selection.
            self.update()
            return

        piece = board.piece_at(square)
        if piece is not None and piece.color == board.turn:
            self._select(square, board)
        else:
            self._clear_selection()
        self.update()

    def _select(self, square: chess.Square, board: chess.Board) -> None:
        self._selected_square = square
        self._legal_destinations = {
            move.to_square for move in board.legal_moves if move.from_square == square
        }
        self.update()

    def _clear_selection(self) -> None:
        self._selected_square = None
        self._legal_destinations = set()

    @staticmethod
    def _build_move(from_square: chess.Square, to_square: chess.Square, board: chess.Board) -> chess.Move:
        move = chess.Move(from_square, to_square)
        if move not in board.legal_moves:
            promoted = chess.Move(from_square, to_square, promotion=chess.QUEEN)
            if promoted in board.legal_moves:
                return promoted
        return move


class BoardPanel(QWidget):
    """The left panel (docs/interactive_ui.md Part 2): side-to-move label + the interactive board."""

    def __init__(self, session_state: SessionState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session_state = session_state

        self.side_to_move_label = QLabel(self)
        self.board_view = _BoardView(session_state, self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.side_to_move_label)
        layout.addWidget(self.board_view)
        layout.addStretch(1)
        self.setLayout(layout)

        self._update_side_to_move_label(session_state.current_node)
        session_state.current_node_changed.connect(self._update_side_to_move_label)

    def _update_side_to_move_label(self, node: chess.pgn.GameNode) -> None:
        board = node.board()
        self.side_to_move_label.setText("White to move" if board.turn == chess.WHITE else "Black to move")
