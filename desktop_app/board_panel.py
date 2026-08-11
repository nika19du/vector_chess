from __future__ import annotations

import chess
import chess.pgn
from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from desktop_app.gl_canvas import compute_square_viewport
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

# V5 (responsive layout): BOARD_PIXELS is now a MAXIMUM, not a fixed size --
# measured (the real remaining blocker at 1280x720 after every pure spacing
# trim available elsewhere was exhausted) that the board's own fixed 480x480
# size was the single largest contributor to MainWindow's true minimum
# height. MIN_BOARD_PIXELS=400 (50px/square) is a bounded floor, not
# unbounded shrinking -- chosen as the smallest reduction that closes the
# measured gap (see tests/test_desktop_app_responsive_layout.py), still
# comfortably playable. Larger windows are unaffected: nothing here changes
# how big the board gets when there's room for the full 480.
BOARD_PIXELS = 480
MIN_BOARD_PIXELS = 400
# The square size at the widget's default/maximum size -- correct for any
# caller (tests included) that never resizes the board below BOARD_PIXELS,
# which is every context except MainWindow's real, space-constrained grid.
# paintEvent/_pixel_to_square/_square_top_left below never read this
# constant themselves; they always compute the *actual* current square size
# fresh from the widget's real dimensions (see _board_geometry), since that
# can now be smaller once a real layout constrains it.
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

    Phase 5e.2 (continuous Timeline scrubbing) adds `set_preview_node`, an
    explicit, narrow override of what gets *painted* while the user is
    dragging the scrub strip -- it never touches `SessionState.current_node`
    and never touches chess state. The ownership split is deliberate and
    total, not merely a convention: `paintEvent` is the only place that ever
    reads `_preview_node`; `mousePressEvent`, `mouseReleaseEvent`, `_select`,
    and `_build_move` all read `_board_node` unconditionally, exactly as
    before this phase. A preview node can therefore never become the basis
    for a chess move, a legal-destination highlight, or a drag/drop target
    -- those code paths simply never look at `_preview_node`, so there is no
    runtime guard to bypass or forget.
    """

    def __init__(self, session_state: SessionState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session_state = session_state
        self._board_node = session_state.current_node
        # Phase 5e.2: paint-only override, set by TimelinePanel's scrub strip
        # via BoardPanel.set_preview_node. None means "no active preview,
        # paint _board_node as usual" -- the default, and the only state
        # outside of an in-progress drag.
        self._preview_node: chess.pgn.GameNode | None = None
        self._selected_square: chess.Square | None = None
        self._legal_destinations: set[chess.Square] = set()

        # V5 (responsive layout): bounded, not fixed -- Expanding lets
        # MainWindow's grid shrink this down to MIN_BOARD_PIXELS when a
        # small window genuinely needs the room (see BOARD_PIXELS's own
        # comment), and grow back up to BOARD_PIXELS otherwise. The
        # explicit resize() gives every OTHER caller (every existing test
        # that constructs a bare BoardPanel with no real layout managing
        # it) the exact same default 480x480 starting size as before this
        # milestone -- min/max/policy only take effect once a real layout
        # actually constrains the widget.
        self.setMinimumSize(MIN_BOARD_PIXELS, MIN_BOARD_PIXELS)
        self.setMaximumSize(BOARD_PIXELS, BOARD_PIXELS)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.resize(BOARD_PIXELS, BOARD_PIXELS)
        session_state.current_node_changed.connect(self._on_current_node_changed)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        """
        A plain QWidget subclass with no internal layout defaults to an
        *invalid* (-1, -1) sizeHint, which silently defeats Qt's
        stretch-based "grow toward my maximum when there's room"
        distribution in the surrounding QVBoxLayout/QGridLayout even with
        an explicit stretch factor and Expanding policy both set -- this
        override exists to give the layout engine a real value instead.

        V7 (desktop workspace layout): investigated and reverted back to
        MIN_BOARD_PIXELS. V7 removes board_panel's rowSpan entirely (it is
        now an ordinary, non-spanning sibling of canvas and layer_panel in
        one QHBoxLayout row), which does let WIDTH grow past the old floor
        for the first time (confirmed via direct offscreen measurement).
        But WIDTH and HEIGHT are assigned by two genuinely independent
        nested layout passes (WIDTH via workspace_row's QHBoxLayout column
        allocation in main_window.py; HEIGHT via BoardPanel's own
        QVBoxLayout main-axis stretch), and nothing couples them: measured
        result was a non-square board (480x422 at 1366x768). Qt's
        heightForWidth/hasHeightForWidth mechanism was tried as the
        sanctioned, non-hacky coupling tool and measured to have *zero*
        effect -- Qt's box-layout stretch redistribution grows an Expanding
        item toward its own fixed setMaximumSize using leftover space,
        independent of heightForWidth (which only informs preferred/minimum
        sizing, not that redistribution), so it cannot fix this. Solving it
        properly would require a resizeEvent-driven or other manual
        geometry coupling between two sibling layouts, which the project's
        explicit constraints rule out. Reverted to MIN_BOARD_PIXELS on both
        this sizeHint and BoardPanel's own stretch (back to 0, see below) --
        the same guaranteed-square-by-both-axes-pinned-at-the-floor
        behavior as before V7 -- rather than ship a widget that can violate
        the existing width==height invariant at some window sizes. Flagged
        to the user as a V7 finding, not silently worked around.
        """
        return QSize(MIN_BOARD_PIXELS, MIN_BOARD_PIXELS)

    def _board_geometry(self) -> tuple[float, float, float]:
        """
        (offset_x, offset_y, square_pixels) for the largest centered square
        that fits this widget's *actual* current rect -- reuses
        MathCanvas's own compute_square_viewport (a plain, GL-independent
        (x, y, side, side) centering computation) rather than duplicating
        the same arithmetic a second time. The board widget itself need not
        be square (Qt's layout doesn't guarantee it at every intermediate
        size within the bounds set in __init__); painting and hit-testing
        both derive from this single method, so they can never disagree
        with each other.
        """
        offset_x, offset_y, side, _ = compute_square_viewport(self.width(), self.height())
        return float(offset_x), float(offset_y), side / 8

    def _on_current_node_changed(self, node: chess.pgn.GameNode) -> None:
        self._board_node = node
        # Defensive: a committed navigation always supersedes any preview,
        # even if a caller committed without first clearing it explicitly.
        self._preview_node = None
        self._clear_selection()
        self.update()

    def set_preview_node(self, node: chess.pgn.GameNode | None) -> None:
        """
        Phase 5e.2: overrides the painted board only, without touching
        `SessionState.current_node` or `_board_node`. `None` reverts to the
        normal `_board_node`-driven display. See the class docstring's
        "preview vs input" split -- painting is the only consumer of this.
        """
        self._preview_node = node
        self._clear_selection()
        self.update()

    def _pixel_to_square(self, x: float, y: float) -> chess.Square | None:
        offset_x, offset_y, square_pixels = self._board_geometry()
        local_x, local_y = x - offset_x, y - offset_y
        side = square_pixels * 8
        if not (0 <= local_x < side and 0 <= local_y < side):
            return None
        file_index = int(local_x // square_pixels)
        visual_row = int(local_y // square_pixels)  # 0 = top of the square board = rank 8
        rank_index = 7 - visual_row
        return chess.square(file_index, rank_index)

    def _square_top_left(self, square: chess.Square) -> tuple[float, float]:
        offset_x, offset_y, square_pixels = self._board_geometry()
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        visual_row = 7 - rank_index
        return offset_x + file_index * square_pixels, offset_y + visual_row * square_pixels

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        _offset_x, _offset_y, square_pixels = self._board_geometry()
        # The only place _preview_node is ever read -- see the class
        # docstring's "preview vs input" split.
        display_node = self._preview_node if self._preview_node is not None else self._board_node
        board = display_node.board()

        for rank_index in range(8):
            for file_index in range(8):
                x, y = self._square_top_left(chess.square(file_index, rank_index))
                is_light = (rank_index + file_index) % 2 == 1
                color = LIGHT_SQUARE_COLOR if is_light else DARK_SQUARE_COLOR
                painter.fillRect(QRectF(x, y, square_pixels, square_pixels), color)

        if self._selected_square is not None:
            x, y = self._square_top_left(self._selected_square)
            painter.fillRect(QRectF(x, y, square_pixels, square_pixels), SELECTED_SQUARE_OVERLAY)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(LEGAL_DESTINATION_DOT)
        radius = square_pixels * 0.12
        for square in self._legal_destinations:
            x, y = self._square_top_left(square)
            center_x, center_y = x + square_pixels / 2, y + square_pixels / 2
            painter.drawEllipse(QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2))

        font = QFont()
        font.setPointSize(max(1, int(square_pixels * 0.5)))
        painter.setFont(font)
        painter.setPen(Qt.GlobalColor.black)
        for square, piece in board.piece_map().items():
            x, y = self._square_top_left(square)
            symbol = UNICODE_PIECES[piece.symbol()]
            painter.drawText(QRectF(x, y, square_pixels, square_pixels), Qt.AlignmentFlag.AlignCenter, symbol)

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
        # V5 (responsive layout): BoardPanel is the widget MainWindow's grid
        # actually places in column 0 -- a grid column only gives extra
        # space to what a *widget's own* sizePolicy asks for, not to
        # whatever its internal children want, so this wrapper needs its
        # own Expanding policy too, on top of board_view's, or board_view's
        # own Expanding/bounded setup below never receives more than its
        # measured floor even with a nonzero column stretch.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.side_to_move_label = QLabel(self)
        self.board_view = _BoardView(session_state, self)

        layout = QVBoxLayout(self)
        # V5 (responsive layout): pure spacing/margin trim, same rationale
        # as main_window.py's grid -- no change to the label/board
        # themselves, only the default 9px margins and 6px spacing between
        # them.
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)
        layout.addWidget(self.side_to_move_label)
        # V6a (canvas-collapse fix), retained under V7: stretch=0. V7
        # removed board_panel's old rowSpan (see main_window.py), which was
        # investigated as a way to also unblock WIDTH growth toward
        # BOARD_PIXELS -- confirmed via direct offscreen measurement that
        # WIDTH growth does work again once the rowSpan is gone, but WIDTH
        # (governed by workspace_row's QHBoxLayout column allocation) and
        # HEIGHT (governed by this QVBoxLayout's own stretch, if nonzero)
        # are two independent layout computations with nothing coupling
        # them -- letting both grow produced a real, measured non-square
        # regression (480x422 at 1366x768). See _BoardView.sizeHint's
        # docstring for why the sanctioned Qt fix (heightForWidth) was
        # tried and found to have no effect, and why a resizeEvent-based
        # coupling was ruled out. stretch=0 keeps HEIGHT pinned at
        # MIN_BOARD_PIXELS (matching this file's own sizeHint override),
        # the same axis WIDTH is *also* pinned at in practice today via
        # this same board_view instance's bounds -- both axes governed
        # identically, guaranteed square, at the cost of not regrowing past
        # the floor. The trailing addStretch(1) absorbs 100% of any extra
        # vertical room in board_panel's own layout.
        layout.addWidget(self.board_view, stretch=0)
        layout.addStretch(1)
        self.setLayout(layout)

        self._update_side_to_move_label(session_state.current_node)
        session_state.current_node_changed.connect(self._update_side_to_move_label)

    def set_preview_node(self, node: chess.pgn.GameNode | None) -> None:
        """Forwards to the interactive board view -- see `_BoardView.set_preview_node`."""
        self.board_view.set_preview_node(node)

    def _update_side_to_move_label(self, node: chess.pgn.GameNode) -> None:
        board = node.board()
        self.side_to_move_label.setText("White to move" if board.turn == chess.WHITE else "Black to move")
