from __future__ import annotations

import chess
import chess.pgn
import matplotlib
import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QResizeEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from desktop_app.board_panel import DARK_SQUARE_COLOR, LIGHT_SQUARE_COLOR
from desktop_app.compare_math import (
    DIFFERENCE_COLORMAP_NAME,
    AttackInfluenceComparison,
    compare_attack_influence,
    difference_colors,
)
from desktop_app.compare_state import CompareState
from desktop_app.gl_canvas import compute_square_viewport
from desktop_app.layers.attack_influence_layer import colorize_matrix
from desktop_app.position_cache import CacheEntryState, PositionCache
from desktop_app.session_state import SessionState
from visualization.board_plot import UNICODE_PIECES

# Visual/layout refinement passes (Branch Comparison V1's mathematics/state
# are frozen and unchanged -- see compare_math.py/compare_state.py, neither
# of which this module modifies). Board/Influence squares scale between
# these bounds instead of a single hardcoded 200px, based on BOTH the
# panel's available width AND its available height (see
# `_square_pixels_for_size`/`resizeEvent` below) -- fixing the "stuck at
# ~200px on large displays" complaint from the first real-screenshot review
# without letting them grow without bound (which would reintroduce the
# original giant-gap problem in a new form) and, just as important, without
# growing so large on a SHORT window that Δ/the summary get pushed below
# the fold again -- the exact complaint that pass existed to fix.
MIN_GRID_PIXELS = 150
MAX_GRID_PIXELS = 340
# Fixed gap between the A block and the B block, and between a board and
# its own influence panel within one block -- deliberately NOT layout
# stretch space, which is what originally let A and B drift to opposite
# edges of the window (see the module's root-cause note in the class
# docstring below).
BLOCK_GAP = 40
BOARD_INFLUENCE_GAP = 12
# Vertical space every chrome element other than the two square rows
# (title/subtitle/column+move labels/divider/close button/margins/result
# card border) actually costs, measured directly after the V1.1 polish pass
# (see the class docstring's "measured" note), plus a small buffer for
# font-metric variance across environments -- lets `_square_pixels_for_size`
# pick a square that fits the panel's real available height with no
# scrolling wherever the window is tall enough to allow it.
FIXED_CHROME_HEIGHT_PX = 170

TITLE_STYLE = "font-weight: bold; font-size: 14px; padding: 4px;"
SUBTITLE_STYLE = "color: #64748b; font-size: 11px; padding-bottom: 2px;"
COLUMN_LABEL_STYLE = "font-weight: bold; color: #eab308; font-size: 13px;"
MOVE_LABEL_STYLE = "color: #94a3b8;"
DIVIDER_STYLE = "color: #2dd4bf; font-weight: bold; padding: 2px;"
RESULT_TITLE_STYLE = "color: #2dd4bf; font-weight: bold; font-size: 15px;"
# Deliberately NO explicit `color` here (V1.1 polish -- real-Windows-app
# review finding): the earlier hardcoded near-white "#e2e8f0" was written
# assuming a dark theme/background this panel never actually sets, so on
# Windows' real light-theme default palette it read as almost-disabled,
# low-contrast text. Omitting `color` lets these labels inherit the
# widget's own default (system/theme) text color instead, which Qt always
# pairs with a readable contrast against ITS OWN default background in
# both light and dark themes -- the "palette-aware" fix, not a new
# hardcoded guess at a different fixed color.
METRIC_LABEL_STYLE = "font-weight: 600;"
METRIC_VALUE_STYLE = ""
LEGEND_LABEL_STYLE = "font-size: 11px;"
LOADING_STYLE = "color: #94a3b8; font-style: italic; padding: 4px;"
HEATMAP_EMPTY_COLOR = QColor("#1e293b")
CLOSE_BUTTON_STYLE = "color: #94a3b8;"
# No background-color set anywhere on this frame -- it inherits the same
# default palette background every other label on the panel sits on, so
# nothing here can reintroduce the light/dark contrast mismatch this pass
# exists to fix. Only a border, to visually group Δ + legend + metrics into
# one "conclusion card" (V1.1 polish, target hierarchy: A/B are inputs, Δ
# is the conclusion) without filling the interface with a colored panel.
RESULT_FRAME_STYLE = "QFrame#compareResultFrame { border: 1px solid #2dd4bf; border-radius: 8px; }"


def _square_pixels_for_size(available_width: int, available_height: int) -> int:
    """
    The largest square (board/influence/Δ tile) that fits BOTH the panel's
    available width and its available height, clamped between
    MIN_GRID_PIXELS and MAX_GRID_PIXELS. Pure function so the sizing rule
    itself is directly testable without a real Qt layout pass.

    Width constraint: four squares (board_a, influence_a, board_b,
    influence_b) side by side, plus the fixed gaps between them.

    Height constraint: TWO square rows stack vertically (the board/influence
    row, then the Δ row inside the result card) on top of
    FIXED_CHROME_HEIGHT_PX of other content -- this is what keeps Δ and the
    summary inside the initial viewport on a short window instead of
    growing the squares so large they push the conclusion below the fold.
    """
    usable_width = available_width - BLOCK_GAP - 2 * BOARD_INFLUENCE_GAP
    square_from_width = usable_width // 4

    usable_height = available_height - FIXED_CHROME_HEIGHT_PX
    square_from_height = usable_height // 2

    square = min(square_from_width, square_from_height)
    return max(MIN_GRID_PIXELS, min(MAX_GRID_PIXELS, int(square)))


class _StaticBoardView(QWidget):
    """
    A read-only board render for one fixed `chess.pgn.GameNode` -- no
    interactivity, no `SessionState` coupling. The only chess state read
    here is `node.board()` itself (Branch Comparison V1's "do not duplicate
    chess state" rule, unchanged by this pass).

    Deliberately NOT a reuse of `desktop_app.board_panel._BoardView`: that
    widget is wired for click-to-move against the single live
    `SessionState.current_node`, which a static comparison board must never
    trigger -- see that class's own docstring for why click/drag and chess
    state are inseparable there. The square-grid/piece-painting arithmetic
    below is consequently a small, deliberate near-duplicate of
    `_BoardView.paintEvent`'s (this widget has no selection state or legal-
    destination highlighting to also share), not an oversight.
    """

    def __init__(self, node: chess.pgn.GameNode, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._node = node
        self.setFixedSize(MIN_GRID_PIXELS, MIN_GRID_PIXELS)

    def set_node(self, node: chess.pgn.GameNode) -> None:
        self._node = node
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        _offset_x, _offset_y, side, _height = compute_square_viewport(self.width(), self.height())
        square_pixels = side / 8
        board = self._node.board()

        for rank_index in range(8):
            for file_index in range(8):
                visual_row = 7 - rank_index
                x, y = file_index * square_pixels, visual_row * square_pixels
                is_light = (rank_index + file_index) % 2 == 1
                color = LIGHT_SQUARE_COLOR if is_light else DARK_SQUARE_COLOR
                painter.fillRect(QRectF(x, y, square_pixels, square_pixels), color)

        font = QFont()
        font.setPointSize(max(1, int(square_pixels * 0.5)))
        painter.setFont(font)
        painter.setPen(Qt.GlobalColor.black)
        for square, piece in board.piece_map().items():
            file_index = chess.square_file(square)
            rank_index = chess.square_rank(square)
            visual_row = 7 - rank_index
            x, y = file_index * square_pixels, visual_row * square_pixels
            symbol = UNICODE_PIECES[piece.symbol()]
            painter.drawText(QRectF(x, y, square_pixels, square_pixels), Qt.AlignmentFlag.AlignCenter, symbol)


class _HeatmapView(QWidget):
    """
    Paints an 8x8 RGBA color grid (row 0 = rank 8, column 0 = file a, same
    convention as `desktop_app.layers.attack_influence_layer.GridColorFrame`)
    -- shared by Influence A/B (the ordinary production RdBu_r palette) and
    the Δ view (a distinct palette, `desktop_app.compare_math.
    difference_colors` -- unchanged by this pass). Never computes a color
    itself; `set_colors` is always handed an already-computed array.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._colors: np.ndarray | None = None
        self.setFixedSize(MIN_GRID_PIXELS, MIN_GRID_PIXELS)

    def set_colors(self, colors: np.ndarray | None) -> None:
        self._colors = colors
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        if self._colors is None:
            painter.fillRect(self.rect(), HEATMAP_EMPTY_COLOR)
            return

        _offset_x, _offset_y, side, _height = compute_square_viewport(self.width(), self.height())
        square_pixels = side / 8
        for row in range(8):
            for column in range(8):
                red, green, blue, alpha = self._colors[row, column]
                color = QColor.fromRgbF(float(red), float(green), float(blue), float(alpha))
                x, y = column * square_pixels, row * square_pixels
                painter.fillRect(QRectF(x, y, square_pixels, square_pixels), color)


class _LegendBar(QWidget):
    """
    Compact horizontal gradient strip sampling the ACTUAL difference
    colormap (`desktop_app.compare_math.DIFFERENCE_COLORMAP_NAME`) end to
    end -- never a hand-picked/guessed pair of colors, so the legend can
    never drift out of sync with what `difference_colors` actually paints.

    Verified mapping (V1.1 polish -- audited, not assumed):
    `difference_colors` normalizes Δφ = φ_B - φ_A to [0.0, 1.0] via
    `Normalize(vmin=-m, vmax=m)`, so t=0.0 is the most negative Δφ (B lower
    than A) and t=1.0 is the most positive Δφ (B higher than A).
    `matplotlib.colormaps["PuOr"]` was sampled directly (not assumed from
    its name) at t=0.0/0.5/1.0: t=0.0 renders ORANGE, t=0.5 renders
    near-white/neutral, t=1.0 renders PURPLE -- i.e. PuOr's low end is
    orange and its high end is purple, the reverse of a naive reading of
    "Pu-Or" as "purple-then-orange". This bar therefore paints left
    (t=0.0) to right (t=1.0), and `ComparePanel` labels the LEFT/orange
    end "A stronger" and the RIGHT/purple end "B stronger" to match.
    """

    WIDTH = 160
    HEIGHT = 14

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(self.WIDTH, self.HEIGHT)
        self._colormap = matplotlib.colormaps[DIFFERENCE_COLORMAP_NAME]

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        for x in range(self.WIDTH):
            t = x / (self.WIDTH - 1)
            red, green, blue, alpha = self._colormap(t)
            painter.fillRect(
                QRectF(x, 0, 1, self.HEIGHT), QColor.fromRgbF(float(red), float(green), float(blue), float(alpha))
            )


def _summary_metrics(comparison: AttackInfluenceComparison) -> list[tuple[str, str]]:
    """
    Exactly the three (label, value) pairs the approved design named --
    same values/rounding/signs as the original `_summary_text` this
    replaces, only reformatted into a compact scannable label/value layout
    (V1.1 polish item 6) instead of full sentences. No metric added or
    removed; no formula, rounding, or sign logic changed.
    """
    metrics = [("Overall change", f"{comparison.mean_absolute_difference:.2f}")]

    if comparison.max_difference_square is not None:
        metrics.append(
            ("Largest change", f"{comparison.max_absolute_difference:.2f} · {comparison.max_difference_square}")
        )
    else:
        metrics.append(("Largest change", "none — identical influence fields"))

    if comparison.balance_shift > 0:
        metrics.append(("Balance shift", f"+{comparison.balance_shift:.2f} → White"))
    elif comparison.balance_shift < 0:
        metrics.append(("Balance shift", f"{comparison.balance_shift:.2f} → Black"))
    else:
        metrics.append(("Balance shift", "none"))

    return metrics


class ComparePanel(QWidget):
    """
    Branch Comparison V1's dedicated workspace (the approved design report,
    sections 7-8). `MainWindow` swaps this in for the normal V7
    `workspace_row` while `SessionState.compare_state` is open, and swaps
    back on close -- the normal workspace is never permanently squeezed to
    make room for it (see `MainWindow`'s own wiring).

    Reuses `PositionCache` exactly as designed (`request`/`get`/
    `position_ready`, both nodes' analyses keyed by their own
    `board_fen()`); there is no `CompareAnalysis` cache. Every board/matrix
    read here comes from `node.board()` or an already-cached
    `FullPositionAnalysis` -- no new game tree, no duplicated chess state.
    None of this changed across either visual refinement pass below.

    Layout refinement pass 1 (real-screenshot review #1): the original
    layout put A's column and B's column each centered within their OWN
    half of a `QGridLayout` that spanned the entire workspace row -- on a
    wide window this put roughly `window_width / 4` of empty space on
    either side of each 200px-fixed square, reading as two unrelated
    islands rather than one comparison. Replaced with two symmetric
    (board | influence) blocks separated by a small FIXED gap (`BLOCK_GAP`),
    centered as one unit via layout stretches on both sides (not per-column
    stretches). Square size (`_square_pixels_for_size`) scales with the
    panel's own width AND height between `MIN_GRID_PIXELS` and
    `MAX_GRID_PIXELS` instead of a single hardcoded constant, recomputed on
    every `resizeEvent`.

    Layout refinement pass 2 / V1.1 polish (real-screenshot review #2, on
    an actual Windows window): (1) Δ + its metrics now live inside a
    bordered "result card" (`RESULT_FRAME_STYLE`) beneath a "DIFFERENCE
    B - A" heading, visually the conclusion of the comparison, not a third
    unrelated heatmap; (2) the metrics text no longer hardcodes a
    dark-theme-only near-white color (see `METRIC_VALUE_STYLE`'s
    docstring-comment) -- Windows' real light theme made it nearly
    unreadable; (3) a `_LegendBar` (audited against the real
    `difference_colors` mapping, not guessed) makes the PuOr palette's
    meaning explicit; (4) the "Show difference (B - A)" checkbox is gone --
    Δ is Compare mode's entire purpose, so it is unconditionally shown once
    both analyses are ready, never gated behind an extra click; (5) Close
    moved from a detached row beneath Δ to the header's top-right corner.

    Entirely reactive to `SessionState.compare_state_changed`: it never
    calls `enter_compare` itself (that's `TimelinePanel`'s "Compare
    variations" affordance) and its own Close button only ever calls
    `exit_compare` -- the single source of truth for whether this panel is
    open remains `SessionState.compare_state`, not any private flag here.
    """

    def __init__(
        self,
        session_state: SessionState,
        position_cache: PositionCache,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._session_state = session_state
        self._position_cache = position_cache
        self._fen_a: str | None = None
        self._fen_b: str | None = None
        self._current_square_pixels = MIN_GRID_PIXELS

        # The panel's own content lives inside a QScrollArea, not directly
        # in this widget's layout -- the same pattern already used for
        # TimelinePanel's move-history row (desktop_app/timeline_panel.py).
        # At most target resolutions the content fits with no scrolling at
        # all (see tests/test_desktop_app_compare_panel_layout.py); at
        # 1280x720 vertical scrolling remains available as a graceful
        # fallback, per the approved refinement's own instruction, rather
        # than shrinking below MIN_GRID_PIXELS to force a no-scroll fit.
        content = QWidget()
        outer = QVBoxLayout(content)
        outer.setContentsMargins(8, 4, 8, 8)
        outer.setSpacing(6)

        # --- header: title (roughly centered) + a top-right Close action,
        # replacing the previous detached "Close" row beneath the result
        # (V1.1 polish item 5) -- Close's own behavior is untouched, only
        # its location and label changed.
        header_row = QHBoxLayout()
        header_row.addStretch(1)
        title = QLabel("BRANCH COMPARISON")
        title.setStyleSheet(TITLE_STYLE)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_row.addWidget(title)
        header_row.addStretch(1)
        self._close_button = QPushButton("× Close comparison")
        self._close_button.setFlat(True)
        self._close_button.setStyleSheet(CLOSE_BUTTON_STYLE)
        self._close_button.clicked.connect(self._session_state.exit_compare)
        header_row.addWidget(self._close_button)
        outer.addLayout(header_row)

        subtitle = QLabel("same position · two futures")
        subtitle.setStyleSheet(SUBTITLE_STYLE)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(subtitle)

        # --- A / B row: two symmetric (board | influence) blocks, a fixed
        # gap apart, centered as ONE unit (stretches on the OUTSIDE of both
        # blocks, never between them) -- this is the fix for the original
        # giant-center-gap bug: a QGridLayout centered each column
        # independently within its own half of the full row width.
        ab_row = QHBoxLayout()
        ab_row.setSpacing(0)
        ab_row.addStretch(1)

        placeholder = chess.pgn.Game()
        self._board_view_a = _StaticBoardView(placeholder)
        self._board_view_b = _StaticBoardView(placeholder)
        self._influence_view_a = _HeatmapView()
        self._influence_view_b = _HeatmapView()

        self._column_label_a = QLabel("A")
        self._column_label_a.setStyleSheet(COLUMN_LABEL_STYLE)
        self._column_label_b = QLabel("B")
        self._column_label_b.setStyleSheet(COLUMN_LABEL_STYLE)
        self._move_label_a = QLabel("")
        self._move_label_a.setStyleSheet(MOVE_LABEL_STYLE)
        self._move_label_b = QLabel("")
        self._move_label_b.setStyleSheet(MOVE_LABEL_STYLE)

        block_a = self._build_variation_block(self._column_label_a, self._move_label_a, self._board_view_a, self._influence_view_a)
        block_b = self._build_variation_block(self._column_label_b, self._move_label_b, self._board_view_b, self._influence_view_b)
        ab_row.addLayout(block_a)
        ab_row.addSpacing(BLOCK_GAP)
        ab_row.addLayout(block_b)
        ab_row.addStretch(1)
        outer.addLayout(ab_row)

        self._loading_label = QLabel("Computing analysis...")
        self._loading_label.setStyleSheet(LOADING_STYLE)
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._loading_label)

        # --- comparison divider: visually marks the result card below as
        # the CONCLUSION drawn from A and B above it.
        divider = QLabel("↓  comparison  ↓")
        divider.setStyleSheet(DIVIDER_STYLE)
        divider.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(divider)

        # --- result card: Δ heatmap + legend + metrics, bordered as one
        # visual group (V1.1 polish item 1) so the conclusion doesn't read
        # as a smaller, weaker echo of the A/B section above it. No
        # background fill (see RESULT_FRAME_STYLE's own comment) -- only a
        # border, so nothing inside needs a background-specific text color.
        result_frame = QFrame()
        result_frame.setObjectName("compareResultFrame")
        result_frame.setStyleSheet(RESULT_FRAME_STYLE)
        result_frame_layout = QVBoxLayout(result_frame)
        result_frame_layout.setContentsMargins(16, 10, 16, 10)
        result_frame_layout.setSpacing(6)

        result_title = QLabel("DIFFERENCE B − A")
        result_title.setStyleSheet(RESULT_TITLE_STYLE)
        result_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_frame_layout.addWidget(result_title)

        result_row = QHBoxLayout()
        result_row.setSpacing(BOARD_INFLUENCE_GAP * 2)
        self._diff_view = _HeatmapView()
        # Hidden until both sides are READY (see _open/_refresh_if_both_ready)
        # -- no checkbox gates this anymore (V1.1 polish item 4), but a
        # fresh, not-yet-computed comparison must not show stale/placeholder
        # Δ content either.
        self._diff_view.setVisible(False)
        result_row.addWidget(self._diff_view)

        legend_and_metrics = QVBoxLayout()
        legend_and_metrics.setSpacing(8)

        # Legend: gradient bar + endpoint/zero labels, mathematically
        # correct against the audited PuOr mapping (see _LegendBar's own
        # docstring) -- "A stronger" is the orange/low end, "B stronger" is
        # the purple/high end, "0" is the visually identifiable midpoint.
        legend_block = QVBoxLayout()
        legend_block.setSpacing(2)
        self._legend_bar = _LegendBar()
        legend_bar_row = QHBoxLayout()
        legend_bar_row.addWidget(self._legend_bar)
        legend_bar_row.addStretch(1)
        legend_block.addLayout(legend_bar_row)

        legend_labels_row = QHBoxLayout()
        a_stronger_label = QLabel("A stronger")
        a_stronger_label.setStyleSheet(LEGEND_LABEL_STYLE)
        zero_label = QLabel("0")
        zero_label.setStyleSheet(LEGEND_LABEL_STYLE)
        b_stronger_label = QLabel("B stronger")
        b_stronger_label.setStyleSheet(LEGEND_LABEL_STYLE)
        legend_labels_row.addWidget(a_stronger_label)
        legend_labels_row.addStretch(1)
        legend_labels_row.addWidget(zero_label)
        legend_labels_row.addStretch(1)
        legend_labels_row.addWidget(b_stronger_label)
        legend_labels_row.setContentsMargins(0, 0, 0, 0)
        legend_block.addLayout(legend_labels_row)
        legend_and_metrics.addLayout(legend_block)

        # Metrics: compact label/value pairs (V1.1 polish item 6),
        # unchanged values/rounding/signs -- see `_summary_metrics`.
        self._metrics_grid = QVBoxLayout()
        self._metrics_grid.setSpacing(2)
        legend_and_metrics.addLayout(self._metrics_grid)
        legend_and_metrics.addStretch(1)

        result_row.addLayout(legend_and_metrics)
        result_frame_layout.addLayout(result_row)

        result_centering_row = QHBoxLayout()
        result_centering_row.addStretch(1)
        result_centering_row.addWidget(result_frame)
        result_centering_row.addStretch(1)
        outer.addLayout(result_centering_row)
        self._result_frame = result_frame

        outer.addStretch(1)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidget(content)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self_layout = QVBoxLayout(self)
        self_layout.setContentsMargins(0, 0, 0, 0)
        self_layout.addWidget(self._scroll_area)

        self._position_cache.position_ready.connect(self._on_position_ready)
        self._session_state.compare_state_changed.connect(self._on_compare_state_changed)

        self.setVisible(False)
        if self._session_state.compare_state is not None:
            self._open(self._session_state.compare_state)

    @staticmethod
    def _build_variation_block(
        column_label: QLabel,
        move_label: QLabel,
        board_view: _StaticBoardView,
        influence_view: _HeatmapView,
    ) -> QVBoxLayout:
        """One symmetric (label / move / board+influence) block -- A and B
        are built from this exact same method, so they can never drift
        apart in structure, only in the data they display."""
        block = QVBoxLayout()
        block.setSpacing(2)
        column_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        move_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        block.addWidget(column_label)
        block.addWidget(move_label)

        board_row = QHBoxLayout()
        board_row.setSpacing(BOARD_INFLUENCE_GAP)
        board_row.addWidget(board_view)
        board_row.addWidget(influence_view)
        block.addLayout(board_row)
        return block

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._apply_responsive_square_size()

    def _apply_responsive_square_size(self) -> None:
        # self.width()/self.height() are the panel's OWN size, i.e. what
        # workspace_row actually handed it -- not the scroll area's viewport
        # (same width/height here; the scroll area only ever adds a
        # VERTICAL scrollbar, see __init__) and not the full MainWindow size.
        square = _square_pixels_for_size(self.width(), self.height())
        if square == self._current_square_pixels:
            return
        self._current_square_pixels = square
        for view in (
            self._board_view_a,
            self._board_view_b,
            self._influence_view_a,
            self._influence_view_b,
            self._diff_view,
        ):
            view.setFixedSize(square, square)

    def _on_compare_state_changed(self) -> None:
        compare_state = self._session_state.compare_state
        if compare_state is None:
            self._close()
        else:
            self._open(compare_state)

    def _open(self, compare_state: CompareState) -> None:
        node_a, node_b = compare_state.node_a, compare_state.node_b
        parent = node_a.parent
        self._move_label_a.setText(parent.board().san(node_a.move))
        self._move_label_b.setText(parent.board().san(node_b.move))
        self._board_view_a.set_node(node_a)
        self._board_view_b.set_node(node_b)

        self._influence_view_a.set_colors(None)
        self._influence_view_b.set_colors(None)
        self._diff_view.set_colors(None)
        self._diff_view.setVisible(False)
        self._clear_metrics()
        self._loading_label.setVisible(True)

        self._fen_a = self._position_cache.request(node_a.board())
        self._fen_b = self._position_cache.request(node_b.board())
        self.setVisible(True)
        self._apply_responsive_square_size()
        self._refresh_if_both_ready()

    def _close(self) -> None:
        self.setVisible(False)
        self._fen_a = None
        self._fen_b = None

    def _on_position_ready(self, fen: str) -> None:
        if fen != self._fen_a and fen != self._fen_b:
            return
        self._refresh_if_both_ready()

    def _clear_metrics(self) -> None:
        while self._metrics_grid.count():
            item = self._metrics_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            layout = item.layout()
            if layout is not None:
                while layout.count():
                    child = layout.takeAt(0)
                    child_widget = child.widget()
                    if child_widget is not None:
                        child_widget.deleteLater()

    def _refresh_if_both_ready(self) -> None:
        if self._fen_a is None or self._fen_b is None:
            return
        entry_a = self._position_cache.get(self._fen_a)
        entry_b = self._position_cache.get(self._fen_b)
        if entry_a.state != CacheEntryState.READY or entry_b.state != CacheEntryState.READY:
            return  # one or both sides still loading -- stay in the loading state

        self._loading_label.setVisible(False)

        matrix_a = np.array(entry_a.analysis.attack_influence_field.matrix, dtype=float)
        matrix_b = np.array(entry_b.analysis.attack_influence_field.matrix, dtype=float)
        self._influence_view_a.set_colors(colorize_matrix(matrix_a).colors)
        self._influence_view_b.set_colors(colorize_matrix(matrix_b).colors)

        comparison = compare_attack_influence(entry_a, entry_b)
        # V1.1 polish: Δ is unconditionally shown once ready -- Compare
        # mode's entire purpose is showing the difference, so there is no
        # longer a "Show difference" checkbox gating it (item 4).
        self._diff_view.set_colors(difference_colors(comparison.difference))
        self._diff_view.setVisible(True)

        self._clear_metrics()
        for label_text, value_text in _summary_metrics(comparison):
            row = QHBoxLayout()
            row.setSpacing(12)
            label = QLabel(label_text)
            label.setStyleSheet(METRIC_LABEL_STYLE)
            value = QLabel(value_text)
            value.setStyleSheet(METRIC_VALUE_STYLE)
            row.addWidget(label)
            row.addStretch(1)
            row.addWidget(value)
            self._metrics_grid.addLayout(row)
