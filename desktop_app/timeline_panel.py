from __future__ import annotations

import chess.pgn
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from desktop_app.session_state import SessionState

# Reuses the same tension-gold accent board_panel.py already uses for
# selection (SELECTED_SQUARE_OVERLAY/LEGAL_DESTINATION_DOT) -- Part 9
# reserves that role for selection/active-highlight, not a new color choice.
CURRENT_MOVE_HIGHLIGHT_STYLE = "background-color: #eab308; color: #1e293b; font-weight: bold;"
MOVE_BUTTON_STYLE = "text-align: left;"
BRANCH_BADGE_STYLE = "color: #eab308; font-weight: bold;"
SIBLING_BUTTON_STYLE = "color: #94a3b8; font-style: italic;"


class TimelinePanel(QWidget):
    """
    The bottom transport strip (docs/interactive_ui.md Part 2/Part 8),
    restrained to discrete navigation for Phase 5e (see the milestone plan --
    continuous slider scrubbing is explicitly out of scope here): First /
    Previous / Next / Last, plus a clickable move history with branch
    switching.

    Communicates exclusively through `SessionState` -- calls its navigation
    methods (`go_to_start`, `undo`, `redo`, `go_to_end`, `set_current_node`)
    and reads `current_node`/`mainline_path()`/`can_redo()`, plus the
    `chess.pgn.GameNode` tree those expose directly (the same precedent
    `board_panel.py` already sets: chess-rules state is read by every panel,
    docs/interactive_ui.md Part 4.2). Never imports or references
    `MathCanvas`, `BoardPanel`, `PositionCache`, or `TransitionController` --
    those all already react to `SessionState.current_node_changed` on their
    own, exactly as they do for moves/undo/redo, so Timeline navigation needs
    no new wiring to reach them.

    Branch UI is deliberately restrained: a small badge appears next to any
    path entry whose parent has more than one child, and clicking it toggles
    an inline row of the OTHER sibling(s)' move labels -- no popup, no
    dedicated tree widget. Clicking a sibling label is just a direct
    `set_current_node` call, identical to clicking any other history entry;
    "switching branches" is not a separate mechanism.

    `current_node_changed`-triggered rebuilds are deferred to the next Qt
    event-loop turn (Qt lifecycle investigation, see the approved diagnosis):
    a nav button's own `.click()` can itself cause `current_node_changed` to
    fire synchronously, and `_rebuild()` unconditionally calls `setEnabled()`
    on all four nav buttons every time it runs. If that ran synchronously
    inside the click, a button whose navigation disables it (e.g. clicking
    Previous from one move deep, landing back at the root) would have
    `setEnabled(False)` called on itself while still mid-dispatch of its own
    click -- a widget mutating its own enabled state from inside its own
    click-handler call stack, which reproduced a native access violation
    under this environment. `QTimer.singleShot(0, ...)` moves the rebuild
    (and its `setEnabled()` calls) to run after the click's own call stack
    has fully unwound, closing the gap. `_rebuild_scheduled` coalesces
    several `current_node_changed` signals that arrive before the scheduled
    rebuild has run into a single rebuild call -- `_rebuild()` always reads
    `SessionState.current_node`/`mainline_path()` fresh when it actually
    runs, so the one rebuild that does happen reflects whatever the latest
    current node is by then, not whichever signal triggered the scheduling.

    The initial build in `__init__` and branch-badge-triggered rebuilds
    (`_toggle_branch_point`) remain synchronous: neither is triggered from
    inside a nav button's own click (the initial build isn't triggered by a
    click at all; a badge is a `_moves_layout` entry that gets replaced, not
    a persistent button whose own enabled state `_rebuild()` ever touches --
    confirmed safe during the investigation, unlike the four nav buttons).
    """

    def __init__(self, session_state: SessionState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session_state = session_state
        # Nodes whose sibling row is currently expanded. Persists across
        # rebuilds (so re-expanding isn't needed after every navigation);
        # entries for nodes that fall off the currently-displayed path are
        # simply never rendered again -- harmless, bounded by branch points
        # ever expanded in this session, not a leak.
        self._expanded_branch_points: set[chess.pgn.GameNode] = set()
        self._move_buttons: dict[chess.pgn.GameNode, QPushButton] = {}
        self._branch_badges: dict[chess.pgn.GameNode, QPushButton] = {}
        # True while a rebuild is scheduled but hasn't run yet -- guards
        # against queuing a second QTimer.singleShot before the first one
        # fires, so several current_node_changed signals arriving before the
        # event loop's next turn collapse into one rebuild.
        self._rebuild_scheduled = False

        outer = QVBoxLayout(self)

        nav_row = QHBoxLayout()
        self._first_button = QPushButton("|◀")
        self._previous_button = QPushButton("◀")
        self._next_button = QPushButton("▶")
        self._last_button = QPushButton("▶|")
        for button in (self._first_button, self._previous_button, self._next_button, self._last_button):
            button.setFixedWidth(36)
            nav_row.addWidget(button)
        nav_row.addStretch(1)
        outer.addLayout(nav_row)

        self._first_button.clicked.connect(session_state.go_to_start)
        self._previous_button.clicked.connect(lambda: session_state.undo())
        self._next_button.clicked.connect(lambda: session_state.redo())
        self._last_button.clicked.connect(session_state.go_to_end)

        self._moves_container = QWidget()
        self._moves_layout = QHBoxLayout(self._moves_container)
        self._moves_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self._moves_container)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setFixedHeight(48)
        outer.addWidget(scroll_area)

        self.setLayout(outer)

        session_state.current_node_changed.connect(self._on_current_node_changed)
        self._rebuild()  # initial build: synchronous, not triggered by a click -- safe as-is

    def _on_current_node_changed(self, node: chess.pgn.GameNode) -> None:
        if self._rebuild_scheduled:
            return
        self._rebuild_scheduled = True
        QTimer.singleShot(0, self._run_scheduled_rebuild)

    def _run_scheduled_rebuild(self) -> None:
        self._rebuild_scheduled = False
        self._rebuild()

    def _rebuild(self) -> None:
        self._move_buttons.clear()
        self._branch_badges.clear()
        while self._moves_layout.count():
            item = self._moves_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        path = self._session_state.mainline_path()
        current_node = self._session_state.current_node

        for index in range(1, len(path)):
            node = path[index]
            parent = path[index - 1]

            label = parent.board().san(node.move)
            button = QPushButton(label)
            button.setFlat(True)
            button.setStyleSheet(CURRENT_MOVE_HIGHLIGHT_STYLE if node is current_node else MOVE_BUTTON_STYLE)
            button.clicked.connect(lambda _checked=False, node=node: self._session_state.set_current_node(node))
            self._moves_layout.addWidget(button)
            self._move_buttons[node] = button

            siblings = [child for child in parent.variations if child is not node]
            if siblings:
                badge = QPushButton(f"↳{len(siblings)}")
                badge.setFlat(True)
                badge.setStyleSheet(BRANCH_BADGE_STYLE)
                badge.clicked.connect(lambda _checked=False, node=node: self._toggle_branch_point(node))
                self._moves_layout.addWidget(badge)
                self._branch_badges[node] = badge

                if node in self._expanded_branch_points:
                    for sibling in siblings:
                        sibling_label = parent.board().san(sibling.move)
                        sibling_button = QPushButton(f"(or: {sibling_label})")
                        sibling_button.setFlat(True)
                        sibling_button.setStyleSheet(SIBLING_BUTTON_STYLE)
                        sibling_button.clicked.connect(
                            lambda _checked=False, sibling=sibling: self._session_state.set_current_node(sibling)
                        )
                        self._moves_layout.addWidget(sibling_button)

        self._moves_layout.addStretch(1)

        self._first_button.setEnabled(current_node.parent is not None)
        self._previous_button.setEnabled(current_node.parent is not None)
        can_go_forward = self._session_state.can_redo()
        self._next_button.setEnabled(can_go_forward)
        self._last_button.setEnabled(can_go_forward)

    def _toggle_branch_point(self, node: chess.pgn.GameNode) -> None:
        if node in self._expanded_branch_points:
            self._expanded_branch_points.discard(node)
        else:
            self._expanded_branch_points.add(node)
        self._rebuild()
