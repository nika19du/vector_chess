from __future__ import annotations

import chess.pgn
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from desktop_app.audio_controller import AudioController
from desktop_app.board_panel import BoardPanel
from desktop_app.scrub import ScrubPosition, clamp_scrub_fraction
from desktop_app.scrub_controller import ScrubController
from desktop_app.session_state import SessionState
from desktop_app.transition_controller import TransitionController

SCRUB_STRIP_STYLE = "background-color: #334155; border-radius: 4px;"

# Reuses the same tension-gold accent board_panel.py already uses for
# selection (SELECTED_SQUARE_OVERLAY/LEGAL_DESTINATION_DOT) -- Part 9
# reserves that role for selection/active-highlight, not a new color choice.
CURRENT_MOVE_HIGHLIGHT_STYLE = "background-color: #eab308; color: #1e293b; font-weight: bold;"
MOVE_BUTTON_STYLE = "text-align: left;"
BRANCH_BADGE_STYLE = "color: #eab308; font-weight: bold;"
SIBLING_BUTTON_STYLE = "color: #94a3b8; font-style: italic;"


class _ScrubStrip(QWidget):
    """
    Phase 5e.2: the continuous drag-to-scrub strip. Sits above the discrete
    move-history row, mapping a mouse x-position across its width to a
    continuous fraction across `SessionState.active_path()` -- the full
    active branch (root..tip), the resolved scope for scrubbing, wider than
    the discrete strip's own `mainline_path()` (root..current_node) below.

    Inert without a `ScrubController`: `TimelinePanel(session_state, parent)`
    (the pre-5e.2 constructor shape, still used by existing discrete-only
    tests and call sites) constructs this strip with `board_panel=None`,
    `transition_controller=None`, `scrub_controller=None`, and every handler
    here no-ops in that case -- discrete Timeline behavior is completely
    unaffected by this class existing.

    Pure pixel<->fraction arithmetic lives here (`_fraction_from_x`, via
    `desktop_app.scrub.clamp_scrub_fraction`); `ScrubController` itself
    never sees a raw pixel, only `ScrubPosition` values -- the same
    separation of concerns `_BoardView` keeps between its own pixel<->square
    mapping and the chess logic it drives.

    Phase 5f.4 adds an optional `audio_controller`, forwarded the exact same
    `ScrubPosition` values at the exact same call sites as `scrub_controller`
    (begin/update/end), never gating whether a scrub is allowed to start --
    that decision (Rule 6, below) is made once, before either controller is
    told about it, which is what keeps visual and audio scrub from ever
    disagreeing about whether one is active.
    """

    FIXED_HEIGHT = 20

    def __init__(
        self,
        session_state: SessionState,
        board_panel: BoardPanel | None,
        transition_controller: TransitionController | None,
        scrub_controller: ScrubController | None,
        parent: QWidget | None = None,
        *,
        audio_controller: AudioController | None = None,
    ) -> None:
        super().__init__(parent)
        self._session_state = session_state
        self._board_panel = board_panel
        self._transition_controller = transition_controller
        self._scrub_controller = scrub_controller
        # Optional and separately guarded, NOT part of `_is_active()` --
        # audio is an additional consumer of the same scrub gesture the
        # visual system already gates, not a second thing that gesture
        # depends on. A scrub strip built without an AudioController (every
        # pre-5f.4 call site, and every visual-only test) behaves exactly
        # as it did before this phase.
        self._audio_controller = audio_controller
        self.setFixedHeight(self.FIXED_HEIGHT)
        self.setStyleSheet(SCRUB_STRIP_STYLE)

    def _is_active(self) -> bool:
        return (
            self._board_panel is not None
            and self._transition_controller is not None
            and self._scrub_controller is not None
        )

    def _fraction_from_x(self, x: float, path_length: int) -> ScrubPosition:
        segment_count = max(1, path_length - 1)
        width = max(1, self.width())
        raw_position = (x / width) * segment_count
        return clamp_scrub_fraction(raw_position, path_length)

    def _apply(self, x: float) -> None:
        path_length = self._scrub_controller.path_length
        if path_length is None:
            return
        position = self._fraction_from_x(x, path_length)
        self._scrub_controller.update(position)
        self._board_panel.set_preview_node(self._scrub_controller.snap_to_nearest_node(position))
        if self._audio_controller is not None:
            self._audio_controller.update_scrub(position)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._is_active() or event.button() != Qt.MouseButton.LeftButton:
            return
        if self._transition_controller.is_animating:
            # Rule 6 (approved plan): scrubbing may only begin from a fully
            # settled state -- a press during an ordinary ~500ms move
            # animation is simply ignored, not queued or force-settled.
            return
        active_path = self._session_state.active_path()
        if len(active_path) < 2:
            return  # nothing to scrub across
        self._scrub_controller.begin(active_path)
        if self._audio_controller is not None:
            self._audio_controller.begin_scrub(active_path)
        self._apply(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._is_active() or not self._scrub_controller.is_scrubbing:
            return
        self._apply(event.position().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._is_active() or not self._scrub_controller.is_scrubbing:
            return
        path_length = self._scrub_controller.path_length
        if path_length is None:
            return
        position = self._fraction_from_x(event.position().x(), path_length)
        node = self._scrub_controller.end(position)
        if self._audio_controller is not None:
            self._audio_controller.end_scrub()
        self._board_panel.set_preview_node(None)
        # The ONE commit: reuses set_current_node's own board_fen() no-op
        # guard if the snapped node is already current. From here on, the
        # ordinary committed-navigation pipeline (MainWindow/BoardPanel/
        # TimelinePanel's own current_node_changed reactions) takes over
        # exactly as it would for any button click or history entry click.
        self._session_state.set_current_node(node)


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

    Phase 5e.2 (continuous Timeline scrubbing) adds a `_ScrubStrip` above the
    discrete move-history row, wired to `board_panel`/`transition_controller`/
    `scrub_controller` -- all keyword-only and optional, defaulting to
    `None`. Existing callers that construct `TimelinePanel(session_state,
    parent)` (unchanged positional shape) get a `_ScrubStrip` that's
    constructed but inert: every mouse handler no-ops when these are `None`,
    so discrete-only behavior is completely unaffected. Only
    `MainWindow`'s full wiring activates scrubbing. `_on_current_node_changed`
    additionally cancels any in-progress scrub and clears the board preview
    on ANY navigation, whether it originated from this panel's own scrub
    commit or from an external source (a branch switch, a button click) --
    see that method's own comment.
    """

    def __init__(
        self,
        session_state: SessionState,
        parent: QWidget | None = None,
        *,
        board_panel: BoardPanel | None = None,
        transition_controller: TransitionController | None = None,
        scrub_controller: ScrubController | None = None,
        audio_controller: AudioController | None = None,
    ) -> None:
        super().__init__(parent)
        self._session_state = session_state
        self._board_panel = board_panel
        self._transition_controller = transition_controller
        self._scrub_controller = scrub_controller
        self._audio_controller = audio_controller
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

        self._scrub_strip = _ScrubStrip(
            session_state,
            board_panel,
            transition_controller,
            scrub_controller,
            audio_controller=audio_controller,
        )
        outer.addWidget(self._scrub_strip)

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
        # Phase 5e.2: cancel any in-progress scrub on ANY navigation --
        # whether it's this panel's own scrub commit (already idle by the
        # time this fires, so a harmless no-op) or an externally-driven
        # change (a branch switch, a button click, a future feature) that
        # invalidates whatever path a scrub snapshotted at press time.
        # Cheap and idempotent, so it runs synchronously, unlike the
        # rebuild below -- only the rebuild's setEnabled() calls were the
        # crash-triggering part the deferral exists to avoid.
        if self._scrub_controller is not None:
            self._scrub_controller.cancel()
        if self._audio_controller is not None:
            self._audio_controller.cancel_scrub()
        if self._board_panel is not None:
            self._board_panel.set_preview_node(None)

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
