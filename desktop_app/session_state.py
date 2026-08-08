from __future__ import annotations

import chess
import chess.pgn
from PySide6.QtCore import QObject, Signal

from chess_engine.models import MoveDetails
from chess_engine.moves import execute_move
from desktop_app.transition import TransitionState


# The six layers registered by Phase 5c (desktop_app/layers/*.py). Default
# visibility deliberately does not turn all six on: with every field layer
# drawn at once the canvas is visually crowded (six overlapping color scales
# and marker sets on one 8x8 grid), so the first view shows a legible
# starting subset -- the two signed color fields plus discrete markers
# (Attack Influence, Equipotential, Critical Points) -- while the two vector
# fields and the cell partition (Gradient, Ridge/Valley, Morse-Smale) start
# hidden, one checkbox away via the layer-strip UI (desktop_app/layer_panel.py).
# All six remain independently toggleable and can still be shown together.
DEFAULT_LAYER_VISIBILITY: dict[str, bool] = {
    "attack_influence": True,
    "equipotential": True,
    "gradient": False,
    "ridge_valley": False,
    "morse_smale": False,
    "critical_points": True,
}

# Opacity is a presentation-only alpha multiplier applied at draw time
# (desktop_app/gl_canvas.py); it never touches analysis data or colors
# computed by a layer's own data_source/renderer. Full opacity by default --
# the "crowded" concern is addressed by DEFAULT_LAYER_VISIBILITY above, not
# by pre-dimming layers a user does choose to show.
DEFAULT_LAYER_OPACITY: dict[str, float] = {
    "attack_influence": 1.0,
    "equipotential": 1.0,
    "gradient": 1.0,
    "ridge_valley": 1.0,
    "morse_smale": 1.0,
    "critical_points": 1.0,
}


class SessionState(QObject):
    """
    Single source of truth for cross-panel state (docs/interactive_ui.md Part 4.1).

    Phase 5b's two slices -- `current_node` and `layer_state` -- each keep their
    own typed Qt signal from Phase 5a, establishing the "one signal per logical
    slice, no monolithic signal, no per-field signal" update propagation
    mechanism the frozen architecture specifies. The remaining slices from Part
    4.1's table (`mixer_state`, `freeze_visualization`, `freeze_audio`,
    `transport_state`, `camera`, `compare_state`) are added by the phases that
    first populate them (5f, 5h), as a mechanical repetition of this same
    pattern -- not a redesign of it.

    `transition_state` (this milestone) is one further mechanical repetition
    of the same pattern for a slice Part 4.1's table doesn't name explicitly
    but the phase's own instructions require ("SessionState remains the
    source of current transition state"): the in-flight move-to-move
    animation (docs/interactive_ui.md Part 6). `desktop_app.
    transition_controller.TransitionController` is its sole writer --
    SessionState itself computes nothing about the transition, exactly like
    every other slice here.

    `current_node` now points into a `chess.pgn.GameNode` tree (Part 4.4),
    since undo and branching are real from this phase on. There is
    deliberately no separate `game_tree` field: nothing in this phase ever
    needs the tree root independently of `current_node` (undo walks
    `.parent`, branch creation/reuse uses `.add_variation`/`.has_variation`/
    `.variation` on the current node), and the root is always reachable later
    via `.parent` if a future phase (5e's Timeline) needs it. Adding it now,
    with no consumer, would be speculative.
    """

    current_node_changed = Signal(object)  # emits the new chess.pgn.GameNode
    layer_state_changed = Signal(str)  # emits the layer_id whose visibility changed
    transition_state_changed = Signal()  # payload lives on session_state.transition_state itself

    def __init__(self, initial_node: chess.pgn.GameNode) -> None:
        super().__init__()
        self._current_node: chess.pgn.GameNode = initial_node
        self._layer_state: dict[str, bool] = dict(DEFAULT_LAYER_VISIBILITY)
        self._layer_opacity: dict[str, float] = dict(DEFAULT_LAYER_OPACITY)
        self._transition_state = TransitionState(
            from_fen=None, to_fen=initial_node.board().board_fen(), progress=1.0
        )
        # Multi-level redo: each undo() pushes the node being left; make_move()
        # with a genuinely new move (not a replay of an existing branch)
        # invalidates it, matching standard undo/redo semantics.
        self._redo_stack: list[chess.pgn.GameNode] = []

    @property
    def current_node(self) -> chess.pgn.GameNode:
        return self._current_node

    def set_current_node(self, node: chess.pgn.GameNode) -> None:
        if node.board().board_fen() == self._current_node.board().board_fen():
            return
        self._current_node = node
        self.current_node_changed.emit(node)

    def make_move(self, move: chess.Move) -> MoveDetails | None:
        """
        The only place a move is executed. Legality is delegated entirely to
        chess_engine.moves.execute_move (python-chess's own board.legal_moves
        underneath it) -- never re-checked here. Validation runs against a
        fresh, throwaway board from `current_node.board()`, so an illegal
        move never mutates the tree.

        Reuses an existing child via has_variation()/variation() rather than
        always calling add_variation(), so replaying a move that was just
        undone (or manually re-clicking the same move) re-enters the same
        branch instead of creating a duplicate sibling.
        """
        candidate_board = self._current_node.board()
        details = execute_move(candidate_board, move.uci())
        if details is None:
            return None

        if self._current_node.has_variation(move):
            next_node = self._current_node.variation(move)
        else:
            next_node = self._current_node.add_variation(move)
            # A genuinely new branch invalidates any pending redo -- redoing
            # into a line that a new move just diverged from would be wrong.
            self._redo_stack.clear()

        self.set_current_node(next_node)
        return details

    def undo(self) -> bool:
        parent = self._current_node.parent
        if parent is None:
            return False
        self._redo_stack.append(self._current_node)
        self.set_current_node(parent)
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        target = self._redo_stack.pop()
        self.set_current_node(target)
        return True

    def layer_visible(self, layer_id: str) -> bool:
        return self._layer_state.get(layer_id, True)

    def set_layer_visible(self, layer_id: str, visible: bool) -> None:
        if self._layer_state.get(layer_id) == visible:
            return
        self._layer_state[layer_id] = visible
        self.layer_state_changed.emit(layer_id)

    def layer_opacity(self, layer_id: str) -> float:
        return self._layer_opacity.get(layer_id, 1.0)

    def set_layer_opacity(self, layer_id: str, opacity: float) -> None:
        """
        Opacity is part of the same `layer_state` slice as visibility (Part
        4.1's table: "visibility / opacity / solo, keyed by layer_id") --
        deliberately reuses `layer_state_changed` rather than adding a
        second signal for what the frozen design treats as one slice.
        """
        opacity = max(0.0, min(1.0, opacity))
        if self._layer_opacity.get(layer_id) == opacity:
            return
        self._layer_opacity[layer_id] = opacity
        self.layer_state_changed.emit(layer_id)

    @property
    def transition_state(self) -> TransitionState:
        return self._transition_state

    def set_transition_state(self, state: TransitionState) -> None:
        if state == self._transition_state:
            return
        self._transition_state = state
        self.transition_state_changed.emit()
