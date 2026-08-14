from __future__ import annotations

import chess
import chess.pgn
from PySide6.QtCore import QObject, Signal

from chess_engine.models import MoveDetails
from chess_engine.moves import execute_move
from desktop_app.compare_state import CompareState, are_valid_siblings
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
    `transport_state`, `camera`) are added by the phases that first populate
    them, as a mechanical repetition of this same pattern -- not a redesign
    of it.

    `compare_state` (Branch Comparison V1, the approved design report):
    `None` when no comparison is open, or a `CompareState(node_a, node_b)`
    holding two canonical sibling `GameNode` references from the SAME tree
    `current_node` points into -- never a copy, never a second tree. Entering
    or exiting a comparison never touches `current_node` or `_active_child`;
    `current_node` remains authoritative the entire time a comparison is
    open. See `enter_compare`/`exit_compare` below and `desktop_app.
    compare_state`'s own docstring for the full read-only contract.

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

    Phase 5e (Timeline/history navigation): `_active_child` replaces the
    earlier LIFO `_redo_stack`. A Timeline can jump to any node directly
    (click a distant move, click into a sibling branch), not just walk one
    step at a time, so "the node most recently undone" is no longer a rich
    enough concept -- what's needed is "the child to follow forward from
    *any* node," recorded for every node on a navigation's path, not only the
    one being left. See `_mark_active_path`.
    """

    current_node_changed = Signal(object)  # emits the new chess.pgn.GameNode
    layer_state_changed = Signal(str)  # emits the layer_id whose visibility changed
    transition_state_changed = Signal()  # payload lives on session_state.transition_state itself
    compare_state_changed = Signal()  # payload lives on session_state.compare_state itself

    def __init__(self, initial_node: chess.pgn.GameNode) -> None:
        super().__init__()
        self._current_node: chess.pgn.GameNode = initial_node
        self._layer_state: dict[str, bool] = dict(DEFAULT_LAYER_VISIBILITY)
        self._layer_opacity: dict[str, float] = dict(DEFAULT_LAYER_OPACITY)
        self._transition_state = TransitionState(
            from_fen=None, to_fen=initial_node.board().board_fen(), progress=1.0
        )
        # Records, per node, which child was most recently navigated to from
        # it -- by play, undo/redo, or a Timeline jump -- so redo()/
        # go_to_end() know which way to go forward from any node, not just
        # the one most recently left. Keyed by the GameNode object itself
        # (default identity hash/eq -- GameNode defines neither): every node
        # in the tree is kept alive for as long as SessionState holds any
        # node (parent links up, variations link back down through the whole
        # tree), so nothing here can go stale or collide.
        #
        # Entries for a branch navigated away from (undo + a different move,
        # or switching into a sibling branch) are never deleted, only
        # shadowed by whichever branch is active now -- they resurface
        # correctly if the user returns to that branch later, so
        # go_to_end() resumes exactly where that branch was last left. This
        # is intentional per-branch memory, bounded by the size of the game
        # tree itself (already retained in full), not a leak.
        self._active_child: dict[chess.pgn.GameNode, chess.pgn.GameNode] = {}
        self._compare_state: CompareState | None = None

    @property
    def current_node(self) -> chess.pgn.GameNode:
        return self._current_node

    @property
    def compare_state(self) -> CompareState | None:
        return self._compare_state

    def enter_compare(self, node_a: chess.pgn.GameNode, node_b: chess.pgn.GameNode) -> None:
        """
        Opens a Branch Comparison between two sibling variations (Branch
        Comparison V1's sole valid domain -- see `are_valid_siblings`).
        Raises `ValueError` for anything else (same node twice, a node with
        no parent, two nodes with different parents) rather than silently
        allowing an arbitrary-node comparison the design was never validated
        for.

        Deliberately does not touch `current_node` or `_active_child`:
        selecting A/B is a read-only view into the existing tree, not a
        navigation event.
        """
        if not are_valid_siblings(node_a, node_b):
            raise ValueError(
                "Branch Comparison V1 only supports two distinct direct "
                "siblings sharing the same parent"
            )
        self._compare_state = CompareState(node_a=node_a, node_b=node_b)
        self.compare_state_changed.emit()

    def exit_compare(self) -> None:
        """Closes the open comparison, if any. A no-op (no signal) if none is open."""
        if self._compare_state is None:
            return
        self._compare_state = None
        self.compare_state_changed.emit()

    def _mark_active_path(self, node: chess.pgn.GameNode) -> None:
        """
        Records `node` as reachable-forward from every ancestor on its root
        path -- "this is the line to follow forward from here" -- for every
        node along the way up to the root. Idempotent, O(depth).
        """
        child = node
        parent = child.parent
        while parent is not None:
            self._active_child[parent] = child
            child = parent
            parent = child.parent

    def set_current_node(self, node: chess.pgn.GameNode) -> None:
        # Branch Comparison V1: every real navigation entry point in this
        # class funnels through here (make_move, undo, redo, go_to_start,
        # go_to_end, plus every direct Timeline/scrub-release call), so
        # exiting compare mode here -- before the no-op guard below -- covers
        # "any real Timeline navigation action" uniformly, with no separate
        # exit-compare call needed at each of those call sites. Exits even
        # when the guard below will make this call a no-op: clicking a nav
        # button that happens not to move the position is still a real
        # navigation *action*.
        self.exit_compare()
        if node.board().board_fen() == self._current_node.board().board_fen():
            return
        # Marking must happen AFTER the equality guard above, not before:
        # board_fen() compares piece placement only, so two nodes on
        # different branches (a transposition, or a simple repetition) can
        # compare equal without being the same tree node. Marking
        # unconditionally would silently rewrite _active_child for a path
        # the current node never actually moved onto whenever such a no-op
        # call happens -- a time-displaced desync between what
        # redo()/go_to_end() do next and what the user actually navigated
        # to. Gating marking on the same guard that gates the real
        # navigation closes this.
        self._mark_active_path(node)
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

        No explicit "invalidate pending redo" step is needed here: a
        genuinely new branch still runs through set_current_node() below,
        whose _mark_active_path() reassigns this node's active child to
        whichever node was actually just navigated to -- uniformly, whether
        that's a brand-new branch, a reused existing child, or (via other
        entry points) an ancestor or a distant Timeline jump.
        """
        candidate_board = self._current_node.board()
        details = execute_move(candidate_board, move.uci())
        if details is None:
            return None

        if self._current_node.has_variation(move):
            next_node = self._current_node.variation(move)
        else:
            next_node = self._current_node.add_variation(move)

        self.set_current_node(next_node)
        return details

    def undo(self) -> bool:
        parent = self._current_node.parent
        if parent is None:
            return False
        self.set_current_node(parent)
        return True

    def redo(self) -> bool:
        """
        Formal contract (Branch Exploration V1): `redo()` always follows
        `_active_child[current_node]` -- the child most recently made active
        FOR THIS PARENT by any navigation, not necessarily the child played
        most recently overall, nor first/last in `.variations` order.
        "Made active" happens uniformly through `set_current_node()`
        (`_mark_active_path`), whether the child was reached by playing a
        move, by `redo()` itself, or by a direct jump into a specific
        sibling (a Timeline branch badge, or the variation selector) -- there
        is deliberately no separate multi-way "redo, but choose among N"
        mode; reaching a non-active sibling is always a direct jump, which
        itself then becomes the new active child going forward. This is
        exactly the existing behavior below, stated as an intended policy
        rather than left as an implementation detail.
        """
        next_node = self._active_child.get(self._current_node)
        if next_node is None:
            return False
        self.set_current_node(next_node)
        return True

    def can_redo(self) -> bool:
        return self._active_child.get(self._current_node) is not None

    def go_to_start(self) -> None:
        """Jumps to the tree root in one navigation/one transition."""
        node = self._current_node
        while node.parent is not None:
            node = node.parent
        self.set_current_node(node)

    def go_to_end(self) -> None:
        """
        Walks the active line forward to its current tip via
        `_active_child`, then jumps there in ONE set_current_node call --
        one transition to the final target, not one per intermediate ply.
        """
        node = self._current_node
        while True:
            next_node = self._active_child.get(node)
            if next_node is None:
                break
            node = next_node
        self.set_current_node(node)

    def mainline_path(self) -> list[chess.pgn.GameNode]:
        """
        Root-to-current_node path, root first -- the line currently on
        screen, for a Timeline widget to render. Always derived from
        `.parent`, never from `_active_child`, so it reflects the line
        actually displayed regardless of what's recorded elsewhere in the
        tree.
        """
        path: list[chess.pgn.GameNode] = []
        node: chess.pgn.GameNode | None = self._current_node
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    def active_path(self) -> list[chess.pgn.GameNode]:
        """
        Root-to-branch-tip path, root first (Phase 5e.2: continuous Timeline
        scrubbing). Extends `mainline_path()` (root..current_node, via
        `.parent`) with a forward walk from `current_node` via
        `_active_child` to the tip of the currently active branch -- the
        same forward-walking logic `go_to_end()` already performs, returning
        the visited nodes as a list instead of jumping to the last one.
        `current_node` itself appears exactly once, at the join between the
        two walks.

        This is a pure, read-only derived view (O(depth), same cost class as
        `mainline_path()`/`go_to_end()`) -- no new field, no new signal.
        `mainline_path()` itself is untouched; this is an addition alongside
        it, not a replacement, since existing consumers (the discrete
        Timeline strip) still want root..current only.
        """
        path = self.mainline_path()
        node = self._current_node
        while True:
            next_node = self._active_child.get(node)
            if next_node is None:
                break
            path.append(next_node)
            node = next_node
        return path

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
