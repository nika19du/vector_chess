from __future__ import annotations

from dataclasses import dataclass

import chess.pgn


@dataclass(frozen=True)
class CompareState:
    """
    SessionState's record of an open Branch Comparison (Branch Comparison V1
    -- see the approved design report for the milestone this implements).

    `node_a`/`node_b` are canonical `chess.pgn.GameNode` references into the
    SAME tree `SessionState.current_node` already points into -- never
    copies, never a second game tree. Identity, not FEN equality, is what
    distinguishes them: two sibling nodes can legally share a resulting FEN
    (a transposition), and they must still remain distinguishable as "A" and
    "B" (see `are_valid_siblings` below and `SessionState.set_current_node`'s
    own no-op-guard docstring for the same FEN-vs-identity distinction
    elsewhere in this class).

    Comparison is read-only with respect to the game tree: entering or
    holding a `CompareState` never calls `add_variation`, never touches
    `_active_child`, and never changes `current_node`. `SessionState.
    current_node` remains authoritative for ordinary navigation the entire
    time a comparison is open.
    """

    node_a: chess.pgn.GameNode
    node_b: chess.pgn.GameNode


def are_valid_siblings(node_a: chess.pgn.GameNode, node_b: chess.pgn.GameNode) -> bool:
    """
    Branch Comparison V1's one validity rule (the approved design report,
    section 3): A and B must be two DISTINCT direct children of the SAME
    parent. This is the only domain V1 supports -- arbitrary two-node
    comparison is explicitly deferred (the correspondence/continuity
    reasoning that justifies "same parent" doesn't extend to positions with
    no shared immediate history).

    Distinctness and shared-parent are both checked by GameNode identity
    (`is`), never by board_fen() equality -- two nodes with the same
    resulting FEN are not automatically "the same node" or "not siblings";
    see the class docstring above.
    """
    if node_a is node_b:
        return False
    if node_a.parent is None or node_b.parent is None:
        return False
    return node_a.parent is node_b.parent
