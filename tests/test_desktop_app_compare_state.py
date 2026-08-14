import chess
import chess.pgn
import pytest

from desktop_app.session_state import SessionState


def _root() -> chess.pgn.Game:
    return chess.pgn.Game()


def _siblings(root: chess.pgn.Game) -> tuple[chess.pgn.GameNode, chess.pgn.GameNode]:
    node_a = root.add_variation(chess.Move.from_uci("e2e4"))
    node_b = root.add_variation(chess.Move.from_uci("d2d4"))
    return node_a, node_b


# ---------------------------------------------------------
# enter/exit lifecycle
# ---------------------------------------------------------


def test_compare_state_starts_closed():
    state = SessionState(_root())

    assert state.compare_state is None


def test_enter_compare_opens_with_the_given_nodes():
    root = _root()
    state = SessionState(root)
    node_a, node_b = _siblings(root)

    state.enter_compare(node_a, node_b)

    assert state.compare_state is not None
    assert state.compare_state.node_a is node_a
    assert state.compare_state.node_b is node_b


def test_enter_compare_fires_compare_state_changed(qtbot):
    root = _root()
    state = SessionState(root)
    node_a, node_b = _siblings(root)

    with qtbot.waitSignal(state.compare_state_changed, timeout=1000):
        state.enter_compare(node_a, node_b)


def test_exit_compare_closes_and_fires_compare_state_changed(qtbot):
    root = _root()
    state = SessionState(root)
    node_a, node_b = _siblings(root)
    state.enter_compare(node_a, node_b)

    with qtbot.waitSignal(state.compare_state_changed, timeout=1000):
        state.exit_compare()

    assert state.compare_state is None


def test_exit_compare_is_a_no_op_when_already_closed(qtbot):
    state = SessionState(_root())

    received = []
    state.compare_state_changed.connect(lambda: received.append(True))

    state.exit_compare()
    qtbot.wait(50)

    assert received == []
    assert state.compare_state is None


def test_rapid_enter_exit_leaves_a_clean_closed_state(qtbot):
    root = _root()
    state = SessionState(root)
    node_a, node_b = _siblings(root)

    for _ in range(20):
        state.enter_compare(node_a, node_b)
        state.exit_compare()

    assert state.compare_state is None


# ---------------------------------------------------------
# sibling-only invariant (Branch Comparison V1's one validity rule)
# ---------------------------------------------------------


def test_enter_compare_rejects_the_same_node_twice():
    root = _root()
    state = SessionState(root)
    node_a, _node_b = _siblings(root)

    with pytest.raises(ValueError):
        state.enter_compare(node_a, node_a)


def test_enter_compare_rejects_the_root_node():
    root = _root()
    state = SessionState(root)
    node_a, _node_b = _siblings(root)

    with pytest.raises(ValueError):
        state.enter_compare(root, node_a)


def test_enter_compare_rejects_nodes_with_different_parents():
    root = _root()
    state = SessionState(root)
    node_a, _node_b = _siblings(root)
    grandchild = node_a.add_variation(chess.Move.from_uci("e7e5"))

    # grandchild's parent is node_a, not root -- not a sibling of node_a.
    with pytest.raises(ValueError):
        state.enter_compare(node_a, grandchild)


def test_enter_compare_rejects_arbitrary_nonsibling_nodes_anywhere_in_the_tree():
    root = _root()
    state = SessionState(root)
    node_a, node_b = _siblings(root)
    deep_a = node_a.add_variation(chess.Move.from_uci("e7e5"))
    deep_b = node_b.add_variation(chess.Move.from_uci("d7d5"))

    with pytest.raises(ValueError):
        state.enter_compare(deep_a, deep_b)


def test_enter_compare_accepts_a_third_sibling_pairing_when_three_or_more_exist():
    root = _root()
    state = SessionState(root)
    node_a, node_b = _siblings(root)
    node_c = root.add_variation(chess.Move.from_uci("c2c4"))

    state.enter_compare(node_a, node_c)

    assert state.compare_state.node_a is node_a
    assert state.compare_state.node_b is node_c

    state.exit_compare()
    state.enter_compare(node_b, node_c)

    assert state.compare_state.node_a is node_b
    assert state.compare_state.node_b is node_c


def test_enter_compare_raises_before_mutating_any_existing_open_comparison():
    root = _root()
    state = SessionState(root)
    node_a, node_b = _siblings(root)
    state.enter_compare(node_a, node_b)

    with pytest.raises(ValueError):
        state.enter_compare(node_a, node_a)

    # The invalid attempt must not have clobbered the already-open comparison.
    assert state.compare_state.node_a is node_a
    assert state.compare_state.node_b is node_b


# ---------------------------------------------------------
# GameNode identity, not FEN equality, distinguishes/validates A/B
# ---------------------------------------------------------


def test_matching_fen_does_not_make_non_sibling_nodes_valid():
    # A genuine transposition: 1.c4 Nf6 2.Nc3 and 1.Nc3 Nf6 2.c4 reach the
    # exact same resulting piece placement, but via two entirely different
    # immediate parents -- neither node's parent is the other's parent, so
    # they are not siblings and must be rejected regardless of FEN equality.
    # This is the identity-vs-FEN distinction `are_valid_siblings` is built
    # around (see `SessionState.set_current_node`'s own no-op-guard
    # docstring for the same distinction elsewhere in this class).
    root = _root()
    state = SessionState(root)

    order_one = root.add_variation(chess.Move.from_uci("c2c4"))
    order_one = order_one.add_variation(chess.Move.from_uci("g8f6"))
    order_one_end = order_one.add_variation(chess.Move.from_uci("b1c3"))

    order_two = root.add_variation(chess.Move.from_uci("b1c3"))
    order_two = order_two.add_variation(chess.Move.from_uci("g8f6"))
    order_two_end = order_two.add_variation(chess.Move.from_uci("c2c4"))

    assert order_one_end.board().board_fen() == order_two_end.board().board_fen()
    assert order_one_end.parent is not order_two_end.parent

    with pytest.raises(ValueError):
        state.enter_compare(order_one_end, order_two_end)


def test_two_sibling_nodes_remain_distinguishable_as_a_and_b_by_identity():
    root = _root()
    state = SessionState(root)
    node_a, node_b = _siblings(root)

    state.enter_compare(node_a, node_b)

    assert state.compare_state.node_a is not state.compare_state.node_b
    assert state.compare_state.node_a is node_a
    assert state.compare_state.node_b is node_b


# ---------------------------------------------------------
# read-only w.r.t. the game tree
# ---------------------------------------------------------


def test_entering_compare_does_not_change_current_node():
    root = _root()
    state = SessionState(root)
    node_a, node_b = _siblings(root)
    original_current = state.current_node

    state.enter_compare(node_a, node_b)

    assert state.current_node is original_current


def test_exiting_compare_does_not_change_current_node():
    root = _root()
    state = SessionState(root)
    node_a, node_b = _siblings(root)
    original_current = state.current_node
    state.enter_compare(node_a, node_b)

    state.exit_compare()

    assert state.current_node is original_current


def test_entering_compare_does_not_change_redo_target():
    root = _root()
    state = SessionState(root)
    node_a, node_b = _siblings(root)
    # node_a is the active child of root (it was navigated to last via
    # set_current_node, below), so redo() from root should still go to
    # node_a after a comparison of node_a vs node_b is opened and closed.
    state.set_current_node(node_a)
    state.undo()
    assert state.can_redo()

    state.enter_compare(node_a, node_b)
    state.exit_compare()

    assert state.redo() is True
    assert state.current_node is node_a


def test_entering_compare_creates_no_new_variations():
    root = _root()
    state = SessionState(root)
    node_a, node_b = _siblings(root)
    variation_count_before = len(root.variations)

    state.enter_compare(node_a, node_b)
    state.exit_compare()

    assert len(root.variations) == variation_count_before


# ---------------------------------------------------------
# navigation exits compare mode
# ---------------------------------------------------------


def test_set_current_node_exits_compare_mode():
    root = _root()
    state = SessionState(root)
    node_a, node_b = _siblings(root)
    state.enter_compare(node_a, node_b)

    state.set_current_node(node_a)

    assert state.compare_state is None


def test_set_current_node_exits_compare_mode_even_when_it_is_a_no_op():
    root = _root()
    state = SessionState(root)
    node_a, node_b = _siblings(root)
    state.enter_compare(node_a, node_b)

    # Same FEN as current_node (root) -- set_current_node's own guard makes
    # this a navigation no-op, but it's still a real navigation *action* and
    # must still exit compare mode.
    same_position_root = _root()
    state.set_current_node(same_position_root)

    assert state.compare_state is None


def test_make_move_exits_compare_mode():
    root = _root()
    state = SessionState(root)
    node_a, node_b = _siblings(root)
    state.enter_compare(node_a, node_b)

    state.make_move(chess.Move.from_uci("e2e4"))

    assert state.compare_state is None


def test_undo_exits_compare_mode():
    root = _root()
    state = SessionState(root)
    node_a, node_b = _siblings(root)
    state.set_current_node(node_a)
    state.enter_compare(node_a, node_b)

    state.undo()

    assert state.compare_state is None


def test_redo_exits_compare_mode():
    root = _root()
    state = SessionState(root)
    node_a, node_b = _siblings(root)
    state.set_current_node(node_a)
    state.undo()
    state.enter_compare(node_a, node_b)

    state.redo()

    assert state.compare_state is None


def test_go_to_start_exits_compare_mode():
    root = _root()
    state = SessionState(root)
    node_a, node_b = _siblings(root)
    state.set_current_node(node_a)
    state.enter_compare(node_a, node_b)

    state.go_to_start()

    assert state.compare_state is None


def test_go_to_end_exits_compare_mode():
    root = _root()
    state = SessionState(root)
    node_a, node_b = _siblings(root)
    state.set_current_node(node_a)
    state.go_to_start()
    state.enter_compare(node_a, node_b)

    state.go_to_end()

    assert state.compare_state is None
