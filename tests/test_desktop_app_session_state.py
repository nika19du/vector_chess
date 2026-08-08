import chess
import chess.pgn

from desktop_app.session_state import DEFAULT_LAYER_VISIBILITY, SessionState
from desktop_app.transition import TransitionState


def _root() -> chess.pgn.Game:
    return chess.pgn.Game()


def test_current_node_starts_as_the_given_node():
    root = _root()
    state = SessionState(root)

    assert state.current_node.board().board_fen() == root.board().board_fen()


def test_current_node_changed_fires_on_an_actual_position_change(qtbot):
    root = _root()
    state = SessionState(root)
    moved = root.add_variation(chess.Move.from_uci("e2e4"))

    with qtbot.waitSignal(state.current_node_changed, timeout=1000) as blocker:
        state.set_current_node(moved)

    assert blocker.args[0].board().board_fen() == moved.board().board_fen()
    assert state.current_node.board().board_fen() == moved.board().board_fen()


def test_current_node_changed_does_not_fire_for_the_same_position(qtbot):
    root = _root()
    state = SessionState(root)

    received = []
    state.current_node_changed.connect(lambda node: received.append(node))

    # A "different" node object that resolves to the exact same position must
    # not be treated as a real change -- set_current_node compares by
    # resulting board_fen, not node identity.
    same_position_root = _root()
    state.set_current_node(same_position_root)
    qtbot.wait(50)

    assert received == []


def test_layer_visible_defaults_match_default_layer_visibility():
    state = SessionState(_root())

    for layer_id, expected in DEFAULT_LAYER_VISIBILITY.items():
        assert state.layer_visible(layer_id) == expected

    # An unregistered layer id defaults to visible, not hidden or an error --
    # visibility is opt-out, not opt-in, matching the frozen design's "solo
    # hides everything else" framing rather than "nothing shows until told to."
    assert state.layer_visible("not_a_real_layer") is True


def test_layer_state_changed_fires_on_an_actual_visibility_change(qtbot):
    state = SessionState(_root())

    with qtbot.waitSignal(state.layer_state_changed, timeout=1000) as blocker:
        state.set_layer_visible("attack_influence", False)

    assert blocker.args[0] == "attack_influence"
    assert state.layer_visible("attack_influence") is False


def test_layer_state_changed_does_not_fire_for_a_redundant_write(qtbot):
    state = SessionState(_root())

    received = []
    state.layer_state_changed.connect(lambda layer_id: received.append(layer_id))

    state.set_layer_visible("attack_influence", True)  # already True by default
    qtbot.wait(50)

    assert received == []


# ---------------------------------------------------------
# layer opacity (same layer_state slice/signal as visibility)
# ---------------------------------------------------------


def test_layer_opacity_defaults_to_full_opacity():
    state = SessionState(_root())

    for layer_id in DEFAULT_LAYER_VISIBILITY:
        assert state.layer_opacity(layer_id) == 1.0
    # An unregistered layer id also defaults to fully opaque, matching
    # layer_visible's own "opt-out, not opt-in" default for unknown ids.
    assert state.layer_opacity("not_a_real_layer") == 1.0


def test_set_layer_opacity_fires_layer_state_changed(qtbot):
    state = SessionState(_root())

    with qtbot.waitSignal(state.layer_state_changed, timeout=1000) as blocker:
        state.set_layer_opacity("gradient", 0.4)

    assert blocker.args[0] == "gradient"
    assert state.layer_opacity("gradient") == 0.4


def test_set_layer_opacity_does_not_fire_for_a_redundant_write(qtbot):
    state = SessionState(_root())

    received = []
    state.layer_state_changed.connect(lambda layer_id: received.append(layer_id))

    state.set_layer_opacity("gradient", 1.0)  # already 1.0 by default
    qtbot.wait(50)

    assert received == []


def test_set_layer_opacity_clamps_to_the_zero_to_one_range():
    state = SessionState(_root())

    state.set_layer_opacity("gradient", 1.5)
    assert state.layer_opacity("gradient") == 1.0

    state.set_layer_opacity("critical_points", -0.5)
    assert state.layer_opacity("critical_points") == 0.0


def test_session_state_only_exposes_the_slices_implemented_so_far():
    # Phase 4.1's full design names eight slices; three are implemented so
    # far (current_node, layer_state, and this milestone's transition_state).
    # This test guards against silently growing unused signal surface -- the
    # other slices are added by the phases that first populate them (5f, 5h).
    signal_names = {
        name
        for name in dir(SessionState)
        if name.endswith("_changed") and not name.startswith("_")
    }
    assert signal_names == {"current_node_changed", "layer_state_changed", "transition_state_changed"}


# ---------------------------------------------------------
# make_move
# ---------------------------------------------------------


def test_make_move_executes_a_legal_move_and_updates_current_node(qtbot):
    state = SessionState(_root())
    move = chess.Move.from_uci("e2e4")

    with qtbot.waitSignal(state.current_node_changed, timeout=1000):
        details = state.make_move(move)

    assert details is not None
    assert details.move == "e2e4"
    assert state.current_node.board().piece_at(chess.E4) is not None
    assert state.current_node.move == move


def test_make_move_rejects_an_illegal_move_and_leaves_current_node_unchanged(qtbot):
    state = SessionState(_root())
    original_node = state.current_node
    illegal_move = chess.Move.from_uci("e2e5")  # pawn cannot jump to e5 from e2

    received = []
    state.current_node_changed.connect(lambda node: received.append(node))

    details = state.make_move(illegal_move)
    qtbot.wait(50)

    assert details is None
    assert state.current_node is original_node
    assert received == []


def test_make_move_reuses_an_existing_child_instead_of_creating_a_duplicate():
    state = SessionState(_root())
    move = chess.Move.from_uci("e2e4")

    state.make_move(move)
    node_after_first_play = state.current_node
    state.undo()
    state.make_move(move)  # replaying the exact same move after undo

    assert state.current_node is node_after_first_play
    assert len(list(state.current_node.parent.variations)) == 1


# ---------------------------------------------------------
# undo / redo
# ---------------------------------------------------------


def test_undo_moves_current_node_to_its_parent(qtbot):
    state = SessionState(_root())
    state.make_move(chess.Move.from_uci("e2e4"))
    node_after_move = state.current_node

    with qtbot.waitSignal(state.current_node_changed, timeout=1000):
        undone = state.undo()

    assert undone is True
    assert state.current_node.move is None  # back at the root
    assert state.current_node is not node_after_move


def test_undo_at_the_root_is_a_no_op(qtbot):
    state = SessionState(_root())
    root_node = state.current_node

    received = []
    state.current_node_changed.connect(lambda node: received.append(node))

    undone = state.undo()
    qtbot.wait(50)

    assert undone is False
    assert state.current_node is root_node
    assert received == []


def test_redo_returns_to_the_node_that_was_undone(qtbot):
    state = SessionState(_root())
    state.make_move(chess.Move.from_uci("e2e4"))
    node_after_move = state.current_node
    state.undo()

    with qtbot.waitSignal(state.current_node_changed, timeout=1000):
        redone = state.redo()

    assert redone is True
    assert state.current_node is node_after_move


def test_redo_supports_multiple_levels():
    state = SessionState(_root())
    state.make_move(chess.Move.from_uci("e2e4"))
    node_1 = state.current_node
    state.make_move(chess.Move.from_uci("e7e5"))
    node_2 = state.current_node

    state.undo()
    state.undo()
    assert state.current_node.move is None

    assert state.redo() is True
    assert state.current_node is node_1
    assert state.redo() is True
    assert state.current_node is node_2


def test_redo_is_a_no_op_when_there_is_nothing_to_redo(qtbot):
    state = SessionState(_root())

    received = []
    state.current_node_changed.connect(lambda node: received.append(node))

    redone = state.redo()
    qtbot.wait(50)

    assert redone is False
    assert received == []


def test_playing_a_different_move_after_undo_creates_a_branch_not_corruption():
    state = SessionState(_root())
    state.make_move(chess.Move.from_uci("e2e4"))
    original_line_node = state.current_node
    root = original_line_node.parent

    state.undo()
    state.make_move(chess.Move.from_uci("d2d4"))  # a genuinely different move
    new_branch_node = state.current_node

    assert new_branch_node is not original_line_node
    # The original line is still intact, not deleted -- both are children of
    # the same root now.
    assert original_line_node in root.variations
    assert new_branch_node in root.variations
    assert len(root.variations) == 2


def test_a_new_branch_after_undo_invalidates_pending_redo(qtbot):
    state = SessionState(_root())
    state.make_move(chess.Move.from_uci("e2e4"))
    state.undo()

    received = []
    state.current_node_changed.connect(lambda node: received.append(node))

    state.make_move(chess.Move.from_uci("d2d4"))  # diverges from the undone line
    received.clear()

    redone = state.redo()
    qtbot.wait(50)

    assert redone is False
    assert received == []


# ---------------------------------------------------------
# transition_state (correspondence/animation milestone)
# ---------------------------------------------------------


def test_transition_state_starts_settled_at_the_initial_position():
    root = _root()
    state = SessionState(root)

    assert state.transition_state.from_fen is None
    assert state.transition_state.to_fen == root.board().board_fen()
    assert state.transition_state.progress == 1.0


def test_set_transition_state_fires_transition_state_changed(qtbot):
    state = SessionState(_root())
    new_state = TransitionState(from_fen="a", to_fen="b", progress=0.5)

    with qtbot.waitSignal(state.transition_state_changed, timeout=1000):
        state.set_transition_state(new_state)

    assert state.transition_state == new_state


def test_set_transition_state_does_not_fire_for_a_redundant_write(qtbot):
    state = SessionState(_root())
    same_state = state.transition_state

    received = []
    state.transition_state_changed.connect(lambda: received.append(True))

    state.set_transition_state(
        TransitionState(from_fen=same_state.from_fen, to_fen=same_state.to_fen, progress=same_state.progress)
    )
    qtbot.wait(50)

    assert received == []
