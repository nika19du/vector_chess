import chess
import chess.pgn

from desktop_app.session_state import SessionState
from desktop_app.timeline_panel import TimelinePanel


def _panel(qapp) -> tuple[TimelinePanel, SessionState]:
    session_state = SessionState(chess.pgn.Game())
    panel = TimelinePanel(session_state, None)
    return panel, session_state


def _two_siblings(session_state: SessionState):
    """Builds root -> {e4, d4} and leaves current_node at d4 (the branch point currently on screen)."""
    session_state.make_move(chess.Move.from_uci("e2e4"))
    e4_node = session_state.current_node
    session_state.undo()
    session_state.make_move(chess.Move.from_uci("d2d4"))
    d4_node = session_state.current_node
    return e4_node, d4_node


def _three_siblings(session_state: SessionState):
    """Builds root -> {e4, d4, c4} and leaves current_node at c4."""
    e4_node, d4_node = _two_siblings(session_state)
    session_state.undo()
    session_state.make_move(chess.Move.from_uci("c2c4"))
    c4_node = session_state.current_node
    return e4_node, d4_node, c4_node


# ---------------------------------------------------------
# no affordance without siblings
# ---------------------------------------------------------


def test_no_compare_button_when_the_current_node_has_no_siblings(qapp, qtbot):
    panel, session_state = _panel(qapp)
    session_state.make_move(chess.Move.from_uci("e2e4"))
    qtbot.wait(10)

    assert panel._compare_buttons == {}


# ---------------------------------------------------------
# exactly two siblings: the button opens comparison directly
# ---------------------------------------------------------


def test_compare_button_appears_at_the_current_branch_point(qapp, qtbot):
    panel, session_state = _panel(qapp)
    _e4_node, d4_node = _two_siblings(session_state)
    qtbot.wait(10)

    assert d4_node in panel._compare_buttons


def test_clicking_compare_with_exactly_one_other_sibling_opens_comparison_directly(qapp, qtbot):
    panel, session_state = _panel(qapp)
    e4_node, d4_node = _two_siblings(session_state)
    qtbot.wait(10)

    panel._compare_buttons[d4_node].click()

    assert session_state.compare_state is not None
    assert session_state.compare_state.node_a is d4_node
    assert session_state.compare_state.node_b is e4_node


# ---------------------------------------------------------
# three or more siblings: an explicit picker row, not auto-comparison
# ---------------------------------------------------------


def test_compare_button_with_three_siblings_expands_a_picker_instead_of_opening_directly(qapp, qtbot):
    panel, session_state = _panel(qapp)
    e4_node, d4_node, c4_node = _three_siblings(session_state)
    qtbot.wait(10)

    panel._compare_buttons[c4_node].click()

    assert session_state.compare_state is None
    assert panel._comparing_from is c4_node
    assert set(panel._compare_target_buttons.keys()) == {e4_node, d4_node}


def test_picking_a_target_from_the_row_opens_comparison_with_the_correct_pair(qapp, qtbot):
    panel, session_state = _panel(qapp)
    e4_node, _d4_node, c4_node = _three_siblings(session_state)
    qtbot.wait(10)
    panel._compare_buttons[c4_node].click()

    panel._compare_target_buttons[e4_node].click()

    assert session_state.compare_state is not None
    assert session_state.compare_state.node_a is c4_node
    assert session_state.compare_state.node_b is e4_node
    assert panel._comparing_from is None


def test_cancel_collapses_the_picker_without_opening_comparison(qapp, qtbot):
    panel, session_state = _panel(qapp)
    _e4_node, _d4_node, c4_node = _three_siblings(session_state)
    qtbot.wait(10)
    panel._compare_buttons[c4_node].click()
    assert panel._comparing_from is c4_node

    panel._compare_cancel_buttons[c4_node].click()

    assert session_state.compare_state is None
    assert panel._comparing_from is None
    assert panel._compare_target_buttons == {}
    assert c4_node in panel._compare_buttons


# ---------------------------------------------------------
# navigation cancels a pending (not-yet-committed) picker
# ---------------------------------------------------------


def test_navigating_away_cancels_a_pending_picker_selection(qapp, qtbot):
    panel, session_state = _panel(qapp)
    e4_node, d4_node, c4_node = _three_siblings(session_state)
    qtbot.wait(10)
    panel._compare_buttons[c4_node].click()
    assert panel._comparing_from is c4_node

    session_state.set_current_node(e4_node)
    qtbot.wait(10)  # let the deferred rebuild run

    assert panel._comparing_from is None
    assert session_state.compare_state is None
