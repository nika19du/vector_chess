import chess
import chess.pgn

from desktop_app.session_state import SessionState
from desktop_app.timeline_panel import CURRENT_MOVE_HIGHLIGHT_STYLE, TimelinePanel


def _panel(qapp) -> tuple[TimelinePanel, SessionState]:
    session_state = SessionState(chess.pgn.Game())
    panel = TimelinePanel(session_state, None)
    return panel, session_state


# ---------------------------------------------------------
# nav buttons call the right SessionState methods
# ---------------------------------------------------------


def test_first_button_calls_go_to_start(qapp, qtbot):
    panel, session_state = _panel(qapp)
    session_state.make_move(chess.Move.from_uci("e2e4"))
    session_state.make_move(chess.Move.from_uci("e7e5"))
    root = session_state.mainline_path()[0]
    qtbot.wait(10)  # let the deferred rebuild enable _first_button before clicking it

    panel._first_button.click()

    assert session_state.current_node is root
    # Drain the rebuild this click itself scheduled before the test ends --
    # `panel` has no Qt parent here (unlike production's MainWindow-parented
    # instance), so a still-pending QTimer.singleShot outliving the test
    # would fire against an already-Python-GC'd object during pytest-qt's
    # own teardown event pump. Not a production hazard, just test hygiene.
    qtbot.wait(10)


def test_previous_button_calls_undo(qapp, qtbot):
    panel, session_state = _panel(qapp)
    session_state.make_move(chess.Move.from_uci("e2e4"))
    root = session_state.mainline_path()[0]
    qtbot.wait(10)

    panel._previous_button.click()

    assert session_state.current_node is root
    qtbot.wait(10)  # drain the rebuild this click scheduled -- see note above


def test_next_button_calls_redo(qapp, qtbot):
    panel, session_state = _panel(qapp)
    session_state.make_move(chess.Move.from_uci("e2e4"))
    tip = session_state.current_node
    session_state.undo()
    qtbot.wait(10)

    panel._next_button.click()

    assert session_state.current_node is tip
    qtbot.wait(10)  # drain the rebuild this click scheduled -- see note above


def test_last_button_calls_go_to_end(qapp, qtbot):
    panel, session_state = _panel(qapp)
    session_state.make_move(chess.Move.from_uci("e2e4"))
    session_state.make_move(chess.Move.from_uci("e7e5"))
    tip = session_state.current_node
    session_state.go_to_start()
    qtbot.wait(10)

    panel._last_button.click()

    assert session_state.current_node is tip
    qtbot.wait(10)  # drain the rebuild this click scheduled -- see note above


# ---------------------------------------------------------
# button enablement
# ---------------------------------------------------------


def test_first_and_previous_disabled_at_the_root(qapp):
    # Initial build (in __init__) is synchronous, not deferred -- no pump needed.
    panel, _ = _panel(qapp)

    assert panel._first_button.isEnabled() is False
    assert panel._previous_button.isEnabled() is False


def test_next_and_last_disabled_at_the_tip(qapp, qtbot):
    panel, session_state = _panel(qapp)
    session_state.make_move(chess.Move.from_uci("e2e4"))
    qtbot.wait(10)  # let the deferred rebuild triggered by current_node_changed run

    assert panel._next_button.isEnabled() is False
    assert panel._last_button.isEnabled() is False


def test_previous_and_next_enabled_in_the_middle_of_a_line(qapp, qtbot):
    panel, session_state = _panel(qapp)
    session_state.make_move(chess.Move.from_uci("e2e4"))
    session_state.make_move(chess.Move.from_uci("e7e5"))
    session_state.undo()
    qtbot.wait(10)

    assert panel._previous_button.isEnabled() is True
    assert panel._next_button.isEnabled() is True


# ---------------------------------------------------------
# clickable move-history entries
# ---------------------------------------------------------


def test_move_entries_use_san_labels(qapp, qtbot):
    panel, session_state = _panel(qapp)
    session_state.make_move(chess.Move.from_uci("e2e4"))
    session_state.make_move(chess.Move.from_uci("g8f6"))
    qtbot.wait(10)

    labels = [button.text() for button in panel._move_buttons.values()]
    assert labels == ["e4", "Nf6"]


def test_clicking_a_move_entry_navigates_directly_to_that_node(qapp, qtbot):
    panel, session_state = _panel(qapp)
    session_state.make_move(chess.Move.from_uci("e2e4"))
    first_move_node = session_state.current_node
    session_state.make_move(chess.Move.from_uci("e7e5"))
    qtbot.wait(10)  # let the deferred rebuild populate _move_buttons for both plies

    panel._move_buttons[first_move_node].click()

    assert session_state.current_node is first_move_node
    qtbot.wait(10)  # drain the rebuild this click scheduled (parentless test widget; see note above)


def test_the_current_nodes_entry_is_highlighted(qapp, qtbot):
    panel, session_state = _panel(qapp)
    session_state.make_move(chess.Move.from_uci("e2e4"))
    session_state.make_move(chess.Move.from_uci("e7e5"))
    qtbot.wait(10)

    current_button = panel._move_buttons[session_state.current_node]
    other_button = panel._move_buttons[session_state.mainline_path()[1]]

    assert current_button.styleSheet() == CURRENT_MOVE_HIGHLIGHT_STYLE
    assert other_button.styleSheet() != CURRENT_MOVE_HIGHLIGHT_STYLE


def test_displayed_row_grows_after_a_new_move_is_played(qapp, qtbot):
    panel, session_state = _panel(qapp)
    assert len(panel._move_buttons) == 0

    session_state.make_move(chess.Move.from_uci("e2e4"))
    qtbot.wait(10)
    assert len(panel._move_buttons) == 1

    session_state.make_move(chess.Move.from_uci("e7e5"))
    qtbot.wait(10)
    assert len(panel._move_buttons) == 2


def test_row_reflects_a_jump_back_after_undo(qapp, qtbot):
    panel, session_state = _panel(qapp)
    session_state.make_move(chess.Move.from_uci("e2e4"))
    session_state.make_move(chess.Move.from_uci("e7e5"))
    qtbot.wait(10)

    session_state.undo()
    qtbot.wait(10)

    assert len(panel._move_buttons) == 1
    assert list(panel._move_buttons.values())[0].text() == "e4"


# ---------------------------------------------------------
# branch badges / sibling selection
# ---------------------------------------------------------


def test_no_badge_when_there_is_no_branch(qapp, qtbot):
    panel, session_state = _panel(qapp)
    session_state.make_move(chess.Move.from_uci("e2e4"))
    qtbot.wait(10)

    assert panel._branch_badges == {}


def test_badge_appears_after_undo_and_a_different_move(qapp, qtbot):
    panel, session_state = _panel(qapp)
    session_state.make_move(chess.Move.from_uci("e2e4"))
    session_state.undo()
    session_state.make_move(chess.Move.from_uci("d2d4"))
    new_branch_node = session_state.current_node
    qtbot.wait(10)

    assert new_branch_node in panel._branch_badges
    assert panel._branch_badges[new_branch_node].text() == "↳1"


def test_clicking_the_badge_reveals_the_sibling_and_clicking_it_switches_branch(qapp, qtbot):
    panel, session_state = _panel(qapp)
    session_state.make_move(chess.Move.from_uci("e2e4"))
    old_branch_node = session_state.current_node
    session_state.undo()
    session_state.make_move(chess.Move.from_uci("d2d4"))
    new_branch_node = session_state.current_node
    qtbot.wait(10)  # let the deferred rebuild populate _branch_badges

    # Before expanding, only the active branch's own move button exists --
    # no stray sibling button anywhere in the layout.
    assert not any(
        panel._moves_layout.itemAt(i).widget().text() == "(or: e4)"
        for i in range(panel._moves_layout.count())
        if panel._moves_layout.itemAt(i).widget() is not None
    )

    panel._branch_badges[new_branch_node].click()  # badge-triggered rebuild is synchronous

    sibling_texts = [
        panel._moves_layout.itemAt(i).widget().text()
        for i in range(panel._moves_layout.count())
        if panel._moves_layout.itemAt(i).widget() is not None
    ]
    assert "(or: e4)" in sibling_texts

    # Clicking the revealed sibling switches into the old branch.
    sibling_button = next(
        panel._moves_layout.itemAt(i).widget()
        for i in range(panel._moves_layout.count())
        if panel._moves_layout.itemAt(i).widget() is not None
        and panel._moves_layout.itemAt(i).widget().text() == "(or: e4)"
    )
    sibling_button.click()

    assert session_state.current_node is old_branch_node
    qtbot.wait(10)  # drain the rebuild this click scheduled (parentless test widget; see note above)


def test_clicking_the_badge_again_collapses_the_sibling_row(qapp, qtbot):
    panel, session_state = _panel(qapp)
    session_state.make_move(chess.Move.from_uci("e2e4"))
    session_state.undo()
    session_state.make_move(chess.Move.from_uci("d2d4"))
    new_branch_node = session_state.current_node
    qtbot.wait(10)

    panel._branch_badges[new_branch_node].click()
    panel._branch_badges[new_branch_node].click()

    sibling_texts = [
        panel._moves_layout.itemAt(i).widget().text()
        for i in range(panel._moves_layout.count())
        if panel._moves_layout.itemAt(i).widget() is not None
    ]
    assert "(or: e4)" not in sibling_texts


# ---------------------------------------------------------
# variation selector (Branch Exploration V1)
# ---------------------------------------------------------


def _three_siblings(session_state):
    """Builds root -> {e4, d4, c4} (in that creation order) and returns the three nodes."""
    session_state.make_move(chess.Move.from_uci("e2e4"))
    e4_node = session_state.current_node
    session_state.undo()
    session_state.make_move(chess.Move.from_uci("d2d4"))
    d4_node = session_state.current_node
    session_state.undo()
    session_state.make_move(chess.Move.from_uci("c2c4"))
    c4_node = session_state.current_node
    return e4_node, d4_node, c4_node


def test_no_variation_selector_when_current_node_has_no_siblings(qapp, qtbot):
    panel, session_state = _panel(qapp)
    session_state.make_move(chess.Move.from_uci("e2e4"))
    qtbot.wait(10)

    assert panel._variation_indicators == {}
    assert panel._variation_prev_buttons == {}
    assert panel._variation_next_buttons == {}


def test_no_variation_selector_at_the_root(qapp):
    panel, _ = _panel(qapp)  # initial build: root only, synchronous

    assert panel._variation_indicators == {}


def test_variation_selector_shows_index_and_total_with_three_siblings(qapp, qtbot):
    panel, session_state = _panel(qapp)
    _, _, c4_node = _three_siblings(session_state)
    qtbot.wait(10)

    assert panel._variation_indicators[c4_node].text() == "variation 3/3"


def test_variation_prev_button_disabled_at_first_variation(qapp, qtbot):
    panel, session_state = _panel(qapp)
    e4_node, _, _ = _three_siblings(session_state)
    qtbot.wait(10)

    session_state.set_current_node(e4_node)  # index 0 of 3
    qtbot.wait(10)

    assert panel._variation_prev_buttons[e4_node].isEnabled() is False
    assert panel._variation_next_buttons[e4_node].isEnabled() is True


def test_variation_next_button_disabled_at_last_variation(qapp, qtbot):
    panel, session_state = _panel(qapp)
    _, _, c4_node = _three_siblings(session_state)  # c4 is index 2 of 3, already current
    qtbot.wait(10)

    assert panel._variation_next_buttons[c4_node].isEnabled() is False
    assert panel._variation_prev_buttons[c4_node].isEnabled() is True


def test_middle_variation_has_both_selector_buttons_enabled(qapp, qtbot):
    panel, session_state = _panel(qapp)
    _, d4_node, _ = _three_siblings(session_state)
    qtbot.wait(10)

    session_state.set_current_node(d4_node)  # index 1 of 3
    qtbot.wait(10)

    assert panel._variation_prev_buttons[d4_node].isEnabled() is True
    assert panel._variation_next_buttons[d4_node].isEnabled() is True


def test_clicking_next_variation_button_navigates_to_the_next_sibling(qapp, qtbot):
    panel, session_state = _panel(qapp)
    e4_node, d4_node, _ = _three_siblings(session_state)
    qtbot.wait(10)

    session_state.set_current_node(e4_node)
    qtbot.wait(10)

    panel._variation_next_buttons[e4_node].click()

    assert session_state.current_node is d4_node
    qtbot.wait(10)  # drain the rebuild this click scheduled (parentless test widget; see note above)


def test_clicking_previous_variation_button_navigates_to_the_previous_sibling(qapp, qtbot):
    panel, session_state = _panel(qapp)
    e4_node, d4_node, _ = _three_siblings(session_state)
    qtbot.wait(10)

    session_state.set_current_node(d4_node)
    qtbot.wait(10)

    panel._variation_prev_buttons[d4_node].click()

    assert session_state.current_node is e4_node
    qtbot.wait(10)


def test_selecting_a_variation_via_the_selector_becomes_the_new_active_child(qapp, qtbot):
    panel, session_state = _panel(qapp)
    e4_node, d4_node, _ = _three_siblings(session_state)  # c4 is active_child(root) right now
    qtbot.wait(10)

    session_state.set_current_node(e4_node)
    qtbot.wait(10)

    panel._variation_next_buttons[e4_node].click()  # e4 -> d4, via the selector
    qtbot.wait(10)

    # redo() from root must now follow d4 -- the selector click re-anchored
    # active_child(root), exactly like a badge/sibling click would.
    session_state.go_to_start()
    assert session_state.redo() is True
    assert session_state.current_node is d4_node


def test_rapid_variation_switching_coalesces_into_one_rebuild_with_the_latest_indicator(qapp, qtbot):
    panel, session_state = _panel(qapp)
    e4_node, d4_node, c4_node = _three_siblings(session_state)
    qtbot.wait(10)

    call_count = {"n": 0}
    original_rebuild = panel._rebuild

    def counting_rebuild():
        call_count["n"] += 1
        original_rebuild()

    panel._rebuild = counting_rebuild

    # Three rapid direct jumps between siblings, no event-loop turn between them.
    session_state.set_current_node(e4_node)
    session_state.set_current_node(d4_node)
    session_state.set_current_node(c4_node)

    assert call_count["n"] == 0  # nothing rebuilt yet -- all three just scheduled

    qtbot.wait(10)

    assert call_count["n"] == 1  # coalesced into exactly one rebuild
    assert panel._variation_indicators[c4_node].text() == "variation 3/3"
    assert e4_node not in panel._variation_indicators  # stale node's selector is gone, not stacked


# ---------------------------------------------------------
# architectural isolation
# ---------------------------------------------------------


def test_widget_holds_no_reference_to_canvas_or_board(qapp):
    panel, _ = _panel(qapp)

    forbidden_attribute_names = {"canvas", "board_panel", "position_cache", "transition_controller"}
    assert forbidden_attribute_names.isdisjoint(vars(panel))


# ---------------------------------------------------------
# Qt lifecycle regression: a nav button disabling itself from inside its
# own .click() call stack must not crash (see the approved Qt/native
# lifecycle investigation and desktop_app/timeline_panel.py's docstring).
# ---------------------------------------------------------


def test_previous_disabling_itself_via_its_own_click_is_deferred_and_safe(qapp, qtbot):
    """
    The exact hazardous shape: one move of history, Previous enabled: click
    Previous (returns to root). This used to crash (setEnabled(False) called
    on _previous_button from inside its own click's call stack). It must now
    (1) not crash, (2) leave _previous_button still enabled immediately
    after the click -- proving the rebuild was genuinely deferred, not just
    made safe some other way -- and (3) become disabled only once the
    deferred rebuild actually runs.
    """
    panel, session_state = _panel(qapp)
    session_state.make_move(chess.Move.from_uci("e2e4"))
    qtbot.wait(10)  # let the deferred rebuild from the move settle first
    assert panel._previous_button.isEnabled() is True

    panel._previous_button.click()  # must not crash

    # SessionState's own navigation is synchronous and unaffected by the
    # deferral -- already correct immediately after the click.
    assert session_state.current_node.parent is None

    # The button's own enabled state has NOT been touched yet -- the
    # rebuild that would disable it hasn't run.
    assert panel._previous_button.isEnabled() is True

    qtbot.wait(10)  # let the deferred rebuild run

    assert panel._previous_button.isEnabled() is False


def test_first_can_safely_disable_itself_after_reaching_root(qapp, qtbot):
    panel, session_state = _panel(qapp)
    session_state.make_move(chess.Move.from_uci("e2e4"))
    session_state.make_move(chess.Move.from_uci("e7e5"))
    qtbot.wait(10)
    assert panel._first_button.isEnabled() is True

    panel._first_button.click()  # must not crash; disables itself once settled

    assert session_state.current_node.parent is None
    assert panel._first_button.isEnabled() is True  # not yet -- deferred

    qtbot.wait(10)

    assert panel._first_button.isEnabled() is False


def test_next_and_last_boundary_transitions_after_their_own_click_are_safe(qapp, qtbot):
    panel, session_state = _panel(qapp)
    session_state.make_move(chess.Move.from_uci("e2e4"))
    session_state.make_move(chess.Move.from_uci("e7e5"))
    tip = session_state.current_node
    session_state.undo()
    qtbot.wait(10)
    assert panel._next_button.isEnabled() is True

    panel._next_button.click()  # redo() -> back at `tip`, which has no
    # children yet, so can_redo() becomes False -- Next (and Last) disable
    # themselves as a direct result of this click. Must not crash.

    assert session_state.current_node is tip
    assert panel._next_button.isEnabled() is True  # deferred, not yet applied
    assert panel._last_button.isEnabled() is True

    qtbot.wait(10)

    assert panel._next_button.isEnabled() is False
    assert panel._last_button.isEnabled() is False


def test_rapid_current_node_changes_coalesce_into_one_rebuild(qapp, qtbot):
    """
    Several current_node_changed signals arriving before the event loop's
    next turn must collapse into exactly one rebuild, and that one rebuild
    must reflect the LATEST current_node -- not the first one that
    triggered scheduling, and not a stale intermediate one.
    """
    panel, session_state = _panel(qapp)

    call_count = {"n": 0}
    original_rebuild = panel._rebuild

    def counting_rebuild():
        call_count["n"] += 1
        original_rebuild()

    panel._rebuild = counting_rebuild

    session_state.make_move(chess.Move.from_uci("e2e4"))
    session_state.make_move(chess.Move.from_uci("e7e5"))
    session_state.make_move(chess.Move.from_uci("g1f3"))
    final_node = session_state.current_node

    # Nothing has rebuilt yet -- all three signals are still just scheduled.
    assert call_count["n"] == 0
    assert panel._move_buttons == {}

    qtbot.wait(10)

    assert call_count["n"] == 1  # three signals coalesced into one rebuild
    assert final_node in panel._move_buttons
    assert panel._move_buttons[final_node].styleSheet() == CURRENT_MOVE_HIGHLIGHT_STYLE
    assert len(panel._move_buttons) == 3  # reflects the full, latest path -- not a stale intermediate one
