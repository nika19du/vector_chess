import chess

from desktop_app.full_position_analysis import build_full_position_analysis
from desktop_app.layers.attack_influence_layer import ATTACK_INFLUENCE_LAYER
from desktop_app.main_window import MainWindow
from desktop_app.transition_controller import TransitionController
from tests.conftest import ready_cache_entry


def _expected_vertex_colors(board: chess.Board):
    entry = ready_cache_entry(board)
    frame = ATTACK_INFLUENCE_LAYER.data_source(entry)
    return ATTACK_INFLUENCE_LAYER.renderer(frame)


class FakeClock:
    """A controllable stand-in for time.monotonic, for deterministic transition tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _install_fake_clock_controller(window: MainWindow) -> FakeClock:
    clock = FakeClock()
    window.transition_controller = TransitionController(
        canvas=window.canvas,
        session_state=window.session_state,
        position_cache=window.position_cache,
        on_settled=window._render_all_layers,
        now_fn=clock,
    )
    window.transition_controller.start_transition(window.session_state.current_node.board().board_fen())
    return clock


def _tick(window: MainWindow, clock: FakeClock, seconds: float) -> None:
    window.transition_controller._timer.stop()
    clock.advance(seconds)
    window.transition_controller._on_tick()


# ---------------------------------------------------------
# board + canvas stay synchronized with Timeline navigation
# ---------------------------------------------------------


def test_headless_construction_with_timeline_panel(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    assert window.timeline_panel is not None


def test_clicking_a_distant_history_entry_syncs_board_and_canvas(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)

    window.session_state.make_move(chess.Move.from_uci("e2e4"))
    window.session_state.make_move(chess.Move.from_uci("e7e5"))
    window.session_state.make_move(chess.Move.from_uci("g1f3"))
    first_move_node = window.session_state.mainline_path()[1]
    qtbot.waitUntil(
        lambda: window.canvas._overlay_colors is not None
        and (window.canvas._overlay_colors == _expected_vertex_colors(window.session_state.current_node.board())).all(),
        timeout=5000,
    )

    window.timeline_panel._move_buttons[first_move_node].click()

    expected = _expected_vertex_colors(first_move_node.board())
    qtbot.waitUntil(
        lambda: window.canvas._overlay_colors is not None and (window.canvas._overlay_colors == expected).all(),
        timeout=5000,
    )
    assert window.session_state.current_node is first_move_node
    assert window.board_panel.board_view._board_node is first_move_node


def test_revisiting_a_cached_position_via_timeline_does_not_recompute(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)

    window.session_state.make_move(chess.Move.from_uci("e2e4"))
    moved_node = window.session_state.current_node
    qtbot.waitUntil(
        lambda: window.canvas._overlay_colors is not None
        and (window.canvas._overlay_colors == _expected_vertex_colors(moved_node.board())).all(),
        timeout=5000,
    )

    call_count = {"n": 0}

    def counting_builder(board: chess.Board):
        call_count["n"] += 1
        return build_full_position_analysis(board)

    window.position_cache._builder = counting_builder

    window.timeline_panel._first_button.click()  # back to root -- already cached
    window.timeline_panel._next_button.click()  # forward again -- already cached
    qtbot.wait(100)

    assert call_count["n"] == 0


def test_distant_jump_produces_one_transition_not_one_per_ply(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    starting_fen = window.session_state.current_node.board().board_fen()

    window.session_state.make_move(chess.Move.from_uci("e2e4"))
    window.session_state.make_move(chess.Move.from_uci("e7e5"))
    window.session_state.make_move(chess.Move.from_uci("g1f3"))
    tip_fen = window.session_state.current_node.board().board_fen()
    qtbot.waitUntil(lambda: window.transition_controller._settled_fen == tip_fen, timeout=5000)

    clock = _install_fake_clock_controller(window)

    window.timeline_panel._first_button.click()  # jump directly to root, 3 plies back

    qtbot.waitUntil(lambda: window.transition_controller._to_fen == starting_fen, timeout=5000)
    assert window.transition_controller._from_fen == tip_fen
    assert window.transition_controller._to_fen == starting_fen

    _tick(window, clock, seconds=1.0)
    assert window.transition_controller._settled_fen == starting_fen


def test_rapid_next_clicks_collapse_to_one_animation(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)

    window.session_state.make_move(chess.Move.from_uci("e2e4"))
    window.session_state.make_move(chess.Move.from_uci("e7e5"))
    tip_fen = window.session_state.current_node.board().board_fen()
    qtbot.waitUntil(lambda: window.transition_controller._settled_fen == tip_fen, timeout=5000)

    window.session_state.go_to_start()
    root_fen = window.session_state.current_node.board().board_fen()
    qtbot.waitUntil(lambda: window.transition_controller._settled_fen == root_fen, timeout=5000)

    clock = _install_fake_clock_controller(window)

    window.timeline_panel._next_button.click()
    window.timeline_panel._next_button.click()  # interrupts the first, before it settles

    qtbot.waitUntil(lambda: window.transition_controller._to_fen == tip_fen, timeout=5000)
    assert window.transition_controller._from_fen == root_fen

    _tick(window, clock, seconds=1.0)
    assert window.transition_controller._settled_fen == tip_fen


def test_no_analysis_recomputation_during_timeline_triggered_animation_ticks(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    clock = _install_fake_clock_controller(window)

    window.session_state.make_move(chess.Move.from_uci("e2e4"))
    moved_fen = window.session_state.current_node.board().board_fen()
    qtbot.waitUntil(lambda: window.transition_controller._to_fen == moved_fen, timeout=5000)
    window.transition_controller._timer.stop()

    call_count = {"n": 0}

    def counting_builder(board: chess.Board):
        call_count["n"] += 1
        return build_full_position_analysis(board)

    window.position_cache._builder = counting_builder

    for _ in range(10):
        _tick(window, clock, seconds=0.05)

    assert call_count["n"] == 0


# ---------------------------------------------------------
# branching through the Timeline UI
# ---------------------------------------------------------


def test_old_branch_remains_reachable_and_restores_its_exact_position_after_a_new_move(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)

    window.session_state.make_move(chess.Move.from_uci("e2e4"))
    old_branch_node = window.session_state.current_node
    old_expected = _expected_vertex_colors(old_branch_node.board())
    qtbot.waitUntil(
        lambda: window.canvas._overlay_colors is not None and (window.canvas._overlay_colors == old_expected).all(),
        timeout=5000,
    )

    window.timeline_panel._previous_button.click()  # back to root
    window.session_state.make_move(chess.Move.from_uci("d2d4"))  # new branch
    new_branch_node = window.session_state.current_node
    new_expected = _expected_vertex_colors(new_branch_node.board())
    qtbot.waitUntil(
        lambda: window.canvas._overlay_colors is not None and (window.canvas._overlay_colors == new_expected).all(),
        timeout=5000,
    )

    # Switch back into the old branch via the Timeline's branch badge -- the
    # badge for new_branch_node is visible while it's on the displayed path,
    # i.e. right now, while it's the current node (no extra navigation
    # needed to reveal it).
    assert window.session_state.current_node is new_branch_node
    window.timeline_panel._branch_badges[new_branch_node].click()  # expand siblings at the branch point

    sibling_button = next(
        window.timeline_panel._moves_layout.itemAt(i).widget()
        for i in range(window.timeline_panel._moves_layout.count())
        if window.timeline_panel._moves_layout.itemAt(i).widget() is not None
        and window.timeline_panel._moves_layout.itemAt(i).widget().text() == "(or: e4)"
    )
    sibling_button.click()

    assert window.session_state.current_node is old_branch_node
    qtbot.waitUntil(
        lambda: window.canvas._overlay_colors is not None and (window.canvas._overlay_colors == old_expected).all(),
        timeout=5000,
    )


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


def test_variation_selector_switch_syncs_board_and_canvas(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)

    e4_node, d4_node, _ = _three_siblings(window.session_state)
    qtbot.wait(10)

    window.session_state.set_current_node(e4_node)
    qtbot.waitUntil(
        lambda: window.canvas._overlay_colors is not None
        and (window.canvas._overlay_colors == _expected_vertex_colors(e4_node.board())).all(),
        timeout=5000,
    )

    window.timeline_panel._variation_next_buttons[e4_node].click()  # e4 -> d4, a cross-branch jump

    expected = _expected_vertex_colors(d4_node.board())
    qtbot.waitUntil(
        lambda: window.canvas._overlay_colors is not None and (window.canvas._overlay_colors == expected).all(),
        timeout=5000,
    )
    assert window.session_state.current_node is d4_node
    assert window.board_panel.board_view._board_node is d4_node


def test_variation_selector_switch_to_a_cached_branch_does_not_recompute(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)

    e4_node, d4_node, _ = _three_siblings(window.session_state)
    qtbot.wait(10)

    # Warm the cache for both e4 and d4 by visiting each once already.
    window.session_state.set_current_node(e4_node)
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    window.session_state.set_current_node(d4_node)
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    window.session_state.set_current_node(e4_node)
    qtbot.wait(10)

    call_count = {"n": 0}

    def counting_builder(board: chess.Board):
        call_count["n"] += 1
        return build_full_position_analysis(board)

    window.position_cache._builder = counting_builder

    window.timeline_panel._variation_next_buttons[e4_node].click()  # e4 -> d4, both already cached
    qtbot.wait(100)

    assert call_count["n"] == 0


def test_variation_selector_switch_produces_one_transition_with_no_stale_state(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)

    e4_node, d4_node, _ = _three_siblings(window.session_state)
    e4_fen = e4_node.board().board_fen()
    d4_fen = d4_node.board().board_fen()
    qtbot.wait(10)

    window.session_state.set_current_node(e4_node)
    qtbot.waitUntil(lambda: window.transition_controller._settled_fen == e4_fen, timeout=5000)

    clock = _install_fake_clock_controller(window)

    window.timeline_panel._variation_next_buttons[e4_node].click()  # e4 -> d4, a cross-branch jump

    qtbot.waitUntil(lambda: window.transition_controller._to_fen == d4_fen, timeout=5000)
    assert window.transition_controller._from_fen == e4_fen  # correct endpoints, no stale target

    _tick(window, clock, seconds=1.0)
    assert window.transition_controller._settled_fen == d4_fen


# ---------------------------------------------------------
# clean shutdown still works
# ---------------------------------------------------------


def test_close_event_still_shuts_down_the_position_cache_cleanly(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)

    window.session_state.make_move(chess.Move.from_uci("e2e4"))
    qtbot.wait(10)  # let the deferred TimelinePanel rebuild run before closing/GC
    window.timeline_panel._first_button.click()
    qtbot.wait(10)  # drain the rebuild this click itself scheduled, same reason

    window.close()

    assert window.position_cache._shutdown_requested.is_set()
