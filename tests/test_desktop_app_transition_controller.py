import chess

from desktop_app.full_position_analysis import build_full_position_analysis
from desktop_app.layers.attack_influence_layer import build_attack_influence_frame, render_attack_influence_frame
from desktop_app.main_window import MainWindow
from desktop_app.position_cache import CacheEntryState
from desktop_app.transition_controller import TransitionController
from tests.conftest import ready_cache_entry


class FakeClock:
    """A controllable stand-in for time.monotonic, for deterministic transition tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _install_fake_clock_controller(window: MainWindow) -> FakeClock:
    """
    Replaces `window.transition_controller` with a fresh instance driven by
    a `FakeClock`, then bootstraps it to the position already on screen --
    otherwise the fresh controller's `_settled_fen` would start as None and
    the *next* move would settle instantly instead of animating.
    """
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
    window.transition_controller._timer.stop()  # real elapsed time must never drive these ticks
    clock.advance(seconds)
    window.transition_controller._on_tick()


# ---------------------------------------------------------
# startup / settling
# ---------------------------------------------------------


def test_first_position_settles_without_animating(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)

    assert window.transition_controller.is_animating is False
    assert window.session_state.transition_state.progress == 1.0
    assert window.session_state.transition_state.from_fen is None


# ---------------------------------------------------------
# a move starts and completes an animation
# ---------------------------------------------------------


def test_move_starts_an_animation_from_the_settled_position(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    starting_fen = window.session_state.current_node.board().board_fen()
    clock = _install_fake_clock_controller(window)

    window.session_state.make_move(chess.Move.from_uci("e2e4"))
    moved_fen = window.session_state.current_node.board().board_fen()
    qtbot.waitUntil(lambda: window.transition_controller._to_fen == moved_fen, timeout=5000)

    assert window.transition_controller.is_animating is True
    assert window.session_state.transition_state.from_fen == starting_fen
    assert window.session_state.transition_state.to_fen == moved_fen
    assert 0.0 <= window.session_state.transition_state.progress < 1.0


def test_transition_settles_exactly_at_the_target_after_enough_ticks(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    clock = _install_fake_clock_controller(window)

    window.session_state.make_move(chess.Move.from_uci("e2e4"))
    moved_fen = window.session_state.current_node.board().board_fen()
    qtbot.waitUntil(lambda: window.transition_controller._to_fen == moved_fen, timeout=5000)

    _tick(window, clock, seconds=1.0)  # well past TRANSITION_DURATION_MS

    assert window.transition_controller.is_animating is False
    assert window.session_state.transition_state.progress == 1.0
    assert window.session_state.transition_state.to_fen == moved_fen

    expected = render_attack_influence_frame(build_attack_influence_frame(ready_cache_entry(window.session_state.current_node.board())))
    assert (window.canvas._overlay_colors == expected).all()


def test_t0_frame_is_applied_synchronously_before_any_tick(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    starting_fen = window.session_state.current_node.board().board_fen()
    clock = _install_fake_clock_controller(window)

    from_entry = window.position_cache.get(starting_fen)
    expected_t0 = render_attack_influence_frame(build_attack_influence_frame(from_entry))

    window.session_state.make_move(chess.Move.from_uci("e2e4"))
    moved_fen = window.session_state.current_node.board().board_fen()
    qtbot.waitUntil(lambda: window.transition_controller._to_fen == moved_fen, timeout=5000)
    window.transition_controller._timer.stop()  # take manual control before any real tick can advance it

    # start_transition applies the t=0 frame synchronously, before the timer
    # ever fires -- the canvas must already show the exact previous state.
    assert (window.canvas._overlay_colors == expected_t0).all()


# ---------------------------------------------------------
# no mathematical recomputation during animation
# ---------------------------------------------------------


def test_no_analysis_recomputation_during_animation_ticks(qapp, qtbot):
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


def test_correspondence_is_computed_once_per_pair_not_per_frame(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    clock = _install_fake_clock_controller(window)

    window.session_state.make_move(chess.Move.from_uci("e2e4"))
    moved_fen = window.session_state.current_node.board().board_fen()
    qtbot.waitUntil(lambda: window.transition_controller._to_fen == moved_fen, timeout=5000)
    window.transition_controller._timer.stop()

    call_count = {"n": 0}
    real_builder = window.transition_controller.correspondence_cache._builder

    def counting_builder(entry_a, entry_b):
        call_count["n"] += 1
        return real_builder(entry_a, entry_b)

    window.transition_controller.correspondence_cache._builder = counting_builder
    window.transition_controller.correspondence_cache._entries.clear()  # force the next tick to (re)build once

    for _ in range(10):
        _tick(window, clock, seconds=0.05)

    assert call_count["n"] == 1


# ---------------------------------------------------------
# rapid retarget safety
# ---------------------------------------------------------


def test_rapid_successive_moves_retarget_cleanly_to_the_latest_position(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    starting_fen = window.session_state.current_node.board().board_fen()
    clock = _install_fake_clock_controller(window)

    window.session_state.make_move(chess.Move.from_uci("e2e4"))
    first_fen = window.session_state.current_node.board().board_fen()
    qtbot.waitUntil(lambda: window.transition_controller._to_fen == first_fen, timeout=5000)
    window.transition_controller._timer.stop()  # interrupt before it ever settles

    window.session_state.make_move(chess.Move.from_uci("e7e5"))
    second_fen = window.session_state.current_node.board().board_fen()
    qtbot.waitUntil(lambda: window.transition_controller._to_fen == second_fen, timeout=5000)

    # Retargeted from the original settled position, not from the abandoned
    # first_fen transition -- never a stale intermediate "from".
    assert window.transition_controller._from_fen == starting_fen

    _tick(window, clock, seconds=1.0)

    assert window.transition_controller.is_animating is False
    assert window.transition_controller._settled_fen == second_fen
    assert window.session_state.transition_state.to_fen == second_fen
    assert window.session_state.transition_state.progress == 1.0
    assert window.session_state.current_node.board().piece_at(chess.E5) is not None


# ---------------------------------------------------------
# undo/redo animation
# ---------------------------------------------------------


def test_undo_animates_back_to_the_previous_position(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    starting_fen = window.session_state.current_node.board().board_fen()
    clock = _install_fake_clock_controller(window)

    window.session_state.make_move(chess.Move.from_uci("e2e4"))
    moved_fen = window.session_state.current_node.board().board_fen()
    qtbot.waitUntil(lambda: window.transition_controller._to_fen == moved_fen, timeout=5000)
    _tick(window, clock, seconds=1.0)  # settle at moved_fen first
    assert window.transition_controller._settled_fen == moved_fen

    window.session_state.undo()
    qtbot.waitUntil(lambda: window.transition_controller._to_fen == starting_fen, timeout=5000)

    assert window.transition_controller._from_fen == moved_fen
    assert window.session_state.transition_state.to_fen == starting_fen

    _tick(window, clock, seconds=1.0)

    assert window.transition_controller._settled_fen == starting_fen
    assert window.session_state.current_node.board() == chess.Board()


# ---------------------------------------------------------
# layer controls remain functional during/after a transition
# ---------------------------------------------------------


def test_layer_visibility_toggle_works_mid_transition(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    clock = _install_fake_clock_controller(window)

    window.session_state.make_move(chess.Move.from_uci("e2e4"))
    moved_fen = window.session_state.current_node.board().board_fen()
    qtbot.waitUntil(lambda: window.transition_controller._to_fen == moved_fen, timeout=5000)
    window.transition_controller._timer.stop()
    assert window.session_state.transition_state.progress < 1.0  # genuinely mid-transition

    window.session_state.set_layer_visible("critical_points", False)

    assert window.canvas._layer_enabled["critical_points"] is False


def test_layer_visibility_toggle_persists_after_settling(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    clock = _install_fake_clock_controller(window)

    window.session_state.make_move(chess.Move.from_uci("e2e4"))
    moved_fen = window.session_state.current_node.board().board_fen()
    qtbot.waitUntil(lambda: window.transition_controller._to_fen == moved_fen, timeout=5000)
    window.transition_controller._timer.stop()

    window.session_state.set_layer_visible("critical_points", False)
    _tick(window, clock, seconds=1.0)  # settle

    assert window.canvas._layer_enabled["critical_points"] is False
