import chess
import chess.pgn
import pytest

from analysis.dynamics import analyze_dynamics
from audio.engine import AccentTrigger
from audio.live_state import sonification_state_from_mapping
from audio.mapping import build_audio_mapping
from chess_engine.analyzer import analyze_position
from chess_engine.moves import execute_move
from desktop_app.audio_controller import AudioController
from desktop_app.session_state import SessionState


class _SpyAudioEngine:
    """
    Test double matching AudioEngine's public surface AudioController
    actually calls (publish/push_event/shutdown) -- records everything so
    the controller's own translation/sequencing logic can be asserted
    precisely, independent of AudioEngine's internal synthesis (already
    covered by tests/test_audio_engine.py).
    """

    def __init__(self) -> None:
        self.published: list = []
        self.triggers: list[AccentTrigger] = []
        self.note_trigger_count = 0
        self.scrub_active_calls: list[bool] = []
        self.shutdown_calls = 0

    def publish(self, state) -> None:
        self.published.append(state)

    def push_event(self, trigger: AccentTrigger) -> None:
        self.triggers.append(trigger)

    def push_note_trigger(self) -> None:
        self.note_trigger_count += 1

    def set_scrub_active(self, active: bool) -> None:
        self.scrub_active_calls.append(active)

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _play(session_state: SessionState, moves: list[str]) -> list[chess.pgn.GameNode]:
    nodes = []
    for move_text in moves:
        details = session_state.make_move(chess.Move.from_uci(move_text))
        assert details is not None, f"expected {move_text} to be legal"
        nodes.append(session_state.current_node)
    return nodes


def _controller(qapp) -> tuple[SessionState, AudioController, _SpyAudioEngine]:
    session_state = SessionState(chess.pgn.Game())
    engine = _SpyAudioEngine()
    controller = AudioController(session_state, engine)
    return session_state, controller, engine


def _offline_mappings(moves: list[str]):
    """
    Independent parity reference: replays `moves` through the exact same
    execute_move/analyze_position/analyze_dynamics/build_audio_mapping
    pipeline console_app/main.py's REPL loop uses, entirely separate from
    SessionState/AudioController -- mirrors the rolling
    previous_analysis convention exactly.
    """

    board = chess.Board()
    previous_analysis = None
    mappings = []

    for move_text in moves:
        details = execute_move(board, move_text)
        assert details is not None, f"expected {move_text} to be legal"
        analysis = analyze_position(board, details)
        dynamics = (
            None
            if previous_analysis is None
            else analyze_dynamics(previous=previous_analysis, current=analysis)
        )
        mappings.append(build_audio_mapping(analysis, dynamics))
        previous_analysis = analysis

    return mappings


# ---------------------------------------------------------
# Root / first-node behavior
# ---------------------------------------------------------


def test_navigating_to_the_root_node_does_not_publish(qapp):
    session_state, controller, engine = _controller(qapp)

    _play(session_state, ["e2e4"])
    assert len(engine.published) == 1

    session_state.go_to_start()  # lands back on the root -- no move to sonify

    assert len(engine.published) == 1  # unchanged
    assert engine.triggers == []


def test_constructing_the_controller_at_the_root_publishes_nothing(qapp):
    session_state = SessionState(chess.pgn.Game())
    engine = _SpyAudioEngine()
    AudioController(session_state, engine)

    assert engine.published == []
    assert engine.triggers == []


# ---------------------------------------------------------
# One normal move
# ---------------------------------------------------------


def test_one_normal_move_publishes_exactly_one_state_with_no_trigger(qapp):
    session_state, controller, engine = _controller(qapp)

    _play(session_state, ["e2e4"])

    assert len(engine.published) == 1
    assert engine.triggers == []

    expected = _offline_mappings(["e2e4"])[0]
    state = engine.published[0]
    assert state.pitch_hz == pytest.approx(expected.pitch_hz)
    assert state.harmonic_richness == expected.harmonic_richness
    assert state.loudness == expected.loudness
    assert state.harmony_interval_ratio == pytest.approx(expected.harmony_interval_ratio)


# ---------------------------------------------------------
# Phase 5f.4a: Melody/Harmony note-trigger fires on every committed
# move, independent of the capture/check accent trigger.
# ---------------------------------------------------------


def test_a_quiet_normal_move_still_pushes_a_note_trigger(qapp):
    session_state, controller, engine = _controller(qapp)

    _play(session_state, ["e2e4"])

    assert engine.triggers == []  # no accent -- Nf3-style quiet move
    assert engine.note_trigger_count == 1  # but the note voices still retrigger


def test_note_trigger_count_matches_publish_count_across_a_mixed_sequence(qapp):
    session_state, controller, engine = _controller(qapp)

    _play(session_state, ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "f1b5"])
    session_state.undo()
    session_state.redo()

    # Every committed navigation retriggers Melody/Harmony -- captures,
    # checks, and ordinary quiet moves alike -- one-to-one with publish.
    assert engine.note_trigger_count == len(engine.published)
    # unaffected by this phase: redo lands back on the checking move
    # (f1b5), so the accent trigger count is unchanged from what it
    # already was pre-5f.4a -- 3 from the original sequence, plus 1 more
    # from redo re-arriving at that same checking node.
    assert len(engine.triggers) == 4


def test_navigating_to_the_root_node_pushes_no_note_trigger(qapp):
    session_state, controller, engine = _controller(qapp)

    _play(session_state, ["e2e4"])
    assert engine.note_trigger_count == 1

    session_state.go_to_start()  # root -- nothing to sonify

    assert engine.note_trigger_count == 1  # unchanged


def test_no_note_trigger_after_shutdown(qapp):
    session_state, controller, engine = _controller(qapp)
    controller.shutdown()

    _play(session_state, ["e2e4"])

    assert engine.note_trigger_count == 0


# ---------------------------------------------------------
# Capture / check / capture+check
# ---------------------------------------------------------


def test_pure_capture_pushes_exactly_one_capture_trigger(qapp):
    session_state, controller, engine = _controller(qapp)

    _play(session_state, ["e2e4", "d7d5", "e4d5"])  # 1.e4 d5 2.exd5 -- capture, no check

    assert len(engine.triggers) == 1
    assert engine.triggers[0].kind == "capture"


def test_pure_check_pushes_exactly_one_check_trigger(qapp):
    session_state, controller, engine = _controller(qapp)

    _play(session_state, ["e2e4", "d7d5", "f1b5"])  # 1.e4 d5 2.Bb5+ -- check, no capture

    assert len(engine.triggers) == 1
    assert engine.triggers[0].kind == "check"


def test_capturing_check_pushes_exactly_one_combined_trigger_not_two(qapp):
    session_state, controller, engine = _controller(qapp)

    # 1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6?? 4.Qxf7+ -- a single move that is both
    # a capture and a check.
    _play(session_state, ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"])

    assert len(engine.triggers) == 1  # one move, one trigger -- not two
    assert engine.triggers[0].kind == "capture+check"


def test_a_quiet_move_pushes_no_trigger(qapp):
    session_state, controller, engine = _controller(qapp)

    _play(session_state, ["g1f3"])  # Nf3 -- no capture, no check

    assert engine.triggers == []


# ---------------------------------------------------------
# White vs. black timbre
# ---------------------------------------------------------


def test_white_move_gets_richness_one_black_move_gets_richness_three(qapp):
    session_state, controller, engine = _controller(qapp)

    _play(session_state, ["e2e4", "e7e5"])

    assert engine.published[0].harmonic_richness == 1  # white
    assert engine.published[1].harmonic_richness == 3  # black


# ---------------------------------------------------------
# Balance/harmony parity and loudness parity across a sequence
# ---------------------------------------------------------


def test_harmony_and_loudness_match_the_offline_pipeline_across_a_sequence(qapp):
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]
    session_state, controller, engine = _controller(qapp)

    _play(session_state, moves)

    expected_mappings = _offline_mappings(moves)
    assert len(engine.published) == len(expected_mappings)

    for published_state, expected in zip(engine.published, expected_mappings):
        assert published_state.harmony_interval_ratio == pytest.approx(expected.harmony_interval_ratio)
        assert published_state.harmony_above_melody == (expected.attack_influence_balance >= 0)
        assert published_state.loudness == pytest.approx(expected.loudness)
        assert published_state.pitch_hz == pytest.approx(expected.pitch_hz)


# ---------------------------------------------------------
# Undo / redo
# ---------------------------------------------------------


def test_undo_sonifies_the_node_it_lands_on_not_the_move_just_left(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])  # published[0]=e4 (white), published[1]=e5 (black)

    session_state.undo()  # lands back on the e4 node

    assert len(engine.published) == 3
    assert engine.published[2].harmonic_richness == engine.published[0].harmonic_richness  # white again
    assert engine.published[2] == engine.published[0]  # same node -> identical settled state


def test_redo_after_undo_matches_the_original_move_again(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])

    session_state.undo()
    session_state.redo()

    assert len(engine.published) == 4
    assert engine.published[3] == engine.published[1]  # back to the e5 node's own state


# ---------------------------------------------------------
# Branch creation / switching
# ---------------------------------------------------------


def test_playing_a_different_move_after_undo_creates_a_branch_with_fresh_analysis(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])  # main line: 1.e4 e5

    session_state.undo()  # back to the e4 node -- black to move (publishes e4's own state again)
    _play(session_state, ["b8c6"])  # a different black reply -> new branch (1...Nc6)

    assert len(engine.published) == 4  # e4, e5, undo->e4, b8c6
    branch_state = engine.published[3]
    main_line_state = engine.published[1]  # the original e5 node's state
    # Phase B1: pitch is no longer a reliable "this is genuinely
    # different data" proxy -- e5 and c6 now happen to quantize to the
    # same scale tone under the compressed 2-octave register (lossy by
    # design: 64 squares snap to 15 discrete tones, preserving ordering
    # but not uniqueness). segment_key is the FEN-pair identity itself
    # and is guaranteed to differ for a genuinely different branch
    # regardless of any future pitch/scale tuning.
    assert branch_state.segment_key != main_line_state.segment_key


def test_switching_back_to_the_original_branch_reuses_its_cached_state(qapp):
    session_state, controller, engine = _controller(qapp)
    e4_node, e5_node = _play(session_state, ["e2e4", "e7e5"])

    session_state.undo()
    _play(session_state, ["b8c6"])  # branch away (1...Nc6 instead of 1...e5)

    session_state.set_current_node(e5_node)  # switch back to the original branch

    assert len(engine.published) == 5  # e4, e5, undo->e4, b8c6, switch back to e5
    assert engine.published[4] == engine.published[1]  # identical to e5's original settled state


# ---------------------------------------------------------
# Distant Timeline jump
# ---------------------------------------------------------


def test_a_distant_jump_publishes_exactly_once_for_the_target_node(qapp):
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]
    session_state, controller, engine = _controller(qapp)
    nodes = _play(session_state, moves)
    assert len(engine.published) == 5

    session_state.set_current_node(nodes[1])  # jump straight from ply 5 to ply 2 (e7e5)

    assert len(engine.published) == 6  # one publish, not four (no intermediate plies)
    assert engine.published[5] == engine.published[1]  # matches e7e5's own original state


def test_a_distant_jump_computes_dynamics_against_the_targets_real_parent(qapp):
    """
    Jumping from ply 5 straight to ply 3 must compare ply 3 against ply
    2's analysis (its real parent), never against whatever node was
    "current" immediately before the jump.
    """

    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]
    session_state, controller, engine = _controller(qapp)
    nodes = _play(session_state, moves)

    session_state.set_current_node(nodes[2])  # jump to ply 3 (Nf3)

    expected = _offline_mappings(moves)[2]
    jumped_state = engine.published[-1]
    assert jumped_state.loudness == pytest.approx(expected.loudness)
    assert jumped_state.harmony_interval_ratio == pytest.approx(expected.harmony_interval_ratio)


# ---------------------------------------------------------
# Node-cache reuse / no duplicate analysis calls
# ---------------------------------------------------------


def test_revisiting_a_node_does_not_recompute_its_analysis(qapp, monkeypatch):
    import desktop_app.audio_controller as audio_controller_module

    call_count = {"analyze_position": 0, "analyze_dynamics": 0}
    real_analyze_position = audio_controller_module.analyze_position
    real_analyze_dynamics = audio_controller_module.analyze_dynamics

    def counting_analyze_position(*args, **kwargs):
        call_count["analyze_position"] += 1
        return real_analyze_position(*args, **kwargs)

    def counting_analyze_dynamics(*args, **kwargs):
        call_count["analyze_dynamics"] += 1
        return real_analyze_dynamics(*args, **kwargs)

    monkeypatch.setattr(audio_controller_module, "analyze_position", counting_analyze_position)
    monkeypatch.setattr(audio_controller_module, "analyze_dynamics", counting_analyze_dynamics)

    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])
    assert call_count["analyze_position"] == 2
    assert call_count["analyze_dynamics"] == 1  # only the second move has a "previous"

    session_state.undo()
    session_state.redo()
    session_state.undo()
    session_state.redo()

    # Every visit after the first is a cache hit -- no growth.
    assert call_count["analyze_position"] == 2
    assert call_count["analyze_dynamics"] == 1


def test_a_distant_jump_back_to_a_previously_visited_node_is_also_a_cache_hit(qapp, monkeypatch):
    import desktop_app.audio_controller as audio_controller_module

    call_count = {"analyze_position": 0}
    real_analyze_position = audio_controller_module.analyze_position

    def counting_analyze_position(*args, **kwargs):
        call_count["analyze_position"] += 1
        return real_analyze_position(*args, **kwargs)

    monkeypatch.setattr(audio_controller_module, "analyze_position", counting_analyze_position)

    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]
    session_state, controller, engine = _controller(qapp)
    nodes = _play(session_state, moves)
    assert call_count["analyze_position"] == 5

    session_state.set_current_node(nodes[1])  # jump back to an already-visited node

    assert call_count["analyze_position"] == 5  # unchanged -- cache hit


# ---------------------------------------------------------
# Exactly one publish per committed navigation / correct trigger count
# ---------------------------------------------------------


def test_exactly_one_publish_per_committed_navigation_across_a_mixed_sequence(qapp):
    session_state, controller, engine = _controller(qapp)

    _play(session_state, ["e2e4", "e7e5", "g1f3"])  # 3 publishes
    session_state.undo()  # 1 publish (lands on e7e5)
    session_state.redo()  # 1 publish (back to g1f3)
    session_state.go_to_start()  # 0 publishes (root)
    session_state.go_to_end()  # 1 publish (back to g1f3, one jump)

    assert len(engine.published) == 6


def test_trigger_count_matches_exactly_the_capturing_or_checking_moves_in_a_sequence(qapp):
    session_state, controller, engine = _controller(qapp)

    # e4(quiet) d5(quiet) exd5(capture) ... Bb5+(check) later
    _play(session_state, ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "f1b5"])
    # moves: 1.e4 d5 2.exd5 Qxd5 3.Nc3 Qa5 4.Bb5+
    # captures: exd5 (white captures pawn), Qxd5 (black captures pawn) -> 2 capture triggers
    # checks: Bb5+ -> 1 check trigger
    assert len(engine.triggers) == 3
    kinds = [t.kind for t in engine.triggers]
    assert kinds == ["capture", "capture", "check"]


def test_rapid_committed_moves_produce_one_publish_and_correct_triggers_each(qapp):
    session_state, controller, engine = _controller(qapp)
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "b1c3", "g8f6"]

    _play(session_state, moves)

    assert len(engine.published) == len(moves)
    assert engine.triggers == []  # none of these moves capture or check


# ---------------------------------------------------------
# Lifecycle: shutdown
# ---------------------------------------------------------


def test_shutdown_disconnects_and_stops_publishing(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4"])
    assert len(engine.published) == 1

    controller.shutdown()
    _play(session_state, ["e7e5"])  # further navigation after shutdown

    assert len(engine.published) == 1  # unchanged -- controller stopped listening


def test_shutdown_calls_engine_shutdown_exactly_once_even_if_called_twice(qapp):
    session_state, controller, engine = _controller(qapp)

    controller.shutdown()
    controller.shutdown()

    assert engine.shutdown_calls == 1


def test_no_publish_or_trigger_after_shutdown_even_for_a_capturing_check(qapp):
    session_state, controller, engine = _controller(qapp)
    controller.shutdown()

    _play(session_state, ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"])

    assert engine.published == []
    assert engine.triggers == []
