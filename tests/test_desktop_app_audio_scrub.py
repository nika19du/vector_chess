import chess
import chess.pgn
import pytest

from analysis.attack_influence import build_attack_influence_field
from audio.mapping import harmony_interval_for_balance
from desktop_app.audio_controller import AudioController
from desktop_app.layers._lerp import ease_in_out
from desktop_app.scrub import ScrubPosition
from desktop_app.session_state import SessionState


class _SpyAudioEngine:
    """Records publish/push_event/shutdown calls -- same test double shape
    as tests/test_desktop_app_audio_controller.py's own, kept local rather
    than cross-imported (matches this codebase's existing per-file test
    helper convention)."""

    def __init__(self) -> None:
        self.published: list = []
        self.triggers: list = []
        self.note_trigger_count = 0
        self.scrub_active_calls: list[bool] = []
        self.shutdown_calls = 0

    def publish(self, state) -> None:
        self.published.append(state)

    def push_event(self, trigger) -> None:
        self.triggers.append(trigger)

    def push_note_trigger(self) -> None:
        self.note_trigger_count += 1

    def set_scrub_active(self, active: bool) -> None:
        self.scrub_active_calls.append(active)

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _controller(qapp) -> tuple[SessionState, AudioController, _SpyAudioEngine]:
    session_state = SessionState(chess.pgn.Game())
    engine = _SpyAudioEngine()
    controller = AudioController(session_state, engine)
    return session_state, controller, engine


def _play(session_state: SessionState, moves: list[str]) -> None:
    for move_text in moves:
        details = session_state.make_move(chess.Move.from_uci(move_text))
        assert details is not None, f"expected {move_text} to be legal"


ROOT_BALANCE = build_attack_influence_field(chess.Board()).balance


# ---------------------------------------------------------
# t=0 / t=1 endpoint behavior
# ---------------------------------------------------------


def test_scrub_at_t_one_reproduces_the_segment_uppers_settled_state(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])
    settled_e4_state = engine.published[0]

    path = session_state.active_path()
    controller.begin_scrub(path)
    controller.update_scrub(ScrubPosition(path_index=0, t=1.0))

    scrub_state = engine.published[-1]
    assert scrub_state.harmony_interval_ratio == pytest.approx(settled_e4_state.harmony_interval_ratio)
    assert scrub_state.harmony_above_melody == settled_e4_state.harmony_above_melody
    assert scrub_state.pitch_hz == settled_e4_state.pitch_hz
    assert scrub_state.harmonic_richness == settled_e4_state.harmonic_richness
    assert scrub_state.loudness == settled_e4_state.loudness


def test_scrub_at_t_zero_uses_the_lower_endpoints_balance(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])
    settled_e4_state = engine.published[0]  # e4 is not a check

    path = session_state.active_path()
    controller.begin_scrub(path)
    controller.update_scrub(ScrubPosition(path_index=0, t=0.0))

    scrub_state = engine.published[-1]
    expected_ratio = harmony_interval_for_balance(ROOT_BALANCE, is_check=False)
    assert scrub_state.harmony_interval_ratio == pytest.approx(expected_ratio)
    # Discrete identity is still e4's, even at t=0 -- held fixed for the
    # whole segment, never the lower endpoint's own (nonexistent) identity.
    assert scrub_state.pitch_hz == settled_e4_state.pitch_hz
    assert scrub_state.harmonic_richness == settled_e4_state.harmonic_richness
    assert scrub_state.loudness == settled_e4_state.loudness


def test_scrub_at_the_tip_reproduces_the_settled_state_via_the_identity_segment_trick(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5", "g1f3"])
    settled_nf3_state = engine.published[-1]

    path = session_state.active_path()
    controller.begin_scrub(path)
    controller.update_scrub(ScrubPosition(path_index=len(path) - 1, t=0.0))

    scrub_state = engine.published[-1]
    # Phase B1: a committed move's harmony_interval_ratio is now
    # snapped to HARMONY_INTERVAL_VOCABULARY (see
    # audio.mapping.build_audio_mapping), while scrub preview
    # deliberately keeps calling the raw, continuous
    # harmony_interval_for_balance directly (see that function's own
    # docstring) so a drag gesture still glides smoothly. The identity-
    # segment trick therefore no longer reproduces the settled state's
    # *quantized* ratio bit-for-bit -- it reproduces the same *raw*
    # value harmony_interval_for_balance gives for this move's own
    # balance, which is the correct ground truth for a continuous
    # preview. Discrete, move-identity fields (pitch/richness/loudness)
    # are unaffected -- both paths read them from the same already-
    # quantized segment_mapping.pitch_hz, held fixed per segment.
    board = chess.Board()
    for move_text in ["e2e4", "e7e5", "g1f3"]:
        board.push(chess.Move.from_uci(move_text))
    nf3_balance = build_attack_influence_field(board).balance
    expected_ratio = harmony_interval_for_balance(nf3_balance, is_check=False)

    assert scrub_state.harmony_interval_ratio == pytest.approx(expected_ratio)
    assert scrub_state.pitch_hz == settled_nf3_state.pitch_hz
    assert scrub_state.loudness == settled_nf3_state.loudness


def test_scrub_before_any_move_exists_publishes_nothing(qapp):
    session_state, controller, engine = _controller(qapp)
    path = session_state.active_path()
    assert len(path) == 1  # just the root

    controller.begin_scrub(path)
    controller.update_scrub(ScrubPosition(path_index=0, t=0.0))

    assert engine.published == []


# ---------------------------------------------------------
# Midpoint interpolation and raw-t (not eased) behavior
# ---------------------------------------------------------


def test_harmony_interval_at_midpoint_is_the_linear_average(qapp):
    session_state, controller, engine = _controller(qapp)
    # a2a3's balance (5.0) stays well under REFERENCE_MAX_BALANCE (20) --
    # unlike e2e4 (balance 40.0), which already saturates the dissonance
    # clamp at both t=0.5 and t=1.0, making "midpoint == linear average"
    # trivially true-but-uninformative there. This segment stays in the
    # unclamped linear region the whole way, actually exercising it.
    _play(session_state, ["a2a3"])

    path = session_state.active_path()
    controller.begin_scrub(path)

    controller.update_scrub(ScrubPosition(path_index=0, t=0.0))
    at_zero = engine.published[-1].harmony_interval_ratio
    controller.update_scrub(ScrubPosition(path_index=0, t=1.0))
    at_one = engine.published[-1].harmony_interval_ratio
    controller.update_scrub(ScrubPosition(path_index=0, t=0.5))
    at_half = engine.published[-1].harmony_interval_ratio

    assert at_zero != pytest.approx(at_one)  # sanity: genuinely different, not both clamped
    assert at_half == pytest.approx((at_zero + at_one) / 2)


def test_scrub_uses_raw_t_not_eased_t(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["a2a3"])  # same unclamped-region reasoning as above

    path = session_state.active_path()
    controller.begin_scrub(path)

    t = 0.25
    assert ease_in_out(t) != pytest.approx(t)  # sanity: easing is genuinely nonlinear here

    controller.update_scrub(ScrubPosition(path_index=0, t=0.0))
    at_zero = engine.published[-1].harmony_interval_ratio
    controller.update_scrub(ScrubPosition(path_index=0, t=1.0))
    at_one = engine.published[-1].harmony_interval_ratio
    controller.update_scrub(ScrubPosition(path_index=0, t=t))
    at_t = engine.published[-1].harmony_interval_ratio

    linear_expected = at_zero + (at_one - at_zero) * t
    eased_expected = at_zero + (at_one - at_zero) * ease_in_out(t)
    assert at_t == pytest.approx(linear_expected)
    assert at_t != pytest.approx(eased_expected, abs=1e-6)


# ---------------------------------------------------------
# Discrete fields fixed within a segment; segment crossing updates them
# ---------------------------------------------------------


def test_pitch_timbre_loudness_stay_constant_across_the_whole_segment(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])

    path = session_state.active_path()
    controller.begin_scrub(path)

    published = []
    for t in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
        controller.update_scrub(ScrubPosition(path_index=0, t=t))
        published.append(engine.published[-1])

    assert all(s.pitch_hz == published[0].pitch_hz for s in published)
    assert all(s.harmonic_richness == published[0].harmonic_richness for s in published)
    assert all(s.loudness == published[0].loudness for s in published)


def test_crossing_a_segment_boundary_updates_discrete_identity(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])  # e4 (white) then e5 (black)

    path = session_state.active_path()
    controller.begin_scrub(path)

    controller.update_scrub(ScrubPosition(path_index=0, t=0.9))  # segment (root, e4)
    in_first_segment = engine.published[-1]
    controller.update_scrub(ScrubPosition(path_index=1, t=0.1))  # segment (e4, e5)
    in_second_segment = engine.published[-1]

    assert in_first_segment.pitch_hz != in_second_segment.pitch_hz
    assert in_first_segment.harmonic_richness != in_second_segment.harmonic_richness  # white vs black
    assert in_first_segment.segment_key != in_second_segment.segment_key


# ---------------------------------------------------------
# No accent during preview; accent only on committed release
# ---------------------------------------------------------


def test_no_capture_accent_fires_while_scrubbing_across_a_capturing_segment(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "d7d5", "e4d5"])  # 2.exd5 -- capture

    path = session_state.active_path()
    engine.triggers.clear()  # discard the trigger the real commit already fired
    controller.begin_scrub(path)

    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        controller.update_scrub(ScrubPosition(path_index=1, t=t))  # segment (d5, exd5)

    assert engine.triggers == []


def test_no_check_accent_fires_while_scrubbing_across_a_checking_segment(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "d7d5", "f1b5"])  # 2.Bb5+ -- check

    path = session_state.active_path()
    engine.triggers.clear()
    controller.begin_scrub(path)

    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        controller.update_scrub(ScrubPosition(path_index=1, t=t))

    assert engine.triggers == []


def test_capture_fires_exactly_once_on_committed_release_after_scrubbing_across_it(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "d7d5"])
    engine.triggers.clear()

    path = session_state.active_path()  # [root, e4, d5] -- exd5 not played yet, so scrub
    # this test instead plays the capture, scrubs across it, then re-commits
    # to it (simulating a drag that ends back exactly where it started).
    node = session_state.make_move(chess.Move.from_uci("e4d5"))
    assert node is not None
    engine.triggers.clear()  # discard the trigger from committing the capture itself

    path = session_state.active_path()
    capture_node = session_state.current_node
    controller.begin_scrub(path)
    for t in (0.0, 0.3, 0.6, 1.0):
        controller.update_scrub(ScrubPosition(path_index=len(path) - 1, t=t))
    assert engine.triggers == []  # still nothing during preview

    controller.end_scrub()
    session_state.set_current_node(capture_node)  # the real release commit -- a no-op navigation

    # A no-op commit (already on capture_node) does not re-fire
    # current_node_changed at all (SessionState's own board_fen() guard) --
    # so no NEW trigger is expected here either; this exercises "release
    # after scrub never fires from the scrub path itself," not a re-fire.
    assert engine.triggers == []


def test_capture_fires_exactly_once_when_release_lands_on_a_different_capturing_node(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "d7d5"])

    path = session_state.active_path()  # [root, e4, d5]
    controller.begin_scrub(path)
    for t in (0.0, 0.5, 1.0):
        controller.update_scrub(ScrubPosition(path_index=1, t=t))
    assert engine.triggers == []  # no accent while merely previewing exd5's segment

    controller.end_scrub()
    # The real release commit: an actual new move, exactly like
    # _ScrubStrip.mouseReleaseEvent's own session_state.set_current_node call.
    node = session_state.make_move(chess.Move.from_uci("e4d5"))
    assert node is not None

    assert len(engine.triggers) == 1
    assert engine.triggers[0].kind == "capture"


# ---------------------------------------------------------
# Phase 5f.4a: scrub must never retrigger the Melody/Harmony note
# envelopes, and must flip the engine's scrub-active flag at the right
# moments.
# ---------------------------------------------------------


def test_begin_scrub_marks_the_engine_scrub_active(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])

    path = session_state.active_path()
    controller.begin_scrub(path)

    assert engine.scrub_active_calls[-1] is True


def test_end_scrub_clears_the_engine_scrub_active_flag(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])

    path = session_state.active_path()
    controller.begin_scrub(path)
    controller.end_scrub()

    assert engine.scrub_active_calls[-1] is False


def test_cancel_scrub_clears_the_engine_scrub_active_flag(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])

    path = session_state.active_path()
    controller.begin_scrub(path)
    controller.cancel_scrub()

    assert engine.scrub_active_calls[-1] is False


def test_no_note_trigger_fires_during_a_dense_scrub_drag(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])
    engine.note_trigger_count = 0  # discard the two commits' own triggers

    path = session_state.active_path()
    controller.begin_scrub(path)
    for i in range(200):
        t = (i % 101) / 100.0
        controller.update_scrub(ScrubPosition(path_index=i % 2, t=t))

    assert engine.note_trigger_count == 0


def test_scrub_release_produces_exactly_one_note_trigger_on_the_real_commit(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "d7d5"])
    engine.note_trigger_count = 0

    path = session_state.active_path()  # [root, e4, d5]
    controller.begin_scrub(path)
    for t in (0.0, 0.5, 1.0):
        controller.update_scrub(ScrubPosition(path_index=1, t=t))
    assert engine.note_trigger_count == 0  # still nothing during preview

    controller.end_scrub()
    node = session_state.make_move(chess.Move.from_uci("e4d5"))  # the real release commit
    assert node is not None

    assert engine.note_trigger_count == 1  # exactly one attack, from the commit alone


# ---------------------------------------------------------
# Rapid scrub: latest-state-wins, no unbounded growth
# ---------------------------------------------------------


def test_rapid_scrub_publishes_only_the_latest_value_each_call_no_queue(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])

    path = session_state.active_path()
    controller.begin_scrub(path)
    baseline = len(engine.published)  # the two committed moves already published

    for i in range(500):
        t = (i % 101) / 100.0
        controller.update_scrub(ScrubPosition(path_index=0, t=t))

    # One publish per update_scrub call -- no batching, no queue, and the
    # final published state matches exactly what the final t alone would
    # produce (proving no stale intermediate value lingers).
    assert len(engine.published) - baseline == 500
    final_t = (499 % 101) / 100.0
    controller.update_scrub(ScrubPosition(path_index=0, t=final_t))
    assert engine.published[-1] == engine.published[-2]


# ---------------------------------------------------------
# Cache / recomputation evidence
# ---------------------------------------------------------


def test_repeated_fractional_updates_within_one_segment_do_not_recompute_analysis(qapp, monkeypatch):
    import desktop_app.audio_controller as audio_controller_module

    call_count = {"analyze_position": 0}
    real_analyze_position = audio_controller_module.analyze_position

    def counting(*args, **kwargs):
        call_count["analyze_position"] += 1
        return real_analyze_position(*args, **kwargs)

    monkeypatch.setattr(audio_controller_module, "analyze_position", counting)

    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])
    assert call_count["analyze_position"] == 2  # from the two committed moves

    path = session_state.active_path()
    controller.begin_scrub(path)
    for t in [i / 50.0 for i in range(51)]:  # 51 fractional updates in one segment
        controller.update_scrub(ScrubPosition(path_index=0, t=t))

    assert call_count["analyze_position"] == 2  # unchanged -- both endpoints already cached


def test_scrubbing_into_a_never_visited_segment_computes_analysis_exactly_once(qapp, monkeypatch):
    import desktop_app.audio_controller as audio_controller_module

    call_count = {"analyze_position": 0}
    real_analyze_position = audio_controller_module.analyze_position

    def counting(*args, **kwargs):
        call_count["analyze_position"] += 1
        return real_analyze_position(*args, **kwargs)

    monkeypatch.setattr(audio_controller_module, "analyze_position", counting)

    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4"])
    assert call_count["analyze_position"] == 1

    # A hypothetical second move is NOT committed -- build a raw path
    # extension the way active_path() would if it had been played, to
    # scrub into a segment whose upper endpoint has never been visited.
    e4_node = session_state.current_node
    e5_node = e4_node.add_variation(chess.Move.from_uci("e7e5"))
    fake_path = [*session_state.active_path(), e5_node]

    controller.begin_scrub(fake_path)
    for t in [i / 20.0 for i in range(21)]:  # many fractional updates
        controller.update_scrub(ScrubPosition(path_index=1, t=t))

    assert call_count["analyze_position"] == 2  # e5_node computed exactly once


def test_crossing_back_into_a_previously_visited_segment_reuses_the_cache(qapp, monkeypatch):
    import desktop_app.audio_controller as audio_controller_module

    call_count = {"analyze_position": 0}
    real_analyze_position = audio_controller_module.analyze_position

    def counting(*args, **kwargs):
        call_count["analyze_position"] += 1
        return real_analyze_position(*args, **kwargs)

    monkeypatch.setattr(audio_controller_module, "analyze_position", counting)

    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5", "g1f3"])
    assert call_count["analyze_position"] == 3

    path = session_state.active_path()
    controller.begin_scrub(path)
    controller.update_scrub(ScrubPosition(path_index=0, t=0.5))  # segment (root, e4)
    controller.update_scrub(ScrubPosition(path_index=1, t=0.5))  # segment (e4, e5)
    controller.update_scrub(ScrubPosition(path_index=0, t=0.2))  # BACK to segment (root, e4)
    controller.update_scrub(ScrubPosition(path_index=1, t=0.8))  # BACK to segment (e4, e5)

    assert call_count["analyze_position"] == 3  # no growth -- every node already cached


# ---------------------------------------------------------
# Branch-aware scrub
# ---------------------------------------------------------


def test_scrub_after_a_branch_switch_uses_the_new_branchs_own_analysis(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])
    # engine.published so far: [0]=e4, [1]=e5

    session_state.undo()  # lands back on e4 -- publishes e4's own state again -> [2]
    _play(session_state, ["b8c6"])  # branch: 1...Nc6 instead of 1...e5 -> [3]

    branch_path = session_state.active_path()  # [root, e4, Nc6]
    controller.begin_scrub(branch_path)
    controller.update_scrub(ScrubPosition(path_index=1, t=1.0))  # settled on Nc6

    scrub_state = engine.published[-1]
    nc6_settled_state = engine.published[3]  # the b8c6 commit's own publish
    assert scrub_state.pitch_hz == pytest.approx(nc6_settled_state.pitch_hz)
    # Phase B1: pitch alone is no longer a reliable "not the old
    # branch's data" proxy -- e5 and c6 can legitimately quantize to
    # the same scale tone under the compressed 2-octave register.
    # segment_key (the FEN-pair identity) is the robust check.
    assert scrub_state.segment_key != engine.published[1].segment_key  # not e5's segment


# ---------------------------------------------------------
# Cancel-on-external-navigation (unit level; panel-level wiring is
# covered separately in tests/test_desktop_app_audio_scrub_integration.py)
# ---------------------------------------------------------


def test_cancel_scrub_stops_further_updates_from_publishing(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])

    path = session_state.active_path()
    controller.begin_scrub(path)
    controller.update_scrub(ScrubPosition(path_index=0, t=0.5))
    assert controller.is_scrubbing is True

    controller.cancel_scrub()
    assert controller.is_scrubbing is False

    published_count_after_cancel = len(engine.published)
    controller.update_scrub(ScrubPosition(path_index=0, t=0.9))  # a stray late mousemove
    assert len(engine.published) == published_count_after_cancel  # safe no-op


def test_a_real_navigation_after_cancel_restores_the_correct_settled_state(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])

    path = session_state.active_path()
    controller.begin_scrub(path)
    controller.update_scrub(ScrubPosition(path_index=0, t=0.5))  # mid-glide preview
    controller.cancel_scrub()

    session_state.undo()  # a real, external, committed navigation

    final_state = engine.published[-1]
    e4_settled_state = engine.published[0]
    assert final_state == e4_settled_state


# ---------------------------------------------------------
# Lifecycle: shutdown mid-scrub
# ---------------------------------------------------------


def test_shutdown_mid_scrub_stops_the_scrub_and_the_engine(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])

    path = session_state.active_path()
    controller.begin_scrub(path)
    controller.update_scrub(ScrubPosition(path_index=0, t=0.5))
    assert controller.is_scrubbing is True

    controller.shutdown()

    assert controller.is_scrubbing is False
    assert engine.shutdown_calls == 1

    published_count = len(engine.published)
    controller.update_scrub(ScrubPosition(path_index=0, t=0.9))  # stray late update
    assert len(engine.published) == published_count  # no publication after shutdown


def test_begin_scrub_after_shutdown_is_a_safe_no_op(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])
    controller.shutdown()

    path = session_state.active_path()
    controller.begin_scrub(path)
    assert controller.is_scrubbing is False

    controller.update_scrub(ScrubPosition(path_index=0, t=0.5))
    assert len(engine.published) == 2  # unchanged from the two committed moves
