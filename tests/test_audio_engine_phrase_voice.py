"""
Audio Layer 2 -- Rhythmic Layer, v2: engine-level tests for finite
phrase playback (audio/engine.py's `_PhraseSchedule`, `_phrase_armed`
gating, `_render_phrase`).

Supersedes tests/test_audio_engine_pulse_voice.py (deleted). That file
locked in v1's continuous-loop contract directly -- tests like
"no dropped or double ticks over long playback" and "ticks land in
exactly the independently predicted blocks" asserted the *infinite*
periodic-firing behavior that was diagnosed as the metronome problem and
has been deliberately removed, not merely retuned. Those specific
contracts are obsolete; the underlying concerns they protected
(determinism, no dropped events, correct timing, scrub freeze, mixer
integration, lifecycle safety) are still tested here, against the new
finite-phrase model.
"""

import time

import numpy as np
import pytest

from audio.backend import FakeAudioBackend
from audio.engine import (
    DEFAULT_BLOCKSIZE,
    DEFAULT_SAMPLE_RATE,
    LATENCY_BUDGET_SECONDS,
    AccentTrigger,
    AudioEngine,
)
from audio.live_state import SonificationState
from audio.phrase import EMPTY_PHRASE, MAX_PHRASE_EVENTS, PhraseDescription, PhraseEvent, build_phrase
from audio.voices import build_default_voice_registry


def _phrase(*events: tuple[float, float, float]) -> PhraseDescription:
    """Builds a PhraseDescription directly from (onset, pitch, amplitude)
    tuples -- precise control for engine-level tests, independent of
    audio/phrase.py's own density->event mapping (already covered by
    tests/test_audio_phrase.py)."""

    return PhraseDescription(
        events=tuple(PhraseEvent(onset_seconds=o, pitch_hz=p, amplitude=a) for o, p, a in events),
        duration_seconds=max((o for o, _, _ in events), default=0.0) + 1.0,
    )


def _state(
    *,
    pitch_hz: float = 440.0,
    harmony_interval_ratio: float = 1.2,
    harmony_above_melody: bool = True,
    harmonic_richness: int = 1,
    loudness: float = 0.3,
    color: str = "white",
    phrase: PhraseDescription = EMPTY_PHRASE,
    master_gain: float = 1.0,
    voice_mute=None,
    voice_solo=None,
) -> SonificationState:
    return SonificationState(
        harmony_interval_ratio=harmony_interval_ratio,
        harmony_above_melody=harmony_above_melody,
        pitch_hz=pitch_hz,
        harmonic_richness=harmonic_richness,
        loudness=loudness,
        segment_key=("fen-a", "fen-b"),
        color=color,
        phrase=phrase,
        master_gain=master_gain,
        voice_mute=voice_mute or {},
        voice_solo=voice_solo or {},
    )


def _running_engine(*, blocksize: int = 64, **kwargs):
    backend = FakeAudioBackend()
    registry = build_default_voice_registry()
    engine = AudioEngine(backend, registry, blocksize=blocksize, **kwargs)
    assert engine.start() is True
    stream = backend.streams[-1]
    return engine, backend, stream


# ---------------------------------------------------------
# Silence before arming; armed only by a real NoteTrigger
# ---------------------------------------------------------


def test_no_audible_phrase_before_the_first_committed_move():
    engine, backend, stream = _running_engine()
    phrase = _phrase((0.0, 880.0, 0.4), (0.2, 880.0, 0.3))
    engine.publish(_state(phrase=phrase, voice_mute={"melody": True, "harmony": True, "accent": True}))
    # deliberately no push_note_trigger() -- nothing committed yet

    block = stream.render_block(64)

    assert engine._phrase_armed is False
    assert np.all(block == 0.0)


def test_phrase_begins_on_the_committed_move():
    engine, backend, stream = _running_engine()
    phrase = _phrase((0.0, 880.0, 0.4))
    engine.publish(_state(phrase=phrase, voice_mute={"melody": True, "harmony": True, "accent": True}))
    engine.push_note_trigger()

    block = stream.render_block(64)

    assert engine._phrase_armed is True
    assert np.abs(block).max() > 0.0


def test_default_empty_phrase_is_silent():
    # SonificationState.phrase defaults to EMPTY_PHRASE -- direct
    # constructions predating this milestone must not crash or produce
    # unexpected sound.
    engine, backend, stream = _running_engine()
    engine.publish(_state(loudness=0.5))  # phrase omitted -> EMPTY_PHRASE
    engine.push_note_trigger()
    block_default = stream.render_block(64)

    engine2, backend2, stream2 = _running_engine()
    engine2.publish(_state(loudness=0.5, voice_mute={"pulse": True}))
    engine2.push_note_trigger()
    block_pulse_muted = stream2.render_block(64)

    assert block_default == pytest.approx(block_pulse_muted)


# ---------------------------------------------------------
# Phrase is finite: each event fires at most once, then true silence
# ---------------------------------------------------------


def test_each_scheduled_event_fires_exactly_once_no_loop():
    engine, backend, stream = _running_engine()
    phrase = _phrase((0.0, 880.0, 0.4), (0.05, 990.0, 0.3))
    engine.publish(_state(phrase=phrase, voice_mute={"melody": True, "harmony": True, "accent": True}))
    engine.push_note_trigger()

    trigger_calls = []
    original_trigger = engine._pulse_voice.trigger

    def counting_trigger(**kwargs):
        trigger_calls.append(kwargs)
        return original_trigger(**kwargs)

    engine._pulse_voice.trigger = counting_trigger

    # Render far longer than the phrase's own span -- if v1's modulo
    # wraparound somehow survived, this would fire many more than 2
    # events. It must fire EXACTLY 2, ever, no matter how long rendering
    # continues afterward.
    for _ in range(2000):  # ~2.9s at blocksize=64/44100Hz
        stream.render_block(64)

    assert len(trigger_calls) == 2


def test_phrase_reaches_true_silence_after_all_events_decay():
    engine, backend, stream = _running_engine(blocksize=256)
    phrase = _phrase((0.0, 880.0, 0.4))
    engine.publish(_state(phrase=phrase, voice_mute={"melody": True, "harmony": True, "accent": True}))
    engine.push_note_trigger()

    # Render well past the single event's own modal decay (damping >=15/s
    # reaches the envelope floor in well under a second).
    for _ in range(200):  # ~1.16s
        stream.render_block(256)

    silent_block = stream.render_block(256)
    assert np.all(silent_block == 0.0)


def test_low_density_produces_fewer_events_than_high_density():
    from audio.mapping import build_audio_mapping
    from chess_engine.analyzer import analyze_position
    from chess_engine.moves import execute_move
    from analysis.dynamics import analyze_dynamics
    import chess

    board = chess.Board()
    d1 = execute_move(board, "g1f3")
    calm_analysis = analyze_position(board, d1)
    calm_mapping = build_audio_mapping(calm_analysis, dynamics=None)

    d2 = execute_move(board, "b8c6")
    tense_analysis = analyze_position(board, d2)
    tense_dynamics = analyze_dynamics(previous=calm_analysis, current=tense_analysis)
    tense_mapping = build_audio_mapping(tense_analysis, tense_dynamics)

    assert len(calm_mapping.phrase.events) <= len(tense_mapping.phrase.events) or calm_mapping.pulse_density <= tense_mapping.pulse_density


# ---------------------------------------------------------
# Determinism
# ---------------------------------------------------------


def test_identical_phrase_and_trigger_sequences_produce_bit_identical_output():
    def _render_sequence():
        engine, backend, stream = _running_engine(smoothing_coefficient=1.0)
        phrase = build_phrase(pulse_density=0.7, pitch_hz=440.0, harmony_interval_ratio=1.3, color="black")
        engine.publish(_state(pitch_hz=440.0, color="black", loudness=0.7, phrase=phrase))
        engine.push_note_trigger()
        engine.push_event(AccentTrigger("capture", loudness=0.7))
        blocks = [stream.render_block(256) for _ in range(30)]
        return np.concatenate(blocks)

    first = _render_sequence()
    second = _render_sequence()
    assert np.array_equal(first, second)


# ---------------------------------------------------------
# Bounded schedule / no unbounded growth
# ---------------------------------------------------------


def test_schedule_never_exceeds_max_phrase_events_even_from_a_larger_phrase():
    engine, backend, stream = _running_engine()
    # Defensively larger than MAX_PHRASE_EVENTS -- audio/phrase.py never
    # actually builds one this large, but the engine's own schedule must
    # still clamp safely if it ever received one.
    oversized = _phrase(*[(i * 0.1, 440.0, 0.1) for i in range(10)])
    engine.publish(_state(phrase=oversized))
    engine.push_note_trigger()
    stream.render_block(64)

    assert engine._phrase_schedule.count <= MAX_PHRASE_EVENTS


def test_repeated_rapid_retriggers_never_grow_the_schedule():
    engine, backend, stream = _running_engine()
    for i in range(50):
        phrase = _phrase((0.0, 440.0 + i, 0.3), (0.1, 550.0 + i, 0.2))
        engine.publish(_state(phrase=phrase))
        engine.push_note_trigger()
        stream.render_block(64)
        assert engine._phrase_schedule.count == 2  # always exactly this phrase's own event count, never accumulating


# ---------------------------------------------------------
# Rapid move policy: newest becomes authoritative, no backlog
# ---------------------------------------------------------


def test_a_new_committed_move_supersedes_the_previous_phrases_unfired_events():
    engine, backend, stream = _running_engine(blocksize=64)
    old_phrase = _phrase((0.0, 100.0, 0.4), (1.0, 200.0, 0.3))  # second event far in the future
    engine.publish(_state(phrase=old_phrase, voice_mute={"melody": True, "harmony": True, "accent": True}))
    engine.push_note_trigger()
    stream.render_block(64)  # fires old_phrase's first event (100.0 Hz) only

    trigger_calls = []
    original_trigger = engine._pulse_voice.trigger

    def counting_trigger(**kwargs):
        trigger_calls.append(kwargs)
        return original_trigger(**kwargs)

    engine._pulse_voice.trigger = counting_trigger

    new_phrase = _phrase((0.0, 900.0, 0.4))
    engine.publish(_state(phrase=new_phrase, voice_mute={"melody": True, "harmony": True, "accent": True}))
    engine.push_note_trigger()

    for _ in range(500):  # render well past where old_phrase's 200.0Hz event would have fired
        stream.render_block(64)

    fired_pitches = [call["fundamental_hz"] for call in trigger_calls]
    assert 900.0 in fired_pitches
    assert 200.0 not in fired_pitches  # the old phrase's not-yet-fired event never fires late


def test_rapid_retrigger_produces_no_non_finite_or_runaway_output():
    engine, backend, stream = _running_engine(blocksize=64)
    for i in range(30):
        phrase = _phrase((0.0, 440.0, 0.4), (0.01, 550.0, 0.3))
        engine.publish(_state(loudness=0.8, phrase=phrase))
        engine.push_note_trigger()
        block = stream.render_block(64)
        assert np.all(np.isfinite(block))
        assert np.abs(block).max() <= 1.0


# ---------------------------------------------------------
# Scrub freeze and resume-only-on-real-commit semantics
# (mechanism unchanged from v1 -- re-verified against the new schedule)
# ---------------------------------------------------------


def test_phrase_disarms_the_instant_scrub_begins():
    engine, backend, stream = _running_engine()
    phrase = _phrase((0.0, 880.0, 0.4), (0.3, 990.0, 0.3))
    engine.publish(_state(phrase=phrase, voice_mute={"melody": True, "harmony": True, "accent": True}))
    engine.push_note_trigger()
    stream.render_block(64)
    assert engine._phrase_armed is True

    engine.set_scrub_active(True)

    assert engine._phrase_armed is False


def test_no_new_phrase_events_fire_while_scrub_is_active():
    engine, backend, stream = _running_engine()
    phrase = _phrase((0.0, 880.0, 0.4))
    engine.publish(_state(phrase=phrase, voice_mute={"melody": True, "harmony": True, "accent": True}))
    engine.push_note_trigger()
    stream.render_block(64)  # the one event armed by the commit itself

    engine.set_scrub_active(True)
    trigger_calls = []
    original_trigger = engine._pulse_voice.trigger

    def counting_trigger(**kwargs):
        trigger_calls.append(kwargs)
        return original_trigger(**kwargs)

    engine._pulse_voice.trigger = counting_trigger
    for _ in range(50):
        stream.render_block(64)

    assert trigger_calls == []


def test_scrub_ending_without_a_following_note_trigger_leaves_phrase_disarmed():
    engine, backend, stream = _running_engine()
    phrase = _phrase((0.0, 880.0, 0.4))
    engine.publish(_state(phrase=phrase, voice_mute={"melody": True, "harmony": True, "accent": True}))
    engine.push_note_trigger()
    stream.render_block(64)

    engine.set_scrub_active(True)
    stream.render_block(64)
    assert engine._phrase_armed is False

    engine.set_scrub_active(False)  # scrub ends -- no push_note_trigger() follows

    assert engine._phrase_armed is False
    block = stream.render_block(64)
    assert engine._phrase_armed is False
    assert np.all(np.isfinite(block))


def test_scrub_release_produces_at_most_one_valid_phrase():
    engine, backend, stream = _running_engine()
    old_phrase = _phrase((0.0, 100.0, 0.4))
    engine.publish(_state(phrase=old_phrase, voice_mute={"melody": True, "harmony": True, "accent": True}))
    engine.push_note_trigger()
    stream.render_block(64)

    engine.set_scrub_active(True)
    # A scrub preview publishing a different (irrelevant) phrase mid-drag
    # must have zero effect while frozen.
    preview_phrase = _phrase((0.0, 500.0, 0.4), (0.2, 600.0, 0.3))
    engine.publish(_state(phrase=preview_phrase, voice_mute={"melody": True, "harmony": True, "accent": True}))
    for _ in range(10):
        stream.render_block(64)
    assert engine._phrase_armed is False

    # The real release commit.
    trigger_calls = []
    original_trigger = engine._pulse_voice.trigger

    def counting_trigger(**kwargs):
        trigger_calls.append(kwargs)
        return original_trigger(**kwargs)

    engine._pulse_voice.trigger = counting_trigger

    final_phrase = _phrase((0.0, 900.0, 0.4))
    engine.set_scrub_active(False)
    engine.publish(_state(phrase=final_phrase, voice_mute={"melody": True, "harmony": True, "accent": True}))
    engine.push_note_trigger()

    for _ in range(50):
        stream.render_block(64)

    assert len(trigger_calls) == 1  # exactly one event -- the final phrase's own single event
    assert trigger_calls[0]["fundamental_hz"] == 900.0


# ---------------------------------------------------------
# Output shape / finiteness / bound
# ---------------------------------------------------------


def test_phrase_output_is_always_finite_and_within_the_clip_bound():
    engine, backend, stream = _running_engine(blocksize=DEFAULT_BLOCKSIZE)
    phrase = build_phrase(pulse_density=1.0, pitch_hz=440.0, harmony_interval_ratio=1.3, color="white")
    engine.publish(_state(loudness=0.9, phrase=phrase))
    engine.push_note_trigger()

    for _ in range(30):
        block = stream.render_block(DEFAULT_BLOCKSIZE)
        assert np.all(np.isfinite(block))
        assert np.abs(block).max() <= 1.0


# ---------------------------------------------------------
# Mute / Solo / Master gain
# ---------------------------------------------------------


def test_muting_pulse_matches_empty_phrase_output():
    phrase = _phrase((0.0, 880.0, 0.4))
    engine_muted, _, stream_muted = _running_engine()
    engine_muted.publish(_state(loudness=0.3, phrase=phrase, voice_mute={"pulse": True}))
    engine_muted.push_note_trigger()
    muted_block = stream_muted.render_block(64)

    engine_reference, _, stream_reference = _running_engine()
    engine_reference.publish(_state(loudness=0.3, phrase=EMPTY_PHRASE))
    engine_reference.push_note_trigger()
    reference_block = stream_reference.render_block(64)

    assert muted_block == pytest.approx(reference_block)


def test_soloing_pulse_isolates_it_from_melody_harmony_accent():
    phrase = _phrase((0.0, 880.0, 0.4))
    engine_soloed, _, stream_soloed = _running_engine()
    engine_soloed.publish(_state(loudness=0.0, phrase=phrase, voice_solo={"pulse": True}))
    engine_soloed.push_note_trigger()
    soloed_block = stream_soloed.render_block(64)

    engine_reference, _, stream_reference = _running_engine()
    engine_reference.publish(
        _state(loudness=0.0, phrase=phrase, voice_mute={"melody": True, "harmony": True, "accent": True})
    )
    engine_reference.push_note_trigger()
    reference_block = stream_reference.render_block(64)

    assert soloed_block == pytest.approx(reference_block)


def test_mute_wins_over_solo_for_pulse_too():
    phrase = _phrase((0.0, 880.0, 0.4))
    engine, _, stream = _running_engine()
    engine.publish(
        _state(loudness=0.0, phrase=phrase, voice_mute={"pulse": True}, voice_solo={"pulse": True})
    )
    engine.push_note_trigger()
    block = stream.render_block(64)

    engine_reference, _, stream_reference = _running_engine()
    engine_reference.publish(_state(loudness=0.0, phrase=EMPTY_PHRASE))
    engine_reference.push_note_trigger()
    reference_block = stream_reference.render_block(64)

    assert block == pytest.approx(reference_block)


def test_master_gain_scales_phrase_output_linearly():
    phrase = _phrase((0.0, 880.0, 0.4))
    engine_full, _, stream_full = _running_engine()
    engine_full.publish(
        _state(loudness=0.0, phrase=phrase, master_gain=1.0, voice_mute={"melody": True, "harmony": True, "accent": True})
    )
    engine_full.push_note_trigger()
    engine_half, _, stream_half = _running_engine()
    engine_half.publish(
        _state(loudness=0.0, phrase=phrase, master_gain=0.5, voice_mute={"melody": True, "harmony": True, "accent": True})
    )
    engine_half.push_note_trigger()

    block_full = stream_full.render_block(64)
    block_half = stream_half.render_block(64)

    assert block_half == pytest.approx(block_full * 0.5)


# ---------------------------------------------------------
# Accent semantics unchanged
# ---------------------------------------------------------


def test_accent_still_fires_only_from_an_explicit_trigger_regardless_of_phrase():
    phrase = build_phrase(pulse_density=1.0, pitch_hz=440.0, harmony_interval_ratio=1.3, color="white")
    engine, backend, stream = _running_engine()
    engine.publish(_state(loudness=0.9, phrase=phrase))
    engine.push_note_trigger()  # a dense phrase, but no AccentTrigger pushed

    stream.render_block(DEFAULT_BLOCKSIZE)
    assert engine._accent_state.active is False


# ---------------------------------------------------------
# Callback latency budget
# ---------------------------------------------------------


def test_callback_stays_within_budget_with_a_dense_phrase_active():
    engine, backend, stream = _running_engine(blocksize=DEFAULT_BLOCKSIZE)
    phrase = build_phrase(pulse_density=1.0, pitch_hz=440.0, harmony_interval_ratio=1.3, color="black")
    engine.publish(_state(pitch_hz=440.0, color="black", loudness=0.9, phrase=phrase))
    engine.push_note_trigger()
    engine.push_event(AccentTrigger("capture+check", loudness=0.9))

    stream.render_block(DEFAULT_BLOCKSIZE)  # warm up

    block_seconds = DEFAULT_BLOCKSIZE / DEFAULT_SAMPLE_RATE
    repeats = 200
    start = time.perf_counter()
    for _ in range(repeats):
        stream.render_block(DEFAULT_BLOCKSIZE)
    elapsed_per_block = (time.perf_counter() - start) / repeats

    assert elapsed_per_block < 0.5 * block_seconds, (
        f"callback took {elapsed_per_block * 1000:.3f}ms, "
        f"budget is {LATENCY_BUDGET_SECONDS * 1000:.1f}ms per block "
        f"({block_seconds * 1000:.2f}ms at this blocksize)"
    )


# ---------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------


def test_repeated_lifecycle_cycles_with_a_dense_phrase_stay_leak_free():
    for cycle in range(20):
        backend = FakeAudioBackend()
        registry = build_default_voice_registry()
        engine = AudioEngine(backend, registry, blocksize=256)

        assert engine.start() is True
        stream = backend.streams[-1]

        phrase = build_phrase(pulse_density=1.0, pitch_hz=220.0 + cycle, harmony_interval_ratio=1.25, color="white")
        engine.publish(_state(pitch_hz=220.0 + cycle, loudness=0.5, phrase=phrase))
        engine.push_note_trigger()
        stream.render_block(256)

        engine.set_scrub_active(True)
        stream.render_block(256)
        engine.set_scrub_active(False)

        engine.stop()
        engine.shutdown()

        assert engine.is_running is False
        assert engine.is_shut_down is True
        assert stream.close_calls == 1


def test_publish_and_push_after_shutdown_do_not_raise_with_a_phrase():
    engine, backend, stream = _running_engine()
    phrase = _phrase((0.0, 440.0, 0.3))
    engine.publish(_state(phrase=phrase))
    engine.push_note_trigger()
    stream.render_block(64)

    engine.shutdown()

    engine.publish(_state(phrase=phrase))  # must not raise
    engine.push_note_trigger()  # must not raise
