import threading

import numpy as np
import pytest

from audio.backend import FakeAudioBackend
from audio.engine import (
    AccentTrigger,
    AudioEngine,
    TriggerBuffer,
    DEFAULT_TRIGGER_BUFFER_CAPACITY,
)
from audio.live_state import SonificationState
from audio.voices import build_default_voice_registry


def _state(
    *,
    pitch_hz: float = 440.0,
    harmony_interval_ratio: float = 1.2,
    harmony_above_melody: bool = True,
    harmonic_richness: int = 1,
    loudness: float = 0.3,
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
        master_gain=master_gain,
        voice_mute=voice_mute or {},
        voice_solo=voice_solo or {},
    )


def _running_engine(
    *,
    blocksize: int = 64,
    smoothing_coefficient: float = 1.0,
    melody_attack_seconds: float = 1e-6,
    melody_plateau_seconds: float = 10.0,
    melody_decay_seconds: float = 10.0,
    harmony_attack_seconds: float = 1e-6,
    harmony_plateau_seconds: float = 10.0,
    harmony_decay_seconds: float = 10.0,
):
    """
    smoothing_coefficient=1.0 makes the very first block after a
    publish() reach the target instantly (value += (target-value)*1.0
    == target) in scrub mode -- this keeps single-block assertions
    exact and deterministic instead of entangled with the smoothing
    ramp.

    Phase 5f.4a: committed/idle mode (the default -- scrub_active is
    False unless a test calls set_scrub_active(True)) drives Melody/
    Harmony through the articulation envelope instead. The near-instant
    attack and long plateau/decay defaults here put a triggered voice at
    (approximately) full peak amplitude for many blocks, mirroring the
    old "instant smoothing" trick so most tests can assert exact,
    steady-state output without rendering through a real attack/decay
    shape. Tests that specifically exercise attack/decay/silence timing
    override these explicitly.
    """

    backend = FakeAudioBackend()
    registry = build_default_voice_registry()
    engine = AudioEngine(
        backend,
        registry,
        blocksize=blocksize,
        smoothing_coefficient=smoothing_coefficient,
        melody_attack_seconds=melody_attack_seconds,
        melody_plateau_seconds=melody_plateau_seconds,
        melody_decay_seconds=melody_decay_seconds,
        harmony_attack_seconds=harmony_attack_seconds,
        harmony_plateau_seconds=harmony_plateau_seconds,
        harmony_decay_seconds=harmony_decay_seconds,
    )
    assert engine.start() is True
    stream = backend.streams[-1]
    return engine, backend, stream


# ---------------------------------------------------------
# TriggerBuffer -- ordering and overflow policy
# ---------------------------------------------------------


def test_trigger_buffer_pops_in_fifo_order():
    buffer = TriggerBuffer(capacity=4)
    a, b, c = AccentTrigger("a"), AccentTrigger("b"), AccentTrigger("c")

    buffer.push(a)
    buffer.push(b)
    buffer.push(c)

    assert buffer.pop() is a
    assert buffer.pop() is b
    assert buffer.pop() is c


def test_trigger_buffer_pop_on_empty_returns_none():
    buffer = TriggerBuffer(capacity=4)

    assert buffer.pop() is None


def test_trigger_buffer_drops_oldest_on_overflow():
    buffer = TriggerBuffer(capacity=4)
    triggers = [AccentTrigger(str(i)) for i in range(6)]

    for trigger in triggers:
        buffer.push(trigger)

    assert len(buffer) == 4
    # triggers[0] and triggers[1] were dropped -- the oldest two, in
    # push order -- leaving triggers[2..5] in FIFO order.
    remaining = [buffer.pop() for _ in range(4)]
    assert remaining == triggers[2:6]
    assert buffer.pop() is None


def test_trigger_buffer_default_capacity_matches_the_engine_default():
    buffer = TriggerBuffer()

    assert buffer.capacity == DEFAULT_TRIGGER_BUFFER_CAPACITY


def test_trigger_buffer_len_tracks_pending_count():
    buffer = TriggerBuffer(capacity=4)

    assert len(buffer) == 0
    buffer.push(AccentTrigger("a"))
    buffer.push(AccentTrigger("b"))
    assert len(buffer) == 2
    buffer.pop()
    assert len(buffer) == 1


# ---------------------------------------------------------
# AudioEngine construction constraints
# ---------------------------------------------------------


def test_engine_rejects_non_mono_channel_counts():
    backend = FakeAudioBackend()
    registry = build_default_voice_registry()

    with pytest.raises(ValueError):
        AudioEngine(backend, registry, channels=2)


# ---------------------------------------------------------
# publish() / latest-state semantics
# ---------------------------------------------------------


def test_publish_before_any_render_leaves_output_silent():
    engine, backend, stream = _running_engine()

    block = stream.render_block(64)

    assert np.all(block == 0.0)


def test_publish_takes_effect_on_the_next_rendered_block():
    engine, backend, stream = _running_engine()

    silent_block = stream.render_block(64)
    assert np.all(silent_block == 0.0)

    engine.publish(_state(loudness=0.3))
    engine.push_note_trigger()  # Phase 5f.4a: publish alone no longer sounds -- a note must be triggered
    audible_block = stream.render_block(64)

    assert np.abs(audible_block).max() > 0.0


def test_publish_only_the_latest_state_is_ever_read():
    engine, backend, stream = _running_engine()

    engine.publish(_state(pitch_hz=220.0, loudness=0.3))
    engine.publish(_state(pitch_hz=880.0, loudness=0.3))  # overwrites, never queued

    block = stream.render_block(64)

    # With instant smoothing, the melody oscillator's target frequency
    # is whatever was published last -- 880Hz, never 220Hz.
    assert engine._melody_osc.smoothed_frequency.value == pytest.approx(880.0)


# ---------------------------------------------------------
# Callback output shape and finiteness
# ---------------------------------------------------------


def test_callback_output_shape_matches_frames_and_channels():
    engine, backend, stream = _running_engine(blocksize=100)
    engine.publish(_state(loudness=0.3))

    block = stream.render_block(100)

    assert block.shape == (100, 1)


def test_callback_output_is_always_finite_before_and_after_publish():
    engine, backend, stream = _running_engine()

    before = stream.render_block(64)
    assert np.all(np.isfinite(before))

    engine.publish(_state(loudness=0.9, harmonic_richness=3))
    engine.push_note_trigger()
    engine.push_event(AccentTrigger("capture", loudness=0.9))
    after = stream.render_block(64)
    assert np.all(np.isfinite(after))


def test_callback_output_never_exceeds_the_clip_bound():
    engine, backend, stream = _running_engine()
    # deliberately hot input -- loudness + harmony + accent easily sums
    # past 1.0 before the final hard clip.
    engine.publish(_state(loudness=1.0, harmonic_richness=3))
    engine.push_note_trigger()
    engine.push_event(AccentTrigger("capture", loudness=1.0))

    block = stream.render_block(64)

    assert np.abs(block).max() <= 1.0


# ---------------------------------------------------------
# Master gain
# ---------------------------------------------------------


def test_master_gain_scales_the_output_linearly():
    engine_full, _, stream_full = _running_engine()
    engine_half, _, stream_half = _running_engine()

    # Low loudness keeps the mix comfortably under the clip bound at
    # gain=1.0, so the comparison stays linear rather than saturating.
    engine_full.publish(_state(loudness=0.2, master_gain=1.0))
    engine_full.push_note_trigger()
    engine_half.publish(_state(loudness=0.2, master_gain=0.5))
    engine_half.push_note_trigger()

    block_full = stream_full.render_block(64)
    block_half = stream_half.render_block(64)

    assert block_half == pytest.approx(block_full * 0.5)


def test_master_gain_zero_produces_silence_even_with_active_voices():
    engine, backend, stream = _running_engine()
    engine.publish(_state(loudness=0.9, master_gain=0.0))
    engine.push_note_trigger()

    block = stream.render_block(64)

    assert np.all(block == 0.0)


# ---------------------------------------------------------
# Mute
# ---------------------------------------------------------


def test_muting_every_voice_produces_silence():
    engine, backend, stream = _running_engine()

    engine.publish(_state(loudness=0.3, voice_mute={"harmony": True, "accent": True}))
    engine.push_note_trigger()
    melody_only_block = stream.render_block(64)
    assert np.abs(melody_only_block).max() > 0.0

    engine.publish(
        _state(loudness=0.3, voice_mute={"melody": True, "harmony": True, "accent": True})
    )
    fully_muted_block = stream.render_block(64)

    assert np.all(fully_muted_block == 0.0)


def test_muted_voice_phase_continues_advancing_across_blocks():
    engine, backend, stream = _running_engine()

    engine.publish(_state(pitch_hz=440.0, loudness=0.3))
    stream.render_block(64)
    phase_before_mute = engine._melody_osc.phase

    engine.publish(_state(pitch_hz=440.0, loudness=0.3, voice_mute={"melody": True}))
    stream.render_block(64)
    phase_after_first_muted_block = engine._melody_osc.phase
    stream.render_block(64)
    phase_after_second_muted_block = engine._melody_osc.phase

    assert phase_after_first_muted_block != pytest.approx(phase_before_mute)
    assert phase_after_second_muted_block != pytest.approx(phase_after_first_muted_block)


def test_unmuting_produces_audible_output_again_without_resetting_phase():
    engine, backend, stream = _running_engine()

    engine.publish(_state(pitch_hz=440.0, loudness=0.3, voice_mute={"melody": True}))
    engine.push_note_trigger()
    stream.render_block(64)
    phase_while_muted = engine._melody_osc.phase

    engine.publish(_state(pitch_hz=440.0, loudness=0.3))  # unmuted again
    block = stream.render_block(64)

    assert np.abs(block).max() > 0.0
    # phase kept advancing from where it was while muted, not reset to 0
    assert engine._melody_osc.phase != pytest.approx(phase_while_muted)


# ---------------------------------------------------------
# Solo
# ---------------------------------------------------------


def test_solo_isolates_the_soloed_voice_regardless_of_mute_flags():
    # Two independently fresh engines (both starting at phase=0.0) so a
    # single-block comparison isn't confounded by phase continuing to
    # advance between two sequential blocks on the same engine.
    engine_muted, _, stream_muted = _running_engine()
    engine_soloed, _, stream_soloed = _running_engine()

    engine_muted.publish(_state(loudness=0.3, voice_mute={"harmony": True, "accent": True}))
    engine_muted.push_note_trigger()
    melody_only = stream_muted.render_block(64)

    engine_soloed.publish(_state(loudness=0.3, voice_solo={"melody": True}))
    engine_soloed.push_note_trigger()
    soloed = stream_soloed.render_block(64)

    assert np.abs(soloed).max() > 0.0
    assert soloed == pytest.approx(melody_only)


def test_mute_wins_over_solo_on_the_same_voice():
    """
    Phase 5f.5: a voice that is BOTH muted and soloed must stay silent --
    mute always wins, even in solo mode. (Discovered while wiring the
    Mixer UI to this contract: the pre-5f.5 `audible()` let solo-mode
    ignore mute entirely for a voice carrying both flags.)
    """

    engine_both, _, stream_both = _running_engine()
    engine_harmony_only, _, stream_harmony_only = _running_engine()

    engine_both.publish(
        _state(loudness=0.3, voice_mute={"melody": True}, voice_solo={"melody": True, "harmony": True})
    )
    engine_both.push_note_trigger()
    both_block = stream_both.render_block(64)

    # Reference: harmony soloed, melody not even mentioned -- if mute
    # correctly wins, muted+soloed melody should contribute nothing,
    # making this identical to "only harmony soloed."
    engine_harmony_only.publish(_state(loudness=0.3, voice_solo={"harmony": True}))
    engine_harmony_only.push_note_trigger()
    harmony_only_block = stream_harmony_only.render_block(64)

    assert both_block == pytest.approx(harmony_only_block)


# ---------------------------------------------------------
# Lifecycle: start / stop / shutdown
# ---------------------------------------------------------


def test_start_opens_and_starts_the_stream():
    backend = FakeAudioBackend()
    engine = AudioEngine(backend, build_default_voice_registry(), blocksize=64)

    assert engine.start() is True
    assert engine.is_running is True
    assert len(backend.streams) == 1
    assert backend.streams[0].start_calls == 1


def test_start_is_idempotent_while_already_running():
    backend = FakeAudioBackend()
    engine = AudioEngine(backend, build_default_voice_registry(), blocksize=64)

    assert engine.start() is True
    assert engine.start() is True

    assert len(backend.streams) == 1
    assert backend.streams[0].start_calls == 1


def test_stop_pauses_without_closing_the_stream():
    engine, backend, stream = _running_engine()

    engine.stop()

    assert engine.is_running is False
    assert stream.stop_calls == 1
    assert stream.close_calls == 0


def test_stop_is_idempotent_when_not_running():
    backend = FakeAudioBackend()
    engine = AudioEngine(backend, build_default_voice_registry(), blocksize=64)

    engine.stop()  # never started
    engine.stop()

    assert engine.is_running is False
    assert backend.streams == []


def test_start_after_stop_resumes_the_same_stream_instance():
    engine, backend, stream = _running_engine()

    engine.stop()
    assert engine.start() is True

    assert len(backend.streams) == 1  # no second stream opened
    assert stream.start_calls == 2


def test_shutdown_closes_the_stream_and_marks_shut_down():
    engine, backend, stream = _running_engine()

    engine.shutdown()

    assert engine.is_shut_down is True
    assert engine.is_running is False
    assert stream.close_calls == 1


def test_shutdown_is_idempotent():
    engine, backend, stream = _running_engine()

    engine.shutdown()
    engine.shutdown()

    assert stream.close_calls == 1


def test_start_after_shutdown_always_returns_false_and_opens_nothing():
    engine, backend, stream = _running_engine()

    engine.shutdown()

    assert engine.start() is False
    assert engine.is_running is False
    assert len(backend.streams) == 1  # no new stream opened


def test_publish_and_push_event_after_shutdown_do_not_raise():
    engine, backend, stream = _running_engine()
    engine.shutdown()

    engine.publish(_state(loudness=0.5))
    engine.push_event(AccentTrigger("capture"))  # must not raise


def test_repeated_create_start_stop_shutdown_cycles_are_stable():
    for _ in range(20):
        backend = FakeAudioBackend()
        engine = AudioEngine(backend, build_default_voice_registry(), blocksize=64)

        assert engine.start() is True
        engine.publish(_state(loudness=0.3))
        stream = backend.streams[-1]
        stream.render_block(64)
        engine.stop()
        engine.shutdown()

        assert engine.is_running is False
        assert engine.is_shut_down is True
        assert stream.close_calls == 1


# ---------------------------------------------------------
# Device-unavailable fallback
# ---------------------------------------------------------


def test_is_available_defaults_true_before_any_start_attempt():
    backend = FakeAudioBackend()
    engine = AudioEngine(backend, build_default_voice_registry(), blocksize=64)

    assert engine.is_available is True


def test_start_with_failing_backend_returns_false_and_never_raises():
    backend = FakeAudioBackend(fail_on_open=True)
    engine = AudioEngine(backend, build_default_voice_registry(), blocksize=64)

    result = engine.start()  # must not raise

    assert result is False
    assert engine.is_running is False
    assert engine.is_available is False
    assert engine.last_open_error is not None


def test_engine_with_unavailable_device_can_still_be_shut_down_safely():
    backend = FakeAudioBackend(fail_on_open=True)
    engine = AudioEngine(backend, build_default_voice_registry(), blocksize=64)

    engine.start()
    engine.shutdown()  # must not raise even though no stream ever opened

    assert engine.is_shut_down is True
    assert engine.start() is False


# ---------------------------------------------------------
# Real-hardware thread-leak check (guarded, skipped if unavailable)
# ---------------------------------------------------------


def test_repeated_real_backend_cycles_do_not_leak_threads():
    """
    Skipped, not failed, when no real audio device/backend is available
    in this environment -- an environment limitation, not evidence of a
    code defect. When a device is available, this is the strongest
    check available for "no thread left alive after shutdown."
    """

    from audio.backend import SoundDeviceBackend

    backend = SoundDeviceBackend()
    baseline = threading.active_count()

    try:
        for _ in range(5):
            engine = AudioEngine(backend, build_default_voice_registry(), blocksize=256)
            if not engine.start():
                pytest.skip(f"no real audio device available: {engine.last_open_error}")
            engine.publish(_state(loudness=0.2))
            engine.shutdown()
    except Exception as error:  # pragma: no cover -- environment-dependent
        pytest.skip(f"real audio backend unusable in this environment: {error}")

    assert threading.active_count() <= baseline + 1
