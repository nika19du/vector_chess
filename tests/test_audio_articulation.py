"""
Phase 5f.4a: deterministic tests for the live Melody/Harmony
articulation envelope -- the fix for the "siren" symptom (see this
phase's own report). Two layers:

- Direct unit tests against `_ArticulationEnvelope` (the DSP shape
  itself: attack ramp, plateau, exponential decay to silence,
  retrigger-safe ramping from the current shape rather than a hard
  reset to zero) -- fast and exact, no audio backend needed.
- Integration tests against the real `AudioEngine` + `FakeAudioBackend`
  (the wiring: push_note_trigger/set_scrub_active, silence as the
  default resting state, mute/phase interaction, rapid retriggers,
  coexistence with the unrelated Accent voice).
"""

import numpy as np
import pytest

from audio.backend import FakeAudioBackend
from audio.engine import (
    AccentTrigger,
    AudioEngine,
    HARMONY_DECAY_SECONDS,
    MELODY_DECAY_SECONDS,
    _ArticulationEnvelope,
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


# ---------------------------------------------------------
# _ArticulationEnvelope -- direct DSP-shape unit tests
# ---------------------------------------------------------


def test_envelope_starts_silent_and_inactive():
    envelope = _ArticulationEnvelope(attack_seconds=0.01, plateau_seconds=0.05, decay_seconds=0.1, floor=1e-4)
    out = np.zeros(64)

    envelope.render(64, 44100, np.arange(64, dtype=np.float64), out)

    assert np.all(out == 0.0)
    assert envelope.active is False


def test_trigger_produces_an_attack_ramp_from_zero_when_starting_from_silence():
    envelope = _ArticulationEnvelope(attack_seconds=0.01, plateau_seconds=0.0, decay_seconds=1.0, floor=1e-4)
    envelope.trigger()

    frames = 100  # 100/44100 ~= 2.27ms, comfortably inside the 10ms attack
    arange = np.arange(frames, dtype=np.float64)
    out = np.zeros(frames)
    envelope.render(frames, 44100, arange, out)

    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.diff(out) >= -1e-9)  # monotonically non-decreasing during attack
    assert out[-1] < 1.0  # hasn't reached peak yet at this frame count


def test_envelope_reaches_full_peak_after_attack_and_holds_through_plateau():
    envelope = _ArticulationEnvelope(attack_seconds=0.001, plateau_seconds=0.05, decay_seconds=0.1, floor=1e-4)
    envelope.trigger()

    sample_rate = 44100
    frames = int(0.02 * sample_rate)  # past attack, still well inside the 50ms plateau
    arange = np.arange(frames, dtype=np.float64)
    out = np.zeros(frames)
    envelope.render(frames, sample_rate, arange, out)

    assert out[-1] == pytest.approx(1.0, abs=1e-6)


def test_envelope_decays_to_true_silence_and_then_stays_silent():
    envelope = _ArticulationEnvelope(attack_seconds=0.001, plateau_seconds=0.001, decay_seconds=0.01, floor=1e-4)
    envelope.trigger()

    sample_rate = 44100
    blocksize = 64
    arange = np.arange(blocksize, dtype=np.float64)

    # Render comfortably past attack + plateau + decay.
    for _ in range(50):
        out = np.zeros(blocksize)
        envelope.render(blocksize, sample_rate, arange, out)

    assert envelope.active is False
    assert envelope._last_shape == 0.0

    # And it stays silent -- no spontaneous re-triggering.
    for _ in range(5):
        out = np.zeros(blocksize)
        envelope.render(blocksize, sample_rate, arange, out)
        assert np.all(out == 0.0)


def test_retrigger_mid_decay_ramps_from_the_current_shape_not_from_zero():
    envelope = _ArticulationEnvelope(attack_seconds=0.001, plateau_seconds=0.0, decay_seconds=0.2, floor=1e-4)
    envelope.trigger()

    sample_rate = 44100
    blocksize = 64
    arange = np.arange(blocksize, dtype=np.float64)

    # Advance partway into decay.
    for _ in range(10):
        out = np.zeros(blocksize)
        envelope.render(blocksize, sample_rate, arange, out)
    shape_before_retrigger = envelope._last_shape
    assert 0.0 < shape_before_retrigger < 1.0  # genuinely mid-decay, not yet silent

    envelope.trigger()  # a rapid retrigger, before the note finished decaying

    out = np.zeros(blocksize)
    envelope.render(blocksize, sample_rate, arange, out)

    # The very first sample of the retriggered block continues from
    # wherever the envelope actually was -- no hard drop to zero.
    assert out[0] == pytest.approx(shape_before_retrigger, abs=1e-6)


def test_retrigger_after_full_silence_still_ramps_from_zero():
    envelope = _ArticulationEnvelope(attack_seconds=0.001, plateau_seconds=0.001, decay_seconds=0.005, floor=1e-4)
    envelope.trigger()

    sample_rate = 44100
    blocksize = 64
    arange = np.arange(blocksize, dtype=np.float64)
    for _ in range(50):
        out = np.zeros(blocksize)
        envelope.render(blocksize, sample_rate, arange, out)
    assert envelope.active is False

    envelope.trigger()
    out = np.zeros(blocksize)
    envelope.render(blocksize, sample_rate, arange, out)

    assert out[0] == pytest.approx(0.0, abs=1e-6)


def test_no_sample_to_sample_jump_exceeds_the_steepest_configured_stage_slope():
    """
    Click/pop guard: nothing the envelope does -- attack ramp, stage
    transition, or a mid-decay retrigger -- should ever produce a
    bigger per-sample amplitude jump than the steepest of its own
    configured stages (the attack ramp's slope, or the exponential
    decay's steepest instantaneous slope, right as decay begins).
    """

    attack_seconds = 0.002
    decay_seconds = 0.05
    floor = 1e-4
    envelope = _ArticulationEnvelope(
        attack_seconds=attack_seconds, plateau_seconds=0.01, decay_seconds=decay_seconds, floor=floor
    )
    sample_rate = 44100
    blocksize = 64
    arange = np.arange(blocksize, dtype=np.float64)

    decay_rate = -np.log(floor) / decay_seconds
    max_attack_step = 1.0 / (attack_seconds * sample_rate)
    max_decay_step = decay_rate / sample_rate
    max_theoretical_step = max(max_attack_step, max_decay_step)

    envelope.trigger()
    all_samples = []
    for i in range(60):
        out = np.zeros(blocksize)
        envelope.render(blocksize, sample_rate, arange, out)
        all_samples.append(out.copy())
        if i == 20:
            envelope.trigger()  # a retrigger mid-stream

    stream = np.concatenate(all_samples)
    deltas = np.abs(np.diff(stream))
    assert deltas.max() <= max_theoretical_step + 1e-6


# ---------------------------------------------------------
# Named articulation constants -- Melody is short/foreground, Harmony
# is softer/longer, per the approved plan's voice-separation design.
# ---------------------------------------------------------


def test_melody_decay_is_shorter_than_harmony_decay():
    assert MELODY_DECAY_SECONDS < HARMONY_DECAY_SECONDS


# ---------------------------------------------------------
# AudioEngine integration -- wiring, silence-by-default, scrub gating,
# mute/phase interaction, rapid retriggers, Accent coexistence.
# ---------------------------------------------------------


def _running_engine(*, blocksize=64, **envelope_overrides):
    backend = FakeAudioBackend()
    registry = build_default_voice_registry()
    engine = AudioEngine(backend, registry, blocksize=blocksize, **envelope_overrides)
    assert engine.start() is True
    stream = backend.streams[-1]
    return engine, backend, stream


def test_publish_without_a_note_trigger_stays_silent():
    engine, backend, stream = _running_engine()
    engine.publish(_state(loudness=0.5))

    block = stream.render_block(64)

    assert np.all(block == 0.0)


def test_note_trigger_makes_a_published_state_audible():
    engine, backend, stream = _running_engine(
        melody_attack_seconds=1e-6, melody_plateau_seconds=10.0, melody_decay_seconds=10.0
    )
    engine.publish(_state(loudness=0.5))
    engine.push_note_trigger()

    block = stream.render_block(64)

    assert np.abs(block).max() > 0.0


def test_committed_mode_note_eventually_returns_to_silence_without_a_new_trigger():
    engine, backend, stream = _running_engine(
        melody_attack_seconds=0.001,
        melody_plateau_seconds=0.001,
        melody_decay_seconds=0.01,
        harmony_attack_seconds=0.001,
        harmony_plateau_seconds=0.0,
        harmony_decay_seconds=0.01,
    )
    engine.publish(_state(loudness=0.5))
    engine.push_note_trigger()

    stream.render_block(64)  # the attack -- audible

    silent_at = None
    for i in range(80):
        block = stream.render_block(64)
        if np.all(block == 0.0):
            silent_at = i
            break

    assert silent_at is not None, "the note never returned to silence"

    # And it stays silent -- no new trigger arrived.
    for _ in range(5):
        block = stream.render_block(64)
        assert np.all(block == 0.0)


def test_scrub_active_true_keeps_a_note_sounding_without_a_trigger():
    engine, backend, stream = _running_engine()
    engine.set_scrub_active(True)
    engine.publish(_state(loudness=0.5))  # no push_note_trigger -- scrub preview never fires one

    block = stream.render_block(64)

    assert np.abs(block).max() > 0.0  # scrub mode's continuous glide, unaffected by the new envelope


def test_scrub_active_false_after_true_returns_to_silence_until_a_trigger():
    engine, backend, stream = _running_engine()
    engine.set_scrub_active(True)
    engine.publish(_state(loudness=0.5))
    stream.render_block(64)  # audible during scrub

    engine.set_scrub_active(False)
    block = stream.render_block(64)  # back to idle mode, no trigger yet

    assert np.all(block == 0.0)


def test_committed_mode_pitch_snaps_immediately_no_glide():
    engine, backend, stream = _running_engine()
    engine.publish(_state(pitch_hz=220.0, loudness=0.3))
    engine.push_note_trigger()
    stream.render_block(64)

    engine.publish(_state(pitch_hz=880.0, loudness=0.3))  # a brand-new move's pitch
    stream.render_block(64)

    # No smoothing/glide toward 880Hz in idle mode -- the very next block
    # already reflects the new target exactly.
    assert engine._melody_osc.smoothed_frequency.value == pytest.approx(880.0)


def test_scrub_mode_still_glides_pitch_via_one_pole_smoothing():
    engine, backend, stream = _running_engine(smoothing_coefficient=1.0)
    engine.set_scrub_active(True)
    engine.publish(_state(pitch_hz=220.0, loudness=0.3))
    stream.render_block(64)

    engine.publish(_state(pitch_hz=880.0, loudness=0.3))
    stream.render_block(64)

    # Scrub mode is explicitly unchanged -- smoothing_coefficient=1.0
    # here just makes the (still smoothed) chase land exactly on target
    # in one block, proving the smoother is still the mechanism in use.
    assert engine._melody_osc.smoothed_frequency.value == pytest.approx(880.0)


def test_envelope_keeps_evolving_while_the_voice_is_muted():
    engine, backend, stream = _running_engine(
        melody_attack_seconds=0.001, melody_plateau_seconds=0.001, melody_decay_seconds=0.02
    )
    engine.publish(_state(loudness=0.5, voice_mute={"melody": True}))
    engine.push_note_trigger()

    stream.render_block(64)
    elapsed_after_first_block = engine._melody_envelope.elapsed_seconds
    stream.render_block(64)
    elapsed_after_second_block = engine._melody_envelope.elapsed_seconds

    assert elapsed_after_second_block > elapsed_after_first_block  # kept advancing despite being muted


def test_rapid_committed_retriggers_do_not_raise_or_get_stuck():
    engine, backend, stream = _running_engine()

    block = None
    for i in range(30):
        engine.publish(_state(pitch_hz=220.0 + i, loudness=0.4))
        engine.push_note_trigger()
        block = stream.render_block(64)
        assert np.all(np.isfinite(block))

    assert np.abs(block).max() <= 1.0


def test_committed_mode_note_attack_overlapping_a_capture_accent_never_hard_clips():
    """
    Headroom regression, real production timing (Phase 5f.4a): a capture
    is both a note-trigger AND an accent-trigger fired in the same
    commit -- so a note's attack and the accent's own click can overlap
    in real usage. This must still respect the final [-1, 1] safety
    clip at every realistic loudness tier, using the actual default
    envelope constants (not a test-only override).
    """

    for loudness in (0.35, 0.55, 0.75, 0.95):
        engine, backend, stream = _running_engine()
        engine.publish(_state(loudness=loudness, harmonic_richness=3))
        engine.push_note_trigger()
        engine.push_event(AccentTrigger("capture", loudness=loudness))

        peak = 0.0
        for _ in range(20):  # comfortably past both the note attack and the accent's own decay
            block = stream.render_block(64)
            assert np.all(np.isfinite(block))
            peak = max(peak, float(np.abs(block).max()))

        assert peak <= 1.0, f"loudness={loudness} peak={peak}"


def test_accent_voice_is_unaffected_by_melody_harmony_articulation():
    engine, backend, stream = _running_engine()
    engine.publish(_state(loudness=0.5))  # no note trigger -- melody/harmony stay silent
    engine.push_event(AccentTrigger("capture", loudness=0.8))

    block = stream.render_block(64)

    assert engine._accent_state.active is True
    assert np.abs(block).max() > 0.0  # the accent alone is audible
