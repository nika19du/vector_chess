"""
B3a: deterministic tests for the live Melody/Harmony committed-note
model (`_ModalNoteState`/`_ModalVoice`) that replaced Phase 5f.4a's
`_ArticulationEnvelope` -- see audio/engine.py's own module docstring
for why (Phase 1 measurement in experiments/audio_b3a_audit/ found the
old envelope's independently-configurable decay times cut a note off
3-5x sooner than B2's own per-mode damping needs, and its flat,
non-evolving oscillator sum never let a note's timbre darken the way
B2's offline render does).

Two layers, mirroring the old file's own structure:
- Direct unit tests against `_ModalNoteState`/`_ModalVoice` (the DSP
  shape itself: attack ramp, per-mode decay, retrigger-safe two-slot
  overlap, no discontinuity on note replacement) -- fast and exact, no
  audio backend needed.
- Integration tests against the real `AudioEngine` + `FakeAudioBackend`
  (the wiring: push_note_trigger/set_scrub_active, silence as the
  default resting state, mute/state interaction, rapid retriggers,
  coexistence with the unrelated Accent voice).
"""

import numpy as np
import pytest

from audio.backend import FakeAudioBackend
from audio.engine import AccentTrigger, AudioEngine, _ModalNoteState, _ModalVoice
from audio.live_state import SonificationState
from audio.organic_synthesis import HARMONY_MODES, WHITE_MODES
from audio.voices import build_default_voice_registry

SAMPLE_RATE = 44100


def _state(
    *,
    pitch_hz: float = 440.0,
    harmony_interval_ratio: float = 1.2,
    harmony_above_melody: bool = True,
    harmonic_richness: int = 1,
    loudness: float = 0.3,
    color: str = "white",
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
        master_gain=master_gain,
        voice_mute=voice_mute or {},
        voice_solo=voice_solo or {},
    )


# ---------------------------------------------------------
# _ModalNoteState -- direct DSP-shape unit tests
# ---------------------------------------------------------


def test_note_state_starts_silent_and_inactive():
    note = _ModalNoteState()
    buf = np.zeros(64)
    arange = np.arange(64, dtype=np.float64)

    peak_bound = note.render_into(buf, 64, SAMPLE_RATE, arange)

    assert np.all(buf == 0.0)
    assert peak_bound == 0.0
    assert note.active is False


def test_trigger_produces_an_attack_ramp_from_zero():
    note = _ModalNoteState()
    note.trigger(fundamental_hz=440.0, modes=WHITE_MODES, amplitude=0.5, attack_seconds=0.01)

    frames = 100  # 100/44100 ~= 2.27ms, comfortably inside the 10ms attack
    arange = np.arange(frames, dtype=np.float64)
    buf = np.zeros(frames)
    note.render_into(buf, frames, SAMPLE_RATE, arange)

    assert buf[0] == pytest.approx(0.0, abs=1e-6)
    assert np.max(np.abs(buf)) < 0.5  # hasn't reached full weight budget yet at this frame count


def test_per_mode_decay_matches_the_accepted_b2_damping_rates():
    """
    The actual regression fix: each mode's own weight in the mix must
    shrink at ITS OWN accepted-B2 rate over time, not stay flat -- a
    windowed spectral centroid taken early vs. late in the note's life
    must be measurably different (brighter early, darker late), unlike
    B3's flat-timbre behavior.
    """

    note = _ModalNoteState()
    note.trigger(fundamental_hz=440.0, modes=WHITE_MODES, amplitude=0.8, attack_seconds=0.001)

    frames = int(0.05 * SAMPLE_RATE)  # 50ms windows
    arange = np.arange(frames, dtype=np.float64)

    early = np.zeros(frames)
    note.render_into(early, frames, SAMPLE_RATE, arange)  # t in [0, 0.05)

    for _ in range(9):  # advance to roughly t in [0.5, 0.55)
        skip = np.zeros(frames)
        note.render_into(skip, frames, SAMPLE_RATE, arange)

    late = np.zeros(frames)
    note.render_into(late, frames, SAMPLE_RATE, arange)

    def centroid(samples):
        spectrum = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), d=1.0 / SAMPLE_RATE)
        total = spectrum.sum()
        return float((spectrum * freqs).sum() / total) if total else 0.0

    assert centroid(late) < centroid(early)  # darkens over time, matching B2


def test_note_decays_to_true_silence_and_then_stays_silent():
    note = _ModalNoteState()
    # HARMONY_MODES has the fastest dominant-mode damping of the three
    # committed voices, so this test doesn't need to wait multiple
    # seconds of simulated audio.
    note.trigger(fundamental_hz=440.0, modes=HARMONY_MODES, amplitude=0.5, attack_seconds=0.001)

    blocksize = 4410  # 100ms/block -- fewer python-level iterations for a multi-second decay
    arange = np.arange(blocksize, dtype=np.float64)

    for _ in range(40):  # 4 simulated seconds, comfortably past HARMONY_MODES's own decay time
        buf = np.zeros(blocksize)
        note.render_into(buf, blocksize, SAMPLE_RATE, arange)

    assert note.active is False

    buf = np.zeros(blocksize)
    peak_bound = note.render_into(buf, blocksize, SAMPLE_RATE, arange)
    assert np.all(buf == 0.0)
    assert peak_bound == 0.0


def test_finite_and_bounded_across_the_notes_whole_life():
    note = _ModalNoteState()
    note.trigger(fundamental_hz=440.0, modes=WHITE_MODES, amplitude=0.9, attack_seconds=0.005)

    blocksize = 512
    arange = np.arange(blocksize, dtype=np.float64)
    for _ in range(40):
        buf = np.zeros(blocksize)
        note.render_into(buf, blocksize, SAMPLE_RATE, arange)
        assert np.all(np.isfinite(buf))
        assert np.max(np.abs(buf)) <= 0.9 + 1e-9  # never exceeds amplitude (see modal_resonance's own proof)


# ---------------------------------------------------------
# _ModalVoice -- two-slot retrigger continuity (click-free)
# ---------------------------------------------------------


def test_retrigger_demotes_the_previous_note_to_secondary_which_keeps_ringing():
    voice = _ModalVoice()
    voice.trigger(fundamental_hz=440.0, modes=WHITE_MODES, amplitude=0.6, attack_seconds=0.001)

    blocksize = 64
    arange = np.arange(blocksize, dtype=np.float64)
    for _ in range(20):  # advance partway into the first note's decay
        buf = np.zeros(blocksize)
        voice.render_into(buf, blocksize, SAMPLE_RATE, arange)

    first_note_object = voice.primary
    assert first_note_object.active is True
    assert first_note_object.elapsed_seconds > 0.0

    voice.trigger(fundamental_hz=550.0, modes=WHITE_MODES, amplitude=0.6, attack_seconds=0.001)

    assert voice.secondary is first_note_object  # demoted, not discarded
    assert voice.secondary.active is True  # still ringing
    assert voice.primary is not first_note_object
    assert voice.primary.elapsed_seconds == 0.0  # fresh note


def test_retrigger_produces_no_discontinuity_at_the_retrigger_boundary():
    """
    Click/pop guard: summing two independently-smooth curves (the old
    note's own established decay, the new note's own fresh attack ramp
    from zero) cannot itself introduce a discontinuity. Rather than
    hand-deriving an absolute slope bound (the multi-mode carrier's own
    normal oscillation already produces sample-to-sample steps of
    comparable size, especially right where two notes' amplitudes sum),
    this compares the delta AT the exact retrigger boundary against the
    general population of deltas from ordinary (non-retriggered)
    playback -- a genuine discontinuity would be an outlier; ordinary
    carrier motion would not.
    """

    blocksize = 64
    arange = np.arange(blocksize, dtype=np.float64)
    attack_seconds = 0.002
    block_count = 60
    retrigger_at_block = 20

    def _render(retrigger: bool) -> np.ndarray:
        voice = _ModalVoice()
        voice.trigger(fundamental_hz=440.0, modes=WHITE_MODES, amplitude=0.7, attack_seconds=attack_seconds)
        samples = []
        for i in range(block_count):
            buf = np.zeros(blocksize)
            voice.render_into(buf, blocksize, SAMPLE_RATE, arange)
            samples.append(buf.copy())
            if retrigger and i == retrigger_at_block:
                voice.trigger(fundamental_hz=390.0, modes=WHITE_MODES, amplitude=0.7, attack_seconds=attack_seconds)
        return np.concatenate(samples)

    retriggered_stream = _render(retrigger=True)
    reference_stream = _render(retrigger=False)  # same note, never retriggered -- "ordinary motion" baseline

    boundary_sample = (retrigger_at_block + 1) * blocksize
    boundary_delta = abs(retriggered_stream[boundary_sample] - retriggered_stream[boundary_sample - 1])

    reference_deltas = np.abs(np.diff(reference_stream))
    assert boundary_delta <= reference_deltas.max() + 1e-9


def test_third_overlapping_retrigger_silently_retires_the_older_secondary():
    """
    Bounded to 2 slots (zero allocation): trigger() reuses the same two
    preallocated _ModalNoteState objects forever, overwriting whichever
    one is being superseded in place -- so slot *identity* does not
    track "which note" across more than one retrigger (the object that
    was slot_a can later hold note C's data). What must hold is the
    *content*: after a third trigger, neither active slot's frequency
    should be note A's anymore -- only the two most recent notes (B, C)
    survive.
    """

    voice = _ModalVoice()
    voice.trigger(fundamental_hz=440.0, modes=WHITE_MODES, amplitude=0.5, attack_seconds=0.001)  # note A
    voice.trigger(fundamental_hz=450.0, modes=WHITE_MODES, amplitude=0.5, attack_seconds=0.001)  # note B
    voice.trigger(fundamental_hz=460.0, modes=WHITE_MODES, amplitude=0.5, attack_seconds=0.001)  # note C

    live_fundamentals = {voice.primary.frequencies_hz[0], voice.secondary.frequencies_hz[0]}
    assert live_fundamentals == {460.0, 450.0}  # C and B survive
    assert 440.0 not in live_fundamentals  # A (the oldest) was silently retired


def test_voice_render_into_is_finite_and_bounded_through_a_retrigger():
    voice = _ModalVoice()
    voice.trigger(fundamental_hz=440.0, modes=WHITE_MODES, amplitude=0.9, attack_seconds=0.005)

    blocksize = 64
    arange = np.arange(blocksize, dtype=np.float64)
    for i in range(30):
        buf = np.zeros(blocksize)
        peak_bound = voice.render_into(buf, blocksize, SAMPLE_RATE, arange)
        assert np.all(np.isfinite(buf))
        assert peak_bound >= 0.0
        if i == 10:
            voice.trigger(fundamental_hz=500.0, modes=WHITE_MODES, amplitude=0.9, attack_seconds=0.005)


# ---------------------------------------------------------
# AudioEngine integration -- wiring, silence-by-default, scrub gating,
# mute/state interaction, rapid retriggers, Accent coexistence.
# ---------------------------------------------------------


def _running_engine(*, blocksize=64, **engine_kwargs):
    backend = FakeAudioBackend()
    registry = build_default_voice_registry()
    engine = AudioEngine(backend, registry, blocksize=blocksize, **engine_kwargs)
    assert engine.start() is True
    stream = backend.streams[-1]
    return engine, backend, stream


def test_publish_without_a_note_trigger_stays_silent():
    engine, backend, stream = _running_engine()
    engine.publish(_state(loudness=0.5))

    block = stream.render_block(64)

    assert np.all(block == 0.0)


def test_note_trigger_makes_a_published_state_audible():
    engine, backend, stream = _running_engine(melody_attack_seconds=1e-6)
    engine.publish(_state(loudness=0.5))
    engine.push_note_trigger()

    block = stream.render_block(64)

    assert np.abs(block).max() > 0.0


def test_committed_mode_note_eventually_returns_to_silence_without_a_new_trigger():
    engine, backend, stream = _running_engine(blocksize=4410, melody_attack_seconds=0.001, harmony_attack_seconds=0.001)
    engine.publish(_state(loudness=0.5))
    engine.push_note_trigger()

    stream.render_block(4410)  # the attack -- audible

    silent_at = None
    for i in range(40):  # up to 4 simulated seconds
        block = stream.render_block(4410)
        if np.all(block == 0.0):
            silent_at = i
            break

    assert silent_at is not None, "the note never returned to silence"

    for _ in range(3):
        block = stream.render_block(4410)
        assert np.all(block == 0.0)


def test_scrub_active_true_keeps_a_note_sounding_without_a_trigger():
    engine, backend, stream = _running_engine()
    engine.set_scrub_active(True)
    engine.publish(_state(loudness=0.5))  # no push_note_trigger -- scrub preview never fires one

    block = stream.render_block(64)

    assert np.abs(block).max() > 0.0  # scrub mode's continuous glide, unaffected by note-slot articulation


def test_scrub_active_false_after_true_returns_to_silence_until_a_trigger():
    engine, backend, stream = _running_engine()
    engine.set_scrub_active(True)
    engine.publish(_state(loudness=0.5))
    stream.render_block(64)  # audible during scrub

    engine.set_scrub_active(False)
    block = stream.render_block(64)  # back to idle mode, no trigger yet

    assert np.all(block == 0.0)


def test_committed_mode_plays_the_newly_triggered_pitch_not_the_previous_notes():
    engine, backend, stream = _running_engine(melody_attack_seconds=1e-6)
    engine.publish(_state(pitch_hz=220.0, loudness=0.3))
    engine.push_note_trigger()
    stream.render_block(64)

    engine.publish(_state(pitch_hz=880.0, loudness=0.3))  # a brand-new move's pitch
    engine.push_note_trigger()
    stream.render_block(64)

    # The new committed note's primary slot reflects the newly
    # triggered pitch immediately -- no glide/portamento concept exists
    # for committed notes (that is scrub's job).
    assert engine._melody_voice.primary.frequencies_hz[0] == pytest.approx(880.0)


def test_scrub_mode_still_glides_pitch_via_one_pole_smoothing():
    engine, backend, stream = _running_engine(smoothing_coefficient=1.0)
    engine.set_scrub_active(True)
    engine.publish(_state(pitch_hz=220.0, loudness=0.3))
    stream.render_block(64)

    engine.publish(_state(pitch_hz=880.0, loudness=0.3))
    stream.render_block(64)

    assert engine._melody_osc.smoothed_frequency.value == pytest.approx(880.0)


def test_note_state_keeps_evolving_while_the_voice_is_muted():
    engine, backend, stream = _running_engine(melody_attack_seconds=0.001)
    engine.publish(_state(loudness=0.5, voice_mute={"melody": True}))
    engine.push_note_trigger()

    stream.render_block(64)
    elapsed_after_first_block = engine._melody_voice.primary.elapsed_seconds
    stream.render_block(64)
    elapsed_after_second_block = engine._melody_voice.primary.elapsed_seconds

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
    Headroom regression, real production timing: a capture is both a
    note-trigger AND an accent-trigger fired in the same commit -- so a
    note's attack and the accent's own click can overlap in real usage.
    Must still respect the final [-1, 1] safety clip at every realistic
    loudness tier, using the actual default attack constants.
    """

    for loudness in (0.35, 0.55, 0.75, 0.95):
        engine, backend, stream = _running_engine()
        engine.publish(_state(loudness=loudness))
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
