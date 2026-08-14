"""
B3: Live Modal/Organic AudioEngine Port -- focused tests for the new
modal DSP wiring itself (color-driven voice selection, scrub's
undamped/sustained contract vs committed's finite articulation, the
multi-mode Accent "knock", determinism, and callback timing under a
worst-case realistic voice combination). Engine lifecycle, mute/solo,
headroom, and articulation-envelope mechanics are already covered by
tests/test_audio_engine.py, tests/test_audio_articulation.py, and
tests/test_audio_live_offline_headroom.py -- unaffected by B3 and not
duplicated here.
"""

import time
from dataclasses import replace

import numpy as np
import pytest

from audio.backend import FakeAudioBackend
from audio.engine import DEFAULT_BLOCKSIZE, DEFAULT_SAMPLE_RATE, LATENCY_BUDGET_SECONDS, AccentTrigger, AudioEngine
from audio.live_state import SonificationState
from audio.voices import build_default_voice_registry


def _state(
    *,
    pitch_hz: float = 440.0,
    harmony_interval_ratio: float = 1.2,
    harmony_above_melody: bool = True,
    harmonic_richness: int = 1,
    loudness: float = 0.5,
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


def _running_engine(*, blocksize: int = DEFAULT_BLOCKSIZE, **kwargs):
    backend = FakeAudioBackend()
    registry = build_default_voice_registry()
    engine = AudioEngine(backend, registry, blocksize=blocksize, **kwargs)
    assert engine.start() is True
    stream = backend.streams[-1]
    return engine, backend, stream


def _spectral_centroid(samples: np.ndarray, sample_rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    total = spectrum.sum()
    return float((spectrum * freqs).sum() / total) if total else 0.0


# ---------------------------------------------------------
# Color -> WHITE_MODES/BLACK_MODES selection (requirement 5)
# ---------------------------------------------------------


def test_white_and_black_committed_notes_produce_different_waveforms_at_the_same_pitch():
    engine, backend, stream = _running_engine(smoothing_coefficient=1.0, melody_attack_seconds=1e-6)
    engine.publish(_state(pitch_hz=440.0, color="white"))
    engine.push_note_trigger()
    white_block = stream.render_block(256)

    engine2, backend2, stream2 = _running_engine(smoothing_coefficient=1.0, melody_attack_seconds=1e-6)
    engine2.publish(_state(pitch_hz=440.0, color="black"))
    engine2.push_note_trigger()
    black_block = stream2.render_block(256)

    assert not np.array_equal(white_block, black_block)


def test_white_and_black_are_distinguishable_during_scrub_too():
    engine, backend, stream = _running_engine(smoothing_coefficient=1.0)
    engine.set_scrub_active(True)
    engine.publish(_state(pitch_hz=440.0, color="white"))
    white_block = stream.render_block(256)

    engine2, backend2, stream2 = _running_engine(smoothing_coefficient=1.0)
    engine2.set_scrub_active(True)
    engine2.publish(_state(pitch_hz=440.0, color="black"))
    black_block = stream2.render_block(256)

    assert not np.array_equal(white_block, black_block)


def test_default_color_is_white_when_state_omits_it():
    # SonificationState.color defaults to "white" -- direct
    # constructions predating B3 (and any caller that never sets it)
    # must not crash or silently render silence.
    engine, backend, stream = _running_engine(smoothing_coefficient=1.0)
    engine.set_scrub_active(True)
    engine.publish(_state(pitch_hz=440.0))  # color omitted -> "white"

    block = stream.render_block(256)

    assert np.all(np.isfinite(block))
    assert np.abs(block).max() > 0.0


# ---------------------------------------------------------
# Scrub must stay sustained/undamped -- not a repeating pluck
# (requirement 7)
# ---------------------------------------------------------


def test_scrub_amplitude_does_not_decay_while_held():
    """
    Requirement 7: scrub must not retrigger plucked/modal events
    continuously -- concretely, a held scrub position must sound at a
    steady amplitude, never fading toward silence the way a committed
    note's modal decay (offline) or articulation envelope (live
    committed mode) would over the same span.
    """

    engine, backend, stream = _running_engine(smoothing_coefficient=1.0)
    engine.set_scrub_active(True)
    engine.publish(_state(pitch_hz=440.0, loudness=0.6))

    # A single block's own max is a noisy proxy for "instantaneous
    # amplitude" (beating between a block boundary and the waveform's
    # own period can make consecutive per-block maxima swing up and
    # down even at genuinely constant amplitude) -- windowed RMS over
    # many consecutive blocks smooths that out and reveals the real
    # trend: does energy decay across the 2-second span, or not.
    window_blocks = 20
    block_count = int(2.0 * DEFAULT_SAMPLE_RATE / 256) + 1
    all_samples = []
    for _ in range(block_count):
        all_samples.append(stream.render_block(256).flatten())
    stream_samples = np.concatenate(all_samples)

    window_size = window_blocks * 256
    early_rms = float(np.sqrt(np.mean(stream_samples[window_size:2 * window_size] ** 2)))
    late_rms = float(np.sqrt(np.mean(stream_samples[-window_size:] ** 2)))

    assert early_rms > 0.0
    assert late_rms > 0.9 * early_rms  # no meaningful decay held over 2 seconds


def test_committed_mode_note_does_decay_unlike_scrub():
    """
    Complements test_scrub_amplitude_does_not_decay_while_held: committed
    notes must show a real, measurable RMS decline over their (now B2-
    faithful, multi-second) life -- unlike scrub, which stays flat.
    Full "reaches true silence" coverage lives in
    tests/test_audio_articulation.py (needs a larger blocksize to reach
    the real B2 decay time in a reasonable number of test iterations).
    """

    engine, backend, stream = _running_engine(melody_attack_seconds=0.001, harmony_attack_seconds=0.001)
    engine.publish(_state(pitch_hz=440.0, loudness=0.6))
    engine.push_note_trigger()

    blocks = [stream.render_block(64).flatten() for _ in range(400)]  # ~0.58s
    stream_samples = np.concatenate(blocks)

    window = 64 * 20
    early_rms = float(np.sqrt(np.mean(stream_samples[:window] ** 2)))
    late_rms = float(np.sqrt(np.mean(stream_samples[-window:] ** 2)))

    assert early_rms > 0.0
    assert late_rms < 0.9 * early_rms  # measurably decayed, unlike scrub's held-flat contract


# ---------------------------------------------------------
# Accent: multi-mode "knock" (requirement 6), same family, capture-only
# ---------------------------------------------------------


def test_accent_spectrum_contains_more_than_a_single_pure_tone():
    """
    B3's Accent uses ACCENT_MODES (3 inharmonic modes), not a single
    fixed-frequency decaying sine -- the rendered spectrum should show
    energy at more than one distinct frequency.
    """

    engine, backend, stream = _running_engine(blocksize=DEFAULT_BLOCKSIZE)
    engine.publish(_state(loudness=0.9))  # accent is only mixed in once a state exists
    engine.push_event(AccentTrigger("capture", loudness=0.9))

    block = stream.render_block(DEFAULT_BLOCKSIZE).flatten()
    spectrum = np.abs(np.fft.rfft(block))
    freqs = np.fft.rfftfreq(len(block), d=1.0 / DEFAULT_SAMPLE_RATE)

    peak_freq = freqs[np.argmax(spectrum)]
    # Energy meaningfully away from the single dominant peak -- proof
    # this isn't one pure sine (which would have ~all its energy in one
    # narrow bin).
    energy_near_peak = spectrum[(freqs > peak_freq - 20) & (freqs < peak_freq + 20)].sum()
    total_energy = spectrum.sum()

    assert energy_near_peak < 0.95 * total_energy


def test_accent_is_finite_and_bounded():
    engine, backend, stream = _running_engine()
    engine.publish(_state(loudness=1.0))
    engine.push_event(AccentTrigger("capture", loudness=1.0))

    for _ in range(30):
        block = stream.render_block(DEFAULT_BLOCKSIZE)
        assert np.all(np.isfinite(block))
        assert np.abs(block).max() <= 1.0


def test_accent_never_fires_without_an_explicit_trigger():
    """Capture-only semantics: publishing state alone (no AccentTrigger)
    must never make the accent voice active, committed or scrub."""

    for scrub in (False, True):
        engine, backend, stream = _running_engine()
        engine.set_scrub_active(scrub)
        engine.publish(_state(loudness=0.9))
        engine.push_note_trigger()

        stream.render_block(DEFAULT_BLOCKSIZE)
        assert engine._accent_state.active is False


# ---------------------------------------------------------
# Determinism (requirement: deterministic behavior)
# ---------------------------------------------------------


def test_identical_trigger_sequences_produce_bit_identical_output():
    def _render_sequence():
        engine, backend, stream = _running_engine(smoothing_coefficient=1.0)
        engine.publish(_state(pitch_hz=440.0, color="black", loudness=0.7))
        engine.push_note_trigger()
        engine.push_event(AccentTrigger("capture", loudness=0.7))
        blocks = [stream.render_block(256) for _ in range(10)]
        return np.concatenate(blocks)

    first = _render_sequence()
    second = _render_sequence()

    assert np.array_equal(first, second)


# ---------------------------------------------------------
# Callback timing under a worst-case realistic voice combination
# ---------------------------------------------------------


def test_callback_stays_comfortably_within_the_block_latency_budget():
    """
    Worst realistic combination: a capturing+checking committed move --
    Melody (5-mode Black) + Harmony (2-mode) + Accent (3-mode) all
    active in the same block, at the engine's real default blocksize.
    """

    engine, backend, stream = _running_engine(blocksize=DEFAULT_BLOCKSIZE)
    engine.publish(_state(pitch_hz=440.0, color="black", loudness=0.9))
    engine.push_note_trigger()
    engine.push_event(AccentTrigger("capture+check", loudness=0.9))

    stream.render_block(DEFAULT_BLOCKSIZE)  # warm up (first call pays import/alloc overhead)

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
# Repeated lifecycle stress, exercising the new modal voices
# ---------------------------------------------------------


def test_repeated_lifecycle_cycles_with_notes_and_accents_stay_leak_free():
    for cycle in range(20):
        backend = FakeAudioBackend()
        registry = build_default_voice_registry()
        engine = AudioEngine(backend, registry, blocksize=256)

        assert engine.start() is True
        stream = backend.streams[-1]

        color = "white" if cycle % 2 == 0 else "black"
        engine.publish(_state(pitch_hz=220.0 + cycle, color=color, loudness=0.5))
        engine.push_note_trigger()
        engine.push_event(AccentTrigger("capture", loudness=0.5))
        stream.render_block(256)

        engine.set_scrub_active(True)
        engine.publish(_state(pitch_hz=440.0, color=color, loudness=0.4))
        stream.render_block(256)
        engine.set_scrub_active(False)

        engine.stop()
        engine.shutdown()

        assert engine.is_running is False
        assert engine.is_shut_down is True
        assert stream.close_calls == 1
