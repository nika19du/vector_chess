import numpy as np
import pytest

from audio.synthesis import (
    additive_tone,
    apply_envelope,
    mix,
    percussive_burst,
    sine_wave,
)

# A low rate keeps tests fast; still well above every frequency used here.
SAMPLE_RATE = 8000


def test_sine_wave_sample_count_and_bounds():
    samples = sine_wave(
        frequency_hz=440.0,
        duration_seconds=0.5,
        sample_rate=SAMPLE_RATE,
        amplitude=0.8,
    )

    assert len(samples) == int(0.5 * SAMPLE_RATE)
    assert np.max(np.abs(samples)) <= 0.8 + 1e-9


def test_additive_tone_with_richness_one_equals_sine_wave():
    sine = sine_wave(220.0, 0.25, SAMPLE_RATE, amplitude=0.7)
    additive = additive_tone(
        220.0,
        harmonic_richness=1,
        duration_seconds=0.25,
        sample_rate=SAMPLE_RATE,
        amplitude=0.7,
    )

    assert np.array_equal(sine, additive)


def test_additive_tone_richer_signal_has_more_high_frequency_energy():
    duration = 0.5
    frequency = 220.0

    pure = additive_tone(
        frequency, harmonic_richness=1, duration_seconds=duration, sample_rate=SAMPLE_RATE
    )
    rich = additive_tone(
        frequency, harmonic_richness=4, duration_seconds=duration, sample_rate=SAMPLE_RATE
    )

    pure_spectrum = np.abs(np.fft.rfft(pure))
    rich_spectrum = np.abs(np.fft.rfft(rich))

    freqs = np.fft.rfftfreq(len(pure), d=1.0 / SAMPLE_RATE)
    above_fundamental = freqs > (frequency * 1.5)

    assert np.sum(rich_spectrum[above_fundamental]) > np.sum(pure_spectrum[above_fundamental])


def test_envelope_starts_and_ends_near_zero():
    samples = np.ones(1000)
    enveloped = apply_envelope(
        samples, attack_seconds=0.01, release_seconds=0.01, sample_rate=SAMPLE_RATE
    )

    assert abs(enveloped[0]) < 1e-9
    assert abs(enveloped[-1]) < 1e-9


def test_envelope_preserves_sustain_region():
    samples = np.full(1000, 0.5)
    enveloped = apply_envelope(
        samples, attack_seconds=0.01, release_seconds=0.01, sample_rate=SAMPLE_RATE
    )

    middle = len(samples) // 2
    assert enveloped[middle] == pytest.approx(0.5)


def test_percussive_burst_decays():
    burst = percussive_burst(duration_seconds=0.2, sample_rate=SAMPLE_RATE, amplitude=1.0)

    half = len(burst) // 2
    first_half_energy = np.sum(burst[:half] ** 2)
    second_half_energy = np.sum(burst[half:] ** 2)

    assert first_half_energy > second_half_energy


def test_mix_single_voice_is_unchanged():
    voice = sine_wave(440.0, 0.1, SAMPLE_RATE)
    mixed = mix(voice)

    assert np.array_equal(mixed, voice)


def test_mix_pads_shorter_voices_with_zero():
    long_voice = np.ones(100)
    short_voice = np.ones(40)

    mixed = mix(long_voice, short_voice)

    assert len(mixed) == 100
    assert mixed[50] == pytest.approx(1.0)  # only long_voice reaches here
    assert mixed[10] == pytest.approx(2.0)  # both voices overlap here


def test_mix_with_no_voices_returns_empty_array():
    assert len(mix()) == 0
