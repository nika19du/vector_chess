import numpy as np

from audio.organic_synthesis import (
    ACCENT_BASE_FREQUENCY_HZ,
    ACCENT_MODAL_PARAMS,
    BLACK_MODAL_PARAMS,
    HARMONY_MODAL_PARAMS,
    WHITE_MODAL_PARAMS,
    modal_resonance,
)

SAMPLE_RATE = 44100
DURATION_SECONDS = 1.6


def _spectral_centroid(samples: np.ndarray, sample_rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    total = spectrum.sum()
    return float((spectrum * freqs).sum() / total) if total else 0.0


def _energy_fraction_above(samples: np.ndarray, sample_rate: int, cutoff_hz: float) -> float:
    power_spectrum = np.abs(np.fft.rfft(samples)) ** 2
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    total = power_spectrum.sum()
    return float(power_spectrum[freqs > cutoff_hz].sum() / total) if total else 0.0


# ---------------------------------------------------------
# Determinism / finiteness / bounds
# ---------------------------------------------------------


def test_modal_resonance_is_deterministic():
    first = modal_resonance(440.0, WHITE_MODAL_PARAMS, DURATION_SECONDS, SAMPLE_RATE, amplitude=0.8)
    second = modal_resonance(440.0, WHITE_MODAL_PARAMS, DURATION_SECONDS, SAMPLE_RATE, amplitude=0.8)

    assert np.array_equal(first, second)


def test_modal_resonance_is_finite_everywhere():
    for params in (WHITE_MODAL_PARAMS, BLACK_MODAL_PARAMS, HARMONY_MODAL_PARAMS, ACCENT_MODAL_PARAMS):
        samples = modal_resonance(440.0, params, DURATION_SECONDS, SAMPLE_RATE, amplitude=0.9)
        assert np.all(np.isfinite(samples))


def test_modal_resonance_peak_never_exceeds_amplitude():
    for amplitude in (0.1, 0.5, 1.0):
        for params in (WHITE_MODAL_PARAMS, BLACK_MODAL_PARAMS, HARMONY_MODAL_PARAMS, ACCENT_MODAL_PARAMS):
            samples = modal_resonance(440.0, params, DURATION_SECONDS, SAMPLE_RATE, amplitude=amplitude)
            assert np.max(np.abs(samples)) <= amplitude + 1e-9


def test_zero_duration_returns_empty_array():
    samples = modal_resonance(440.0, WHITE_MODAL_PARAMS, 0.0, SAMPLE_RATE, amplitude=1.0)
    assert len(samples) == 0


# ---------------------------------------------------------
# Physical attack / decay / silence
# ---------------------------------------------------------


def test_decays_toward_silence_across_the_clip():
    samples = modal_resonance(440.0, WHITE_MODAL_PARAMS, DURATION_SECONDS, SAMPLE_RATE, amplitude=1.0)

    tenth = len(samples) // 10
    first_rms = np.sqrt(np.mean(samples[:tenth] ** 2))
    last_rms = np.sqrt(np.mean(samples[-tenth:] ** 2))

    assert last_rms < 0.1 * first_rms


def test_click_and_pop_safety_at_clip_boundaries():
    for params in (WHITE_MODAL_PARAMS, BLACK_MODAL_PARAMS, HARMONY_MODAL_PARAMS, ACCENT_MODAL_PARAMS):
        samples = modal_resonance(440.0, params, DURATION_SECONDS, SAMPLE_RATE, amplitude=1.0)
        assert abs(samples[0]) < 1e-9
        assert abs(samples[-1]) < 1e-9


def test_accent_energy_is_concentrated_early():
    accent = modal_resonance(
        ACCENT_BASE_FREQUENCY_HZ, ACCENT_MODAL_PARAMS, 0.2, SAMPLE_RATE, amplitude=1.0, attack_seconds=0.0
    )

    half = len(accent) // 2
    first_half_energy = np.sum(accent[:half] ** 2)
    second_half_energy = np.sum(accent[half:] ** 2)

    assert first_half_energy > second_half_energy


# ---------------------------------------------------------
# White / Black timbre distinction (item 3/11 of the B2 spec)
# ---------------------------------------------------------


def test_white_and_black_are_distinct_at_the_same_pitch():
    white = modal_resonance(440.0, WHITE_MODAL_PARAMS, DURATION_SECONDS, SAMPLE_RATE, amplitude=1.0)
    black = modal_resonance(440.0, BLACK_MODAL_PARAMS, DURATION_SECONDS, SAMPLE_RATE, amplitude=1.0)

    assert not np.array_equal(white, black)


def test_black_spectral_centroid_stays_bounded_below_white_at_every_register():
    """
    Black's mode bank includes a sub-octave (the original docs/audio.md
    intent, never previously implemented -- see
    experiments/audio_musicality/REPORT.md) plus fast-damped upper
    modes, so it should measure as consistently *darker* than White at
    the same fundamental, not shrill -- replacing B1's hard frequency
    ceiling with a measured, bounded relationship.
    """

    for fundamental_hz in (220.0, 440.0, 880.0):
        white = modal_resonance(fundamental_hz, WHITE_MODAL_PARAMS, DURATION_SECONDS, SAMPLE_RATE, amplitude=1.0)
        black = modal_resonance(fundamental_hz, BLACK_MODAL_PARAMS, DURATION_SECONDS, SAMPLE_RATE, amplitude=1.0)

        white_centroid = _spectral_centroid(white, SAMPLE_RATE)
        black_centroid = _spectral_centroid(black, SAMPLE_RATE)

        assert black_centroid < white_centroid
        assert black_centroid > 0.3 * white_centroid  # still recognizably the same instrument family


def test_no_uncontrolled_high_frequency_energy_for_either_color():
    for fundamental_hz in (220.0, 440.0, 880.0):
        for params in (WHITE_MODAL_PARAMS, BLACK_MODAL_PARAMS):
            samples = modal_resonance(fundamental_hz, params, DURATION_SECONDS, SAMPLE_RATE, amplitude=1.0)
            assert _energy_fraction_above(samples, SAMPLE_RATE, 8000.0) < 0.01
