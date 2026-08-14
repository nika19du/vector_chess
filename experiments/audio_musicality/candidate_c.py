"""
Candidate C -- hybrid mathematical.

Changes the LEAST about production's own philosophy: keeps the
additive-synthesis-as-Attack-Influence-analog identity (production's
own justification for additive synthesis: "the synthesis method *is*
the mathematical method," docs/audio.md Section 2) and keeps White/
Black distinguished exactly as production does today -- by partial
count (1 vs. up to 3) -- rather than moving identity to filtering
(Candidate B) or timbre family (Candidate A).

The single fix is a capped ABSOLUTE partial frequency, not an
arbitrary partial-count cut: partials are omitted once
frequency_hz * partial_index exceeds HARMONIC_CEILING_HZ, so a
low-register Black move keeps its full richness while a high-register
one legitimately gets fewer partials -- itself one-sentence-explainable
("Black's timbre richness is capped by an absolute frequency ceiling,
not an arbitrary partial count").

HARMONIC_CEILING_HZ is chosen from measure.py's actual measured data
(experiments/audio_musicality/output/diagnostics/measurements.csv),
not asserted blind: across the 66 Black partials measured over 3 real
games, 21.2% exceed 4000 Hz but only 6.1% exceed 6000 Hz and 1.5%
exceed 8000 Hz -- 4200 Hz sits at the "elbow" of that distribution,
capping the most extreme quintile of Black's harmonic content while
leaving the majority (roughly 79%) of Black's richness untouched.
"""

import numpy as np

from audio.models import AudioMapping
from audio.synthesis import apply_envelope, mix, percussive_burst, sine_wave

import experiments.audio_musicality.register as register
from experiments.audio_musicality.synth_common import one_pole_lowpass

HARMONIC_CEILING_HZ = 4200.0
"""Data-derived (see module docstring) -- the elbow of Black's measured
partial-frequency distribution across 3 real games."""

UNIFORM_FILTER_CUTOFF = 0.5
"""Mild low-pass, same for both colors -- unlike Candidate B, White/
Black identity here stays entirely in partial count, not filtering."""

HARMONY_VOICE_WEIGHT = 0.6
SUSTAINED_PEAK_HEADROOM = 0.85
CAPTURE_ACCENT_DURATION_SECONDS = 0.2

WHITE_DEMO_SQUARE = "d4"
BLACK_DEMO_SQUARE = "d5"


def capped_additive_tone(
    frequency_hz: float,
    harmonic_richness: int,
    duration_seconds: float,
    sample_rate: int,
    amplitude: float = 1.0,
    ceiling_hz: float = HARMONIC_CEILING_HZ,
) -> np.ndarray:
    """
    Same weighted-partial-sum identity as production's additive_tone
    (weight 1/n per n-th partial), except partials whose absolute
    frequency (frequency_hz * partial_index) exceeds ceiling_hz are
    omitted entirely rather than summed and then hoping downstream
    filtering saves them.
    """

    n = int(round(duration_seconds * sample_rate))
    t = np.linspace(0.0, duration_seconds, n, endpoint=False)
    total = np.zeros(n)
    weight_sum = 0.0

    for partial_index in range(1, harmonic_richness + 1):
        partial_hz = frequency_hz * partial_index
        if partial_hz > ceiling_hz:
            continue

        weight = 1.0 / partial_index
        weight_sum += weight
        total += weight * np.sin(2 * np.pi * partial_hz * t)

    if weight_sum == 0.0:
        # Every partial (even the fundamental) exceeded the ceiling --
        # fall back to an unweighted fundamental rather than silence,
        # since a move must always produce some audible melody note.
        return amplitude * np.sin(2 * np.pi * frequency_hz * t)

    return total * (amplitude / weight_sum)


def _melody_voice(mapping: AudioMapping, melody_pitch_hz: float, sample_rate: int, duration_seconds: float) -> np.ndarray:
    tone = capped_additive_tone(
        melody_pitch_hz,
        harmonic_richness=mapping.harmonic_richness,
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        amplitude=mapping.loudness,
    )
    tone = apply_envelope(tone, attack_seconds=0.01, release_seconds=0.15, sample_rate=sample_rate)
    return one_pole_lowpass(tone, UNIFORM_FILTER_CUTOFF)


def _harmony_voice(mapping: AudioMapping, melody_pitch_hz: float, sample_rate: int, duration_seconds: float) -> np.ndarray:
    quantized_ratio = register.quantize_ratio_to_scale(
        mapping.harmony_interval_ratio, register.DIATONIC_HARMONY_RATIOS
    )
    harmony_hz = (
        melody_pitch_hz * quantized_ratio
        if mapping.attack_influence_balance >= 0
        else melody_pitch_hz / quantized_ratio
    )

    tone = apply_envelope(
        sine_wave(harmony_hz, duration_seconds, sample_rate, amplitude=mapping.loudness * HARMONY_VOICE_WEIGHT),
        attack_seconds=0.01, release_seconds=0.15, sample_rate=sample_rate,
    )
    return one_pole_lowpass(tone, UNIFORM_FILTER_CUTOFF)


def render_move(mapping: AudioMapping, sample_rate: int = 44100, duration_seconds: float = 1.6) -> np.ndarray:
    melody_pitch_hz = register.pitch_for_square(mapping.destination_square)

    melody = _melody_voice(mapping, melody_pitch_hz, sample_rate, duration_seconds)
    harmony = _harmony_voice(mapping, melody_pitch_hz, sample_rate, duration_seconds)

    sustained = mix(melody, harmony)
    peak = float(np.max(np.abs(sustained))) if sustained.size else 0.0
    if peak > SUSTAINED_PEAK_HEADROOM and peak > 0.0:
        sustained = sustained * (SUSTAINED_PEAK_HEADROOM / peak)

    if mapping.is_capture:
        accent_duration = min(CAPTURE_ACCENT_DURATION_SECONDS, duration_seconds)
        accent = apply_envelope(
            percussive_burst(duration_seconds=accent_duration, sample_rate=sample_rate, amplitude=mapping.loudness),
            attack_seconds=0.0,
            release_seconds=min(0.15, accent_duration),
            sample_rate=sample_rate,
        )
        combined = mix(sustained, accent)
    else:
        combined = sustained

    return np.clip(combined, -1.0, 1.0)


def render_isolated_white_note(sample_rate: int = 44100, duration_seconds: float = 1.6) -> np.ndarray:
    pitch_hz = register.pitch_for_square(WHITE_DEMO_SQUARE)
    tone = apply_envelope(
        capped_additive_tone(pitch_hz, harmonic_richness=1, duration_seconds=duration_seconds, sample_rate=sample_rate, amplitude=0.8),
        attack_seconds=0.01, release_seconds=0.15, sample_rate=sample_rate,
    )
    return np.clip(one_pole_lowpass(tone, UNIFORM_FILTER_CUTOFF), -1.0, 1.0)


def render_isolated_black_note(sample_rate: int = 44100, duration_seconds: float = 1.6) -> np.ndarray:
    pitch_hz = register.pitch_for_square(BLACK_DEMO_SQUARE)
    tone = apply_envelope(
        capped_additive_tone(pitch_hz, harmonic_richness=3, duration_seconds=duration_seconds, sample_rate=sample_rate, amplitude=0.8),
        attack_seconds=0.01, release_seconds=0.15, sample_rate=sample_rate,
    )
    return np.clip(one_pole_lowpass(tone, UNIFORM_FILTER_CUTOFF), -1.0, 1.0)


def render_isolated_harmony(sample_rate: int = 44100, duration_seconds: float = 1.6) -> np.ndarray:
    pitch_hz = register.pitch_for_square(WHITE_DEMO_SQUARE)
    ratio = register.quantize_ratio_to_scale(1.2599, register.DIATONIC_HARMONY_RATIOS)
    tone = apply_envelope(
        sine_wave(pitch_hz * ratio, duration_seconds, sample_rate, amplitude=0.7),
        attack_seconds=0.01, release_seconds=0.15, sample_rate=sample_rate,
    )
    return np.clip(one_pole_lowpass(tone, UNIFORM_FILTER_CUTOFF), -1.0, 1.0)


def render_isolated_accent(sample_rate: int = 44100, duration_seconds: float = 0.4) -> np.ndarray:
    return np.clip(
        apply_envelope(
            percussive_burst(duration_seconds, sample_rate, amplitude=0.9),
            attack_seconds=0.0, release_seconds=0.15, sample_rate=sample_rate,
        ),
        -1.0, 1.0,
    )
