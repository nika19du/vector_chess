"""
Candidate B -- restrained electronic / ambient.

Both colors use the SAME synthesis technique (a soft, steeply-rolled-
off additive oscillator) and the SAME 2-octave-clamped, diatonically-
quantized register as melody pitch already uses in production. White
and Black are distinguished purely by one-pole low-pass filter
brightness -- a direct test of requirement 4 (identity via timbre/
filtering, not pitch/register). Slower attack/release than production
for a softer, more "restrained" character. Tension (dynamics_label)
additionally brightens the filter slightly, a second explainable
tension channel layered on top of the unchanged loudness signal.
"""

import numpy as np

from audio.models import AudioMapping
from audio.synthesis import apply_envelope, mix, percussive_burst, sine_wave

import experiments.audio_musicality.register as register
from experiments.audio_musicality.synth_common import one_pole_lowpass

NUM_PARTIALS = 2
"""Fixed for both colors -- distinguishing White/Black is the filter's
job here, not the partial count (unlike production and Candidate A)."""

WHITE_CUTOFF = 0.35  # brighter
BLACK_CUTOFF = 0.18  # darker/duller -- the "smallest possible audible
                      # distinction," moved from harmonic count to filter
HARMONY_CUTOFF = 0.22  # fixed, color-independent -- harmony reflects
                        # position balance, not which side just moved

TENSION_CUTOFF_OFFSET: dict[str, float] = {
    "calm": 0.0,
    "active": 0.04,
    "tense": 0.08,
    "chaotic": 0.14,
}
MAX_CUTOFF = 0.9

ATTACK_SECONDS = 0.04
RELEASE_SECONDS = 0.35

HARMONY_VOICE_WEIGHT = 0.6
SUSTAINED_PEAK_HEADROOM = 0.85
CAPTURE_ACCENT_DURATION_SECONDS = 0.2

WHITE_DEMO_SQUARE = "d4"
BLACK_DEMO_SQUARE = "d5"


def _tension_offset(dynamics_label: str | None) -> float:
    if dynamics_label is None:
        return 0.0
    return TENSION_CUTOFF_OFFSET.get(dynamics_label, 0.0)


def soft_additive_tone(
    frequency_hz: float,
    duration_seconds: float,
    sample_rate: int,
    amplitude: float = 1.0,
    num_partials: int = NUM_PARTIALS,
) -> np.ndarray:
    """
    Additive tone with a steep 1/n^2 partial-amplitude rolloff (vs.
    production additive_tone's 1/n) -- much less energy reaches upper
    partials by construction, independent of any register clamp, so
    this candidate tests whether reducing partial *weighting* alone
    (not just count or ceiling) softens perceived harshness.
    """

    n = int(round(duration_seconds * sample_rate))
    t = np.linspace(0.0, duration_seconds, n, endpoint=False)
    total = np.zeros(n)
    weight_sum = 0.0

    for partial_index in range(1, num_partials + 1):
        weight = 1.0 / (partial_index ** 2)
        weight_sum += weight
        total += weight * np.sin(2 * np.pi * frequency_hz * partial_index * t)

    return total * (amplitude / weight_sum)


def _melody_voice(mapping: AudioMapping, melody_pitch_hz: float, sample_rate: int, duration_seconds: float) -> np.ndarray:
    tone = soft_additive_tone(melody_pitch_hz, duration_seconds, sample_rate, amplitude=mapping.loudness)
    tone = apply_envelope(tone, attack_seconds=ATTACK_SECONDS, release_seconds=RELEASE_SECONDS, sample_rate=sample_rate)

    base_cutoff = WHITE_CUTOFF if mapping.color == "white" else BLACK_CUTOFF
    cutoff = min(base_cutoff + _tension_offset(mapping.dynamics_label), MAX_CUTOFF)

    return one_pole_lowpass(tone, cutoff)


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
        attack_seconds=ATTACK_SECONDS, release_seconds=RELEASE_SECONDS, sample_rate=sample_rate,
    )

    cutoff = min(HARMONY_CUTOFF + _tension_offset(mapping.dynamics_label), MAX_CUTOFF)
    return one_pole_lowpass(tone, cutoff)


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
        soft_additive_tone(pitch_hz, duration_seconds, sample_rate, amplitude=0.8),
        attack_seconds=ATTACK_SECONDS, release_seconds=RELEASE_SECONDS, sample_rate=sample_rate,
    )
    return np.clip(one_pole_lowpass(tone, WHITE_CUTOFF), -1.0, 1.0)


def render_isolated_black_note(sample_rate: int = 44100, duration_seconds: float = 1.6) -> np.ndarray:
    pitch_hz = register.pitch_for_square(BLACK_DEMO_SQUARE)
    tone = apply_envelope(
        soft_additive_tone(pitch_hz, duration_seconds, sample_rate, amplitude=0.8),
        attack_seconds=ATTACK_SECONDS, release_seconds=RELEASE_SECONDS, sample_rate=sample_rate,
    )
    return np.clip(one_pole_lowpass(tone, BLACK_CUTOFF), -1.0, 1.0)


def render_isolated_harmony(sample_rate: int = 44100, duration_seconds: float = 1.6) -> np.ndarray:
    pitch_hz = register.pitch_for_square(WHITE_DEMO_SQUARE)
    ratio = register.quantize_ratio_to_scale(1.2599, register.DIATONIC_HARMONY_RATIOS)
    tone = apply_envelope(
        sine_wave(pitch_hz * ratio, duration_seconds, sample_rate, amplitude=0.7),
        attack_seconds=ATTACK_SECONDS, release_seconds=RELEASE_SECONDS, sample_rate=sample_rate,
    )
    return np.clip(one_pole_lowpass(tone, HARMONY_CUTOFF), -1.0, 1.0)


def render_isolated_accent(sample_rate: int = 44100, duration_seconds: float = 0.4) -> np.ndarray:
    return np.clip(
        apply_envelope(
            percussive_burst(duration_seconds, sample_rate, amplitude=0.9),
            attack_seconds=0.0, release_seconds=0.15, sample_rate=sample_rate,
        ),
        -1.0, 1.0,
    )
