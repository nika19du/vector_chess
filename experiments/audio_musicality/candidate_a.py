"""
Candidate A -- warm / acoustic-like.

White = Karplus-Strong plucked string (a new synthesis technique, not
present in audio/synthesis.py). Black = a capped-partial, fast-decay
"mallet" tone built from production's own additive_tone (read-only
import), so Black still traces back to the same harmonic_richness
signal production uses -- just capped and reshaped, not replaced by an
unrelated timbre.

Register: 2-octave clamp (register.py) + pentatonic quantization --
the most restrictive, most consonant scale of the three candidates,
matching an acoustic/melodic identity.

OPEN DESIGN QUESTION, surfaced for the report rather than resolved
here: docs/audio.md Section 2 states physical modeling (plucked/bowed
strings) is "reserved, at most, for a single rare cadential gesture at
checkmate" -- explicitly NOT a base identity. Using Karplus-Strong as
White's voice on every single move is a base-identity use of physical
modeling, and directly conflicts with that stated principle. This
candidate is still built and rendered so it can be *heard* and
evaluated, but the conflict must be stated plainly in REPORT.md, not
quietly worked around.
"""

from collections import deque

import numpy as np

from audio.mapping import SCALE_RATIOS
from audio.models import AudioMapping
from audio.synthesis import additive_tone, apply_envelope, mix, percussive_burst

import experiments.audio_musicality.register as register

DECAY = 0.995
"""Single named, explainable Karplus-Strong feedback parameter --
controls how quickly the plucked string's energy dissipates. Not
tuned per-note; one constant for the whole voice."""

MALLET_DECAY_RATE = 6.0  # per second -- fast, percussive but tonal
MALLET_ATTACK_SECONDS = 0.005

HARMONY_VOICE_WEIGHT = 0.6  # mirrors audio/renderer.py's own weighting
SUSTAINED_PEAK_HEADROOM = 0.85  # mirrors audio/renderer.py's own headroom
CAPTURE_ACCENT_DURATION_SECONDS = 0.2

WHITE_DEMO_SQUARE = "d4"
BLACK_DEMO_SQUARE = "d5"


def quantized_pitch_for_square(square: str) -> float:
    """register.pitch_for_square's formula, with the file->scale-degree
    ratio additionally snapped to the pentatonic set before the
    (still-continuous) octave multiplier is applied."""

    file_letter = square[0]
    rank_number = int(square[1])

    diatonic_ratio = SCALE_RATIOS[file_letter]
    quantized_ratio = register.quantize_ratio_to_scale(diatonic_ratio, register.PENTATONIC_RATIOS)
    octave_multiplier = 2.0 ** ((rank_number - 1) * register.CLAMPED_OCTAVE_SPAN / register.RANK_STEPS)

    return register.BASE_FREQUENCY_HZ * quantized_ratio * octave_multiplier


def karplus_strong_pluck(
    frequency_hz: float,
    duration_seconds: float,
    sample_rate: int,
    amplitude: float = 1.0,
    decay: float = DECAY,
) -> np.ndarray:
    """
    Classic Karplus-Strong plucked string: a delay line of length
    N = round(sample_rate / frequency_hz) is initialized with a short
    excitation, then repeatedly averages its front two samples (a
    one-zero low-pass) and feeds the (decayed) result back in.

    Excitation choice: a short, deterministic burst built from three
    odd harmonics (1st, 3rd, 5th) tapered to zero across the delay
    line -- NOT the textbook random-noise burst. This keeps the whole
    candidate reproducible without a fixed-seed RNG. A seeded white-
    noise burst (numpy.random.default_rng(seed=0)) is the more
    traditional excitation and would still be fully deterministic
    under docs/audio.md's "same input, same output" rule -- flagged
    here as an open alternative for the report, not silently chosen.
    """

    n_total = int(round(duration_seconds * sample_rate))
    delay_length = max(2, int(round(sample_rate / frequency_hz)))

    t = np.arange(delay_length)
    excitation = np.zeros(delay_length)
    odd_harmonics = (1, 3, 5)
    weight_sum = 0.0

    for harmonic in odd_harmonics:
        weight = 1.0 / harmonic
        weight_sum += weight
        excitation += weight * np.sin(2 * np.pi * harmonic * frequency_hz * t / sample_rate)

    excitation *= amplitude / weight_sum
    excitation *= np.linspace(1.0, 0.0, delay_length)  # taper -- smooth onset, no hard click

    buffer = deque(excitation.tolist())
    output = np.empty(n_total)

    for i in range(n_total):
        output[i] = buffer[0]
        averaged = decay * 0.5 * (buffer[0] + buffer[1])
        buffer.append(averaged)
        buffer.popleft()

    return output


def mallet_tone(
    frequency_hz: float,
    harmonic_richness: int,
    duration_seconds: float,
    sample_rate: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Black's voice: production's additive_tone, capped at 2 partials
    (vs. production's uncapped harmonic_richness, up to 3), reshaped
    with a fast exponential decay -- a struck, resonant mallet-like
    character rather than a sustained tone."""

    capped_richness = min(harmonic_richness, 2)
    tone = additive_tone(
        frequency_hz=frequency_hz,
        harmonic_richness=capped_richness,
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        amplitude=amplitude,
    )

    n = tone.size
    t = np.linspace(0.0, duration_seconds, n, endpoint=False)
    decay_envelope = np.exp(-MALLET_DECAY_RATE * t)

    attack_n = min(int(round(MALLET_ATTACK_SECONDS * sample_rate)), n)
    attack_envelope = np.ones(n)
    if attack_n > 0:
        attack_envelope[:attack_n] = np.linspace(0.0, 1.0, attack_n, endpoint=False)

    return tone * decay_envelope * attack_envelope


def _melody_voice(mapping: AudioMapping, melody_pitch_hz: float, sample_rate: int, duration_seconds: float) -> np.ndarray:
    if mapping.color == "white":
        return karplus_strong_pluck(
            frequency_hz=melody_pitch_hz,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            amplitude=mapping.loudness,
        )

    return mallet_tone(
        frequency_hz=melody_pitch_hz,
        harmonic_richness=mapping.harmonic_richness,
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        amplitude=mapping.loudness,
    )


def _harmony_voice(mapping: AudioMapping, melody_pitch_hz: float, sample_rate: int, duration_seconds: float) -> np.ndarray:
    quantized_ratio = register.quantize_ratio_to_scale(
        mapping.harmony_interval_ratio, register.DIATONIC_HARMONY_RATIOS
    )
    harmony_hz = (
        melody_pitch_hz * quantized_ratio
        if mapping.attack_influence_balance >= 0
        else melody_pitch_hz / quantized_ratio
    )

    from audio.synthesis import sine_wave

    return apply_envelope(
        sine_wave(
            frequency_hz=harmony_hz,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            amplitude=mapping.loudness * HARMONY_VOICE_WEIGHT,
        ),
        attack_seconds=0.02,
        release_seconds=0.2,
        sample_rate=sample_rate,
    )


def render_move(mapping: AudioMapping, sample_rate: int = 44100, duration_seconds: float = 1.6) -> np.ndarray:
    melody_pitch_hz = quantized_pitch_for_square(mapping.destination_square)

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
    pitch_hz = quantized_pitch_for_square(WHITE_DEMO_SQUARE)
    return np.clip(karplus_strong_pluck(pitch_hz, duration_seconds, sample_rate, amplitude=0.8), -1.0, 1.0)


def render_isolated_black_note(sample_rate: int = 44100, duration_seconds: float = 1.6) -> np.ndarray:
    pitch_hz = quantized_pitch_for_square(BLACK_DEMO_SQUARE)
    return np.clip(mallet_tone(pitch_hz, harmonic_richness=3, duration_seconds=duration_seconds, sample_rate=sample_rate, amplitude=0.8), -1.0, 1.0)


def render_isolated_harmony(sample_rate: int = 44100, duration_seconds: float = 1.6) -> np.ndarray:
    from audio.synthesis import sine_wave

    pitch_hz = quantized_pitch_for_square(WHITE_DEMO_SQUARE)
    ratio = register.quantize_ratio_to_scale(1.2599, register.DIATONIC_HARMONY_RATIOS)
    return np.clip(
        apply_envelope(
            sine_wave(pitch_hz * ratio, duration_seconds, sample_rate, amplitude=0.7),
            attack_seconds=0.02, release_seconds=0.2, sample_rate=sample_rate,
        ),
        -1.0, 1.0,
    )


def render_isolated_accent(sample_rate: int = 44100, duration_seconds: float = 0.4) -> np.ndarray:
    return np.clip(
        apply_envelope(
            percussive_burst(duration_seconds, sample_rate, amplitude=0.9),
            attack_seconds=0.0, release_seconds=0.15, sample_rate=sample_rate,
        ),
        -1.0, 1.0,
    )
