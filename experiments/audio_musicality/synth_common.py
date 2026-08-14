"""
Shared DSP/glue helpers for the Phase A candidate prototypes: a simple
one-pole low-pass filter (the one genuinely new "conservative filtering"
technique requirement 7 asks for), RMS-based loudness measurement and
normalization (so the three candidate.wav comparisons are fair), and a
thin wav-writing wrapper around the *production* audio.export.write_wav
(read-only import -- reused, not reimplemented, so container format
matches production output exactly).
"""

from pathlib import Path

import numpy as np

from audio.export import write_wav
from audio.models import AudioClip

DEFAULT_SAMPLE_RATE = 44100

# Conservative headroom target for the comparison tracks -- deliberately
# below the hard-clip ceiling, mirroring production's own
# SUSTAINED_PEAK_HEADROOM philosophy (audio/renderer.py) of leaving
# room before the final [-1, 1] safety bound rather than relying on it.
TARGET_RMS = 0.2


def one_pole_lowpass(samples: np.ndarray, cutoff_coefficient: float) -> np.ndarray:
    """
    y[n] = y[n-1] + cutoff_coefficient * (x[n] - y[n-1])

    A single-parameter recursive low-pass: higher cutoff_coefficient
    (closer to 1.0) lets more high-frequency content through
    (brighter); lower (closer to 0.0) rolls it off more aggressively
    (darker/duller). Deterministic, no dependency beyond numpy.
    """

    if samples.size == 0:
        return samples

    output = np.empty_like(samples)
    previous = 0.0

    for i in range(samples.size):
        previous = previous + cutoff_coefficient * (samples[i] - previous)
        output[i] = previous

    return output


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def peak(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


def normalize_to_rms(samples: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    """
    Scales samples so their RMS matches target_rms, applied identically
    across all three candidates before writing -- so an A/B/C listening
    comparison isn't confounded by one candidate simply being louder.
    Never divides by zero: silent input is returned unchanged.
    """

    current = rms(samples)
    if current <= 0.0:
        return samples

    return samples * (target_rms / current)


def write_clip(samples: np.ndarray, path: Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Path:
    """Wraps samples in the production AudioClip dataclass and writes
    via the production write_wav -- both imported read-only, neither
    modified. Keeps the experimental WAVs byte-identical in container
    format to production's own offline output."""

    clip = AudioClip(samples=samples, sample_rate=sample_rate)
    return write_wav(clip, path)
