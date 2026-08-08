from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioMapping:
    """
    The complete, deterministic musical description of one move --
    pure data, no audio samples yet.

    Every field here must trace back to a named value in
    MoveAnalysis / DynamicsAnalysis, per docs/audio.md's
    "no arbitrary mappings" rule. Fields kept purely for
    traceability (destination_square, attack_influence_balance)
    are marked below.
    """

    move: str
    color: str  # "white" | "black" -> harmonic_richness (timbre)

    destination_square: str  # traceability: which square drove pitch_hz
    pitch_hz: float  # destination_square -> pitch

    harmonic_richness: int  # color -> timbre partial count

    is_capture: bool  # -> percussive accent layer
    is_check: bool  # -> forced dissonance floor

    attack_influence_balance: float  # traceability: raw balance driving harmony
    harmony_interval_ratio: float  # balance -> consonant..dissonant interval

    dynamics_label: str | None  # None only on the first move
    loudness: float  # dynamics_label -> master gain, 0..1


@dataclass(frozen=True)
class AudioClip:
    """Rendered PCM audio -- mono, float64, range [-1.0, 1.0]."""

    samples: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class RenderConfig:
    sample_rate: int = 44100
    clip_duration_seconds: float = 1.6
    attack_seconds: float = 0.01
    release_seconds: float = 0.15
