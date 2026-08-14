"""
Historical reconstruction of the pre-B3a live engine's Melody/Harmony
DSP formula, for the final A/B/C comparison only.

audio/engine.py was rewritten in place for B3a (no separate git commit
boundary exists between B3 and B3a in this session's history to check
out), so this file reconstructs -- faithfully, not approximately -- the
exact formula B3's own report and engine.py module docstring described:
B2's mode weights/ratios (WHITE_MODES/BLACK_MODES/HARMONY_MODES,
undamped) multiplied by the old `_ArticulationEnvelope` shape (attack ->
plateau -> exponential decay to silence), using the exact constants B3
shipped with (MELODY_ATTACK_SECONDS=0.012/PLATEAU=0.12/DECAY=0.55,
HARMONY_ATTACK_SECONDS=0.045/PLATEAU=0.0/DECAY=1.1).

Not imported by production code. Read-only against
audio/organic_synthesis.py (reuses WHITE_MODES/BLACK_MODES/HARMONY_MODES
so the mode content itself is identical to what B3 actually used, not
re-derived).
"""

import numpy as np

from audio.organic_synthesis import BLACK_MODES, HARMONY_MODES, WHITE_MODES

MELODY_ATTACK_SECONDS = 0.012
MELODY_PLATEAU_SECONDS = 0.12
MELODY_DECAY_SECONDS = 0.55

HARMONY_ATTACK_SECONDS = 0.045
HARMONY_PLATEAU_SECONDS = 0.0
HARMONY_DECAY_SECONDS = 1.1

FLOOR = 1e-4


def _articulation_shape(t: np.ndarray, attack_seconds: float, plateau_seconds: float, decay_seconds: float) -> np.ndarray:
    """Exact reconstruction of audio/engine.py's pre-B3a `_ArticulationEnvelope.render`
    closed form, evaluated from a fresh trigger at t=0 (no mid-decay
    retrigger continuity needed for this one-shot reference render)."""

    plateau_end = attack_seconds + plateau_seconds
    decay_rate = -np.log(FLOOR) / decay_seconds if decay_seconds > 0.0 else float("inf")

    shape = np.ones_like(t)
    attack_mask = t < attack_seconds
    plateau_mask = (~attack_mask) & (t < plateau_end)
    decay_mask = ~attack_mask & ~plateau_mask

    shape[attack_mask] = t[attack_mask] / attack_seconds
    shape[plateau_mask] = 1.0
    shape[decay_mask] = np.exp(-decay_rate * (t[decay_mask] - plateau_end))

    return np.clip(shape, 0.0, 1.0)


def render_legacy_b3_melody(fundamental_hz: float, color: str, duration_seconds: float, sample_rate: int, amplitude: float) -> np.ndarray:
    modes = BLACK_MODES if color == "black" else WHITE_MODES
    n = int(round(duration_seconds * sample_rate))
    t = np.arange(n) / sample_rate

    total = np.zeros(n)
    for weight, ratio in zip(modes.weights, modes.ratios):
        total += weight * np.sin(2 * np.pi * ratio * fundamental_hz * t)
    total *= amplitude / modes.weight_sum

    envelope = _articulation_shape(t, MELODY_ATTACK_SECONDS, MELODY_PLATEAU_SECONDS, MELODY_DECAY_SECONDS)
    return total * envelope


def render_legacy_b3_harmony(fundamental_hz: float, duration_seconds: float, sample_rate: int, amplitude: float) -> np.ndarray:
    modes = HARMONY_MODES
    n = int(round(duration_seconds * sample_rate))
    t = np.arange(n) / sample_rate

    total = np.zeros(n)
    for weight, ratio in zip(modes.weights, modes.ratios):
        total += weight * np.sin(2 * np.pi * ratio * fundamental_hz * t)
    total *= amplitude / modes.weight_sum

    envelope = _articulation_shape(t, HARMONY_ATTACK_SECONDS, HARMONY_PLATEAU_SECONDS, HARMONY_DECAY_SECONDS)
    return total * envelope
