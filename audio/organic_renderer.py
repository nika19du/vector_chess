"""
B2: OrganicAudioRenderer -- an offline-only, parallel renderer that
turns an AudioMapping into an AudioClip using damped modal resonance
synthesis (audio/organic_synthesis.py) instead of production's
oscillator-based AudioRenderer (audio/renderer.py).

Deliberately isolated, not a drop-in replacement:
tests/test_audio_live_offline_headroom.py asserts the live AudioEngine's
peak matches AudioRenderer's offline peak exactly -- the live engine
still uses additive oscillators, so changing AudioRenderer's own output
in place would silently break that regression test. OrganicAudioRenderer
therefore lives alongside AudioRenderer as a new, independent path:
same public shape (`render(mapping) -> AudioClip`, `RenderConfig`
constructor) so it's a drop-in comparison target, but it shares no
private helpers with audio/renderer.py (the harmony-direction rule
below is re-derived, not imported, exactly as
experiments/audio_musicality/measure.py already does for the same
reason). Not wired into AudioEngine or console_app -- see the B2 plan.
"""

import numpy as np

from audio.models import AudioClip, AudioMapping, RenderConfig
from audio.organic_synthesis import (
    ACCENT_BASE_FREQUENCY_HZ,
    ACCENT_MODAL_PARAMS,
    BLACK_MODAL_PARAMS,
    HARMONY_MODAL_PARAMS,
    ModalVoiceParams,
    WHITE_MODAL_PARAMS,
    modal_resonance,
)
from audio.synthesis import mix

# Mirrors audio/renderer.py's own HARMONY_VOICE_WEIGHT convention: the
# harmony voice is quieter than melody by design (foreground = the
# move itself, background = the position's balance).
HARMONY_GAIN = 0.6

# Mirrors audio/renderer.py's own SUSTAINED_PEAK_HEADROOM policy: the
# melody+harmony bed is normalized to this peak (scaled down only,
# never up) independently of whether a capture accent exists, so the
# accent's own transient doesn't retroactively shrink the sustained bed.
SUSTAINED_PEAK_HEADROOM = 0.85

CAPTURE_ACCENT_DURATION_SECONDS = 0.2


def _melody_params(color: str) -> ModalVoiceParams:
    return WHITE_MODAL_PARAMS if color == "white" else BLACK_MODAL_PARAMS


def _harmony_frequency(mapping: AudioMapping) -> float:
    """
    Same balance-sign direction rule as audio/renderer.py's own private
    _harmony_frequency (balance >= 0 raises the interval above the
    melody pitch, balance < 0 lowers it below) -- re-derived here
    rather than imported, since it is a private symbol of a module this
    renderer is deliberately decoupled from.
    """

    if mapping.attack_influence_balance >= 0:
        return mapping.pitch_hz * mapping.harmony_interval_ratio

    return mapping.pitch_hz / mapping.harmony_interval_ratio


def _normalize_to_peak(samples: np.ndarray, target_peak: float) -> np.ndarray:
    """Scales down deterministically if the signal exceeds target_peak; never scales up."""

    peak = float(np.max(np.abs(samples))) if samples.size else 0.0

    if peak > target_peak and peak > 0.0:
        return samples * (target_peak / peak)

    return samples


class OrganicAudioRenderer:
    def __init__(self, config: RenderConfig = RenderConfig()) -> None:
        self.config = config

    def render(self, mapping: AudioMapping) -> AudioClip:
        """
        Voice precedence mirrors AudioRenderer.render exactly (Melody
        foreground, Harmony background, Capture accent layered on top)
        -- only the synthesis technique for each voice changes.
        """

        config = self.config

        melody_voice = modal_resonance(
            fundamental_hz=mapping.pitch_hz,
            params=_melody_params(mapping.color),
            duration_seconds=config.clip_duration_seconds,
            sample_rate=config.sample_rate,
            amplitude=mapping.loudness,
            attack_seconds=config.attack_seconds,
        )

        harmony_voice = modal_resonance(
            fundamental_hz=_harmony_frequency(mapping),
            params=HARMONY_MODAL_PARAMS,
            duration_seconds=config.clip_duration_seconds,
            sample_rate=config.sample_rate,
            amplitude=mapping.loudness * HARMONY_GAIN,
            attack_seconds=config.attack_seconds,
        )

        sustained = _normalize_to_peak(
            mix(melody_voice, harmony_voice),
            SUSTAINED_PEAK_HEADROOM,
        )

        if mapping.is_capture:
            accent_duration = min(
                CAPTURE_ACCENT_DURATION_SECONDS,
                config.clip_duration_seconds,
            )

            accent = modal_resonance(
                fundamental_hz=ACCENT_BASE_FREQUENCY_HZ,
                params=ACCENT_MODAL_PARAMS,
                duration_seconds=accent_duration,
                sample_rate=config.sample_rate,
                amplitude=mapping.loudness,
                attack_seconds=0.0,
            )

            combined = mix(sustained, accent)
        else:
            combined = sustained

        # Final safety bound, same rationale as AudioRenderer.render:
        # a hard clip (not a proportional rescale) so an accent
        # transient can't retroactively shrink the sustained bed.
        normalized = np.clip(combined, -1.0, 1.0)

        return AudioClip(samples=normalized, sample_rate=config.sample_rate)
