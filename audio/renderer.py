import numpy as np

from audio.models import AudioClip, AudioMapping, RenderConfig
from audio.synthesis import additive_tone, apply_envelope, mix, percussive_burst, sine_wave

# The harmony voice sits underneath the melodic (destination-square)
# voice -- foreground = the move itself, background = the balance of
# the position -- so it is mixed in at reduced weight.
HARMONY_VOICE_WEIGHT = 0.6

# Capture accents are short transients layered on top of the sustained
# voices, independent of clip_duration_seconds.
CAPTURE_ACCENT_DURATION_SECONDS = 0.2

# The sustained bed (melody + harmony) is normalized to this peak,
# independently of whether a capture accent exists. Reserving headroom
# here -- rather than normalizing the whole mix including the accent
# in one pass -- means the sustained bed renders identically whether
# or not the move is a capture; a global normalization pass that saw
# the accent too could otherwise shrink the *entire* clip by a
# different factor depending on is_capture, occasionally leaving a
# capture's opening window with less normalized energy than a
# non-capture clip even though the raw signal has more.
SUSTAINED_PEAK_HEADROOM = 0.85


def _harmony_frequency(mapping: AudioMapping) -> float:
    """
    balance >= 0 (White ahead or even) raises the harmonizing interval
    above the melody pitch; balance < 0 (Black ahead) lowers it below --
    the sign of the same balance value used to pick harmony_interval_ratio's
    magnitude also picks its direction.

    Check is NOT handled here. audio.mapping already folds the check
    dissonance floor into mapping.harmony_interval_ratio before this
    function ever sees it -- the renderer trusts that result rather
    than re-deriving it from mapping.is_check, so there is exactly one
    place in the whole pipeline that decides what "check" sounds like.
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


class AudioRenderer:
    def __init__(self, config: RenderConfig = RenderConfig()) -> None:
        self.config = config

    def render(self, mapping: AudioMapping) -> AudioClip:
        """
        Voice precedence for combined events:

        1. Melody (foreground, always present) -- the move itself:
           destination-square pitch, color-driven timbre.
        2. Harmony (background, always present, reduced weight) -- the
           position's balance: direction and dissonance both come
           from mapping.harmony_interval_ratio / attack_influence_balance,
           which already incorporates the check floor (see
           _harmony_frequency's docstring).
        3. Capture accent (independent transient, only when
           mapping.is_capture) -- layered on top via mix(), not
           substituted for either sustained voice.

        A checking capture therefore renders as all three at once:
        the harmony voice already carries check's dissonance floor,
        and the accent is added independently on top of it.
        """

        config = self.config

        melody_voice = apply_envelope(
            additive_tone(
                frequency_hz=mapping.pitch_hz,
                harmonic_richness=mapping.harmonic_richness,
                duration_seconds=config.clip_duration_seconds,
                sample_rate=config.sample_rate,
                amplitude=mapping.loudness,
            ),
            attack_seconds=config.attack_seconds,
            release_seconds=config.release_seconds,
            sample_rate=config.sample_rate,
        )

        harmony_voice = apply_envelope(
            sine_wave(
                frequency_hz=_harmony_frequency(mapping),
                duration_seconds=config.clip_duration_seconds,
                sample_rate=config.sample_rate,
                amplitude=mapping.loudness * HARMONY_VOICE_WEIGHT,
            ),
            attack_seconds=config.attack_seconds,
            release_seconds=config.release_seconds,
            sample_rate=config.sample_rate,
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

            # percussive_burst's own exponential decay gets close to
            # zero but never exactly reaches it before mix() zero-pads
            # past its length -- an explicit release closes that last
            # fraction of a click deterministically, same as the
            # sustained voices above.
            accent = apply_envelope(
                percussive_burst(
                    duration_seconds=accent_duration,
                    sample_rate=config.sample_rate,
                    amplitude=mapping.loudness,
                ),
                attack_seconds=0.0,
                release_seconds=min(config.release_seconds, accent_duration),
                sample_rate=config.sample_rate,
            )

            combined = mix(sustained, accent)
        else:
            combined = sustained

        # Final safety bound: a hard clip, not a proportional rescale.
        # A rescale here would divide the *entire* clip (sustained bed
        # included) by whatever pushed the peak over 1.0 -- almost
        # always the accent -- which would shrink the sustained bed's
        # own contribution to the opening window in exactly the
        # is_capture-dependent way SUSTAINED_PEAK_HEADROOM above is
        # meant to avoid. A hard clip only truncates the handful of
        # samples that actually exceed the range, leaving the rest of
        # the signal (including most of the accent) untouched.
        normalized = np.clip(combined, -1.0, 1.0)

        return AudioClip(samples=normalized, sample_rate=config.sample_rate)
