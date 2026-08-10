from dataclasses import dataclass, field
from typing import Mapping

from audio.mapping import harmony_interval_for_balance
from audio.models import AudioMapping


@dataclass(frozen=True)
class SonificationState:
    """
    What should currently be audible -- decoupled from AudioMapping (one
    committed move's clip parameters) so the same shape can describe
    either a settled move or a live scrub preview between two moves.

    Fields split into two groups, per the Phase 5f plan's signal
    categorization (see docs/interactive_ui.md's scrub design and
    audio/mapping.py's six signals):

    - continuous / position-derived: safe to have changed from the
      previously published state at any time (interpolated while
      scrubbing).
    - discrete / move-identity-derived: held constant for as long as
      segment_key is unchanged -- a move has no meaningful "halfway"
      identity, so these never interpolate.
    """

    # continuous, position-derived (Attack Influence balance -> harmony)
    harmony_interval_ratio: float
    harmony_above_melody: bool

    # discrete, move-identity-derived -- constant while segment_key is
    # unchanged
    pitch_hz: float
    harmonic_richness: int
    loudness: float
    segment_key: tuple[str, str]

    # transport / mixer
    master_gain: float = 1.0
    voice_mute: Mapping[str, bool] = field(default_factory=dict)
    voice_solo: Mapping[str, bool] = field(default_factory=dict)


def _lerp(a: float, b: float, t: float) -> float:
    """
    Same exact-endpoint discipline as desktop_app/layers/_lerp.py's lerp
    (t<=0 returns a verbatim, t>=1 returns b verbatim, so scrub endpoints
    reproduce the settled state bit-for-bit) -- duplicated here rather
    than imported, so audio/ stays decoupled from desktop_app.
    """

    if t <= 0.0:
        return a
    if t >= 1.0:
        return b
    return a + (b - a) * t


def sonification_state_from_mapping(
    mapping: AudioMapping,
    segment_key: tuple[str, str],
    *,
    master_gain: float = 1.0,
    voice_mute: Mapping[str, bool] | None = None,
    voice_solo: Mapping[str, bool] | None = None,
) -> SonificationState:
    """
    The settled (non-interpolated) SonificationState for one committed
    move's already-computed AudioMapping -- what a move-driven update
    (session_state.current_node_changed) publishes.
    """

    return SonificationState(
        harmony_interval_ratio=mapping.harmony_interval_ratio,
        harmony_above_melody=mapping.attack_influence_balance >= 0,
        pitch_hz=mapping.pitch_hz,
        harmonic_richness=mapping.harmonic_richness,
        loudness=mapping.loudness,
        segment_key=segment_key,
        master_gain=master_gain,
        voice_mute=dict(voice_mute) if voice_mute is not None else {},
        voice_solo=dict(voice_solo) if voice_solo is not None else {},
    )


def interpolate_sonification_state(
    segment_mapping: AudioMapping,
    balance_a: float,
    balance_b: float,
    t: float,
    segment_key: tuple[str, str],
    *,
    master_gain: float = 1.0,
    voice_mute: Mapping[str, bool] | None = None,
    voice_solo: Mapping[str, bool] | None = None,
) -> SonificationState:
    """
    Pure, deterministic scrub-preview state for a segment between two
    real positions A and B, connected by the real move that
    segment_mapping already describes.

    Only the one position-derived signal (Attack Influence balance ->
    harmony) is continuously interpolated, by linearly interpolating the
    raw balance at A and B and re-running it through the same pure
    harmony_interval_for_balance already used by build_audio_mapping --
    no new math, the exact same formula. Every move-identity-derived
    signal (pitch, timbre, loudness) is held constant at
    segment_mapping's values: a move has no meaningful halfway pitch or
    timbre, only the position it connects has a continuously varying
    field. is_check is likewise held constant for the whole segment
    (segment_mapping.is_check), even though it is technically a property
    of position B alone -- it describes which move this segment *is*,
    not a threshold balance crosses partway through the drag.
    """

    interpolated_balance = _lerp(balance_a, balance_b, t)

    return SonificationState(
        harmony_interval_ratio=harmony_interval_for_balance(
            interpolated_balance, segment_mapping.is_check
        ),
        harmony_above_melody=interpolated_balance >= 0,
        pitch_hz=segment_mapping.pitch_hz,
        harmonic_richness=segment_mapping.harmonic_richness,
        loudness=segment_mapping.loudness,
        segment_key=segment_key,
        master_gain=master_gain,
        voice_mute=dict(voice_mute) if voice_mute is not None else {},
        voice_solo=dict(voice_solo) if voice_solo is not None else {},
    )
