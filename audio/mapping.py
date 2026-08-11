import math

from audio.models import AudioMapping
from chess_engine.models import DynamicsAnalysis, MoveAnalysis

# --- Signal 1: color -> timbre ------------------------------------
# White is the simplest member of the palette (pure tone); Black adds
# harmonic richness. The smallest possible audible distinction, per
# docs/audio.md. This is the *base* richness color implies -- the
# actual value stored on AudioMapping additionally passes through
# _effective_harmonic_richness below, which can reduce it further as a
# spectral-safety measure. Kept as its own function (rather than folded
# into the ceiling policy) so "what richness does color intend" and
# "what richness is it safe to actually play" stay two separately
# testable questions.
WHITE_HARMONIC_RICHNESS = 1
BLACK_HARMONIC_RICHNESS = 3

# --- Signal 2: destination square -> pitch -------------------------
# File (a..h) walks a fixed just-intonation major scale (do..do), one
# full octave across the 8 files -- files and scale steps line up
# exactly. Rank (1..8) shifts register.
#
# Phase B product decision (see experiments/audio_musicality/REPORT.md,
# Phase A audit): the register was narrowed from a 3-octave sweep
# (220-3520 Hz) to 2 octaves (220-880 Hz) -- comfortable and alias-free,
# and this alone drops Black's worst-case 3rd-harmonic partial from
# ~10.5 kHz to ~2.6 kHz. OCTAVE_SPAN still describes the *raw*,
# pre-quantization exponential sweep across the 7 rank-to-rank steps;
# the actual sounded pitch is then snapped to the nearest tone of
# LEGAL_MELODY_PITCHES_HZ below (see _pitch_for_square), so the fixed
# per-rank-step ratio the old 3-octave design relied on no longer
# holds exactly -- only the coarser scale-degree/register result does.
BASE_FREQUENCY_HZ = 220.0  # A3

SCALE_RATIOS: dict[str, float] = {
    "a": 1.0,
    "b": 9 / 8,
    "c": 5 / 4,
    "d": 4 / 3,
    "e": 3 / 2,
    "f": 5 / 3,
    "g": 15 / 8,
    "h": 2.0,
}

OCTAVE_SPAN = 2.0  # was 3.0 -- Phase B register-compression decision
RANK_STEPS = 7.0  # ranks 1..8 -> 7 steps


def _build_legal_melody_pitches() -> tuple[float, ...]:
    """
    Every frequency _pitch_for_square is allowed to actually sound:
    the diatonic scale ratios (SCALE_RATIOS) repeated at each whole
    octave layer within the clamped register, from BASE_FREQUENCY_HZ
    up to BASE_FREQUENCY_HZ * 2**OCTAVE_SPAN. With OCTAVE_SPAN=2.0 this
    is 220-880 Hz -- 15 unique tones (7 distinct scale degrees x 2
    octave layers, plus the shared top/bottom boundary at 880 Hz where
    layer 1's "h" ratio (2.0) coincides with the scale's own top).
    Computed once at import time (pure function of fixed constants).
    """

    pitches: set[float] = set()

    for octave_layer in range(int(OCTAVE_SPAN)):
        for ratio in SCALE_RATIOS.values():
            pitches.add(BASE_FREQUENCY_HZ * ratio * (2.0 ** octave_layer))

    return tuple(sorted(pitches))


LEGAL_MELODY_PITCHES_HZ: tuple[float, ...] = _build_legal_melody_pitches()


def _quantize_to_nearest(raw_value: float, legal_values: tuple[float, ...]) -> float:
    """
    Snaps raw_value to its nearest neighbor in legal_values, measured in
    log2 space (perceptual/musical distance, not linear distance -- the
    gap between ratio 1.0 and 1.05 is far more audible than between 1.9
    and 1.95, even though both differ by 0.05 linearly). Shared by both
    the melody-pitch and harmony-interval quantization below so the two
    use one consistent notion of "nearest."
    """

    target_log = math.log2(raw_value)
    return min(legal_values, key=lambda value: abs(math.log2(value) - target_log))


# --- Spectral safety: absolute harmonic ceiling ---------------------
# Phase A measured Black's summed partials (fundamental + 2nd + 3rd,
# via audio.synthesis.additive_tone) reaching as high as ~9.9 kHz under
# the old 3-octave register -- squarely in the range the ear reads as
# piercing/hiss rather than tone. The OCTAVE_SPAN=2.0 register clamp
# above already bounds this: the worst case is now
# BASE_FREQUENCY_HZ * 2**OCTAVE_SPAN * BLACK_HARMONIC_RICHNESS =
# 880 * 3 = 2640 Hz, comfortably under any piercing threshold on its
# own. HARMONIC_CEILING_HZ is kept as an explicit, independently
# testable second safeguard (chosen from the same Phase A data --
# 4200 Hz sits at the elbow of Black's measured partial-frequency
# distribution: 21% of measured partials exceeded 4000 Hz, only 6%
# exceeded 6000 Hz) so the invariant "no partial energy above this
# ceiling" holds even if the register constants above are ever changed
# independently later, rather than relying solely on today's register
# bound as an accidental side effect.
HARMONIC_CEILING_HZ = 4200.0

# --- Signal 5: Attack Influence balance -> harmony ------------------
# balance == 0 -> unison (most consonant). |balance| growing pushes
# the harmonizing interval toward a tritone (most dissonant interval
# in the 12-tone vocabulary), normalized against a fixed reference
# magnitude.
CONSONANT_RATIO = 1.0
DISSONANT_RATIO = 2.0 ** 0.5
REFERENCE_MAX_BALANCE = 20.0

# --- Signal 4: check -> dissonance ----------------------------------
# Check imposes a dissonance *floor* on top of the balance-driven
# interval (does not replace it) -- an acute, discrete event layered
# onto the continuous harmony bed, per docs/audio.md's "match the
# domain to the mathematics" principle.
CHECK_DISSONANCE_FLOOR = 1.25

# --- Committed-note harmony quantization -----------------------------
# harmony_interval_for_balance (below) stays a continuous function on
# purpose: audio.live_state.interpolate_sonification_state calls it
# directly with a continuously-interpolated balance value to drive
# scrub preview's continuous glide, and that behavior must not change
# in Phase B1. Quantization is therefore applied only once, at the
# point build_audio_mapping stores a *committed* move's ratio -- see
# _quantize_harmony_ratio and its call site below.
#
# Vocabulary: CONSONANT_RATIO (1.0, unison) and DISSONANT_RATIO
# (sqrt(2), tritone) are production's own existing named endpoints,
# kept exactly so quantization never moves the two most meaningful
# values. CHECK_DISSONANCE_FLOOR (1.25) is algebraically 5/4, a just
# major third, and already a SCALE_RATIOS-consistent value -- included
# so the check-dissonance floor invariant survives quantization
# exactly (see test_check_forces_dissonance_floor_even_at_zero_balance).
# 9/8 and 4/3 fill the gap with two more SCALE_RATIOS-consistent steps,
# keeping harmony's vocabulary a subset of melody's own scale rather
# than an unrelated set.
HARMONY_INTERVAL_VOCABULARY: tuple[float, ...] = (
    CONSONANT_RATIO,
    9 / 8,
    CHECK_DISSONANCE_FLOOR,
    4 / 3,
    DISSONANT_RATIO,
)

# --- Signal 6: Dynamics label -> volume/density ---------------------
# The more a position just changed (analysis.dynamics.label, computed
# by analysis.dynamics.analyze_dynamics), the louder the move sounds --
# a direct discrete-bucket-to-loudness-tier mapping. The first move has
# no previous position to compare against, so it gets a neutral,
# non-committal loudness rather than silence or a guessed label.
LOUDNESS_BY_LABEL: dict[str, float] = {
    "calm": 0.35,
    "active": 0.55,
    "tense": 0.75,
    "chaotic": 0.95,
}
NEUTRAL_LOUDNESS = 0.5  # first move: no previous position to compare


def _pitch_for_square(square: str) -> float:
    """
    raw chess value (file, rank)
      -> normalized value (raw_hz: the continuous exponential formula
         below, unchanged in shape from the pre-Phase-B design, just a
         narrower OCTAVE_SPAN)
      -> scale degree + octave/register (snapping raw_hz to the
         nearest tone of LEGAL_MELODY_PITCHES_HZ, which already encodes
         "which of the 7 diatonic scale degrees, in which of the 2
         octave layers")
      -> final frequency (the snapped result, always a member of
         LEGAL_MELODY_PITCHES_HZ)

    Strictly monotonic non-decreasing in rank for a fixed file, and in
    file for a fixed rank (see tests/test_audio_mapping.py) -- ordering
    information is preserved even though the raw continuous value is
    now snapped to a fixed, comfortable, musically-constrained
    vocabulary instead of landing on an arbitrary frequency.
    """

    file_letter = square[0]
    rank_number = int(square[1])

    scale_ratio = SCALE_RATIOS[file_letter]
    octave_multiplier = 2.0 ** ((rank_number - 1) * OCTAVE_SPAN / RANK_STEPS)
    raw_hz = BASE_FREQUENCY_HZ * scale_ratio * octave_multiplier

    return _quantize_to_nearest(raw_hz, LEGAL_MELODY_PITCHES_HZ)


def _harmonic_richness_for_color(color: str) -> int:
    """The richness color *intends* -- see _effective_harmonic_richness
    for the spectral-safety-adjusted value actually stored on
    AudioMapping."""

    return WHITE_HARMONIC_RICHNESS if color == "white" else BLACK_HARMONIC_RICHNESS


def _effective_harmonic_richness(base_richness: int, pitch_hz: float) -> int:
    """
    Reduces base_richness (color's intended partial count) so that the
    highest partial audio.synthesis.additive_tone will actually sum
    (pitch_hz * richness) never exceeds HARMONIC_CEILING_HZ -- enforced
    at the mapping/spectral-policy source, before any synthesis runs,
    rather than by attenuating volume or clipping frequencies after the
    fact. Never returns less than 1 (a move must always produce some
    audible melody note).
    """

    for richness in range(base_richness, 0, -1):
        if pitch_hz * richness <= HARMONIC_CEILING_HZ:
            return richness

    return 1


def harmony_interval_for_balance(balance: float, is_check: bool) -> float:
    """
    Continuous by design -- audio.live_state.interpolate_sonification_state
    calls this directly with a continuously-interpolated balance value
    to drive scrub preview's continuous glide (see docs/interactive_ui.md
    and audio/live_state.py). Do not add quantization here; see
    _quantize_harmony_ratio for the committed-note-only quantization
    step build_audio_mapping applies to this function's result.
    """

    normalized = min(abs(balance) / REFERENCE_MAX_BALANCE, 1.0)
    ratio = CONSONANT_RATIO + normalized * (DISSONANT_RATIO - CONSONANT_RATIO)

    if is_check:
        ratio = max(ratio, CHECK_DISSONANCE_FLOOR)

    return ratio


def _quantize_harmony_ratio(raw_ratio: float) -> float:
    """Snaps a harmony_interval_for_balance result to the nearest member
    of HARMONY_INTERVAL_VOCABULARY -- applied only to committed-move
    mappings (see build_audio_mapping), never to the continuous scrub
    path."""

    return _quantize_to_nearest(raw_ratio, HARMONY_INTERVAL_VOCABULARY)


def _loudness_for_dynamics_label(label: str | None) -> float:
    if label is None:
        return NEUTRAL_LOUDNESS

    return LOUDNESS_BY_LABEL.get(label, NEUTRAL_LOUDNESS)


def build_audio_mapping(
    analysis: MoveAnalysis,
    dynamics: DynamicsAnalysis | None,
) -> AudioMapping:
    """
    Translates one move's already-computed math-layer analysis into a
    fully deterministic musical description. No sound is generated
    here -- see audio.renderer for that.
    """

    dynamics_label = dynamics.label if dynamics is not None else None
    balance = analysis.attack_influence_field.balance
    pitch_hz = _pitch_for_square(analysis.to_square)

    return AudioMapping(
        move=analysis.move,
        color=analysis.color,
        destination_square=analysis.to_square,
        pitch_hz=pitch_hz,
        harmonic_richness=_effective_harmonic_richness(
            _harmonic_richness_for_color(analysis.color), pitch_hz
        ),
        # Signal 3: capture -> accent. Passed through unchanged --
        # the percussive sound itself is synthesized in
        # audio.renderer, not here; mapping.py's only job for this
        # signal is to carry analysis.is_capture forward untouched.
        is_capture=analysis.is_capture,
        is_check=analysis.is_check,
        attack_influence_balance=balance,
        harmony_interval_ratio=_quantize_harmony_ratio(
            harmony_interval_for_balance(balance, analysis.is_check)
        ),
        dynamics_label=dynamics_label,
        loudness=_loudness_for_dynamics_label(dynamics_label),
    )
