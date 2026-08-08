from audio.models import AudioMapping
from chess_engine.models import DynamicsAnalysis, MoveAnalysis

# --- Signal 1: color -> timbre ------------------------------------
# White is the simplest member of the palette (pure tone); Black adds
# harmonic richness. The smallest possible audible distinction, per
# docs/audio.md.
WHITE_HARMONIC_RICHNESS = 1
BLACK_HARMONIC_RICHNESS = 3

# --- Signal 2: destination square -> pitch -------------------------
# File (a..h) walks a fixed just-intonation major scale (do..do), one
# full octave across the 8 files -- files and scale steps line up
# exactly. Rank (1..8) shifts register: each successive rank raises
# pitch by a fixed ratio, spanning 3 octaves in total across the 7
# rank-to-rank steps, chosen to keep the whole board's pitch range
# (220 Hz - 3520 Hz) comfortably audible and alias-free.
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

OCTAVE_SPAN = 3.0
RANK_STEPS = 7.0  # ranks 1..8 -> 7 steps

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
    file_letter = square[0]
    rank_number = int(square[1])

    scale_ratio = SCALE_RATIOS[file_letter]
    octave_multiplier = 2.0 ** ((rank_number - 1) * OCTAVE_SPAN / RANK_STEPS)

    return BASE_FREQUENCY_HZ * scale_ratio * octave_multiplier


def _harmonic_richness_for_color(color: str) -> int:
    return WHITE_HARMONIC_RICHNESS if color == "white" else BLACK_HARMONIC_RICHNESS


def _harmony_interval_for_balance(balance: float, is_check: bool) -> float:
    normalized = min(abs(balance) / REFERENCE_MAX_BALANCE, 1.0)
    ratio = CONSONANT_RATIO + normalized * (DISSONANT_RATIO - CONSONANT_RATIO)

    if is_check:
        ratio = max(ratio, CHECK_DISSONANCE_FLOOR)

    return ratio


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

    return AudioMapping(
        move=analysis.move,
        color=analysis.color,
        destination_square=analysis.to_square,
        pitch_hz=_pitch_for_square(analysis.to_square),
        harmonic_richness=_harmonic_richness_for_color(analysis.color),
        # Signal 3: capture -> accent. Passed through unchanged --
        # the percussive sound itself is synthesized in
        # audio.renderer, not here; mapping.py's only job for this
        # signal is to carry analysis.is_capture forward untouched.
        is_capture=analysis.is_capture,
        is_check=analysis.is_check,
        attack_influence_balance=balance,
        harmony_interval_ratio=_harmony_interval_for_balance(
            balance, analysis.is_check
        ),
        dynamics_label=dynamics_label,
        loudness=_loudness_for_dynamics_label(dynamics_label),
    )
