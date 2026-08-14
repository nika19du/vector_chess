"""
Shared register/scale primitives for the Phase A candidate prototypes.

Not part of production audio/ -- these are experimental replacements
for two spots the diagnosis (see measure.py's output) identified as
"too literal": the melody register spans a full 3 octaves per move
(220-3520 Hz) purely from destination square, and the harmony voice's
interval ratio is continuous/unquantized. Both primitives here keep
the same *ordering* information production encodes (higher rank still
sounds higher; a larger |balance| still sounds more dissonant) while
constraining the *musical vocabulary* the sound is drawn from.

All three candidates import from here rather than duplicating this
logic, so any observed difference between candidates is attributable
to their timbre/filter choices, not to inconsistent register math.
"""

import math

from audio.mapping import SCALE_RATIOS

# Same anchor as production (audio/mapping.py::BASE_FREQUENCY_HZ) --
# reusing 220.0 Hz keeps candidates comparable to production's own
# pitch floor rather than introducing an unrelated new constant.
BASE_FREQUENCY_HZ = 220.0

# Production spans 3 octaves (OCTAVE_SPAN) across the 7 rank-to-rank
# steps -- 220..3520 Hz. Clamping to 2 octaves keeps the top of the
# range at 880 Hz (A5), a comfortable, non-piercing register, while
# still preserving strict monotonic ordering (higher rank -> higher
# pitch) -- ordering information survives; only the absolute spread
# shrinks.
CLAMPED_OCTAVE_SPAN = 2.0
RANK_STEPS = 7.0  # ranks 1..8 -> 7 steps, unchanged from production


def pitch_for_square(square: str) -> float:
    """
    Same formula shape as audio/mapping.py::_pitch_for_square (not
    imported -- that symbol is private to the mapping module), with
    CLAMPED_OCTAVE_SPAN substituted for production's OCTAVE_SPAN=3.0.
    File -> scale-degree correspondence (SCALE_RATIOS) is identical to
    production, imported read-only.
    """

    file_letter = square[0]
    rank_number = int(square[1])

    scale_ratio = SCALE_RATIOS[file_letter]
    octave_multiplier = 2.0 ** ((rank_number - 1) * CLAMPED_OCTAVE_SPAN / RANK_STEPS)

    return BASE_FREQUENCY_HZ * scale_ratio * octave_multiplier


# Harmony-ratio quantization ------------------------------------------
#
# Production's harmony_interval_ratio (audio/mapping.py) is a
# continuous real number in [1.0, sqrt(2)] (or >= 1.25 on check) --
# never snapped to a discrete musical interval, unlike melody pitch
# which is already quantized via SCALE_RATIOS. quantize_ratio_to_scale
# fixes that asymmetry.

DIATONIC_HARMONY_RATIOS: tuple[float, ...] = tuple(SCALE_RATIOS.values())
"""Reuses melody's own scale vocabulary -- a harmony interval always
lands on a ratio the listener has already heard as a melodic step
somewhere on the board, keeping the two voices in one consistent
vocabulary."""

PENTATONIC_RATIOS: tuple[float, ...] = (1.0, 9 / 8, 5 / 4, 3 / 2, 5 / 3)
"""Major pentatonic -- drops the 4th (4/3) and 7th (15/8) scale
degrees, the two steps most responsible for perceived tension/color in
a major scale, leaving a smaller, more uniformly consonant set. Chosen
for candidates that want a more forgiving/ambient result, trading away
some of the check/tension floor's audible bite in exchange."""


def quantize_ratio_to_scale(raw_ratio: float, allowed_ratios: tuple[float, ...]) -> float:
    """
    Snaps raw_ratio to its nearest neighbor in allowed_ratios, measured
    in log-frequency space (perceptual/musical-interval distance, not
    linear ratio distance -- e.g. the perceptual gap between ratio 1.0
    and 1.05 is much larger than between 1.9 and 1.95, even though both
    pairs differ by 0.05 linearly).
    """

    target_log = math.log2(raw_ratio)
    return min(allowed_ratios, key=lambda ratio: abs(math.log2(ratio) - target_log))
