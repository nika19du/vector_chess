from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import numpy as np
from matplotlib.colors import Normalize

from desktop_app.layers._color_scale import symmetric_max_abs
from desktop_app.position_cache import CacheEntry

# Deliberately NOT RdBu_r: that colormap already means White-influence <->
# Black-influence balance everywhere else in the app
# (desktop_app/layers/attack_influence_layer.py, visualization/
# attack_influence_plot.py). Reusing it here for a completely different axis
# of meaning ("B stronger" <-> "A stronger") would let the two meanings
# visually collide -- the approved Branch Comparison V1 design explicitly
# forbids this. PuOr (purple/orange) shares no hue with RdBu_r's red/blue,
# so a difference panel can never be mistaken for a White/Black balance
# panel at a glance.
DIFFERENCE_COLORMAP_NAME = "PuOr"
DIFFERENCE_ALPHA = 1.0


@dataclass(frozen=True)
class AttackInfluenceComparison:
    """
    A pure, ephemeral comparison of two sibling positions' Attack Influence
    fields -- computed fresh at render/summary time from two already-cached
    `FullPositionAnalysis` results (`PositionCache`, reused exactly as
    designed). Never stored back into `PositionCache`; there is no
    `CompareAnalysis` cache (Branch Comparison V1's approved scope).

    `difference = matrix_b - matrix_a`: positive means B's square is more
    White-favoring (or less Black-favoring) than A's at that square;
    negative means the reverse. This describes how B differs from A, a
    comparison between two counterfactual alternatives -- not a White/Black
    balance statement on its own, and it must never be rendered with the
    same colormap as one (see `difference_colors`).
    """

    matrix_a: np.ndarray
    matrix_b: np.ndarray
    difference: np.ndarray
    mean_absolute_difference: float
    max_absolute_difference: float
    max_difference_square: str | None
    balance_a: float
    balance_b: float
    balance_shift: float


def _square_name(row: int, column: int) -> str:
    """
    row 0 = rank 8, column 0 = file a -- the same convention every other 8x8
    field in this codebase uses (see desktop_app/layers/
    attack_influence_layer.py's GridColorFrame docstring, and analysis/
    geometry.py's matrix_rc_to_square/matrix_row_to_rank).
    """
    file_letter = chr(ord("a") + column)
    rank_number = 8 - row
    return f"{file_letter}{rank_number}"


def compare_attack_influence(entry_a: CacheEntry, entry_b: CacheEntry) -> AttackInfluenceComparison:
    """
    The one mathematical operation Branch Comparison V1 performs: subtracts
    two already-computed Attack Influence matrices, and derives a small set
    of human-interpretable summaries from that difference. Reads
    `entry.analysis.attack_influence_field` exactly as
    `attack_influence_layer.build_attack_influence_frame` does -- the
    production Attack Influence formula (analysis/attack_influence.py) is
    never reimplemented or re-run here, only its cached output is read.
    """
    if entry_a.analysis is None or entry_b.analysis is None:
        raise ValueError("both cache entries must be READY to compare")

    matrix_a = np.array(entry_a.analysis.attack_influence_field.matrix, dtype=float)
    matrix_b = np.array(entry_b.analysis.attack_influence_field.matrix, dtype=float)
    difference = matrix_b - matrix_a

    absolute_difference = np.abs(difference)
    max_absolute_difference = float(np.max(absolute_difference))
    mean_absolute_difference = float(np.mean(absolute_difference))

    # All-zero difference case, handled explicitly rather than reporting a
    # meaningless "largest change" square for a field that didn't change
    # anywhere.
    max_difference_square: str | None = None
    if max_absolute_difference > 0.0:
        row, column = np.unravel_index(np.argmax(absolute_difference), absolute_difference.shape)
        max_difference_square = _square_name(int(row), int(column))

    balance_a = entry_a.analysis.attack_influence_field.balance
    balance_b = entry_b.analysis.attack_influence_field.balance

    return AttackInfluenceComparison(
        matrix_a=matrix_a,
        matrix_b=matrix_b,
        difference=difference,
        mean_absolute_difference=mean_absolute_difference,
        max_absolute_difference=max_absolute_difference,
        max_difference_square=max_difference_square,
        balance_a=balance_a,
        balance_b=balance_b,
        balance_shift=balance_b - balance_a,
    )


def difference_colors(difference: np.ndarray) -> np.ndarray:
    """
    Zero-centered diverging colorization of a B-minus-A difference field,
    on a colormap distinct from RdBu_r (see module docstring) so it can
    never be mistaken for the ordinary White/Black Attack Influence
    palette. The UI must always label this "B - A" alongside it (Branch
    Comparison V1's approved scope) -- this function only supplies color,
    never a legend string.

    Normalization is `symmetric_max_abs(difference)` -- the same
    zero-clamped-at-1.0 scheme every other signed field in this codebase
    uses (desktop_app/layers/_color_scale.py). This handles the all-zero
    difference case safely: `Normalize(-1.0, 1.0)` applied to an all-zero
    matrix produces the colormap's exact midpoint color everywhere, never a
    division by zero.
    """
    max_absolute_value = symmetric_max_abs(difference)
    normalization = Normalize(vmin=-max_absolute_value, vmax=max_absolute_value)
    colormap = matplotlib.colormaps[DIFFERENCE_COLORMAP_NAME]
    colors = colormap(normalization(difference)).astype(np.float32)
    colors[..., 3] *= DIFFERENCE_ALPHA
    return colors
