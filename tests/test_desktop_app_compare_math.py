import chess
import numpy as np
import pytest

from desktop_app.compare_math import (
    DIFFERENCE_COLORMAP_NAME,
    compare_attack_influence,
    difference_colors,
)
from desktop_app.layers.attack_influence_layer import COLORMAP_NAME as ATTACK_INFLUENCE_COLORMAP_NAME
from desktop_app.position_cache import CacheEntry, CacheEntryState
from tests.conftest import ready_cache_entry


class _FakeField:
    def __init__(self, matrix, balance):
        self.matrix = matrix
        self.balance = balance


class _FakeAnalysis:
    def __init__(self, matrix, balance):
        self.attack_influence_field = _FakeField(matrix, balance)


def _fake_entry(matrix, balance=0.0) -> CacheEntry:
    return CacheEntry(state=CacheEntryState.READY, analysis=_FakeAnalysis(matrix, balance))


# ---------------------------------------------------------
# compare_attack_influence: exact difference, real production pipeline
# ---------------------------------------------------------


def test_compare_attack_influence_reads_from_the_real_production_pipeline():
    entry_a = ready_cache_entry(chess.Board())
    board_b = chess.Board()
    board_b.push(chess.Move.from_uci("e2e4"))
    entry_b = ready_cache_entry(board_b)

    matrix_a = np.array(entry_a.analysis.attack_influence_field.matrix, dtype=float)
    matrix_b = np.array(entry_b.analysis.attack_influence_field.matrix, dtype=float)

    comparison = compare_attack_influence(entry_a, entry_b)

    assert (comparison.matrix_a == matrix_a).all()
    assert (comparison.matrix_b == matrix_b).all()
    assert (comparison.difference == (matrix_b - matrix_a)).all()
    assert comparison.balance_a == entry_a.analysis.attack_influence_field.balance
    assert comparison.balance_b == entry_b.analysis.attack_influence_field.balance
    assert comparison.balance_shift == pytest.approx(comparison.balance_b - comparison.balance_a)


def test_compare_attack_influence_zero_difference_when_comparing_a_position_with_itself():
    entry = ready_cache_entry(chess.Board())

    comparison = compare_attack_influence(entry, entry)

    assert (comparison.difference == 0.0).all()
    assert comparison.mean_absolute_difference == 0.0
    assert comparison.max_absolute_difference == 0.0
    assert comparison.max_difference_square is None
    assert comparison.balance_shift == 0.0


def test_compare_attack_influence_raises_if_entry_a_is_not_ready():
    entry_a = CacheEntry(state=CacheEntryState.MISSING)
    entry_b = ready_cache_entry(chess.Board())

    with pytest.raises(ValueError):
        compare_attack_influence(entry_a, entry_b)


def test_compare_attack_influence_raises_if_entry_b_is_not_ready():
    entry_a = ready_cache_entry(chess.Board())
    entry_b = CacheEntry(state=CacheEntryState.COMPUTING)

    with pytest.raises(ValueError):
        compare_attack_influence(entry_a, entry_b)


# ---------------------------------------------------------
# max_difference_square: exact arithmetic control via fake entries
# ---------------------------------------------------------


def test_max_difference_square_identifies_the_correct_board_square():
    matrix_a = np.zeros((8, 8))
    matrix_b = np.zeros((8, 8))
    # row 0 = rank 8, column 0 = file a -- so (row=0, column=4) is e8.
    matrix_b[0, 4] = 5.0

    comparison = compare_attack_influence(_fake_entry(matrix_a), _fake_entry(matrix_b))

    assert comparison.max_difference_square == "e8"
    assert comparison.max_absolute_difference == pytest.approx(5.0)


def test_max_difference_square_uses_absolute_value_for_a_negative_change():
    matrix_a = np.zeros((8, 8))
    matrix_b = np.zeros((8, 8))
    # row 7 = rank 1, column 0 = file a -- a1.
    matrix_b[7, 0] = -3.0

    comparison = compare_attack_influence(_fake_entry(matrix_a), _fake_entry(matrix_b))

    assert comparison.max_difference_square == "a1"
    assert comparison.max_absolute_difference == pytest.approx(3.0)


def test_mean_absolute_difference_matches_manual_computation():
    matrix_a = np.zeros((8, 8))
    matrix_b = np.zeros((8, 8))
    matrix_b[0, 0] = 2.0
    matrix_b[1, 1] = -4.0

    comparison = compare_attack_influence(_fake_entry(matrix_a), _fake_entry(matrix_b))

    expected_mean = (2.0 + 4.0) / 64
    assert comparison.mean_absolute_difference == pytest.approx(expected_mean)
    assert comparison.max_absolute_difference == pytest.approx(4.0)


def test_balance_shift_reflects_which_side_gained():
    comparison = compare_attack_influence(
        _fake_entry(np.zeros((8, 8)), balance=1.0),
        _fake_entry(np.zeros((8, 8)), balance=4.0),
    )

    assert comparison.balance_a == 1.0
    assert comparison.balance_b == 4.0
    assert comparison.balance_shift == pytest.approx(3.0)


# ---------------------------------------------------------
# difference_colors: distinct palette, safe zero-difference handling
# ---------------------------------------------------------


def test_difference_colormap_is_distinct_from_the_attack_influence_colormap():
    # The core semantic-collision guard from the approved design: a B-minus-A
    # difference must never be rendered with the same colormap as the
    # ordinary White/Black Attack Influence field.
    assert DIFFERENCE_COLORMAP_NAME != ATTACK_INFLUENCE_COLORMAP_NAME


def test_difference_colors_shape_matches_the_input_grid():
    difference = np.random.default_rng(0).normal(size=(8, 8))

    colors = difference_colors(difference)

    assert colors.shape == (8, 8, 4)


def test_difference_colors_handles_the_all_zero_case_safely():
    colors = difference_colors(np.zeros((8, 8)))

    assert np.isfinite(colors).all()
    # Every cell must land on the colormap's exact midpoint color -- an
    # all-zero field carries no information about which square changed more
    # than another, so every cell must render identically.
    first_cell = colors[0, 0]
    assert (colors == first_cell).all()


def test_difference_colors_are_symmetric_around_zero():
    difference = np.zeros((8, 8))
    difference[0, 0] = 3.0
    difference[1, 1] = -3.0

    colors = difference_colors(difference)

    # Equal-magnitude opposite-sign differences must be equidistant from
    # (but not equal to) the midpoint color on a zero-centered diverging map.
    assert not (colors[0, 0] == colors[1, 1]).all()
    midpoint = difference_colors(np.zeros((8, 8)))[0, 0]
    assert not (colors[0, 0] == midpoint).all()
    assert not (colors[1, 1] == midpoint).all()
