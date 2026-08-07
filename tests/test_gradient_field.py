import math

import pytest

from analysis.gradient_field import (
    build_gradient_field,
    calculate_gradient_magnitude,
    calculate_x_gradient,
    calculate_y_gradient,
)


def _column_ramp_matrix() -> list[list[float]]:
    """Every row is identical: value == column index (0..7)."""

    return [[float(column) for column in range(8)] for _ in range(8)]


def _row_ramp_matrix() -> list[list[float]]:
    """Every column is identical: value == row index (0..7)."""

    return [[float(row) for _ in range(8)] for row in range(8)]


# ---------------------------------------------------------
# calculate_x_gradient: interior central difference + both boundaries
# ---------------------------------------------------------
#
# matrix[row][column] == column is exactly linear in column, so both
# the central-difference interior formula and the one-sided boundary
# formulas must all recover the exact same slope: 1.0. This pins down
# the boundary handling with a hand-computable expected value, not
# just "does it run."


def test_x_gradient_interior_matches_exact_slope_of_a_linear_ramp():
    matrix = _column_ramp_matrix()

    for column in range(1, 7):
        assert calculate_x_gradient(matrix, row=3, column=column) == pytest.approx(1.0)


def test_x_gradient_left_boundary_uses_forward_difference():
    matrix = _column_ramp_matrix()
    assert calculate_x_gradient(matrix, row=3, column=0) == pytest.approx(1.0)


def test_x_gradient_right_boundary_uses_backward_difference():
    matrix = _column_ramp_matrix()
    assert calculate_x_gradient(matrix, row=3, column=7) == pytest.approx(1.0)


def test_x_gradient_sign_flips_for_a_descending_ramp():
    matrix = [[float(7 - column) for column in range(8)] for _ in range(8)]

    for column in range(1, 7):
        assert calculate_x_gradient(matrix, row=3, column=column) == pytest.approx(-1.0)


# ---------------------------------------------------------
# calculate_y_gradient: interior central difference + both boundaries
# ---------------------------------------------------------
#
# Row 0 is rank 8 and mathematical y increases toward rank 8 (i.e. y
# increases as row DECREASES) -- the whole point of this file's
# existence is to pin that flip down with an exact hand-computable
# case instead of leaving it to only be "probably right" from reading
# the docstring. matrix[row][column] == row means the value increases
# with row, i.e. DECREASES as y increases (moving from row to row-1,
# toward higher y/rank, lands on a smaller stored value) -- so dF/dy
# must come out to exactly -1.0 everywhere, interior and boundaries
# alike, for this exactly-linear-in-row matrix.


def test_y_gradient_interior_matches_exact_slope_accounting_for_the_row_y_flip():
    matrix = _row_ramp_matrix()

    for row in range(1, 7):
        assert calculate_y_gradient(matrix, row=row, column=3) == pytest.approx(-1.0)


def test_y_gradient_top_boundary_rank_8():
    matrix = _row_ramp_matrix()
    assert calculate_y_gradient(matrix, row=0, column=3) == pytest.approx(-1.0)


def test_y_gradient_bottom_boundary_rank_1():
    matrix = _row_ramp_matrix()
    assert calculate_y_gradient(matrix, row=7, column=3) == pytest.approx(-1.0)


def test_y_gradient_sign_flips_when_value_increases_with_y():
    # value == (7 - row): increases as row decreases, i.e. increases
    # as mathematical y increases -- dF/dy must now be +1.0.
    matrix = [[float(7 - row) for _ in range(8)] for row in range(8)]

    for row in range(1, 7):
        assert calculate_y_gradient(matrix, row=row, column=3) == pytest.approx(1.0)

    assert calculate_y_gradient(matrix, row=0, column=3) == pytest.approx(1.0)
    assert calculate_y_gradient(matrix, row=7, column=3) == pytest.approx(1.0)


# ---------------------------------------------------------
# calculate_gradient_magnitude
# ---------------------------------------------------------


def test_gradient_magnitude_is_euclidean_norm():
    assert calculate_gradient_magnitude(3.0, 4.0) == pytest.approx(5.0)
    assert calculate_gradient_magnitude(0.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------
# build_gradient_field: shape validation, vector contract, max_magnitude
# ---------------------------------------------------------


def test_build_gradient_field_rejects_wrong_row_count():
    with pytest.raises(ValueError):
        build_gradient_field([[0.0] * 8 for _ in range(7)])


def test_build_gradient_field_rejects_wrong_column_count():
    matrix = [[0.0] * 8 for _ in range(8)]
    matrix[3] = [0.0] * 7

    with pytest.raises(ValueError):
        build_gradient_field(matrix)


def test_build_gradient_field_zero_matrix_has_zero_everything():
    field = build_gradient_field([[0.0] * 8 for _ in range(8)])

    assert len(field.vectors) == 64
    assert field.max_magnitude == 0.0
    assert all(value == 0.0 for row in field.x_matrix for value in row)
    assert all(value == 0.0 for row in field.y_matrix for value in row)
    assert all(value == 0.0 for row in field.magnitude_matrix for value in row)


def test_build_gradient_field_max_magnitude_matches_the_true_maximum():
    field = build_gradient_field(_column_ramp_matrix())

    expected_max = max(
        calculate_gradient_magnitude(dx, dy)
        for row_dx, row_dy in zip(field.x_matrix, field.y_matrix)
        for dx, dy in zip(row_dx, row_dy)
    )
    assert field.max_magnitude == pytest.approx(expected_max)


def test_build_gradient_field_vector_metadata_matches_its_own_matrices():
    field = build_gradient_field(_column_ramp_matrix())

    # Every GradientVector must carry the same delta_x/delta_y/magnitude
    # already stored in x_matrix/y_matrix/magnitude_matrix for its own
    # (mathematical) coordinates -- the per-vector list and the three
    # matrices are two views of the same computation, not independent
    # ones that could silently drift apart.
    for vector in field.vectors:
        # mathematical (x, y): x = column (0..7), y = 7 - row.
        row = 7 - vector.y
        column = vector.x

        assert field.x_matrix[row][column] == pytest.approx(vector.delta_x)
        assert field.y_matrix[row][column] == pytest.approx(vector.delta_y)
        assert field.magnitude_matrix[row][column] == pytest.approx(vector.magnitude)
        assert vector.magnitude == pytest.approx(
            calculate_gradient_magnitude(vector.delta_x, vector.delta_y)
        )
