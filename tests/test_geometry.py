import chess

from analysis.geometry import (
    math_coords_to_matrix_rc,
    math_coords_to_square,
    math_to_plot_coords,
    matrix_rc_to_math_coords,
    matrix_rc_to_square,
    matrix_row_to_rank,
    plot_coords_to_square,
    plot_to_math_coords,
    square_to_math_coords,
    square_to_matrix_rc,
    square_to_plot_coords,
)


def test_square_to_math_coords_corners():
    assert square_to_math_coords(chess.A1) == (0, 0)
    assert square_to_math_coords(chess.H1) == (7, 0)
    assert square_to_math_coords(chess.A8) == (0, 7)
    assert square_to_math_coords(chess.H8) == (7, 7)


def test_square_to_math_coords_center_squares():
    assert square_to_math_coords(chess.D4) == (3, 3)
    assert square_to_math_coords(chess.E4) == (4, 3)
    assert square_to_math_coords(chess.D5) == (3, 4)
    assert square_to_math_coords(chess.E5) == (4, 4)


def test_square_to_plot_coords_matches_existing_7_minus_y_convention():
    # This pins the exact formula every hand-written `7 - chess.square_rank(...)`
    # call site used before the extraction, so the refactor can't silently
    # change orientation.
    assert square_to_plot_coords(chess.A1) == (0, 7)
    assert square_to_plot_coords(chess.H1) == (7, 7)
    assert square_to_plot_coords(chess.A8) == (0, 0)
    assert square_to_plot_coords(chess.H8) == (7, 0)


def test_square_to_matrix_rc_row_zero_is_rank_8():
    assert square_to_matrix_rc(chess.A8) == (0, 0)
    assert square_to_matrix_rc(chess.H8) == (0, 7)
    assert square_to_matrix_rc(chess.A1) == (7, 0)
    assert square_to_matrix_rc(chess.H1) == (7, 7)


def test_matrix_row_to_rank():
    assert matrix_row_to_rank(0) == 8
    assert matrix_row_to_rank(7) == 1


def test_square_round_trip_via_math_coords():
    for square in chess.SQUARES:
        x, y = square_to_math_coords(square)
        assert math_coords_to_square(x, y) == square


def test_square_round_trip_via_plot_coords():
    for square in chess.SQUARES:
        x, y = square_to_plot_coords(square)
        assert plot_coords_to_square(x, y) == square


def test_square_round_trip_via_matrix_rc():
    for square in chess.SQUARES:
        row, column = square_to_matrix_rc(square)
        assert matrix_rc_to_square(row, column) == square


def test_math_and_plot_coords_are_mutual_inverses():
    for x in range(8):
        for y in range(8):
            plot_x, plot_y = math_to_plot_coords(x, y)
            assert plot_to_math_coords(plot_x, plot_y) == (x, y)


def test_matrix_rc_and_math_coords_are_mutual_inverses():
    for row in range(8):
        for column in range(8):
            x, y = matrix_rc_to_math_coords(row, column)
            assert math_coords_to_matrix_rc(x, y) == (row, column)
