import chess
import pytest

from analysis.attack_influence import (
    PIECE_WEIGHTS,
    build_attack_influence_cells,
    build_attack_influence_field,
    build_attack_influence_matrix,
    calculate_square_attack_influence,
    find_strongest_square,
    get_piece_weight,
)
from analysis.geometry import matrix_rc_to_square


# ---------------------------------------------------------
# PIECE_WEIGHTS / get_piece_weight
# ---------------------------------------------------------


def test_get_piece_weight_matches_piece_weights_table_for_every_piece_type():
    for piece_type, expected_weight in PIECE_WEIGHTS.items():
        piece = chess.Piece(piece_type, chess.WHITE)
        assert get_piece_weight(piece) == expected_weight


def test_king_weight_is_finite_not_a_sentinel_for_infinity():
    # Documented design decision (analysis/attack_influence.py): the
    # king does not get an infinite/sentinel weight, because that
    # would destroy the color scale used by every downstream field.
    assert PIECE_WEIGHTS[chess.KING] == 4.0


# ---------------------------------------------------------
# calculate_square_attack_influence
# ---------------------------------------------------------


def test_square_attack_influence_is_zero_with_no_attackers():
    board = chess.Board(None)
    board.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))

    # h8 is not attacked by a rook sitting on a1.
    assert calculate_square_attack_influence(board, chess.H8, chess.WHITE) == 0.0


def test_square_attack_influence_sums_weights_of_all_attackers_of_one_color():
    board = chess.Board(None)
    board.set_piece_at(chess.D4, chess.Piece(chess.ROOK, chess.WHITE))
    board.set_piece_at(chess.B3, chess.Piece(chess.KNIGHT, chess.WHITE))

    # d2 is attacked by both the d4 rook (along the file) and the b3
    # knight (b3 -> d2 is a valid knight move) -- verified directly
    # against python-chess's own board.attackers() before writing this
    # test, not just assumed.
    influence = calculate_square_attack_influence(board, chess.D2, chess.WHITE)
    assert influence == PIECE_WEIGHTS[chess.ROOK] + PIECE_WEIGHTS[chess.KNIGHT]


def test_square_attack_influence_only_counts_the_requested_color():
    board = chess.Board(None)
    board.set_piece_at(chess.D4, chess.Piece(chess.ROOK, chess.WHITE))
    board.set_piece_at(chess.F5, chess.Piece(chess.KNIGHT, chess.BLACK))

    # A piece never attacks the square it stands on -- the rook does
    # not contribute to its own square's white influence, only the
    # black knight (f5 -> d4 is a valid knight move) contributes, and
    # only to the black side.
    assert calculate_square_attack_influence(board, chess.D4, chess.WHITE) == 0.0
    assert (
        calculate_square_attack_influence(board, chess.D4, chess.BLACK)
        == PIECE_WEIGHTS[chess.KNIGHT]
    )


# ---------------------------------------------------------
# build_attack_influence_cells / build_attack_influence_field
# ---------------------------------------------------------


def test_empty_board_has_zero_influence_everywhere_and_balanced_totals():
    board = chess.Board(None)
    field = build_attack_influence_field(board)

    assert all(cell.white_attack_influence == 0.0 for cell in field.cells)
    assert all(cell.black_attack_influence == 0.0 for cell in field.cells)
    assert all(cell.difference == 0.0 for cell in field.cells)
    assert field.total_white_attack_influence == 0.0
    assert field.total_black_attack_influence == 0.0
    assert field.balance == 0.0
    assert field.strongest_white_square is None
    assert field.strongest_black_square is None


def test_mixed_position_produces_the_expected_cell_values_and_balance():
    board = chess.Board(None)
    board.set_piece_at(chess.D4, chess.Piece(chess.ROOK, chess.WHITE))
    board.set_piece_at(chess.B3, chess.Piece(chess.KNIGHT, chess.WHITE))
    board.set_piece_at(chess.F5, chess.Piece(chess.KNIGHT, chess.BLACK))

    field = build_attack_influence_field(board)
    cells_by_square = {cell.square: cell for cell in field.cells}

    d2 = cells_by_square["d2"]
    assert d2.white_attack_influence == PIECE_WEIGHTS[chess.ROOK] + PIECE_WEIGHTS[chess.KNIGHT]
    assert d2.black_attack_influence == 0.0
    assert d2.difference == pytest.approx(d2.white_attack_influence)

    # d4 is the rook's own square (a piece never attacks the square it
    # stands on, so the rook contributes nothing here) but IS attacked
    # by both the white b3 knight (b3 -> d4 is a valid knight move) and
    # the black f5 knight (f5 -> d4) -- a genuinely mixed square where
    # both colors' attacker computations must be independently correct
    # and simply subtracted for difference.
    d4 = cells_by_square["d4"]
    assert d4.white_attack_influence == PIECE_WEIGHTS[chess.KNIGHT]
    assert d4.black_attack_influence == PIECE_WEIGHTS[chess.KNIGHT]
    assert d4.difference == pytest.approx(0.0)

    h8 = cells_by_square["h8"]
    assert h8.white_attack_influence == 0.0
    assert h8.black_attack_influence == 0.0

    # Totals/balance must reconcile with the per-cell values, not be
    # computed independently.
    assert field.total_white_attack_influence == sum(
        cell.white_attack_influence for cell in field.cells
    )
    assert field.total_black_attack_influence == sum(
        cell.black_attack_influence for cell in field.cells
    )
    assert field.balance == pytest.approx(
        field.total_white_attack_influence - field.total_black_attack_influence
    )

    # d2 (8.0) is a strictly higher, unique maximum for white -- no
    # other square is attacked by both pieces at once, so this does
    # not depend on find_strongest_square's tie-breaking behavior.
    # (Black's strongest square is genuinely tied here -- a single f5
    # knight attacks several squares equally -- so it is deliberately
    # not asserted in this test; see the dedicated tie-free scenario
    # below.)
    assert field.strongest_white_square == "d2"


# ---------------------------------------------------------
# build_attack_influence_matrix: row/column <-> square mapping
# ---------------------------------------------------------


def test_matrix_assembly_matches_cells_via_the_shared_coordinate_convention():
    board = chess.Board(None)
    board.set_piece_at(chess.D4, chess.Piece(chess.ROOK, chess.WHITE))
    board.set_piece_at(chess.B3, chess.Piece(chess.KNIGHT, chess.WHITE))
    board.set_piece_at(chess.F5, chess.Piece(chess.KNIGHT, chess.BLACK))

    cells = build_attack_influence_cells(board)
    matrix = build_attack_influence_matrix(cells)

    assert len(matrix) == 8
    assert all(len(row) == 8 for row in matrix)

    cells_by_square = {cell.square: cell for cell in cells}

    # Cross-check every (row, column) against the shared
    # matrix_rc_to_square conversion used everywhere else in the
    # pipeline (analysis/geometry.py) -- if the row/column <-> square
    # mapping ever regresses, this catches it independently of
    # geometry.py's own tests.
    for row in range(8):
        for column in range(8):
            square_name = chess.square_name(matrix_rc_to_square(row, column))
            assert matrix[row][column] == cells_by_square[square_name].difference

    # Row 0 is rank 8 (the board's own documented convention, so the
    # matrix prints top-down like a real board) -- d4's row/column
    # in this convention is row=4 (8 - rank 4), column=3 (file d).
    assert matrix[4][3] == cells_by_square["d4"].difference


# ---------------------------------------------------------
# find_strongest_square
# ---------------------------------------------------------


def test_find_strongest_square_returns_none_for_empty_cell_list():
    assert find_strongest_square([], chess.WHITE) is None
    assert find_strongest_square([], chess.BLACK) is None


def test_find_strongest_square_returns_none_when_all_influence_is_zero():
    board = chess.Board(None)
    cells = build_attack_influence_cells(board)

    assert find_strongest_square(cells, chess.WHITE) is None
    assert find_strongest_square(cells, chess.BLACK) is None


def test_find_strongest_square_identifies_the_unique_maximum_per_color():
    board = chess.Board(None)
    board.set_piece_at(chess.D4, chess.Piece(chess.ROOK, chess.WHITE))
    board.set_piece_at(chess.B3, chess.Piece(chess.KNIGHT, chess.WHITE))
    # b1 and a2 both attack c3 (verified directly against
    # board.attacks() before writing this test) -- a genuinely unique
    # combined-weight maximum for black, distinct from the white
    # pieces' own unique maximum at d2, so neither assertion depends
    # on tie-breaking order among equal-influence squares.
    board.set_piece_at(chess.B1, chess.Piece(chess.KNIGHT, chess.BLACK))
    board.set_piece_at(chess.A2, chess.Piece(chess.KNIGHT, chess.BLACK))

    cells = build_attack_influence_cells(board)

    assert find_strongest_square(cells, chess.WHITE) == "d2"
    assert find_strongest_square(cells, chess.BLACK) == "c3"
