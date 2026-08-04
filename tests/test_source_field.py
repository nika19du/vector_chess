import chess

from analysis.potential_field import PIECE_WEIGHTS
from analysis.potential_field import build_potential_matrix, build_potential_cells
from analysis.source_field import build_source_field, build_source_matrix, build_source_cells


def material_balance(board: chess.Board) -> float:
    """Independent reference calculation, not reusing build_source_field."""

    total = 0.0

    for piece in board.piece_map().values():
        weight = PIECE_WEIGHTS[piece.piece_type]
        total += weight if piece.color == chess.WHITE else -weight

    return total


def test_conservation_matches_material_balance():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("d5")
    board.push_san("exd5")

    source_field = build_source_field(board)

    assert source_field.balance == material_balance(board)
    assert source_field.total_white_mass - source_field.total_black_mass == material_balance(board)


def test_empty_board_source_field_is_all_zero():
    board = chess.Board(None)

    source_field = build_source_field(board)

    assert source_field.total_white_mass == 0.0
    assert source_field.total_black_mass == 0.0
    assert source_field.balance == 0.0

    for cell in source_field.cells:
        assert cell.white_mass == 0.0
        assert cell.black_mass == 0.0
        assert cell.difference == 0.0

    for row in source_field.matrix:
        for value in row:
            assert value == 0.0


def test_color_swap_negates_source_field():
    white_board = chess.Board(None)
    white_board.set_piece_at(chess.D4, chess.Piece(chess.QUEEN, chess.WHITE))

    black_board = chess.Board(None)
    black_board.set_piece_at(chess.D4, chess.Piece(chess.QUEEN, chess.BLACK))

    white_field = build_source_field(white_board)
    black_field = build_source_field(black_board)

    for white_row, black_row in zip(white_field.matrix, black_field.matrix):
        for white_value, black_value in zip(white_row, black_row):
            assert white_value == -black_value


def test_single_piece_mass_is_localized_and_signed():
    board = chess.Board(None)
    board.set_piece_at(chess.E5, chess.Piece(chess.ROOK, chess.WHITE))

    source_field = build_source_field(board)

    nonzero_cells = [
        cell for cell in source_field.cells if cell.difference != 0.0
    ]

    assert len(nonzero_cells) == 1
    assert nonzero_cells[0].square == "e5"
    assert nonzero_cells[0].difference == PIECE_WEIGHTS[chess.ROOK]
    assert nonzero_cells[0].white_mass == PIECE_WEIGHTS[chess.ROOK]
    assert nonzero_cells[0].black_mass == 0.0


def test_matrix_coordinate_convention_matches_potential_field():
    """
    build_source_matrix must lay out rows/columns exactly like
    build_potential_matrix (row 0 = rank 8, column 0 = file a),
    so the two fields stay directly comparable square-for-square.
    """

    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")

    # build_potential_matrix / build_potential_cells are imported
    # only to confirm both modules agree on an 8x8, list-of-lists
    # shape; the actual convention check is done via a known piece
    # placement below.
    assert len(build_potential_matrix(build_potential_cells(board))) == 8

    source_matrix = build_source_matrix(
        build_source_cells(board)
    )

    assert len(source_matrix) == 8
    assert all(len(row) == 8 for row in source_matrix)

    # A white queen still on d1 must land at row=7 (rank 1),
    # column=3 (file d) — row 0 is rank 8, column 0 is file a.
    queen_row = 8 - 1  # rank 1 -> row index 7
    queen_column = ord("d") - ord("a")

    assert source_matrix[queen_row][queen_column] == PIECE_WEIGHTS[chess.QUEEN]

    # And nowhere else on the board should show that exact value
    # by coincidence for this position.
    occurrences = sum(
        1
        for row in source_matrix
        for value in row
        if value == PIECE_WEIGHTS[chess.QUEEN]
    )
    assert occurrences == 1
