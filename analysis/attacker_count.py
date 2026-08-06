import chess

from analysis.geometry import matrix_rc_to_square, matrix_row_to_rank
from chess_engine.models import AttackerCountCell, AttackerCountField


def build_attacker_count_cells(board: chess.Board) -> list[AttackerCountCell]:
    cells: list[AttackerCountCell] = []

    for square in chess.SQUARES:
        white_attackers = len(board.attackers(chess.WHITE, square))
        black_attackers = len(board.attackers(chess.BLACK, square))
        # board.attackers: колко фигури атакуват дадено поле
        cells.append(AttackerCountCell(
            square = chess.square_name(square),
            white_attackers = white_attackers,
            black_attackers = black_attackers,
            difference = white_attackers - black_attackers
        ))

    return cells


def build_attacker_count_field(board: chess.Board) -> AttackerCountField:
    """
    Връща матрица 8x8

    Положително число:
    повече бели фигури атакуват полето.

    Отриц. число:
    повече черни фигури атакуват полето.

    Нула:
    равен брой атакуващи или липса на такива
    """

    cells = build_attacker_count_cells(board)

    values_by_square = {
        cell.square: cell.difference
        for cell in cells
    }

    matrix: list[list[int]] = []

    # започваме от 8ред за да изглежка като шахматна дъска

    for row_index in range(8):
        row: list[int] = []

        for column_index in range(8):
            square_name = chess.square_name(
                matrix_rc_to_square(row_index, column_index)
            )
            row.append(values_by_square[square_name])

        matrix.append(row)

    return AttackerCountField(
        cells=cells,
        matrix=matrix,
    )


def print_heatmap(matrix: list[list[int]]) -> None:
    print("\nAttacker count heatmap")
    print("     a   b   c   d   e   f   g   h")

    for index, row in enumerate(matrix):
        rank = matrix_row_to_rank(index)
        formatted_row = " ".join(f"{value:>3}" for value in row)
        print(f"{rank}  {formatted_row}")
