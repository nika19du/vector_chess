import chess

from analysis.geometry import matrix_rc_to_square, matrix_row_to_rank
from chess_engine.models import (
    AttackInfluenceCell,
    AttackInfluenceField,
)


# Тежестта показва колко силно даден тип фигура
# влияе върху полетата, които атакува или защитава.
PIECE_WEIGHTS: dict[int, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,

    # Царят не получава безкрайна стойност,
    # защото това би унищожило цветовата скала.
    chess.KING: 4.0,
}


def get_piece_weight(
    piece: chess.Piece,
) -> float:
    """
    Връща математическата тежест на фигурата.
    """

    return PIECE_WEIGHTS[piece.piece_type]


def calculate_square_attack_influence(
    board: chess.Board,
    square: chess.Square,
    color: chess.Color,
) -> float:
    """
    Изчислява атаковото влияние на един цвят върху едно поле.

    За всяка фигура, която влияе върху полето,
    добавяме нейната тежест.

    Пример:
        пешка + кон влияят върху d5

        attack_influence = 1.0 + 3.0 = 4.0
    """

    total_attack_influence = 0.0

    attackers = board.attackers(
        color,
        square,
    )

    for attacker_square in attackers:
        piece = board.piece_at(
            attacker_square
        )

        if piece is None:
            continue

        total_attack_influence += get_piece_weight(
            piece
        )

    return total_attack_influence


def build_attack_influence_cells(
    board: chess.Board,
) -> list[AttackInfluenceCell]:
    """
    Изгражда стойност на атаково влияние за всичките 64 полета.
    """

    cells: list[AttackInfluenceCell] = []

    for square in chess.SQUARES:
        white_attack_influence = calculate_square_attack_influence(
            board=board,
            square=square,
            color=chess.WHITE,
        )

        black_attack_influence = calculate_square_attack_influence(
            board=board,
            square=square,
            color=chess.BLACK,
        )

        difference = (
            white_attack_influence
            - black_attack_influence
        )

        cells.append(
            AttackInfluenceCell(
                square=chess.square_name(square),
                white_attack_influence=white_attack_influence,
                black_attack_influence=black_attack_influence,
                difference=difference,
            )
        )

    return cells


def build_attack_influence_matrix(
    cells: list[AttackInfluenceCell],
) -> list[list[float]]:
    """
    Превръща списъка с клетки в матрица 8x8.

    Матрицата започва от rank 8,
    за да изглежда като шахматна дъска.
    """

    values_by_square = {
        cell.square: cell.difference
        for cell in cells
    }

    matrix: list[list[float]] = []

    for row in range(8):
        row_values: list[float] = []

        for column in range(8):
            square_name = chess.square_name(
                matrix_rc_to_square(row, column)
            )

            row_values.append(
                values_by_square[square_name]
            )

        matrix.append(row_values)

    return matrix


def find_strongest_square(
    cells: list[AttackInfluenceCell],
    color: chess.Color,
) -> str | None:
    """
    Намира полето с най-силно атаково влияние
    за избрания цвят.
    """

    if not cells:
        return None

    if color == chess.WHITE:
        strongest_cell = max(
            cells,
            key=lambda cell: (
                cell.white_attack_influence
            ),
        )

        if strongest_cell.white_attack_influence == 0:
            return None

    else:
        strongest_cell = max(
            cells,
            key=lambda cell: (
                cell.black_attack_influence
            ),
        )

        if strongest_cell.black_attack_influence == 0:
            return None

    return strongest_cell.square


def build_attack_influence_field(
    board: chess.Board,
) -> AttackInfluenceField:
    """
    Изгражда цялото атаково поле на влияние.

    Balance:
        положително -> повече бяло атаково влияние;
        отрицателно -> повече черно атаково влияние;
        нула       -> равновесие.
    """

    cells = build_attack_influence_cells(board)

    matrix = build_attack_influence_matrix(cells)

    total_white_attack_influence = sum(
        cell.white_attack_influence
        for cell in cells
    )

    total_black_attack_influence = sum(
        cell.black_attack_influence
        for cell in cells
    )

    balance = (
        total_white_attack_influence
        - total_black_attack_influence
    )

    strongest_white_square = (
        find_strongest_square(
            cells,
            chess.WHITE,
        )
    )

    strongest_black_square = (
        find_strongest_square(
            cells,
            chess.BLACK,
        )
    )

    return AttackInfluenceField(
        cells=cells,
        matrix=matrix,
        total_white_attack_influence=(
            total_white_attack_influence
        ),
        total_black_attack_influence=(
            total_black_attack_influence
        ),
        balance=balance,
        strongest_white_square=(
            strongest_white_square
        ),
        strongest_black_square=(
            strongest_black_square
        ),
    )


def print_attack_influence_matrix(
    matrix: list[list[float]],
) -> None:
    """
    Отпечатва матрицата на атаковото влияние в терминала.
    """

    print("\nAttack influence field")
    print(
        "       a      b      c      d"
        "      e      f      g      h"
    )

    for row_index, row in enumerate(matrix):
        rank = matrix_row_to_rank(row_index)

        formatted_row = " ".join(
            f"{value:>6.2f}"
            for value in row
        )

        print(
            f"{rank}  {formatted_row}"
        )
