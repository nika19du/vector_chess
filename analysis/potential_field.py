import chess

from chess_engine.models import (
    PotentialCell,
    PotentialField,
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


def calculate_square_potential(
    board: chess.Board,
    square: chess.Square,
    color: chess.Color,
) -> float:
    """
    Изчислява потенциала на един цвят върху едно поле.

    За всяка фигура, която влияе върху полето,
    добавяме нейната тежест.

    Пример:
        пешка + кон влияят върху d5

        potential = 1.0 + 3.0 = 4.0
    """

    total_potential = 0.0

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

        total_potential += get_piece_weight(
            piece
        )

    return total_potential


def build_potential_cells(
    board: chess.Board,
) -> list[PotentialCell]:
    """
    Изгражда потенциална стойност за всичките 64 полета.
    """

    cells: list[PotentialCell] = []

    for square in chess.SQUARES:
        white_potential = calculate_square_potential(
            board=board,
            square=square,
            color=chess.WHITE,
        )

        black_potential = calculate_square_potential(
            board=board,
            square=square,
            color=chess.BLACK,
        )

        difference = (
            white_potential
            - black_potential
        )

        cells.append(
            PotentialCell(
                square=chess.square_name(square),
                white_potential=white_potential,
                black_potential=black_potential,
                difference=difference,
            )
        )

    return cells


def build_potential_matrix(
    cells: list[PotentialCell],
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

    for rank in range(8, 0, -1):
        row: list[float] = []

        for file_name in "abcdefgh":
            square_name = (
                f"{file_name}{rank}"
            )

            row.append(
                values_by_square[square_name]
            )

        matrix.append(row)

    return matrix


def find_strongest_square(
    cells: list[PotentialCell],
    color: chess.Color,
) -> str | None:
    """
    Намира полето с най-силен потенциал
    за избрания цвят.
    """

    if not cells:
        return None

    if color == chess.WHITE:
        strongest_cell = max(
            cells,
            key=lambda cell: (
                cell.white_potential
            ),
        )

        if strongest_cell.white_potential == 0:
            return None

    else:
        strongest_cell = max(
            cells,
            key=lambda cell: (
                cell.black_potential
            ),
        )

        if strongest_cell.black_potential == 0:
            return None

    return strongest_cell.square


def build_potential_field(
    board: chess.Board,
) -> PotentialField:
    """
    Изгражда цялото потенциално поле.

    Balance:
        положително -> повече бял потенциал;
        отрицателно -> повече черен потенциал;
        нула       -> равновесие.
    """

    cells = build_potential_cells(board)

    matrix = build_potential_matrix(cells)

    total_white_potential = sum(
        cell.white_potential
        for cell in cells
    )

    total_black_potential = sum(
        cell.black_potential
        for cell in cells
    )

    balance = (
        total_white_potential
        - total_black_potential
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

    return PotentialField(
        cells=cells,
        matrix=matrix,
        total_white_potential=(
            total_white_potential
        ),
        total_black_potential=(
            total_black_potential
        ),
        balance=balance,
        strongest_white_square=(
            strongest_white_square
        ),
        strongest_black_square=(
            strongest_black_square
        ),
    )


def print_potential_matrix(
    matrix: list[list[float]],
) -> None:
    """
    Отпечатва потенциалната матрица в терминала.
    """

    print("\nPotential field")
    print(
        "       a      b      c      d"
        "      e      f      g      h"
    )

    for row_index, row in enumerate(matrix):
        rank = 8 - row_index

        formatted_row = " ".join(
            f"{value:>6.2f}"
            for value in row
        )

        print(
            f"{rank}  {formatted_row}"
        )