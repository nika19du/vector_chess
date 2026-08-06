import chess

from analysis.attack_influence import PIECE_WEIGHTS
from analysis.geometry import matrix_rc_to_square
from chess_engine.models import SourceCell, SourceField


def calculate_square_mass(
    board: chess.Board,
    square: chess.Square,
) -> float:
    """
    Изчислява знаковата маса ρ на едно поле.

    За разлика от attack_influence.calculate_square_attack_influence,
    тук НЕ гледаме кой атакува полето — гледаме единствено
    каква фигура стои върху него. Няма board.attackers(),
    няма ray-casting, няма blocking — чисто occupancy.

    Празно поле -> 0.
    Бяла фигура -> +тежест.
    Черна фигура -> -тежест.
    """

    piece = board.piece_at(square)

    if piece is None:
        return 0.0

    weight = PIECE_WEIGHTS[piece.piece_type]

    if piece.color == chess.WHITE:
        return weight

    return -weight


def build_source_cells(
    board: chess.Board,
) -> list[SourceCell]:
    """
    Изгражда ρ за всичките 64 полета.
    """

    cells: list[SourceCell] = []

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece is None:
            white_mass = 0.0
            black_mass = 0.0
        elif piece.color == chess.WHITE:
            white_mass = PIECE_WEIGHTS[piece.piece_type]
            black_mass = 0.0
        else:
            white_mass = 0.0
            black_mass = PIECE_WEIGHTS[piece.piece_type]

        difference = white_mass - black_mass

        cells.append(
            SourceCell(
                square=chess.square_name(square),
                white_mass=white_mass,
                black_mass=black_mass,
                difference=difference,
            )
        )

    return cells


def build_source_matrix(
    cells: list[SourceCell],
) -> list[list[float]]:
    """
    Превръща списъка с клетки в матрица 8x8.

    Използва точно същата подредба като
    attack_influence.build_attack_influence_matrix: ред 0 е rank 8,
    за да съвпадат двата модела координатно.
    """

    values_by_square = {
        cell.square: cell.difference
        for cell in cells
    }

    matrix: list[list[float]] = []

    for row_index in range(8):
        row: list[float] = []

        for column_index in range(8):
            square_name = chess.square_name(
                matrix_rc_to_square(row_index, column_index)
            )

            row.append(
                values_by_square[square_name]
            )

        matrix.append(row)

    return matrix


def build_source_field(
    board: chess.Board,
) -> SourceField:
    """
    Изгражда цялото изходно поле ρ.

    Balance:
        положително -> повече бяла маса на дъската;
        отрицателно -> повече черна маса;
        нула        -> материален баланс.
    """

    cells = build_source_cells(board)

    matrix = build_source_matrix(cells)

    total_white_mass = sum(
        cell.white_mass for cell in cells
    )

    total_black_mass = sum(
        cell.black_mass for cell in cells
    )

    balance = total_white_mass - total_black_mass

    return SourceField(
        cells=cells,
        matrix=matrix,
        total_white_mass=total_white_mass,
        total_black_mass=total_black_mass,
        balance=balance,
    )
