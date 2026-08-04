import chess
from chess_engine.models import ControlSummary, SquareControl


def count_control(board: chess.Board) -> ControlSummary:
    white_squares = get_controlled_squares(board, chess.WHITE)
    black_squares = get_controlled_squares(board, chess.BLACK)

    white_count = len(white_squares)
    black_count = len(black_squares)

    return ControlSummary(
        white_controlled_squares = white_count,
        black_controlled_squares = black_count,
        difference = white_count - black_count,
    )

def get_controlled_squares(
    board: chess.Board,
    color: chess.Color
) -> set[chess.Square]:
    controlled_squares: set[chess.Square] = set()

    for square, piece in board.piece_map().items():
        if piece.color != color:
            continue

        attacks = board.attacks(square)
        controlled_squares.update(attacks)

    return controlled_squares

def build_control_map(board: chess.Board) -> list[SquareControl]:
    control_map: list[SquareControl] = []

    for square in chess.SQUARES:
        white_attackers = len(board.attackers(chess.WHITE, square))
        black_attackers = len(board.attackers(chess.BLACK, square))
        # board.attackers: колко фигури атакуват дадено поле
        control_map.append(SquareControl(
            square = chess.square_name(square),
            white_attackers = white_attackers,
            black_attackers = black_attackers,
            difference = white_attackers - black_attackers
        ))

    return control_map