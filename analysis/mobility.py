import chess
from chess_engine.models import MobilitySummary


def count_mobility(board: chess.Board) -> MobilitySummary:
    white_squares = get_reachable_squares(board, chess.WHITE)
    black_squares = get_reachable_squares(board, chess.BLACK)

    white_count = len(white_squares)
    black_count = len(black_squares)

    return MobilitySummary(
        white_reachable_squares = white_count,
        black_reachable_squares = black_count,
        difference = white_count - black_count,
    )

def get_reachable_squares(
    board: chess.Board,
    color: chess.Color
) -> set[chess.Square]:
    reachable_squares: set[chess.Square] = set()

    for square, piece in board.piece_map().items():
        if piece.color != color:
            continue

        attacks = board.attacks(square)
        reachable_squares.update(attacks)

    return reachable_squares
