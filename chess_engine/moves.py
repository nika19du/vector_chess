import chess
from chess_engine.models import MoveDetails, MoveAnalysis


def execute_move(board: chess.Board, move_text: str) -> MoveDetails | None:
    """
    Validates and executes a move.
    Returns MoveDetails on a successful move.
    Returns None on an invalid format or illegal move.
    """

    try:
        move = chess.Move.from_uci(move_text.strip().lower())
    except ValueError:
        return None

    if move not in board.legal_moves:
        return None

    moved_piece = board.piece_at(move.from_square)
    if moved_piece is None:
        return None

    is_capture = board.is_capture(move)

    piece_name = chess.piece_name(moved_piece.piece_type)
    color = "white" if moved_piece.color == chess.WHITE else "black"

    from_square = chess.square_name(move.from_square)
    to_square = chess.square_name(move.to_square)

    board.push(move)

    is_check = board.is_check()

    return MoveDetails(
        move=move.uci(),
        piece_name=piece_name,
        color=color,
        from_square=from_square,
        to_square=to_square,
        is_capture=is_capture,
        is_check=is_check
    )