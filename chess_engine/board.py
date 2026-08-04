import chess
from chess_engine.models import MoveDetails
from chess_engine.moves import execute_move


class ChessGame:
    def __init__(self) -> None:
        self.board = chess.Board()

    def print_board(self) -> None:
        print(self.board)

    def make_move(self, move_text: str) -> MoveDetails | None:
        return execute_move(self.board, move_text)

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def current_turn(self) -> str:
        return "white" if self.board.turn == chess.WHITE else "black"
