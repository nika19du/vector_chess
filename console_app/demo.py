import chess

from chess_engine.board import ChessGame
from chess_engine.analyzer import analyze_position
game = ChessGame()

moves = [
    "e2e4",
    "d7d5",
    "e4d5",
]

game.print_board()

for move_text in moves:
    print(f"\nОпит за ход: {move_text}")

    result = game.make_move(move_text)

    if result is None:
        print("Невалиден или непозволен ход.")
        continue

    print(result)

    analysis = analyze_position(game.board, result)

    print("\nPosition analysis:")
    print(analysis)

game.print_board()
