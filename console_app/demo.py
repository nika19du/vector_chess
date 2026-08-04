import chess

from chess_engine.board import ChessGame
from analysis.control import (
    build_control_map,
    count_control,
    get_controlled_squares
)
from analysis.heatmap import build_heatmap, print_heatmap
from analysis.center_of_mass import calculate_control_center
from analysis.vectors import calculate_control_vector
from analysis.attack_vectors import generate_attack_vectors

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
    control = count_control(game.board)
    print(f"White controls: {control.white_controlled_squares}")
    print(f"Black controls: {control.black_controlled_squares}")
    print(f"Difference: {control.difference}")

    white_squares = get_controlled_squares(game.board, chess.WHITE)
    black_squares = get_controlled_squares(game.board, chess.BLACK)

    white_names = sorted(chess.square_name(square) for square in white_squares)
    black_names = sorted(chess.square_name(square) for square in black_squares)
    print("White controlled squares:", ",".join(white_names))
    print("Black controlled squares:", ",".join(black_names))

    heatmap = build_heatmap(game.board)
    print_heatmap(heatmap)

    white_center = calculate_control_center(game.board, chess.WHITE)
    black_center = calculate_control_center(game.board, chess.BLACK)

    print("\nControl centers:")
    if white_center is not None:
        print(
            f"White: x={white_center[0]:.2f}, "
            f"y={white_center[1]:.2f}"
        )
    if black_center is not None:
        print(
            f"Black: x={black_center[0]:.2f}, "
            f"y={black_center[1]:.2f}"
        )

    control_vector = calculate_control_vector(game.board)

    if control_vector is not None:
        print("\nControl vector:")
        print(
            f"Start: ({control_vector.start_x:.2f}, "
            f"{control_vector.start_y:.2f})"
        )
        print(
            f"End: ({control_vector.end_x:.2f}, "
            f"{control_vector.end_y:.2f})"
        )
        print(
            f"Delta: ({control_vector.delta_x:.2f}, "
            f"{control_vector.delta_y:.2f})"
        )
        print(f"Magnitude: {control_vector.magnitude:.2f}")

    attack_vectors = generate_attack_vectors(game.board)

    print(f"\nAttack vectors count: {len(attack_vectors)}")

    for vector in attack_vectors:
        print(
            f"{vector.color} {vector.piece_name}: "
            f"{vector.from_square} -> {vector.to_square}, "
            f"delta=({vector.delta_x}, {vector.delta_y})"
        )

game.print_board()
