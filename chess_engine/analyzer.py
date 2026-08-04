import chess

from analysis.attack_vectors import generate_attack_vectors
from analysis.center_of_mass import calculate_control_center
from analysis.control import count_control
from analysis.heatmap import build_heatmap
from analysis.vectors import calculate_control_vector
from chess_engine.models import MoveAnalysis, MoveDetails

def analyze_position(
    board:chess.Board,
    move_details:MoveDetails
) -> MoveAnalysis:
    """
    Събира всички анализи за позицията след извършения ход
    """

    control = count_control(board)
    heatmap = build_heatmap(board)

    white_center= calculate_control_center(board, chess.WHITE)
    black_center= calculate_control_center(board, chess.BLACK)

    control_vector = calculate_control_vector(board)
    attack_vectors = generate_attack_vectors(board)

    return MoveAnalysis(
        move = move_details.move,
        piece_name = move_details.piece_name,
        color = move_details.color,
        from_square = move_details.from_square,
        to_square = move_details.to_square,
        is_capture = move_details.is_capture,
        is_check = move_details.is_check,
        control = control,
        heatmap = heatmap,
        white_center = white_center,
        black_center = black_center,
        control_vector = control_vector,
        attack_vectors = attack_vectors
    )