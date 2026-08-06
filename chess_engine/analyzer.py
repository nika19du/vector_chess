import chess

from analysis.attack_vectors import (
    generate_attack_vectors,
)
from analysis.center_of_mass import (
    calculate_control_center,
)
from analysis.mobility import count_mobility
from analysis.attacker_count import build_attacker_count_field
from analysis.attack_influence import (
    build_attack_influence_field,
)
from analysis.vectors import (
    calculate_control_vector,
)
from chess_engine.models import (
    MoveAnalysis,
    MoveDetails,
)
from analysis.gradient_field import (
    build_gradient_field,
)


def analyze_position(
    board: chess.Board,
    move_details: MoveDetails,
) -> MoveAnalysis:
    """
    Събира всички математически анализи
    за позицията след извършения ход.
    """

    mobility = count_mobility(board)

    attacker_count_field = build_attacker_count_field(board)

    white_center = calculate_control_center(
        board,
        chess.WHITE,
    )

    black_center = calculate_control_center(
        board,
        chess.BLACK,
    )

    control_vector = calculate_control_vector(
        board
    )

    attack_vectors = generate_attack_vectors(
        board
    )

    attack_influence_field = build_attack_influence_field(
        board
    )

    gradient_field = build_gradient_field(
        attack_influence_field.matrix
    )

    return MoveAnalysis(
        move=move_details.move,
        piece_name=move_details.piece_name,
        color=move_details.color,
        from_square=move_details.from_square,
        to_square=move_details.to_square,
        is_capture=move_details.is_capture,
        is_check=move_details.is_check,
        mobility=mobility,
        attacker_count_field=attacker_count_field,
        white_center=white_center,
        black_center=black_center,
        control_vector=control_vector,
        attack_vectors=attack_vectors,
        attack_influence_field=attack_influence_field,
        gradient_field=gradient_field,
    )