import chess


def calculate_control_center(
    board: chess.Board,
    color: chess.Color
) -> tuple[float, float] | None:
    total_weight = 0
    weighted_x= 0
    weighted_y = 0

    for square in chess.SQUARES:
        attackers_count = len(board.attackers(color, square))

        if attackers_count == 0:
            continue

        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)

        weighted_x += file_index * attackers_count
        weighted_y += rank_index * attackers_count
        total_weight += attackers_count

    if total_weight == 0:
        return None

    center_x = weighted_x / total_weight
    center_y = weighted_y / total_weight

    return center_x, center_y
