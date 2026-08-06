from chess_engine.models import DynamicsAnalysis, MoveAnalysis

def calculate_heatmap_change(
    previous_heatmap: list[list[int]],
    current_heatmap: list[list[int]]
) -> int:
    """
    Изчислява общата абсолютна промяна между две heatmap матрици

    За всяко поле:
        |current_value - previous_value|

    След това събира промените на всичките 64 полета
    """

    total_change = 0

    for row_index in range(8):
        for column_index in range(8):
            previous_value = previous_heatmap[row_index][column_index]
            current_value = current_heatmap[row_index][column_index]

            cell_change = abs(current_value - previous_value)
            total_change += cell_change

    return total_change


def analyze_dynamics(
    previous: MoveAnalysis,
    current: MoveAnalysis
) -> DynamicsAnalysis:
    """
    Сравнява две последователни шахматни позиции

    F = white_mobility - black_mobility
    delta F = f_current - f_previous
    """

    previous_force = previous.mobility.difference
    current_force = current.mobility.difference

    delta_force = current_force - previous_force

    white_mobility_delta = (
        current.mobility.white_reachable_squares
        - previous.mobility.white_reachable_squares
    )

    black_mobility_delta = (
        current.mobility.black_reachable_squares
        - previous.mobility.black_reachable_squares
    )

    attack_vectors_delta = (
        len(current.attack_vectors)
        - len(previous.attack_vectors)
    )

    heatmap_change = calculate_heatmap_change(
        previous_heatmap=previous.attacker_count_field.matrix,
        current_heatmap=current.attacker_count_field.matrix
    )

    intensity = (
        abs(delta_force)
        + abs(white_mobility_delta)
        + abs(black_mobility_delta)
        + abs(attack_vectors_delta) * 0.5
        + heatmap_change * 0.25
    )

    if intensity < 5:
        label = "calm"
    elif intensity < 12:
        label = "active"
    elif intensity < 20:
        label = "tense"
    else:
        label = "chaotic"

    return DynamicsAnalysis(
        previous_force=previous_force,
        current_force=current_force,
        delta_force=delta_force,
        white_mobility_delta=white_mobility_delta,
        black_mobility_delta=black_mobility_delta,
        attack_vectors_delta = attack_vectors_delta,
        heatmap_change=heatmap_change,
        intensity=intensity,
        label=label
    )
