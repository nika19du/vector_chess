import math

import chess
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from analysis.geometry import math_to_plot_coords, square_to_plot_coords
from chess_engine.models import (
    AttackVector,
    DynamicsAnalysis,
    MoveAnalysis,
)


# ---------------------------------------------------------
# Visual configuration
# ---------------------------------------------------------

# Показва числата във всяка heatmap клетка.
DEBUG = False

# Възможни стойности:
# "last_move" - само векторите на последно преместената фигура;
# "all"       - всички вектори;
# "none"      - без локални вектори.
VECTOR_MODE = "last_move"

# Скрива векторите, които завършват върху собствена фигура.
# Празните контролирани полета остават видими.
HIDE_DEFENSE_VECTORS = True

# None означава без ограничение.
# Например 3.0 показва само вектори с дължина до 3 полета.
MAX_VECTOR_LENGTH: float | None = 3.0

HEATMAP_ALPHA = 0.72
LOCAL_VECTOR_ALPHA = 0.48


UNICODE_PIECES = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}


def plot_position(
    board: chess.Board,
    analysis: MoveAnalysis,
    dynamics: DynamicsAnalysis | None = None,
) -> None:
    """
    Визуализира шахматната позиция като многослоен модел:

    1. шахматна дъска;
    2. heatmap на броя атакуващи;
    3. локални вектори на влияние;
    4. шахматни фигури;
    5. глобален вектор;
    6. центрове на контрол.
    """

    figure, axes = plt.subplots(figsize=(10, 10))

    draw_board_background(axes)

    heatmap_image = draw_heatmap_overlay(
        axes=axes,
        analysis=analysis,
    )

    draw_attack_vectors(
        axes=axes,
        board=board,
        analysis=analysis,
    )

    draw_pieces(
        axes=axes,
        board=board,
    )

    draw_control_vector(
        axes=axes,
        analysis=analysis,
    )

    draw_control_centers(
        axes=axes,
        analysis=analysis,
    )

    configure_axes(
        axes=axes,
        analysis=analysis,
        dynamics=dynamics,
    )

    # Малка цветова скала извън дъската.
    colorbar = figure.colorbar(
        heatmap_image,
        ax=axes,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label(
        "Black attacker count  ←  balance  →  White attacker count"
    )

    figure.tight_layout()
    plt.show()


def draw_board_background(axes: Axes) -> None:
    """Рисува класическия шахматен фон."""

    for plotted_row in range(8):
        for file_index in range(8):
            is_light_square = (
                plotted_row + file_index
            ) % 2 == 0

            square_color = (
                "#f0d9b5"
                if is_light_square
                else "#b58863"
            )

            axes.add_patch(
                plt.Rectangle(
                    (file_index, plotted_row),
                    1,
                    1,
                    facecolor=square_color,
                    edgecolor="none",
                    zorder=1,
                )
            )


def draw_heatmap_overlay(
    axes: Axes,
    analysis: MoveAnalysis,
):
    """
    Показва разликата:

        white attackers - black attackers

    Отрицателно:
        синьо, преимущество на Черните.

    Нула:
        бяло, равновесие.

    Положително:
        червено, преимущество на Белите.
    """

    heatmap = np.array(
        analysis.attacker_count_field.matrix,
        dtype=float,
    )

    max_absolute_value = max(
        1.0,
        float(np.max(np.abs(heatmap))),
    )

    normalization = Normalize(
        vmin=-max_absolute_value,
        vmax=max_absolute_value,
    )

    image = axes.imshow(
        heatmap,
        extent=(0, 8, 8, 0),
        cmap="RdBu_r",
        norm=normalization,
        alpha=HEATMAP_ALPHA,
        interpolation="nearest",
        zorder=2,
    )

    if DEBUG:
        draw_heatmap_values(
            axes=axes,
            heatmap=heatmap,
        )

    return image


def draw_heatmap_values(
    axes: Axes,
    heatmap: np.ndarray,
) -> None:
    """Показва числовите стойности само в debug режим."""

    for matrix_row in range(8):
        for matrix_column in range(8):
            value = int(
                heatmap[matrix_row][matrix_column]
            )

            axes.text(
                matrix_column + 0.5,
                matrix_row + 0.5,
                str(value),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                zorder=10,
            )


def draw_attack_vectors(
    axes: Axes,
    board: chess.Board,
    analysis: MoveAnalysis,
) -> None:
    """
    Рисува филтрирани локални вектори.

    По подразбиране се показват само векторите,
    генерирани от последно преместената фигура.
    """

    vectors = filter_attack_vectors(
        board=board,
        analysis=analysis,
    )

    for vector in vectors:
        draw_single_attack_vector(
            axes=axes,
            vector=vector,
        )


def filter_attack_vectors(
    board: chess.Board,
    analysis: MoveAnalysis,
) -> list[AttackVector]:
    """Филтрира векторите според визуалните настройки."""

    if VECTOR_MODE == "none":
        return []

    vectors = analysis.attack_vectors

    if VECTOR_MODE == "last_move":
        vectors = [
            vector
            for vector in vectors
            if vector.from_square == analysis.to_square
        ]

    filtered_vectors: list[AttackVector] = []

    for vector in vectors:
        length = math.sqrt(
            vector.delta_x ** 2
            + vector.delta_y ** 2
        )

        if (
            MAX_VECTOR_LENGTH is not None
            and length > MAX_VECTOR_LENGTH
        ):
            continue

        if (
            HIDE_DEFENSE_VECTORS
            and targets_own_piece(
                board=board,
                vector=vector,
            )
        ):
            continue

        filtered_vectors.append(vector)

    return filtered_vectors


def targets_own_piece(
    board: chess.Board,
    vector: AttackVector,
) -> bool:
    """
    Проверява дали крайната точка съдържа фигура
    от същия цвят като началната фигура.
    """

    target_square = chess.parse_square(
        vector.to_square
    )

    target_piece = board.piece_at(target_square)

    if target_piece is None:
        return False

    vector_is_white = vector.color == "white"

    return target_piece.color == vector_is_white


def draw_single_attack_vector(
    axes: Axes,
    vector: AttackVector,
) -> None:
    """Рисува един локален вектор на влияние."""

    plot_start_x, plot_start_y = math_to_plot_coords(
        vector.start_x, vector.start_y
    )
    plot_end_x, plot_end_y = math_to_plot_coords(
        vector.end_x, vector.end_y
    )

    start_x = plot_start_x + 0.5
    start_y = plot_start_y + 0.5

    end_x = plot_end_x + 0.5
    end_y = plot_end_y + 0.5

    delta_x = end_x - start_x
    delta_y = end_y - start_y

    # Стрелката започва малко след центъра на фигурата
    # и приключва преди центъра на целевото поле.
    start_offset = 0.17
    end_factor = 0.80

    visible_start_x = (
        start_x + delta_x * start_offset
    )
    visible_start_y = (
        start_y + delta_y * start_offset
    )

    visible_end_x = (
        start_x + delta_x * end_factor
    )
    visible_end_y = (
        start_y + delta_y * end_factor
    )

    vector_color = (
        "#d73027"
        if vector.color == "white"
        else "#2166ac"
    )

    axes.annotate(
        "",
        xy=(visible_end_x, visible_end_y),
        xytext=(
            visible_start_x,
            visible_start_y,
        ),
        arrowprops={
            "arrowstyle": "->",
            "color": vector_color,
            "linewidth": 1.8,
            "alpha": LOCAL_VECTOR_ALPHA,
            "mutation_scale": 13,
        },
        zorder=4,
    )


def draw_pieces(
    axes: Axes,
    board: chess.Board,
) -> None:
    """Рисува Unicode шахматните фигури."""

    for square, piece in board.piece_map().items():
        x, plotted_y = square_to_plot_coords(square)

        symbol = UNICODE_PIECES[piece.symbol()]

        axes.text(
            x + 0.5,
            plotted_y + 0.5,
            symbol,
            fontsize=32,
            ha="center",
            va="center",
            color="black",
            zorder=7,
        )


def draw_control_centers(
    axes: Axes,
    analysis: MoveAnalysis,
) -> None:
    """Рисува двата центъра на контрол."""

    if analysis.white_center is not None:
        white_x, white_y = analysis.white_center
        plot_white_x, plot_white_y = math_to_plot_coords(white_x, white_y)

        axes.scatter(
            plot_white_x + 0.5,
            plot_white_y + 0.5,
            s=250,
            marker="o",
            facecolor="#d73027",
            edgecolor="black",
            linewidth=1.8,
            alpha=0.95,
            zorder=10,
        )

    if analysis.black_center is not None:
        black_x, black_y = analysis.black_center
        plot_black_x, plot_black_y = math_to_plot_coords(black_x, black_y)

        axes.scatter(
            plot_black_x + 0.5,
            plot_black_y + 0.5,
            s=250,
            marker="X",
            facecolor="#2166ac",
            edgecolor="black",
            linewidth=1.8,
            alpha=0.95,
            zorder=10,
        )


def draw_control_vector(
    axes: Axes,
    analysis: MoveAnalysis,
) -> None:
    """
    Рисува глобалния вектор от белия център
    към черния център.
    """

    vector = analysis.control_vector

    if vector is None:
        return

    plot_start_x, plot_start_y = math_to_plot_coords(
        vector.start_x, vector.start_y
    )
    plot_end_x, plot_end_y = math_to_plot_coords(
        vector.end_x, vector.end_y
    )

    start_x = plot_start_x + 0.5
    start_y = plot_start_y + 0.5

    end_x = plot_end_x + 0.5
    end_y = plot_end_y + 0.5

    axes.annotate(
        "",
        xy=(end_x, end_y),
        xytext=(start_x, start_y),
        arrowprops={
            "arrowstyle": "->",
            "color": "black",
            "linewidth": 4,
            "mutation_scale": 22,
        },
        zorder=9,
    )


def configure_axes(
    axes: Axes,
    analysis: MoveAnalysis,
    dynamics: DynamicsAnalysis | None,
) -> None:
    """Настройва координатите и заглавието."""

    axes.set_xlim(0, 8)
    axes.set_ylim(8, 0)
    axes.set_aspect("equal")

    axes.set_xticks(
        [index + 0.5 for index in range(8)]
    )
    axes.set_xticklabels(list("abcdefgh"))

    axes.set_yticks(
        [index + 0.5 for index in range(8)]
    )
    axes.set_yticklabels(range(8, 0, -1))

    axes.set_xlabel("File")
    axes.set_ylabel("Rank")

    white_mobility = (
        analysis.mobility.white_reachable_squares
    )
    black_mobility = (
        analysis.mobility.black_reachable_squares
    )
    mobility_balance = analysis.mobility.difference

    if dynamics is None:
        dynamics_line = (
            "ΔF: N/A | "
            "Intensity: first analyzed move"
        )
    else:
        dynamics_line = (
            f"ΔF: {dynamics.delta_force:+d} | "
            f"Intensity: {dynamics.intensity:.2f} "
            f"— {dynamics.label}"
        )

    axes.set_title(
        "VectorChess\n"
        f"Move: {analysis.move} | "
        f"{dynamics_line}\n"
        f"White mobility: {white_mobility} | "
        f"Black mobility: {black_mobility} | "
        f"Balance: {mobility_balance:+d}",
        pad=16,
    )

    # Няма legend, защото покрива част от дъската.