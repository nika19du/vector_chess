import chess
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from analysis.geometry import square_to_plot_coords
from chess_engine.models import MoveAnalysis


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


def plot_attack_influence_field(
    board: chess.Board,
    analysis: MoveAnalysis,
) -> None:
    """
    Рисува претегленото поле на атаково влияние.

    Червено:
        по-силно бяло атаково влияние.

    Синьо:
        по-силно черно атаково влияние.

    Бяло:
        равновесие.
    """

    figure, axes = plt.subplots(figsize=(10, 10))

    draw_board_background(axes)

    image = draw_attack_influence_overlay(
        axes=axes,
        analysis=analysis,
    )

    draw_pieces(
        axes=axes,
        board=board,
    )

    draw_strongest_squares(
        axes=axes,
        analysis=analysis,
    )

    configure_axes(
        axes=axes,
        analysis=analysis,
    )

    colorbar = figure.colorbar(
        image,
        ax=axes,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Black attack influence  ←  balance  →  White attack influence"
    )

    figure.tight_layout()
    plt.show()


def draw_board_background(
    axes: Axes,
) -> None:
    for row in range(8):
        for column in range(8):
            is_light_square = (
                row + column
            ) % 2 == 0

            color = (
                "#f0d9b5"
                if is_light_square
                else "#b58863"
            )

            axes.add_patch(
                plt.Rectangle(
                    (column, row),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="none",
                    zorder=1,
                )
            )


def draw_attack_influence_overlay(
    axes: Axes,
    analysis: MoveAnalysis,
):
    matrix = np.array(
        analysis.attack_influence_field.matrix,
        dtype=float,
    )

    max_absolute_value = max(
        1.0,
        float(np.max(np.abs(matrix))),
    )

    normalization = Normalize(
        vmin=-max_absolute_value,
        vmax=max_absolute_value,
    )

    return axes.imshow(
        matrix,
        extent=(0, 8, 8, 0),
        cmap="RdBu_r",
        norm=normalization,
        alpha=0.72,
        interpolation="nearest",
        zorder=2,
    )


def draw_pieces(
    axes: Axes,
    board: chess.Board,
) -> None:
    for square, piece in board.piece_map().items():
        x, y = square_to_plot_coords(square)

        symbol = UNICODE_PIECES[piece.symbol()]

        axes.text(
            x + 0.5,
            y + 0.5,
            symbol,
            fontsize=32,
            ha="center",
            va="center",
            color="black",
            zorder=5,
        )


def draw_strongest_squares(
    axes: Axes,
    analysis: MoveAnalysis,
) -> None:
    attack_influence = analysis.attack_influence_field

    if attack_influence.strongest_white_square is not None:
        draw_square_marker(
            axes=axes,
            square_name=attack_influence.strongest_white_square,
            marker="o",
            facecolor="#d73027",
        )

    if attack_influence.strongest_black_square is not None:
        draw_square_marker(
            axes=axes,
            square_name=attack_influence.strongest_black_square,
            marker="X",
            facecolor="#2166ac",
        )


def draw_square_marker(
    axes: Axes,
    square_name: str,
    marker: str,
    facecolor: str,
) -> None:
    square = chess.parse_square(square_name)

    x, y = square_to_plot_coords(square)

    axes.scatter(
        x + 0.5,
        y + 0.5,
        s=260,
        marker=marker,
        facecolor=facecolor,
        edgecolor="black",
        linewidth=1.8,
        alpha=0.95,
        zorder=8,
    )


def configure_axes(
    axes: Axes,
    analysis: MoveAnalysis,
) -> None:
    attack_influence = analysis.attack_influence_field

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

    axes.set_title(
        "VectorChess Attack Influence Field\n"
        f"Move: {analysis.move} | "
        f"White attack influence: "
        f"{attack_influence.total_white_attack_influence:.2f} | "
        f"Black attack influence: "
        f"{attack_influence.total_black_attack_influence:.2f}\n"
        f"Balance: {attack_influence.balance:+.2f} | "
        f"White peak: "
        f"{attack_influence.strongest_white_square} | "
        f"Black peak: "
        f"{attack_influence.strongest_black_square}",
        pad=16,
    )