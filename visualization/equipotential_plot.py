import chess
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from analysis.attack_influence_surface import build_attack_influence_surface
from analysis.geometry import square_to_plot_coords
from chess_engine.models import AttackInfluenceSurface, MoveAnalysis


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


ATTACK_INFLUENCE_ALPHA = 0.72

CONTOUR_LEVEL_COUNT = 13

CONTOUR_LINE_COLOR = "#202124"
CONTOUR_LINE_WIDTH = 0.7
CONTOUR_LINE_ALPHA = 0.55


def plot_equipotential_field(
    board: chess.Board,
    analysis: MoveAnalysis,
) -> None:
    """
    Рисува еквипотенциалните линии на Attack Influence Field.

    Всяка линия свързва точки с еднакво атаково влияние — същият
    принцип като изобарите на карта на атмосферното налягане.
    Полето е математически консервативно (градиент на скаларен
    потенциал), затова еквипотенциалните линии, а не flow paths,
    са коректната визуализация за него.
    """

    figure, axes = plt.subplots(figsize=(10, 10))

    surface = build_attack_influence_surface(
        analysis.attack_influence_field
    )

    max_absolute_value = max(
        1.0,
        float(np.max(np.abs(surface.z))),
    )

    levels = np.linspace(
        -max_absolute_value,
        max_absolute_value,
        CONTOUR_LEVEL_COUNT,
    )

    draw_board_background(axes)

    filled_contours = draw_attack_influence_surface(
        axes=axes,
        surface=surface,
        levels=levels,
        max_absolute_value=max_absolute_value,
    )

    draw_equipotential_lines(
        axes=axes,
        surface=surface,
        levels=levels,
    )

    draw_pieces(
        axes=axes,
        board=board,
    )

    configure_axes(
        axes=axes,
        analysis=analysis,
    )

    colorbar = figure.colorbar(
        filled_contours,
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

            square_color = (
                "#f0d9b5"
                if is_light_square
                else "#b58863"
            )

            axes.add_patch(
                plt.Rectangle(
                    (column, row),
                    1,
                    1,
                    facecolor=square_color,
                    edgecolor="none",
                    zorder=1,
                )
            )


def draw_attack_influence_surface(
    axes: Axes,
    surface: AttackInfluenceSurface,
    levels: np.ndarray,
    max_absolute_value: float,
):
    normalization = Normalize(
        vmin=-max_absolute_value,
        vmax=max_absolute_value,
    )

    return axes.contourf(
        surface.x,
        surface.y,
        surface.z,
        levels=levels,
        cmap="RdBu_r",
        norm=normalization,
        alpha=ATTACK_INFLUENCE_ALPHA,
        zorder=2,
    )


def draw_equipotential_lines(
    axes: Axes,
    surface: AttackInfluenceSurface,
    levels: np.ndarray,
) -> None:
    contour_set = axes.contour(
        surface.x,
        surface.y,
        surface.z,
        levels=levels,
        colors=CONTOUR_LINE_COLOR,
        linewidths=CONTOUR_LINE_WIDTH,
        alpha=CONTOUR_LINE_ALPHA,
        zorder=3,
    )

    axes.clabel(
        contour_set,
        contour_set.levels[::2],
        fontsize=7,
        fmt="%.1f",
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
            fontsize=31,
            ha="center",
            va="center",
            color="black",
            zorder=5,
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
        "VectorChess Equipotential Field\n"
        f"Move: {analysis.move} | "
        f"Attack influence balance: {attack_influence.balance:+.2f}\n"
        f"White peak: {attack_influence.strongest_white_square} | "
        f"Black peak: {attack_influence.strongest_black_square}",
        pad=16,
    )
