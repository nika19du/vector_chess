import chess
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

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


def plot_potential_field(
    board: chess.Board,
    analysis: MoveAnalysis,
) -> None:
    """
    Рисува претегленото потенциално поле.

    Червено:
        по-силен бял потенциал.

    Синьо:
        по-силен черен потенциал.

    Бяло:
        равновесие.
    """

    figure, axes = plt.subplots(figsize=(10, 10))

    draw_board_background(axes)

    image = draw_potential_overlay(
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
        "Black potential  ←  balance  →  White potential"
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


def draw_potential_overlay(
    axes: Axes,
    analysis: MoveAnalysis,
):
    matrix = np.array(
        analysis.potential_field.matrix,
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
        x = chess.square_file(square)
        y = 7 - chess.square_rank(square)

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
    potential = analysis.potential_field

    if potential.strongest_white_square is not None:
        draw_square_marker(
            axes=axes,
            square_name=potential.strongest_white_square,
            marker="o",
            facecolor="#d73027",
        )

    if potential.strongest_black_square is not None:
        draw_square_marker(
            axes=axes,
            square_name=potential.strongest_black_square,
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

    x = chess.square_file(square)
    y = 7 - chess.square_rank(square)

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
    potential = analysis.potential_field

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
        "VectorChess Potential Field\n"
        f"Move: {analysis.move} | "
        f"White potential: "
        f"{potential.total_white_potential:.2f} | "
        f"Black potential: "
        f"{potential.total_black_potential:.2f}\n"
        f"Balance: {potential.balance:+.2f} | "
        f"White peak: "
        f"{potential.strongest_white_square} | "
        f"Black peak: "
        f"{potential.strongest_black_square}",
        pad=16,
    )