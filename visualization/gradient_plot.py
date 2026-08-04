import chess
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from chess_engine.models import GradientVector, MoveAnalysis


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


# ---------------------------------------------------------
# Visual configuration
# ---------------------------------------------------------

# Не показваме вектори, които са по-слаби от този
# процент спрямо най-силния градиент в позицията.
MIN_RELATIVE_STRENGTH = 0.16

# Абсолютен минимален праг. Полезен е при позиции,
# в които максималният градиент е много голям.
MIN_MAGNITUDE = 1.5

# При False не показваме градиент върху поле,
# на което има шахматна фигура.
SHOW_VECTORS_ON_PIECES = False

POTENTIAL_ALPHA = 0.68

# Минимална и максимална дължина на стрелката
# в единици от една шахматна клетка.
MIN_ARROW_LENGTH = 0.22
MAX_ARROW_LENGTH = 0.64


def plot_gradient_field(
    board: chess.Board,
    analysis: MoveAnalysis,
) -> None:
    """
    Рисува потенциалното поле и неговия градиент.

    Всяка стрелка сочи към посоката, в която
    потенциалът на Белите спрямо Черните
    нараства най-бързо.
    """

    figure, axes = plt.subplots(figsize=(10, 10))

    draw_board_background(axes)

    potential_image = draw_potential_overlay(
        axes=axes,
        analysis=analysis,
    )

    visible_vectors_count = draw_gradient_vectors(
        axes=axes,
        board=board,
        analysis=analysis,
    )

    draw_pieces(
        axes=axes,
        board=board,
    )

    configure_axes(
        axes=axes,
        analysis=analysis,
        visible_vectors_count=visible_vectors_count,
    )

    colorbar = figure.colorbar(
        potential_image,
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
    """Рисува класическата шахматна дъска."""

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


def draw_potential_overlay(
    axes: Axes,
    analysis: MoveAnalysis,
):
    """Рисува Potential Field като полупрозрачен фон."""

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
        alpha=POTENTIAL_ALPHA,
        interpolation="nearest",
        zorder=2,
    )


def draw_gradient_vectors(
    axes: Axes,
    board: chess.Board,
    analysis: MoveAnalysis,
) -> int:
    """
    Филтрира и рисува значимите градиентни вектори.

    Връща броя на реално показаните стрелки.
    """

    gradient_field = analysis.gradient_field
    max_magnitude = gradient_field.max_magnitude

    if max_magnitude <= 0:
        return 0

    visible_vectors = [
        vector
        for vector in gradient_field.vectors
        if should_draw_vector(
            board=board,
            vector=vector,
            max_magnitude=max_magnitude,
        )
    ]

    for vector in visible_vectors:
        draw_single_gradient_vector(
            axes=axes,
            analysis=analysis,
            vector=vector,
            max_magnitude=max_magnitude,
        )

    return len(visible_vectors)


def should_draw_vector(
    board: chess.Board,
    vector: GradientVector,
    max_magnitude: float,
) -> bool:
    """Определя дали градиентният вектор е достатъчно значим."""

    if vector.magnitude < MIN_MAGNITUDE:
        return False

    relative_strength = (
        vector.magnitude / max_magnitude
    )

    if relative_strength < MIN_RELATIVE_STRENGTH:
        return False

    if not SHOW_VECTORS_ON_PIECES:
        square = chess.parse_square(vector.square)

        if board.piece_at(square) is not None:
            return False

    return True


def draw_single_gradient_vector(
    axes: Axes,
    analysis: MoveAnalysis,
    vector: GradientVector,
    max_magnitude: float,
) -> None:
    """Рисува един нормализиран градиентен вектор."""

    if vector.magnitude <= 0:
        return

    center_x = vector.x + 0.5
    center_y = 7 - vector.y + 0.5

    unit_x = (
        vector.delta_x / vector.magnitude
    )

    # Математически Y расте нагоре,
    # но при графиката Y расте надолу.
    unit_y = -(
        vector.delta_y / vector.magnitude
    )

    relative_strength = (
        vector.magnitude / max_magnitude
    )

    arrow_length = (
        MIN_ARROW_LENGTH
        + relative_strength
        * (MAX_ARROW_LENGTH - MIN_ARROW_LENGTH)
    )

    # Малко допълнително ограничение,
    # за да не доближава краищата на клетката.
    arrow_length = min(arrow_length, 0.58)

    delta_x = unit_x * arrow_length
    delta_y = unit_y * arrow_length

    start_x = center_x - delta_x / 2
    start_y = center_y - delta_y / 2

    end_x = center_x + delta_x / 2
    end_y = center_y + delta_y / 2

    arrow_color = get_contrasting_arrow_color(
        analysis=analysis,
        vector=vector,
    )

    axes.annotate(
        "",
        xy=(end_x, end_y),
        xytext=(start_x, start_y),
        arrowprops={
            "arrowstyle": "-|>",
            "color": arrow_color,
            "linewidth": (
                1.2 + relative_strength * 2.2
            ),
            "alpha": (
                0.72 + relative_strength * 0.25
            ),
            "mutation_scale": (
                11 + relative_strength * 10
            ),
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=6,
    )


def get_contrasting_arrow_color(
    analysis: MoveAnalysis,
    vector: GradientVector,
) -> str:
    """
    Избира светла или тъмна стрелка според
    локалната стойност на Potential Field.
    """

    matrix_row = 7 - vector.y
    matrix_column = vector.x

    local_potential = (
        analysis.potential_field.matrix
        [matrix_row][matrix_column]
    )

    max_absolute_value = max(
        1.0,
        max(
            abs(value)
            for row in analysis.potential_field.matrix
            for value in row
        ),
    )

    relative_background_strength = (
        abs(local_potential)
        / max_absolute_value
    )

    # Върху силно наситен фон използваме бяло.
    if relative_background_strength >= 0.30:
        return "#ffffff"

    return "#202124"


def draw_pieces(
    axes: Axes,
    board: chess.Board,
) -> None:
    """Рисува шахматните фигури над останалите слоеве."""

    for square, piece in board.piece_map().items():
        x = chess.square_file(square)
        y = 7 - chess.square_rank(square)

        symbol = UNICODE_PIECES[piece.symbol()]

        axes.text(
            x + 0.5,
            y + 0.5,
            symbol,
            fontsize=31,
            ha="center",
            va="center",
            color="black",
            zorder=9,
        )


def configure_axes(
    axes: Axes,
    analysis: MoveAnalysis,
    visible_vectors_count: int,
) -> None:
    """Настройва координатите и заглавието."""

    potential = analysis.potential_field
    gradient = analysis.gradient_field

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
        "VectorChess Gradient Field\n"
        f"Move: {analysis.move} | "
        f"Potential balance: {potential.balance:+.2f}\n"
        f"Max gradient: {gradient.max_magnitude:.2f} | "
        f"Visible vectors: {visible_vectors_count}",
        pad=16,
    )