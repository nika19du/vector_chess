import matplotlib.pyplot as plt
import numpy as np

from chess_engine.models import MoveAnalysis


def plot_position_analysis(
    analysis: MoveAnalysis,
) -> None:
    """
    Визуализира:
    - control heatmap;
    - центъра на белия контрол;
    - центъра на черния контрол;
    - глобалния вектор между двата центъра.
    """

    heatmap = np.array(analysis.heatmap)

    figure, axes = plt.subplots(figsize=(8, 8))

    image = axes.imshow(
        heatmap,
        origin="upper",
        extent=(-0.5, 7.5, -0.5, 7.5),
    )

    figure.colorbar(
        image,
        ax=axes,
        label="White control - Black control",
    )

    axes.set_xticks(range(8))
    axes.set_xticklabels(list("abcdefgh"))

    axes.set_yticks(range(8))
    axes.set_yticklabels(range(8, 0, -1))

    axes.set_xlabel("File")
    axes.set_ylabel("Rank")

    axes.set_title(
        f"VectorChess control field after {analysis.move}"
    )

    # Показваме числената стойност във всяка клетка.
    for matrix_row in range(8):
        for matrix_column in range(8):
            value = heatmap[matrix_row][matrix_column]

            axes.text(
                matrix_column,
                matrix_row,
                str(value),
                ha="center",
                va="center",
            )

    if analysis.white_center is not None:
        white_x, white_y = analysis.white_center

        # Heatmap матрицата е обърната вертикално:
        # ред 8 е на индекс 0, а ред 1 е на индекс 7.
        plotted_white_y = 7 - white_y

        axes.scatter(
            white_x,
            plotted_white_y,
            s=180,
            marker="o",
            label="White control center",
        )

    if analysis.black_center is not None:
        black_x, black_y = analysis.black_center
        plotted_black_y = 7 - black_y

        axes.scatter(
            black_x,
            plotted_black_y,
            s=180,
            marker="X",
            label="Black control center",
        )

    if analysis.control_vector is not None:
        vector = analysis.control_vector

        start_x = vector.start_x
        start_y = 7 - vector.start_y

        end_x = vector.end_x
        end_y = 7 - vector.end_y

        axes.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 2,
            },
        )

    axes.set_xlim(-0.5, 7.5)
    axes.set_ylim(7.5, -0.5)
    axes.set_aspect("equal")

    axes.grid(
        visible=True,
        linewidth=0.5,
    )

    axes.legend()
    figure.tight_layout()

    plt.show()