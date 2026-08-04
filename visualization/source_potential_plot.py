import chess
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from analysis.field_reconstruction import build_potential_surface
from analysis.source_field import build_source_field
from analysis.source_potential import build_source_potential_surface
from chess_engine.models import MoveAnalysis, SourcePotentialSurface
from visualization.equipotential_plot import (
    CONTOUR_LEVEL_COUNT,
    draw_board_background,
    draw_equipotential_lines,
    draw_pieces,
    draw_potential_surface,
)


SOURCE_CONTOUR_ALPHA = 0.72


def plot_field_comparison(
    board: chess.Board,
    analysis: MoveAnalysis,
    kernel: str = "gaussian",
    kernel_params: dict[str, float] | None = None,
) -> None:
    """
    Рисува Attack Influence Field и Source Potential Field
    един до друг, за да е видно, че това са ДВА отделни
    математически модела, а не един и същ, различно изгладен.

    Ляво:
        Attack Influence — атаково поле от analysis/potential_field.py
        (зависи от board.attackers(), блокиране, геометрия на ходовете).

    Дясно:
        Source Potential — Φ_source, генерирано директно от
        ρ (позиция + тежест на фигурите), без никаква връзка
        с атакова геометрия.

    Всеки панел използва собствена, независима нормализация на
    цвета — общите магнитуди на двете полета не са сравними
    директно, само пространствената им структура е.
    """

    figure, (left_axes, right_axes) = plt.subplots(
        1,
        2,
        figsize=(18, 9),
    )

    source_field = build_source_field(board)

    source_surface = build_source_potential_surface(
        source_field=source_field,
        kernel=kernel,
        kernel_params=kernel_params,
    )

    draw_attack_panel(
        axes=left_axes,
        board=board,
        analysis=analysis,
        figure=figure,
    )

    draw_source_panel(
        axes=right_axes,
        board=board,
        source_surface=source_surface,
        figure=figure,
    )

    figure.suptitle(
        f"Move: {analysis.move}  |  "
        "independent color scales per panel — "
        "compare shape/sign, not magnitude, across panels",
        fontsize=11,
    )

    figure.tight_layout()
    plt.show()


def draw_attack_panel(
    axes: Axes,
    board: chess.Board,
    analysis: MoveAnalysis,
    figure,
) -> None:
    surface = build_potential_surface(
        analysis.potential_field
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

    filled_contours = draw_potential_surface(
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

    configure_panel_axes(
        axes=axes,
        title=(
            "Attack Influence Field\n"
            "(depends on board.attackers, blocking, move geometry)"
        ),
    )

    colorbar = figure.colorbar(
        filled_contours,
        ax=axes,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Black attacks  ←  independent scale  →  White attacks"
    )


def draw_source_panel(
    axes: Axes,
    board: chess.Board,
    source_surface: SourcePotentialSurface,
    figure,
) -> None:
    draw_board_background(axes)

    filled_contours = draw_source_potential_overlay(
        axes=axes,
        surface=source_surface,
    )

    draw_source_equipotential_lines(
        axes=axes,
        surface=source_surface,
    )

    draw_pieces(
        axes=axes,
        board=board,
    )

    kernel_description = ", ".join(
        f"{name}={value:g}"
        for name, value in source_surface.kernel_params.items()
    )

    configure_panel_axes(
        axes=axes,
        title=(
            "Source Potential Field  Φ_source\n"
            f"(kernel={source_surface.kernel_name}, "
            f"{kernel_description}; occupancy only, "
            "no attack geometry)"
        ),
    )

    colorbar = figure.colorbar(
        filled_contours,
        ax=axes,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Black material  ←  independent scale  →  White material"
    )


def draw_source_potential_overlay(
    axes: Axes,
    surface: SourcePotentialSurface,
):
    max_absolute_value = max(
        1.0,
        float(np.max(np.abs(surface.z))),
    )

    normalization = Normalize(
        vmin=-max_absolute_value,
        vmax=max_absolute_value,
    )

    return axes.contourf(
        surface.x,
        surface.y,
        surface.z,
        levels=CONTOUR_LEVEL_COUNT,
        cmap="RdBu_r",
        norm=normalization,
        alpha=SOURCE_CONTOUR_ALPHA,
        zorder=2,
    )


def draw_source_equipotential_lines(
    axes: Axes,
    surface: SourcePotentialSurface,
) -> None:
    contour_set = axes.contour(
        surface.x,
        surface.y,
        surface.z,
        levels=CONTOUR_LEVEL_COUNT,
        colors="#202124",
        linewidths=0.7,
        alpha=0.55,
        zorder=3,
    )

    axes.clabel(
        contour_set,
        contour_set.levels[::2],
        fontsize=7,
        fmt="%.1f",
    )


def configure_panel_axes(
    axes: Axes,
    title: str,
) -> None:
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

    axes.set_title(title, pad=12, fontsize=11)
