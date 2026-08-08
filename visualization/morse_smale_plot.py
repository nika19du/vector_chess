import chess
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis.morse_smale import compute_cell_geometry
from chess_engine.models import (
    AttackInfluenceSurface,
    MorseSmaleCell,
    MorseSmaleCellQualityAssessment,
    MorseSmaleComplex,
    MoveAnalysis,
)
from visualization.critical_points_plot import (
    draw_critical_points,
    draw_critical_points_legend,
)
from visualization.equipotential_plot import (
    CONTOUR_LEVEL_COUNT,
    draw_attack_influence_surface,
    draw_board_background,
    draw_equipotential_lines,
    draw_pieces,
)


# Приета (затворена И качествено одобрена) клетка -- плътен тъмнозелен
# контур + полупрозрачна запълнена площ. Умишлено различен от ЦЯЛАТА
# досегашна палитра (RdBu_r фон, ridge оранжево #e65100, valley тъмно
# синьо #0d47a1, критични точки злато/светло синьо/тъмнотюркоазен-x/
# лилаво) -- критичните точки се рисуват на СЪЩИЯ график (виж
# plot_morse_smale_cells), затова тъмнотюркоазеният saddle маркер
# (#00695c) умишлено НЕ се преизползва тук, за да не се бърка
# границата на клетка със самия saddle маркер.
ACCEPTED_CELL_FILL_COLOR = "#2e7d32"
ACCEPTED_CELL_LINE_COLOR = "#1b5e20"
ACCEPTED_CELL_FILL_ALPHA = 0.22
ACCEPTED_CELL_LINEWIDTH = 2.2

# Отхвърлени (но затворени) вериги -- само в debug режим
# (show_only_accepted=False). Сиво пунктирано -- същия принцип като
# REJECTED_LINE_COLOR в ridge_valley_plot.py: връщащ се потребител
# вече знае "сиво пунктирано = отхвърлено" от другите графики.
REJECTED_CELL_LINE_COLOR = "#757575"
REJECTED_CELL_LINEWIDTH = 1.6
REJECTED_CELL_ALPHA = 0.5

# Отворени (непълни, is_closed=False) клетки -- също само в debug
# режим. Същия сив цвят като отхвърлените (все още "не напълно
# надежден резултат"), но РАЗЛИЧЕН щрих (точковиден, не пунктиран) --
# "отворено" (непълна топология -- обходът на границата не се е
# затворил) е качествено ДРУГО състояние от "отхвърлено" (топологията
# Е пълна, просто не достатъчно качествена), затова цветът остава
# споделен, но линията умишлено се различава.
OPEN_CELL_LINE_COLOR = "#757575"
OPEN_CELL_LINEWIDTH = 1.2
OPEN_CELL_ALPHA = 0.4

# Над equipotential линиите (zorder=3), под фигурите (zorder=5) --
# същия принцип като RIDGE_VALLEY_ZORDER в ridge_valley_plot.py.
CELL_ZORDER = 4

# По-малко от 2 точки няма смислена линия за начертаване (matplotlib
# изисква поне 2 точки за отсечка).
MIN_POINTS_TO_DRAW = 2

CELL_LEGEND_LOC = "lower right"


def plot_morse_smale_cells(
    board: chess.Board,
    analysis: MoveAnalysis,
    surface: AttackInfluenceSurface,
    morse_smale_complex: MorseSmaleComplex,
    cell_assessments: list[MorseSmaleCellQualityAssessment],
    show_only_accepted: bool = True,
) -> None:
    """
    Рисува Attack Influence Surface + eквипотенциални линии като база
    (същите helper-и като plot_equipotential_field/plot_critical_points
    /plot_ridge_valley от equipotential_plot.py), после overlay-ва:

    1. 2-клетките на Morse-Smale комплекса (analysis/morse_smale.py,
       Фаза 2/3) -- границите им СА сепаратрисите, визуално различни
       от тънките черни equipotential линии (различен цвят и по-
       дебела линия, виж CELL_ZORDER/*_LINEWIDTH).
    2. Критичните точки (morse_smale_complex.vertices -- точно
       множеството критични точки, засегнати от поне една
       сепаратриса), overlay-нати чрез същия draw_critical_points,
       който вече използват critical_points_plot.py/ridge_valley_plot
       .py -- нищо не се преоткрива тук.

    За разлика от plot_critical_points/plot_ridge_valley (Фаза 0),
    тази функция НЕ приема опционални предварително изчислени
    резултати с fallback към преизчисляване -- surface,
    morse_smale_complex и cell_assessments са ЗАДЪЛЖИТЕЛНИ. Целият
    Morse-Smale pipeline (surface -> критични точки -> качество ->
    сепаратриси -> сглобяване на клетки -> качество на клетките) е
    твърде скъп, за да се преизчислява мълчаливо вътре в renderer-а --
    визуализацията само рисува вече изчисленото, никога не преизчислява
    топология.

    show_only_accepted (по подразбиране True):
        контролира кое СЕ ПОКАЗВА, не кое се изчислява:
        - True  -- само приетите (is_closed=True И is_accepted=True)
                   клетки.
        - False -- добавя и отхвърлените затворени клетки (сиво
                   пунктирано) и отворените/непълните клетки (сиво
                   точковидно, само тяхната ЧАСТИЧНА граница -- виж
                   draw_morse_smale_cells) за debug инспекция.

    Визуализацията не създава и не преизчислява математически данни:
    локализацията, класификацията, трасирането, сглобяването на
    клетки и оценките за качество идват изцяло от analysis/ -- тук
    само се рисува вече изчисленото.
    """

    figure, axes = plt.subplots(figsize=(10, 10))

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

    draw_morse_smale_cells(
        axes=axes,
        cells=morse_smale_complex.cells,
        cell_assessments=cell_assessments,
        show_only_accepted=show_only_accepted,
    )

    draw_critical_points(
        axes=axes,
        classified_points=morse_smale_complex.vertices,
    )

    # Две отделни легенди на едни оси -- същата matplotlib идиома,
    # която вече използва ridge_valley_plot.py спрямо
    # critical_points_plot.py: draw_critical_points_legend прави
    # axes.legend() свой активен обект -- превръщаме го в постоянен
    # artist (axes.add_artist), преди draw_morse_smale_legend да
    # направи следващия axes.legend() свой собствен активен обект.
    draw_critical_points_legend(axes=axes)
    critical_points_legend = axes.get_legend()
    if critical_points_legend is not None:
        axes.add_artist(critical_points_legend)

    draw_morse_smale_legend(
        axes=axes,
        show_debug=not show_only_accepted,
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


def _closed_cell_polygon_xy(
    cell: MorseSmaleCell,
) -> tuple[list[float], list[float]] | None:
    geometry = compute_cell_geometry(cell)

    if geometry is None:
        return None

    return (
        [point[0] for point in geometry.polygon],
        [point[1] for point in geometry.polygon],
    )


def _open_cell_partial_boundary_xy(
    cell: MorseSmaleCell,
) -> tuple[list[float], list[float]]:
    """
    Плътната, следваща кривите ЧАСТИЧНА граница на отворена клетка --
    същия принцип на сглобяване като
    analysis.morse_smale.compute_cell_geometry за затворени клетки
    (връх + собствените точки на всяка гранична сепаратриса, в реда
    на обхождането), само БЕЗ затваряне на цикъла -- за отворена
    клетка обходът не се е върнал в началото, затова няма какво да се
    затвори.

    Дублирано тук нарочно, не преизнесено в analysis/morse_smale.py:
    това е чисто форматиране за рисуване на вече изчислени координати
    (separatrix.points[i].x/y, vertex.x/y) -- визуализационен слой,
    не преизчисляване на топология или геометрия.
    """

    x_values: list[float] = []
    y_values: list[float] = []

    for separatrix, is_forward, vertex in zip(
        cell.boundary_separatrices,
        cell.boundary_traversal_directions,
        cell.boundary_critical_points,
    ):
        origin = separatrix.start_saddle if is_forward else separatrix.end_critical_point
        points = separatrix.points if is_forward else list(reversed(separatrix.points))

        if not x_values:
            x_values.append(origin.x)
            y_values.append(origin.y)

        x_values.extend(point.x for point in points)
        y_values.extend(point.y for point in points)

        x_values.append(vertex.x)
        y_values.append(vertex.y)

    return x_values, y_values


def draw_morse_smale_cells(
    axes: Axes,
    cells: list[MorseSmaleCell],
    cell_assessments: list[MorseSmaleCellQualityAssessment],
    show_only_accepted: bool = True,
) -> None:
    """
    Overlay-ва MorseSmaleCell-овете -- НЕ извиква сглобяването или
    оценката на качеството, само рисува вече изчисления резултат,
    точно както draw_critical_points/draw_ridge_valley_chains спрямо
    своите съответни analysis/ функции.

    Всяка клетка се съпоставя с MorseSmaleCellQualityAssessment по
    cell_id (не по позиция в списъка -- устойчиво на различен ред
    между cells и cell_assessments). Клетка без съответна оценка се
    третира като неприета (консервативно -- никога не се рисува като
    "приета" без изрично потвърждение).

    Затворени, приети клетки: плътна запълнена площ + плътен контур
    (ACCEPTED_CELL_*). Затворени, отхвърлени клетки: само контур,
    сиво пунктирано, единствено ако show_only_accepted=False.
    Отворени клетки: НЯМА запълване (compute_cell_geometry връща None
    -- полигон не е дефиниран без затворена граница) -- рисува се само
    ЧАСТИЧНАТА проследена граница, сиво точковидно, единствено ако
    show_only_accepted=False. Липсата на запълване на отворена клетка
    не е пропуск -- отсъствието на площ е самата точка: няма валиден
    затворен регион за запълване.
    """

    assessment_by_cell_id = {
        assessment.cell.cell_id: assessment for assessment in cell_assessments
    }

    for cell in cells:
        assessment = assessment_by_cell_id.get(cell.cell_id)
        is_accepted = assessment is not None and assessment.is_accepted

        if cell.is_closed:
            polygon_xy = _closed_cell_polygon_xy(cell)

            if polygon_xy is None:
                continue

            x_values, y_values = polygon_xy

            if len(x_values) < MIN_POINTS_TO_DRAW:
                continue

            if is_accepted:
                axes.fill(
                    x_values,
                    y_values,
                    facecolor=ACCEPTED_CELL_FILL_COLOR,
                    edgecolor="none",
                    alpha=ACCEPTED_CELL_FILL_ALPHA,
                    zorder=CELL_ZORDER,
                )
                axes.plot(
                    x_values,
                    y_values,
                    color=ACCEPTED_CELL_LINE_COLOR,
                    linewidth=ACCEPTED_CELL_LINEWIDTH,
                    linestyle="-",
                    zorder=CELL_ZORDER,
                )
            elif not show_only_accepted:
                axes.plot(
                    x_values,
                    y_values,
                    color=REJECTED_CELL_LINE_COLOR,
                    linewidth=REJECTED_CELL_LINEWIDTH,
                    linestyle="--",
                    alpha=REJECTED_CELL_ALPHA,
                    zorder=CELL_ZORDER,
                )
        else:
            if show_only_accepted:
                continue

            x_values, y_values = _open_cell_partial_boundary_xy(cell)

            if len(x_values) < MIN_POINTS_TO_DRAW:
                continue

            axes.plot(
                x_values,
                y_values,
                color=OPEN_CELL_LINE_COLOR,
                linewidth=OPEN_CELL_LINEWIDTH,
                linestyle=":",
                alpha=OPEN_CELL_ALPHA,
                zorder=CELL_ZORDER,
            )


def draw_morse_smale_legend(
    axes: Axes,
    show_debug: bool = False,
) -> None:
    """
    Компактна легенда за Morse-Smale клетките -- отделна от
    draw_critical_points_legend (различен ъгъл на осите, виж
    CELL_LEGEND_LOC), винаги включва "Accepted cell", и добавя
    "Rejected cell"/"Open cell" само когато show_debug=True -- същия
    принцип като draw_ridge_valley_legend(show_rejected=...):
    легендата отразява какво РЕАЛНО може да се появи на текущата
    рисунка, не статичен изчерпателен списък.
    """

    handles = [
        Patch(
            facecolor=ACCEPTED_CELL_FILL_COLOR,
            edgecolor=ACCEPTED_CELL_LINE_COLOR,
            alpha=ACCEPTED_CELL_FILL_ALPHA,
            label="Accepted cell",
        ),
    ]

    if show_debug:
        handles.append(
            Line2D(
                [0],
                [0],
                color=REJECTED_CELL_LINE_COLOR,
                linewidth=REJECTED_CELL_LINEWIDTH,
                linestyle="--",
                label="Rejected cell",
            )
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=OPEN_CELL_LINE_COLOR,
                linewidth=OPEN_CELL_LINEWIDTH,
                linestyle=":",
                label="Open cell",
            )
        )

    axes.legend(
        handles=handles,
        loc=CELL_LEGEND_LOC,
        fontsize=8,
        title="Morse-Smale cells",
        title_fontsize=8,
        framealpha=0.85,
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
        "VectorChess Morse-Smale Complex\n"
        f"Move: {analysis.move} | "
        f"Attack influence balance: {attack_influence.balance:+.2f}",
        pad=16,
    )
