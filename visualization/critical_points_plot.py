import math

import chess
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from analysis.attack_influence_surface import build_attack_influence_surface
from analysis.critical_point_quality import assess_critical_point_quality
from analysis.critical_points import classify_critical_points, locate_critical_points
from chess_engine.models import (
    AttackInfluenceSurface,
    ClassifiedCriticalPoint,
    CriticalPointQualityAssessment,
    MoveAnalysis,
)
from visualization.equipotential_plot import (
    CONTOUR_LEVEL_COUNT,
    draw_attack_influence_surface,
    draw_board_background,
    draw_equipotential_lines,
    draw_pieces,
)


# Реда, в който се рисуват и изброяват в легендата типовете --
# фиксиран и изчерпателен: точно тези 4 стойности, никога
# "unclassified" (виж draw_critical_points).
CLASSIFICATION_ORDER = ("maximum", "minimum", "saddle", "degenerate")

# Маркер за всеки тип по docs/mathematics.md, Раздел 9 -- формата и
# цветът трябва да остават еднакви във всички бъдещи визуализации
# (същият принцип като audio.md: "vocabulary only grows"). Плътен
# цвят + черен кант за maximum/minimum/saddle; куха окръжност за
# degenerate, за да изглежда видимо различна, а не просто "четвърти
# плътен маркер".
MARKER_SPECS: dict[str, dict] = {
    "maximum": {
        "marker": "^",
        "facecolor": "#fdd835",
        "edgecolor": "#202124",
        "label": "Maximum",
    },
    "minimum": {
        "marker": "v",
        "facecolor": "#42a5f5",
        "edgecolor": "#202124",
        "label": "Minimum",
    },
    "saddle": {
        "marker": "x",
        "facecolor": "#00695c",
        "edgecolor": "#00695c",
        "label": "Saddle",
    },
    "degenerate": {
        "marker": "o",
        "facecolor": "none",
        "edgecolor": "#8e24aa",
        "label": "Degenerate",
    },
}

# Над draw_pieces (zorder=5) в equipotential_plot.py -- маркерите за
# критични точки трябва да стоят видимо над фигурите, не под тях.
CRITICAL_POINT_ZORDER = 6

# Маркери без запълнена площ (само линии, напр. "x") нямат отделен
# edgecolor -- matplotlib го игнорира и издава предупреждение, ако
# им подадем facecolors/edgecolors поотделно. За тях се подава един
# общ цвят вместо това.
UNFILLED_LINE_MARKERS = {"x"}

MARKER_SIZE = 220
MARKER_LINEWIDTH = 1.8

LEGEND_MARKER_SIZE = 11

# Milestone C (VECTORCHESS_MODEL_V2_INTEGRATION_AUDIT.md; Experiment 007
# REPORT.md): matplotlib's `s=` is marker AREA in points^2, so an area
# scale of 1.35**2 gives the same LINEAR size increase (~35%) as the
# desktop canvas's PROMINENT_MARKER_SCALE in
# desktop_app/layers/critical_points_layer.py -- same visual cue, correct
# units for this renderer.
PROMINENT_MARKER_SIZE = round(MARKER_SIZE * 1.35**2)


def critical_point_prominence(point: ClassifiedCriticalPoint) -> float:
    """
    Experiment 007's own prominence measure, reused verbatim (see
    VECTORCHESS_MATHEMATICAL_MODEL_V2.md Sec. 8.3 and
    VECTORCHESS_MODEL_V2_INTEGRATION_AUDIT.md Sec. 9): combined curvature
    magnitude, hypot(eigenvalue_min, eigenvalue_max). Meaningful only for a
    converged, classified point (eigenvalue_min/max not None) -- every
    accepted point satisfies this by construction
    (analysis/critical_point_quality.py only accepts status=="converged"
    points, which always have both eigenvalues set).

    Not a new scoring function -- the exact formula
    experiments/geometric_move_prediction/experiment_007_reconstruction_
    stability/stability_metrics.py's `_curvature_strength` already used to
    rank critical points for that experiment's stability measurements.

    Every real, production-built accepted point has both eigenvalues set
    (see above) -- but this function is also handed hand-built points in
    tests and, defensively, any future caller that hasn't guaranteed that
    invariant, so a missing eigenvalue resolves to 0.0 (lowest possible
    prominence, never mistaken for "strongest") rather than raising.
    """

    if point.eigenvalue_min is None or point.eigenvalue_max is None:
        return 0.0
    return math.hypot(point.eigenvalue_min, point.eigenvalue_max)


def strongest_critical_point(
    points: list[ClassifiedCriticalPoint],
) -> ClassifiedCriticalPoint | None:
    """
    The single most prominent point among `points` (expected to already be
    one position's accepted set), by `critical_point_prominence`, highest
    first. None for an empty list.

    Deterministic tie-break: Python's `max()` returns the FIRST
    maximal-value element it encounters for an exact tie, never the last --
    so ties resolve by `points`' own existing order, which is already
    deterministic end to end (`locate_critical_points` sorts candidates by
    `(round(y,6), round(x,6))`, and `critical_point_assessments` preserves
    that order through classification and quality assessment). This
    function never reorders or re-sorts `points` itself.

    The ONE canonical ranking helper for this milestone -- both
    `desktop_app/layers/critical_points_layer.py` (GL canvas) and this
    module's own `draw_critical_points` (matplotlib) call this, so the
    definition of "strongest" is never duplicated or allowed to drift
    between the two renderers.
    """

    if not points:
        return None
    return max(points, key=critical_point_prominence)


def plot_critical_points(
    board: chess.Board,
    analysis: MoveAnalysis,
    show_only_accepted: bool = True,
    surface: AttackInfluenceSurface | None = None,
    classified_points: list[ClassifiedCriticalPoint] | None = None,
    assessments: list[CriticalPointQualityAssessment] | None = None,
) -> None:
    """
    Рисува Attack Influence Surface + eквипотенциални линии като база
    (същите helper-и като plot_equipotential_field от
    equipotential_plot.py), после overlay-ва класифицираните критични
    точки (maximum/minimum/saddle/degenerate) от
    analysis/critical_points.py.

    Визуализацията не създава информация: локализацията и
    класификацията идват изцяло от математическия слой -- тук само
    се рисуват вече изчислените резултати. "unclassified" кандидати
    (Newton не е сходил) никога не се показват, защото не са реални
    критични точки на повърхността, само неуспешни опити.

    show_only_accepted (по подразбиране True):
        филтрира допълнително и през
        analysis.critical_point_quality.assess_critical_point_quality,
        показвайки само точките, преминали прага за надеждност (виж
        docs/mathematics.md, Раздел 9, за защо не всяка критична
        точка на сплайна е шахматно значима). False връща предишното
        поведение -- показва всички класифицирани точки, независимо
        от качеството им.

    surface / classified_points / assessments (по подразбиране None):
        позволяват на вече изчислени резултати (напр. от
        console_app/main.py, което ги отпечатва в резюмето) да бъдат
        подадени директно, вместо тази функция да ги преизчислява
        наново -- чисто архитектурна оптимизация (Phase 0 hardening),
        математиката остава същата. Ако липсва даден параметър, се
        изчислява точно както досега.
    """

    figure, axes = plt.subplots(figsize=(10, 10))

    if surface is None:
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

    if classified_points is None:
        candidates = locate_critical_points(surface)
        classified_points = classify_critical_points(candidates, surface)

    # Milestone C: prominence tiering is defined over the ACCEPTED set only
    # (the same set Experiment 007 ranked) -- in the show_only_accepted=False
    # debug path, "strongest" would require computing assessments purely to
    # support that rare path, so it is left None there rather than forcing
    # extra computation in a debug-only branch (no points are hidden either
    # way -- only the size cue is skipped).
    strongest: ClassifiedCriticalPoint | None = None

    if show_only_accepted:
        if assessments is None:
            assessments = assess_critical_point_quality(classified_points, surface)
        points_to_draw = [
            assessment.point for assessment in assessments if assessment.is_accepted
        ]
        strongest = strongest_critical_point(points_to_draw)
    else:
        points_to_draw = classified_points

    draw_critical_points(
        axes=axes,
        classified_points=points_to_draw,
        strongest=strongest,
    )

    draw_critical_points_legend(axes=axes)

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


def draw_critical_points(
    axes: Axes,
    classified_points: list[ClassifiedCriticalPoint],
    strongest: ClassifiedCriticalPoint | None = None,
) -> None:
    """
    Overlay-ва САМО класифицирани критични точки
    (maximum/minimum/saddle/degenerate). "unclassified" точки никога
    не се рисуват -- CLASSIFICATION_ORDER изчерпателно изброява
    единствените 4 стойности, които тази функция изобразява.

    Използва x, y директно в екранни координати -- същата конвенция
    като draw_pieces/draw_attack_influence_surface -- затова
    непрекъснатите (извън-мрежови) позиции на Newton уточняването се
    рисуват точно там, където са намерени, без да се "прилепват"
    към клетка.

    strongest (Milestone C, по подразбиране None): ако е подаден и
    съвпада (по идентичност) с точка от classified_points, тази точка
    се рисува с PROMINENT_MARKER_SIZE вместо MARKER_SIZE -- същият
    визуален знак ("структурно най-силна в тази позиция"), който
    desktop_app.layers.critical_points_layer вече прилага през размера
    на GL маркера. Формата и цветът остават напълно непроменени; никакъв
    нов draw call не се въвежда -- `s=` вече приема масив, един за всяка
    точка в existing scatter извикването.
    """

    for classification in CLASSIFICATION_ORDER:
        points = [
            point
            for point in classified_points
            if point.classification == classification
        ]

        if not points:
            continue

        spec = MARKER_SPECS[classification]

        color_kwargs = (
            {"color": spec["facecolor"]}
            if spec["marker"] in UNFILLED_LINE_MARKERS
            else {"facecolors": spec["facecolor"], "edgecolors": spec["edgecolor"]}
        )

        sizes = [PROMINENT_MARKER_SIZE if point is strongest else MARKER_SIZE for point in points]

        axes.scatter(
            [point.x for point in points],
            [point.y for point in points],
            marker=spec["marker"],
            s=sizes,
            linewidths=MARKER_LINEWIDTH,
            zorder=CRITICAL_POINT_ZORDER,
            **color_kwargs,
        )


def draw_critical_points_legend(
    axes: Axes,
) -> None:
    """
    Компактна легенда с точно 4-те поддържани типа
    (CLASSIFICATION_ORDER), винаги -- независимо кои типове реално
    се появяват в текущата позиция, за да остане речникът на
    маркерите постоянен между отделните извиквания.
    """

    handles = [
        Line2D(
            [0],
            [0],
            marker=MARKER_SPECS[classification]["marker"],
            color="none",
            markerfacecolor=MARKER_SPECS[classification]["facecolor"],
            markeredgecolor=MARKER_SPECS[classification]["edgecolor"],
            markeredgewidth=MARKER_LINEWIDTH,
            markersize=LEGEND_MARKER_SIZE,
            linestyle="None",
            label=MARKER_SPECS[classification]["label"],
        )
        for classification in CLASSIFICATION_ORDER
    ]

    axes.legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
        title="Critical points",
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
        "VectorChess Critical Points\n"
        f"Move: {analysis.move} | "
        f"Attack influence balance: {attack_influence.balance:+.2f}",
        pad=16,
    )
