import chess
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from analysis.attack_influence_surface import build_attack_influence_surface
from analysis.critical_point_quality import assess_critical_point_quality
from analysis.critical_points import classify_critical_points, locate_critical_points
from analysis.ridge_valley import assess_ridge_valley_quality, locate_ridge_valley_chains
from chess_engine.models import (
    AttackInfluenceSurface,
    ClassifiedCriticalPoint,
    CriticalPointQualityAssessment,
    MoveAnalysis,
    RidgeValleyQualityAssessment,
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


# Цветове за ridge/valley линиите -- умишлено различни както от
# MARKER_SPECS в critical_points_plot.py (злато/светло синьо/тъмно
# зелено/лилаво), така и от полупрозрачния RdBu_r contourf и тънките
# черни equipotential линии, за да не се бъркат вериги с контури.
# Топъл наситен оранжев за ridge (спътник на "White" интуицията),
# студен наситен тъмносин за valley (спътник на "Black" интуицията)
# -- същия принцип като White/Black разграничението в docs/audio.md.
RIDGE_LINE_COLOR = "#e65100"
VALLEY_LINE_COLOR = "#0d47a1"

# Отхвърлените (is_accepted=False) вериги, показани само при
# show_only_accepted=False -- пунктир + намален alpha, за да останат
# визуално еднозначно "провизорни", не просто по-тънка версия на
# приетите.
REJECTED_LINE_COLOR = "#757575"

RIDGE_VALLEY_LINEWIDTH = 3.0
REJECTED_LINEWIDTH = 2.0

ACCEPTED_ALPHA = 0.95
REJECTED_ALPHA = 0.35

# Над equipotential линиите (zorder=3), под фигурите (zorder=5) --
# веригите трябва да се виждат ясно над контурите, но не да скриват
# фигурите върху дъската.
RIDGE_VALLEY_ZORDER = 4

RIDGE_VALLEY_LEGEND_LOC = "lower left"

# Верига с по-малко от 2 точки няма смислена линия за начертаване
# (matplotlib изисква поне 2 точки за отсечка) -- такива вериги вече
# носят собствена rejection_reasons анотация от
# assess_ridge_valley_quality, тук просто се пропускат мълчаливо.
MIN_POINTS_TO_DRAW = 2


def plot_ridge_valley(
    board: chess.Board,
    analysis: MoveAnalysis,
    show_only_accepted: bool = True,
    surface: AttackInfluenceSurface | None = None,
    classified_points: list[ClassifiedCriticalPoint] | None = None,
    critical_point_assessments: list[CriticalPointQualityAssessment] | None = None,
    ridge_assessments: list[RidgeValleyQualityAssessment] | None = None,
    valley_assessments: list[RidgeValleyQualityAssessment] | None = None,
) -> None:
    """
    Рисува Attack Influence Surface + eквипотенциални линии като база
    (същите helper-и като plot_equipotential_field/plot_critical_points
    от equipotential_plot.py), после overlay-ва:

    1. Ridge/valley веригите (analysis/ridge_valley.py) -- трасирани
       ВИНАГИ от вече качествено-филтрираните (assess_critical_point_
       quality) критични точки като анкери, независимо от
       show_only_accepted -- това е математическото решение от
       Раздел 10 на docs/mathematics.md ("Section 9's already-
       quality-filtered critical points"), не визуален избор.
    2. Класифицираните критични точки (analysis/critical_points.py),
       overlay-нати чрез същия draw_critical_points, който вече
       използва critical_points_plot.py -- нищо не се преоткрива тук.

    Визуализацията не създава и не преизчислява математически данни:
    локализацията, класификацията, трасирането и оценките за качество
    идват изцяло от analysis/ -- тук само се рисува вече изчисленото.

    show_only_accepted (по подразбиране True):
        контролира кое СЕ ПОКАЗВА, не кое се изчислява:
        - True  -- само приетите (is_accepted=True) ridge/valley
                   вериги и само приетите критични точки.
        - False -- добавя и отхвърлените вериги (пунктир, намален
                   alpha -- виж REJECTED_*) и всички класифицирани
                   критични точки, независимо от качеството им, за
                   debug инспекция на какво точно е било отхвърлено и
                   защо (rejection_reasons остава достъпно на самите
                   обекти, дори когато не се изобразява текстово).

    surface / classified_points / critical_point_assessments /
    ridge_assessments / valley_assessments (по подразбиране None):
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

    if critical_point_assessments is None:
        critical_point_assessments = assess_critical_point_quality(
            classified_points, surface
        )
    accepted_critical_points = [
        assessment.point
        for assessment in critical_point_assessments
        if assessment.is_accepted
    ]

    if ridge_assessments is None:
        ridge_chains = locate_ridge_valley_chains(
            surface, accepted_critical_points, kind="ridge"
        )
        ridge_assessments = assess_ridge_valley_quality(ridge_chains, surface)

    if valley_assessments is None:
        valley_chains = locate_ridge_valley_chains(
            surface, accepted_critical_points, kind="valley"
        )
        valley_assessments = assess_ridge_valley_quality(valley_chains, surface)

    draw_ridge_valley_chains(
        axes=axes,
        assessments=ridge_assessments + valley_assessments,
        show_only_accepted=show_only_accepted,
    )

    critical_points_to_draw = (
        accepted_critical_points if show_only_accepted else classified_points
    )

    draw_critical_points(
        axes=axes,
        classified_points=critical_points_to_draw,
    )

    # Две отделни легенди на едни оси: draw_critical_points_legend
    # прави axes.legend() свой активен обект -- превръщаме го в
    # постоянен artist (axes.add_artist), преди draw_ridge_valley_legend
    # да направи следващия axes.legend() свой собствен активен обект.
    # Стандартната matplotlib идиома за няколко легенди на едни оси;
    # critical_points_plot.py не се променя по никакъв начин.
    draw_critical_points_legend(axes=axes)
    critical_points_legend = axes.get_legend()
    if critical_points_legend is not None:
        axes.add_artist(critical_points_legend)

    draw_ridge_valley_legend(
        axes=axes,
        show_rejected=not show_only_accepted,
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


def draw_ridge_valley_chains(
    axes: Axes,
    assessments: list[RidgeValleyQualityAssessment],
    show_only_accepted: bool = True,
) -> None:
    """
    Overlay-ва RidgeValleyChain-овете, обвити в
    RidgeValleyQualityAssessment (analysis.ridge_valley
    .assess_ridge_valley_quality) -- НЕ извиква трасирането или
    оценката на качеството, само рисува вече изчисления резултат,
    точно както draw_critical_points спрямо
    locate_critical_points/classify_critical_points.

    Използва x, y директно от RidgeValleyPoint -- същите истински
    непрекъснати координати, върнати от analysis.ridge_valley, без
    закръгляне или прилепване към клетка (същия принцип като
    draw_critical_points спрямо Newton-уточнените координати).

    Ridge вериги -- RIDGE_LINE_COLOR; valley -- VALLEY_LINE_COLOR
    (chain.kind). Отхвърлени (is_accepted=False) вериги се рисуват
    само ако show_only_accepted=False, тогава винаги с
    REJECTED_LINE_COLOR + пунктир + намален alpha -- визуално
    еднозначно "провизорно/отхвърлено", независимо дали са ridge или
    valley (цветовото кодиране ridge/valley важи само за приетите
    вериги; смесването на "тип" и "качество" в един цвят би
    объркало и двете съобщения).
    """

    for assessment in assessments:
        chain = assessment.chain

        if not assessment.is_accepted:
            if show_only_accepted:
                continue
            color = REJECTED_LINE_COLOR
            linestyle = "--"
            linewidth = REJECTED_LINEWIDTH
            alpha = REJECTED_ALPHA
        else:
            color = RIDGE_LINE_COLOR if chain.kind == "ridge" else VALLEY_LINE_COLOR
            linestyle = "-"
            linewidth = RIDGE_VALLEY_LINEWIDTH
            alpha = ACCEPTED_ALPHA

        if len(chain.points) < MIN_POINTS_TO_DRAW:
            continue

        axes.plot(
            [point.x for point in chain.points],
            [point.y for point in chain.points],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
            zorder=RIDGE_VALLEY_ZORDER,
        )


def draw_ridge_valley_legend(
    axes: Axes,
    show_rejected: bool = False,
) -> None:
    """
    Компактна легенда за Ridge/Valley -- отделна от
    draw_critical_points_legend (различен ъгъл на осите, виж
    RIDGE_VALLEY_LEGEND_LOC), винаги включва Ridge и Valley, и
    добавя трети запис "Rejected" само когато show_rejected=True --
    легендата отразява какво РЕАЛНО може да се появи на текущата
    рисунка (show_only_accepted управлява и двете), не статичен
    изчерпателен списък като CLASSIFICATION_ORDER в
    critical_points_plot.py, защото "отхвърлено" не е отделен
    математически тип, а състояние на качествената оценка.
    """

    handles = [
        Line2D(
            [0],
            [0],
            color=RIDGE_LINE_COLOR,
            linewidth=RIDGE_VALLEY_LINEWIDTH,
            linestyle="-",
            label="Ridge",
        ),
        Line2D(
            [0],
            [0],
            color=VALLEY_LINE_COLOR,
            linewidth=RIDGE_VALLEY_LINEWIDTH,
            linestyle="-",
            label="Valley",
        ),
    ]

    if show_rejected:
        handles.append(
            Line2D(
                [0],
                [0],
                color=REJECTED_LINE_COLOR,
                linewidth=REJECTED_LINEWIDTH,
                linestyle="--",
                label="Rejected (debug)",
            )
        )

    axes.legend(
        handles=handles,
        loc=RIDGE_VALLEY_LEGEND_LOC,
        fontsize=8,
        title="Ridge / Valley",
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
        "VectorChess Ridge / Valley\n"
        f"Move: {analysis.move} | "
        f"Attack influence balance: {attack_influence.balance:+.2f}",
        pad=16,
    )
