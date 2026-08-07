import copy

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from scipy.interpolate import RectBivariateSpline

from analysis.critical_points import classify_critical_points, locate_critical_points
from analysis.ridge_valley import assess_ridge_valley_quality, locate_ridge_valley_chains
from chess_engine.analyzer import analyze_position
from chess_engine.board import ChessGame
from chess_engine.models import (
    AttackInfluenceSurface,
    RidgeValleyChain,
    RidgeValleyPoint,
    RidgeValleyQualityAssessment,
)
from visualization.critical_points_plot import (
    draw_critical_points,
    draw_critical_points_legend,
)
from visualization.equipotential_plot import (
    draw_attack_influence_surface,
    draw_board_background,
    draw_equipotential_lines,
    draw_pieces,
)
from visualization.ridge_valley_plot import (
    RIDGE_LINE_COLOR,
    VALLEY_LINE_COLOR,
    REJECTED_LINE_COLOR,
    draw_ridge_valley_chains,
    draw_ridge_valley_legend,
    plot_ridge_valley,
)


DOMAIN_MIN = 0.5
DOMAIN_MAX = 7.5


def new_axes():
    return plt.subplots()


def make_ridge_valley_point(
    x: float, y: float, kind: str, eigenvalue_cross: float = -1.0
) -> RidgeValleyPoint:
    return RidgeValleyPoint(
        x=x,
        y=y,
        value=0.0,
        eigenvalue_cross=eigenvalue_cross,
        alignment_residual=0.0,
        kind=kind,
    )


def make_assessment(
    kind: str,
    coordinates: list[tuple[float, float]],
    is_accepted: bool = True,
    rejection_reasons: list[str] | None = None,
) -> RidgeValleyQualityAssessment:
    points = [make_ridge_valley_point(x, y, kind) for x, y in coordinates]
    chain = RidgeValleyChain(
        kind=kind,
        points=points,
        anchor=None,
        start_status="left_domain",
        end_status="left_domain",
    )
    return RidgeValleyQualityAssessment(
        chain=chain,
        is_accepted=is_accepted,
        rejection_reasons=rejection_reasons or [],
    )


def real_position_analysis(moves: list[str]):
    game = ChessGame()
    move_details = None
    for move in moves:
        move_details = game.make_move(move)
    analysis = analyze_position(game.board, move_details)
    return game, analysis


def surface_from_function(
    func, node_count: int = 61, resolution: int = 81
) -> AttackInfluenceSurface:
    """Same construction as tests/test_ridge_valley.py's helper."""

    row_coords = np.linspace(DOMAIN_MIN, DOMAIN_MAX, node_count)
    column_coords = np.linspace(DOMAIN_MIN, DOMAIN_MAX, node_count)

    matrix = np.array(
        [[func(column, row) for column in column_coords] for row in row_coords]
    )

    spline = RectBivariateSpline(row_coords, column_coords, matrix, kx=3, ky=3)

    row_fine = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
    column_fine = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)

    z = spline(row_fine, column_fine)

    return AttackInfluenceSurface(
        x=column_fine, y=row_fine, z=z, resolution=resolution, spline=spline
    )


def anisotropic_gaussian_bump(
    cx: float, cy: float, amplitude: float = 10.0, sigma_x: float = 2.0, sigma_y: float = 0.5
):
    def value(x, y):
        return amplitude * np.exp(
            -((x - cx) ** 2 / (2 * sigma_x**2) + (y - cy) ** 2 / (2 * sigma_y**2))
        )

    return value


# ---------------------------------------------------------
# Accepted ridge / valley chains rendered with distinct colors
# ---------------------------------------------------------


def test_accepted_ridge_chain_is_rendered_with_ridge_color():
    assessment = make_assessment("ridge", [(2.0, 2.0), (3.0, 2.0), (4.0, 2.0)])

    figure, axes = new_axes()
    draw_ridge_valley_chains(axes, [assessment])

    lines = axes.get_lines()
    assert len(lines) == 1

    line = lines[0]
    assert to_rgba(line.get_color()) == pytest.approx(to_rgba(RIDGE_LINE_COLOR))
    assert line.get_linestyle() == "-"
    assert list(line.get_xdata()) == pytest.approx([2.0, 3.0, 4.0])
    assert list(line.get_ydata()) == pytest.approx([2.0, 2.0, 2.0])

    plt.close(figure)


def test_accepted_valley_chain_is_rendered_with_valley_color():
    assessment = make_assessment("valley", [(2.0, 5.0), (3.0, 5.0), (4.0, 5.0)])

    figure, axes = new_axes()
    draw_ridge_valley_chains(axes, [assessment])

    lines = axes.get_lines()
    assert len(lines) == 1
    assert to_rgba(lines[0].get_color()) == pytest.approx(to_rgba(VALLEY_LINE_COLOR))

    plt.close(figure)


def test_ridge_and_valley_colors_are_distinct_from_each_other_and_from_rejected():
    colors = {RIDGE_LINE_COLOR, VALLEY_LINE_COLOR, REJECTED_LINE_COLOR}
    assert len(colors) == 3


# ---------------------------------------------------------
# Rejected chains: omitted by default, visible in debug mode
# ---------------------------------------------------------


def test_rejected_chains_omitted_by_default():
    accepted = make_assessment("ridge", [(2.0, 2.0), (3.0, 2.0)], is_accepted=True)
    rejected = make_assessment(
        "valley", [(5.0, 5.0), (6.0, 5.0)], is_accepted=False,
        rejection_reasons=["chain too short"],
    )

    figure, axes = new_axes()
    draw_ridge_valley_chains(axes, [accepted, rejected], show_only_accepted=True)

    lines = axes.get_lines()
    assert len(lines) == 1
    assert to_rgba(lines[0].get_color()) == pytest.approx(to_rgba(RIDGE_LINE_COLOR))

    plt.close(figure)


def test_rejected_chains_shown_when_debug_flag_is_set():
    accepted = make_assessment("ridge", [(2.0, 2.0), (3.0, 2.0)], is_accepted=True)
    rejected = make_assessment(
        "valley", [(5.0, 5.0), (6.0, 5.0)], is_accepted=False,
        rejection_reasons=["chain too short"],
    )

    figure, axes = new_axes()
    draw_ridge_valley_chains(axes, [accepted, rejected], show_only_accepted=False)

    lines = axes.get_lines()
    assert len(lines) == 2

    rejected_lines = [
        line for line in lines
        if to_rgba(line.get_color()) == pytest.approx(to_rgba(REJECTED_LINE_COLOR))
    ]
    assert len(rejected_lines) == 1
    assert rejected_lines[0].get_linestyle() == "--"
    assert rejected_lines[0].get_alpha() < 0.5

    plt.close(figure)


def test_rejected_chain_keeps_gray_regardless_of_its_own_kind():
    # A rejected chain's color signals "rejected", not "ridge"/"valley"
    # -- mixing the two messages into one color would blur both.
    rejected_ridge = make_assessment(
        "ridge", [(2.0, 2.0), (3.0, 2.0)], is_accepted=False, rejection_reasons=["x"],
    )

    figure, axes = new_axes()
    draw_ridge_valley_chains(axes, [rejected_ridge], show_only_accepted=False)

    lines = axes.get_lines()
    assert len(lines) == 1
    assert to_rgba(lines[0].get_color()) == pytest.approx(to_rgba(REJECTED_LINE_COLOR))
    assert to_rgba(lines[0].get_color()) != pytest.approx(to_rgba(RIDGE_LINE_COLOR))

    plt.close(figure)


# ---------------------------------------------------------
# Off-grid continuous coordinates preserved
# ---------------------------------------------------------


def test_exact_continuous_coordinates_preserved():
    coordinates = [(2.1347, 3.8821), (2.6821, 4.1234), (3.0456, 4.4009)]
    assessment = make_assessment("ridge", coordinates)

    figure, axes = new_axes()
    draw_ridge_valley_chains(axes, [assessment])

    line = axes.get_lines()[0]
    assert list(line.get_xdata()) == pytest.approx([c[0] for c in coordinates], abs=1e-9)
    assert list(line.get_ydata()) == pytest.approx([c[1] for c in coordinates], abs=1e-9)

    plt.close(figure)


# ---------------------------------------------------------
# Empty / degenerate chain lists
# ---------------------------------------------------------


def test_empty_chain_list_renders_without_error():
    figure, axes = new_axes()
    draw_ridge_valley_chains(axes, [])

    assert axes.get_lines() == []

    plt.close(figure)


def test_single_point_chain_is_not_drawn_as_a_degenerate_line():
    # A chain that terminated immediately in both directions (e.g. an
    # isotropic anchor) has only its anchor point -- not enough to
    # draw a line, and must not raise or produce a zero-length mark.
    assessment = make_assessment("ridge", [(4.0, 4.0)])

    figure, axes = new_axes()
    draw_ridge_valley_chains(axes, [assessment])

    assert axes.get_lines() == []

    plt.close(figure)


# ---------------------------------------------------------
# Legend correctness
# ---------------------------------------------------------


def test_legend_default_shows_ridge_and_valley_only():
    figure, axes = new_axes()
    draw_ridge_valley_legend(axes)

    legend = axes.get_legend()
    assert legend is not None

    labels = {text.get_text() for text in legend.get_texts()}
    assert labels == {"Ridge", "Valley"}

    plt.close(figure)


def test_legend_with_show_rejected_includes_rejected_entry():
    figure, axes = new_axes()
    draw_ridge_valley_legend(axes, show_rejected=True)

    labels = {text.get_text() for text in axes.get_legend().get_texts()}
    assert labels == {"Ridge", "Valley", "Rejected (debug)"}

    plt.close(figure)


def test_legend_is_independent_of_which_chains_are_actually_plotted():
    figure, axes = new_axes()
    draw_ridge_valley_chains(axes, [make_assessment("ridge", [(2.0, 2.0), (3.0, 2.0)])])
    draw_ridge_valley_legend(axes)

    labels = {text.get_text() for text in axes.get_legend().get_texts()}
    assert labels == {"Ridge", "Valley"}

    plt.close(figure)


def test_critical_point_and_ridge_valley_legends_coexist_on_plot_ridge_valley():
    game, analysis = real_position_analysis(["e2e4"])

    plot_ridge_valley(game.board, analysis)

    figure = plt.gcf()
    axes = figure.axes[0]

    all_texts = {
        text.get_text()
        for legend in (axes.get_legend(),) if legend is not None
        for text in legend.get_texts()
    }
    for artist in axes.artists:
        if hasattr(artist, "get_texts"):
            all_texts |= {text.get_text() for text in artist.get_texts()}

    assert {"Ridge", "Valley"}.issubset(all_texts)
    assert {"Maximum", "Minimum", "Saddle", "Degenerate"}.issubset(all_texts)

    plt.close("all")


# ---------------------------------------------------------
# No mutation
# ---------------------------------------------------------


def test_draw_ridge_valley_chains_does_not_mutate_assessments():
    assessments = [
        make_assessment("ridge", [(2.0, 2.0), (3.0, 2.0)]),
        make_assessment("valley", [(5.0, 5.0), (6.0, 5.0)], is_accepted=False,
                         rejection_reasons=["chain too short"]),
    ]
    assessments_before = copy.deepcopy(assessments)

    figure, axes = new_axes()
    draw_ridge_valley_chains(axes, assessments, show_only_accepted=False)

    assert assessments == assessments_before

    plt.close(figure)


def test_plot_ridge_valley_does_not_mutate_board_analysis_or_computed_chains():
    game, analysis = real_position_analysis(
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]
    )

    fen_before = game.board.fen()
    attack_influence_before = copy.deepcopy(analysis.attack_influence_field)

    from analysis.attack_influence_surface import build_attack_influence_surface

    surface = build_attack_influence_surface(analysis.attack_influence_field)
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)
    ridge_chains_before = copy.deepcopy(
        locate_ridge_valley_chains(surface, classified, kind="ridge")
    )

    plot_ridge_valley(game.board, analysis)

    assert game.board.fen() == fen_before
    assert analysis.attack_influence_field == attack_influence_before

    ridge_chains_after = locate_ridge_valley_chains(surface, classified, kind="ridge")
    assert ridge_chains_after == ridge_chains_before

    plt.close("all")


# ---------------------------------------------------------
# Headless rendering
# ---------------------------------------------------------


def test_headless_rendering_succeeds_for_a_real_position():
    game, analysis = real_position_analysis(["e2e4"])

    plot_ridge_valley(game.board, analysis)  # must not raise under Agg

    assert plt.get_fignums()

    plt.close("all")


def test_headless_rendering_succeeds_for_the_synthetic_ridge_valley_surface():
    surface = surface_from_function(
        anisotropic_gaussian_bump(4.0, 4.0, amplitude=10.0, sigma_x=2.0, sigma_y=0.5)
    )
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)

    ridge_chains = locate_ridge_valley_chains(surface, classified, kind="ridge")
    valley_chains = locate_ridge_valley_chains(surface, classified, kind="valley")

    ridge_assessments = assess_ridge_valley_quality(ridge_chains, surface)
    valley_assessments = assess_ridge_valley_quality(valley_chains, surface)

    figure, axes = new_axes()
    draw_board_background(axes)
    draw_ridge_valley_chains(axes, ridge_assessments + valley_assessments)
    draw_ridge_valley_legend(axes)

    assert plt.get_fignums()
    assert len(axes.get_lines()) >= 1

    plt.close(figure)


# ---------------------------------------------------------
# Existing critical-point / equipotential helpers unchanged
# ---------------------------------------------------------


def test_existing_critical_point_and_equipotential_helpers_remain_unchanged():
    from analysis.attack_influence_surface import build_attack_influence_surface

    game, analysis = real_position_analysis(["e2e4"])
    surface = build_attack_influence_surface(analysis.attack_influence_field)

    figure, axes = new_axes()

    draw_board_background(axes)
    assert len(axes.patches) == 64

    max_absolute_value = max(1.0, float(np.max(np.abs(surface.z))))
    levels = np.linspace(-max_absolute_value, max_absolute_value, 13)

    filled_contours = draw_attack_influence_surface(
        axes=axes, surface=surface, levels=levels, max_absolute_value=max_absolute_value,
    )
    assert filled_contours is not None

    draw_equipotential_lines(axes=axes, surface=surface, levels=levels)

    text_count_before_pieces = len(axes.texts)
    draw_pieces(axes=axes, board=game.board)
    assert len(axes.texts) - text_count_before_pieces == len(game.board.piece_map())

    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)
    draw_critical_points(axes=axes, classified_points=classified)
    draw_critical_points_legend(axes=axes)

    legend = axes.get_legend()
    assert {text.get_text() for text in legend.get_texts()} == {
        "Maximum", "Minimum", "Saddle", "Degenerate",
    }

    plt.close(figure)
