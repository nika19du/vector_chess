import copy

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba

from analysis.attack_influence_surface import build_attack_influence_surface
from analysis.critical_point_quality import assess_critical_point_quality
from analysis.critical_points import classify_critical_points, locate_critical_points
from chess_engine.analyzer import analyze_position
from chess_engine.board import ChessGame
from chess_engine.models import ClassifiedCriticalPoint
from visualization.critical_points_plot import (
    CLASSIFICATION_ORDER,
    MARKER_SPECS,
    draw_critical_points,
    draw_critical_points_legend,
    plot_critical_points,
)
from visualization.equipotential_plot import (
    draw_attack_influence_surface,
    draw_board_background,
    draw_equipotential_lines,
    draw_pieces,
)


def make_point(x: float, y: float, classification: str, **overrides) -> ClassifiedCriticalPoint:
    fields = dict(
        x=x,
        y=y,
        value=1.0,
        gradient_norm=1e-9,
        status="converged",
        iterations=2,
        f_xx=1.0,
        f_xy=0.0,
        f_yx=0.0,
        f_yy=1.0,
        eigenvalue_min=1.0,
        eigenvalue_max=1.0,
        classification=classification,
    )
    fields.update(overrides)
    return ClassifiedCriticalPoint(**fields)


def new_axes():
    return plt.subplots()


def path_collections(axes):
    """
    Isolates scatter-created markers from contourf/contour artists on
    the same axes -- in this matplotlib version (3.11), contourf and
    contour do not register as PathCollection instances in
    axes.collections, only ax.scatter(...) calls do, so this filter
    reliably finds only what draw_critical_points added, regardless
    of what base layer was already drawn underneath.
    """

    return [c for c in axes.collections if isinstance(c, PathCollection)]


def facecolor_matches(collection, spec_facecolor) -> bool:
    actual = collection.get_facecolor()
    if spec_facecolor == "none":
        return len(actual) == 0
    return len(actual) == 1 and np.allclose(actual[0], to_rgba(spec_facecolor))


def edgecolor_matches(collection, spec_edgecolor) -> bool:
    actual = collection.get_edgecolor()
    return len(actual) == 1 and np.allclose(actual[0], to_rgba(spec_edgecolor))


def real_position_analysis(moves: list[str]):
    game = ChessGame()
    move_details = None
    for move in moves:
        move_details = game.make_move(move)
    analysis = analyze_position(game.board, move_details)
    return game, analysis


# ---------------------------------------------------------
# One marker per classified point, correct marker type
# ---------------------------------------------------------


def test_one_marker_per_classified_point_with_correct_specs():
    points = [
        make_point(2.0, 2.0, "maximum"),
        make_point(4.0, 4.0, "minimum"),
        make_point(6.0, 6.0, "saddle"),
        make_point(1.0, 6.0, "degenerate"),
    ]

    figure, axes = new_axes()
    draw_critical_points(axes, points)

    collections = path_collections(axes)
    assert len(collections) == len(CLASSIFICATION_ORDER)

    total_plotted = sum(len(collection.get_offsets()) for collection in collections)
    assert total_plotted == len(points)

    for classification in CLASSIFICATION_ORDER:
        spec = MARKER_SPECS[classification]
        expected_point = next(p for p in points if p.classification == classification)

        matches = [
            collection
            for collection in collections
            if facecolor_matches(collection, spec["facecolor"])
            and edgecolor_matches(collection, spec["edgecolor"])
        ]
        assert len(matches) == 1

        offsets = matches[0].get_offsets()
        assert len(offsets) == 1
        assert offsets[0][0] == pytest.approx(expected_point.x)
        assert offsets[0][1] == pytest.approx(expected_point.y)

    plt.close(figure)


def test_multiple_points_of_same_type_share_one_collection():
    points = [
        make_point(2.0, 2.0, "maximum"),
        make_point(5.0, 3.0, "maximum"),
        make_point(6.5, 6.5, "maximum"),
    ]

    figure, axes = new_axes()
    draw_critical_points(axes, points)

    collections = path_collections(axes)
    assert len(collections) == 1
    assert len(collections[0].get_offsets()) == 3

    plt.close(figure)


# ---------------------------------------------------------
# Off-grid coordinates preserved
# ---------------------------------------------------------


def test_off_grid_coordinates_preserved_exactly():
    point = make_point(3.2743, 5.8821, "maximum")

    figure, axes = new_axes()
    draw_critical_points(axes, [point])

    offsets = path_collections(axes)[0].get_offsets()
    assert offsets[0][0] == pytest.approx(3.2743, abs=1e-9)
    assert offsets[0][1] == pytest.approx(5.8821, abs=1e-9)

    plt.close(figure)


# ---------------------------------------------------------
# Unclassified points omitted
# ---------------------------------------------------------


def test_unclassified_points_are_omitted():
    points = [
        make_point(2.0, 2.0, "maximum"),
        make_point(
            5.0,
            5.0,
            "unclassified",
            status="left_domain",
            f_xx=None,
            f_xy=None,
            f_yx=None,
            f_yy=None,
            eigenvalue_min=None,
            eigenvalue_max=None,
        ),
    ]

    figure, axes = new_axes()
    draw_critical_points(axes, points)

    collections = path_collections(axes)
    total_plotted = sum(len(collection.get_offsets()) for collection in collections)
    assert total_plotted == 1

    for collection in collections:
        for x, y in collection.get_offsets():
            assert (float(x), float(y)) != pytest.approx((5.0, 5.0))

    plt.close(figure)


# ---------------------------------------------------------
# Degenerate points displayed distinctly
# ---------------------------------------------------------


def test_degenerate_marker_is_a_distinct_hollow_circle():
    figure, axes = new_axes()
    draw_critical_points(axes, [make_point(4.0, 4.0, "degenerate")])

    collections = path_collections(axes)
    assert len(collections) == 1
    assert len(collections[0].get_facecolor()) == 0  # hollow -- no fill at all

    assert MARKER_SPECS["degenerate"]["marker"] == "o"
    for other_classification in ("maximum", "minimum", "saddle"):
        assert MARKER_SPECS[other_classification]["facecolor"] != "none"
        assert MARKER_SPECS[other_classification]["marker"] != MARKER_SPECS["degenerate"]["marker"]

    plt.close(figure)


# ---------------------------------------------------------
# Empty critical-point list
# ---------------------------------------------------------


def test_empty_critical_point_list_renders_without_error():
    figure, axes = new_axes()
    draw_critical_points(axes, [])

    assert path_collections(axes) == []

    plt.close(figure)


# ---------------------------------------------------------
# Legend
# ---------------------------------------------------------


def test_legend_contains_exactly_the_supported_classifications():
    figure, axes = new_axes()
    draw_critical_points_legend(axes)

    legend = axes.get_legend()
    assert legend is not None

    labels = {text.get_text() for text in legend.get_texts()}
    assert labels == {"Maximum", "Minimum", "Saddle", "Degenerate"}
    assert len(legend.get_texts()) == 4

    plt.close(figure)


def test_legend_is_independent_of_which_points_are_actually_plotted():
    figure, axes = new_axes()
    draw_critical_points(axes, [make_point(4.0, 4.0, "maximum")])
    draw_critical_points_legend(axes)

    labels = {text.get_text() for text in axes.get_legend().get_texts()}
    assert labels == {"Maximum", "Minimum", "Saddle", "Degenerate"}

    plt.close(figure)


# ---------------------------------------------------------
# No mutation
# ---------------------------------------------------------


def test_draw_critical_points_does_not_mutate_input_list():
    points = [make_point(2.0, 2.0, "maximum"), make_point(4.0, 4.0, "minimum")]
    points_before = copy.deepcopy(points)

    figure, axes = new_axes()
    draw_critical_points(axes, points)

    assert points == points_before

    plt.close(figure)


def test_plot_critical_points_does_not_mutate_board_or_analysis():
    game, analysis = real_position_analysis(
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]
    )

    fen_before = game.board.fen()
    attack_influence_before = copy.deepcopy(analysis.attack_influence_field)

    plot_critical_points(game.board, analysis)

    assert game.board.fen() == fen_before
    assert analysis.attack_influence_field == attack_influence_before

    plt.close("all")


# ---------------------------------------------------------
# Headless rendering
# ---------------------------------------------------------


def test_headless_rendering_succeeds_for_a_real_position():
    game, analysis = real_position_analysis(["e2e4"])

    plot_critical_points(game.board, analysis)  # must not raise under Agg

    assert plt.get_fignums()

    plt.close("all")


# ---------------------------------------------------------
# show_only_accepted (Phase 4a quality filtering) wiring
# ---------------------------------------------------------


def test_show_only_accepted_default_draws_strictly_fewer_or_equal_points():
    game, analysis = real_position_analysis(
        ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"]
    )

    surface = build_attack_influence_surface(analysis.attack_influence_field)
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)
    assessments = assess_critical_point_quality(classified, surface)

    converged_count = sum(1 for point in classified if point.status == "converged")
    accepted_count = sum(1 for assessment in assessments if assessment.is_accepted)

    # This specific, previously-inspected position is known (Phase 4
    # smoke render) to contain at least one converged point close
    # enough to the board edge to be rejected by the default quality
    # filter -- so filtered must be strictly fewer, not just <=.
    assert accepted_count < converged_count

    figure_filtered, axes_filtered = new_axes()
    draw_critical_points(axes_filtered, [a.point for a in assessments if a.is_accepted])
    filtered_total = sum(
        len(collection.get_offsets()) for collection in path_collections(axes_filtered)
    )
    plt.close(figure_filtered)

    figure_unfiltered, axes_unfiltered = new_axes()
    draw_critical_points(axes_unfiltered, classified)
    unfiltered_total = sum(
        len(collection.get_offsets()) for collection in path_collections(axes_unfiltered)
    )
    plt.close(figure_unfiltered)

    assert filtered_total == accepted_count
    assert unfiltered_total == converged_count
    assert filtered_total < unfiltered_total


def test_plot_critical_points_show_only_accepted_true_is_the_default():
    import inspect

    signature = inspect.signature(plot_critical_points)
    assert signature.parameters["show_only_accepted"].default is True


# ---------------------------------------------------------
# Existing equipotential helpers remain unchanged (regression baseline)
# ---------------------------------------------------------


def test_existing_equipotential_helpers_remain_unchanged():
    from analysis.attack_influence_surface import build_attack_influence_surface

    game, analysis = real_position_analysis(["e2e4"])
    surface = build_attack_influence_surface(analysis.attack_influence_field)

    figure, axes = new_axes()

    draw_board_background(axes)
    assert len(axes.patches) == 64

    max_absolute_value = max(1.0, float(np.max(np.abs(surface.z))))
    levels = np.linspace(-max_absolute_value, max_absolute_value, 13)

    filled_contours = draw_attack_influence_surface(
        axes=axes,
        surface=surface,
        levels=levels,
        max_absolute_value=max_absolute_value,
    )
    assert filled_contours is not None

    draw_equipotential_lines(axes=axes, surface=surface, levels=levels)

    # draw_equipotential_lines' own clabel(...) call already added
    # contour-label Text artists -- compare the DELTA draw_pieces
    # contributes, not the running total.
    text_count_before_pieces = len(axes.texts)

    draw_pieces(axes=axes, board=game.board)

    assert len(axes.texts) - text_count_before_pieces == len(game.board.piece_map())

    plt.close(figure)
