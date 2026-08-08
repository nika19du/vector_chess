import copy

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon
from scipy.interpolate import RectBivariateSpline

from analysis.attack_influence import build_attack_influence_field
from analysis.attack_influence_surface import build_attack_influence_surface
from analysis.critical_point_quality import assess_critical_point_quality
from analysis.critical_points import classify_critical_points, locate_critical_points
from analysis.morse_smale import (
    assemble_morse_smale_cells,
    assess_morse_smale_cell_quality,
    compute_cell_geometry,
    locate_morse_smale_separatrices,
)
from chess_engine.analyzer import analyze_position
from chess_engine.board import ChessGame
from chess_engine.models import (
    AttackInfluenceSurface,
    ClassifiedCriticalPoint,
    MorseSmaleCell,
    MorseSmaleCellQualityAssessment,
    MorseSmaleComplex,
    SeparatrixPath,
)
import visualization.critical_points_plot as critical_points_plot_module
import visualization.equipotential_plot as equipotential_plot_module
import visualization.morse_smale_plot as morse_smale_plot_module
from visualization.morse_smale_plot import (
    ACCEPTED_CELL_FILL_COLOR,
    ACCEPTED_CELL_LINE_COLOR,
    OPEN_CELL_LINE_COLOR,
    REJECTED_CELL_LINE_COLOR,
    draw_morse_smale_cells,
    draw_morse_smale_legend,
    plot_morse_smale_cells,
)


DOMAIN_MIN = 0.5
DOMAIN_MAX = 7.5


# ---------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------


def new_axes():
    return plt.subplots()


def make_point(x: float, y: float, classification: str) -> ClassifiedCriticalPoint:
    return ClassifiedCriticalPoint(
        x=x, y=y, value=0.0, gradient_norm=0.0, status="converged", iterations=1,
        f_xx=-1.0, f_xy=0.0, f_yx=0.0, f_yy=-1.0,
        eigenvalue_min=-1.0, eigenvalue_max=1.0,
        classification=classification,
    )


def zero_length_separatrix(
    start: ClassifiedCriticalPoint,
    end: ClassifiedCriticalPoint,
    flow_direction: str = "ascending",
) -> SeparatrixPath:
    return SeparatrixPath(
        start_saddle=start, flow_direction=flow_direction, points=[],
        end_critical_point=end, termination_status="reached_critical_point",
        step_count=0, path_length=0.0,
    )


def make_quad_cell(cell_id: int = 0) -> MorseSmaleCell:
    saddle_a = make_point(1.75, 1.75, "saddle")
    saddle_b = make_point(3.25, 1.75, "saddle")
    maximum = make_point(2.5, 2.5, "maximum")
    minimum = make_point(2.5, 1.0, "minimum")

    sep_a_max = zero_length_separatrix(saddle_a, maximum, "ascending")
    sep_b_max = zero_length_separatrix(saddle_b, maximum, "ascending")
    sep_b_min = zero_length_separatrix(saddle_b, minimum, "descending")
    sep_a_min = zero_length_separatrix(saddle_a, minimum, "descending")

    return MorseSmaleCell(
        cell_id=cell_id,
        boundary_separatrices=[sep_a_max, sep_b_max, sep_b_min, sep_a_min],
        boundary_traversal_directions=[True, False, True, False],
        boundary_critical_points=[maximum, saddle_b, minimum, saddle_a],
        is_closed=True,
        open_boundaries=[],
    )


def make_open_cell(cell_id: int = 0) -> MorseSmaleCell:
    saddle = make_point(2.5, 4.0, "saddle")
    maximum = make_point(3.5, 4.0, "maximum")
    sep = zero_length_separatrix(saddle, maximum, "ascending")

    blocker = SeparatrixPath(
        start_saddle=saddle, flow_direction="descending", points=[],
        end_critical_point=None, termination_status="left_domain",
        step_count=0, path_length=0.0,
    )

    return MorseSmaleCell(
        cell_id=cell_id,
        boundary_separatrices=[sep],
        boundary_traversal_directions=[True],
        boundary_critical_points=[maximum],
        is_closed=False,
        open_boundaries=[blocker],
    )


def make_assessment(
    cell: MorseSmaleCell, is_accepted: bool
) -> MorseSmaleCellQualityAssessment:
    return MorseSmaleCellQualityAssessment(
        cell=cell, is_accepted=is_accepted,
        rejection_reasons=[] if is_accepted else [],
    )


def surface_from_function(func, node_count: int = 81, resolution: int = 121) -> AttackInfluenceSurface:
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


def egg_crate(x0: float, y0: float, amplitude: float = 10.0, period: float = 1.5):
    def value(x, y):
        return amplitude * np.cos((x - x0) * (np.pi / period)) * np.cos(
            (y - y0) * (np.pi / period)
        )

    return value


def real_position_pipeline(moves: list[str]):
    game = ChessGame()
    move_details = None
    for move in moves:
        move_details = game.make_move(move)
    analysis = analyze_position(game.board, move_details)

    surface = build_attack_influence_surface(analysis.attack_influence_field)
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)
    assessments = assess_critical_point_quality(classified, surface)
    separatrices = locate_morse_smale_separatrices(surface, assessments)
    complex_ = assemble_morse_smale_cells(separatrices)
    cell_assessments = assess_morse_smale_cell_quality(complex_.cells, complex_.topology_issues)

    return game, analysis, surface, complex_, cell_assessments


def polygon_patches(axes):
    return [patch for patch in axes.patches if isinstance(patch, Polygon)]


# ---------------------------------------------------------
# Accepted cell rendered
# ---------------------------------------------------------


def test_accepted_cell_is_filled_and_outlined():
    cell = make_quad_cell()
    assessment = make_assessment(cell, is_accepted=True)

    figure, axes = new_axes()
    draw_morse_smale_cells(axes, [cell], [assessment])

    fills = polygon_patches(axes)
    assert len(fills) == 1
    assert np.allclose(fills[0].get_facecolor(), to_rgba(ACCEPTED_CELL_FILL_COLOR, alpha=fills[0].get_facecolor()[3]))

    lines = axes.get_lines()
    assert len(lines) == 1
    assert to_rgba(lines[0].get_color()) == pytest.approx(to_rgba(ACCEPTED_CELL_LINE_COLOR))
    assert lines[0].get_linestyle() == "-"

    plt.close(figure)


# ---------------------------------------------------------
# Rejected cell: hidden by default, visible in debug mode
# ---------------------------------------------------------


def test_rejected_closed_cell_hidden_by_default():
    cell = make_quad_cell()
    assessment = make_assessment(cell, is_accepted=False)

    figure, axes = new_axes()
    draw_morse_smale_cells(axes, [cell], [assessment], show_only_accepted=True)

    assert polygon_patches(axes) == []
    assert axes.get_lines() == []

    plt.close(figure)


def test_rejected_closed_cell_visible_in_debug_mode():
    cell = make_quad_cell()
    assessment = make_assessment(cell, is_accepted=False)

    figure, axes = new_axes()
    draw_morse_smale_cells(axes, [cell], [assessment], show_only_accepted=False)

    # Rejected cells are never filled -- only accepted ones are.
    assert polygon_patches(axes) == []

    lines = axes.get_lines()
    assert len(lines) == 1
    assert to_rgba(lines[0].get_color()) == pytest.approx(to_rgba(REJECTED_CELL_LINE_COLOR))
    assert lines[0].get_linestyle() == "--"
    assert lines[0].get_alpha() < 1.0

    plt.close(figure)


# ---------------------------------------------------------
# Open cells: no polygon fill, distinct dotted line, debug-only
# ---------------------------------------------------------


def test_open_cell_hidden_by_default():
    cell = make_open_cell()
    assessment = make_assessment(cell, is_accepted=False)

    figure, axes = new_axes()
    draw_morse_smale_cells(axes, [cell], [assessment], show_only_accepted=True)

    assert polygon_patches(axes) == []
    assert axes.get_lines() == []

    plt.close(figure)


def test_open_cell_drawn_without_fill_in_debug_mode():
    cell = make_open_cell()
    assessment = make_assessment(cell, is_accepted=False)

    figure, axes = new_axes()
    draw_morse_smale_cells(axes, [cell], [assessment], show_only_accepted=False)

    # No valid polygon exists for an open cell -- must never be filled.
    assert polygon_patches(axes) == []
    assert compute_cell_geometry(cell) is None

    lines = axes.get_lines()
    assert len(lines) == 1
    assert to_rgba(lines[0].get_color()) == pytest.approx(to_rgba(OPEN_CELL_LINE_COLOR))
    assert lines[0].get_linestyle() == ":"

    plt.close(figure)


def test_open_and_rejected_cells_use_different_linestyles():
    # Same gray color family ("not fully trustworthy"), but distinct
    # dash patterns -- "open" (incomplete topology) must remain
    # visually distinguishable from "rejected" (complete topology,
    # low quality).
    quad = make_quad_cell(cell_id=0)
    quad_assessment = make_assessment(quad, is_accepted=False)
    open_cell = make_open_cell(cell_id=1)
    open_assessment = make_assessment(open_cell, is_accepted=False)

    figure, axes = new_axes()
    draw_morse_smale_cells(
        axes, [quad, open_cell], [quad_assessment, open_assessment],
        show_only_accepted=False,
    )

    linestyles = {line.get_linestyle() for line in axes.get_lines()}
    assert linestyles == {"--", ":"}

    plt.close(figure)


# ---------------------------------------------------------
# Exact continuous polygon coordinates preserved
# ---------------------------------------------------------


def test_exact_continuous_polygon_coordinates_preserved():
    cell = make_quad_cell()
    assessment = make_assessment(cell, is_accepted=True)
    expected_polygon = compute_cell_geometry(cell).polygon

    figure, axes = new_axes()
    draw_morse_smale_cells(axes, [cell], [assessment])

    line = axes.get_lines()[0]
    assert list(line.get_xdata()) == pytest.approx(
        [p[0] for p in expected_polygon], abs=1e-9
    )
    assert list(line.get_ydata()) == pytest.approx(
        [p[1] for p in expected_polygon], abs=1e-9
    )

    plt.close(figure)


# ---------------------------------------------------------
# Critical points preserved
# ---------------------------------------------------------


def test_critical_points_are_drawn_from_complex_vertices():
    from matplotlib.collections import PathCollection

    game, analysis, surface, complex_, cell_assessments = real_position_pipeline(["e2e4"])

    plot_morse_smale_cells(game.board, analysis, surface, complex_, cell_assessments)

    figure = plt.gcf()
    axes = figure.axes[0]
    collections = [c for c in axes.collections if isinstance(c, PathCollection)]
    total_plotted = sum(len(c.get_offsets()) for c in collections)

    expected_classified = sum(
        1 for point in complex_.vertices if point.classification != "unclassified"
    )
    assert total_plotted == expected_classified
    assert total_plotted > 0

    plt.close("all")


# ---------------------------------------------------------
# Empty complex renders cleanly
# ---------------------------------------------------------


def test_empty_complex_renders_without_error():
    game, analysis, surface, _complex, _assessments = real_position_pipeline(["e2e4"])

    empty_complex = MorseSmaleComplex(
        vertices=[], edges=[], cells=[], unassigned_separatrices=[], topology_issues=[]
    )

    plot_morse_smale_cells(game.board, analysis, surface, empty_complex, [])

    figure = plt.gcf()
    axes = figure.axes[0]
    assert polygon_patches(axes) == []
    assert axes.get_legend() is not None

    plt.close("all")


# ---------------------------------------------------------
# Legend correctness
# ---------------------------------------------------------


def test_legend_default_shows_only_accepted_entry():
    figure, axes = new_axes()
    draw_morse_smale_legend(axes)

    labels = {text.get_text() for text in axes.get_legend().get_texts()}
    assert labels == {"Accepted cell"}

    plt.close(figure)


def test_legend_debug_mode_shows_all_three_entries():
    figure, axes = new_axes()
    draw_morse_smale_legend(axes, show_debug=True)

    labels = {text.get_text() for text in axes.get_legend().get_texts()}
    assert labels == {"Accepted cell", "Rejected cell", "Open cell"}

    plt.close(figure)


def test_legend_is_independent_of_which_cells_are_actually_plotted():
    figure, axes = new_axes()
    draw_morse_smale_cells(axes, [], [])
    draw_morse_smale_legend(axes)

    labels = {text.get_text() for text in axes.get_legend().get_texts()}
    assert labels == {"Accepted cell"}

    plt.close(figure)


# ---------------------------------------------------------
# No mutation
# ---------------------------------------------------------


def test_draw_morse_smale_cells_does_not_mutate_inputs():
    cell = make_quad_cell()
    assessment = make_assessment(cell, is_accepted=True)
    cells_before = copy.deepcopy([cell])
    assessments_before = copy.deepcopy([assessment])

    figure, axes = new_axes()
    draw_morse_smale_cells(axes, [cell], [assessment])

    assert cell.is_closed == cells_before[0].is_closed
    assert len(cell.boundary_separatrices) == len(cells_before[0].boundary_separatrices)
    assert assessment.is_accepted == assessments_before[0].is_accepted

    plt.close(figure)


def test_plot_morse_smale_cells_does_not_mutate_board_analysis_or_complex():
    game, analysis, surface, complex_, cell_assessments = real_position_pipeline(
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]
    )

    fen_before = game.board.fen()
    attack_influence_before = copy.deepcopy(analysis.attack_influence_field)
    complex_cells_before = copy.deepcopy(complex_.cells)

    plot_morse_smale_cells(game.board, analysis, surface, complex_, cell_assessments)

    assert game.board.fen() == fen_before
    assert analysis.attack_influence_field == attack_influence_before
    assert len(complex_.cells) == len(complex_cells_before)
    for cell_after, cell_before in zip(complex_.cells, complex_cells_before):
        assert cell_after.is_closed == cell_before.is_closed
        assert len(cell_after.boundary_separatrices) == len(cell_before.boundary_separatrices)

    plt.close("all")


# ---------------------------------------------------------
# Headless rendering
# ---------------------------------------------------------


def test_headless_rendering_succeeds_for_a_real_position():
    game, analysis, surface, complex_, cell_assessments = real_position_pipeline(["e2e4"])

    plot_morse_smale_cells(game.board, analysis, surface, complex_, cell_assessments)

    assert plt.get_fignums()

    plt.close("all")


def test_headless_rendering_succeeds_for_the_egg_crate_surface():
    surface = surface_from_function(egg_crate(4.0, 4.0))
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)
    assessments = assess_critical_point_quality(classified, surface)
    separatrices = locate_morse_smale_separatrices(surface, assessments)
    complex_ = assemble_morse_smale_cells(separatrices)
    cell_assessments = assess_morse_smale_cell_quality(complex_.cells, complex_.topology_issues)

    figure, axes = new_axes()
    from visualization.equipotential_plot import draw_board_background

    draw_board_background(axes)
    draw_morse_smale_cells(axes, complex_.cells, cell_assessments)

    assert plt.get_fignums()
    assert len(polygon_patches(axes)) == sum(1 for a in cell_assessments if a.is_accepted)

    plt.close(figure)


# ---------------------------------------------------------
# Backward-compatible helper reuse (genuine import, not
# reimplementation under the same name)
# ---------------------------------------------------------


def test_reuses_existing_helpers_by_identity_not_reimplementation():
    assert (
        morse_smale_plot_module.draw_board_background
        is equipotential_plot_module.draw_board_background
    )
    assert (
        morse_smale_plot_module.draw_attack_influence_surface
        is equipotential_plot_module.draw_attack_influence_surface
    )
    assert (
        morse_smale_plot_module.draw_equipotential_lines
        is equipotential_plot_module.draw_equipotential_lines
    )
    assert morse_smale_plot_module.draw_pieces is equipotential_plot_module.draw_pieces
    assert (
        morse_smale_plot_module.draw_critical_points
        is critical_points_plot_module.draw_critical_points
    )
    assert (
        morse_smale_plot_module.draw_critical_points_legend
        is critical_points_plot_module.draw_critical_points_legend
    )


def test_plot_morse_smale_cells_actually_invokes_the_reused_helpers(monkeypatch):
    game, analysis, surface, complex_, cell_assessments = real_position_pipeline(["e2e4"])

    call_counts = {"board": 0, "pieces": 0, "critical_points": 0}

    original_board = morse_smale_plot_module.draw_board_background
    original_pieces = morse_smale_plot_module.draw_pieces
    original_critical_points = morse_smale_plot_module.draw_critical_points

    def counting_board(*args, **kwargs):
        call_counts["board"] += 1
        return original_board(*args, **kwargs)

    def counting_pieces(*args, **kwargs):
        call_counts["pieces"] += 1
        return original_pieces(*args, **kwargs)

    def counting_critical_points(*args, **kwargs):
        call_counts["critical_points"] += 1
        return original_critical_points(*args, **kwargs)

    monkeypatch.setattr(morse_smale_plot_module, "draw_board_background", counting_board)
    monkeypatch.setattr(morse_smale_plot_module, "draw_pieces", counting_pieces)
    monkeypatch.setattr(
        morse_smale_plot_module, "draw_critical_points", counting_critical_points
    )

    plot_morse_smale_cells(game.board, analysis, surface, complex_, cell_assessments)

    assert call_counts == {"board": 1, "pieces": 1, "critical_points": 1}

    plt.close("all")
