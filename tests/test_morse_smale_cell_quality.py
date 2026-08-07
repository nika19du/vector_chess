import copy

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from analysis.critical_point_quality import assess_critical_point_quality
from analysis.critical_points import classify_critical_points, locate_critical_points
from analysis.morse_smale import (
    MIN_CELL_AREA,
    MIN_CELL_PERIMETER,
    assemble_morse_smale_cells,
    assess_morse_smale_cell_quality,
    compute_cell_geometry,
    locate_morse_smale_separatrices,
)
from chess_engine.models import (
    AttackInfluenceSurface,
    ClassifiedCriticalPoint,
    MorseSmaleCell,
    MorseSmaleCellRejectionReasonKind,
    SeparatrixPath,
    SeparatrixPoint,
    TopologyIssueKind,
)


DOMAIN_MIN = 0.5
DOMAIN_MAX = 7.5


# ---------------------------------------------------------
# Shared helpers (surface construction mirrors
# tests/test_morse_smale_cells.py)
# ---------------------------------------------------------


def surface_from_function(
    func, node_count: int = 81, resolution: int = 121
) -> AttackInfluenceSurface:
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


def two_bumps_along_x(
    cx1: float, cx2: float, cy: float, amplitude: float = 10.0, sigma: float = 1.2
):
    def value(x, y):
        return amplitude * np.exp(
            -((x - cx1) ** 2 + (y - cy) ** 2) / (2 * sigma**2)
        ) + amplitude * np.exp(-((x - cx2) ** 2 + (y - cy) ** 2) / (2 * sigma**2))

    return value


def assemble_from_surface(surface: AttackInfluenceSurface):
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)
    assessments = assess_critical_point_quality(classified, surface)
    separatrices = locate_morse_smale_separatrices(surface, assessments)
    return separatrices, assemble_morse_smale_cells(separatrices)


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
    """A closed separatrix that reached its target on the very first step."""

    return SeparatrixPath(
        start_saddle=start, flow_direction=flow_direction, points=[],
        end_critical_point=end, termination_status="reached_critical_point",
        step_count=0, path_length=0.0,
    )


def make_quad_cell(
    saddle_a: ClassifiedCriticalPoint,
    saddle_b: ClassifiedCriticalPoint,
    maximum: ClassifiedCriticalPoint,
    minimum: ClassifiedCriticalPoint,
) -> MorseSmaleCell:
    """
    A hand-built, already-closed 4-vertex quad -- same shape family as
    the real egg-crate quad in tests/test_morse_smale_cells.py, but
    constructed directly (not via assemble_morse_smale_cells) so tests
    can control exact coordinates/scale/classification.
    """

    sep_a_max = zero_length_separatrix(saddle_a, maximum, "ascending")
    sep_b_max = zero_length_separatrix(saddle_b, maximum, "ascending")
    sep_b_min = zero_length_separatrix(saddle_b, minimum, "descending")
    sep_a_min = zero_length_separatrix(saddle_a, minimum, "descending")

    return MorseSmaleCell(
        cell_id=0,
        boundary_separatrices=[sep_a_max, sep_b_max, sep_b_min, sep_a_min],
        boundary_traversal_directions=[True, False, True, False],
        boundary_critical_points=[maximum, saddle_b, minimum, saddle_a],
        is_closed=True,
        open_boundaries=[],
    )


# ---------------------------------------------------------
# Egg-crate reference surface: all closed cells accepted
# ---------------------------------------------------------


def test_egg_crate_reference_cells_are_all_accepted_by_default():
    surface = surface_from_function(egg_crate(4.0, 4.0))
    _separatrices, complex_ = assemble_from_surface(surface)

    quality = assess_morse_smale_cell_quality(complex_.cells, complex_.topology_issues)

    assert len(quality) == 25
    assert all(assessment.is_accepted for assessment in quality)
    assert all(assessment.rejection_reasons == [] for assessment in quality)
    # Non-destructive: the original cell objects are preserved by identity.
    for assessment, cell in zip(quality, complex_.cells):
        assert assessment.cell is cell


# ---------------------------------------------------------
# Open cells: rejected, geometry-dependent checks skipped not passed
# ---------------------------------------------------------


def test_open_cells_are_rejected_and_geometry_checks_are_skipped_not_passed():
    surface = surface_from_function(
        two_bumps_along_x(2.5, 5.5, 4.0, amplitude=10.0, sigma=1.2),
        node_count=61, resolution=81,
    )
    _separatrices, complex_ = assemble_from_surface(surface)

    quality = assess_morse_smale_cell_quality(complex_.cells, complex_.topology_issues)

    assert len(quality) == 2
    for assessment in quality:
        assert not assessment.is_accepted
        reason_kinds = [reason.kind for reason in assessment.rejection_reasons]
        assert MorseSmaleCellRejectionReasonKind.NOT_CLOSED in reason_kinds
        # Geometry-dependent criteria must never appear for an open
        # cell -- not because they "passed" (compute_cell_geometry
        # returns None, there is no polygon to measure), simply
        # inapplicable.
        assert MorseSmaleCellRejectionReasonKind.AREA_TOO_SMALL not in reason_kinds
        assert MorseSmaleCellRejectionReasonKind.PERIMETER_TOO_SMALL not in reason_kinds
        assert compute_cell_geometry(assessment.cell) is None


# ---------------------------------------------------------
# Topology-issue cross-referencing
# ---------------------------------------------------------


def test_cell_touching_a_topology_issue_is_rejected_and_references_its_kind():
    saddle_a = make_point(1.0, 1.0, "saddle")
    saddle_b = make_point(2.0, 1.0, "saddle")
    other_max = make_point(1.0, 2.0, "maximum")
    other_min = make_point(1.0, 0.0, "minimum")

    separatrices = [
        zero_length_separatrix(saddle_a, saddle_b, "ascending"),
        zero_length_separatrix(saddle_a, other_max, "ascending"),
        zero_length_separatrix(saddle_a, other_min, "descending"),
        zero_length_separatrix(saddle_a, other_min, "descending"),
    ]
    complex_ = assemble_morse_smale_cells(separatrices)

    saddle_to_saddle_issues = [
        issue
        for issue in complex_.topology_issues
        if issue.kind == TopologyIssueKind.SADDLE_TO_SADDLE_CONNECTIVITY
    ]
    assert len(saddle_to_saddle_issues) == 1

    affected_cell = next(
        cell
        for cell in complex_.cells
        if saddle_to_saddle_issues[0].separatrix in cell.boundary_separatrices
    )

    quality = assess_morse_smale_cell_quality(complex_.cells, complex_.topology_issues)
    assessment = next(a for a in quality if a.cell is affected_cell)

    assert not assessment.is_accepted
    related_kinds = [
        reason.related_topology_issue
        for reason in assessment.rejection_reasons
        if reason.kind == MorseSmaleCellRejectionReasonKind.AFFECTED_BY_TOPOLOGY_ISSUE
    ]
    assert TopologyIssueKind.SADDLE_TO_SADDLE_CONNECTIVITY in related_kinds

    # topology_issues and rejection_reasons stay structurally separate
    # -- the original TopologyIssue objects are never copied in.
    assert all(
        not hasattr(reason, "separatrix") for reason in assessment.rejection_reasons
    )


# ---------------------------------------------------------
# Minimum area / perimeter
# ---------------------------------------------------------


def test_tiny_closed_cell_is_rejected_for_area_and_perimeter():
    saddle_a = make_point(0.60, 0.60, "saddle")
    saddle_b = make_point(0.61, 0.60, "saddle")
    maximum = make_point(0.605, 0.605, "maximum")
    minimum = make_point(0.605, 0.595, "minimum")

    cell = make_quad_cell(saddle_a, saddle_b, maximum, minimum)
    geometry = compute_cell_geometry(cell)
    assert geometry.area < MIN_CELL_AREA
    assert geometry.perimeter < MIN_CELL_PERIMETER

    quality = assess_morse_smale_cell_quality([cell], [])
    assessment = quality[0]

    assert not assessment.is_accepted
    reason_kinds = {reason.kind for reason in assessment.rejection_reasons}
    assert MorseSmaleCellRejectionReasonKind.AREA_TOO_SMALL in reason_kinds
    assert MorseSmaleCellRejectionReasonKind.PERIMETER_TOO_SMALL in reason_kinds


def test_realistically_sized_closed_cell_clears_area_and_perimeter():
    # Same shape/scale as the verified real egg-crate quad (side
    # spacing 1.5): area 1.125, perimeter ~4.24 -- both comfortably
    # above the thresholds.
    saddle_a = make_point(1.75, 1.75, "saddle")
    saddle_b = make_point(3.25, 1.75, "saddle")
    maximum = make_point(2.5, 2.5, "maximum")
    minimum = make_point(2.5, 1.0, "minimum")

    cell = make_quad_cell(saddle_a, saddle_b, maximum, minimum)

    quality = assess_morse_smale_cell_quality([cell], [])
    assert quality[0].is_accepted
    assert quality[0].rejection_reasons == []


# ---------------------------------------------------------
# Minimum distinct boundary points (degenerate bigon)
# ---------------------------------------------------------


def test_closed_bigon_is_rejected_for_too_few_distinct_boundary_points():
    # A saddle whose two same-type branches both happen to reach the
    # identical target -- a legitimately CLOSED 2-vertex loop (zero
    # area, since it's a degenerate back-and-forth line, not a real
    # polygon), verified directly before writing this assertion.
    saddle = make_point(1.0, 1.0, "saddle")
    maximum = make_point(1.5, 1.0, "maximum")

    sep_a = zero_length_separatrix(saddle, maximum, "ascending")
    sep_b = zero_length_separatrix(saddle, maximum, "ascending")

    bigon = MorseSmaleCell(
        cell_id=0,
        boundary_separatrices=[sep_a, sep_b],
        boundary_traversal_directions=[True, False],
        boundary_critical_points=[maximum, saddle],
        is_closed=True,
        open_boundaries=[],
    )

    quality = assess_morse_smale_cell_quality([bigon], [])
    assessment = quality[0]

    assert not assessment.is_accepted
    reason_kinds = {reason.kind for reason in assessment.rejection_reasons}
    assert MorseSmaleCellRejectionReasonKind.TOO_FEW_DISTINCT_BOUNDARY_POINTS in reason_kinds


# ---------------------------------------------------------
# Optional: boundary proximity
# ---------------------------------------------------------


def _minimal_surface(domain_min: float, domain_max: float) -> AttackInfluenceSurface:
    x = np.linspace(domain_min, domain_max, 10)
    row_coords = np.arange(8) + 0.5
    matrix = np.zeros((8, 8))
    spline = RectBivariateSpline(row_coords, row_coords, matrix, kx=3, ky=3)
    return AttackInfluenceSurface(x=x, y=x, z=spline(x, x), resolution=10, spline=spline)


def test_boundary_margin_is_opt_in_and_requires_a_surface():
    saddle_a = make_point(0.55, 1.0, "saddle")
    saddle_b = make_point(0.90, 1.0, "saddle")
    maximum = make_point(0.70, 1.20, "maximum")
    minimum = make_point(0.70, 0.80, "minimum")

    cell = make_quad_cell(saddle_a, saddle_b, maximum, minimum)
    surface = _minimal_surface(DOMAIN_MIN, DOMAIN_MAX)

    # Off by default: no surface needed, cell is otherwise healthy.
    default_quality = assess_morse_smale_cell_quality([cell], [])
    assert default_quality[0].is_accepted

    # Passing boundary_margin without a surface is a caller error.
    with pytest.raises(ValueError):
        assess_morse_smale_cell_quality([cell], [], boundary_margin=0.2)

    # Enabled explicitly: this cell sits within 0.2 of the domain's
    # x_min edge on every vertex.
    strict_quality = assess_morse_smale_cell_quality(
        [cell], [], surface=surface, boundary_margin=0.2
    )
    assert not strict_quality[0].is_accepted
    reason_kinds = {reason.kind for reason in strict_quality[0].rejection_reasons}
    assert MorseSmaleCellRejectionReasonKind.NEAR_BOUNDARY in reason_kinds


# ---------------------------------------------------------
# Optional: degenerate vertex participation
# ---------------------------------------------------------


def test_degenerate_vertex_participation_is_opt_in():
    saddle_a = make_point(1.75, 1.75, "saddle")
    saddle_b = make_point(3.25, 1.75, "saddle")
    maximum = make_point(2.5, 2.5, "degenerate")
    minimum = make_point(2.5, 1.0, "minimum")

    cell = make_quad_cell(saddle_a, saddle_b, maximum, minimum)

    default_quality = assess_morse_smale_cell_quality([cell], [])
    assert default_quality[0].is_accepted

    strict_quality = assess_morse_smale_cell_quality(
        [cell], [], reject_degenerate_vertex_participation=True
    )
    assert not strict_quality[0].is_accepted
    reason_kinds = {reason.kind for reason in strict_quality[0].rejection_reasons}
    assert MorseSmaleCellRejectionReasonKind.DEGENERATE_VERTEX_PARTICIPATION in reason_kinds


# ---------------------------------------------------------
# Determinism and no mutation
# ---------------------------------------------------------


def test_assess_morse_smale_cell_quality_is_deterministic():
    surface = surface_from_function(egg_crate(4.0, 4.0))
    _separatrices, complex_ = assemble_from_surface(surface)

    first_run = assess_morse_smale_cell_quality(complex_.cells, complex_.topology_issues)
    second_run = assess_morse_smale_cell_quality(complex_.cells, complex_.topology_issues)

    assert len(first_run) == len(second_run)
    for assessment_a, assessment_b in zip(first_run, second_run):
        assert assessment_a.is_accepted == assessment_b.is_accepted
        assert len(assessment_a.rejection_reasons) == len(assessment_b.rejection_reasons)


def test_assess_morse_smale_cell_quality_does_not_mutate_inputs():
    surface = surface_from_function(egg_crate(4.0, 4.0))
    _separatrices, complex_ = assemble_from_surface(surface)

    cells_before = copy.deepcopy(complex_.cells)

    assess_morse_smale_cell_quality(complex_.cells, complex_.topology_issues)

    for cell_after, cell_before in zip(complex_.cells, cells_before):
        assert cell_after.is_closed == cell_before.is_closed
        assert len(cell_after.boundary_separatrices) == len(cell_before.boundary_separatrices)
        assert cell_after.boundary_traversal_directions == cell_before.boundary_traversal_directions


# ---------------------------------------------------------
# Real-position smoke test
# ---------------------------------------------------------


def test_real_position_pipeline_runs_end_to_end_without_crashing():
    import chess

    from analysis.attack_influence import build_attack_influence_field
    from analysis.attack_influence_surface import build_attack_influence_surface

    board = chess.Board()
    for move in ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"]:
        board.push_uci(move)

    field = build_attack_influence_field(board)
    surface = build_attack_influence_surface(field)
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)
    assessments = assess_critical_point_quality(classified, surface)
    separatrices = locate_morse_smale_separatrices(surface, assessments)
    complex_ = assemble_morse_smale_cells(separatrices)

    quality = assess_morse_smale_cell_quality(complex_.cells, complex_.topology_issues)

    assert len(quality) == len(complex_.cells)
    accepted_count = sum(1 for assessment in quality if assessment.is_accepted)
    rejected_count = len(quality) - accepted_count
    assert accepted_count + rejected_count == len(quality)
    for assessment in quality:
        assert assessment.is_accepted == (assessment.rejection_reasons == [])
