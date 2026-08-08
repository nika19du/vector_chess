import copy
import math

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from analysis.critical_point_quality import assess_critical_point_quality
from analysis.critical_points import classify_critical_points, locate_critical_points
from analysis.morse_smale import (
    assemble_morse_smale_cells,
    compute_cell_geometry,
    locate_morse_smale_separatrices,
)
from chess_engine.models import (
    AttackInfluenceSurface,
    ClassifiedCriticalPoint,
    SeparatrixPath,
    SeparatrixPoint,
    TopologyIssueKind,
)


DOMAIN_MIN = 0.5
DOMAIN_MAX = 7.5


# ---------------------------------------------------------
# Shared helpers
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
    """
    The textbook Morse function amplitude*cos(x)*cos(y): a checkerboard
    of maxima/minima at integer-period grid points with saddles at the
    half-period midpoints between them -- verified directly against
    the real locate_critical_points/classify_critical_points/
    locate_morse_smale_separatrices pipeline (see conversation) before
    writing any test against it: on [0.5, 7.5]^2 with period=1.5 this
    produces 25 extrema, 16 saddles, and all 64 separatrices closed
    (none open) -- a fully-tiled reference complex.
    """

    def value(x, y):
        return amplitude * np.cos((x - x0) * (np.pi / period)) * np.cos(
            (y - y0) * (np.pi / period)
        )

    return value


def two_bumps_along_x(
    cx1: float, cx2: float, cy: float, amplitude: float = 10.0, sigma: float = 1.2
):
    """Same construction as tests/test_morse_smale.py -- one saddle, two maxima."""

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


def find_cell_by_vertices(complex_, expected_coords: set[tuple[float, float]]):
    for cell in complex_.cells:
        coords = {
            (round(point.x, 2), round(point.y, 2))
            for point in cell.boundary_critical_points
        }
        if coords == expected_coords:
            return cell
    return None


def make_point(x: float, y: float, classification: str) -> ClassifiedCriticalPoint:
    return ClassifiedCriticalPoint(
        x=x, y=y, value=0.0, gradient_norm=0.0, status="converged", iterations=1,
        f_xx=-1.0, f_xy=0.0, f_yx=0.0, f_yy=-1.0,
        eigenvalue_min=-1.0, eigenvalue_max=1.0,
        classification=classification,
    )


def make_closed_separatrix(
    start: ClassifiedCriticalPoint,
    end: ClassifiedCriticalPoint,
    flow_direction: str = "ascending",
) -> SeparatrixPath:
    points = [SeparatrixPoint(x=start.x + 0.05, y=start.y, value=0.0)]
    return SeparatrixPath(
        start_saddle=start, flow_direction=flow_direction, points=points,
        end_critical_point=end, termination_status="reached_critical_point",
        step_count=1, path_length=0.05,
    )


# ---------------------------------------------------------
# Egg-crate reference surface: fully closed, tiled complex
# ---------------------------------------------------------


def test_egg_crate_reference_surface_produces_all_closed_cells():
    surface = surface_from_function(egg_crate(4.0, 4.0))
    separatrices, complex_ = assemble_from_surface(surface)

    assert len(separatrices) == 64
    assert all(s.termination_status == "reached_critical_point" for s in separatrices)

    # Empirically confirmed by direct execution (not derived by hand --
    # not every cell in this construction is a simple 4-edge quad near
    # the board boundary, so the count isn't a trivial closed-form
    # formula): 25 cells, all closed, nothing left over.
    assert len(complex_.cells) == 25
    assert all(cell.is_closed for cell in complex_.cells)
    assert all(cell.open_boundaries == [] for cell in complex_.cells)
    assert complex_.unassigned_separatrices == []
    assert complex_.topology_issues == []
    assert complex_.edges == separatrices


def test_specific_quad_has_the_expected_four_boundary_vertices():
    surface = surface_from_function(egg_crate(4.0, 4.0))
    _separatrices, complex_ = assemble_from_surface(surface)

    # Hand-identified and verified directly against the real traced
    # separatrices: saddle(1.75,1.75) and saddle(3.25,1.75) both reach
    # maximum(2.5,2.5) (ascending) and minimum(2.5,1.0) (descending).
    cell = find_cell_by_vertices(
        complex_, {(1.75, 1.75), (3.25, 1.75), (2.5, 2.5), (2.5, 1.0)}
    )

    assert cell is not None
    assert cell.is_closed
    assert len(cell.boundary_separatrices) == 4
    assert len(cell.boundary_traversal_directions) == 4
    assert len(cell.boundary_critical_points) == 4
    assert cell.open_boundaries == []


def test_quad_area_and_centroid_match_the_exact_rhombus_geometry():
    surface = surface_from_function(egg_crate(4.0, 4.0))
    _separatrices, complex_ = assemble_from_surface(surface)

    cell = find_cell_by_vertices(
        complex_, {(1.75, 1.75), (3.25, 1.75), (2.5, 2.5), (2.5, 1.0)}
    )
    geometry = compute_cell_geometry(cell)

    # The 4 vertices form a rhombus with diagonals d1=1.5 (between the
    # two saddles) and d2=1.5 (between max and min): area = d1*d2/2
    # exactly, independent of how finely the curved boundary is
    # sampled -- a real external check, not just "trust the same
    # shoelace code that computed it."
    assert geometry.area == pytest.approx(1.5 * 1.5 / 2.0, abs=1e-3)
    assert geometry.centroid[0] == pytest.approx(2.5, abs=1e-3)
    assert geometry.centroid[1] == pytest.approx(1.75, abs=1e-3)
    assert geometry.polygon[0] == geometry.polygon[-1]


def test_quad_perimeter_equals_reconstructed_edge_lengths():
    # NOTE: perimeter is NOT simply the sum of each boundary
    # separatrix's own path_length -- every separatrix stops
    # REACHED_CRITICAL_POINT_DISTANCE short of its true target (by
    # Phase 1 design, verified directly), so the polygon includes one
    # additional "closing gap" segment per edge (from its last traced
    # point to the true critical point) that path_length does not
    # cover. This test reconstructs each edge's true contribution
    # (path_length + closing gap, computed independently from the raw
    # points/start/end, not from compute_cell_geometry's own
    # perimeter logic) and compares totals.
    surface = surface_from_function(egg_crate(4.0, 4.0))
    _separatrices, complex_ = assemble_from_surface(surface)

    cell = find_cell_by_vertices(
        complex_, {(1.75, 1.75), (3.25, 1.75), (2.5, 2.5), (2.5, 1.0)}
    )
    geometry = compute_cell_geometry(cell)

    def closing_gap(separatrix):
        last = separatrix.points[-1] if separatrix.points else separatrix.start_saddle
        return math.hypot(
            last.x - separatrix.end_critical_point.x,
            last.y - separatrix.end_critical_point.y,
        )

    reconstructed = sum(
        separatrix.path_length + closing_gap(separatrix)
        for separatrix in cell.boundary_separatrices
    )

    assert geometry.perimeter == pytest.approx(reconstructed, abs=1e-9)


def test_adjacent_cells_share_one_separatrix_with_opposite_direction():
    surface = surface_from_function(egg_crate(4.0, 4.0))
    _separatrices, complex_ = assemble_from_surface(surface)

    cell_a = find_cell_by_vertices(
        complex_, {(1.75, 1.75), (3.25, 1.75), (2.5, 2.5), (2.5, 1.0)}
    )
    cell_b = find_cell_by_vertices(
        complex_, {(1.75, 1.75), (1.75, 3.25), (2.5, 2.5), (1.0, 2.5)}
    )
    assert cell_a is not None and cell_b is not None

    shared_ids = set(id(s) for s in cell_a.boundary_separatrices) & set(
        id(s) for s in cell_b.boundary_separatrices
    )
    assert len(shared_ids) == 1

    shared_id = next(iter(shared_ids))
    index_a = [id(s) for s in cell_a.boundary_separatrices].index(shared_id)
    index_b = [id(s) for s in cell_b.boundary_separatrices].index(shared_id)

    # The SAME SeparatrixPath object, traversed in opposite directions
    # by the two cells it separates.
    assert cell_a.boundary_separatrices[index_a] is cell_b.boundary_separatrices[index_b]
    assert (
        cell_a.boundary_traversal_directions[index_a]
        != cell_b.boundary_traversal_directions[index_b]
    )


# ---------------------------------------------------------
# Open / incomplete cells
# ---------------------------------------------------------


def test_single_saddle_dumbbell_produces_only_open_cells():
    # Verified directly: one saddle, 2 ascending branches reach the
    # two maxima (closed), 2 descending branches leave the domain
    # (open) -- with no second saddle to close a loop, both faces this
    # produces must be open, each blocked by exactly one open
    # (left_domain) separatrix.
    surface = surface_from_function(
        two_bumps_along_x(2.5, 5.5, 4.0, amplitude=10.0, sigma=1.2),
        node_count=61, resolution=81,
    )
    separatrices, complex_ = assemble_from_surface(surface)

    ascending = [s for s in separatrices if s.flow_direction == "ascending"]
    descending = [s for s in separatrices if s.flow_direction == "descending"]
    assert all(s.termination_status == "reached_critical_point" for s in ascending)
    assert all(s.termination_status == "left_domain" for s in descending)

    assert len(complex_.cells) == 2
    for cell in complex_.cells:
        assert cell.is_closed is False
        assert len(cell.open_boundaries) == 1
        assert cell.open_boundaries[0] in descending
        assert compute_cell_geometry(cell) is None

    assert complex_.unassigned_separatrices == []
    assert complex_.topology_issues == []


# ---------------------------------------------------------
# Determinism
# ---------------------------------------------------------


def test_assemble_morse_smale_cells_is_deterministic():
    surface = surface_from_function(egg_crate(4.0, 4.0))
    separatrices, _complex = assemble_from_surface(surface)

    first_run = assemble_morse_smale_cells(separatrices)
    second_run = assemble_morse_smale_cells(separatrices)

    assert len(first_run.cells) == len(second_run.cells)

    for cell_a, cell_b in zip(first_run.cells, second_run.cells):
        assert cell_a.cell_id == cell_b.cell_id
        assert cell_a.is_closed == cell_b.is_closed
        assert len(cell_a.boundary_separatrices) == len(cell_b.boundary_separatrices)
        assert cell_a.boundary_traversal_directions == cell_b.boundary_traversal_directions

        for point_a, point_b in zip(
            cell_a.boundary_critical_points, cell_b.boundary_critical_points
        ):
            assert (point_a.x, point_a.y) == (point_b.x, point_b.y)


# ---------------------------------------------------------
# No mutation of inputs
# ---------------------------------------------------------


def test_assemble_morse_smale_cells_does_not_mutate_inputs():
    surface = surface_from_function(egg_crate(4.0, 4.0))
    separatrices, _complex = assemble_from_surface(surface)

    separatrices_before = copy.deepcopy(separatrices)

    assemble_morse_smale_cells(separatrices)

    for after, before in zip(separatrices, separatrices_before):
        assert after.start_saddle.x == before.start_saddle.x
        assert after.termination_status == before.termination_status
        assert len(after.points) == len(before.points)


# ---------------------------------------------------------
# Invalid / ambiguous topology
# ---------------------------------------------------------


def test_saddle_to_saddle_connectivity_is_included_and_flagged():
    saddle_a = make_point(1.0, 1.0, "saddle")
    saddle_b = make_point(2.0, 1.0, "saddle")
    other_max = make_point(1.0, 2.0, "maximum")
    other_min = make_point(1.0, 0.0, "minimum")

    separatrices = [
        make_closed_separatrix(saddle_a, saddle_b, "ascending"),
        make_closed_separatrix(saddle_a, other_max, "ascending"),
        make_closed_separatrix(saddle_a, other_min, "descending"),
        make_closed_separatrix(saddle_a, other_min, "descending"),
    ]

    complex_ = assemble_morse_smale_cells(separatrices)

    saddle_issues = [
        issue
        for issue in complex_.topology_issues
        if issue.kind == TopologyIssueKind.SADDLE_TO_SADDLE_CONNECTIVITY
    ]
    assert len(saddle_issues) == 1
    assert saddle_issues[0].separatrix is separatrices[0]

    # The saddle-to-saddle edge must still be present in the graph --
    # not silently dropped.
    assert any(s.end_critical_point is saddle_b for s in complex_.edges)


def test_saddle_branch_count_mismatch_is_flagged_not_raised():
    saddle = make_point(1.0, 1.0, "saddle")
    max_point = make_point(2.0, 1.0, "maximum")

    # Only one separatrix for this saddle instead of the expected 4.
    separatrices = [make_closed_separatrix(saddle, max_point, "ascending")]

    complex_ = assemble_morse_smale_cells(separatrices)

    mismatch_issues = [
        issue
        for issue in complex_.topology_issues
        if issue.kind == TopologyIssueKind.SADDLE_BRANCH_COUNT_MISMATCH
    ]
    assert len(mismatch_issues) == 1
    assert mismatch_issues[0].critical_point is saddle
    assert len(complex_.cells) >= 0  # must not raise


def test_zero_length_edge_is_flagged():
    saddle = make_point(1.0, 1.0, "saddle")
    almost_coincident = make_point(1.05, 1.0, "maximum")

    zero_length = SeparatrixPath(
        start_saddle=saddle, flow_direction="ascending", points=[],
        end_critical_point=almost_coincident, termination_status="reached_critical_point",
        step_count=0, path_length=0.0,
    )

    complex_ = assemble_morse_smale_cells([zero_length])

    zero_length_issues = [
        issue
        for issue in complex_.topology_issues
        if issue.kind == TopologyIssueKind.ZERO_LENGTH_EDGE
    ]
    assert len(zero_length_issues) == 1
    assert zero_length_issues[0].separatrix is zero_length
