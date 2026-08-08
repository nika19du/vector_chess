import copy
import dataclasses

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from analysis.critical_point_quality import assess_critical_point_quality
from analysis.critical_points import classify_critical_points, locate_critical_points
from analysis.morse_smale import STEP_SIZE, locate_morse_smale_separatrices
import analysis.morse_smale as morse_smale_module
from chess_engine.models import (
    AttackInfluenceSurface,
    ClassifiedCriticalPoint,
    CriticalPointQualityAssessment,
)


DOMAIN_MIN = 0.5
DOMAIN_MAX = 7.5


# ---------------------------------------------------------
# Shared helpers (same construction as tests/test_ridge_valley.py)
# ---------------------------------------------------------


def surface_from_function(
    func, node_count: int = 61, resolution: int = 81
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


def two_bumps_along_x(
    cx1: float,
    cx2: float,
    cy: float,
    amplitude: float = 10.0,
    sigma: float = 1.2,
):
    """
    Two identical Gaussian bumps (amplitude > 0: two peaks; amplitude
    < 0: two dips) centered on the same y = cy, spaced far enough
    apart relative to sigma that their midpoint is a genuine saddle:
    curvature along x (connecting the two bumps) has the opposite
    sign from curvature along y (perpendicular), by symmetry. Verified
    numerically against the real locate_critical_points/
    classify_critical_points pipeline before writing any test against
    it (see conversation) -- not just asserted analytically.

    amplitude > 0: saddle's ascending (eigenvalue_max) direction runs
        along x, toward the two peaks (maxima).
    amplitude < 0: saddle's descending (eigenvalue_min) direction runs
        along x, toward the two dips (minima).
    Either way, the perpendicular (y) direction has no target -- it
    decays toward the flat background.
    """

    def value(x, y):
        return amplitude * np.exp(
            -((x - cx1) ** 2 + (y - cy) ** 2) / (2 * sigma**2)
        ) + amplitude * np.exp(
            -((x - cx2) ** 2 + (y - cy) ** 2) / (2 * sigma**2)
        )

    return value


def classify_surface(
    surface: AttackInfluenceSurface,
) -> tuple[list[ClassifiedCriticalPoint], list[CriticalPointQualityAssessment]]:
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)
    assessments = assess_critical_point_quality(classified, surface)
    return classified, assessments


def make_fake_assessment(
    x: float,
    y: float,
    is_accepted: bool,
    classification: str = "maximum",
) -> CriticalPointQualityAssessment:
    """
    A CriticalPointQualityAssessment standing in for an unrelated,
    already-classified (and already quality-assessed) critical point
    elsewhere on the surface -- only x, y and is_accepted are read by
    the separatrix reconnection checks, so the other fields carry
    plausible but otherwise unused values. Mirrors
    tests/test_ridge_valley.py's make_fake_other_point, one layer up
    (wrapped in the quality assessment Morse-Smale actually consumes).
    """

    point = ClassifiedCriticalPoint(
        x=x,
        y=y,
        value=0.0,
        gradient_norm=0.0,
        status="converged",
        iterations=1,
        f_xx=-1.0,
        f_xy=0.0,
        f_yx=0.0,
        f_yy=-1.0,
        eigenvalue_min=-1.0,
        eigenvalue_max=-1.0,
        classification=classification,
    )

    return CriticalPointQualityAssessment(
        point=point,
        is_accepted=is_accepted,
        rejection_reasons=[] if is_accepted else ["forced for test"],
    )


# ---------------------------------------------------------
# Ascending branches reach known maxima
# ---------------------------------------------------------


def test_ascending_branches_reach_the_two_known_maxima():
    surface = surface_from_function(
        two_bumps_along_x(2.5, 5.5, 4.0, amplitude=10.0, sigma=1.2)
    )
    classified, assessments = classify_surface(surface)

    maxima = [p for p in classified if p.classification == "maximum"]
    saddles = [p for p in classified if p.classification == "saddle"]
    assert len(maxima) == 2
    assert len(saddles) == 1

    paths = locate_morse_smale_separatrices(surface, assessments)
    ascending_paths = [p for p in paths if p.flow_direction == "ascending"]
    assert len(ascending_paths) == 2

    for path in ascending_paths:
        assert path.start_saddle is saddles[0]
        assert path.termination_status == "reached_critical_point"
        assert path.end_critical_point in maxima
        assert path.step_count == len(path.points)
        assert path.path_length > 0.0

    # Each of the two maxima is reached by exactly one ascending branch.
    # ClassifiedCriticalPoint has no __hash__ (plain dataclass), so
    # identity is compared via id() rather than via a set.
    reached_ids = sorted(id(path.end_critical_point) for path in ascending_paths)
    expected_ids = sorted(id(point) for point in maxima)
    assert reached_ids == expected_ids


# ---------------------------------------------------------
# Descending branches reach known minima
# ---------------------------------------------------------


def test_descending_branches_reach_the_two_known_minima():
    surface = surface_from_function(
        two_bumps_along_x(2.5, 5.5, 4.0, amplitude=-10.0, sigma=1.2)
    )
    classified, assessments = classify_surface(surface)

    minima = [p for p in classified if p.classification == "minimum"]
    saddles = [p for p in classified if p.classification == "saddle"]
    assert len(minima) == 2
    assert len(saddles) == 1

    paths = locate_morse_smale_separatrices(surface, assessments)
    descending_paths = [p for p in paths if p.flow_direction == "descending"]
    assert len(descending_paths) == 2

    for path in descending_paths:
        assert path.start_saddle is saddles[0]
        assert path.termination_status == "reached_critical_point"
        assert path.end_critical_point in minima

    reached_ids = sorted(id(path.end_critical_point) for path in descending_paths)
    expected_ids = sorted(id(point) for point in minima)
    assert reached_ids == expected_ids


# ---------------------------------------------------------
# Boundary termination
# ---------------------------------------------------------


def test_perpendicular_branches_terminate_on_domain_exit():
    # Same two-peak surface as the ascending test: the perpendicular
    # (descending) direction has no target and runs straight toward
    # the board edge.
    surface = surface_from_function(
        two_bumps_along_x(2.5, 5.5, 4.0, amplitude=10.0, sigma=1.2)
    )
    _classified, assessments = classify_surface(surface)

    paths = locate_morse_smale_separatrices(surface, assessments)
    descending_paths = [p for p in paths if p.flow_direction == "descending"]
    assert len(descending_paths) == 2

    for path in descending_paths:
        assert path.termination_status == "left_domain"
        assert path.end_critical_point is None
        assert path.step_count > 0


# ---------------------------------------------------------
# Near-zero-gradient stagnation
# ---------------------------------------------------------


def test_perpendicular_branches_stagnate_before_reaching_the_boundary():
    # A narrow, low-amplitude bump pair: far along the perpendicular
    # direction the field flattens toward the background well before
    # either branch reaches the domain edge.
    surface = surface_from_function(
        two_bumps_along_x(2.5, 5.5, 4.0, amplitude=1.0, sigma=0.5),
        node_count=81,
        resolution=121,
    )
    classified, assessments = classify_surface(surface)
    saddles = [p for p in classified if p.classification == "saddle"]
    assert len(saddles) == 1

    paths = locate_morse_smale_separatrices(surface, assessments)
    descending_paths = [p for p in paths if p.flow_direction == "descending"]
    assert len(descending_paths) == 2

    for path in descending_paths:
        assert path.termination_status == "gradient_stagnation"
        assert path.end_critical_point is None
        assert path.step_count > 0


# ---------------------------------------------------------
# Rejected / unreliable critical-point termination
# ---------------------------------------------------------


def test_ascending_branch_stops_at_a_rejected_critical_point():
    surface = surface_from_function(
        two_bumps_along_x(2.5, 5.5, 4.0, amplitude=10.0, sigma=1.2)
    )
    classified, assessments = classify_surface(surface)

    maxima = [p for p in classified if p.classification == "maximum"]
    left_maximum = min(maxima, key=lambda p: p.x)
    saddle = next(p for p in classified if p.classification == "saddle")

    # Planted directly on the ascending branch's path toward the left
    # maximum, well before it -- a real, converged critical point that
    # quality assessment has rejected (as opposed to accepted).
    fake_x = (saddle.x + left_maximum.x) / 2
    fake_rejected = make_fake_assessment(fake_x, saddle.y, is_accepted=False)

    augmented = assessments + [fake_rejected]
    paths = locate_morse_smale_separatrices(surface, augmented)
    ascending_paths = [p for p in paths if p.flow_direction == "ascending"]
    assert len(ascending_paths) == 2

    intercepted = [
        p for p in ascending_paths if p.termination_status == "reached_unreliable_point"
    ]
    assert len(intercepted) == 1

    intercepted_path = intercepted[0]
    assert intercepted_path.end_critical_point is fake_rejected.point
    # The branch must stop short of (not past) the real maximum it was
    # heading toward.
    for point in intercepted_path.points:
        assert abs(point.x - saddle.x) < abs(left_maximum.x - saddle.x)

    # The OTHER ascending branch (toward the right maximum) is
    # unaffected by the unrelated rejected point.
    unaffected = [p for p in ascending_paths if p is not intercepted_path]
    assert len(unaffected) == 1
    assert unaffected[0].termination_status == "reached_critical_point"


# ---------------------------------------------------------
# Self-intersection prevention (wiring, not a naturally
# self-intersecting field -- gradient flow lines of a smooth scalar
# field provably cannot cross themselves away from critical points,
# so this verifies _trace_gradient_flow_direction actually consults
# and obeys check_self_intersection, the same reused function
# analysis/ridge_valley.py already tests in isolation).
# ---------------------------------------------------------


def test_self_intersection_check_stops_the_branch(monkeypatch):
    surface = surface_from_function(
        two_bumps_along_x(2.5, 5.5, 4.0, amplitude=10.0, sigma=1.2)
    )
    _classified, assessments = classify_surface(surface)

    # Force every candidate step to be flagged as a self-intersection,
    # regardless of geometry -- proves _trace_gradient_flow_direction
    # actually consults and obeys check_self_intersection (the same
    # reused function analysis/ridge_valley.py already tests in
    # isolation), rather than asserting on a naturally-occurring loop:
    # gradient flow lines of a smooth scalar field provably cannot
    # cross themselves away from critical points, so no real synthetic
    # surface would ever exercise this path organically.
    monkeypatch.setattr(
        morse_smale_module, "check_self_intersection", lambda *args, **kwargs: True
    )

    paths = locate_morse_smale_separatrices(surface, assessments)

    assert len(paths) == 4
    for path in paths:
        assert path.termination_status == "self_intersection"
        assert path.step_count == 0
        assert path.points == []
        assert path.end_critical_point is None


# ---------------------------------------------------------
# Max steps reached
# ---------------------------------------------------------


def test_branches_stop_at_max_steps_when_budget_is_small():
    surface = surface_from_function(
        two_bumps_along_x(2.5, 5.5, 4.0, amplitude=10.0, sigma=1.2)
    )
    _classified, assessments = classify_surface(surface)

    paths = locate_morse_smale_separatrices(surface, assessments, max_steps=3)

    assert len(paths) == 4
    for path in paths:
        assert path.termination_status == "max_steps_reached"
        assert path.step_count == 3


# ---------------------------------------------------------
# Only quality-accepted critical points become anchors
# ---------------------------------------------------------


def test_a_rejected_saddle_produces_no_separatrices():
    surface = surface_from_function(
        two_bumps_along_x(2.5, 5.5, 4.0, amplitude=10.0, sigma=1.2)
    )
    _classified, assessments = classify_surface(surface)

    forced_rejected = [
        dataclasses.replace(
            assessment, is_accepted=False, rejection_reasons=["forced for test"]
        )
        if assessment.point.classification == "saddle"
        else assessment
        for assessment in assessments
    ]

    paths = locate_morse_smale_separatrices(surface, forced_rejected)
    assert paths == []


# ---------------------------------------------------------
# Continuous, off-grid coordinates
# ---------------------------------------------------------


def test_separatrix_preserves_true_continuous_coordinates_off_grid():
    # An off-grid saddle center -- neither bump position nor cy is
    # aligned with any 0.5-spaced board-cell boundary or the sparse
    # 61-node fitting lattice.
    surface = surface_from_function(
        two_bumps_along_x(2.2, 5.9, 4.13, amplitude=10.0, sigma=1.2)
    )
    classified, assessments = classify_surface(surface)
    saddle = next(p for p in classified if p.classification == "saddle")

    assert saddle.x == pytest.approx(4.05, abs=0.01)
    assert saddle.y == pytest.approx(4.13, abs=1e-3)

    paths = locate_morse_smale_separatrices(surface, assessments)

    for path in paths:
        assert len(path.points) > 3
        # Consecutive points (a genuinely continuous march) are
        # separated by exactly one marching step.
        for first, second in zip(path.points, path.points[1:]):
            distance = ((second.x - first.x) ** 2 + (second.y - first.y) ** 2) ** 0.5
            assert distance == pytest.approx(STEP_SIZE, abs=1e-6)


# ---------------------------------------------------------
# Determinism
# ---------------------------------------------------------


def test_locate_morse_smale_separatrices_is_deterministic():
    surface = surface_from_function(
        two_bumps_along_x(2.5, 5.5, 4.0, amplitude=10.0, sigma=1.2)
    )
    _classified, assessments = classify_surface(surface)

    first_run = locate_morse_smale_separatrices(surface, assessments)
    second_run = locate_morse_smale_separatrices(surface, assessments)

    assert len(first_run) == len(second_run)

    for path_a, path_b in zip(first_run, second_run):
        assert path_a.flow_direction == path_b.flow_direction
        assert path_a.termination_status == path_b.termination_status
        assert path_a.step_count == path_b.step_count
        assert path_a.path_length == path_b.path_length

        for point_a, point_b in zip(path_a.points, path_b.points):
            assert point_a.x == point_b.x
            assert point_a.y == point_b.y
            assert point_a.value == point_b.value


# ---------------------------------------------------------
# No mutation of surface or critical points
# ---------------------------------------------------------


def test_locate_morse_smale_separatrices_does_not_mutate_inputs():
    surface = surface_from_function(
        two_bumps_along_x(2.5, 5.5, 4.0, amplitude=10.0, sigma=1.2)
    )
    _classified, assessments = classify_surface(surface)

    x_before = np.copy(surface.x)
    y_before = np.copy(surface.y)
    z_before = np.copy(surface.z)
    assessments_before = copy.deepcopy(assessments)

    locate_morse_smale_separatrices(surface, assessments)

    assert np.array_equal(surface.x, x_before)
    assert np.array_equal(surface.y, y_before)
    assert np.array_equal(surface.z, z_before)

    for assessment_after, assessment_before in zip(assessments, assessments_before):
        assert assessment_after == assessment_before
