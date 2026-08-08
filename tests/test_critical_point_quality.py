import copy

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from analysis.critical_point_quality import (
    BOUNDARY_MARGIN,
    MIN_CURVATURE_STRENGTH,
    MIN_EIGENVALUE_MAGNITUDE,
    assess_critical_point_quality,
)
from analysis.critical_points import classify_critical_points, locate_critical_points
from chess_engine.analyzer import analyze_position
from chess_engine.board import ChessGame
from chess_engine.models import AttackInfluenceSurface, ClassifiedCriticalPoint


DOMAIN_MIN = 0.5
DOMAIN_MAX = 7.5


def surface_from_function(func, node_count: int = 41, resolution: int = 61) -> AttackInfluenceSurface:
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


def gaussian_bump(cx, cy, amplitude=10.0, sigma=1.2):
    def value(x, y):
        return amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))

    return value


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


def real_position_analysis(moves: list[str]):
    game = ChessGame()
    move_details = None
    for move in moves:
        move_details = game.make_move(move)
    return game, analyze_position(game.board, move_details)


def by_status(assessments):
    accepted = [a for a in assessments if a.is_accepted]
    rejected = [a for a in assessments if not a.is_accepted]
    return accepted, rejected


# ---------------------------------------------------------
# Raw points always preserved, never mutated
# ---------------------------------------------------------


def test_all_input_points_are_preserved_in_the_output():
    points = [
        make_point(4.0, 4.0, "maximum"),
        make_point(0.5, 0.5, "minimum"),
        make_point(4.0, 4.0, "unclassified", status="left_domain", f_xx=None, f_xy=None, f_yx=None, f_yy=None, eigenvalue_min=None, eigenvalue_max=None),
    ]
    surface = surface_from_function(gaussian_bump(4.0, 4.0))

    assessments = assess_critical_point_quality(points, surface)

    assert len(assessments) == len(points)
    assert [a.point for a in assessments] == points


def test_assess_does_not_mutate_input_points_or_surface():
    points = [make_point(4.0, 4.0, "maximum"), make_point(0.6, 0.6, "minimum")]
    points_before = copy.deepcopy(points)

    surface = surface_from_function(gaussian_bump(4.0, 4.0))
    x_before = np.copy(surface.x)
    probe_before = surface.spline(4.0, 4.0, dx=1, dy=1)[0, 0]

    assess_critical_point_quality(points, surface)

    assert points == points_before
    assert np.array_equal(surface.x, x_before)
    assert surface.spline(4.0, 4.0, dx=1, dy=1)[0, 0] == pytest.approx(probe_before)


def test_every_rejected_point_has_at_least_one_reason():
    points = [
        make_point(4.0, 4.0, "maximum"),
        make_point(0.5, 0.5, "minimum"),  # boundary
        make_point(
            4.0, 4.0, "unclassified", status="left_domain",
            f_xx=None, f_xy=None, f_yx=None, f_yy=None,
            eigenvalue_min=None, eigenvalue_max=None,
        ),
    ]
    surface = surface_from_function(gaussian_bump(4.0, 4.0))

    assessments = assess_critical_point_quality(points, surface)

    for assessment in assessments:
        if not assessment.is_accepted:
            assert len(assessment.rejection_reasons) >= 1
        else:
            assert assessment.rejection_reasons == []


# ---------------------------------------------------------
# Non-converged candidates are always rejected
# ---------------------------------------------------------


def test_non_converged_point_is_always_rejected():
    point = make_point(
        4.0, 4.0, "unclassified", status="max_iterations_reached",
        f_xx=None, f_xy=None, f_yx=None, f_yy=None,
        eigenvalue_min=None, eigenvalue_max=None,
    )
    surface = surface_from_function(gaussian_bump(4.0, 4.0))

    assessment = assess_critical_point_quality([point], surface)[0]

    assert not assessment.is_accepted
    assert "not a converged critical point" in assessment.rejection_reasons[0]


# ---------------------------------------------------------
# Boundary margin
# ---------------------------------------------------------


def test_point_within_boundary_margin_is_rejected():
    point = make_point(DOMAIN_MIN + BOUNDARY_MARGIN / 2, 4.0, "maximum", eigenvalue_min=-5.0, eigenvalue_max=-5.0)
    surface = surface_from_function(gaussian_bump(4.0, 4.0))

    assessment = assess_critical_point_quality([point], surface)[0]

    assert not assessment.is_accepted
    assert any("boundary" in reason for reason in assessment.rejection_reasons)


def test_point_safely_inside_boundary_margin_is_accepted():
    point = make_point(4.0, 4.0, "maximum", eigenvalue_min=-5.0, eigenvalue_max=-5.0)
    surface = surface_from_function(gaussian_bump(4.0, 4.0))

    assessment = assess_critical_point_quality([point], surface)[0]

    assert assessment.is_accepted


# ---------------------------------------------------------
# Curvature strength thresholds
# ---------------------------------------------------------


def test_weak_but_non_degenerate_curvature_is_rejected_for_a_different_reason_than_degenerate():
    weak_eigenvalue = MIN_EIGENVALUE_MAGNITUDE / 2  # non-degenerate per Phase 3, weak per Phase 4a
    point = make_point(
        4.0, 4.0, "minimum", eigenvalue_min=weak_eigenvalue, eigenvalue_max=weak_eigenvalue * 1.1
    )
    surface = surface_from_function(gaussian_bump(4.0, 4.0))

    assessment = assess_critical_point_quality([point], surface)[0]

    assert not assessment.is_accepted
    assert any("weak curvature" in reason for reason in assessment.rejection_reasons)
    assert not any("degenerate classification" in reason for reason in assessment.rejection_reasons)
    assert assessment.point.classification == "minimum"  # Phase 3's own classification untouched


def test_strong_curvature_on_both_axes_is_accepted():
    point = make_point(4.0, 4.0, "saddle", eigenvalue_min=-3.0, eigenvalue_max=3.0)
    surface = surface_from_function(gaussian_bump(4.0, 4.0))

    assessment = assess_critical_point_quality([point], surface)[0]

    assert assessment.is_accepted


def test_curvature_strength_threshold_is_independent_of_eigenvalue_magnitude_threshold():
    """
    Both eigenvalues individually clear MIN_EIGENVALUE_MAGNITUDE, but
    their combined strength is set just under MIN_CURVATURE_STRENGTH
    -- this must be caught by the separate curvature_strength check,
    not silently passed because the per-axis check alone passed.
    """

    eigenvalue = MIN_EIGENVALUE_MAGNITUDE * 1.05
    combined = float(np.hypot(eigenvalue, eigenvalue))
    assert eigenvalue >= MIN_EIGENVALUE_MAGNITUDE
    assert combined < MIN_CURVATURE_STRENGTH

    point = make_point(4.0, 4.0, "minimum", eigenvalue_min=eigenvalue, eigenvalue_max=eigenvalue)
    surface = surface_from_function(gaussian_bump(4.0, 4.0))

    assessment = assess_critical_point_quality([point], surface)[0]

    assert not assessment.is_accepted
    assert any("overall curvature too weak" in reason for reason in assessment.rejection_reasons)


# ---------------------------------------------------------
# Multiple simultaneous failing reasons
# ---------------------------------------------------------


def test_point_failing_multiple_criteria_reports_all_of_them():
    point = make_point(
        DOMAIN_MIN + BOUNDARY_MARGIN / 2,
        4.0,
        "minimum",
        eigenvalue_min=MIN_EIGENVALUE_MAGNITUDE / 4,
        eigenvalue_max=MIN_EIGENVALUE_MAGNITUDE / 4,
    )
    surface = surface_from_function(gaussian_bump(4.0, 4.0))

    assessment = assess_critical_point_quality([point], surface)[0]

    assert not assessment.is_accepted
    assert len(assessment.rejection_reasons) >= 2
    assert any("boundary" in reason for reason in assessment.rejection_reasons)
    assert any("weak curvature" in reason for reason in assessment.rejection_reasons)


# ---------------------------------------------------------
# Degenerate suppression, toggleable
# ---------------------------------------------------------


def test_degenerate_suppressed_by_default():
    point = make_point(4.0, 4.0, "degenerate", eigenvalue_min=1e-8, eigenvalue_max=5.0)
    surface = surface_from_function(gaussian_bump(4.0, 4.0))

    assessment = assess_critical_point_quality([point], surface)[0]

    assert not assessment.is_accepted
    assert any("degenerate classification" in reason for reason in assessment.rejection_reasons)


def test_degenerate_suppression_is_toggleable():
    point = make_point(4.0, 4.0, "degenerate", eigenvalue_min=5.0, eigenvalue_max=5.0)
    surface = surface_from_function(gaussian_bump(4.0, 4.0))

    assessment = assess_critical_point_quality(
        [point], surface, suppress_near_flat_degenerate=False
    )[0]

    # Strong eigenvalues chosen so no OTHER criterion rejects it once
    # suppression is turned off.
    assert assessment.is_accepted


# ---------------------------------------------------------
# Optional gradient-quality threshold
# ---------------------------------------------------------


def test_gradient_quality_threshold_disabled_by_default():
    point = make_point(4.0, 4.0, "maximum", eigenvalue_min=-5.0, eigenvalue_max=-5.0, gradient_norm=9.9e-7)
    surface = surface_from_function(gaussian_bump(4.0, 4.0))

    assessment = assess_critical_point_quality([point], surface)[0]

    assert assessment.is_accepted


def test_gradient_quality_threshold_rejects_when_explicitly_set():
    point = make_point(4.0, 4.0, "maximum", eigenvalue_min=-5.0, eigenvalue_max=-5.0, gradient_norm=9.9e-7)
    surface = surface_from_function(gaussian_bump(4.0, 4.0))

    assessment = assess_critical_point_quality(
        [point], surface, gradient_quality_threshold=1e-8
    )[0]

    assert not assessment.is_accepted
    assert any("gradient norm" in reason for reason in assessment.rejection_reasons)


# ---------------------------------------------------------
# End-to-end: synthetic surface with true interior extrema
# ---------------------------------------------------------


def test_true_interior_extrema_survive_end_to_end_filtering():
    def three_features(x, y):
        maximum_bump = 9.0 * np.exp(-((x - 1.8) ** 2 + (y - 1.8) ** 2) / (2 * 0.9**2))
        minimum_bump = -9.0 * np.exp(-((x - 6.2) ** 2 + (y - 6.2) ** 2) / (2 * 0.9**2))
        u, v = x - 5.0, y - 2.0
        saddle_bump = 12.0 * u * v * np.exp(-(u**2 + v**2) / (2 * 1.1**2))
        return maximum_bump + minimum_bump + saddle_bump

    surface = surface_from_function(three_features)
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)
    assessments = assess_critical_point_quality(classified, surface)

    accepted, _ = by_status(assessments)

    def closest_accepted(x, y):
        return min(accepted, key=lambda a: np.hypot(a.point.x - x, a.point.y - y))

    maximum_match = closest_accepted(1.8, 1.8)
    minimum_match = closest_accepted(6.2, 6.2)
    saddle_match = closest_accepted(5.0, 2.0)

    assert maximum_match.point.classification == "maximum"
    assert maximum_match.point.x == pytest.approx(1.8, abs=0.05)
    assert minimum_match.point.classification == "minimum"
    assert minimum_match.point.x == pytest.approx(6.2, abs=0.05)
    assert saddle_match.point.classification == "saddle"
    assert saddle_match.point.x == pytest.approx(5.0, abs=0.1)


def test_boundary_artifacts_are_flagged_on_a_synthetic_surface():
    """
    A single bump's spline fit reliably produces small, real (but
    chess-meaningless) near-zero-gradient artifacts at the board
    corners -- documented and tested in Phase 2/4. The quality filter
    must flag every one of them.
    """

    surface = surface_from_function(gaussian_bump(4.0, 4.0))
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)
    assessments = assess_critical_point_quality(classified, surface)

    corner_assessments = [
        assessment
        for assessment in assessments
        if min(
            assessment.point.x - DOMAIN_MIN,
            DOMAIN_MAX - assessment.point.x,
            assessment.point.y - DOMAIN_MIN,
            DOMAIN_MAX - assessment.point.y,
        )
        < BOUNDARY_MARGIN
    ]

    assert corner_assessments  # the scenario must actually produce some
    assert all(not assessment.is_accepted for assessment in corner_assessments)

    true_peak = min(assessments, key=lambda a: np.hypot(a.point.x - 4.0, a.point.y - 4.0))
    assert true_peak.is_accepted
    assert true_peak.point.classification == "maximum"


# ---------------------------------------------------------
# End-to-end: a real chess position
# ---------------------------------------------------------


def test_real_position_boundary_points_rejected_interior_points_kept():
    """
    A different move sequence than any used in prior smoke tests, so
    the default thresholds are shown to generalize rather than being
    tuned to one specific game.
    """

    from analysis.attack_influence_surface import build_attack_influence_surface

    _game, analysis = real_position_analysis(
        ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"]
    )
    surface = build_attack_influence_surface(analysis.attack_influence_field)
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)
    assessments = assess_critical_point_quality(classified, surface)

    assert assessments  # sanity: the position must produce candidates at all

    for assessment in assessments:
        distance_to_boundary = min(
            assessment.point.x - DOMAIN_MIN,
            DOMAIN_MAX - assessment.point.x,
            assessment.point.y - DOMAIN_MIN,
            DOMAIN_MAX - assessment.point.y,
        )
        if distance_to_boundary < BOUNDARY_MARGIN:
            assert not assessment.is_accepted
            # A non-converged candidate near the boundary (Newton
            # itself often stops right at the edge -- "left_domain")
            # is short-circuit-rejected on status alone, never
            # reaching the boundary check; only a "converged" point
            # this close to the edge must carry the boundary reason.
            if assessment.point.status == "converged":
                assert any(
                    "boundary" in reason for reason in assessment.rejection_reasons
                )

    accepted, rejected = by_status(assessments)
    assert accepted  # this real, contested position must have SOME reliable point
