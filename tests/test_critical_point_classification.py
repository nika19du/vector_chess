import copy

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from analysis.critical_points import (
    EIGENVALUE_ZERO_THRESHOLD,
    classify_critical_points,
    locate_critical_points,
    refine_seed,
)
from chess_engine.models import AttackInfluenceSurface, CriticalPointCandidate


DOMAIN_MIN = 0.5
DOMAIN_MAX = 7.5


def surface_from_function(
    func, node_count: int = 41, resolution: int = 61
) -> AttackInfluenceSurface:
    """
    Same construction as tests/test_critical_point_localization.py's
    helper: a fine node grid avoids the 8x8-undersampling spline
    ringing already documented and tested in
    tests/test_attack_influence_surface.py, keeping these tests
    focused on classification given a known, clean Hessian.
    """

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


def closest_to(points, x, y):
    return min(points, key=lambda point: np.hypot(point.x - x, point.y - y))


# ---------------------------------------------------------
# Definiteness -> classification
# ---------------------------------------------------------


def test_positive_definite_hessian_classifies_as_minimum():
    def bowl(x, y):
        return 3.0 * (x - 4.0) ** 2 + 7.0 * (y - 4.0) ** 2

    surface = surface_from_function(bowl)
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)

    converged = [c for c in classified if c.status == "converged"]
    assert len(converged) == 1

    point = converged[0]
    assert point.classification == "minimum"
    assert point.eigenvalue_min == pytest.approx(6.0, abs=1e-6)
    assert point.eigenvalue_max == pytest.approx(14.0, abs=1e-6)
    assert point.f_xx == pytest.approx(6.0, abs=1e-6)
    assert point.f_yy == pytest.approx(14.0, abs=1e-6)


def test_negative_definite_hessian_classifies_as_maximum():
    def bump(x, y):
        return 10.0 * np.exp(-((x - 4.0) ** 2 + (y - 4.0) ** 2) / (2 * 1.2**2))

    surface = surface_from_function(bump)
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)

    peak = closest_to([c for c in classified if c.status == "converged"], 4.0, 4.0)

    assert peak.classification == "maximum"
    assert peak.eigenvalue_max < 0
    assert peak.eigenvalue_min < 0


def test_indefinite_hessian_classifies_as_saddle():
    def saddle(x, y):
        return 5.0 * (x - 4.0) ** 2 - 5.0 * (y - 4.0) ** 2

    surface = surface_from_function(saddle)
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)

    converged = [c for c in classified if c.status == "converged"]
    assert len(converged) == 1

    point = converged[0]
    assert point.classification == "saddle"
    assert point.eigenvalue_min < 0 < point.eigenvalue_max


# ---------------------------------------------------------
# Degenerate cases
# ---------------------------------------------------------


def test_one_near_zero_eigenvalue_is_degenerate():
    # Sharp curvature in x, a curvature in y far below
    # EIGENVALUE_ZERO_THRESHOLD -- only one eigenvalue is "flat".
    def one_flat_axis(x, y):
        return 2.0 * (x - 4.0) ** 2 + 1e-8 * (y - 4.0) ** 2

    surface = surface_from_function(one_flat_axis)
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)

    point = closest_to([c for c in classified if c.status == "converged"], 4.0, 4.0)

    assert point.classification == "degenerate"
    assert abs(point.eigenvalue_min) < EIGENVALUE_ZERO_THRESHOLD
    assert abs(point.eigenvalue_max) >= EIGENVALUE_ZERO_THRESHOLD


def test_both_near_zero_eigenvalues_are_degenerate():
    def both_flat(x, y):
        return 1e-8 * (x - 4.0) ** 2 + 1e-8 * (y - 4.0) ** 2

    surface = surface_from_function(both_flat)
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)

    point = closest_to([c for c in classified if c.status == "converged"], 4.0, 4.0)

    assert point.classification == "degenerate"
    assert abs(point.eigenvalue_min) < EIGENVALUE_ZERO_THRESHOLD
    assert abs(point.eigenvalue_max) < EIGENVALUE_ZERO_THRESHOLD


def test_degenerate_never_silently_reported_as_definite():
    """
    A near-zero eigenvalue must never be rounded into "minimum" or
    "maximum" by numerical luck -- explicitly checks the boundary
    case is caught, not just the deep-degenerate case above.
    """

    def barely_below_threshold(x, y):
        # f_xx = 2 * curvature must itself land below the threshold,
        # not the coefficient -- 0.25x keeps the eigenvalue at half
        # EIGENVALUE_ZERO_THRESHOLD with clear margin either side.
        curvature = EIGENVALUE_ZERO_THRESHOLD * 0.25
        return curvature * (x - 4.0) ** 2 + 3.0 * (y - 4.0) ** 2

    surface = surface_from_function(barely_below_threshold)
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)

    point = closest_to([c for c in classified if c.status == "converged"], 4.0, 4.0)

    assert point.classification == "degenerate"


# ---------------------------------------------------------
# Non-converged candidates
# ---------------------------------------------------------


def test_non_converged_candidate_is_unclassified_with_no_hessian():
    def far_off_vertex(x, y):
        return (x - 20.0) ** 2 + (y - 20.0) ** 2

    surface = surface_from_function(far_off_vertex, node_count=9)

    candidate = refine_seed(
        surface.spline,
        seed_y=4.0,
        seed_x=4.0,
        x_min=DOMAIN_MIN,
        x_max=DOMAIN_MAX,
        y_min=DOMAIN_MIN,
        y_max=DOMAIN_MAX,
    )
    assert candidate.status == "left_domain"

    classified = classify_critical_points([candidate], surface)

    assert len(classified) == 1
    point = classified[0]
    assert point.classification == "unclassified"
    assert point.f_xx is None
    assert point.f_xy is None
    assert point.f_yx is None
    assert point.f_yy is None
    assert point.eigenvalue_min is None
    assert point.eigenvalue_max is None
    # Localization metadata must still be preserved unchanged.
    assert point.x == candidate.x
    assert point.y == candidate.y
    assert point.value == candidate.value
    assert point.gradient_norm == candidate.gradient_norm
    assert point.status == candidate.status
    assert point.iterations == candidate.iterations


# ---------------------------------------------------------
# Off-grid classified critical point
# ---------------------------------------------------------


def test_off_grid_critical_point_is_classified_correctly():
    true_x, true_y = 3.13, 5.71

    def bump(x, y):
        return 10.0 * np.exp(-((x - true_x) ** 2 + (y - true_y) ** 2) / (2 * 1.2**2))

    surface = surface_from_function(bump)
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)

    point = closest_to(
        [c for c in classified if c.status == "converged"], true_x, true_y
    )

    assert point.x == pytest.approx(true_x, abs=1e-3)
    assert point.y == pytest.approx(true_y, abs=1e-3)
    assert point.classification == "maximum"


# ---------------------------------------------------------
# Multiple points of different types in one surface
# ---------------------------------------------------------


def test_multiple_distinct_types_on_one_surface():
    def mixed(x, y):
        positive_bump = 10.0 * np.exp(
            -((x - 2.0) ** 2 + (y - 2.0) ** 2) / (2 * 0.7**2)
        )
        negative_bump = -10.0 * np.exp(
            -((x - 6.0) ** 2 + (y - 6.0) ** 2) / (2 * 0.7**2)
        )
        return positive_bump + negative_bump

    surface = surface_from_function(mixed)
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)

    converged = [c for c in classified if c.status == "converged"]

    maximum_point = closest_to(converged, 2.0, 2.0)
    minimum_point = closest_to(converged, 6.0, 6.0)

    assert maximum_point.classification == "maximum"
    assert minimum_point.classification == "minimum"
    assert maximum_point is not minimum_point


# ---------------------------------------------------------
# Determinant/trace vs. eigenvalue interpretation consistency
# ---------------------------------------------------------


def test_determinant_trace_test_agrees_with_eigenvalue_classification():
    """
    Cross-checks the classic det/trace second-derivative test against
    the eigenvalue-based classification, using the SAME stored f_xx,
    f_xy, f_yx, f_yy values -- both must describe the same conclusion
    for every non-degenerate, non-unclassified point produced across
    a range of surfaces.
    """

    def bowl(x, y):
        return 3.0 * (x - 4.0) ** 2 + 7.0 * (y - 4.0) ** 2

    def bump(x, y):
        return 10.0 * np.exp(-((x - 4.0) ** 2 + (y - 4.0) ** 2) / (2 * 1.2**2))

    def saddle(x, y):
        return 5.0 * (x - 4.0) ** 2 - 5.0 * (y - 4.0) ** 2

    all_classified = []
    for surface_function in (bowl, bump, saddle):
        surface = surface_from_function(surface_function)
        candidates = locate_critical_points(surface)
        all_classified.extend(classify_critical_points(candidates, surface))

    checked_definite_or_saddle = 0

    for point in all_classified:
        if point.classification not in ("minimum", "maximum", "saddle"):
            continue

        mixed = 0.5 * (point.f_xy + point.f_yx)
        determinant = point.f_xx * point.f_yy - mixed**2
        trace = point.f_xx + point.f_yy

        if point.classification == "minimum":
            assert determinant > 0
            assert trace > 0
        elif point.classification == "maximum":
            assert determinant > 0
            assert trace < 0
        else:  # saddle
            assert determinant < 0

        checked_definite_or_saddle += 1

    assert checked_definite_or_saddle == 3


# ---------------------------------------------------------
# Determinism and stable ordering
# ---------------------------------------------------------


def test_classify_critical_points_is_deterministic():
    def mixed(x, y):
        positive_bump = 10.0 * np.exp(
            -((x - 2.0) ** 2 + (y - 2.0) ** 2) / (2 * 0.7**2)
        )
        negative_bump = -10.0 * np.exp(
            -((x - 6.0) ** 2 + (y - 6.0) ** 2) / (2 * 0.7**2)
        )
        return positive_bump + negative_bump

    surface = surface_from_function(mixed)
    candidates = locate_critical_points(surface)

    first_run = classify_critical_points(candidates, surface)
    second_run = classify_critical_points(candidates, surface)

    assert len(first_run) == len(second_run)

    for a, b in zip(first_run, second_run):
        assert a.x == b.x
        assert a.y == b.y
        assert a.classification == b.classification
        assert a.eigenvalue_min == b.eigenvalue_min
        assert a.eigenvalue_max == b.eigenvalue_max


def test_classify_critical_points_preserves_input_order():
    def bowl(x, y):
        return 3.0 * (x - 4.0) ** 2 + 7.0 * (y - 4.0) ** 2

    surface = surface_from_function(bowl)
    candidates = locate_critical_points(surface)  # already sorted by (y, x)

    classified = classify_critical_points(candidates, surface)

    assert [(c.x, c.y) for c in classified] == [
        (candidate.x, candidate.y) for candidate in candidates
    ]


# ---------------------------------------------------------
# No mutation of candidates or surface
# ---------------------------------------------------------


def test_classify_critical_points_does_not_mutate_candidates_or_surface():
    def bowl(x, y):
        return 3.0 * (x - 4.0) ** 2 + 7.0 * (y - 4.0) ** 2

    surface = surface_from_function(bowl)
    candidates = locate_critical_points(surface)
    candidates_before = copy.deepcopy(candidates)

    x_before = np.copy(surface.x)
    y_before = np.copy(surface.y)
    z_before = np.copy(surface.z)
    probe_before = surface.spline(4.0, 4.0, dx=1, dy=1)[0, 0]

    classify_critical_points(candidates, surface)

    assert candidates == candidates_before
    assert np.array_equal(surface.x, x_before)
    assert np.array_equal(surface.y, y_before)
    assert np.array_equal(surface.z, z_before)

    probe_after = surface.spline(4.0, 4.0, dx=1, dy=1)[0, 0]
    assert probe_after == pytest.approx(probe_before)
