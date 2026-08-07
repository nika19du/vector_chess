import copy
import math

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from analysis.critical_points import (
    classify_critical_points,
    compute_symmetric_eigenvalues,
    locate_critical_points,
)
from analysis.ridge_valley import (
    ALIGNMENT_RELATIVE_TOLERANCE,
    SELF_INTERSECTION_DISTANCE,
    SELF_INTERSECTION_MIN_STEP_GAP,
    STEP_SIZE,
    _canonical_direction_sign,
    _consistent_direction,
    check_self_intersection,
    compute_symmetric_eigenvectors,
    locate_ridge_valley_chains,
)
from chess_engine.models import AttackInfluenceSurface, ClassifiedCriticalPoint


DOMAIN_MIN = 0.5
DOMAIN_MAX = 7.5


# ---------------------------------------------------------
# Shared helpers (same construction as
# tests/test_critical_point_localization.py and
# tests/test_critical_point_classification.py)
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


def anisotropic_gaussian_bump(
    cx: float,
    cy: float,
    amplitude: float = 10.0,
    sigma_x: float = 2.0,
    sigma_y: float = 0.5,
):
    """
    A Gaussian bump stretched along x (sigma_x > sigma_y): curvature
    is strongly negative across y (the cross-ridge direction) and
    only weakly negative along x near the peak, decaying slowly --
    the axis-aligned reference case docs/mathematics.md Section 10
    asks for, with a known ridge running along y = cy through the
    peak, so no rotated closed form needs to be hand-verified.
    """

    def value(x, y):
        return amplitude * np.exp(
            -(
                (x - cx) ** 2 / (2 * sigma_x**2)
                + (y - cy) ** 2 / (2 * sigma_y**2)
            )
        )

    return value


def anchors_by_classification(
    surface: AttackInfluenceSurface, classification: str
) -> tuple[list[ClassifiedCriticalPoint], list[ClassifiedCriticalPoint]]:
    """Returns (matching_anchors, all_classified_points)."""

    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)

    matching = [
        point
        for point in classified
        if point.status == "converged" and point.classification == classification
    ]

    return matching, classified


def make_fake_other_point(x: float, y: float) -> ClassifiedCriticalPoint:
    """
    A ClassifiedCriticalPoint standing in for an unrelated, already-
    classified critical point elsewhere on the surface -- only its
    x, y are read by the ridge/valley reconnection check, so the
    other fields carry plausible but otherwise unused values.
    """

    return ClassifiedCriticalPoint(
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
        classification="saddle",
    )


# ---------------------------------------------------------
# Closed-form eigenvector extraction
# ---------------------------------------------------------


def test_eigenvector_recovery_axis_aligned_hessian():
    # f_xx=3, f_yy=7, mixed=0 -- already diagonal: e_max must align
    # with the y-axis (eigenvalue_max=7=f_yy), e_min with the x-axis.
    eigenvectors = compute_symmetric_eigenvectors(f_xx=3.0, mixed=0.0, f_yy=7.0)
    assert eigenvectors is not None

    e_max, e_min = eigenvectors

    assert abs(e_max[0]) == pytest.approx(0.0, abs=1e-9)
    assert abs(e_max[1]) == pytest.approx(1.0, abs=1e-9)
    assert abs(e_min[0]) == pytest.approx(1.0, abs=1e-9)
    assert abs(e_min[1]) == pytest.approx(0.0, abs=1e-9)


def test_eigenvector_recovery_matches_eigenvalue_equation_on_rotated_hessian():
    # A generic, non-axis-aligned Hessian -- verified generically via
    # H @ e = lambda * e rather than by hand-deriving the rotation
    # angle, which is robust to any algebraic slip in the derivation.
    f_xx, mixed, f_yy = 5.0, 2.0, 1.0

    eigenvalue_max, eigenvalue_min, _determinant, _trace = compute_symmetric_eigenvalues(
        f_xx, mixed, f_yy
    )
    eigenvectors = compute_symmetric_eigenvectors(f_xx, mixed, f_yy)
    assert eigenvectors is not None

    e_max, e_min = eigenvectors
    hessian = np.array([[f_xx, mixed], [mixed, f_yy]])

    for eigenvalue, eigenvector in ((eigenvalue_max, e_max), (eigenvalue_min, e_min)):
        vector = np.array(eigenvector)
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-9)

        product = hessian @ vector
        assert product == pytest.approx(eigenvalue * vector, abs=1e-9)

    # Eigenvectors of a symmetric matrix with distinct eigenvalues are
    # orthogonal.
    assert np.dot(e_max, e_min) == pytest.approx(0.0, abs=1e-9)


def test_eigenvector_extraction_returns_none_on_isotropic_point():
    # f_xx == f_yy and mixed == 0: every direction is an eigenvector,
    # the ridge/valley direction is genuinely undefined here.
    assert compute_symmetric_eigenvectors(f_xx=4.0, mixed=0.0, f_yy=4.0) is None


# ---------------------------------------------------------
# Sign-consistent eigenvector continuation
# ---------------------------------------------------------


def test_consistent_direction_keeps_vector_already_pointing_forward():
    result = _consistent_direction((1.0, 0.0), previous_direction=(1.0, 0.0))
    assert result == (1.0, 0.0)


def test_consistent_direction_flips_vector_pointing_backward():
    result = _consistent_direction((-1.0, 0.0), previous_direction=(1.0, 0.0))
    assert result == (1.0, 0.0)


def test_consistent_direction_flips_on_perpendicular_axis_too():
    result = _consistent_direction((0.0, 1.0), previous_direction=(0.0, -1.0))
    assert result == (0.0, -1.0)


def test_canonical_direction_sign_prefers_positive_x():
    assert _canonical_direction_sign((-1.0, 0.5)) == (1.0, -0.5)
    assert _canonical_direction_sign((1.0, -0.5)) == (1.0, -0.5)


def test_canonical_direction_sign_falls_back_to_y_when_x_is_zero():
    assert _canonical_direction_sign((0.0, -1.0)) == (0.0, 1.0)
    assert _canonical_direction_sign((0.0, 1.0)) == (0.0, 1.0)


# ---------------------------------------------------------
# Self-intersection / near-loop detection
# ---------------------------------------------------------


def test_check_self_intersection_detects_return_to_old_point():
    visited = [(0.0, 0.0)] + [(float(i) * 0.5, 0.0) for i in range(1, 30)]
    # 30 points; the cutoff excludes only the most recent
    # SELF_INTERSECTION_MIN_STEP_GAP entries, so (0.0, 0.0) is still a
    # valid comparison target.
    assert check_self_intersection(0.02, 0.0, visited) is True


def test_check_self_intersection_ignores_the_recent_window():
    visited = [(0.0, 0.0)] + [(float(i) * 0.5, 0.0) for i in range(1, 30)]
    recent_point = visited[-1]

    # Geometrically identical to a recent point -- this is normal path
    # continuation, not a loop, and must not be flagged.
    result = check_self_intersection(recent_point[0], recent_point[1], visited)
    assert result is False


def test_check_self_intersection_false_when_history_too_short():
    visited = [(0.0, 0.0), (0.05, 0.0), (0.1, 0.0)]
    assert len(visited) <= SELF_INTERSECTION_MIN_STEP_GAP
    assert check_self_intersection(0.0, 0.0, visited) is False


def test_self_intersection_distance_is_two_step_sizes():
    assert SELF_INTERSECTION_DISTANCE == pytest.approx(2 * STEP_SIZE)


# ---------------------------------------------------------
# One analytic ridge / one analytic valley
# ---------------------------------------------------------


def test_analytic_ridge_runs_along_the_stretched_axis():
    surface = surface_from_function(
        anisotropic_gaussian_bump(4.0, 4.0, amplitude=10.0, sigma_x=2.0, sigma_y=0.5)
    )

    maxima, classified = anchors_by_classification(surface, "maximum")
    assert len(maxima) == 1

    chains = locate_ridge_valley_chains(surface, classified, kind="ridge")
    assert len(chains) == 1

    chain = chains[0]
    assert chain.kind == "ridge"
    assert chain.anchor is maxima[0]
    # A genuine ridge, not just the anchor by itself.
    assert len(chain.points) > 10

    # The ridge is expected to hug y = 4.0 (the stretched axis) while
    # spreading out in x on both sides of the peak.
    for point in chain.points:
        assert point.y == pytest.approx(4.0, abs=0.05)

    xs = [point.x for point in chain.points]
    assert min(xs) < 3.0
    assert max(xs) > 5.0

    # Every retained point must satisfy the ridge sign condition.
    for point in chain.points:
        assert point.eigenvalue_cross < 0.0

    assert chain.start_status in (
        "left_domain",
        "weak_curvature",
        "max_steps_reached",
    )
    assert chain.end_status in (
        "left_domain",
        "weak_curvature",
        "max_steps_reached",
    )


def test_analytic_valley_runs_along_the_stretched_axis():
    surface = surface_from_function(
        anisotropic_gaussian_bump(4.0, 4.0, amplitude=-10.0, sigma_x=2.0, sigma_y=0.5)
    )

    minima, classified = anchors_by_classification(surface, "minimum")
    assert len(minima) == 1

    chains = locate_ridge_valley_chains(surface, classified, kind="valley")
    assert len(chains) == 1

    chain = chains[0]
    assert chain.kind == "valley"
    assert len(chain.points) > 10

    for point in chain.points:
        assert point.y == pytest.approx(4.0, abs=0.05)

    xs = [point.x for point in chain.points]
    assert min(xs) < 3.0
    assert max(xs) > 5.0

    for point in chain.points:
        assert point.eigenvalue_cross > 0.0


# ---------------------------------------------------------
# Off-grid coordinate preservation and bidirectional tracing
# ---------------------------------------------------------


def test_ridge_preserves_true_continuous_coordinates_off_grid():
    # An off-grid peak center -- not aligned with any 0.5-spaced board
    # cell boundary.
    surface = surface_from_function(
        anisotropic_gaussian_bump(4.13, 3.87, amplitude=10.0, sigma_x=2.0, sigma_y=0.5)
    )

    maxima, classified = anchors_by_classification(surface, "maximum")
    chains = locate_ridge_valley_chains(surface, classified, kind="ridge")
    chain = chains[0]

    assert chain.anchor.x == pytest.approx(4.13, abs=1e-3)
    assert chain.anchor.y == pytest.approx(3.87, abs=1e-3)

    # None of the traced x-coordinates should land on a 0.5-spaced
    # board-cell boundary -- the chain is not snapped to any lattice.
    for point in chain.points:
        nearest_half_step = round(point.x * 2) / 2
        assert abs(point.x - nearest_half_step) > 1e-6

    # Consecutive points (away from the anchor, where step direction
    # is unambiguous) are separated by exactly one marching step.
    for first, second in zip(chain.points, chain.points[1:]):
        distance = math.hypot(second.x - first.x, second.y - first.y)
        assert distance == pytest.approx(STEP_SIZE, abs=1e-6)


def test_ridge_traces_bidirectionally_from_the_anchor():
    surface = surface_from_function(
        anisotropic_gaussian_bump(4.0, 4.0, amplitude=10.0, sigma_x=2.0, sigma_y=0.5)
    )

    maxima, classified = anchors_by_classification(surface, "maximum")
    chain = locate_ridge_valley_chains(surface, classified, kind="ridge")[0]

    anchor_x = chain.anchor.x
    points_before = [p for p in chain.points if p.x < anchor_x - STEP_SIZE / 2]
    points_after = [p for p in chain.points if p.x > anchor_x + STEP_SIZE / 2]

    assert len(points_before) > 5
    assert len(points_after) > 5

    # The anchor itself must appear exactly once, at the seam between
    # the two directions.
    anchor_matches = [
        p for p in chain.points if p.x == anchor_x and p.y == chain.anchor.y
    ]
    assert len(anchor_matches) == 1


# ---------------------------------------------------------
# Determinism and no mutation
# ---------------------------------------------------------


def test_locate_ridge_valley_chains_is_deterministic():
    surface = surface_from_function(
        anisotropic_gaussian_bump(4.0, 4.0, amplitude=10.0, sigma_x=2.0, sigma_y=0.5)
    )
    _maxima, classified = anchors_by_classification(surface, "maximum")

    first_run = locate_ridge_valley_chains(surface, classified, kind="ridge")
    second_run = locate_ridge_valley_chains(surface, classified, kind="ridge")

    assert len(first_run) == len(second_run)

    for chain_a, chain_b in zip(first_run, second_run):
        assert chain_a.start_status == chain_b.start_status
        assert chain_a.end_status == chain_b.end_status
        assert len(chain_a.points) == len(chain_b.points)

        for point_a, point_b in zip(chain_a.points, chain_b.points):
            assert point_a.x == point_b.x
            assert point_a.y == point_b.y
            assert point_a.eigenvalue_cross == point_b.eigenvalue_cross


def test_locate_ridge_valley_chains_does_not_mutate_inputs():
    surface = surface_from_function(
        anisotropic_gaussian_bump(4.0, 4.0, amplitude=10.0, sigma_x=2.0, sigma_y=0.5)
    )
    _maxima, classified = anchors_by_classification(surface, "maximum")

    x_before = np.copy(surface.x)
    y_before = np.copy(surface.y)
    z_before = np.copy(surface.z)
    classified_before = copy.deepcopy(classified)

    locate_ridge_valley_chains(surface, classified, kind="ridge")

    assert np.array_equal(surface.x, x_before)
    assert np.array_equal(surface.y, y_before)
    assert np.array_equal(surface.z, z_before)

    for point_after, point_before in zip(classified, classified_before):
        assert point_after == point_before


# ---------------------------------------------------------
# Termination: boundary, low curvature, reconnection
# ---------------------------------------------------------


def test_ridge_terminates_on_domain_exit_near_the_board_edge():
    # A broad ridge (large sigma_x keeps cross-curvature healthy far
    # from the peak) centered close to the right edge of the board --
    # marching toward that edge must hit "left_domain" well before
    # curvature has any chance to collapse.
    surface = surface_from_function(
        anisotropic_gaussian_bump(7.2, 4.0, amplitude=10.0, sigma_x=2.0, sigma_y=0.5)
    )

    maxima, classified = anchors_by_classification(surface, "maximum")
    assert len(maxima) == 1

    chain = locate_ridge_valley_chains(surface, classified, kind="ridge")[0]

    assert "left_domain" in (chain.start_status, chain.end_status)


def test_ridge_terminates_on_weak_curvature_far_from_a_narrow_peak():
    # A narrow, low-amplitude bump: cross-curvature decays below
    # CROSS_EIGENVALUE_COLLAPSE_THRESHOLD well inside the board,
    # before either direction reaches the domain edge.
    surface = surface_from_function(
        anisotropic_gaussian_bump(4.0, 4.0, amplitude=1.0, sigma_x=0.5, sigma_y=0.3),
        node_count=81,
        resolution=121,
    )

    maxima, classified = anchors_by_classification(surface, "maximum")
    assert len(maxima) == 1

    chain = locate_ridge_valley_chains(surface, classified, kind="ridge")[0]

    assert chain.start_status == "weak_curvature"
    assert chain.end_status == "weak_curvature"


def test_ridge_terminates_on_reconnection_to_another_critical_point():
    surface = surface_from_function(
        anisotropic_gaussian_bump(4.0, 4.0, amplitude=10.0, sigma_x=2.0, sigma_y=0.5)
    )

    maxima, classified = anchors_by_classification(surface, "maximum")
    assert len(maxima) == 1
    anchor = maxima[0]

    other_ahead = make_fake_other_point(anchor.x + 1.0, anchor.y)
    other_behind = make_fake_other_point(anchor.x - 1.0, anchor.y)

    augmented = classified + [other_ahead, other_behind]

    chain = locate_ridge_valley_chains(surface, augmented, kind="ridge")[0]

    assert chain.start_status == "reached_critical_point"
    assert chain.end_status == "reached_critical_point"

    # The chain must stop short of (not past) the reconnection points.
    for point in chain.points:
        assert point.x < other_ahead.x
        assert point.x > other_behind.x


# ---------------------------------------------------------
# Ridge / valley symmetry on sign-inverted surfaces
# ---------------------------------------------------------


def test_ridge_and_valley_are_symmetric_under_surface_negation():
    positive_surface = surface_from_function(
        anisotropic_gaussian_bump(4.0, 4.0, amplitude=6.0, sigma_x=2.0, sigma_y=0.5)
    )
    negative_surface = surface_from_function(
        anisotropic_gaussian_bump(4.0, 4.0, amplitude=-6.0, sigma_x=2.0, sigma_y=0.5)
    )

    maxima, classified_positive = anchors_by_classification(
        positive_surface, "maximum"
    )
    minima, classified_negative = anchors_by_classification(
        negative_surface, "minimum"
    )
    assert len(maxima) == 1
    assert len(minima) == 1

    ridge_chain = locate_ridge_valley_chains(
        positive_surface, classified_positive, kind="ridge"
    )[0]
    valley_chain = locate_ridge_valley_chains(
        negative_surface, classified_negative, kind="valley"
    )[0]

    assert ridge_chain.start_status == valley_chain.start_status
    assert ridge_chain.end_status == valley_chain.end_status
    assert len(ridge_chain.points) == len(valley_chain.points)

    for ridge_point, valley_point in zip(ridge_chain.points, valley_chain.points):
        assert ridge_point.x == pytest.approx(valley_point.x, abs=1e-6)
        assert ridge_point.y == pytest.approx(valley_point.y, abs=1e-6)
        assert ridge_point.eigenvalue_cross == pytest.approx(
            -valley_point.eigenvalue_cross, abs=1e-6
        )
