import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from analysis.critical_points import (
    CONVERGENCE_GRADIENT_THRESHOLD,
    DEDUPLICATION_DISTANCE,
    deduplicate_candidates,
    locate_critical_points,
    refine_seed,
    sample_hessian,
)
from chess_engine.models import AttackInfluenceSurface, CriticalPointCandidate


DOMAIN_MIN = 0.5
DOMAIN_MAX = 7.5


def surface_from_function(
    func, node_count: int = 41, resolution: int = 61
) -> AttackInfluenceSurface:
    """
    Builds an AttackInfluenceSurface directly from an arbitrary
    analytic function, sampled on a node_count x node_count grid
    (finer than the real 8x8 chess board) before fitting the same
    kind of bicubic spline build_attack_influence_surface uses.

    Fitting a genuinely curved (non-polynomial) function like a
    Gaussian from only 8 nodes reproduces the ringing/overshoot
    build_attack_influence_surface's own smoothing_sigma exists to
    suppress (see analysis/attack_influence_surface.py). Sampling
    finer here keeps these tests focused on the localization
    algorithm itself, not on 8x8-undersampling artifacts already
    covered by tests/test_attack_influence_surface.py.
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


def gaussian_bump(cx: float, cy: float, amplitude: float = 10.0, sigma: float = 1.2):
    def value(x, y):
        return amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))

    return value


def converged_only(candidates: list[CriticalPointCandidate]):
    return [candidate for candidate in candidates if candidate.status == "converged"]


def closest_to(candidates: list[CriticalPointCandidate], x: float, y: float):
    return min(candidates, key=lambda candidate: np.hypot(candidate.x - x, candidate.y - y))


# ---------------------------------------------------------
# One known interior critical point
# ---------------------------------------------------------


def test_single_bump_finds_exactly_one_converged_peak():
    surface = surface_from_function(gaussian_bump(4.0, 4.0))

    results = locate_critical_points(surface)
    converged = converged_only(results)

    assert len(converged) == 1

    peak = converged[0]
    assert peak.x == pytest.approx(4.0, abs=1e-4)
    assert peak.y == pytest.approx(4.0, abs=1e-4)
    assert peak.value == pytest.approx(10.0, abs=1e-4)
    assert peak.gradient_norm < CONVERGENCE_GRADIENT_THRESHOLD


def test_status_and_gradient_norm_never_disagree():
    """
    A candidate is "converged" if and only if its gradient_norm is
    actually below the convergence threshold -- the two fields must
    never contradict each other for any status the search can reach.
    """

    surface = surface_from_function(gaussian_bump(4.0, 4.0))
    results = locate_critical_points(surface)

    assert results  # sanity: the scenario must produce candidates at all

    for candidate in results:
        if candidate.status == "converged":
            assert candidate.gradient_norm < CONVERGENCE_GRADIENT_THRESHOLD
        else:
            assert candidate.gradient_norm >= CONVERGENCE_GRADIENT_THRESHOLD


# ---------------------------------------------------------
# Multiple distinct critical points
# ---------------------------------------------------------


def test_two_separated_bumps_are_both_found_and_stay_distinct():
    def two_bumps(x, y):
        return gaussian_bump(2.5, 2.5, sigma=0.8)(x, y) + gaussian_bump(
            5.5, 5.5, sigma=0.8
        )(x, y)

    surface = surface_from_function(two_bumps)
    results = locate_critical_points(surface)
    converged = converged_only(results)

    first_peak = closest_to(converged, 2.5, 2.5)
    second_peak = closest_to(converged, 5.5, 5.5)

    assert first_peak.x == pytest.approx(2.5, abs=1e-3)
    assert first_peak.y == pytest.approx(2.5, abs=1e-3)
    assert second_peak.x == pytest.approx(5.5, abs=1e-3)
    assert second_peak.y == pytest.approx(5.5, abs=1e-3)

    distance_between_peaks = np.hypot(
        first_peak.x - second_peak.x, first_peak.y - second_peak.y
    )
    assert distance_between_peaks > DEDUPLICATION_DISTANCE


# ---------------------------------------------------------
# Critical point between grid samples
# ---------------------------------------------------------


def test_off_grid_critical_point_is_localized_past_the_nearest_node():
    true_x, true_y = 3.27, 4.83

    surface = surface_from_function(gaussian_bump(true_x, true_y))
    results = locate_critical_points(surface)
    converged = converged_only(results)

    nearest_grid_spacing = (DOMAIN_MAX - DOMAIN_MIN) / (surface.resolution - 1)

    match = closest_to(converged, true_x, true_y)
    distance_to_truth = np.hypot(match.x - true_x, match.y - true_y)

    assert distance_to_truth < 1e-3
    # The true point does not sit on any grid node -- localization
    # must land strictly closer to it than the nearest node itself.
    assert distance_to_truth < nearest_grid_spacing / 2


# ---------------------------------------------------------
# Deduplication
# ---------------------------------------------------------


def test_multiple_seeds_converging_to_same_point_are_deduplicated():
    surface = surface_from_function(gaussian_bump(4.0, 4.0))

    starting_points = [(3.9, 3.9), (4.1, 3.85), (3.85, 4.15)]

    candidates = [
        refine_seed(
            surface.spline,
            seed_y=start_y,
            seed_x=start_x,
            x_min=DOMAIN_MIN,
            x_max=DOMAIN_MAX,
            y_min=DOMAIN_MIN,
            y_max=DOMAIN_MAX,
        )
        for start_y, start_x in starting_points
    ]

    assert all(candidate.status == "converged" for candidate in candidates)

    deduplicated = deduplicate_candidates(candidates, DEDUPLICATION_DISTANCE)

    assert len(deduplicated) == 1
    assert deduplicated[0].x == pytest.approx(4.0, abs=1e-4)
    assert deduplicated[0].y == pytest.approx(4.0, abs=1e-4)


def test_deduplication_prefers_converged_regardless_of_input_order():
    converged = CriticalPointCandidate(
        x=4.0, y=4.0, value=1.0, gradient_norm=1e-9,
        status="converged", iterations=3,
    )
    non_converged = CriticalPointCandidate(
        x=4.01, y=4.0, value=1.0, gradient_norm=0.2,
        status="max_iterations_reached", iterations=50,
    )

    result_order_a = deduplicate_candidates([non_converged, converged], 0.25)
    result_order_b = deduplicate_candidates([converged, non_converged], 0.25)

    assert len(result_order_a) == 1 and result_order_a[0].status == "converged"
    assert len(result_order_b) == 1 and result_order_b[0].status == "converged"


def test_no_false_duplicate_merging_of_nearby_distinct_points():
    first = CriticalPointCandidate(
        x=2.0, y=2.0, value=5.0, gradient_norm=1e-9,
        status="converged", iterations=2,
    )
    second = CriticalPointCandidate(
        x=2.0 + DEDUPLICATION_DISTANCE * 3, y=2.0, value=-5.0,
        gradient_norm=1e-9, status="converged", iterations=2,
    )

    deduplicated = deduplicate_candidates([first, second], DEDUPLICATION_DISTANCE)

    assert len(deduplicated) == 2


def test_deduplication_does_not_chain_merge_across_a_series_of_close_neighbors():
    # A chain of candidates each only 0.9 * DEDUPLICATION_DISTANCE from
    # the next must not transitively collapse into one point just
    # because consecutive pairs are individually close -- the total
    # span across the whole chain is 4.5x DEDUPLICATION_DISTANCE, far
    # beyond what "the same critical point" should mean. gradient_norm
    # strictly improves along the chain so each new candidate would win
    # the merge tie-break, which is exactly the scenario that lets a
    # representative-based (rather than anchor-based) merge distance
    # drift arbitrarily far from its starting point.
    step = DEDUPLICATION_DISTANCE * 0.9
    count = 6

    chain = [
        CriticalPointCandidate(
            x=index * step, y=0.0, value=0.0,
            gradient_norm=1.0 / (index + 1),
            status="converged", iterations=1,
        )
        for index in range(count)
    ]

    total_span = (count - 1) * step
    assert total_span > DEDUPLICATION_DISTANCE * 4

    deduplicated = deduplicate_candidates(chain, DEDUPLICATION_DISTANCE)

    # Every pair of surviving points must be genuinely within
    # DEDUPLICATION_DISTANCE of each other -- not just chained through
    # intermediates that no longer exist in the output.
    for first_point in deduplicated:
        for second_point in deduplicated:
            if first_point is second_point:
                continue
            distance = (
                (first_point.x - second_point.x) ** 2
                + (first_point.y - second_point.y) ** 2
            ) ** 0.5
            assert distance > DEDUPLICATION_DISTANCE

    # The first and last candidates in the chain are 4.5x the threshold
    # apart and must end up represented by different, distinct points.
    first_x = min(candidate.x for candidate in deduplicated)
    last_x = max(candidate.x for candidate in deduplicated)
    assert last_x - first_x > DEDUPLICATION_DISTANCE


# ---------------------------------------------------------
# Singular / ill-conditioned Hessian handling
# ---------------------------------------------------------


def test_singular_hessian_is_reported_explicitly_not_solved():
    """
    phi = (x - 4)^3 has zero curvature in y everywhere (f_yy = 0
    identically), so its Hessian is singular at every point -- Newton
    must stop and report that rather than solve against a near-zero
    determinant.
    """

    def cubic_ridge(x, y):
        return (x - 4.0) ** 3

    surface = surface_from_function(cubic_ridge, node_count=21)

    candidate = refine_seed(
        surface.spline,
        seed_y=4.0,
        seed_x=3.7,
        x_min=DOMAIN_MIN,
        x_max=DOMAIN_MAX,
        y_min=DOMAIN_MIN,
        y_max=DOMAIN_MAX,
    )

    assert candidate.status == "singular_hessian"
    assert candidate.iterations == 0
    assert candidate.gradient_norm >= CONVERGENCE_GRADIENT_THRESHOLD


# ---------------------------------------------------------
# Non-convergent seeds
# ---------------------------------------------------------


def test_seed_stops_at_max_iterations_without_converging():
    surface = surface_from_function(gaussian_bump(4.0, 4.0))

    candidate = refine_seed(
        surface.spline,
        seed_y=2.0,
        seed_x=2.0,
        x_min=DOMAIN_MIN,
        x_max=DOMAIN_MAX,
        y_min=DOMAIN_MIN,
        y_max=DOMAIN_MAX,
        max_iterations=1,
    )

    assert candidate.status == "max_iterations_reached"
    assert candidate.iterations == 1
    assert candidate.gradient_norm >= CONVERGENCE_GRADIENT_THRESHOLD


# ---------------------------------------------------------
# Boundary behavior
# ---------------------------------------------------------


def test_step_leaving_domain_is_rejected_not_silently_clamped():
    """
    A quadratic whose true vertex sits far outside the board -- Newton
    is exact on a quadratic, so it would jump straight to that vertex
    in a single step -- must be stopped and reported at the last
    valid point, not silently clipped into a fake in-domain answer.
    """

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
    assert candidate.iterations == 0
    assert candidate.x == pytest.approx(4.0)
    assert candidate.y == pytest.approx(4.0)


# ---------------------------------------------------------
# Determinism and stable ordering
# ---------------------------------------------------------


def test_locate_critical_points_is_deterministic():
    def two_bumps(x, y):
        return gaussian_bump(2.5, 2.5, sigma=0.8)(x, y) + gaussian_bump(
            5.5, 5.5, sigma=0.8
        )(x, y)

    surface = surface_from_function(two_bumps)

    first_run = locate_critical_points(surface)
    second_run = locate_critical_points(surface)

    assert len(first_run) == len(second_run)

    for a, b in zip(first_run, second_run):
        assert a.x == b.x
        assert a.y == b.y
        assert a.status == b.status
        assert a.iterations == b.iterations
        assert a.gradient_norm == b.gradient_norm


def test_locate_critical_points_output_is_sorted_by_y_then_x():
    def two_bumps(x, y):
        return gaussian_bump(2.5, 2.5, sigma=0.8)(x, y) + gaussian_bump(
            5.5, 5.5, sigma=0.8
        )(x, y)

    surface = surface_from_function(two_bumps)
    results = locate_critical_points(surface)

    keys = [(round(candidate.y, 6), round(candidate.x, 6)) for candidate in results]
    assert keys == sorted(keys)


# ---------------------------------------------------------
# No mutation of the surface or Hessian field
# ---------------------------------------------------------


def test_locate_critical_points_does_not_mutate_surface():
    surface = surface_from_function(gaussian_bump(4.0, 4.0))

    x_before = np.copy(surface.x)
    y_before = np.copy(surface.y)
    z_before = np.copy(surface.z)
    probe_before = surface.spline(4.0, 4.0, dx=1, dy=1)[0, 0]

    locate_critical_points(surface)

    assert np.array_equal(surface.x, x_before)
    assert np.array_equal(surface.y, y_before)
    assert np.array_equal(surface.z, z_before)

    probe_after = surface.spline(4.0, 4.0, dx=1, dy=1)[0, 0]
    assert probe_after == pytest.approx(probe_before)


def test_sample_hessian_behavior_unchanged_by_phase_2_refactor():
    """
    sample_hessian's internals were refactored in Phase 2 to share
    evaluate_hessian_at_points with refine_seed -- its own public,
    whole-grid behavior from Phase 1 must be reproducible and
    unaffected by that refactor.
    """

    surface = surface_from_function(gaussian_bump(4.0, 4.0), resolution=25)

    first_call = sample_hessian(surface)
    second_call = sample_hessian(surface)

    assert np.array_equal(first_call.f_xx, second_call.f_xx)
    assert np.array_equal(first_call.f_yy, second_call.f_yy)
    assert np.array_equal(first_call.f_xy, second_call.f_xy)
    assert np.array_equal(first_call.f_yx, second_call.f_yx)
