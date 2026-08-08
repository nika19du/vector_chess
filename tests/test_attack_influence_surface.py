import numpy as np
import pytest

from analysis.attack_influence_surface import (
    DEFAULT_RESOLUTION,
    build_attack_influence_surface,
    sample_gradient,
)
from chess_engine.models import AttackInfluenceField


def _field_from_matrix(matrix: list[list[float]]) -> AttackInfluenceField:
    return AttackInfluenceField(
        cells=[],
        matrix=matrix,
        total_white_attack_influence=0.0,
        total_black_attack_influence=0.0,
        balance=0.0,
        strongest_white_square=None,
        strongest_black_square=None,
    )


def _zero_matrix() -> list[list[float]]:
    return [[0.0 for _ in range(8)] for _ in range(8)]


def _constant_matrix(value: float) -> list[list[float]]:
    return [[value for _ in range(8)] for _ in range(8)]


def _gaussian_bump_matrix(amplitude: float = 10.0, sigma: float = 1.5) -> list[list[float]]:
    """
    Synthetic 8x8 matrix shaped like a single Gaussian bump centered
    exactly on the board (row_coords/column_coords midpoint = 4.0),
    independent of any chess position.

    Its unique critical point is a maximum at the board center by
    construction -- this is the reference case Phase 1's Hessian and
    critical-point machinery will later be validated against, and the
    case this file uses to pin down build_attack_influence_surface's
    and sample_gradient's existing, currently-untested behavior.
    """

    matrix: list[list[float]] = []

    for row in range(8):
        row_values: list[float] = []

        for column in range(8):
            # row_coords[i] = i + 0.5, so (row - 3.5) == row_coords[row] - 4.0.
            distance_squared = (row - 3.5) ** 2 + (column - 3.5) ** 2
            row_values.append(
                amplitude * np.exp(-distance_squared / (2 * sigma**2))
            )

        matrix.append(row_values)

    return matrix


# ---------------------------------------------------------
# build_attack_influence_surface: shape and grid contract
# ---------------------------------------------------------


def test_default_resolution_matches_module_constant():
    surface = build_attack_influence_surface(_field_from_matrix(_zero_matrix()))

    assert surface.resolution == DEFAULT_RESOLUTION
    assert surface.z.shape == (DEFAULT_RESOLUTION, DEFAULT_RESOLUTION)
    assert surface.x.shape == (DEFAULT_RESOLUTION,)
    assert surface.y.shape == (DEFAULT_RESOLUTION,)


def test_custom_resolution_is_respected():
    surface = build_attack_influence_surface(
        _field_from_matrix(_zero_matrix()), resolution=32
    )

    assert surface.resolution == 32
    assert surface.z.shape == (32, 32)


def test_grid_spans_cell_center_coordinates():
    surface = build_attack_influence_surface(_field_from_matrix(_zero_matrix()))

    assert surface.x[0] == pytest.approx(0.5)
    assert surface.x[-1] == pytest.approx(7.5)
    assert surface.y[0] == pytest.approx(0.5)
    assert surface.y[-1] == pytest.approx(7.5)


def test_empty_field_surface_is_zero_everywhere():
    surface = build_attack_influence_surface(_field_from_matrix(_zero_matrix()))

    assert np.allclose(surface.z, 0.0)


def test_constant_field_surface_is_constant():
    surface = build_attack_influence_surface(
        _field_from_matrix(_constant_matrix(3.0)), smoothing_sigma=0.0
    )

    assert np.allclose(surface.z, 3.0)


# ---------------------------------------------------------
# Interpolation correctness (no pre-smoothing)
# ---------------------------------------------------------


def test_unsmoothed_spline_exactly_reproduces_matrix_at_cell_centers():
    """
    With smoothing_sigma=0, RectBivariateSpline is fit with its default
    smoothing factor s=0, which means exact interpolation at the input
    nodes. Evaluating the fitted spline back at the original 8x8 cell
    centers must reproduce the source matrix bit-for-bit (within
    floating point tolerance) -- the surface must not silently distort
    the discrete field it was built from.
    """

    matrix = _gaussian_bump_matrix()
    field = _field_from_matrix(matrix)

    surface = build_attack_influence_surface(field, smoothing_sigma=0.0)

    row_coords = np.arange(8) + 0.5
    column_coords = np.arange(8) + 0.5

    reconstructed = surface.spline(row_coords, column_coords)

    assert np.allclose(reconstructed, np.array(matrix), atol=1e-8)


def test_smoothing_sigma_changes_the_fitted_surface():
    matrix = _gaussian_bump_matrix()
    field = _field_from_matrix(matrix)

    unsmoothed = build_attack_influence_surface(field, smoothing_sigma=0.0)
    smoothed = build_attack_influence_surface(field, smoothing_sigma=1.0)

    assert not np.allclose(unsmoothed.z, smoothed.z)
    # A Gaussian pre-filter can only ever flatten a single sharp peak,
    # never raise it above the discrete field's own maximum.
    assert smoothed.z.max() < unsmoothed.z.max()


# ---------------------------------------------------------
# Gaussian-bump reference case: maximum at the board center
# ---------------------------------------------------------


def test_gaussian_bump_surface_peaks_at_board_center():
    field = _field_from_matrix(_gaussian_bump_matrix())

    surface = build_attack_influence_surface(field, resolution=101, smoothing_sigma=0.0)

    peak_row, peak_column = np.unravel_index(np.argmax(surface.z), surface.z.shape)

    center_index = 50  # resolution=101 -> midpoint index is exact

    assert peak_row == pytest.approx(center_index, abs=1)
    assert peak_column == pytest.approx(center_index, abs=1)


def test_gaussian_bump_gradient_vanishes_at_board_center():
    field = _field_from_matrix(_gaussian_bump_matrix())

    surface = build_attack_influence_surface(field, smoothing_sigma=0.0)

    delta_x_at_center = surface.spline(4.0, 4.0, dx=0, dy=1)[0, 0]
    delta_y_at_center = surface.spline(4.0, 4.0, dx=1, dy=0)[0, 0]

    assert delta_x_at_center == pytest.approx(0.0, abs=1e-6)
    assert delta_y_at_center == pytest.approx(0.0, abs=1e-6)


def test_gaussian_bump_gradient_points_away_from_center():
    """
    On the descending flank of a bump, the surface decreases as you
    move away from the center, so the gradient component along that
    axis must be negative moving outward (positive on the near side,
    negative on the far side) -- i.e. it points back toward the peak.
    """

    field = _field_from_matrix(_gaussian_bump_matrix())
    surface = build_attack_influence_surface(field, smoothing_sigma=0.0)

    delta_x_left = surface.spline(4.0, 2.0, dx=0, dy=1)[0, 0]
    delta_x_right = surface.spline(4.0, 6.0, dx=0, dy=1)[0, 0]

    assert delta_x_left > 0
    assert delta_x_right < 0


# ---------------------------------------------------------
# sample_gradient
# ---------------------------------------------------------


def test_sample_gradient_shape_matches_surface_grid():
    surface = build_attack_influence_surface(
        _field_from_matrix(_gaussian_bump_matrix()), resolution=40
    )

    delta_x, delta_y = sample_gradient(surface)

    assert delta_x.shape == (40, 40)
    assert delta_y.shape == (40, 40)


def test_sample_gradient_is_zero_for_constant_field():
    surface = build_attack_influence_surface(
        _field_from_matrix(_constant_matrix(5.0)), smoothing_sigma=0.0
    )

    delta_x, delta_y = sample_gradient(surface)

    assert np.allclose(delta_x, 0.0, atol=1e-8)
    assert np.allclose(delta_y, 0.0, atol=1e-8)


def test_sample_gradient_is_antisymmetric_across_bump_center():
    """
    The Gaussian bump is symmetric under reflection through the board
    center, so the gradient field must be antisymmetric: the gradient
    at a point mirrored through the center is the negation of the
    gradient at the original point.
    """

    field = _field_from_matrix(_gaussian_bump_matrix())
    surface = build_attack_influence_surface(field, resolution=41, smoothing_sigma=0.0)

    delta_x, delta_y = sample_gradient(surface)

    assert np.allclose(delta_x, -np.flip(delta_x), atol=1e-6)
    assert np.allclose(delta_y, -np.flip(delta_y), atol=1e-6)


def test_unknown_field_shape_raises():
    malformed_field = _field_from_matrix([[0.0, 0.0], [0.0, 0.0]])

    with pytest.raises(Exception):
        build_attack_influence_surface(malformed_field)
