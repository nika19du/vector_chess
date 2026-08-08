import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from analysis.critical_points import sample_hessian
from chess_engine.models import AttackInfluenceSurface


# ---------------------------------------------------------
# Synthetic analytic surfaces
# ---------------------------------------------------------
#
# phi(x, y) = a*(x-4)^2 + b*(y-4)^2 + c*(x-4)*(y-4)
#
# is a pure quadratic form centered on the board. Its true Hessian is
# the constant matrix [[2a, c], [c, 2b]] everywhere -- a closed-form
# reference no numerical routine needs to approximate, exactly the
# kind of case docs/mathematics.md Section 9 asks Phase 1 to validate
# against before any real chess position is involved.


def _quadratic_surface(
    a: float, b: float, c: float, resolution: int = 41
) -> AttackInfluenceSurface:
    row_coords = np.arange(8) + 0.5
    column_coords = np.arange(8) + 0.5

    def value(x: float, y: float) -> float:
        return a * (x - 4) ** 2 + b * (y - 4) ** 2 + c * (x - 4) * (y - 4)

    matrix = np.array(
        [
            [value(column, row) for column in column_coords]
            for row in row_coords
        ]
    )

    spline = RectBivariateSpline(row_coords, column_coords, matrix, kx=3, ky=3)

    row_fine = np.linspace(row_coords[0], row_coords[-1], resolution)
    column_fine = np.linspace(column_coords[0], column_coords[-1], resolution)

    z = spline(row_fine, column_fine)

    return AttackInfluenceSurface(
        x=column_fine,
        y=row_fine,
        z=z,
        resolution=resolution,
        spline=spline,
    )


# ---------------------------------------------------------
# Definiteness cases
# ---------------------------------------------------------


def test_positive_definite_hessian_on_convex_bowl():
    # phi = 3(x-4)^2 + 7(y-4)^2 -- a bowl opening upward on both axes.
    surface = _quadratic_surface(a=3.0, b=7.0, c=0.0)

    hessian = sample_hessian(surface)

    assert np.allclose(hessian.f_xx, 6.0, atol=1e-6)
    assert np.allclose(hessian.f_yy, 14.0, atol=1e-6)

    determinant = hessian.f_xx * hessian.f_yy - hessian.f_xy * hessian.f_yx
    assert np.all(determinant > 0)
    assert np.all(hessian.f_xx > 0)


def test_negative_definite_hessian_on_concave_bowl():
    # phi = -3(x-4)^2 - 7(y-4)^2 -- a dome opening downward on both axes.
    surface = _quadratic_surface(a=-3.0, b=-7.0, c=0.0)

    hessian = sample_hessian(surface)

    assert np.allclose(hessian.f_xx, -6.0, atol=1e-6)
    assert np.allclose(hessian.f_yy, -14.0, atol=1e-6)

    determinant = hessian.f_xx * hessian.f_yy - hessian.f_xy * hessian.f_yx
    assert np.all(determinant > 0)
    assert np.all(hessian.f_xx < 0)


def test_indefinite_hessian_on_saddle_surface():
    # phi = 5(x-4)^2 - 5(y-4)^2 -- rises along x, falls along y: a saddle.
    surface = _quadratic_surface(a=5.0, b=-5.0, c=0.0)

    hessian = sample_hessian(surface)

    assert np.allclose(hessian.f_xx, 10.0, atol=1e-6)
    assert np.allclose(hessian.f_yy, -10.0, atol=1e-6)

    determinant = hessian.f_xx * hessian.f_yy - hessian.f_xy * hessian.f_yx
    assert np.all(determinant < 0)


# ---------------------------------------------------------
# Mixed derivative
# ---------------------------------------------------------


def test_zero_mixed_derivative_for_axis_aligned_quadratic():
    # No cross term -- the two axes are independent, f_xy must vanish.
    surface = _quadratic_surface(a=4.0, b=2.0, c=0.0)

    hessian = sample_hessian(surface)

    assert np.allclose(hessian.f_xy, 0.0, atol=1e-6)
    assert np.allclose(hessian.f_yx, 0.0, atol=1e-6)


def test_nonzero_mixed_derivative_for_cross_term_quadratic():
    surface = _quadratic_surface(a=1.0, b=1.0, c=2.5)

    hessian = sample_hessian(surface)

    assert np.allclose(hessian.f_xy, 2.5, atol=1e-6)
    assert np.allclose(hessian.f_yx, 2.5, atol=1e-6)


# ---------------------------------------------------------
# f_xy vs f_yx symmetry -- the actual numerical cross-check,
# since the two are computed via independent finite-difference paths.
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,c",
    [
        (1.0, 1.0, 0.0),
        (3.0, 7.0, 1.5),
        (5.0, -5.0, -2.0),
        (-2.0, -6.0, 4.0),
    ],
)
def test_mixed_partials_agree_within_tolerance(a, b, c):
    surface = _quadratic_surface(a=a, b=b, c=c)

    hessian = sample_hessian(surface)

    assert np.allclose(hessian.f_xy, hessian.f_yx, atol=1e-6)


# ---------------------------------------------------------
# Constant Hessian across a quadratic surface
# ---------------------------------------------------------


def test_hessian_is_constant_across_quadratic_surface():
    surface = _quadratic_surface(a=2.0, b=5.0, c=1.0)

    hessian = sample_hessian(surface)

    assert np.std(hessian.f_xx) < 1e-6
    assert np.std(hessian.f_yy) < 1e-6
    assert np.std(hessian.f_xy) < 1e-6
    assert np.std(hessian.f_yx) < 1e-6


# ---------------------------------------------------------
# Coordinate handling
# ---------------------------------------------------------


def test_curvature_axes_are_not_swapped():
    """
    Uses a strongly asymmetric quadratic (a far from b) to catch an
    accidental transpose between the file/x axis and the rank/y axis
    in sample_hessian's spline calls -- f_xx must track the x-axis
    coefficient and f_yy the y-axis coefficient, not the reverse.
    """

    surface = _quadratic_surface(a=1.0, b=9.0, c=0.0)

    hessian = sample_hessian(surface)

    assert np.allclose(hessian.f_xx, 2.0, atol=1e-6)
    assert np.allclose(hessian.f_yy, 18.0, atol=1e-6)
    assert not np.allclose(hessian.f_xx, 18.0, atol=1e-3)
    assert not np.allclose(hessian.f_yy, 2.0, atol=1e-3)


def test_hessian_shape_matches_surface_grid():
    surface = _quadratic_surface(a=1.0, b=1.0, c=0.0, resolution=25)

    hessian = sample_hessian(surface)

    assert hessian.f_xx.shape == (25, 25)
    assert hessian.f_yy.shape == (25, 25)
    assert hessian.f_xy.shape == (25, 25)
    assert hessian.f_yx.shape == (25, 25)


# ---------------------------------------------------------
# No mutation of AttackInfluenceSurface
# ---------------------------------------------------------


def test_sample_hessian_does_not_mutate_surface():
    surface = _quadratic_surface(a=3.0, b=4.0, c=1.0)

    x_before = np.copy(surface.x)
    y_before = np.copy(surface.y)
    z_before = np.copy(surface.z)

    probe_point = surface.spline(4.0, 4.0, dx=1, dy=1)[0, 0]

    sample_hessian(surface)

    assert np.array_equal(surface.x, x_before)
    assert np.array_equal(surface.y, y_before)
    assert np.array_equal(surface.z, z_before)

    probe_point_after = surface.spline(4.0, 4.0, dx=1, dy=1)[0, 0]
    assert probe_point_after == pytest.approx(probe_point)
