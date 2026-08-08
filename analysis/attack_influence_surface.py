import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

from chess_engine.models import AttackInfluenceField, AttackInfluenceSurface


# Резолюция на гъстата мрежа, върху която се
# оценява напаснатата повърхност.
DEFAULT_RESOLUTION = 200

# Piece-blocking прави потенциала генуинно прекъснат между
# съседни клетки (напр. отворена линия на дама). Точна bicubic
# интерполация (kx=ky=3, s=0) звъни (overshoot) около такива
# скокове — производната ѝ там може да е 3-4x по-голяма от
# истинската. Лек Gaussian pre-filter потушава този ефект,
# преди да напаснем сплайна.
DEFAULT_SMOOTHING_SIGMA = 0.6


def build_attack_influence_surface(
    attack_influence_field: AttackInfluenceField,
    resolution: int = DEFAULT_RESOLUTION,
    smoothing_sigma: float = DEFAULT_SMOOTHING_SIGMA,
) -> AttackInfluenceSurface:
    """
    Напасва bicubic сплайн повърхност през 8x8 матрицата на
    атаковото влияние.

    Дискретното поле е известно само в 64 точки — центровете
    на клетките. Тази функция изгражда действително непрекъсната
    апроксимация φ̂(x, y), която е математическата предпоставка
    за всяка визуализация от continuous домейна (equipotential
    линии, дивергенция, критични точки). Без нея методите работят
    върху nearest-neighbor стъпаловидна функция, а не върху
    непрекъснато поле.

    Координатите съвпадат с екранната конвенция, използвана от
    board_plot/attack_influence_plot/gradient_plot: x = колона (файл),
    y = ред (0 = rank 8 най-отгоре).
    """

    matrix = np.array(
        attack_influence_field.matrix,
        dtype=float,
    )

    if smoothing_sigma > 0:
        matrix = gaussian_filter(
            matrix,
            sigma=smoothing_sigma,
            mode="nearest",
        )

    row_coords = np.arange(8) + 0.5
    column_coords = np.arange(8) + 0.5

    spline = RectBivariateSpline(
        row_coords,
        column_coords,
        matrix,
        kx=3,
        ky=3,
    )

    row_fine = np.linspace(
        row_coords[0],
        row_coords[-1],
        resolution,
    )

    column_fine = np.linspace(
        column_coords[0],
        column_coords[-1],
        resolution,
    )

    z = spline(row_fine, column_fine)

    return AttackInfluenceSurface(
        x=column_fine,
        y=row_fine,
        z=z,
        resolution=resolution,
        spline=spline,
    )


def sample_gradient(
    surface: AttackInfluenceSurface,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Оценява аналитичния градиент на напаснатата повърхност
    върху нейната гъста мрежа.

    Връща (delta_x, delta_y) в същата екранна конвенция като
    surface.x/surface.y — за разлика от analysis.gradient_field,
    тук няма обръщане на Y, защото не минаваме през
    математическата (bottom-up) конвенция.
    """

    spline = surface.spline

    delta_column = spline(surface.y, surface.x, dx=0, dy=1)
    delta_row = spline(surface.y, surface.x, dx=1, dy=0)

    return delta_column, delta_row
