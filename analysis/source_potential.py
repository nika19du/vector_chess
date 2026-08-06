import chess
import numpy as np

from analysis.geometry import square_to_plot_coords
from chess_engine.models import SourceField, SourcePotentialSurface


# Резолюция на гъстата мрежа, върху която се
# оценява Φ_source.
DEFAULT_RESOLUTION = 200

DEFAULT_KERNEL_PARAMS: dict[str, dict[str, float]] = {
    "gaussian": {"sigma": 1.0},
    "inverse_distance": {"epsilon": 1.0},
}


def gaussian_kernel(
    r: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    K(r) = exp(-r^2 / (2 * sigma^2))

    Гладко, без сингулярност при r=0 (K(0)=1), бърз локален
    спад (~3*sigma ефективна опора).
    """

    return np.exp(-(r ** 2) / (2 * sigma ** 2))


def softened_inverse_distance_kernel(
    r: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """
    K(r) = 1 / sqrt(1 + (r / epsilon)^2)

    Plummer-softened обратно разстояние: ограничено при r=0
    (K(0)=1), но с по-дълга опашка от Гаусиана — спада като
    epsilon/r за r >> epsilon.
    """

    return 1.0 / np.sqrt(1.0 + (r / epsilon) ** 2)


KERNELS = {
    "gaussian": gaussian_kernel,
    "inverse_distance": softened_inverse_distance_kernel,
}


def extract_source_points(
    source_field: SourceField,
) -> list[tuple[float, float, float]]:
    """
    Извлича (x, y, q) за всяка непразна клетка на ρ, в същите
    екранни координати като останалата визуализация
    (виж analysis.geometry.square_to_plot_coords; x = файл + 0.5,
    y = плоскостно y на клетката + 0.5).

    Никакво позоваване на board.attackers() или геометрия на
    ходовете — само позиция и знакова маса.
    """

    points: list[tuple[float, float, float]] = []

    for cell in source_field.cells:
        if cell.difference == 0.0:
            continue

        square = chess.parse_square(cell.square)

        plot_x, plot_y = square_to_plot_coords(square)
        x = plot_x + 0.5
        y = plot_y + 0.5

        points.append((x, y, cell.difference))

    return points


def resolve_kernel(
    kernel: str,
    kernel_params: dict[str, float] | None,
):
    """
    Валидира името на ядрото и допълва подадените параметри
    с подразбиращите се стойности.

    Връща (kernel_function, resolved_params).
    """

    if kernel not in KERNELS:
        raise ValueError(
            f"Unknown kernel '{kernel}'. "
            f"Available kernels: {sorted(KERNELS)}."
        )

    resolved_params = dict(
        DEFAULT_KERNEL_PARAMS[kernel]
    )

    if kernel_params:
        resolved_params.update(kernel_params)

    return KERNELS[kernel], resolved_params


def evaluate_source_potential(
    source_field: SourceField,
    x: float,
    y: float,
    kernel: str = "gaussian",
    kernel_params: dict[str, float] | None = None,
) -> float:
    """
    Оценява Φ_source в произволна точка (x, y) — без да минава
    през гъстата мрежа. Използва се за точкови проверки
    (симетрия, монотонност) там, където мрежовото резолюция
    би замъглила резултата.
    """

    kernel_function, resolved_params = resolve_kernel(
        kernel,
        kernel_params,
    )

    total = 0.0

    for source_x, source_y, mass in extract_source_points(
        source_field
    ):
        distance = (
            (x - source_x) ** 2
            + (y - source_y) ** 2
        ) ** 0.5

        total += mass * kernel_function(
            distance,
            **resolved_params,
        )

    return total


def build_source_potential_surface(
    source_field: SourceField,
    kernel: str = "gaussian",
    kernel_params: dict[str, float] | None = None,
    resolution: int = DEFAULT_RESOLUTION,
) -> SourcePotentialSurface:
    """
    Генерира Φ_source(x, y) чрез директно сумиране на ρ с
    decay kernel — НЕ чрез сплайн интерполация през 64 точки.

    Φ_source(x, y) = Σ_p q(p) * K(|(x,y) - (x_p, y_p)|)

    Няма blocking, няма ray-casting, няма зависимост от
    board.attackers(): само позицията и знаковата маса на
    всяка фигура.
    """

    kernel_function, resolved_params = resolve_kernel(
        kernel,
        kernel_params,
    )

    x_fine = np.linspace(0.5, 7.5, resolution)
    y_fine = np.linspace(0.5, 7.5, resolution)

    grid_y, grid_x = np.meshgrid(
        y_fine,
        x_fine,
        indexing="ij",
    )

    z = np.zeros_like(grid_x)

    for source_x, source_y, mass in extract_source_points(
        source_field
    ):
        distance = np.sqrt(
            (grid_x - source_x) ** 2
            + (grid_y - source_y) ** 2
        )

        z += mass * kernel_function(
            distance,
            **resolved_params,
        )

    return SourcePotentialSurface(
        x=x_fine,
        y=y_fine,
        z=z,
        resolution=resolution,
        kernel_name=kernel,
        kernel_params=resolved_params,
    )
