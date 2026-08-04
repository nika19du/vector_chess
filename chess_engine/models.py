from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RectBivariateSpline


@dataclass
class ChessVector:
    from_square: str
    to_square: str
    piece_name: str
    color: str
    strength: float


@dataclass
class ControlSummary:
    white_controlled_squares: int
    black_controlled_squares: int
    difference: int


@dataclass
class MoveDetails:
    move: str
    piece_name: str
    color: str
    from_square: str
    to_square: str
    is_capture: bool
    is_check: bool


@dataclass
class SquareControl:
    square: str
    white_attackers: int
    black_attackers: int
    difference: int


@dataclass
class FieldVector:
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    delta_x: float
    delta_y: float
    magnitude: float


@dataclass
class AttackVector:
    piece_name: str
    color: str

    from_square: str
    to_square: str

    start_x: int
    start_y: int

    end_x: int
    end_y: int

    delta_x: int
    delta_y: int


@dataclass
class PotentialCell:
    """
    Потенциалната стойност на едно шахматно поле.

    white_potential:
        сумата от тежестите на белите фигури,
        които влияят върху полето.

    black_potential:
        сумата от тежестите на черните фигури,
        които влияят върху полето.

    difference:
        white_potential - black_potential
    """

    square: str
    white_potential: float
    black_potential: float
    difference: float


@dataclass
class PotentialField:
    """
    Цялото потенциално поле на шахматната дъска.
    """

    cells: list[PotentialCell]

    matrix: list[list[float]]

    total_white_potential: float
    total_black_potential: float

    balance: float

    strongest_white_square: str | None
    strongest_black_square: str | None


@dataclass
class SourceCell:
    """
    Изходната маса (ρ) на едно поле — чисто occupancy,
    без атакова геометрия.

    white_mass / black_mass:
        тежестта на фигурата, ако полето е заето от
        съответния цвят, иначе 0.

    difference:
        white_mass - black_mass (нетен заряд).
    """

    square: str
    white_mass: float
    black_mass: float
    difference: float


@dataclass
class SourceField:
    """
    Дискретното ρ(x, y) — знаково разпределение на маса
    върху 64-те полета, независимо от board.attackers().
    """

    cells: list[SourceCell]

    matrix: list[list[float]]

    total_white_mass: float
    total_black_mass: float

    balance: float


@dataclass
class PotentialSurface:
    """
    Непрекъсната bicubic апроксимация на дискретното
    8x8 потенциално поле.

    x, y:
        координати на гъстата мрежа в екранни координати
        (x = файл/колона, y = ред; y=0 съответства на rank 8,
        нагоре по дъската, за да съвпада с imshow/board_plot).

    z:
        стойностите на потенциала върху гъстата мрежа,
        z.shape == (resolution, resolution).

    spline:
        напаснатата повърхност (RectBivariateSpline),
        използвана за градиент, дивергенция и критични точки.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    resolution: int

    spline: RectBivariateSpline


@dataclass
class SourcePotentialSurface:
    """
    Непрекъснатото Φ_source(x, y), генерирано чрез директно
    сумиране на ρ с decay kernel — НЕ чрез сплайн интерполация.

    За разлика от PotentialSurface, тук няма spline обект,
    защото повърхността не минава през дискретни точки чрез
    интерполация, а е директна суперпозиция на källов принос
    от всяка фигура.

    x, y, z, resolution:
        същата екранна конвенция като PotentialSurface, за да
        могат двете повърхности да се рисуват с общ код.

    kernel_name / kernel_params:
        коя ядрова функция и с какви параметри е използвана
        (напр. "gaussian", {"sigma": 1.0}), за проследимост.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    resolution: int

    kernel_name: str
    kernel_params: dict[str, float]


@dataclass
class GradientVector:
    square: str

    x: int
    y: int

    delta_x: float
    delta_y: float

    magnitude: float


@dataclass
class GradientField:
    vectors: list[GradientVector]

    x_matrix: list[list[float]]
    y_matrix: list[list[float]]
    magnitude_matrix: list[list[float]]

    max_magnitude: float

@dataclass
class MoveAnalysis:
    move: str
    piece_name: str
    color: str

    from_square: str
    to_square: str

    is_capture: bool
    is_check: bool

    control: ControlSummary
    heatmap: list[list[int]]

    white_center: tuple[float, float] | None
    black_center: tuple[float, float] | None

    control_vector: FieldVector | None
    attack_vectors: list[AttackVector]

    potential_field: PotentialField
    gradient_field: GradientField


@dataclass
class DynamicsAnalysis:
    previous_force: int
    current_force: int
    delta_force: int

    white_control_delta: int
    black_control_delta: int

    attack_vectors_delta: int
    heatmap_change: int

    intensity: float
    label: str