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
class MobilitySummary:
    white_reachable_squares: int
    black_reachable_squares: int
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
class AttackerCountCell:
    square: str
    white_attackers: int
    black_attackers: int
    difference: int


@dataclass
class AttackerCountField:
    cells: list[AttackerCountCell]
    matrix: list[list[int]]


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
class AttackInfluenceCell:
    """
    Стойността на атаково влияние на едно шахматно поле.

    white_attack_influence:
        сумата от тежестите на белите фигури,
        които влияят върху полето.

    black_attack_influence:
        сумата от тежестите на черните фигури,
        които влияят върху полето.

    difference:
        white_attack_influence - black_attack_influence
    """

    square: str
    white_attack_influence: float
    black_attack_influence: float
    difference: float


@dataclass
class AttackInfluenceField:
    """
    Цялото поле на атаково влияние на шахматната дъска.
    """

    cells: list[AttackInfluenceCell]

    matrix: list[list[float]]

    total_white_attack_influence: float
    total_black_attack_influence: float

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
class AttackInfluenceSurface:
    """
    Непрекъсната bicubic апроксимация на дискретното
    8x8 поле на атаково влияние.

    x, y:
        координати на гъстата мрежа в екранни координати
        (x = файл/колона, y = ред; y=0 съответства на rank 8,
        нагоре по дъската, за да съвпада с imshow/board_plot).

    z:
        стойностите на атаковото влияние върху гъстата мрежа,
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
class HessianField:
    """
    Матрицата на вторите частни производни (Hessian) на
    AttackInfluenceSurface, оценена върху нейната собствена
    гъста мрежа (surface.y, surface.x) -- същата мрежа, върху
    която работи и sample_gradient.

    f_xx, f_yy:
        чисти втори производни (кривина по колона/файл,
        съответно по ред/ранг), взети директно от bicubic
        сплайна.

    f_xy, f_yx:
        смесената производна ∂²φ/∂x∂y, изчислена по два
        независими пътя (диференциране на ∂φ/∂y спрямо x,
        и диференциране на ∂φ/∂x спрямо y) -- не се приема
        предварително, че си съвпадат по теоремата на Клеро,
        а се проверява числено.

    Не локализира и не класифицира критични точки -- само
    оценява H(x, y) навсякъде по мрежата. Виж
    analysis/critical_points.py и docs/mathematics.md,
    Раздел 9.
    """

    f_xx: np.ndarray
    f_xy: np.ndarray
    f_yx: np.ndarray
    f_yy: np.ndarray


@dataclass
class CriticalPointCandidate:
    """
    Резултат от Newton локализация на един кандидат за критична
    точка върху AttackInfluenceSurface -- НЕ е класифициран
    (максимум/минимум/седло); класификацията е следваща фаза.
    Виж analysis/critical_points.py и docs/mathematics.md, Раздел 9.

    x, y:
        крайната позиция в екранни координати -- напасната директно
        върху непрекъснатия сплайн, не е ограничена до възлите на
        дискретната мрежа.

    value:
        стойността на повърхността в тази точка.

    gradient_norm:
        |∇φ| в крайната точка -- колко близо е тя до истинско ∇φ=0.

    status:
        "converged" -- |∇φ| падна под прага за сходимост.
        "max_iterations_reached" -- изчерпан бюджет от Newton стъпки.
        "singular_hessian" -- Hessian-ът е сингулярен или зле
            обусловен в текущата точка; итерацията спира там, вместо
            да раздели с почти нулев детерминант.
        "left_domain" -- следващата Newton стъпка би напуснала
            валидния диапазон на дъската; итерацията спира на
            последната валидна точка вместо да я приема.

    iterations:
        брой извършени Newton стъпки преди спиране.
    """

    x: float
    y: float
    value: float
    gradient_norm: float
    status: str
    iterations: int


@dataclass
class ClassifiedCriticalPoint:
    """
    Класифициран резултат от analysis.critical_points, обвиващ
    CriticalPointCandidate с типа му (максимум/минимум/седло) и
    Hessian-а, използван за да се стигне до него -- за пълна
    проследимост. Виж docs/mathematics.md, Раздел 9.

    x, y, value, gradient_norm, status, iterations:
        запазени непроменени от оригиналния CriticalPointCandidate --
        класификацията никога не пренаписва локализационните метаданни.

    f_xx, f_xy, f_yx, f_yy:
        стойностите на Hessian-а точно в (x, y), взети чрез
        evaluate_hessian_at_points -- None, ако status != "converged"
        (Hessian не се оценява в некон converged точка).

    eigenvalue_min, eigenvalue_max:
        собствените числа на симетризирания Hessian
        ([[f_xx, mixed], [mixed, f_yy]], mixed = (f_xy+f_yx)/2),
        подредени числово (eigenvalue_min <= eigenvalue_max) --
        None при status != "converged".

    classification:
        "maximum"      -- eigenvalue_max < 0 (двете отрицателни).
        "minimum"      -- eigenvalue_min > 0 (двете положителни).
        "saddle"       -- различни знаци.
        "degenerate"   -- собствено число ~0 ИЛИ f_xy/f_yx не
                          съвпадат в рамките на HESSIAN_SYMMETRY_TOLERANCE
                          -- тестът е недостоверен, отчита се честно
                          вместо да се гадае.
        "unclassified" -- кандидатът не е "converged"; няма смисъл
                          да се класифицира точка, която не е
                          истинска критична точка.
    """

    x: float
    y: float
    value: float
    gradient_norm: float
    status: str
    iterations: int

    f_xx: float | None
    f_xy: float | None
    f_yx: float | None
    f_yy: float | None

    eigenvalue_min: float | None
    eigenvalue_max: float | None

    classification: str


@dataclass
class CriticalPointQualityAssessment:
    """
    Обвива ClassifiedCriticalPoint с преценка за качество -- НЕ
    променя локализацията или класификацията. Единствената цел е да
    раздели "математически открита критична точка на напаснатия
    сплайн" от "точка, достатъчно надеждна, за да се представи като
    шахматно значима". Виж analysis/critical_point_quality.py и
    docs/mathematics.md, Раздел 9.

    point:
        оригиналният ClassifiedCriticalPoint, напълно непроменен.

    is_accepted:
        True само ако точката премине всички приложени критерии.

    rejection_reasons:
        ВСИЧКИ провалени критерии, четимо описани -- никога само
        "първата причина", за да не се скрие информация. Празен
        списък точно когато is_accepted е True.
    """

    point: ClassifiedCriticalPoint
    is_accepted: bool
    rejection_reasons: list[str]


@dataclass
class SourcePotentialSurface:
    """
    Непрекъснатото Φ_source(x, y), генерирано чрез директно
    сумиране на ρ с decay kernel — НЕ чрез сплайн интерполация.

    За разлика от AttackInfluenceSurface, тук няма spline обект,
    защото повърхността не минава през дискретни точки чрез
    интерполация, а е директна суперпозиция на källов принос
    от всяка фигура.

    x, y, z, resolution:
        същата екранна конвенция като AttackInfluenceSurface, за да
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

    mobility: MobilitySummary
    attacker_count_field: AttackerCountField

    white_center: tuple[float, float] | None
    black_center: tuple[float, float] | None

    control_vector: FieldVector | None
    attack_vectors: list[AttackVector]

    attack_influence_field: AttackInfluenceField
    gradient_field: GradientField


@dataclass
class DynamicsAnalysis:
    previous_force: int
    current_force: int
    delta_force: int

    white_mobility_delta: int
    black_mobility_delta: int

    attack_vectors_delta: int
    heatmap_change: int

    intensity: float
    label: str