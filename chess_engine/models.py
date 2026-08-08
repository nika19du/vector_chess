from dataclasses import dataclass
from enum import Enum

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

    Идентичност (identity contract):
        "Същата критична точка" в analysis.ridge_valley и
        analysis.morse_smale се определя ИЗКЛЮЧИТЕЛНО чрез Python
        object identity (`is`/`id()`), НЕ чрез равенство на стойности
        (`==`). Например locate_ridge_valley_chains изключва анкера
        от списъка "други точки" чрез `point is not anchor`, а
        locate_morse_smale_separatrices прави същото и допълнително
        връща действителния обект (не копие или пресъздадена точка) в
        SeparatrixPath.end_critical_point, когато трасирането стигне
        до него.

        Следствие: ако списък от ClassifiedCriticalPoint мине през
        copy.deepcopy, dataclasses.replace, сериализация/десериализация
        или каквато и да е реконструкция, връзката на идентичност се
        губи МЪЛЧАЛИВО -- копие на анкера вече не е `is` оригиналния
        анкер, "изключи себе си" филтрите спират да работят коректно,
        а съвпаденията по разстояние в трасирането може погрешно да
        третират копието като "друга" точка. Никой код в тази
        кодова база в момента не копира/пресъздава тези обекти между
        локализация и трасиране -- затова договорът засега се спазва
        мълчаливо, но всеки бъдещ код, който го прави, трябва явно да
        възстанови идентичността (напр. чрез повторно свързване по
        (x, y) или чрез запазване на същите обекти), а не да приема,
        че `==` е достатъчно.
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


@dataclass
class RidgeValleyPoint:
    """
    Една проследена точка от ridge/valley верига върху
    AttackInfluenceSurface. Виж analysis/ridge_valley.py и
    docs/mathematics.md, Раздел 10.

    x, y:
        екранни координати -- директно по непрекъснатия сплайн,
        никога закръглени до възлите на дискретната мрежа.

    value:
        стойността на повърхността в тази точка.

    eigenvalue_cross:
        собственото число на Hessian-а в КРОСОВАТА (перпендикулярна
        на билото/долината) посока в тази точка -- eigenvalue_min
        за "ridge", eigenvalue_max за "valley". Трябва да пази
        нужния знак (отрицателно за ridge, положително за valley)
        навсякъде по веригата -- иначе маршируването щеше да е
        спряло преди тази точка.

    alignment_residual:
        |∇φ · e_cross| в тази точка -- колко близо е тя до истинско
        ridge/valley условие (нула означава без наклон в кросовата
        посока). За анкерната точка (самата критична точка) винаги
        0.0, защото градиентът там вече е ~0 във всяка посока.

    kind:
        "ridge" или "valley".
    """

    x: float
    y: float
    value: float

    eigenvalue_cross: float
    alignment_residual: float

    kind: str


@dataclass
class RidgeValleyChain:
    """
    Наредена верига от RidgeValleyPoint, трасирана в двете посоки от
    един анкер (класифициран maximum за "ridge", minimum за
    "valley"). Виж analysis/ridge_valley.py и docs/mathematics.md,
    Раздел 10.

    kind:
        "ridge" или "valley".

    points:
        наредени от единия край до другия, преминаващи през анкера
        по средата -- НЕ само от анкера навън в една посока.

    anchor:
        оригиналният ClassifiedCriticalPoint, от който е тръгнало
        трасирането. None само в случаите, в които анкерът не може
        да бъде представен като такъв (не се очаква при нормална
        употреба -- locate_ridge_valley_chains винаги го подава).

    start_status, end_status:
        причината за спиране във всеки край (началото на points
        списъка и края му, съответно) -- никога "успех" сам по
        себе си, само описание защо трасирането е спряло точно там:

        "left_domain" -- следващата стъпка би напуснала
            [x_min,x_max] x [y_min,y_max].
        "weak_curvature" -- |eigenvalue_cross| падна под
            CROSS_EIGENVALUE_COLLAPSE_THRESHOLD -- кривината в
            кросовата посока вече е неразличима от числения шум.
        "misaligned" -- |∇φ·e_cross| / |∇φ| надхвърли
            ALIGNMENT_RELATIVE_TOLERANCE -- точката вече не лежи на
            билото/долината.
        "reached_critical_point" -- трасирането стигна до РАЗЛИЧНА
            (не анкерната) вече класифицирана критична точка.
        "self_intersection" -- новата точка се върна твърде близо до
            по-стара точка от СЪЩАТА посока -- маршируването би
            затворило контур вместо да продължи напред.
        "max_steps_reached" -- изчерпан бюджет от стъпки
            (MAX_MARCH_STEPS) в тази посока.
        "isotropic_point" -- Hessian-ът е изотропен (собствените
            вектори не са дефинирани) точно в анкера -- веригата
            съдържа само анкерната точка и в двата края.
    """

    kind: str
    points: list[RidgeValleyPoint]
    anchor: ClassifiedCriticalPoint | None

    start_status: str
    end_status: str


@dataclass
class RidgeValleyQualityAssessment:
    """
    Обвива RidgeValleyChain с преценка за качество -- НЕ променя
    трасираната верига по никакъв начин. Аналогично на
    CriticalPointQualityAssessment, но на ниво цяла верига, а не
    отделна точка. Виж analysis/ridge_valley.py и
    docs/mathematics.md, Раздел 10.

    chain:
        оригиналната RidgeValleyChain, напълно непроменена.

    is_accepted:
        True само ако веригата премине всички приложени критерии.

    rejection_reasons:
        ВСИЧКИ провалени критерии, четимо описани -- никога само
        "първата причина". Празен списък точно когато is_accepted
        е True.
    """

    chain: RidgeValleyChain
    is_accepted: bool
    rejection_reasons: list[str]


@dataclass
class SeparatrixPoint:
    """
    Една проследена точка от Morse-Smale сепаратриса (интегрална крива
    на градиентния поток) върху AttackInfluenceSurface. Виж
    analysis/morse_smale.py.

    x, y:
        екранни координати -- директно по непрекъснатия сплайн,
        никога закръглени до възлите на дискретната мрежа (същото
        съглашение като RidgeValleyPoint).

    value:
        стойността на повърхността в тази точка.
    """

    x: float
    y: float
    value: float


@dataclass
class SeparatrixPath:
    """
    Едно-посочна интегрална крива на градиентния поток, тръгваща от
    седлова точка (ClassifiedCriticalPoint с classification="saddle").
    Виж analysis/morse_smale.py.

    За разлика от RidgeValleyChain (която съчетава двете посоки на
    ЕДНО и също било/долина в една верига), тук всяка от четирите
    сепаратриси на едно седло (2 възходящи + 2 низходящи клона) е
    отделен SeparatrixPath -- възходящ и низходящ поток отиват към
    качествено различни цели (максимум срещу минимум), затова не могат
    смислено да се съчетаят в един обект.

    start_saddle:
        оригиналният ClassifiedCriticalPoint (седло), от който е
        тръгнало трасирането -- НЕ е включен в points.

    flow_direction:
        "ascending" -- следва +∇φ (посока на нарастване), тръгва по
            собствения вектор на положителното собствено число на
            Hessian-а в седлото (eigenvalue_max).
        "descending" -- следва -∇φ (посока на намаляване), тръгва по
            собствения вектор на отрицателното собствено число
            (eigenvalue_min).

    points:
        наредени от седлото навън (седлото самото НЕ е включено).

    end_critical_point:
        ClassifiedCriticalPoint, до който трасирането е стигнало
        (termination_status "reached_critical_point" или
        "reached_unreliable_point"), иначе None.

    termination_status:
        причината за спиране -- никога "успех" сам по себе си, само
        описание защо трасирането е спряло точно там:

        "left_domain" -- следващата стъпка би напуснала
            [x_min,x_max] x [y_min,y_max].
        "reached_critical_point" -- трасирането стигна до ДРУГА
            (не start_saddle), качествено ПРИЕТА класифицирана
            критична точка.
        "reached_unreliable_point" -- трасирането стигна до качествено
            ОТХВЪРЛЕНА (rejected) класифицирана критична точка --
            спира там вместо тихо да продължи през ненадеждна точка.
        "gradient_stagnation" -- |∇φ| падна под
            GRADIENT_STAGNATION_THRESHOLD -- посоката на потока вече е
            числено недефинирана (нормализиране на почти нулев вектор).
        "self_intersection" -- новата точка се върна твърде близо до
            по-стара точка от СЪЩИЯ клон -- трасирането би затворило
            контур вместо да продължи напред.
        "max_steps_reached" -- изчерпан бюджет от стъпки
            (MAX_MARCH_STEPS) в тази посока.
        "isotropic_point" -- Hessian-ът в start_saddle е изотропен
            (собствените вектори не са дефинирани) -- points е празен.
            Математически не се очаква за истинско седло (изисква
            собствени числа с противоположни знаци), но
            compute_symmetric_eigenvectors може по договор да върне
            None, затова се обработва явно.

    step_count:
        len(points) -- брой успешно приети стъпки в тази посока.

    path_length:
        сборът от евклидовите разстояния между последователните точки
        (започвайки от start_saddle), а не просто step_count *
        step_size -- изчислен директно от точките, не по формула.
    """

    start_saddle: ClassifiedCriticalPoint
    flow_direction: str

    points: list[SeparatrixPoint]
    end_critical_point: ClassifiedCriticalPoint | None

    termination_status: str
    step_count: int
    path_length: float


class TopologyIssueKind(Enum):
    """
    Категории аномалии, които analysis.morse_smale.assemble_morse_smale
    _cells може да засече, докато сглобява клетки от вече трасираните
    SeparatrixPath -- структуриран enum вместо низове с представки, за
    да могат consumers да филтрират/броят по категория без парсене на
    текст (за разлика от categorize_rejection_reason в console_app
    /main.py, който съзнателно НЕ се копира тук).

    SADDLE_TO_SADDLE_CONNECTIVITY:
        сепаратриса е стигнала до ДРУГО седло (не до максимум/минимум).
        "Чист" Morse-Smale комплекс предполага седло<->екстремум
        свързаност; включена в графа въпреки това, само отбелязана.

    SADDLE_BRANCH_COUNT_MISMATCH:
        седло с брой сепаратриси, различен от очакваните 4 (2
        възходящи + 2 низходящи) -- нарушава собствената гаранция на
        locate_morse_smale_separatrices; сглобяването продължава с
        каквото е налично, но отбелязва несъответствието.

    DEGENERATE_ANGULAR_ORDERING:
        две полу-ребра, тръгващи от един и същ връх, имат посоки в
        рамките на числен epsilon една от друга -- сортирането по ъгъл
        е ненадеждно точно там.

    FACE_TRACE_BUDGET_EXCEEDED:
        обхождането на границата на клетка надхвърли бюджета от стъпки
        (предпазна мярка срещу безкраен цикъл) -- изоставено, вместо
        да увисне.

    ZERO_LENGTH_EDGE:
        затворена сепаратриса със step_count == 0 -- стигнала целта си
        на самата първа стъпка (началната и крайната критична точка са
        практически съвпадащи).
    """

    SADDLE_TO_SADDLE_CONNECTIVITY = "saddle_to_saddle_connectivity"
    SADDLE_BRANCH_COUNT_MISMATCH = "saddle_branch_count_mismatch"
    DEGENERATE_ANGULAR_ORDERING = "degenerate_angular_ordering"
    FACE_TRACE_BUDGET_EXCEEDED = "face_trace_budget_exceeded"
    ZERO_LENGTH_EDGE = "zero_length_edge"


@dataclass
class TopologyIssue:
    """
    Една засегната аномалия -- НЕ спира сглобяването на клетки, само
    го анотира честно (същия принцип като rejection_reasons в
    CriticalPointQualityAssessment/RidgeValleyQualityAssessment: никога
    тихо изтриване на информация).

    kind:
        категорията, виж TopologyIssueKind.

    critical_point:
        засегнатият връх, ако е приложимо (напр. седлото с грешен брой
        клонове) -- иначе None.

    separatrix:
        засегнатата сепаратриса, ако е приложимо (напр. седло-до-седло
        връзка) -- иначе None.

    detail:
        четимо описание на конкретиката (напр. точния ъгъл на
        разминаване, точния брой клонове) -- допълва kind, не го
        замества.
    """

    kind: TopologyIssueKind
    critical_point: ClassifiedCriticalPoint | None
    separatrix: SeparatrixPath | None
    detail: str


@dataclass
class MorseSmaleCell:
    """
    ЕДНА 2-клетка (басейн) на Morse-Smale комплекса -- ЧИСТО
    топологичен обект. Геометрията (полигон/площ/центроид/периметър)
    умишлено НЕ живее тук -- виж MorseSmaleCellGeometry и
    analysis.morse_smale.compute_cell_geometry, извиквана при нужда от
    caller-а, а не изчислявана автоматично при сглобяването. Виж
    analysis/morse_smale.py.

    cell_id:
        пореден номер по реда на откриване (детерминиран за фиксиран
        входен списък от SeparatrixPath -- виж assemble_morse_smale
        _cells).

    boundary_separatrices:
        оригиналните SeparatrixPath обекти по границата, в реда на
        обхождането -- споделени по идентичност (`is`) със съседните
        клетки, които границата разделя (виж boundary_traversal
        _directions).

    boundary_traversal_directions:
        успореден списък на boundary_separatrices: True означава
        обхождане start_saddle -> end_critical_point (напред), False
        -- обратно. Нужен е, защото една и съща сепаратриса се обхожда
        в противоположни посоки от двете клетки от двете ѝ страни.

    boundary_critical_points:
        върхът, в който се пристига след всяка гранична сепаратриса, в
        реда на обхождането -- len() == len(boundary_separatrices).

    is_closed:
        True само ако обхождането се е върнало в началната полу-ребро
        -- иначе граничният обход е спрян преждевременно (виж
        open_boundaries) и клетката е непълна.

    open_boundaries:
        сепаратрисите (не задължително точно една -- представена като
        списък, за да поддържа естествено бъдещо разширение), които са
        спрели обхождането -- празен списък точно когато is_closed е
        True.
    """

    cell_id: int
    boundary_separatrices: list[SeparatrixPath]
    boundary_traversal_directions: list[bool]
    boundary_critical_points: list[ClassifiedCriticalPoint]
    is_closed: bool
    open_boundaries: list[SeparatrixPath]


@dataclass
class MorseSmaleComplex:
    """
    Резултатът от analysis.morse_smale.assemble_morse_smale_cells --
    0-клетките (vertices) и 1-клетките (edges) вече съществуват от
    Фаза 1, тук се добавят само 2-клетките (cells), сглобени
    изключително от вече трасираните SeparatrixPath. Не пресъздава и
    не променя нито едно от входните полета.

    vertices:
        различните ClassifiedCriticalPoint, засегнати от поне една
        сепаратриса (по идентичност).

    edges:
        == входния списък от SeparatrixPath, непроменен.

    cells:
        затворени И отворени/непълни MorseSmaleCell (виж is_closed).

    unassigned_separatrices:
        затворени сепаратриси, чиито полу-ребра са били отбелязани
        като посетени по време на ИЗОСТАВЕНО обхождане
        (FACE_TRACE_BUDGET_EXCEEDED) -- при нормална работа този
        списък е празен, защото всяко полу-ребро на затворена
        сепаратриса принадлежи точно на едно (затворено или отворено)
        обхождане.

    topology_issues:
        виж TopologyIssue -- никога не спира сглобяването, само го
        анотира.
    """

    vertices: list[ClassifiedCriticalPoint]
    edges: list[SeparatrixPath]
    cells: list[MorseSmaleCell]
    unassigned_separatrices: list[SeparatrixPath]
    topology_issues: list[TopologyIssue]


@dataclass
class MorseSmaleCellGeometry:
    """
    Изведена (derived) геометрия на ЕДНА затворена MorseSmaleCell --
    НИКОГА не се съхранява на самата клетка, винаги се изчислява при
    нужда чрез analysis.morse_smale.compute_cell_geometry (None за
    отворени клетки, за които полигон не е добре дефиниран). Виж
    analysis/morse_smale.py.

    polygon:
        плътната, следваща кривите граница -- за всяко гранично ребро,
        връх + собствените му SeparatrixPoint (обърнати, ако реброто е
        обходено назад), не права линия между критичните точки.
        Затворен цикъл: polygon[0] == polygon[-1].

    area:
        формулата на Гаус (shoelace) върху polygon.

    centroid:
        стандартната формула за центроид на полигон (претеглена по
        shoelace), дефинирана за прост (несамопресичащ се) полигон.

    perimeter:
        сборът от последователните разстояния по polygon. НЕ е просто
        сборът от path_length на boundary_separatrices -- проверено
        директно при изграждането на Фаза 2: всяка сепаратриса спира
        REACHED_CRITICAL_POINT_DISTANCE преди истинската си цел (по
        дизайн на Фаза 1), затова полигонът включва по един
        допълнителен "затварящ" сегмент на ребро (от последната
        проследена точка до истинския връх), който path_length не
        покрива.
    """

    polygon: list[tuple[float, float]]
    area: float
    centroid: tuple[float, float]
    perimeter: float


class MorseSmaleCellRejectionReasonKind(Enum):
    """
    Категории причини, поради които assess_morse_smale_cell_quality
    отхвърля клетка -- структуриран enum, по същия принцип като
    TopologyIssueKind, вместо низове с представки. Виж
    analysis/morse_smale.py.

    NOT_CLOSED:
        cell.is_closed е False -- геометрията (полигон/площ/центроид
        /периметър) не е дефинирана.

    UNRELIABLE_BOUNDARY_TERMINATION:
        защитна проверка, която в момента не се очаква да задейства:
        по конструкция на Фаза 2 затворена клетка може да съдържа само
        сепаратриси със termination_status="reached_critical_point"
        -- запазена изрично като отделен критерий, в случай че това
        гарантирано някога се разхлаби.

    AFFECTED_BY_TOPOLOGY_ISSUE:
        някоя гранична сепаратриса или критична точка на клетката
        съвпада (по идентичност) с .separatrix/.critical_point на
        TopologyIssue от MorseSmaleComplex.topology_issues --
        related_topology_issue носи кой точно TopologyIssueKind.

    AREA_TOO_SMALL, PERIMETER_TOO_SMALL:
        геометрията е дефинирана (клетката Е затворена), но е под
        числовия шумов праг -- вижте MIN_CELL_AREA/MIN_CELL_PERIMETER
        в analysis/morse_smale.py за обосновката на прага.

    TOO_FEW_DISTINCT_BOUNDARY_POINTS:
        по-малко от MIN_DISTINCT_BOUNDARY_POINTS различни (по
        идентичност) върха по границата -- напр. изроден "бигон"
        (седло, чиито два клона от един и същ тип стигат до една и
        съща цел).

    NEAR_BOUNDARY:
        (опционален критерий, изключен по подразбиране) повечето
        гранични върхове са твърде близо до ръба на дъската.

    DEGENERATE_VERTEX_PARTICIPATION:
        (опционален критерий, изключен по подразбиране) гранична
        критична точка с classification="degenerate".
    """

    NOT_CLOSED = "not_closed"
    UNRELIABLE_BOUNDARY_TERMINATION = "unreliable_boundary_termination"
    AFFECTED_BY_TOPOLOGY_ISSUE = "affected_by_topology_issue"
    AREA_TOO_SMALL = "area_too_small"
    PERIMETER_TOO_SMALL = "perimeter_too_small"
    TOO_FEW_DISTINCT_BOUNDARY_POINTS = "too_few_distinct_boundary_points"
    NEAR_BOUNDARY = "near_boundary"
    DEGENERATE_VERTEX_PARTICIPATION = "degenerate_vertex_participation"


@dataclass
class MorseSmaleCellRejectionReason:
    """
    Една провалена проверка за качество на клетка -- НЕ спира
    оценяването, само го анотира (същия принцип като TopologyIssue и
    CriticalPointQualityAssessment/RidgeValleyQualityAssessment).

    kind:
        категорията, виж MorseSmaleCellRejectionReasonKind.

    detail:
        четимо описание на конкретиката (напр. точната измерена площ
        спрямо прага) -- допълва kind, не го замества.

    related_topology_issue:
        само за kind=AFFECTED_BY_TOPOLOGY_ISSUE -- кой точно
        TopologyIssueKind е засегнал тази клетка. None за всички
        останали видове причини.
    """

    kind: MorseSmaleCellRejectionReasonKind
    detail: str
    related_topology_issue: TopologyIssueKind | None


@dataclass
class MorseSmaleCellQualityAssessment:
    """
    Обвива MorseSmaleCell с преценка за качество -- НЕ променя
    топологията на клетката по никакъв начин. Аналогично на
    CriticalPointQualityAssessment/RidgeValleyQualityAssessment, но за
    2-клетки. Виж analysis/morse_smale.py.

    cell:
        оригиналната MorseSmaleCell, напълно непроменена.

    is_accepted:
        True само ако клетката премине всички приложени критерии.

    rejection_reasons:
        ВСИЧКИ провалени критерии, четимо описани -- никога само
        "първата причина". Празен списък точно когато is_accepted е
        True.

        За ОТВОРЕНИ клетки (is_closed=False): геометрично-зависимите
        критерии (AREA_TOO_SMALL, PERIMETER_TOO_SMALL) никога не се
        добавят -- не защото клетката ги "минава", а защото площ и
        периметър просто не са дефинирани без валиден полигон
        (compute_cell_geometry връща None за такива клетки).
        NOT_CLOSED вече е достатъчна причина за отхвърляне; останалите
        (не-геометрични) критерии продължават да се проверяват и могат
        да добавят допълнителни причини.
    """

    cell: MorseSmaleCell
    is_accepted: bool
    rejection_reasons: list[MorseSmaleCellRejectionReason]