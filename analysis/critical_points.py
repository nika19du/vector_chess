import numpy as np

from analysis.attack_influence_surface import sample_gradient
from chess_engine.models import (
    AttackInfluenceSurface,
    ClassifiedCriticalPoint,
    CriticalPointCandidate,
    HessianField,
)


# Стъпка за централна крайна разлика при кръстосаната проверка
# f_xy == f_yx. Достатъчно малка за точност, но не толкова малка,
# че да пресича границата на 8x8 мрежата (0.5 .. 7.5) съществено.
MIXED_PARTIAL_STEP = 1e-3

# |∇φ| прагът, под който клетка от съществуващата гъста мрежа
# (surface.x/surface.y) става кандидат-семе за Newton уточняване.
# Мрежата се използва САМО за да предложи начални точки -- самото
# уточняване винаги продължава върху непрекъснатия сплайн.
SEED_GRADIENT_THRESHOLD = 2.0

# Newton спира със статус "converged", щом |∇φ| падне под този праг.
CONVERGENCE_GRADIENT_THRESHOLD = 1e-6

# Максимален брой Newton стъпки на семе, преди статус
# "max_iterations_reached".
MAX_NEWTON_ITERATIONS = 50

# Две уточнени точки, отдалечени на не повече от това разстояние
# (в екранни координати), се третират като едно и също кандидатно
# критично място и се сливат.
DEDUPLICATION_DISTANCE = 0.25

# Ако абсолютната стойност на кое да е собствено число на Hessian-а
# падне под този праг, той се счита за сингулярен -- статус
# "singular_hessian" вместо разделяне с почти нулев детерминант.
HESSIAN_MIN_EIGENVALUE = 1e-6

# Фаза 3 -- класификация. Точка се класифицира "degenerate" вместо
# maximum/minimum/saddle, ако е изпълнено поне едно от:
#   (а) |собствено число| < EIGENVALUE_ZERO_THRESHOLD за което и да е
#       от двете собствени числа на Hessian-а в тази точка;
#   (б) |f_xy - f_yx| > HESSIAN_SYMMETRY_TOLERANCE -- тестът за
#       втора производна предполага симетричен Hessian; нарушение
#       прави собствените числа ненадеждни, затова резултатът се
#       отчита честно като недостоверен, вместо да се гадае.
EIGENVALUE_ZERO_THRESHOLD = 1e-6
HESSIAN_SYMMETRY_TOLERANCE = 1e-4

# Отношението |най-голямо собствено число| / |най-малко собствено
# число|, над което Hessian-ът се третира като зле обусловен --
# също статус "singular_hessian".
HESSIAN_CONDITION_THRESHOLD = 1e8


def evaluate_hessian_at_points(
    spline,
    y_points: np.ndarray,
    x_points: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
):
    """
    Общата (grid или единична точка) оценка на Hessian-а, споделена
    между sample_hessian (цялата мрежа на повърхността) и Newton
    уточняването (една точка на итерация). Няма промяна в поведението
    на sample_hessian спрямо Фаза 1 -- само извличане на споделена
    логика.
    """

    f_xx = spline(y_points, x_points, dx=0, dy=2)
    f_yy = spline(y_points, x_points, dx=2, dy=0)

    x_plus = np.clip(x_points + MIXED_PARTIAL_STEP, x_min, x_max)
    x_minus = np.clip(x_points - MIXED_PARTIAL_STEP, x_min, x_max)

    y_plus = np.clip(y_points + MIXED_PARTIAL_STEP, y_min, y_max)
    y_minus = np.clip(y_points - MIXED_PARTIAL_STEP, y_min, y_max)

    # f_yx = ∂/∂x( ∂φ/∂y ): вземаме ∂φ/∂y (dx=1, dy=0) и я
    # диференцираме по x чрез централна крайна разлика.
    phi_y_at_x_plus = spline(y_points, x_plus, dx=1, dy=0)
    phi_y_at_x_minus = spline(y_points, x_minus, dx=1, dy=0)

    denominator_x = x_plus - x_minus
    denominator_x = np.where(
        denominator_x == 0, 2 * MIXED_PARTIAL_STEP, denominator_x
    )

    f_yx = (phi_y_at_x_plus - phi_y_at_x_minus) / denominator_x

    # f_xy = ∂/∂y( ∂φ/∂x ): вземаме ∂φ/∂x (dx=0, dy=1) и я
    # диференцираме по y чрез централна крайна разлика.
    phi_x_at_y_plus = spline(y_plus, x_points, dx=0, dy=1)
    phi_x_at_y_minus = spline(y_minus, x_points, dx=0, dy=1)

    denominator_y = y_plus - y_minus
    denominator_y = np.where(
        denominator_y == 0, 2 * MIXED_PARTIAL_STEP, denominator_y
    )

    f_xy = (phi_x_at_y_plus - phi_x_at_y_minus) / denominator_y[:, np.newaxis]

    return f_xx, f_xy, f_yx, f_yy


def sample_hessian(
    surface: AttackInfluenceSurface,
) -> HessianField:
    """
    Оценява Hessian-а на напаснатия сплайн на AttackInfluenceSurface
    върху неговата собствена гъста мрежа (surface.y, surface.x) --
    точно мрежата, върху която вече работи sample_gradient.

    f_xx, f_yy идват директно от аналитичните производни на
    bicubic сплайна. f_xy и f_yx НЕ се приемат за равни по теоремата
    на Клеро -- всяка се изчислява по собствен, независим път, за да
    може съвпадението им да бъде истинска числена проверка.

    Не локализира критични точки (∇φ ≈ 0) и не ги класифицира --
    само връща H(x, y) навсякъде по мрежата. Не променя
    AttackInfluenceSurface по никакъв начин.
    """

    x_min, x_max = float(surface.x[0]), float(surface.x[-1])
    y_min, y_max = float(surface.y[0]), float(surface.y[-1])

    f_xx, f_xy, f_yx, f_yy = evaluate_hessian_at_points(
        surface.spline, surface.y, surface.x, x_min, x_max, y_min, y_max
    )

    return HessianField(
        f_xx=f_xx,
        f_xy=f_xy,
        f_yx=f_yx,
        f_yy=f_yy,
    )


def generate_seeds(
    surface: AttackInfluenceSurface,
    delta_x: np.ndarray,
    delta_y: np.ndarray,
    threshold: float,
) -> list[tuple[float, float]]:
    """
    Използва СЪЩЕСТВУВАЩАТА гъста мрежа (и вече изчисления градиент
    от sample_gradient) единствено за да предложи начални точки за
    Newton уточняване -- никога не връща крайни критични точки.

    Семе е клетка от мрежата, чиято |∇φ|^2 е (а) под threshold^2 и
    (б) локален минимум сред своите до 8 съседа -- комбинацията
    избягва да предложи цял "плосък" облак от съседни семена там,
    където дедупликацията и без друго би ги слепила.
    """

    gradient_magnitude_squared = delta_x**2 + delta_y**2

    resolution_y, resolution_x = gradient_magnitude_squared.shape
    threshold_squared = threshold**2

    seeds: list[tuple[float, float]] = []

    for row in range(resolution_y):
        for column in range(resolution_x):
            value = gradient_magnitude_squared[row, column]

            if value > threshold_squared:
                continue

            is_local_minimum = True

            for row_offset in (-1, 0, 1):
                for column_offset in (-1, 0, 1):
                    if row_offset == 0 and column_offset == 0:
                        continue

                    neighbor_row = row + row_offset
                    neighbor_column = column + column_offset

                    if not (0 <= neighbor_row < resolution_y):
                        continue

                    if not (0 <= neighbor_column < resolution_x):
                        continue

                    if (
                        gradient_magnitude_squared[neighbor_row, neighbor_column]
                        < value
                    ):
                        is_local_minimum = False
                        break

                if not is_local_minimum:
                    break

            if is_local_minimum:
                seeds.append(
                    (float(surface.y[row]), float(surface.x[column]))
                )

    return seeds


def compute_symmetric_eigenvalues(
    f_xx: float,
    mixed: float,
    f_yy: float,
) -> tuple[float, float, float, float]:
    """
    Затворена форма за собствените числа на симетричната 2x2 матрица
    [[f_xx, mixed], [mixed, f_yy]], споделена между refine_seed
    (Фаза 2 -- за проверка на обусловеността) и
    classify_critical_points (Фаза 3 -- за втория тест на
    производната). Връща (eigenvalue_a, eigenvalue_b, determinant,
    trace); eigenvalue_a >= eigenvalue_b винаги, по конструкция.
    """

    trace = f_xx + f_yy
    determinant = f_xx * f_yy - mixed**2
    discriminant = max((trace / 2) ** 2 - determinant, 0.0)
    sqrt_discriminant = np.sqrt(discriminant)

    eigenvalue_a = trace / 2 + sqrt_discriminant
    eigenvalue_b = trace / 2 - sqrt_discriminant

    return eigenvalue_a, eigenvalue_b, determinant, trace


def refine_seed(
    spline,
    seed_y: float,
    seed_x: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    convergence_threshold: float = CONVERGENCE_GRADIENT_THRESHOLD,
    max_iterations: int = MAX_NEWTON_ITERATIONS,
    min_eigenvalue: float = HESSIAN_MIN_EIGENVALUE,
    condition_threshold: float = HESSIAN_CONDITION_THRESHOLD,
) -> CriticalPointCandidate:
    """
    Уточнява едно семе чрез Newton итерация директно върху
    непрекъснатия сплайн (не е ограничено до възлите на мрежата):

        H(x, y) * delta = -∇φ(x, y)

    на всяка стъпка, докато |∇φ| падне под convergence_threshold,
    бюджетът от итерации се изчерпи, Hessian-ът стане сингулярен/зле
    обусловен, или следващата стъпка би напуснала [x_min,x_max] x
    [y_min,y_max] -- във всеки от тези случаи итерацията спира явно
    на последната валидна точка вместо да продължи без основание.
    """

    x, y = seed_x, seed_y

    for iteration in range(max_iterations):
        grad_x = float(spline(y, x, dx=0, dy=1)[0, 0])
        grad_y = float(spline(y, x, dx=1, dy=0)[0, 0])
        gradient_norm = float(np.hypot(grad_x, grad_y))

        if gradient_norm < convergence_threshold:
            value = float(spline(y, x)[0, 0])
            return CriticalPointCandidate(
                x=x,
                y=y,
                value=value,
                gradient_norm=gradient_norm,
                status="converged",
                iterations=iteration,
            )

        f_xx, f_xy, f_yx, f_yy = evaluate_hessian_at_points(
            spline,
            np.array([y]),
            np.array([x]),
            x_min,
            x_max,
            y_min,
            y_max,
        )

        f_xx_value = float(f_xx[0, 0])
        f_yy_value = float(f_yy[0, 0])
        mixed_value = float(0.5 * (f_xy[0, 0] + f_yx[0, 0]))

        eigenvalue_a, eigenvalue_b, _determinant, _trace = compute_symmetric_eigenvalues(
            f_xx_value, mixed_value, f_yy_value
        )

        min_abs_eigenvalue = min(abs(eigenvalue_a), abs(eigenvalue_b))
        max_abs_eigenvalue = max(abs(eigenvalue_a), abs(eigenvalue_b))

        is_singular = min_abs_eigenvalue < min_eigenvalue
        is_ill_conditioned = (
            not is_singular
            and (max_abs_eigenvalue / min_abs_eigenvalue) > condition_threshold
        )

        if is_singular or is_ill_conditioned:
            value = float(spline(y, x)[0, 0])
            return CriticalPointCandidate(
                x=x,
                y=y,
                value=value,
                gradient_norm=gradient_norm,
                status="singular_hessian",
                iterations=iteration,
            )

        hessian = np.array(
            [[f_xx_value, mixed_value], [mixed_value, f_yy_value]]
        )
        gradient = np.array([grad_x, grad_y])

        delta = np.linalg.solve(hessian, -gradient)

        new_x = x + float(delta[0])
        new_y = y + float(delta[1])

        if not (x_min <= new_x <= x_max) or not (y_min <= new_y <= y_max):
            value = float(spline(y, x)[0, 0])
            return CriticalPointCandidate(
                x=x,
                y=y,
                value=value,
                gradient_norm=gradient_norm,
                status="left_domain",
                iterations=iteration,
            )

        x, y = new_x, new_y

    grad_x = float(spline(y, x, dx=0, dy=1)[0, 0])
    grad_y = float(spline(y, x, dx=1, dy=0)[0, 0])
    gradient_norm = float(np.hypot(grad_x, grad_y))
    value = float(spline(y, x)[0, 0])

    return CriticalPointCandidate(
        x=x,
        y=y,
        value=value,
        gradient_norm=gradient_norm,
        status="max_iterations_reached",
        iterations=max_iterations,
    )


def deduplicate_candidates(
    candidates: list[CriticalPointCandidate],
    distance_threshold: float,
) -> list[CriticalPointCandidate]:
    """
    Слива кандидати, чиито крайни (x, y) са на разстояние <=
    distance_threshold от АНКЕРНАТА (първата видяна) точка на
    съответния клъстер -- различни семена, уточнени до едно и също
    критично място.

    Разстоянието се мери спрямо анкерната точка на клъстера, НЕ спрямо
    текущия представител (accepted[merge_index]), който може да се
    смени при всяко по-добро сливане. Сравняване спрямо текущия
    представител би позволило "верижно" сливане: представителят се
    измества стъпка по стъпка, всяка стъпка в рамките на
    distance_threshold от ПРЕДИШНИЯ представител, но натрупаното
    изместване по цялата верига може да надхвърли distance_threshold
    от първата точка с произволно голяма стойност -- два кандидата,
    които реално НЕ са на разстояние <= distance_threshold едно от
    друго, биха се слели транзитивно през междинни кандидати. Анкерът
    остава фиксиран за целия живот на клъстера, затова радиусът му е
    коректно ограничен до distance_threshold, независимо от броя
    кандидати, слети в него.

    При сливане се предпочита "converged" пред всеки друг статус, а
    при равен статус -- по-малкото gradient_norm; изборът на
    ПРЕДСТАВИТЕЛ (отделно от анкерната точка, използвана само за
    разстоянието) е детерминиран за фиксиран входен ред.
    """

    status_priority = {"converged": 0}

    accepted: list[CriticalPointCandidate] = []
    anchor_positions: list[tuple[float, float]] = []

    for candidate in candidates:
        merge_index = None

        for index, (anchor_x, anchor_y) in enumerate(anchor_positions):
            distance = float(
                np.hypot(candidate.x - anchor_x, candidate.y - anchor_y)
            )

            if distance <= distance_threshold:
                merge_index = index
                break

        if merge_index is None:
            accepted.append(candidate)
            anchor_positions.append((candidate.x, candidate.y))
            continue

        existing = accepted[merge_index]

        candidate_key = (
            status_priority.get(candidate.status, 1),
            candidate.gradient_norm,
        )
        existing_key = (
            status_priority.get(existing.status, 1),
            existing.gradient_norm,
        )

        if candidate_key < existing_key:
            accepted[merge_index] = candidate

    return accepted


def locate_critical_points(
    surface: AttackInfluenceSurface,
    seed_gradient_threshold: float = SEED_GRADIENT_THRESHOLD,
    convergence_threshold: float = CONVERGENCE_GRADIENT_THRESHOLD,
    max_iterations: int = MAX_NEWTON_ITERATIONS,
    deduplication_distance: float = DEDUPLICATION_DISTANCE,
    min_eigenvalue: float = HESSIAN_MIN_EIGENVALUE,
    condition_threshold: float = HESSIAN_CONDITION_THRESHOLD,
) -> list[CriticalPointCandidate]:
    """
    Локализира кандидати за критични точки на AttackInfluenceSurface:

    1. sample_gradient дава градиента върху съществуващата мрежа.
    2. generate_seeds предлага начални точки от тази мрежа --
       мрежата НИКОГА не е крайният отговор, само отправна точка.
    3. Всяко семе се уточнява с Newton директно върху непрекъснатия
       сплайн (refine_seed), докато сходи, изчерпи бюджета си,
       срещне сингулярен Hessian, или напусне валидната дъска.
    4. deduplicate_candidates слива уточнени точки, попаднали
       практически на едно и също място.
    5. Резултатът се сортира по (y, x) за детерминиран, стабилен ред,
       независим от вътрешния ред на откриване на семената.

    Не класифицира точките (максимум/минимум/седло) -- това е
    следваща фаза. Не променя surface или каквото и да е негово
    поле.
    """

    delta_x, delta_y = sample_gradient(surface)

    seeds = generate_seeds(surface, delta_x, delta_y, seed_gradient_threshold)

    x_min, x_max = float(surface.x[0]), float(surface.x[-1])
    y_min, y_max = float(surface.y[0]), float(surface.y[-1])

    candidates = [
        refine_seed(
            surface.spline,
            seed_y,
            seed_x,
            x_min,
            x_max,
            y_min,
            y_max,
            convergence_threshold,
            max_iterations,
            min_eigenvalue,
            condition_threshold,
        )
        for seed_y, seed_x in seeds
    ]

    deduplicated = deduplicate_candidates(candidates, deduplication_distance)

    deduplicated.sort(key=lambda candidate: (round(candidate.y, 6), round(candidate.x, 6)))

    return deduplicated


def classify_critical_points(
    candidates: list[CriticalPointCandidate],
    surface: AttackInfluenceSurface,
    eigenvalue_zero_threshold: float = EIGENVALUE_ZERO_THRESHOLD,
    symmetry_tolerance: float = HESSIAN_SYMMETRY_TOLERANCE,
) -> list[ClassifiedCriticalPoint]:
    """
    Класифицира кандидатите, върнати от locate_critical_points --
    не ги локализира наново и не променя locate_critical_points или
    surface по никакъв начин.

    За всеки candidate със status != "converged": класификацията е
    "unclassified" -- точка, която самата Newton итерация не е
    приела за истинска критична точка, няма смисъл да се
    класифицира по тип, и Hessian-ът НЕ се оценява там (f_xx..f_yy
    и eigenvalue_min/max остават None).

    За всеки "converged" candidate: Hessian-ът се оценява точно в
    (candidate.x, candidate.y) чрез evaluate_hessian_at_points (същата
    механика като refine_seed/sample_hessian -- никаква нова
    координатна трансформация), после:

    1. Ако |f_xy - f_yx| > symmetry_tolerance -- тестът за втора
       производна разчита на симетричен Hessian; нарушение прави
       собствените числа ненадеждни -- "degenerate", вместо да се
       гадае кой знак печели.
    2. Ако |кое да е собствено число| < eigenvalue_zero_threshold --
       "degenerate" -- класическият тест е неубедителен точно тук.
    3. Иначе -- класическият тест за втора производна върху
       собствените числа на симетризирания Hessian
       ([[f_xx, (f_xy+f_yx)/2], [(f_xy+f_yx)/2, f_yy]]):
       и двете положителни -> "minimum";
       и двете отрицателни -> "maximum";
       различни знаци -> "saddle".

    Оригиналните локализационни полета (x, y, value, gradient_norm,
    status, iterations) се копират непроменени -- класификацията
    никога не ги пренаписва.
    """

    x_min, x_max = float(surface.x[0]), float(surface.x[-1])
    y_min, y_max = float(surface.y[0]), float(surface.y[-1])

    classified: list[ClassifiedCriticalPoint] = []

    for candidate in candidates:
        if candidate.status != "converged":
            classified.append(
                ClassifiedCriticalPoint(
                    x=candidate.x,
                    y=candidate.y,
                    value=candidate.value,
                    gradient_norm=candidate.gradient_norm,
                    status=candidate.status,
                    iterations=candidate.iterations,
                    f_xx=None,
                    f_xy=None,
                    f_yx=None,
                    f_yy=None,
                    eigenvalue_min=None,
                    eigenvalue_max=None,
                    classification="unclassified",
                )
            )
            continue

        f_xx, f_xy, f_yx, f_yy = evaluate_hessian_at_points(
            surface.spline,
            np.array([candidate.y]),
            np.array([candidate.x]),
            x_min,
            x_max,
            y_min,
            y_max,
        )

        f_xx_value = float(f_xx[0, 0])
        f_yy_value = float(f_yy[0, 0])
        f_xy_value = float(f_xy[0, 0])
        f_yx_value = float(f_yx[0, 0])
        mixed_value = 0.5 * (f_xy_value + f_yx_value)

        eigenvalue_a, eigenvalue_b, _determinant, _trace = compute_symmetric_eigenvalues(
            f_xx_value, mixed_value, f_yy_value
        )

        eigenvalue_min = min(eigenvalue_a, eigenvalue_b)
        eigenvalue_max = max(eigenvalue_a, eigenvalue_b)

        is_asymmetric = abs(f_xy_value - f_yx_value) > symmetry_tolerance
        has_near_zero_eigenvalue = (
            abs(eigenvalue_min) < eigenvalue_zero_threshold
            or abs(eigenvalue_max) < eigenvalue_zero_threshold
        )

        if is_asymmetric or has_near_zero_eigenvalue:
            classification = "degenerate"
        elif eigenvalue_min > 0:
            classification = "minimum"
        elif eigenvalue_max < 0:
            classification = "maximum"
        else:
            classification = "saddle"

        classified.append(
            ClassifiedCriticalPoint(
                x=candidate.x,
                y=candidate.y,
                value=candidate.value,
                gradient_norm=candidate.gradient_norm,
                status=candidate.status,
                iterations=candidate.iterations,
                f_xx=f_xx_value,
                f_xy=f_xy_value,
                f_yx=f_yx_value,
                f_yy=f_yy_value,
                eigenvalue_min=eigenvalue_min,
                eigenvalue_max=eigenvalue_max,
                classification=classification,
            )
        )

    return classified
