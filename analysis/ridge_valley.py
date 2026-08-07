import math

import numpy as np

from analysis.critical_points import (
    compute_symmetric_eigenvalues,
    evaluate_hessian_at_points,
)
from chess_engine.models import (
    AttackInfluenceSurface,
    ClassifiedCriticalPoint,
    RidgeValleyChain,
    RidgeValleyPoint,
    RidgeValleyQualityAssessment,
)


# Праг, под който f_xx == f_yy И mixed == 0 се третират като
# изотропна (изродена) точка на Hessian-а -- собствените вектори не
# са дефинирани там (всяка посока е собствен вектор с еднакво
# собствено число), затова билна/долинна посока не съществува.
# Съизмерим с EIGENVALUE_ZERO_THRESHOLD от analysis/critical_points.py
# -- същия ред на числения шум.
ISOTROPIC_HESSIAN_TOLERANCE = 1e-6

# Дължина на една стъпка на маршируването, в екранни координати
# (клетка = 1.0 единица). ~1/20 от клетка -- достатъчно фино да
# проследи кривината на билото/долината, без прекомерно много стъпки
# за типичната дължина на дъската (~7 единици в едната посока).
STEP_SIZE = 0.05

# Максимален брой стъпки на ЕДНА посока (напред ИЛИ назад поотделно),
# преди статус "max_steps_reached". 500 * STEP_SIZE = 25 единици --
# много повече от диагонала на дъската (~9.9 единици), т.е. на
# практика никога не е ограничаващият фактор при нормално поведение;
# съществува само като предпазител срещу безкраен цикъл.
MAX_MARCH_STEPS = 500

# Праг за "изтъняване" на кросовата (перпендикулярна на билото/
# долината) крива -- ако |eigenvalue_cross| падне под този праг,
# маршируването спира със статус "weak_curvature". Стойността
# умишлено съвпада с EIGENVALUE_ZERO_THRESHOLD от
# analysis/critical_points.py -- същият праг, същата обосновка: под
# това ниво собственото число е неразличимо от числения шум на
# недосемплирания 8x8 сплайн.
CROSS_EIGENVALUE_COLLAPSE_THRESHOLD = 1e-6

# Ако |∇φ| в текущата точка е под този праг, тестът за подравняване
# (alignment, виж ALIGNMENT_RELATIVE_TOLERANCE) НЕ се прилага -- твърде
# близо до критична точка (където градиентът е ~0 във всяка посока,
# включително кросовата) делението |∇φ·e_cross| / |∇φ| е числено
# безсмислено (0/0), а не признак за отклонение от билото/долината.
# Такива точки естествено се хващат от REACHED_CRITICAL_POINT_DISTANCE
# вместо от alignment проверката.
MIN_GRADIENT_FOR_ALIGNMENT_CHECK = 1e-4

# Относителен праг за подравняване: |∇φ·e_cross| / |∇φ| над този праг
# означава, че точката вече не лежи на билото/долината -- останал е
# осезаем наклон в кросовата посока, не само по билото. 0.1 позволява
# малко числено отклонение от точното условие ∇φ·e_cross=0 (сплайнът
# е приближение), но хваща истинско отклонение от билната/долинна
# линия.
ALIGNMENT_RELATIVE_TOLERANCE = 0.1

# Разстояние, под което маршируването се счита за "стигнало" до друга
# (различна от анкера) вече класифицирана критична точка -- статус
# "reached_critical_point". Съизмеримо с DEDUPLICATION_DISTANCE от
# analysis/critical_points.py (същия ред на "практически едно и също
# място" в тези координати).
REACHED_CRITICAL_POINT_DISTANCE = 0.25

# Брой стъпки, които трябва да изминат в дадена посока, преди новата
# точка изобщо да се сравнява с по-стари точки от СЪЩАТА посока за
# self-intersection -- без този пропуск всяка гладка крива би
# "открила" фалшиво пресичане само защото съседните ѝ стъпки
# естествено са близо една до друга.
SELF_INTERSECTION_MIN_STEP_GAP = 20

# Разстояние, под което текуща точка се счита за връщане към по-стара
# точка от СЪЩАТА посока на маршируване -- статус "self_intersection".
# 2 * STEP_SIZE: по-близо от две стъпки означава реално затваряне на
# контур, не просто естествена кривина на пътя.
SELF_INTERSECTION_DISTANCE = 2 * STEP_SIZE

# Толеранс за избор на детерминиран начален знак на along-собствения
# вектор в анкера (виж _canonical_direction_sign) -- под този праг
# x-компонентата се счита за практически нула и вместо нея се използва
# знакът на y-компонентата за развалата на връзката.
SIGN_TIE_BREAK_TOLERANCE = 1e-9

# Пост-трасиращ качествен филтър (assess_ridge_valley_quality) --
# аналогичен на analysis/critical_point_quality.py, но на ниво цяла
# верига, а не отделна точка.
CHAIN_BOUNDARY_MARGIN = 0.3
MIN_CROSS_CURVATURE_MAGNITUDE = 0.05
MIN_CHAIN_POINTS = 3


def compute_symmetric_eigenvectors(
    f_xx: float,
    mixed: float,
    f_yy: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """
    Затворена форма (Jacobi ъгъл на въртене) за собствените вектори на
    симетричната 2x2 матрица [[f_xx, mixed], [mixed, f_yy]] -- същата
    матрица, чиито собствени ЧИСЛА вече връща
    analysis.critical_points.compute_symmetric_eigenvalues.

        theta = 0.5 * atan2(2*mixed, f_xx - f_yy)
        e_max = (cos theta, sin theta)     -- за eigenvalue_max
        e_min = (-sin theta, cos theta)    -- за eigenvalue_min

    Използва atan2 вместо директно деление, за да остане устойчива
    дори когато mixed == 0 (диагонален Hessian).

    Връща (e_max, e_min) като единични вектори -- собствени вектори са
    дефинирани само с точност до знак; тази функция НЕ прави избор на
    знак съгласуван между съседни точки -- това е грижа на
    _canonical_direction_sign / _consistent_direction в извикващия
    код, не на самата затворена форма.

    Връща None точно в изотропната изродена точка (f_xx == f_yy И
    mixed == 0, в рамките на ISOTROPIC_HESSIAN_TOLERANCE) -- там
    atan2(0, 0) е математически недефинирано и всяка посока е
    собствен вектор; билна/долинна посока не съществува в такава
    точка.
    """

    if (
        abs(f_xx - f_yy) < ISOTROPIC_HESSIAN_TOLERANCE
        and abs(mixed) < ISOTROPIC_HESSIAN_TOLERANCE
    ):
        return None

    theta = 0.5 * math.atan2(2.0 * mixed, f_xx - f_yy)

    e_max = (math.cos(theta), math.sin(theta))
    e_min = (-math.sin(theta), math.cos(theta))

    return e_max, e_min


def _canonical_direction_sign(
    vector: tuple[float, float],
) -> tuple[float, float]:
    """
    Избира детерминиран знак за посока, дефинирана само с точност до
    знак (собствен вектор) -- предпочита положителна x-компонента;
    ако x е практически нула, предпочита положителна y-компонента.

    Използва се САМО за да фиксира изходната посока при първата
    стъпка от даден анкер (нужно за детерминизъм -- иначе изборът на
    начален знак би зависел от произволни особености на затворената
    форма). Всяка следваща стъпка вместо това пази съгласуваност
    спрямо предходната действителна стъпка (виж _consistent_direction),
    не спрямо това правило.
    """

    x, y = vector

    if abs(x) > SIGN_TIE_BREAK_TOLERANCE:
        return vector if x > 0 else (-x, -y)

    return vector if y >= 0 else (-x, -y)


def _consistent_direction(
    vector: tuple[float, float],
    previous_direction: tuple[float, float],
) -> tuple[float, float]:
    """
    Избира знака на vector (собствен вектор, дефиниран само с
    точност до знак), който продължава НАПРЕД спрямо
    previous_direction -- т.е. знака, за който скаларното
    произведение с previous_direction е неотрицателно.

    Пази маршируването от "скачане назад" стъпка по стъпка само
    защото затворената форма на собствения вектор
    (compute_symmetric_eigenvectors) може произволно да смени знак
    между съседни точки -- eigenvector(-p) е математически същата
    посока като eigenvector(p), но не е гарантирано да излезе със
    същия знак от atan2/cos/sin.
    """

    x, y = vector

    dot = x * previous_direction[0] + y * previous_direction[1]

    if dot < 0:
        return (-x, -y)

    return vector


def _evaluate_at_point(
    spline,
    x: float,
    y: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[float, float, float, float, float]:
    """
    Връща (grad_x, grad_y, f_xx, mixed, f_yy) в една-единствена точка
    (x, y), директно върху непрекъснатия сплайн -- без никаква нова
    координатна трансформация. mixed е симетризираната смесена
    производна (f_xy + f_yx) / 2, същото съглашение, което вече
    използва analysis.critical_points.classify_critical_points.
    """

    grad_x = float(spline(y, x, dx=0, dy=1)[0, 0])
    grad_y = float(spline(y, x, dx=1, dy=0)[0, 0])

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
    mixed_value = 0.5 * (float(f_xy[0, 0]) + float(f_yx[0, 0]))

    return grad_x, grad_y, f_xx_value, mixed_value, f_yy_value


def _cross_and_along(
    kind: str,
    eigenvalue_max: float,
    eigenvalue_min: float,
    e_max: tuple[float, float],
    e_min: tuple[float, float],
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """
    Избира кросовата (перпендикулярна на билото/долината) и along
    (по билото/долината) двойки (собствено число, собствен вектор) в
    зависимост от kind -- единствената точка в модула, в която ridge
    и valley логиката се разминават, за да остане симетрията иначе
    навсякъде другаде явна и споделена.

    "ridge"  -- кросова посока = eigenvalue_min / e_min (изисква се
                отрицателна, като при maximum); along = eigenvalue_max
                / e_max.
    "valley" -- точният дуал: кросова = eigenvalue_max / e_max
                (изисква се положителна, като при minimum); along =
                eigenvalue_min / e_min.

    Връща (cross_eigenvalue, e_cross, e_along).
    """

    if kind == "ridge":
        return eigenvalue_min, e_min, e_max

    if kind == "valley":
        return eigenvalue_max, e_max, e_min

    raise ValueError(f"kind must be 'ridge' or 'valley', got {kind!r}")


def _is_cross_curvature_valid(kind: str, cross_eigenvalue: float) -> bool:
    """
    Проверява дали cross_eigenvalue все още пази знака, изискван от
    kind, отвъд числения шум (CROSS_EIGENVALUE_COLLAPSE_THRESHOLD):
    отрицателно за "ridge", положително за "valley". Общата
    (по абсолютна стойност) граница пази симетрията изрична, вместо
    два отделни, лесни за разминаване прага.
    """

    if kind == "ridge":
        return cross_eigenvalue < -CROSS_EIGENVALUE_COLLAPSE_THRESHOLD

    return cross_eigenvalue > CROSS_EIGENVALUE_COLLAPSE_THRESHOLD


def check_self_intersection(
    candidate_x: float,
    candidate_y: float,
    visited: list[tuple[float, float]],
    min_step_gap: int = SELF_INTERSECTION_MIN_STEP_GAP,
    distance_threshold: float = SELF_INTERSECTION_DISTANCE,
) -> bool:
    """
    True ако (candidate_x, candidate_y) е достатъчно близо
    (<= distance_threshold) до точка от visited, поне min_step_gap
    записа назад от края на visited -- т.е. маршируването се връща
    към собствения си по-стар път, вместо да продължава напред.

    Изрично изолирана като самостоятелна, чиста функция, за да може
    механизмът за self-intersection да се тества директно със
    синтетични visited списъци, без да е нужно реално поле, чиято
    собствена-векторна посока действително да образува контур.
    """

    if len(visited) <= min_step_gap:
        return False

    for old_x, old_y in visited[: len(visited) - min_step_gap]:
        if math.hypot(candidate_x - old_x, candidate_y - old_y) <= distance_threshold:
            return True

    return False


def _trace_direction(
    spline,
    anchor_x: float,
    anchor_y: float,
    initial_direction: tuple[float, float],
    kind: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    other_critical_points: list[tuple[float, float]],
    step_size: float = STEP_SIZE,
    max_steps: int = MAX_MARCH_STEPS,
) -> tuple[list[RidgeValleyPoint], str]:
    """
    Маршира в ЕДНА посока от (anchor_x, anchor_y), следвайки на всяка
    стъпка along-собствения вектор на Hessian-а в текущата точка,
    пазейки съгласуваност на знака спрямо предходната действителна
    стъпка (_consistent_direction). Анкерната точка НЕ се включва в
    резултата (тя се добавя отделно, веднъж, при сглобяването на
    пълната верига в _trace_chain_from_anchor) -- връща само новите
    точки от тази посока, в реда, в който са изминати (от анкера
    навън), плюс явна причина за спиране.

    Всяка кандидатна нова точка се проверява ПРЕДИ да бъде приета
    (домейн -> кривина -> подравняване -> reconnection -> self-
    intersection); неуспешна проверка спира трасирането веднага и
    връща причината -- провалилата се точка НЕ се добавя към
    резултата, точно както CriticalPointCandidate.status не преправя
    последната валидна точка при Newton уточняването.
    """

    points: list[RidgeValleyPoint] = []
    visited: list[tuple[float, float]] = [(anchor_x, anchor_y)]

    x, y = anchor_x, anchor_y
    direction = initial_direction

    for _step in range(max_steps):
        new_x = x + step_size * direction[0]
        new_y = y + step_size * direction[1]

        if not (x_min <= new_x <= x_max) or not (y_min <= new_y <= y_max):
            return points, "left_domain"

        grad_x, grad_y, f_xx, mixed, f_yy = _evaluate_at_point(
            spline, new_x, new_y, x_min, x_max, y_min, y_max
        )

        eigenvalue_a, eigenvalue_b, _determinant, _trace = (
            compute_symmetric_eigenvalues(f_xx, mixed, f_yy)
        )

        eigenvectors = compute_symmetric_eigenvectors(f_xx, mixed, f_yy)
        if eigenvectors is None:
            return points, "isotropic_point"

        e_max, e_min = eigenvectors

        cross_eigenvalue, e_cross, e_along = _cross_and_along(
            kind, eigenvalue_a, eigenvalue_b, e_max, e_min
        )

        if not _is_cross_curvature_valid(kind, cross_eigenvalue):
            return points, "weak_curvature"

        gradient_norm = math.hypot(grad_x, grad_y)
        alignment_residual = abs(grad_x * e_cross[0] + grad_y * e_cross[1])

        if gradient_norm >= MIN_GRADIENT_FOR_ALIGNMENT_CHECK:
            if alignment_residual / gradient_norm > ALIGNMENT_RELATIVE_TOLERANCE:
                return points, "misaligned"

        for other_x, other_y in other_critical_points:
            if math.hypot(new_x - other_x, new_y - other_y) <= REACHED_CRITICAL_POINT_DISTANCE:
                return points, "reached_critical_point"

        if check_self_intersection(new_x, new_y, visited):
            return points, "self_intersection"

        value = float(spline(new_y, new_x)[0, 0])

        points.append(
            RidgeValleyPoint(
                x=new_x,
                y=new_y,
                value=value,
                eigenvalue_cross=cross_eigenvalue,
                alignment_residual=alignment_residual,
                kind=kind,
            )
        )

        direction = _consistent_direction(e_along, direction)
        visited.append((new_x, new_y))
        x, y = new_x, new_y

    return points, "max_steps_reached"


def _trace_chain_from_anchor(
    surface: AttackInfluenceSurface,
    anchor: ClassifiedCriticalPoint,
    kind: str,
    other_points: list[ClassifiedCriticalPoint],
    step_size: float = STEP_SIZE,
    max_steps: int = MAX_MARCH_STEPS,
) -> RidgeValleyChain:
    """
    Сглобява пълната RidgeValleyChain за един анкер: анкерната точка
    тривиално удовлетворява ridge/valley условието (градиентът там е
    ~0 във всяка посока, включително кросовата, а изискваното
    собствено число вече има нужния знак по конструкция на
    "maximum"/"minimum" класификацията) -- затова се добавя директно,
    без да минава през _trace_direction.

    Двете посоки (напред/назад) тръгват с точно противоположен
    начален знак на along-собствения вектор в анкера
    (_canonical_direction_sign за едната, обратното за другата), после
    всяка продължава независимо, пазейки собствена съгласуваност на
    знака спрямо предходната си стъпка.
    """

    spline = surface.spline
    x_min, x_max = float(surface.x[0]), float(surface.x[-1])
    y_min, y_max = float(surface.y[0]), float(surface.y[-1])

    anchor_cross_eigenvalue = (
        anchor.eigenvalue_min if kind == "ridge" else anchor.eigenvalue_max
    )

    anchor_point = RidgeValleyPoint(
        x=anchor.x,
        y=anchor.y,
        value=anchor.value,
        eigenvalue_cross=anchor_cross_eigenvalue,
        alignment_residual=0.0,
        kind=kind,
    )

    mixed = 0.5 * (anchor.f_xy + anchor.f_yx)
    eigenvectors = compute_symmetric_eigenvectors(anchor.f_xx, mixed, anchor.f_yy)

    if eigenvectors is None:
        return RidgeValleyChain(
            kind=kind,
            points=[anchor_point],
            anchor=anchor,
            start_status="isotropic_point",
            end_status="isotropic_point",
        )

    e_max, e_min = eigenvectors
    e_along = e_max if kind == "ridge" else e_min

    forward_direction = _canonical_direction_sign(e_along)
    backward_direction = (-forward_direction[0], -forward_direction[1])

    other_critical_point_coords = [(point.x, point.y) for point in other_points]

    forward_points, forward_status = _trace_direction(
        spline,
        anchor.x,
        anchor.y,
        forward_direction,
        kind,
        x_min,
        x_max,
        y_min,
        y_max,
        other_critical_point_coords,
        step_size,
        max_steps,
    )

    backward_points, backward_status = _trace_direction(
        spline,
        anchor.x,
        anchor.y,
        backward_direction,
        kind,
        x_min,
        x_max,
        y_min,
        y_max,
        other_critical_point_coords,
        step_size,
        max_steps,
    )

    ordered_points = list(reversed(backward_points)) + [anchor_point] + forward_points

    return RidgeValleyChain(
        kind=kind,
        points=ordered_points,
        anchor=anchor,
        start_status=backward_status,
        end_status=forward_status,
    )


def locate_ridge_valley_chains(
    surface: AttackInfluenceSurface,
    classified_points: list[ClassifiedCriticalPoint],
    kind: str,
    step_size: float = STEP_SIZE,
    max_steps: int = MAX_MARCH_STEPS,
) -> list[RidgeValleyChain]:
    """
    Тръгва от всяка вече класифицирана точка от нужния тип ("maximum"
    за kind="ridge", "minimum" за kind="valley") в classified_points и
    я маршира в двете посоки, връщайки по една RidgeValleyChain на
    анкер.

    НЕ извиква сам classify_critical_points или
    assess_critical_point_quality -- очаква вече изчислен резултат от
    caller-а (както classify_critical_points очаква вече изчислени
    locate_critical_points кандидати). Ако caller-ът иска само
    "качествени" анкери, филтрирането по CriticalPointQualityAssessment
    .is_accepted е негова грижа преди извикването -- този модул не
    налага собствено мнение по въпроса кой критична точка е достатъчно
    надеждна, за да зачене верига.

    Не филтрира по качество на самите вериги -- виж
    assess_ridge_valley_quality за пост-трасиращия филтър, отделна
    стъпка, аналогична на localize/classify спрямо
    critical_point_quality в Раздел 9.

    Не променя surface или classified_points по никакъв начин.
    """

    if kind not in ("ridge", "valley"):
        raise ValueError(f"kind must be 'ridge' or 'valley', got {kind!r}")

    anchor_classification = "maximum" if kind == "ridge" else "minimum"

    anchors = [
        point
        for point in classified_points
        if point.status == "converged" and point.classification == anchor_classification
    ]

    converged_points = [
        point for point in classified_points if point.status == "converged"
    ]

    chains: list[RidgeValleyChain] = []

    for anchor in anchors:
        other_points = [point for point in converged_points if point is not anchor]

        chain = _trace_chain_from_anchor(
            surface, anchor, kind, other_points, step_size, max_steps
        )
        chains.append(chain)

    return chains


def assess_ridge_valley_quality(
    chains: list[RidgeValleyChain],
    surface: AttackInfluenceSurface,
    boundary_margin: float = CHAIN_BOUNDARY_MARGIN,
    min_curvature_magnitude: float = MIN_CROSS_CURVATURE_MAGNITUDE,
    min_points: int = MIN_CHAIN_POINTS,
) -> list[RidgeValleyQualityAssessment]:
    """
    Пост-трасираща стъпка за интерпретируемост -- НЕ променя
    trасирането (locate_ridge_valley_chains) по никакъв начин.
    Аналогична на analysis.critical_point_quality
    .assess_critical_point_quality, но на ниво цяла верига: разделя
    "математически трасирана верига по напаснатия сплайн" от "верига,
    достатъчно надеждна, за да се представи като шахматно значима".

    Всяка входна верига се запазва в резултата непроменена (обвита в
    RidgeValleyQualityAssessment) -- функцията никога тихо не изтрива
    информация, само я анотира. Всяка отхвърлена верига носи ВСИЧКИ
    провалени критерии, не само първия.

    Приложени критерии:

    1. Брой точки във веригата < min_points -- твърде къса, за да
       представлява нещо повече от самата анкерна критична точка.
    2. Повече от половината точки от веригата са на разстояние <
       boundary_margin от ръба на дъската -- веригата прекарва
       по-голямата част от дължината си в зоната, за която Раздел 10
       на docs/mathematics.md предупреждава, че е по-вероятно артефакт
       на сплайн-напасването, отколкото шахматна структура.
    3. Средната стойност на |eigenvalue_cross| по веригата <
       min_curvature_magnitude -- кривината в кросовата посока е като
       цяло твърде слаба, за да е различима от числения шум на
       недосемплирания 8x8 сплайн (аналогично на
       MIN_EIGENVALUE_MAGNITUDE в critical_point_quality.py).
    """

    x_min, x_max = float(surface.x[0]), float(surface.x[-1])
    y_min, y_max = float(surface.y[0]), float(surface.y[-1])

    assessments: list[RidgeValleyQualityAssessment] = []

    for chain in chains:
        reasons: list[str] = []

        point_count = len(chain.points)

        if point_count < min_points:
            reasons.append(
                "chain too short "
                f"(points={point_count} < min_points={min_points})"
            )

        if point_count > 0:
            distances = [
                min(
                    point.x - x_min,
                    x_max - point.x,
                    point.y - y_min,
                    y_max - point.y,
                )
                for point in chain.points
            ]

            near_boundary_fraction = sum(
                1 for distance in distances if distance < boundary_margin
            ) / point_count

            if near_boundary_fraction > 0.5:
                reasons.append(
                    "chain spends most of its length near the board boundary "
                    f"(fraction={near_boundary_fraction:.2f} > 0.5, "
                    f"boundary_margin={boundary_margin:.4f})"
                )

            average_curvature = sum(
                abs(point.eigenvalue_cross) for point in chain.points
            ) / point_count

            if average_curvature < min_curvature_magnitude:
                reasons.append(
                    "average cross-direction curvature too weak "
                    f"(|eigenvalue_cross| avg={average_curvature:.6f} "
                    f"< min_curvature_magnitude={min_curvature_magnitude:.6f})"
                )

        assessments.append(
            RidgeValleyQualityAssessment(
                chain=chain,
                is_accepted=(len(reasons) == 0),
                rejection_reasons=reasons,
            )
        )

    return assessments
