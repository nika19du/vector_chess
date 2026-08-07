import math
from dataclasses import dataclass

from analysis.ridge_valley import check_self_intersection, compute_symmetric_eigenvectors
from chess_engine.models import (
    AttackInfluenceSurface,
    ClassifiedCriticalPoint,
    CriticalPointQualityAssessment,
    MorseSmaleCell,
    MorseSmaleCellGeometry,
    MorseSmaleCellQualityAssessment,
    MorseSmaleCellRejectionReason,
    MorseSmaleCellRejectionReasonKind,
    MorseSmaleComplex,
    SeparatrixPath,
    SeparatrixPoint,
    TopologyIssue,
    TopologyIssueKind,
)


# Дължина на една стъпка на маршируването, в екранни координати
# (клетка = 1.0 единица). Умишлено СЪЩАТА стойност като STEP_SIZE в
# analysis/ridge_valley.py -- еднакво фина резолюция и за двата вида
# трасиране по повърхността.
STEP_SIZE = 0.05

# Максимален брой стъпки на ЕДНА посока (един клон), преди статус
# "max_steps_reached". Същата стойност и обосновка като MAX_MARCH_STEPS
# в analysis/ridge_valley.py.
MAX_MARCH_STEPS = 500

# Разстояние, под което маршируването се счита за "стигнало" до друга
# (различна от start_saddle) вече класифицирана критична точка.
# Съизмеримо с REACHED_CRITICAL_POINT_DISTANCE в
# analysis/ridge_valley.py -- същия ред на "практически едно и също
# място" в тези координати.
REACHED_CRITICAL_POINT_DISTANCE = 0.25

# Ако |∇φ| в текущата точка падне под този праг, посоката на потока
# (нормализираният градиент) вече е числено недефинирана -- статус
# "gradient_stagnation". Съизмерим по ред на величината с
# MIN_GRADIENT_FOR_ALIGNMENT_CHECK в analysis/ridge_valley.py (същата
# обосновка: под това ниво градиентът е неразличим от числения шум).
GRADIENT_STAGNATION_THRESHOLD = 1e-4

# Брой стъпки, които трябва да изминат в дадена посока, преди новата
# точка изобщо да се сравнява с по-стари точки от СЪЩИЯ клон за
# self-intersection. Реекспортирано поведение -- виж
# check_self_intersection по-долу, внесена директно от
# analysis/ridge_valley.py (публична, споделена функция, не
# preизобретена тук).


def _evaluate_gradient_at_point(
    spline,
    x: float,
    y: float,
) -> tuple[float, float]:
    """
    Аналитичният градиент на напаснатия сплайн в ЕДНА непрекъсната
    точка -- същата математическа дефиниция, която
    analysis.attack_influence_surface.sample_gradient оценява върху
    цялата гъста мрежа на повърхността (spline(y, x, dx=0, dy=1) /
    dx=1, dy=0, без y-flip). sample_gradient самата не приема
    произволна (x, y), само surface.x/surface.y -- затова
    маршируването тук пресмята същата производна поточково, по
    същия модел на повторна употреба, който вече използва
    analysis.ridge_valley._evaluate_at_point за своята Hessian-оценка
    в точка.
    """

    grad_x = float(spline(y, x, dx=0, dy=1)[0, 0])
    grad_y = float(spline(y, x, dx=1, dy=0)[0, 0])

    return grad_x, grad_y


def _normalize(
    vector_x: float,
    vector_y: float,
) -> tuple[float, float]:
    """
    Единичен вектор в посоката (vector_x, vector_y). Извиква се само
    след като градиентната норма вече е проверена срещу
    GRADIENT_STAGNATION_THRESHOLD, затова делението никога не е
    близо до нула тук.
    """

    norm = math.hypot(vector_x, vector_y)

    return vector_x / norm, vector_y / norm


def _polyline_length(
    anchor_x: float,
    anchor_y: float,
    points: list[SeparatrixPoint],
) -> float:
    """
    Сборът от евклидовите разстояния между последователните точки на
    клона, започвайки от анкера (start_saddle) -- изчислен директно от
    точките, а не по формула step_count * step_size, за да остане
    коректен дори ако бъдещо разширение направи стъпката адаптивна.
    """

    if not points:
        return 0.0

    total = math.hypot(points[0].x - anchor_x, points[0].y - anchor_y)

    for previous_point, current_point in zip(points, points[1:]):
        total += math.hypot(
            current_point.x - previous_point.x,
            current_point.y - previous_point.y,
        )

    return total


def _trace_gradient_flow_direction(
    spline,
    anchor_x: float,
    anchor_y: float,
    initial_direction: tuple[float, float],
    flow_direction: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    accepted_other_points: list[ClassifiedCriticalPoint],
    rejected_points: list[ClassifiedCriticalPoint],
    step_size: float = STEP_SIZE,
    max_steps: int = MAX_MARCH_STEPS,
) -> tuple[list[SeparatrixPoint], str, ClassifiedCriticalPoint | None]:
    """
    Маршира в ЕДНА посока (един клон) от (anchor_x, anchor_y): първата
    стъпка тръгва по initial_direction (собствения вектор на Hessian-а
    в седлото -- градиентът в самото седло е ~0 по конструкция, затова
    посоката там не може да идва от градиента), всяка следваща стъпка
    следва нормализирания (+ или -, според flow_direction) аналитичен
    градиент в текущата точка -- детерминирана експлицитна интеграция
    с фиксирана дължина на стъпката, никога зависима от произволен
    избор на знак (за разлика от собствения вектор в
    analysis/ridge_valley.py, градиентната посока е еднозначна --
    +∇φ или -∇φ, без нужда от _consistent_direction).

    Анкерната точка НЕ се включва в резултата -- връща само новите
    точки от този клон, в реда, в който са изминати, плюс явна причина
    за спиране и, ако е приложимо, критичната точка, до която е
    стигнало трасирането.

    Всяка кандидатна нова точка се проверява ПРЕДИ да бъде приета
    (домейн -> достигната приета критична точка -> достигната
    отхвърлена/ненадеждна критична точка -> self-intersection ->
    стагнация на градиента); неуспешна проверка спира трасирането
    веднага и връща причината -- провалилата се точка НЕ се добавя
    към резултата.
    """

    points: list[SeparatrixPoint] = []
    visited: list[tuple[float, float]] = [(anchor_x, anchor_y)]

    x, y = anchor_x, anchor_y
    direction = initial_direction

    sign = 1.0 if flow_direction == "ascending" else -1.0

    for _step in range(max_steps):
        new_x = x + step_size * direction[0]
        new_y = y + step_size * direction[1]

        if not (x_min <= new_x <= x_max) or not (y_min <= new_y <= y_max):
            return points, "left_domain", None

        for other_point in accepted_other_points:
            distance = math.hypot(new_x - other_point.x, new_y - other_point.y)
            if distance <= REACHED_CRITICAL_POINT_DISTANCE:
                return points, "reached_critical_point", other_point

        for other_point in rejected_points:
            distance = math.hypot(new_x - other_point.x, new_y - other_point.y)
            if distance <= REACHED_CRITICAL_POINT_DISTANCE:
                return points, "reached_unreliable_point", other_point

        if check_self_intersection(new_x, new_y, visited):
            return points, "self_intersection", None

        grad_x, grad_y = _evaluate_gradient_at_point(spline, new_x, new_y)
        gradient_norm = math.hypot(grad_x, grad_y)

        if gradient_norm < GRADIENT_STAGNATION_THRESHOLD:
            return points, "gradient_stagnation", None

        value = float(spline(new_y, new_x)[0, 0])

        points.append(
            SeparatrixPoint(
                x=new_x,
                y=new_y,
                value=value,
            )
        )

        direction = _normalize(sign * grad_x, sign * grad_y)
        visited.append((new_x, new_y))
        x, y = new_x, new_y

    return points, "max_steps_reached", None


def _trace_separatrices_from_saddle(
    surface: AttackInfluenceSurface,
    anchor: ClassifiedCriticalPoint,
    accepted_other_points: list[ClassifiedCriticalPoint],
    rejected_points: list[ClassifiedCriticalPoint],
    step_size: float = STEP_SIZE,
    max_steps: int = MAX_MARCH_STEPS,
) -> list[SeparatrixPath]:
    """
    Построява четирите SeparatrixPath за едно седло: 2 възходящи клона
    (по + и - посоката на собствения вектор на eigenvalue_max, следвани
    от +∇φ) и 2 низходящи клона (по + и - посоката на собствения
    вектор на eigenvalue_min, следвани от -∇φ).

    eigenvalue_max/eigenvalue_min идват директно от вече изчислените
    полета на anchor (classify_critical_points ги е попълнила) -- не
    се преизчисляват тук. За истинско седло eigenvalue_max > 0 >
    eigenvalue_min по конструкция на класификацията, затова e_max е
    винаги възходящата (unstable) посока, e_min винаги низходящата
    (stable) -- без нужда от допълнителна проверка на знака.

    Ако Hessian-ът в седлото се окаже изотропен (compute_symmetric_
    eigenvectors връща None) -- математически неочаквано за истинско
    седло (изисква противоположни знаци на собствените числа, което
    изключва изотропност), но обработено явно, защото функцията може
    по договор да върне None -- връща 4 празни SeparatrixPath със
    status "isotropic_point", вместо да разпадне повикващия код.
    """

    x_min, x_max = float(surface.x[0]), float(surface.x[-1])
    y_min, y_max = float(surface.y[0]), float(surface.y[-1])

    mixed = 0.5 * (anchor.f_xy + anchor.f_yx)
    eigenvectors = compute_symmetric_eigenvectors(anchor.f_xx, mixed, anchor.f_yy)

    if eigenvectors is None:
        return [
            SeparatrixPath(
                start_saddle=anchor,
                flow_direction=flow_direction,
                points=[],
                end_critical_point=None,
                termination_status="isotropic_point",
                step_count=0,
                path_length=0.0,
            )
            for flow_direction in ("ascending", "ascending", "descending", "descending")
        ]

    e_max, e_min = eigenvectors

    branch_specs = (
        ("ascending", e_max),
        ("ascending", (-e_max[0], -e_max[1])),
        ("descending", e_min),
        ("descending", (-e_min[0], -e_min[1])),
    )

    paths: list[SeparatrixPath] = []

    for flow_direction, initial_direction in branch_specs:
        points, status, end_point = _trace_gradient_flow_direction(
            surface.spline,
            anchor.x,
            anchor.y,
            initial_direction,
            flow_direction,
            x_min,
            x_max,
            y_min,
            y_max,
            accepted_other_points,
            rejected_points,
            step_size,
            max_steps,
        )

        paths.append(
            SeparatrixPath(
                start_saddle=anchor,
                flow_direction=flow_direction,
                points=points,
                end_critical_point=end_point,
                termination_status=status,
                step_count=len(points),
                path_length=_polyline_length(anchor.x, anchor.y, points),
            )
        )

    return paths


def locate_morse_smale_separatrices(
    surface: AttackInfluenceSurface,
    assessments: list[CriticalPointQualityAssessment],
    step_size: float = STEP_SIZE,
    max_steps: int = MAX_MARCH_STEPS,
) -> list[SeparatrixPath]:
    """
    Тръгва от всяка качествено ПРИЕТА (is_accepted=True), класифицирана
    като "saddle" точка в assessments и я маршира в четирите ѝ клона
    (2 възходящи + 2 низходящи, виж _trace_separatrices_from_saddle),
    връщайки по 4 SeparatrixPath на седло.

    Само качествено приети критични точки могат да бъдат анкери --
    съответства на изискването "use only quality-accepted classified
    critical points as valid Morse-Smale anchors" и на общия принцип
    на critical_point_quality.py: не всяка математически открита
    критична точка на сплайна е достатъчно надеждна, за да носи
    Morse-Smale структура. Отхвърлените точки все пак участват в
    трасирането -- като ВЪЗМОЖНИ ЦЕЛИ ("reached_unreliable_point"), не
    като анкери -- трасирането спира там вместо тихо да продължи през
    ненадеждна точка.

    НЕ извиква сам classify_critical_points или
    assess_critical_point_quality -- очаква вече изчислен резултат от
    caller-а, същия модел като locate_ridge_valley_chains спрямо
    вече класифицираните точки.

    НЕ сглобява Morse-Smale клетки (2-cells) -- само 0-клетките
    (критичните точки, вече съществуващи от по-рано) и 1-клетките
    (сепаратрисите, върнати тук). Сглобяването на клетките е следваща
    фаза, извън обхвата на този модул.

    Не променя surface или assessments по никакъв начин.
    """

    accepted_points = [
        assessment.point
        for assessment in assessments
        if assessment.is_accepted and assessment.point.status == "converged"
    ]

    rejected_points = [
        assessment.point
        for assessment in assessments
        if not assessment.is_accepted and assessment.point.status == "converged"
    ]

    saddle_anchors = [
        point for point in accepted_points if point.classification == "saddle"
    ]

    paths: list[SeparatrixPath] = []

    for anchor in saddle_anchors:
        other_accepted = [point for point in accepted_points if point is not anchor]

        paths.extend(
            _trace_separatrices_from_saddle(
                surface,
                anchor,
                other_accepted,
                rejected_points,
                step_size,
                max_steps,
            )
        )

    return paths


# ===========================================================
# Фаза 2 -- сглобяване на 2-клетки (басейни) ИЗКЛЮЧИТЕЛНО от вече
# трасираните SeparatrixPath (locate_morse_smale_separatrices по-горе,
# непроменена). Стандартна планарно-графова конструкция ("rotation
# system" / DCEL face trace): всяка затворена сепаратриса поражда две
# полу-ребра (напред и назад); на всеки връх полу-ребрата, тръгващи от
# него, се подреждат по ъгъл; обхождането на границата на клетка
# продължава на всяка стъпка към СЛЕДВАЩОТО полу-ребро (по ъгъл) в
# точката, в която пристига -- стандартното "next" правило, гарантира
# че всяко полу-ребро принадлежи на точно едно лице (клетка).
# ===========================================================

# Праг за откриване на изродено (numerically unstable) ъглово
# подреждане -- две полу-ребра, тръгващи от един и същ връх, с ъглова
# разлика под този праг, се отбелязват като DEGENERATE_ANGULAR
# _ORDERING. Съизмерим по ред на величината с останалите "числов шум"
# прагове в модула (EIGENVALUE_ZERO_THRESHOLD и др. в
# analysis/critical_points.py).
ANGULAR_DEGENERACY_TOLERANCE = 1e-6


@dataclass
class _HalfEdge:
    """
    Вътрешно (не-експортирано) представяне на едно полу-ребро на
    планарния граф -- НЕ е математически обект от Фаза 1/чиста
    топология, а само счетоводна структура за обхождането на границите
    в тази фаза. Не се появява в chess_engine.models.

    origin:
        върхът, от който полу-реброто тръгва.

    destination:
        върхът, до който полу-реброто стига -- None означава
        "блокер": полу-ребро, представящо ОТВОРЕНА (незатворена)
        сепаратриса от седло -- обхождане, стигнало до него, спира
        (виж open_boundaries в MorseSmaleCell), никога не се "изминава"
        по-нататък.

    separatrix:
        оригиналната SeparatrixPath, която поражда това полу-ребро.

    is_forward:
        True -- полу-реброто следва separatrix от start_saddle към
        end_critical_point; False -- обратната посока (само за
        затворени сепаратриси; блокерите винаги са is_forward=True,
        тръгвайки от start_saddle).

    angle:
        посоката (atan2), в която полу-реброто напуска origin --
        използва се единствено за циклично подреждане около всеки
        връх.
    """

    origin: ClassifiedCriticalPoint
    destination: ClassifiedCriticalPoint | None
    separatrix: SeparatrixPath
    is_forward: bool
    angle: float


def _forward_leaving_direction(
    vertex: ClassifiedCriticalPoint,
    separatrix: SeparatrixPath,
) -> tuple[float, float] | None:
    """
    Посоката, в която separatrix напуска vertex (== separatrix
    .start_saddle), в реда: (1) действително проследената първа точка
    (points[0]) -- най-точна, отразява реалния път; (2)
    end_critical_point, ако сепаратрисата е затворена, но е стигнала
    целта си на самата първа стъпка (points е празен, "нулева дължина",
    виж ZERO_LENGTH_EDGE); (3) собственият вектор на Hessian-а на
    седлото по правилната ос (eigenvalue_max за "ascending",
    eigenvalue_min за "descending") -- само когато сепаратрисата няма
    НИТО една проследена точка, НИТО достигната цел (напр. "left
    _domain" или "gradient_stagnation" на самата първа стъпка, или
    "isotropic_point"). Тази стъпка (3) е реконструкция, не действително
    проследена стойност -- знакът (+ или -) на собствения вектор не се
    възстановява от separatrix (Фаза 1 не го пази изрично), затова се
    връща суровият знак на compute_symmetric_eigenvectors, който е
    детерминиран, но не гарантирано съвпада с оригиналния знак,
    избран при трасирането. Рядък ъглов случай (седло на по-малко от
    една стъпка от ръба на дъската, чиято първа стъпка веднага напуска
    домейна) -- документиран тук, не пренаписва математиката на
    Фаза 1.

    Връща None само ако дори (3) е недефинирана (изотропен Hessian в
    седлото) -- полу-реброто просто не участва в ъгловото подреждане
    (безвредно: изотропно седло няма затворени клонове, до които изобщо
    да пристигне обхождане, затова не е нужно да бъде подредено спрямо
    нещо).
    """

    if separatrix.points:
        target = separatrix.points[0]
        return (target.x - vertex.x, target.y - vertex.y)

    if separatrix.end_critical_point is not None:
        target = separatrix.end_critical_point
        return (target.x - vertex.x, target.y - vertex.y)

    mixed = 0.5 * (vertex.f_xy + vertex.f_yx)
    eigenvectors = compute_symmetric_eigenvectors(vertex.f_xx, mixed, vertex.f_yy)

    if eigenvectors is None:
        return None

    e_max, e_min = eigenvectors

    return e_max if separatrix.flow_direction == "ascending" else e_min


def _backward_leaving_direction(
    vertex: ClassifiedCriticalPoint,
    separatrix: SeparatrixPath,
) -> tuple[float, float]:
    """
    Посоката, в която separatrix (винаги ЗАТВОРЕНА тук -- само
    затворени сепаратриси пораждат "назад" полу-ребро) напуска
    vertex (== separatrix.end_critical_point), обратно към start
    _saddle: действително проследената последна точка (points[-1]),
    или -- за "нулева дължина" сепаратриса без точки -- директно
    към start_saddle.
    """

    if separatrix.points:
        target = separatrix.points[-1]
    else:
        target = separatrix.start_saddle

    return (target.x - vertex.x, target.y - vertex.y)


def _angle_difference(first: float, second: float) -> float:
    """Най-малката разлика между два ъгъла по кръг (0..pi)."""

    difference = abs(first - second) % (2 * math.pi)

    return min(difference, 2 * math.pi - difference)


def _find_index_by_identity(items: list[_HalfEdge], target: _HalfEdge) -> int:
    """
    Индексът на target в items по ИДЕНТИЧНОСТ (`is`), не по `==` --
    _HalfEdge не носи собствена идентичност-съобразена == семантика, а
    сравняване по стойност би могло погрешно да съвпадне две различни
    полу-ребра с случайно еднакви полета (напр. две "нулева дължина"
    сепаратриси). Съответства на identity contract-а на
    ClassifiedCriticalPoint, документиран в chess_engine/models.py.
    """

    for index, item in enumerate(items):
        if item is target:
            return index

    raise ValueError("half-edge not found in its own vertex's cyclic order")


def _build_half_edges(
    separatrices: list[SeparatrixPath],
) -> tuple[
    dict[int, list[_HalfEdge]],
    dict[int, ClassifiedCriticalPoint],
    list[_HalfEdge],
    list[TopologyIssue],
]:
    """
    Изгражда edges_at (id(връх) -> цикличо подредени по ъгъл полу-
    ребра, тръгващи от него, ВКЛЮЧИТЕЛНО блокери за отворени
    сепаратриси от седла -- виж _HalfEdge.destination), vertex_by_id
    (id(връх) -> самия обект), ordered_closed_half_edges (плосък
    списък от ВСИЧКИ полу-ребра на затворени сепаратриси, в
    ДЕТЕРМИНИРАН ред: за всяка separatrices[i] по входния ред -- първо
    нейното "напред" полу-ребро, после "назад" -- използван по-късно
    като фиксиран ред за сканиране на начални полу-ребра при
    обхождането на границите) и списък от TopologyIssue, засечени по
    време на изграждането (branch count mismatch, saddle-to-saddle,
    zero-length edge, degenerate angular ordering).

    Не променя separatrices или каквото и да е негово поле.
    """

    edges_at: dict[int, list[_HalfEdge]] = {}
    vertex_by_id: dict[int, ClassifiedCriticalPoint] = {}
    ordered_closed_half_edges: list[_HalfEdge] = []
    topology_issues: list[TopologyIssue] = []

    def register_vertex(vertex: ClassifiedCriticalPoint) -> None:
        vertex_by_id.setdefault(id(vertex), vertex)
        edges_at.setdefault(id(vertex), [])

    branch_counts: dict[int, int] = {}

    for separatrix in separatrices:
        register_vertex(separatrix.start_saddle)
        branch_counts[id(separatrix.start_saddle)] = (
            branch_counts.get(id(separatrix.start_saddle), 0) + 1
        )

    for saddle_id, count in branch_counts.items():
        if count != 4:
            topology_issues.append(
                TopologyIssue(
                    kind=TopologyIssueKind.SADDLE_BRANCH_COUNT_MISMATCH,
                    critical_point=vertex_by_id[saddle_id],
                    separatrix=None,
                    detail=(
                        f"saddle has {count} separatrices in the input, expected 4"
                    ),
                )
            )

    for separatrix in separatrices:
        start = separatrix.start_saddle
        is_closed = (
            separatrix.termination_status == "reached_critical_point"
            and separatrix.end_critical_point is not None
        )

        forward_direction = _forward_leaving_direction(start, separatrix)

        forward_half_edge: _HalfEdge | None = None

        if forward_direction is not None:
            forward_half_edge = _HalfEdge(
                origin=start,
                destination=(separatrix.end_critical_point if is_closed else None),
                separatrix=separatrix,
                is_forward=True,
                angle=math.atan2(forward_direction[1], forward_direction[0]),
            )
            edges_at[id(start)].append(forward_half_edge)

        if not is_closed:
            continue

        end = separatrix.end_critical_point
        register_vertex(end)

        backward_direction = _backward_leaving_direction(end, separatrix)
        backward_half_edge = _HalfEdge(
            origin=end,
            destination=start,
            separatrix=separatrix,
            is_forward=False,
            angle=math.atan2(backward_direction[1], backward_direction[0]),
        )
        edges_at[id(end)].append(backward_half_edge)

        if forward_half_edge is not None:
            ordered_closed_half_edges.append(forward_half_edge)
        ordered_closed_half_edges.append(backward_half_edge)

        if end.classification == "saddle":
            topology_issues.append(
                TopologyIssue(
                    kind=TopologyIssueKind.SADDLE_TO_SADDLE_CONNECTIVITY,
                    critical_point=None,
                    separatrix=separatrix,
                    detail=(
                        f"separatrix from saddle at ({start.x:.3f}, {start.y:.3f}) "
                        f"reached another saddle at ({end.x:.3f}, {end.y:.3f})"
                    ),
                )
            )

        if separatrix.step_count == 0:
            topology_issues.append(
                TopologyIssue(
                    kind=TopologyIssueKind.ZERO_LENGTH_EDGE,
                    critical_point=None,
                    separatrix=separatrix,
                    detail=(
                        "closed separatrix reached its target on the very first step "
                        f"(saddle at ({start.x:.3f}, {start.y:.3f}))"
                    ),
                )
            )

    for vertex_id, half_edges in edges_at.items():
        half_edges.sort(key=lambda half_edge: half_edge.angle)

        if len(half_edges) < 2:
            continue

        for first, second in zip(half_edges, half_edges[1:] + half_edges[:1]):
            difference = _angle_difference(first.angle, second.angle)

            if difference < ANGULAR_DEGENERACY_TOLERANCE:
                topology_issues.append(
                    TopologyIssue(
                        kind=TopologyIssueKind.DEGENERATE_ANGULAR_ORDERING,
                        critical_point=vertex_by_id[vertex_id],
                        separatrix=None,
                        detail=(
                            "two half-edges leave the same vertex within "
                            f"{difference:.2e} radians of each other"
                        ),
                    )
                )

    return edges_at, vertex_by_id, ordered_closed_half_edges, topology_issues


def _index_half_edges(
    edges_at: dict[int, list[_HalfEdge]],
) -> dict[tuple[int, bool], _HalfEdge]:
    """
    (id(separatrix), is_forward) -> _HalfEdge -- позволява намирането
    на "twin" (другата посока на СЪЩАТА сепаратриса) на всяко полу-
    ребро без изрично поле-указател (което би създало кръгова връзка
    при конструирането).
    """

    index: dict[tuple[int, bool], _HalfEdge] = {}

    for half_edges in edges_at.values():
        for half_edge in half_edges:
            index[(id(half_edge.separatrix), half_edge.is_forward)] = half_edge

    return index


def _next_half_edge(
    current: _HalfEdge,
    edges_at: dict[int, list[_HalfEdge]],
    half_edge_index: dict[tuple[int, bool], _HalfEdge],
) -> _HalfEdge:
    """
    Стандартното "next" правило за обхождане на границата на лице в
    планарен граф: намира "twin" на current (другата посока на
    същата сепаратриса, тръгваща от current.destination), после
    връща СЛЕДВАЩОТО (по ъгъл) полу-ребро в цикличния списък на
    twin.origin -- границата продължава от там. Може да върне блокер
    (destination=None), ако следващото по ъгъл полу-ребро там
    представя отворена сепаратриса -- extractor-ът на клетки
    (_trace_faces) решава какво значи това, не тази функция.
    """

    twin = half_edge_index[(id(current.separatrix), not current.is_forward)]

    ordered = edges_at[id(twin.origin)]
    twin_index = _find_index_by_identity(ordered, twin)

    return ordered[(twin_index + 1) % len(ordered)]


def _trace_faces(
    ordered_closed_half_edges: list[_HalfEdge],
    edges_at: dict[int, list[_HalfEdge]],
    half_edge_index: dict[tuple[int, bool], _HalfEdge],
) -> tuple[list[MorseSmaleCell], list[SeparatrixPath], list[TopologyIssue]]:
    """
    Обхожда всяко все още непосетено полу-ребро от
    ordered_closed_half_edges (в неговия фиксиран, детерминиран ред)
    като начало на ново лице (MorseSmaleCell), следвайки _next_half
    _edge, докато: (а) обходът се върне в началното полу-ребро --
    затворена клетка; (б) следващото полу-ребро е блокер (отворена
    сепаратриса) -- непълна клетка, open_boundaries=[тази сепаратриса];
    (в) бюджетът от стъпки (2 * общия брой полу-ребра + 1 -- повече от
    достатъчен за всяко легитимно лице) се изчерпи -- изоставен обход,
    отбелязан FACE_TRACE_BUDGET_EXCEEDED, вместо да увисне.

    Всяко полу-ребро принадлежи на точно едно обхождане (маркирано
    посетено веднага щом бъде включено в текущата граница) --
    структурно изключва дублирани клетки.
    """

    visited_ids: set[int] = set()
    cells: list[MorseSmaleCell] = []
    unassigned: list[SeparatrixPath] = []
    topology_issues: list[TopologyIssue] = []

    max_steps_per_face = 2 * len(ordered_closed_half_edges) + 1

    for start_half_edge in ordered_closed_half_edges:
        if id(start_half_edge) in visited_ids:
            continue

        boundary_separatrices: list[SeparatrixPath] = []
        boundary_directions: list[bool] = []
        boundary_vertices: list[ClassifiedCriticalPoint] = []
        open_boundaries: list[SeparatrixPath] = []

        current = start_half_edge
        is_closed_cell = False
        abandoned = True

        for _step in range(max_steps_per_face):
            visited_ids.add(id(current))
            boundary_separatrices.append(current.separatrix)
            boundary_directions.append(current.is_forward)
            boundary_vertices.append(current.destination)

            next_half_edge = _next_half_edge(current, edges_at, half_edge_index)

            if next_half_edge.destination is None:
                open_boundaries.append(next_half_edge.separatrix)
                abandoned = False
                break

            if next_half_edge is start_half_edge:
                is_closed_cell = True
                abandoned = False
                break

            current = next_half_edge

        if abandoned:
            unassigned.extend(boundary_separatrices)
            topology_issues.append(
                TopologyIssue(
                    kind=TopologyIssueKind.FACE_TRACE_BUDGET_EXCEEDED,
                    critical_point=None,
                    separatrix=start_half_edge.separatrix,
                    detail=(
                        "face trace starting at "
                        f"({start_half_edge.origin.x:.3f}, "
                        f"{start_half_edge.origin.y:.3f}) exceeded "
                        f"{max_steps_per_face} steps"
                    ),
                )
            )
            continue

        cells.append(
            MorseSmaleCell(
                cell_id=len(cells),
                boundary_separatrices=boundary_separatrices,
                boundary_traversal_directions=boundary_directions,
                boundary_critical_points=boundary_vertices,
                is_closed=is_closed_cell,
                open_boundaries=open_boundaries,
            )
        )

    return cells, unassigned, topology_issues


def assemble_morse_smale_cells(
    separatrices: list[SeparatrixPath],
) -> MorseSmaleComplex:
    """
    Сглобява 2-клетките на Morse-Smale комплекса ИЗКЛЮЧИТЕЛНО от вече
    трасираните separatrices (locate_morse_smale_separatrices,
    непроменена в тази фаза) -- не преизчислява градиент, Hessian, или
    каквато и да е математика на повърхността, само комбинаторика над
    вече изчислените криви и техните крайни точки.

    0-клетките (vertices) и 1-клетките (edges) вече съществуват --
    връщат се тук само за пълнота на резултата, непроменени.
    2-клетките (cells) -- виж MorseSmaleCell -- могат да бъдат
    затворени или отворени/непълни (виж is_closed/open_boundaries в
    MorseSmaleCell). Не филтрира по качество -- вижте
    compute_cell_geometry за геометрията на затворените клетки,
    изведена отделно при поискване, не изчислена автоматично тук.

    Не променя separatrices или каквото и да е негово поле.
    """

    edges_at, vertex_by_id, ordered_closed_half_edges, build_issues = (
        _build_half_edges(separatrices)
    )
    half_edge_index = _index_half_edges(edges_at)

    cells, unassigned, trace_issues = _trace_faces(
        ordered_closed_half_edges, edges_at, half_edge_index
    )

    return MorseSmaleComplex(
        vertices=list(vertex_by_id.values()),
        edges=list(separatrices),
        cells=cells,
        unassigned_separatrices=unassigned,
        topology_issues=build_issues + trace_issues,
    )


def _shoelace_signed_area(polygon: list[tuple[float, float]]) -> float:
    """
    Формулата на Гаус (shoelace) -- ЗНАКОВАТА площ (положителна за
    обхождане обратно на часовниковата стрелка, отрицателна за по
    часовниковата) -- знакът се пази изрично, защото центроидната
    формула по-долу се нуждае от него, не само от абсолютната стойност.
    """

    total = 0.0

    for (x0, y0), (x1, y1) in zip(polygon, polygon[1:]):
        total += x0 * y1 - x1 * y0

    return total / 2.0


def _polygon_centroid(
    polygon: list[tuple[float, float]],
    signed_area: float,
) -> tuple[float, float]:
    """
    Стандартната формула за центроид на прост (несамопресичащ се)
    полигон, претеглена по shoelace -- изисква ЗНАКОВАТА площ (не
    abs), за да остане правилна независимо от посоката на обхождане.
    За изроден (нулева площ) полигон -- средно аритметично на върховете
    вместо деление на нула.
    """

    if signed_area == 0.0:
        unique_points = polygon[:-1] or polygon
        x_sum = sum(point[0] for point in unique_points)
        y_sum = sum(point[1] for point in unique_points)
        return (x_sum / len(unique_points), y_sum / len(unique_points))

    centroid_x = 0.0
    centroid_y = 0.0

    for (x0, y0), (x1, y1) in zip(polygon, polygon[1:]):
        cross = x0 * y1 - x1 * y0
        centroid_x += (x0 + x1) * cross
        centroid_y += (y0 + y1) * cross

    return (
        centroid_x / (6.0 * signed_area),
        centroid_y / (6.0 * signed_area),
    )


def _polygon_perimeter(polygon: list[tuple[float, float]]) -> float:
    return sum(
        math.hypot(x1 - x0, y1 - y0)
        for (x0, y0), (x1, y1) in zip(polygon, polygon[1:])
    )


def compute_cell_geometry(
    cell: MorseSmaleCell,
) -> MorseSmaleCellGeometry | None:
    """
    Изведена геометрия на ЕДНА клетка -- изчислена при поискване,
    НИКОГА не се съхранява на самата MorseSmaleCell (чисто
    топологична). Връща None за отворени/непълни клетки (is_closed
    =False) -- полигон/площ/центроид/периметър не са добре дефинирани
    за незатворена граница, а измислянето им би нарушило принципа
    "никога не изобретявай данни".

    polygon: за всяко гранично ребро, в реда на обхождането, връх +
    собствените му SeparatrixPoint (обърнати, ако реброто е обходено
    назад -- boundary_traversal_directions[i]=False) -- следва
    ДЕЙСТВИТЕЛНО проследените криви, не прави линии между критичните
    точки. Затваря се естествено (polygon[-1] == polygon[0]), защото
    последният връх на обхождането по конструкция е същият обект като
    началния (обходът се връща в start_half_edge.origin) -- не се
    добавя изрично затваряща точка.

    perimeter: сборът от последователните разстояния по polygon -- НЕ
    просто сборът от path_length на boundary_separatrices (проверено
    директно: всяка сепаратриса спира REACHED_CRITICAL_POINT_DISTANCE
    преди истинската си цел, затова полигонът включва по един
    допълнителен "затварящ" сегмент на ребро, който path_length не
    покрива).
    """

    if not cell.is_closed:
        return None

    polygon: list[tuple[float, float]] = []

    for separatrix, is_forward, vertex in zip(
        cell.boundary_separatrices,
        cell.boundary_traversal_directions,
        cell.boundary_critical_points,
    ):
        origin = separatrix.start_saddle if is_forward else separatrix.end_critical_point
        points = separatrix.points if is_forward else list(reversed(separatrix.points))

        if not polygon:
            polygon.append((origin.x, origin.y))

        polygon.extend((point.x, point.y) for point in points)
        polygon.append((vertex.x, vertex.y))

    signed_area = _shoelace_signed_area(polygon)

    return MorseSmaleCellGeometry(
        polygon=polygon,
        area=abs(signed_area),
        centroid=_polygon_centroid(polygon, signed_area),
        perimeter=_polygon_perimeter(polygon),
    )


# ===========================================================
# Фаза 3 -- оценка на качеството на 2-клетките. НЕ променя топологията
# (MorseSmaleCell/MorseSmaleComplex от Фаза 2) или трасирането
# (SeparatrixPath от Фаза 1) по никакъв начин -- само анотира вече
# сглобените клетки, аналогично на critical_point_quality.py и
# assess_ridge_valley_quality спрямо своите обекти.
# ===========================================================

# Измерено при истинска (не синтетична адвокатска) референтна
# повърхност (egg-crate, виж tests/) -- минимална реална площ 1.125,
# минимален реален периметър 4.24. Числовият шумов праг на едно
# marching-стъпало е STEP_SIZE^2 = 0.0025 за площ. Същата логика като
# MIN_EIGENVALUE_MAGNITUDE в analysis/critical_point_quality.py: 3-4
# порядъка над шума, 1-2 под типична истинска клетка.
MIN_CELL_AREA = 0.05

# Граница, под която целият периметър на клетката е по-къс от 4
# marching стъпки -- неразличимо от единично изродено стъпало, не
# истински затворен цикъл. Обвързано с STEP_SIZE, не произволна нова
# константа.
MIN_CELL_PERIMETER = 4 * STEP_SIZE

# Отхвърля само напълно изродения 2-върхов "бигон" (седло, чиито два
# клона от един и същ тип стигат до една и съща цел) -- най-малката
# легитимна затворена клетка (2 седла + 1 максимум + 1 минимум) има 4.
MIN_DISTINCT_BOUNDARY_POINTS = 3

# Опционален критерий (изключен по подразбиране, boundary_margin=None)
# -- същата стойност и обосновка като BOUNDARY_MARGIN в
# analysis/critical_point_quality.py и CHAIN_BOUNDARY_MARGIN в
# analysis/ridge_valley.py.
CELL_BOUNDARY_MARGIN = 0.3


def _cell_topology_issue_reasons(
    cell: MorseSmaleCell,
    topology_issues: list[TopologyIssue],
) -> list[MorseSmaleCellRejectionReason]:
    """
    За всеки TopologyIssue, чийто .separatrix е (по идентичност) в
    cell.boundary_separatrices, или чийто .critical_point е (по
    идентичност) в cell.boundary_critical_points -- по една причина
    за отхвърляне на kind=AFFECTED_BY_TOPOLOGY_ISSUE. Не пренаписва и
    не копира самите TopologyIssue обекти в резултата -- само реферира
    кой TopologyIssueKind е засегнал клетката (related_topology_issue),
    пазейки двата списъка (topology_issues и rejection_reasons)
    структурно разделени.
    """

    reasons: list[MorseSmaleCellRejectionReason] = []

    for issue in topology_issues:
        affected = (
            issue.separatrix is not None
            and any(issue.separatrix is s for s in cell.boundary_separatrices)
        ) or (
            issue.critical_point is not None
            and any(issue.critical_point is p for p in cell.boundary_critical_points)
        )

        if not affected:
            continue

        reasons.append(
            MorseSmaleCellRejectionReason(
                kind=MorseSmaleCellRejectionReasonKind.AFFECTED_BY_TOPOLOGY_ISSUE,
                detail=(
                    f"cell touches a {issue.kind.value} topology issue: {issue.detail}"
                ),
                related_topology_issue=issue.kind,
            )
        )

    return reasons


def assess_morse_smale_cell_quality(
    cells: list[MorseSmaleCell],
    topology_issues: list[TopologyIssue],
    surface: AttackInfluenceSurface | None = None,
    min_area: float = MIN_CELL_AREA,
    min_perimeter: float = MIN_CELL_PERIMETER,
    min_distinct_boundary_points: int = MIN_DISTINCT_BOUNDARY_POINTS,
    boundary_margin: float | None = None,
    reject_degenerate_vertex_participation: bool = False,
) -> list[MorseSmaleCellQualityAssessment]:
    """
    Пост-сглобяваща стъпка за интерпретируемост -- НЕ променя
    сглобяването (assemble_morse_smale_cells) по никакъв начин.
    Единствената ѝ задача е да раздели "коректно затворена клетка на
    комплекса" от "клетка, достатъчно надеждна, за да се представи
    като значима" -- аналогично на
    analysis.critical_point_quality.assess_critical_point_quality и
    analysis.ridge_valley.assess_ridge_valley_quality, но за 2-клетки.

    Всяка входна клетка се запазва в резултата непроменена (обвита в
    MorseSmaleCellQualityAssessment) -- функцията никога тихо не
    изтрива информация, само я анотира. Всяка отхвърлена клетка носи
    ВСИЧКИ провалени критерии, не само първия.

    surface е нужна САМО ако boundary_margin е зададен (за границите
    на домейна) -- по подразбиране None и не се изисква, освен ако
    NEAR_BOUNDARY критерият изрично не е поискан.

    За ОТВОРЕНИ клетки (is_closed=False): compute_cell_geometry връща
    None (площ/периметър не са дефинирани без валиден полигон), затова
    AREA_TOO_SMALL/PERIMETER_TOO_SMALL просто НЕ се проверяват за тях
    -- това НЕ означава, че отворената клетка ги "минава", а че тези
    два критерия са неприложими без геометрия. NOT_CLOSED вече е
    достатъчна причина за отхвърляне; останалите, негеометрични
    критерии (topology issues, брой различни върхове, по избор:
    близост до ръба, изродени върхове) продължават да се проверяват
    и могат да добавят допълнителни причини към същата отхвърлена
    клетка.

    Приложени критерии (всеки именуван и документиран по-горе или
    чрез своя параметър):

    1. not cell.is_closed -> NOT_CLOSED.
    2. Гранична сепаратриса с termination_status различен от
       "reached_critical_point" -> UNRELIABLE_BOUNDARY_TERMINATION
       (защитна проверка -- по конструкция на Фаза 2 не се очаква да
       задейства за затворена клетка).
    3. Клетката съвпада по идентичност с .separatrix/.critical_point
       на запис от topology_issues -> AFFECTED_BY_TOPOLOGY_ISSUE
       (по един запис на засегнат TopologyIssue).
    4. area < min_area -> AREA_TOO_SMALL (само за затворени клетки).
    5. perimeter < min_perimeter -> PERIMETER_TOO_SMALL (само за
       затворени клетки).
    6. По-малко от min_distinct_boundary_points различни (по
       идентичност) гранични критични точки -> TOO_FEW_DISTINCT
       _BOUNDARY_POINTS.
    7. (по избор) boundary_margin зададен и повече от половината
       гранични точки са на разстояние < boundary_margin от ръба на
       домейна -> NEAR_BOUNDARY.
    8. (по избор) reject_degenerate_vertex_participation=True и някоя
       гранична точка има classification="degenerate" ->
       DEGENERATE_VERTEX_PARTICIPATION.
    """

    if boundary_margin is not None and surface is None:
        raise ValueError("surface is required when boundary_margin is provided")

    if boundary_margin is not None:
        x_min, x_max = float(surface.x[0]), float(surface.x[-1])
        y_min, y_max = float(surface.y[0]), float(surface.y[-1])

    assessments: list[MorseSmaleCellQualityAssessment] = []

    for cell in cells:
        reasons: list[MorseSmaleCellRejectionReason] = []

        if not cell.is_closed:
            reasons.append(
                MorseSmaleCellRejectionReason(
                    kind=MorseSmaleCellRejectionReasonKind.NOT_CLOSED,
                    detail=(
                        "cell boundary walk did not close "
                        f"({len(cell.open_boundaries)} open boundary/boundaries)"
                    ),
                    related_topology_issue=None,
                )
            )

        unreliable = [
            separatrix
            for separatrix in cell.boundary_separatrices
            if separatrix.termination_status != "reached_critical_point"
        ]
        if unreliable:
            reasons.append(
                MorseSmaleCellRejectionReason(
                    kind=(
                        MorseSmaleCellRejectionReasonKind
                        .UNRELIABLE_BOUNDARY_TERMINATION
                    ),
                    detail=(
                        f"{len(unreliable)} boundary separatrix/separatrices did not "
                        "terminate with reached_critical_point"
                    ),
                    related_topology_issue=None,
                )
            )

        reasons.extend(_cell_topology_issue_reasons(cell, topology_issues))

        geometry = compute_cell_geometry(cell)

        if geometry is not None:
            if geometry.area < min_area:
                reasons.append(
                    MorseSmaleCellRejectionReason(
                        kind=MorseSmaleCellRejectionReasonKind.AREA_TOO_SMALL,
                        detail=(
                            f"area={geometry.area:.6f} < min_area={min_area:.6f}"
                        ),
                        related_topology_issue=None,
                    )
                )

            if geometry.perimeter < min_perimeter:
                reasons.append(
                    MorseSmaleCellRejectionReason(
                        kind=MorseSmaleCellRejectionReasonKind.PERIMETER_TOO_SMALL,
                        detail=(
                            f"perimeter={geometry.perimeter:.6f} "
                            f"< min_perimeter={min_perimeter:.6f}"
                        ),
                        related_topology_issue=None,
                    )
                )

        distinct_boundary_point_ids = {
            id(point) for point in cell.boundary_critical_points
        }
        if len(distinct_boundary_point_ids) < min_distinct_boundary_points:
            reasons.append(
                MorseSmaleCellRejectionReason(
                    kind=(
                        MorseSmaleCellRejectionReasonKind
                        .TOO_FEW_DISTINCT_BOUNDARY_POINTS
                    ),
                    detail=(
                        f"{len(distinct_boundary_point_ids)} distinct boundary "
                        f"points < min_distinct_boundary_points="
                        f"{min_distinct_boundary_points}"
                    ),
                    related_topology_issue=None,
                )
            )

        if boundary_margin is not None and cell.boundary_critical_points:
            distances = [
                min(
                    point.x - x_min,
                    x_max - point.x,
                    point.y - y_min,
                    y_max - point.y,
                )
                for point in cell.boundary_critical_points
            ]
            near_boundary_fraction = sum(
                1 for distance in distances if distance < boundary_margin
            ) / len(distances)

            if near_boundary_fraction > 0.5:
                reasons.append(
                    MorseSmaleCellRejectionReason(
                        kind=MorseSmaleCellRejectionReasonKind.NEAR_BOUNDARY,
                        detail=(
                            f"fraction={near_boundary_fraction:.2f} > 0.5 near the "
                            f"domain edge (boundary_margin={boundary_margin:.4f})"
                        ),
                        related_topology_issue=None,
                    )
                )

        if reject_degenerate_vertex_participation:
            degenerate_points = [
                point
                for point in cell.boundary_critical_points
                if point.classification == "degenerate"
            ]
            if degenerate_points:
                reasons.append(
                    MorseSmaleCellRejectionReason(
                        kind=(
                            MorseSmaleCellRejectionReasonKind
                            .DEGENERATE_VERTEX_PARTICIPATION
                        ),
                        detail=(
                            f"{len(degenerate_points)} boundary point(s) classified "
                            "as degenerate"
                        ),
                        related_topology_issue=None,
                    )
                )

        assessments.append(
            MorseSmaleCellQualityAssessment(
                cell=cell,
                is_accepted=(len(reasons) == 0),
                rejection_reasons=reasons,
            )
        )

    return assessments
