import numpy as np

from chess_engine.models import (
    AttackInfluenceSurface,
    ClassifiedCriticalPoint,
    CriticalPointQualityAssessment,
)


# Точка, чието разстояние до най-близкия ръб на дъската
# ([x_min,x_max] x [y_min,y_max], т.е. 0.5..7.5) е под този праг, се
# отхвърля. Обосновка: и Фаза 2, и реалният рендър на позиция във
# Фаза 4 показаха, че напаснатият сплайн може да има истински (не
# NaN, не грешка) критични точки точно до ръба, произтичащи от
# резонанс между bicubic напасването на само 8 възела и затихващата
# опашка на полето там -- не от шахматна структура. 0.3 е ~1/3 от
# ширината на клетка (клетка = 1.0 в тези координати) -- достатъчно,
# за да отсее точки буквално долепени до ръба, без да отхвърля
# истински точки близо до крайна редица/файл.
BOUNDARY_MARGIN = 0.3

# Най-слабото (по модул) собствено число на Hessian-а в точката
# трябва да е поне толкова голямо. Обосновка за мащаба: тежестите в
# PIECE_WEIGHTS (analysis/attack_influence.py) варират 1..9, а
# кривината на истинска, добре оформена структура (напр. квадратче,
# защитавано от няколко фигури, срещу празно поле наблизо) е от
# порядъка на 1..10 на квадратна координатна единица. Числов шум от
# недосемплирания 8x8 сплайн край границата е измерен в Фаза 2/4 на
# нива 1e-6..1e-8 -- 0.05 е с 3-4 порядъка над шума, но с 1-2 порядъка
# под типична истинска структура, т.е. запазва реални точки, докато
# маха численен шум. Не е напасната по конкретна партия.
MIN_EIGENVALUE_MAGNITUDE = 0.05

# Комбинираната "сила" на Hessian-а -- sqrt(eigenvalue_min^2 +
# eigenvalue_max^2), т.е. Frobenius-подобна норма на диагонализирания
# Hessian. За разлика от MIN_EIGENVALUE_MAGNITUDE (която гледа само
# НАЙ-СЛАБАТА посока), тази гледа общата "изразеност" на точката --
# полезна за отхвърляне на точки, при които и двете посоки са
# поотделно над MIN_EIGENVALUE_MAGNITUDE, но все пак цялостно плитки.
# Прагът е нарочно малко над MIN_EIGENVALUE_MAGNITUDE, за да е
# самостоятелен, а не просто изведен от него.
MIN_CURVATURE_STRENGTH = 0.1


def assess_critical_point_quality(
    classified_points: list[ClassifiedCriticalPoint],
    surface: AttackInfluenceSurface,
    boundary_margin: float = BOUNDARY_MARGIN,
    min_eigenvalue_magnitude: float = MIN_EIGENVALUE_MAGNITUDE,
    min_curvature_strength: float = MIN_CURVATURE_STRENGTH,
    gradient_quality_threshold: float | None = None,
    suppress_near_flat_degenerate: bool = True,
) -> list[CriticalPointQualityAssessment]:
    """
    Пост-класификационна стъпка за интерпретируемост -- НЕ променя
    локализацията (analysis.critical_points.locate_critical_points)
    или класификацията (classify_critical_points) по никакъв начин.
    Единствената ѝ задача е да раздели "математически открита точка
    на напаснатия сплайн" от "точка, достатъчно надеждна, за да се
    представи като шахматно значима" -- вижте docs/mathematics.md,
    Раздел 9 за пълния разбор защо не всяка критична точка на
    сплайна е шахматно значима.

    Всяка входна точка се запазва в резултата непроменена (обвита в
    CriticalPointQualityAssessment) -- функцията никога тихо не
    изтрива информация, само я анотира. Всяка отхвърлена точка носи
    ВСИЧКИ провалени критерии, не само първия.

    Приложени критерии (всеки именуван и документиран по-горе или
    чрез своя параметър):

    1. status != "converged" -> винаги отхвърлена (никога не е била
       истинска критична точка; eigenvalue_min/max са None, така че
       никой друг критерий не е приложим).
    2. Разстояние до ръба на дъската < boundary_margin.
    3. suppress_near_flat_degenerate=True (по подразбиране) и
       classification == "degenerate".
    4. |най-слабото собствено число| < min_eigenvalue_magnitude.
    5. Комбинирана сила на кривината < min_curvature_strength.
    6. (по избор) gradient_norm > gradient_quality_threshold, ако е
       зададен -- по подразбиране None (изключен), за по-строг праг
       от Фаза 2 сходимостта, ако бъде поискан изрично.
    """

    x_min, x_max = float(surface.x[0]), float(surface.x[-1])
    y_min, y_max = float(surface.y[0]), float(surface.y[-1])

    assessments: list[CriticalPointQualityAssessment] = []

    for point in classified_points:
        reasons: list[str] = []

        if point.status != "converged":
            reasons.append(
                f"not a converged critical point (status={point.status!r})"
            )
            assessments.append(
                CriticalPointQualityAssessment(
                    point=point,
                    is_accepted=False,
                    rejection_reasons=reasons,
                )
            )
            continue

        distance_to_boundary = min(
            point.x - x_min,
            x_max - point.x,
            point.y - y_min,
            y_max - point.y,
        )

        if distance_to_boundary < boundary_margin:
            reasons.append(
                "too close to board boundary "
                f"(distance={distance_to_boundary:.4f} "
                f"< boundary_margin={boundary_margin:.4f})"
            )

        if suppress_near_flat_degenerate and point.classification == "degenerate":
            reasons.append(
                "degenerate classification: second-derivative test "
                "was inconclusive at this point"
            )

        # status == "converged" guarantees eigenvalue_min/max are not None.
        weakest_eigenvalue_magnitude = min(
            abs(point.eigenvalue_min), abs(point.eigenvalue_max)
        )

        if weakest_eigenvalue_magnitude < min_eigenvalue_magnitude:
            reasons.append(
                "weak curvature in at least one principal direction "
                f"(|eigenvalue|={weakest_eigenvalue_magnitude:.6f} "
                f"< min_eigenvalue_magnitude={min_eigenvalue_magnitude:.6f})"
            )

        curvature_strength = float(
            np.hypot(point.eigenvalue_min, point.eigenvalue_max)
        )

        if curvature_strength < min_curvature_strength:
            reasons.append(
                "overall curvature too weak "
                f"(strength={curvature_strength:.6f} "
                f"< min_curvature_strength={min_curvature_strength:.6f})"
            )

        if (
            gradient_quality_threshold is not None
            and point.gradient_norm > gradient_quality_threshold
        ):
            reasons.append(
                "gradient norm above the requested quality threshold "
                f"(|∇φ|={point.gradient_norm:.2e} "
                f"> gradient_quality_threshold={gradient_quality_threshold:.2e})"
            )

        assessments.append(
            CriticalPointQualityAssessment(
                point=point,
                is_accepted=(len(reasons) == 0),
                rejection_reasons=reasons,
            )
        )

    return assessments
