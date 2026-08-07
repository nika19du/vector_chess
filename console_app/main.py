from analysis.dynamics import analyze_dynamics
from analysis.attacker_count import print_heatmap
from chess_engine.analyzer import analyze_position
from chess_engine.board import ChessGame
from chess_engine.models import (
    ClassifiedCriticalPoint,
    CriticalPointCandidate,
    CriticalPointQualityAssessment,
    DynamicsAnalysis,
    MorseSmaleCellQualityAssessment,
    MorseSmaleCellRejectionReasonKind,
    MorseSmaleComplex,
    MoveAnalysis,
    RidgeValleyChain,
    RidgeValleyQualityAssessment,
)
from console_app.export import (
    export_game_analysis,
    export_game_audio,
    export_move_audio,
)
from visualization.heatmap_plot import plot_position_analysis
from visualization.position_plot import plot_position
from analysis.attack_influence import (
    print_attack_influence_matrix,
)
from analysis.attack_influence_surface import build_attack_influence_surface
from analysis.critical_point_quality import assess_critical_point_quality
from analysis.critical_points import classify_critical_points, locate_critical_points
from analysis.ridge_valley import assess_ridge_valley_quality, locate_ridge_valley_chains
from analysis.morse_smale import (
    assemble_morse_smale_cells,
    assess_morse_smale_cell_quality,
    locate_morse_smale_separatrices,
)
from visualization.attack_influence_plot import plot_attack_influence_field
from analysis.gradient_field import (
    print_gradient_field,
)
from visualization.critical_points_plot import plot_critical_points
from visualization.equipotential_plot import (
    plot_equipotential_field,
)
from visualization.gradient_plot import plot_gradient_field
from visualization.morse_smale_plot import plot_morse_smale_cells
from visualization.ridge_valley_plot import plot_ridge_valley
from visualization.source_potential_plot import (
    plot_field_comparison,
)


def print_position_analysis(
    analysis: MoveAnalysis,
) -> None:
    print("\n--- Анализ на хода ---")
    print(f"Ход: {analysis.move}")
    print(f"Фигура: {analysis.color} {analysis.piece_name}")
    print(f"От: {analysis.from_square}")
    print(f"До: {analysis.to_square}")
    print(f"Вземане: {'да' if analysis.is_capture else 'не'}")
    print(f"Шах: {'да' if analysis.is_check else 'не'}")

    print("\n--- Мобилност ---")
    print(
        f"Белите достигат: "
        f"{analysis.mobility.white_reachable_squares} полета"
    )
    print(
        f"Черните достигат: "
        f"{analysis.mobility.black_reachable_squares} полета"
    )
    print(f"Разлика: {analysis.mobility.difference:+d}")

    print_heatmap(analysis.attacker_count_field.matrix)

    print("\n--- Центрове на контрол ---")

    if analysis.white_center is not None:
        print(
            f"Бели: "
            f"x={analysis.white_center[0]:.2f}, "
            f"y={analysis.white_center[1]:.2f}"
        )

    if analysis.black_center is not None:
        print(
            f"Черни: "
            f"x={analysis.black_center[0]:.2f}, "
            f"y={analysis.black_center[1]:.2f}"
        )

    if analysis.control_vector is not None:
        vector = analysis.control_vector

        print("\n--- Глобален вектор ---")
        print(
            f"Начало: "
            f"({vector.start_x:.2f}, {vector.start_y:.2f})"
        )
        print(
            f"Край: "
            f"({vector.end_x:.2f}, {vector.end_y:.2f})"
        )
        print(
            f"Посока: "
            f"({vector.delta_x:.2f}, {vector.delta_y:.2f})"
        )
        print(f"Дължина: {vector.magnitude:.2f}")

    print(
        f"\nБрой локални вектори на атака: "
        f"{len(analysis.attack_vectors)}"
    )

    print("Първите 10 вектора:")

    for vector in analysis.attack_vectors[:10]:
        print(
            f"  {vector.color} {vector.piece_name}: "
            f"{vector.from_square} -> {vector.to_square}, "
            f"delta=({vector.delta_x}, {vector.delta_y})"
        )

    attack_influence = analysis.attack_influence_field

    print("\n--- Атаково влияние ---")

    print(
        f"Общо бяло атаково влияние: "
        f"{attack_influence.total_white_attack_influence:.2f}"
    )

    print(
        f"Общо черно атаково влияние: "
        f"{attack_influence.total_black_attack_influence:.2f}"
    )

    print(
        f"Баланс на атаковото влияние: "
        f"{attack_influence.balance:+.2f}"
    )

    print(
        f"Най-силно бяло поле: "
        f"{attack_influence.strongest_white_square}"
    )

    print(
        f"Най-силно черно поле: "
        f"{attack_influence.strongest_black_square}"
    )

    print_attack_influence_matrix(attack_influence.matrix)

    print("\n--- Най-силни gradient vectors ---")

    strongest_vectors = sorted(
        analysis.gradient_field.vectors,
        key=lambda vector: vector.magnitude,
        reverse=True,
    )[:10]

    for vector in strongest_vectors:
        print(
            f"{vector.square}: "
            f"({vector.delta_x:+.2f}, "
            f"{vector.delta_y:+.2f}), "
            f"magnitude={vector.magnitude:.2f}"
        )

def print_dynamics(
    dynamics: DynamicsAnalysis,
) -> None:
    print("\n--- Динамика спрямо предишния ход ---")

    print(
        f"Сила на предишната позиция: "
        f"{dynamics.previous_force:+d}"
    )
    print(
        f"Сила на текущата позиция: "
        f"{dynamics.current_force:+d}"
    )
    print(f"ΔF: {dynamics.delta_force:+d}")

    print(
        f"Промяна в бялата мобилност: "
        f"{dynamics.white_mobility_delta:+d}"
    )
    print(
        f"Промяна в черната мобилност: "
        f"{dynamics.black_mobility_delta:+d}"
    )
    print(
        f"Промяна в броя вектори: "
        f"{dynamics.attack_vectors_delta:+d}"
    )
    print(
        f"Обща промяна в heatmap-а: "
        f"{dynamics.heatmap_change}"
    )

    print(f"Интензивност: {dynamics.intensity:.2f}")
    print(f"Състояние: {dynamics.label}")


def print_history(
    analysis_history: list[MoveAnalysis],
    dynamics_history: list[DynamicsAnalysis],
) -> None:
    print("\n--- История на партията ---")

    if not analysis_history:
        print("Все още няма изиграни ходове.")
        return

    for index, analysis in enumerate(
        analysis_history,
        start=1,
    ):
        print(
            f"Ход {index}: "
            f"{analysis.move} "
            f"({analysis.color} {analysis.piece_name})"
        )

        if index == 1:
            print("  Няма предишна позиция за сравнение.")
            continue

        dynamics = dynamics_history[index - 2]

        print(
            f"  ΔF={dynamics.delta_force:+d}, "
            f"intensity={dynamics.intensity:.2f}, "
            f"state={dynamics.label}"
        )


# Групиране на причини за отхвърляне по КАТЕГОРИЯ (не по точния низ
# с конкретните числа във всяка причина -- иначе почти всяка причина
# би била уникална и "групирано" резюме не би било компактно).
# Съответства едно-към-едно на префиксите, генерирани от
# analysis/critical_point_quality.py.
REJECTION_REASON_CATEGORIES: list[tuple[str, str]] = [
    ("not a converged critical point", "Newton не е сходил"),
    ("too close to board boundary", "твърде близо до ръба на дъската"),
    ("degenerate classification", "изродена (degenerate) класификация"),
    (
        "weak curvature in at least one principal direction",
        "слаба кривина в поне една посока",
    ),
    ("overall curvature too weak", "цялостно твърде слаба кривина"),
    (
        "gradient norm above the requested quality threshold",
        "градиентът е над зададения праг за качество",
    ),
]


def categorize_rejection_reason(
    reason: str,
) -> str:
    for prefix, category_label in REJECTION_REASON_CATEGORIES:
        if reason.startswith(prefix):
            return category_label

    return reason


def print_critical_points_summary(
    candidates: list[CriticalPointCandidate],
    classified_points: list[ClassifiedCriticalPoint],
    assessments: list[CriticalPointQualityAssessment],
) -> None:
    """
    Компактно резюме на изхода на критичните точки -- НЕ преизчислява
    нищо, само отчита вече готовите резултати от
    analysis/critical_points.py и analysis/critical_point_quality.py.
    """

    classified_count = sum(
        1
        for point in classified_points
        if point.classification != "unclassified"
    )

    accepted_count = sum(
        1 for assessment in assessments if assessment.is_accepted
    )

    rejected_count = len(assessments) - accepted_count

    print("\n--- Критични точки ---")
    print(f"Открити кандидати (Newton): {len(candidates)}")
    print(f"Класифицирани (сходили): {classified_count}")
    print(f"Приети (надеждни): {accepted_count}")
    print(f"Отхвърлени: {rejected_count}")

    if rejected_count == 0:
        return

    reason_counts: dict[str, int] = {}

    for assessment in assessments:
        if assessment.is_accepted:
            continue

        category_label = categorize_rejection_reason(
            assessment.rejection_reasons[0]
        )

        reason_counts[category_label] = (
            reason_counts.get(category_label, 0) + 1
        )

    print("Причини за отхвърляне:")

    for category_label, count in reason_counts.items():
        print(f"  {category_label}: {count}")


# Групиране на причини за отхвърляне на ridge/valley вериги по
# КАТЕГОРИЯ -- същия принцип като REJECTION_REASON_CATEGORIES по-горе,
# но за префиксите, генерирани от
# analysis/ridge_valley.py::assess_ridge_valley_quality.
RIDGE_VALLEY_REJECTION_REASON_CATEGORIES: list[tuple[str, str]] = [
    ("chain too short", "веригата е твърде къса"),
    (
        "chain spends most of its length near the board boundary",
        "веригата е предимно близо до ръба на дъската",
    ),
    (
        "average cross-direction curvature too weak",
        "цялостно твърде слаба кросова кривина",
    ),
]


def categorize_ridge_valley_rejection_reason(
    reason: str,
) -> str:
    for prefix, category_label in RIDGE_VALLEY_REJECTION_REASON_CATEGORIES:
        if reason.startswith(prefix):
            return category_label

    return reason


def print_ridge_valley_summary(
    ridge_chains: list[RidgeValleyChain],
    valley_chains: list[RidgeValleyChain],
    ridge_assessments: list[RidgeValleyQualityAssessment],
    valley_assessments: list[RidgeValleyQualityAssessment],
) -> None:
    """
    Компактно резюме на изхода от ridge/valley трасирането -- НЕ
    преизчислява нищо, само отчита вече готовите резултати от
    analysis/ridge_valley.py. Огледален формат на
    print_critical_points_summary по-горе.
    """

    accepted_ridge_count = sum(
        1 for assessment in ridge_assessments if assessment.is_accepted
    )
    accepted_valley_count = sum(
        1 for assessment in valley_assessments if assessment.is_accepted
    )

    all_assessments = ridge_assessments + valley_assessments
    rejected_count = (
        len(all_assessments) - accepted_ridge_count - accepted_valley_count
    )

    print("\n--- Ridge / Valley ---")
    print(f"Открити ridge вериги: {len(ridge_chains)}")
    print(f"Открити valley вериги: {len(valley_chains)}")
    print(f"Приети ridge вериги: {accepted_ridge_count}")
    print(f"Приети valley вериги: {accepted_valley_count}")
    print(f"Отхвърлени вериги: {rejected_count}")

    if rejected_count == 0:
        return

    reason_counts: dict[str, int] = {}

    for assessment in all_assessments:
        if assessment.is_accepted:
            continue

        category_label = categorize_ridge_valley_rejection_reason(
            assessment.rejection_reasons[0]
        )

        reason_counts[category_label] = (
            reason_counts.get(category_label, 0) + 1
        )

    print("Причини за отхвърляне:")

    for category_label, count in reason_counts.items():
        print(f"  {category_label}: {count}")


# Четими Български етикети за MorseSmaleCellRejectionReasonKind
# (analysis/morse_smale.py) -- директен речник по enum стойност, не
# съвпадение по представка на низ, за разлика от
# REJECTION_REASON_CATEGORIES/RIDGE_VALLEY_REJECTION_REASON_CATEGORIES
# по-горе -- структурираният enum от Фаза 3 прави сравняването по низ
# ненужно тук.
MORSE_SMALE_REJECTION_REASON_LABELS: dict[MorseSmaleCellRejectionReasonKind, str] = {
    MorseSmaleCellRejectionReasonKind.NOT_CLOSED: "клетката не е затворена",
    MorseSmaleCellRejectionReasonKind.UNRELIABLE_BOUNDARY_TERMINATION: (
        "ненадеждно завършена гранична сепаратриса"
    ),
    MorseSmaleCellRejectionReasonKind.AFFECTED_BY_TOPOLOGY_ISSUE: (
        "засегната от топологичен проблем"
    ),
    MorseSmaleCellRejectionReasonKind.AREA_TOO_SMALL: "твърде малка площ",
    MorseSmaleCellRejectionReasonKind.PERIMETER_TOO_SMALL: "твърде малък периметър",
    MorseSmaleCellRejectionReasonKind.TOO_FEW_DISTINCT_BOUNDARY_POINTS: (
        "твърде малко различни гранични точки"
    ),
    MorseSmaleCellRejectionReasonKind.NEAR_BOUNDARY: (
        "твърде близо до ръба на дъската"
    ),
    MorseSmaleCellRejectionReasonKind.DEGENERATE_VERTEX_PARTICIPATION: (
        "изродена гранична критична точка"
    ),
}


def print_morse_smale_summary(
    critical_point_assessments: list[CriticalPointQualityAssessment],
    morse_smale_complex: MorseSmaleComplex,
    cell_assessments: list[MorseSmaleCellQualityAssessment],
) -> None:
    """
    Компактно резюме на изхода от Morse-Smale pipeline-а -- НЕ
    преизчислява нищо, само отчита вече готовите резултати от
    analysis/morse_smale.py.

    За разлика от print_critical_points_summary/print_ridge_valley
    _summary (които групират само по ПЪРВАТА причина за отхвърляне на
    даден елемент), тук се броят ВСИЧКИ причини на всяка отхвърлена
    клетка -- съответства на изричното решение от Фаза 3
    (MorseSmaleCellQualityAssessment.rejection_reasons никога не се
    съкращава до първата причина), затова резюмето не трябва тихо да
    изхвърля тази информация в момента на отчитане.
    """

    accepted_points = [
        assessment.point
        for assessment in critical_point_assessments
        if assessment.is_accepted
    ]
    accepted_saddle_count = sum(
        1 for point in accepted_points if point.classification == "saddle"
    )

    closed_cell_count = sum(
        1 for cell in morse_smale_complex.cells if cell.is_closed
    )
    open_cell_count = len(morse_smale_complex.cells) - closed_cell_count

    accepted_cell_count = sum(
        1 for assessment in cell_assessments if assessment.is_accepted
    )
    rejected_cell_count = len(cell_assessments) - accepted_cell_count

    print("\n--- Morse-Smale Complex ---")
    print(f"Приети критични точки: {len(accepted_points)}")
    print(f"Използвани седла: {accepted_saddle_count}")
    print(f"Сепаратриси: {len(morse_smale_complex.edges)}")
    print(f"Затворени клетки: {closed_cell_count}")
    print(f"Отворени/непълни клетки: {open_cell_count}")
    print(f"Приети клетки: {accepted_cell_count}")
    print(f"Отхвърлени клетки: {rejected_cell_count}")
    print(f"Топологични проблеми: {len(morse_smale_complex.topology_issues)}")

    if rejected_cell_count == 0:
        return

    reason_counts: dict[str, int] = {}

    for assessment in cell_assessments:
        if assessment.is_accepted:
            continue

        for reason in assessment.rejection_reasons:
            category_label = MORSE_SMALE_REJECTION_REASON_LABELS[reason.kind]
            reason_counts[category_label] = (
                reason_counts.get(category_label, 0) + 1
            )

    print("Причини за отхвърляне:")

    for category_label, count in reason_counts.items():
        print(f"  {category_label}: {count}")


def main() -> None:
    game = ChessGame()

    previous_analysis: MoveAnalysis | None = None

    analysis_history: list[MoveAnalysis] = []
    dynamics_history: list[DynamicsAnalysis] = []

    print("VectorChess — Console Analysis")
    print("Въвеждай ходове във формат UCI, например:")
    print("e2e4, e7e5, g1f3")
    print("board - показва шахматната дъска")
    print("За история напиши: history")
    print("За запис в JSON напиши: export")
    print("За графична визуализация напиши: plot")
    print("За attack influence field визуализация напиши: attack_influence_plot")
    print("За gradient field визуализация напиши: gradient_plot")
    print(
        "За equipotential визуализация "
        "напиши: equipotential_plot"
    )
    print(
        "За сравнение Attack vs Source Potential "
        "напиши: source_plot"
    )
    print(
        "За критични точки (maximum/minimum/saddle) "
        "напиши: critical_points_plot"
    )
    print(
        "За ridge/valley вериги "
        "напиши: ridge_valley_plot"
    )
    print(
        "За Morse-Smale комплекс "
        "напиши: morse_smale_plot"
    )
    print(
        "За аудио клип на последния ход "
        "напиши: audio"
    )
    print(
        "За аудио клипове на цялата партия дотук "
        "напиши: export_audio"
    )
    print(
        "При край на партията (шах и мат, пат и т.н.) аудио "
        "клиповете за всички изиграни ходове се записват "
        "автоматично в output/audio."
    )
    print("За изход напиши: quit")

    while not game.is_game_over():
        print("\nТекуща позиция:")
        game.print_board()

        print(f"\nРед на: {game.current_turn()}")

        move_text = input("\nХод: ").strip().lower()

        if move_text in {"quit", "exit"}:
            print("Играта беше прекратена.")
            return

        if move_text == "plot":
            if previous_analysis is None:
                print(
                    "Все още няма позиция "
                    "за визуализиране."
                )
                continue

            latest_dynamics = (
                dynamics_history[-1]
                if dynamics_history
                else None
            )

            plot_position(
                board=game.board,
                analysis=previous_analysis,
                dynamics=latest_dynamics,
            )

            continue

        if move_text == "attack_influence_plot":
            if previous_analysis is None:
                print(
                    "Все още няма позиция "
                    "за визуализиране."
                )
                continue

            plot_attack_influence_field(
                board=game.board,
                analysis=previous_analysis,
            )

            continue

        if move_text == "gradient_plot":
            if previous_analysis is None:
                print(
                    "Все още няма позиция "
                    "за визуализиране."
                )
                continue

            plot_gradient_field(
                board=game.board,
                analysis=previous_analysis,
            )

            continue

        if move_text == "equipotential_plot":
            if previous_analysis is None:
                print(
                    "Все още няма позиция "
                    "за визуализиране."
                )
                continue

            plot_equipotential_field(
                board=game.board,
                analysis=previous_analysis,
            )

            continue

        if move_text == "source_plot":
            if previous_analysis is None:
                print(
                    "Все още няма позиция "
                    "за визуализиране."
                )
                continue

            plot_field_comparison(
                board=game.board,
                analysis=previous_analysis,
            )

            continue

        if move_text == "critical_points_plot":
            if previous_analysis is None:
                print(
                    "Все още няма позиция "
                    "за визуализиране."
                )
                continue

            surface = build_attack_influence_surface(
                previous_analysis.attack_influence_field
            )

            candidates = locate_critical_points(surface)
            classified_points = classify_critical_points(
                candidates, surface
            )
            assessments = assess_critical_point_quality(
                classified_points, surface
            )

            print_critical_points_summary(
                candidates=candidates,
                classified_points=classified_points,
                assessments=assessments,
            )

            plot_critical_points(
                board=game.board,
                analysis=previous_analysis,
                surface=surface,
                classified_points=classified_points,
                assessments=assessments,
            )

            continue

        if move_text == "ridge_valley_plot":
            if previous_analysis is None:
                print(
                    "Все още няма позиция "
                    "за визуализиране."
                )
                continue

            surface = build_attack_influence_surface(
                previous_analysis.attack_influence_field
            )

            candidates = locate_critical_points(surface)
            classified_points = classify_critical_points(
                candidates, surface
            )
            critical_point_assessments = assess_critical_point_quality(
                classified_points, surface
            )
            accepted_critical_points = [
                assessment.point
                for assessment in critical_point_assessments
                if assessment.is_accepted
            ]

            ridge_chains = locate_ridge_valley_chains(
                surface, accepted_critical_points, kind="ridge"
            )
            valley_chains = locate_ridge_valley_chains(
                surface, accepted_critical_points, kind="valley"
            )

            ridge_assessments = assess_ridge_valley_quality(
                ridge_chains, surface
            )
            valley_assessments = assess_ridge_valley_quality(
                valley_chains, surface
            )

            print_ridge_valley_summary(
                ridge_chains=ridge_chains,
                valley_chains=valley_chains,
                ridge_assessments=ridge_assessments,
                valley_assessments=valley_assessments,
            )

            plot_ridge_valley(
                board=game.board,
                analysis=previous_analysis,
                surface=surface,
                classified_points=classified_points,
                critical_point_assessments=critical_point_assessments,
                ridge_assessments=ridge_assessments,
                valley_assessments=valley_assessments,
            )

            continue

        if move_text == "morse_smale_plot":
            if previous_analysis is None:
                print(
                    "Все още няма позиция "
                    "за визуализиране."
                )
                continue

            surface = build_attack_influence_surface(
                previous_analysis.attack_influence_field
            )

            candidates = locate_critical_points(surface)
            classified_points = classify_critical_points(
                candidates, surface
            )
            critical_point_assessments = assess_critical_point_quality(
                classified_points, surface
            )

            separatrices = locate_morse_smale_separatrices(
                surface, critical_point_assessments
            )
            morse_smale_complex = assemble_morse_smale_cells(separatrices)
            cell_assessments = assess_morse_smale_cell_quality(
                morse_smale_complex.cells, morse_smale_complex.topology_issues
            )

            print_morse_smale_summary(
                critical_point_assessments=critical_point_assessments,
                morse_smale_complex=morse_smale_complex,
                cell_assessments=cell_assessments,
            )

            plot_morse_smale_cells(
                board=game.board,
                analysis=previous_analysis,
                surface=surface,
                morse_smale_complex=morse_smale_complex,
                cell_assessments=cell_assessments,
            )

            continue

        if move_text == "audio":
            if previous_analysis is None:
                print(
                    "Все още няма позиция "
                    "за визуализиране."
                )
                continue

            latest_dynamics = (
                dynamics_history[-1]
                if dynamics_history
                else None
            )

            output_path = export_move_audio(
                analysis=previous_analysis,
                dynamics=latest_dynamics,
                index=len(analysis_history),
            )

            print(f"Аудио клипът беше записан в {output_path}")
            continue

        if move_text == "history":
            print_history(
                analysis_history=analysis_history,
                dynamics_history=dynamics_history,
            )
            continue

        if move_text == "export":
            if not analysis_history:
                print("Все още няма анализ за записване.")
                continue

            output_path = export_game_analysis(
                analysis_history= analysis_history,
                dynamics_history = dynamics_history
            )

            print(f"Анализът беше записан в {output_path}")
            continue

        if move_text == "export_audio":
            if not analysis_history:
                print("Все още няма анализ за озвучаване.")
                continue

            output_paths = export_game_audio(
                analysis_history=analysis_history,
                dynamics_history=dynamics_history,
            )

            output_directory = output_paths[0].parent

            print(
                f"Аудио файловете за партията бяха записани в "
                f"{output_directory} ({len(output_paths)} файла)."
            )

            continue

        if move_text == "plot":
            if previous_analysis is None:
                print("Все още няма позиция за визуализиране.")
                continue

            plot_position_analysis(previous_analysis)
            continue

        move_details = game.make_move(move_text)

        if move_details is None:
            print("Невалиден или непозволен ход.")
            continue

        analysis = analyze_position(
            game.board,
            move_details,
        )

        analysis_history.append(analysis)

        print_position_analysis(analysis)

        if previous_analysis is None:
            print(
                "\n--- Динамика ---"
                "\nТова е първият ход и все още няма "
                "предишна позиция за сравнение."
            )
        else:
            dynamics = analyze_dynamics(
                previous=previous_analysis,
                current=analysis,
            )

            dynamics_history.append(dynamics)

            print_dynamics(dynamics)

        previous_analysis = analysis

    print("\nИграта приключи.")
    print(f"Резултат: {game.board.result()}")

    # The while loop above only exits this way after at least one move
    # has been played and appended -- the quit/exit branch returns
    # directly from inside the loop and never reaches this code, so
    # this never fires just from quitting an unfinished game. This is
    # the fix for the final (e.g. checkmating) move's audio otherwise
    # being unreachable: the REPL loop itself is intentionally left
    # unchanged, and every move's clip -- including the last one -- is
    # exported here instead, once the game is actually over.
    if analysis_history:
        output_paths = export_game_audio(
            analysis_history=analysis_history,
            dynamics_history=dynamics_history,
        )

        output_directory = output_paths[0].parent

        print(
            f"Играта приключи — аудио клиповете за цялата партия "
            f"бяха автоматично записани в {output_directory} "
            f"({len(output_paths)} файла)."
        )

    print_history(
        analysis_history=analysis_history,
        dynamics_history=dynamics_history,
    )


if __name__ == "__main__":
    main()