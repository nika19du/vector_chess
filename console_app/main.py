from analysis.dynamics import analyze_dynamics
from analysis.attacker_count import print_heatmap
from chess_engine.analyzer import analyze_position
from chess_engine.board import ChessGame
from chess_engine.models import (
    DynamicsAnalysis,
    MoveAnalysis,
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
from visualization.attack_influence_plot import plot_attack_influence_field
from analysis.gradient_field import (
    print_gradient_field,
)
from visualization.equipotential_plot import (
    plot_equipotential_field,
)
from visualization.gradient_plot import plot_gradient_field
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