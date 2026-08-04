from analysis.dynamics import analyze_dynamics
from analysis.heatmap import print_heatmap
from chess_engine.analyzer import analyze_position
from chess_engine.board import ChessGame
from chess_engine.models import (
    DynamicsAnalysis,
    MoveAnalysis,
)
from console_app.export import export_game_analysis
from visualization.heatmap_plot import plot_position_analysis
from visualization.position_plot import plot_position


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

    print("\n--- Контрол ---")
    print(
        f"Белите контролират: "
        f"{analysis.control.white_controlled_squares} полета"
    )
    print(
        f"Черните контролират: "
        f"{analysis.control.black_controlled_squares} полета"
    )
    print(f"Разлика: {analysis.control.difference:+d}")

    print_heatmap(analysis.heatmap)

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
        f"Промяна в белия контрол: "
        f"{dynamics.white_control_delta:+d}"
    )
    print(
        f"Промяна в черния контрол: "
        f"{dynamics.black_control_delta:+d}"
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

    print_history(
        analysis_history=analysis_history,
        dynamics_history=dynamics_history,
    )


if __name__ == "__main__":
    main()