import json
from dataclasses import asdict
from pathlib import Path

from chess_engine.models import DynamicsAnalysis, MoveAnalysis


def export_game_analysis(
    analysis_history: list[MoveAnalysis],
    dynamics_history: list[DynamicsAnalysis],
    file_name: str = "game_analysis.json",
) -> Path:
    """
    Записва анализа на партията в JSON файл.
    """

    output_directory = Path("output")
    output_directory.mkdir(exist_ok=True)

    output_path = output_directory / file_name

    data = {
        "moves": [
            asdict(analysis)
            for analysis in analysis_history
        ],
        "dynamics": [
            asdict(dynamics)
            for dynamics in dynamics_history
        ],
    }

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path