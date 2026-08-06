import json
from dataclasses import asdict
from pathlib import Path

from audio.export import write_wav
from audio.mapping import build_audio_mapping
from audio.renderer import AudioRenderer
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


def export_move_audio(
    analysis: MoveAnalysis,
    dynamics: DynamicsAnalysis | None,
    index: int | None = None,
    file_name: str | None = None,
) -> Path:
    """
    Renders one move's Audio Layer MVP clip and writes it to
    output/audio/. Move -> AudioMapping -> AudioRenderer -> .wav,
    per docs/audio.md.
    """

    output_directory = Path("output") / "audio"
    output_directory.mkdir(parents=True, exist_ok=True)

    if file_name is None:
        prefix = f"{index:03d}_" if index is not None else ""
        file_name = f"{prefix}{analysis.move}.wav"

    output_path = output_directory / file_name

    mapping = build_audio_mapping(analysis, dynamics)
    clip = AudioRenderer().render(mapping)

    return write_wav(clip, output_path)