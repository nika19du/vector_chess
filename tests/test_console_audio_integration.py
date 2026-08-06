import re
import wave
from pathlib import Path

import console_app.export as console_export
import console_app.main as console_main

FILENAME_PATTERN = re.compile(r"^\d+_[a-h][1-8][a-h][1-8][a-z]?\.wav$")


def _run_console(monkeypatch, tmp_path, commands: list[str]) -> None:
    monkeypatch.chdir(tmp_path)

    commands_iter = iter(commands)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(commands_iter))

    console_main.main()


def _audio_dir(tmp_path: Path) -> Path:
    return tmp_path / "output" / "audio"


# ---------------------------------------------------------
# audio before any move
# ---------------------------------------------------------


def test_audio_before_any_move_prints_guard_and_writes_nothing(
    monkeypatch, tmp_path, capsys
):
    _run_console(monkeypatch, tmp_path, ["audio", "quit"])

    captured = capsys.readouterr()
    assert "Все още няма позиция" in captured.out
    assert not _audio_dir(tmp_path).exists()


# ---------------------------------------------------------
# Dynamics wiring: first move (None) vs later move (reused)
# ---------------------------------------------------------


def test_first_move_audio_uses_no_dynamics(monkeypatch, tmp_path):
    recorded_dynamics = []
    original_build_audio_mapping = console_export.build_audio_mapping

    def spy(analysis, dynamics):
        recorded_dynamics.append(dynamics)
        return original_build_audio_mapping(analysis, dynamics)

    monkeypatch.setattr(console_export, "build_audio_mapping", spy)

    _run_console(monkeypatch, tmp_path, ["e2e4", "audio", "quit"])

    assert recorded_dynamics == [None]

    files = list(_audio_dir(tmp_path).glob("*.wav"))
    assert len(files) == 1
    assert FILENAME_PATTERN.match(files[0].name)
    assert "e2e4" in files[0].name


def test_later_move_audio_reuses_the_latest_dynamics(monkeypatch, tmp_path):
    recorded_dynamics = []
    original_build_audio_mapping = console_export.build_audio_mapping

    def spy(analysis, dynamics):
        recorded_dynamics.append(dynamics)
        return original_build_audio_mapping(analysis, dynamics)

    monkeypatch.setattr(console_export, "build_audio_mapping", spy)

    _run_console(monkeypatch, tmp_path, ["e2e4", "e7e5", "audio", "quit"])

    assert len(recorded_dynamics) == 1
    assert recorded_dynamics[0] is not None
    assert recorded_dynamics[0].label in {"calm", "active", "tense", "chaotic"}


# ---------------------------------------------------------
# No duplicate analysis
# ---------------------------------------------------------


def test_audio_does_not_trigger_duplicate_analysis(monkeypatch, tmp_path):
    call_count = {"n": 0}
    original_analyze_position = console_main.analyze_position

    def counting_analyze_position(board, move_details):
        call_count["n"] += 1
        return original_analyze_position(board, move_details)

    monkeypatch.setattr(console_main, "analyze_position", counting_analyze_position)

    _run_console(monkeypatch, tmp_path, ["e2e4", "audio", "audio", "quit"])

    # One real move was played; "audio" was invoked twice for it.
    # analyze_position must have run exactly once, not three times.
    assert call_count["n"] == 1


# ---------------------------------------------------------
# Output location, naming, and format
# ---------------------------------------------------------


def test_output_lands_in_output_audio_with_a_readable_filename(monkeypatch, tmp_path):
    _run_console(monkeypatch, tmp_path, ["g1f3", "audio", "quit"])

    files = list(_audio_dir(tmp_path).glob("*.wav"))
    assert len(files) == 1
    assert FILENAME_PATTERN.match(files[0].name)
    assert "g1f3" in files[0].name


def test_generated_wav_is_valid(monkeypatch, tmp_path):
    _run_console(monkeypatch, tmp_path, ["e2e4", "audio", "quit"])

    files = list(_audio_dir(tmp_path).glob("*.wav"))
    assert len(files) == 1

    with wave.open(str(files[0]), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() > 0
        assert wav_file.getnframes() > 0


def test_consecutive_moves_produce_distinct_non_overwritten_files(
    monkeypatch, tmp_path
):
    _run_console(monkeypatch, tmp_path, ["e2e4", "audio", "e7e5", "audio", "quit"])

    files = sorted(_audio_dir(tmp_path).glob("*.wav"))
    assert len(files) == 2
    assert files[0].name != files[1].name
    assert "e2e4" in files[0].name
    assert "e7e5" in files[1].name


# ---------------------------------------------------------
# Existing commands unaffected, output printed in Bulgarian
# ---------------------------------------------------------


def test_existing_commands_remain_unaffected(monkeypatch, tmp_path, capsys):
    _run_console(
        monkeypatch, tmp_path, ["e2e4", "history", "export", "audio", "quit"]
    )

    captured = capsys.readouterr()
    assert "История на партията" in captured.out
    assert (tmp_path / "output" / "game_analysis.json").exists()
    assert len(list(_audio_dir(tmp_path).glob("*.wav"))) == 1


def test_prints_the_output_path_in_bulgarian(monkeypatch, tmp_path, capsys):
    _run_console(monkeypatch, tmp_path, ["e2e4", "audio", "quit"])

    captured = capsys.readouterr()
    assert "Аудио клипът беше записан в" in captured.out
