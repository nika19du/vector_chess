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


# ---------------------------------------------------------
# export_audio: whole-game batch export
# ---------------------------------------------------------


def test_export_audio_before_any_move_prints_guard_and_writes_nothing(
    monkeypatch, tmp_path, capsys
):
    _run_console(monkeypatch, tmp_path, ["export_audio", "quit"])

    captured = capsys.readouterr()
    assert "Все още няма анализ" in captured.out
    assert not _audio_dir(tmp_path).exists()


def test_export_audio_produces_one_file_per_move(monkeypatch, tmp_path):
    _run_console(
        monkeypatch, tmp_path, ["e2e4", "e7e5", "g1f3", "export_audio", "quit"]
    )

    files = sorted(_audio_dir(tmp_path).glob("*.wav"))
    assert len(files) == 3
    assert [f.name for f in files] == sorted(f.name for f in files)
    assert "e2e4" in files[0].name
    assert "e7e5" in files[1].name
    assert "g1f3" in files[2].name


def test_export_audio_matches_individual_audio_exports_byte_for_byte(
    monkeypatch, tmp_path
):
    individually_dir = tmp_path / "individually"
    batch_dir = tmp_path / "batch"
    individually_dir.mkdir()
    batch_dir.mkdir()

    _run_console(
        monkeypatch,
        individually_dir,
        ["e2e4", "audio", "e7e5", "audio", "g1f3", "audio", "quit"],
    )
    _run_console(
        monkeypatch, batch_dir, ["e2e4", "e7e5", "g1f3", "export_audio", "quit"]
    )

    individually_files = sorted(_audio_dir(individually_dir).glob("*.wav"))
    batch_files = sorted(_audio_dir(batch_dir).glob("*.wav"))

    assert [f.name for f in individually_files] == [f.name for f in batch_files]
    for one, other in zip(individually_files, batch_files):
        assert one.read_bytes() == other.read_bytes()


def test_export_audio_pairs_dynamics_correctly_across_the_whole_game(
    monkeypatch, tmp_path
):
    recorded_dynamics = []
    original_build_audio_mapping = console_export.build_audio_mapping

    def spy(analysis, dynamics):
        recorded_dynamics.append(dynamics)
        return original_build_audio_mapping(analysis, dynamics)

    monkeypatch.setattr(console_export, "build_audio_mapping", spy)

    _run_console(
        monkeypatch, tmp_path, ["e2e4", "e7e5", "g1f3", "export_audio", "quit"]
    )

    assert len(recorded_dynamics) == 3
    assert recorded_dynamics[0] is None
    assert recorded_dynamics[1] is not None
    assert recorded_dynamics[2] is not None


def test_export_audio_is_idempotent(monkeypatch, tmp_path):
    _run_console(
        monkeypatch,
        tmp_path,
        ["e2e4", "e7e5", "export_audio", "export_audio", "quit"],
    )

    files = sorted(_audio_dir(tmp_path).glob("*.wav"))
    assert len(files) == 2  # re-running did not create duplicates


# ---------------------------------------------------------
# Game-end automatic export -- the checkmate-reachability fix
# ---------------------------------------------------------


def test_checkmate_without_any_audio_command_still_exports_every_move(
    monkeypatch, tmp_path, capsys
):
    # 1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6?? 4.Qxf7# -- Scholar's mate. No
    # "audio"/"export_audio" command is ever typed; the loop ends
    # itself the moment checkmate is reached.
    _run_console(
        monkeypatch,
        tmp_path,
        ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"],
    )

    files = sorted(_audio_dir(tmp_path).glob("*.wav"))
    assert len(files) == 7

    # The move that was previously unreachable: the mating move itself.
    assert any(f.name == "007_h5f7.wav" for f in files)

    captured = capsys.readouterr()
    assert "Играта приключи" in captured.out
    assert "Резултат: 1-0" in captured.out


def test_stalemate_without_any_audio_command_still_exports_every_move(
    monkeypatch, tmp_path
):
    # A verified fastest-stalemate line (not checkmate), proving the
    # fix is not special-cased to checkmate: it fires for any
    # game.is_game_over() reason.
    moves = [
        "e2e3", "a7a5", "d1h5", "a8a6", "h5a5", "h7h5", "a5c7", "a6h6",
        "h2h4", "f7f6", "c7d7", "e8f7", "d7b7", "d8d3", "b7b8", "d3h7",
        "b8c8", "f7g6", "c8e6",
    ]

    _run_console(monkeypatch, tmp_path, moves)

    files = sorted(_audio_dir(tmp_path).glob("*.wav"))
    assert len(files) == 19
    assert any(f.name == "019_c8e6.wav" for f in files)


def test_game_end_summary_is_distinguishable_from_manual_export_audio(
    monkeypatch, tmp_path, capsys
):
    manual_dir = tmp_path / "manual"
    auto_dir = tmp_path / "auto"
    manual_dir.mkdir()
    auto_dir.mkdir()

    _run_console(monkeypatch, manual_dir, ["e2e4", "export_audio", "quit"])
    manual_message = capsys.readouterr().out

    _run_console(
        monkeypatch,
        auto_dir,
        ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"],
    )
    auto_message = capsys.readouterr().out

    # The help banner (printed at the start of every session, per the
    # "document this clearly" requirement) also mentions "автоматично",
    # so the distinguishing check must target the actual summary
    # phrasing, not just that one word's presence anywhere in stdout.
    assert "Аудио файловете за партията бяха записани в" in manual_message
    assert "бяха автоматично записани" not in manual_message

    assert "бяха автоматично записани" in auto_message
    assert "Играта приключи" in auto_message
    assert "Аудио файловете за партията бяха записани в" not in auto_message


def test_quit_before_game_over_does_not_trigger_automatic_export(
    monkeypatch, tmp_path
):
    _run_console(monkeypatch, tmp_path, ["e2e4", "quit"])

    assert not _audio_dir(tmp_path).exists()


def test_game_over_messaging_still_prints_result_and_history(
    monkeypatch, tmp_path, capsys
):
    _run_console(
        monkeypatch,
        tmp_path,
        ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"],
    )

    captured = capsys.readouterr()
    assert "Играта приключи." in captured.out
    assert "Резултат: 1-0" in captured.out
    assert "История на партията" in captured.out
