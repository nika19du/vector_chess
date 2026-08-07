import matplotlib

matplotlib.use("Agg")

import console_app.main as console_main
from analysis.attack_influence_surface import build_attack_influence_surface
from analysis.critical_point_quality import assess_critical_point_quality
from analysis.critical_points import classify_critical_points, locate_critical_points
from analysis.morse_smale import (
    assemble_morse_smale_cells,
    assess_morse_smale_cell_quality,
    locate_morse_smale_separatrices,
)
from chess_engine.analyzer import analyze_position
from chess_engine.board import ChessGame
from chess_engine.models import (
    ClassifiedCriticalPoint,
    MorseSmaleCell,
    MorseSmaleCellQualityAssessment,
    MorseSmaleCellRejectionReason,
    MorseSmaleCellRejectionReasonKind,
    MorseSmaleComplex,
    SeparatrixPath,
    TopologyIssue,
    TopologyIssueKind,
)


def _run_console(monkeypatch, tmp_path, commands: list[str]) -> None:
    monkeypatch.chdir(tmp_path)

    commands_iter = iter(commands)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(commands_iter))

    console_main.main()


def _real_pipeline_morse_smale(moves: list[str]):
    game = ChessGame()
    move_details = None
    for move in moves:
        move_details = game.make_move(move)
    analysis = analyze_position(game.board, move_details)

    surface = build_attack_influence_surface(analysis.attack_influence_field)
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)
    critical_point_assessments = assess_critical_point_quality(classified, surface)

    separatrices = locate_morse_smale_separatrices(surface, critical_point_assessments)
    morse_smale_complex = assemble_morse_smale_cells(separatrices)
    cell_assessments = assess_morse_smale_cell_quality(
        morse_smale_complex.cells, morse_smale_complex.topology_issues
    )

    return critical_point_assessments, morse_smale_complex, cell_assessments


# ---------------------------------------------------------
# Command before any move
# ---------------------------------------------------------


def test_morse_smale_plot_before_any_move_prints_guard(monkeypatch, tmp_path, capsys):
    _run_console(monkeypatch, tmp_path, ["morse_smale_plot", "quit"])

    captured = capsys.readouterr()
    assert "Все още няма позиция" in captured.out


# ---------------------------------------------------------
# Command after one valid move / headless rendering succeeds
# ---------------------------------------------------------


def test_morse_smale_plot_after_one_move_succeeds_headless(monkeypatch, tmp_path, capsys):
    _run_console(monkeypatch, tmp_path, ["e2e4", "morse_smale_plot", "quit"])

    captured = capsys.readouterr()
    assert "Morse-Smale Complex" in captured.out


# ---------------------------------------------------------
# Summary counts match the real pipeline
# ---------------------------------------------------------


def test_summary_counts_match_the_real_pipeline(monkeypatch, tmp_path, capsys):
    critical_point_assessments, morse_smale_complex, cell_assessments = (
        _real_pipeline_morse_smale(["e2e4"])
    )

    accepted_points = [a for a in critical_point_assessments if a.is_accepted]
    accepted_saddles = sum(
        1 for a in accepted_points if a.point.classification == "saddle"
    )
    closed_cells = sum(1 for c in morse_smale_complex.cells if c.is_closed)
    open_cells = len(morse_smale_complex.cells) - closed_cells
    accepted_cells = sum(1 for a in cell_assessments if a.is_accepted)
    rejected_cells = len(cell_assessments) - accepted_cells

    _run_console(monkeypatch, tmp_path, ["e2e4", "morse_smale_plot", "quit"])

    captured = capsys.readouterr()
    assert f"Приети критични точки: {len(accepted_points)}" in captured.out
    assert f"Използвани седла: {accepted_saddles}" in captured.out
    assert f"Сепаратриси: {len(morse_smale_complex.edges)}" in captured.out
    assert f"Затворени клетки: {closed_cells}" in captured.out
    assert f"Отворени/непълни клетки: {open_cells}" in captured.out
    assert f"Приети клетки: {accepted_cells}" in captured.out
    assert f"Отхвърлени клетки: {rejected_cells}" in captured.out
    assert (
        f"Топологични проблеми: {len(morse_smale_complex.topology_issues)}"
        in captured.out
    )


# ---------------------------------------------------------
# print_morse_smale_summary: zero-cell case
# ---------------------------------------------------------


def test_print_morse_smale_summary_handles_zero_cells(capsys):
    empty_complex = MorseSmaleComplex(
        vertices=[], edges=[], cells=[], unassigned_separatrices=[], topology_issues=[]
    )

    console_main.print_morse_smale_summary(
        critical_point_assessments=[],
        morse_smale_complex=empty_complex,
        cell_assessments=[],
    )

    captured = capsys.readouterr()
    assert "Приети критични точки: 0" in captured.out
    assert "Използвани седла: 0" in captured.out
    assert "Сепаратриси: 0" in captured.out
    assert "Затворени клетки: 0" in captured.out
    assert "Отворени/непълни клетки: 0" in captured.out
    assert "Приети клетки: 0" in captured.out
    assert "Отхвърлени клетки: 0" in captured.out
    assert "Топологични проблеми: 0" in captured.out
    assert "Причини за отхвърляне:" not in captured.out  # nothing to group


# ---------------------------------------------------------
# print_morse_smale_summary: topology-issue reporting and
# all-reasons-counted grouping
# ---------------------------------------------------------


def test_print_morse_smale_summary_reports_topology_issues_and_all_reasons():
    def make_point(x, y, classification):
        return ClassifiedCriticalPoint(
            x=x, y=y, value=0.0, gradient_norm=0.0, status="converged", iterations=1,
            f_xx=-1.0, f_xy=0.0, f_yx=0.0, f_yy=-1.0,
            eigenvalue_min=-1.0, eigenvalue_max=1.0, classification=classification,
        )

    saddle = make_point(1.0, 1.0, "saddle")
    maximum = make_point(1.5, 1.0, "maximum")
    separatrix = SeparatrixPath(
        start_saddle=saddle, flow_direction="ascending", points=[],
        end_critical_point=maximum, termination_status="reached_critical_point",
        step_count=0, path_length=0.0,
    )

    cell = MorseSmaleCell(
        cell_id=0,
        boundary_separatrices=[separatrix],
        boundary_traversal_directions=[True],
        boundary_critical_points=[maximum],
        is_closed=False,
        open_boundaries=[separatrix],
    )

    complex_with_issue = MorseSmaleComplex(
        vertices=[saddle, maximum],
        edges=[separatrix],
        cells=[cell],
        unassigned_separatrices=[],
        topology_issues=[
            TopologyIssue(
                kind=TopologyIssueKind.ZERO_LENGTH_EDGE,
                critical_point=None,
                separatrix=separatrix,
                detail="synthetic issue for the test",
            )
        ],
    )

    assessment = MorseSmaleCellQualityAssessment(
        cell=cell,
        is_accepted=False,
        rejection_reasons=[
            MorseSmaleCellRejectionReason(
                kind=MorseSmaleCellRejectionReasonKind.NOT_CLOSED,
                detail="cell boundary walk did not close",
                related_topology_issue=None,
            ),
            MorseSmaleCellRejectionReason(
                kind=MorseSmaleCellRejectionReasonKind.TOO_FEW_DISTINCT_BOUNDARY_POINTS,
                detail="1 distinct boundary points < min_distinct_boundary_points=3",
                related_topology_issue=None,
            ),
        ],
    )

    import io
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        console_main.print_morse_smale_summary(
            critical_point_assessments=[],
            morse_smale_complex=complex_with_issue,
            cell_assessments=[assessment],
        )
    output = buffer.getvalue()

    assert "Топологични проблеми: 1" in output
    assert "Отхвърлени клетки: 1" in output
    # BOTH reasons on the single rejected cell must be counted, not
    # just the first -- matches Phase 3's "never just the first
    # reason" design.
    assert "клетката не е затворена: 1" in output
    assert "твърде малко различни гранични точки: 1" in output


# ---------------------------------------------------------
# Accepted-only rendering by default
# ---------------------------------------------------------


def test_plot_morse_smale_cells_is_called_without_overriding_the_default(
    monkeypatch, tmp_path
):
    recorded_calls = []
    original_plot = console_main.plot_morse_smale_cells

    def spy(*args, **kwargs):
        recorded_calls.append(kwargs)
        return original_plot(*args, **kwargs)

    monkeypatch.setattr(console_main, "plot_morse_smale_cells", spy)

    _run_console(monkeypatch, tmp_path, ["e2e4", "morse_smale_plot", "quit"])

    assert len(recorded_calls) == 1
    # console_app never passes show_only_accepted itself -- it relies
    # entirely on visualization/morse_smale_plot.py's own default
    # (True), same principle as critical_points_plot/ridge_valley_plot.
    assert "show_only_accepted" not in recorded_calls[0]


# ---------------------------------------------------------
# No duplicate analysis / no duplicate Morse-Smale computation
# ---------------------------------------------------------


def test_morse_smale_plot_does_not_trigger_duplicate_analysis(monkeypatch, tmp_path):
    call_count = {"n": 0}
    original_analyze_position = console_main.analyze_position

    def counting_analyze_position(board, move_details):
        call_count["n"] += 1
        return original_analyze_position(board, move_details)

    monkeypatch.setattr(console_main, "analyze_position", counting_analyze_position)

    _run_console(
        monkeypatch,
        tmp_path,
        ["e2e4", "morse_smale_plot", "morse_smale_plot", "quit"],
    )

    assert call_count["n"] == 1


def test_morse_smale_plot_computes_the_pipeline_exactly_once_per_command(
    monkeypatch, tmp_path
):
    # Unlike critical_points_plot/ridge_valley_plot,
    # visualization.morse_smale_plot does not import any of these
    # pipeline functions at all (Phase 4's design: complex and
    # cell_assessments are required, never recomputed inside the
    # renderer) -- so there is exactly one place left that could call
    # them: console_app.main itself. Patching only that reference is
    # therefore a complete check, not merely half of one.
    call_counts = {
        "surface": 0, "critical_points": 0, "separatrices": 0, "cells": 0,
    }

    original_surface = console_main.build_attack_influence_surface
    original_critical_points = console_main.locate_critical_points
    original_separatrices = console_main.locate_morse_smale_separatrices
    original_cells = console_main.assemble_morse_smale_cells

    def counting(name, original):
        def wrapper(*args, **kwargs):
            call_counts[name] += 1
            return original(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(
        console_main, "build_attack_influence_surface", counting("surface", original_surface)
    )
    monkeypatch.setattr(
        console_main, "locate_critical_points", counting("critical_points", original_critical_points)
    )
    monkeypatch.setattr(
        console_main, "locate_morse_smale_separatrices",
        counting("separatrices", original_separatrices),
    )
    monkeypatch.setattr(
        console_main, "assemble_morse_smale_cells", counting("cells", original_cells)
    )

    _run_console(monkeypatch, tmp_path, ["e2e4", "morse_smale_plot", "quit"])

    assert call_counts == {
        "surface": 1, "critical_points": 1, "separatrices": 1, "cells": 1,
    }


def test_morse_smale_plot_passes_the_same_objects_used_for_the_summary(
    monkeypatch, tmp_path
):
    created_complexes = []
    original_assemble = console_main.assemble_morse_smale_cells

    def capturing_assemble(*args, **kwargs):
        result = original_assemble(*args, **kwargs)
        created_complexes.append(result)
        return result

    monkeypatch.setattr(console_main, "assemble_morse_smale_cells", capturing_assemble)

    recorded_calls = []
    original_plot = console_main.plot_morse_smale_cells

    def spy(*args, **kwargs):
        recorded_calls.append(kwargs)
        return original_plot(*args, **kwargs)

    monkeypatch.setattr(console_main, "plot_morse_smale_cells", spy)

    _run_console(monkeypatch, tmp_path, ["e2e4", "morse_smale_plot", "quit"])

    assert len(created_complexes) == 1
    assert recorded_calls[0]["morse_smale_complex"] is created_complexes[0]
    assert recorded_calls[0]["cell_assessments"] is not None
    assert recorded_calls[0]["surface"] is not None


# ---------------------------------------------------------
# No mutation of recorded history
# ---------------------------------------------------------


def test_morse_smale_plot_does_not_change_recorded_history(monkeypatch, tmp_path, capsys):
    _run_console(
        monkeypatch,
        tmp_path,
        ["e2e4", "e7e5", "morse_smale_plot", "morse_smale_plot", "history", "quit"],
    )

    captured = capsys.readouterr()
    assert "Ход 1: e2e4" in captured.out
    assert "Ход 2: e7e5" in captured.out
    assert captured.out.count("Ход 1:") == 1
    assert captured.out.count("Ход 2:") == 1


# ---------------------------------------------------------
# Existing commands unaffected
# ---------------------------------------------------------


def test_existing_commands_remain_unaffected(monkeypatch, tmp_path, capsys):
    _run_console(
        monkeypatch,
        tmp_path,
        [
            "e2e4",
            "history",
            "export",
            "critical_points_plot",
            "ridge_valley_plot",
            "audio",
            "morse_smale_plot",
            "quit",
        ],
    )

    captured = capsys.readouterr()
    assert "История на партията" in captured.out
    assert (tmp_path / "output" / "game_analysis.json").exists()
    assert "Критични точки" in captured.out
    assert "Ridge / Valley" in captured.out
    assert "Аудио клипът беше записан" in captured.out
    assert "Morse-Smale Complex" in captured.out
    assert (tmp_path / "output" / "audio").exists()
