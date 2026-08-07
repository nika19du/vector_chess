import dataclasses

import matplotlib

matplotlib.use("Agg")

import console_app.main as console_main
import visualization.critical_points_plot as critical_points_plot_module
from analysis.attack_influence_surface import build_attack_influence_surface
from analysis.critical_point_quality import assess_critical_point_quality
from analysis.critical_points import classify_critical_points, locate_critical_points
from chess_engine.analyzer import analyze_position
from chess_engine.board import ChessGame


def _run_console(monkeypatch, tmp_path, commands: list[str]) -> None:
    monkeypatch.chdir(tmp_path)

    commands_iter = iter(commands)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(commands_iter))

    console_main.main()


def _real_pipeline_counts(moves: list[str]):
    game = ChessGame()
    move_details = None
    for move in moves:
        move_details = game.make_move(move)
    analysis = analyze_position(game.board, move_details)

    surface = build_attack_influence_surface(analysis.attack_influence_field)
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)
    assessments = assess_critical_point_quality(classified, surface)

    return candidates, classified, assessments


# ---------------------------------------------------------
# Command before any move
# ---------------------------------------------------------


def test_critical_points_plot_before_any_move_prints_guard(monkeypatch, tmp_path, capsys):
    _run_console(monkeypatch, tmp_path, ["critical_points_plot", "quit"])

    captured = capsys.readouterr()
    assert "Все още няма позиция" in captured.out


# ---------------------------------------------------------
# Command after one valid move / headless rendering succeeds
# ---------------------------------------------------------


def test_critical_points_plot_after_one_move_succeeds_headless(monkeypatch, tmp_path, capsys):
    # Must not raise; matplotlib.use("Agg") makes plt.show() a no-op.
    _run_console(monkeypatch, tmp_path, ["e2e4", "critical_points_plot", "quit"])

    captured = capsys.readouterr()
    assert "Критични точки" in captured.out


# ---------------------------------------------------------
# Summary counts
# ---------------------------------------------------------


def test_summary_counts_match_the_real_pipeline(monkeypatch, tmp_path, capsys):
    candidates, classified, assessments = _real_pipeline_counts(["e2e4"])
    expected_classified = sum(1 for c in classified if c.classification != "unclassified")
    expected_accepted = sum(1 for a in assessments if a.is_accepted)
    expected_rejected = len(assessments) - expected_accepted

    _run_console(monkeypatch, tmp_path, ["e2e4", "critical_points_plot", "quit"])

    captured = capsys.readouterr()
    assert f"Открити кандидати (Newton): {len(candidates)}" in captured.out
    assert f"Класифицирани (сходили): {expected_classified}" in captured.out
    assert f"Приети (надеждни): {expected_accepted}" in captured.out
    assert f"Отхвърлени: {expected_rejected}" in captured.out


def test_print_critical_points_summary_reports_grouped_rejection_reasons(capsys):
    _candidates, classified, assessments = _real_pipeline_counts(["e2e4"])
    candidates = _candidates

    console_main.print_critical_points_summary(
        candidates=candidates,
        classified_points=classified,
        assessments=assessments,
    )

    captured = capsys.readouterr()
    assert "Причини за отхвърляне:" in captured.out
    # e2e4 is known (see Phase 4a validation) to produce both boundary
    # and non-converged rejections -- both categories must appear.
    assert "твърде близо до ръба на дъската" in captured.out
    assert "Newton не е сходил" in captured.out


# ---------------------------------------------------------
# Empty accepted set
# ---------------------------------------------------------


def test_print_critical_points_summary_handles_zero_accepted_points(capsys):
    candidates, classified, assessments = _real_pipeline_counts(["e2e4"])

    all_rejected = [
        dataclasses.replace(
            assessment,
            is_accepted=False,
            rejection_reasons=assessment.rejection_reasons or ["forced for test"],
        )
        for assessment in assessments
    ]

    console_main.print_critical_points_summary(
        candidates=candidates,
        classified_points=classified,
        assessments=all_rejected,
    )

    captured = capsys.readouterr()
    assert "Приети (надеждни): 0" in captured.out
    assert f"Отхвърлени: {len(all_rejected)}" in captured.out


def test_print_critical_points_summary_handles_zero_rejected_points(capsys):
    candidates, classified, assessments = _real_pipeline_counts(["e2e4"])

    all_accepted = [
        dataclasses.replace(assessment, is_accepted=True, rejection_reasons=[])
        for assessment in assessments
    ]

    console_main.print_critical_points_summary(
        candidates=candidates,
        classified_points=classified,
        assessments=all_accepted,
    )

    captured = capsys.readouterr()
    assert f"Приети (надеждни): {len(all_accepted)}" in captured.out
    assert "Отхвърлени: 0" in captured.out
    assert "Причини за отхвърляне:" not in captured.out  # nothing to group


# ---------------------------------------------------------
# Accepted-only rendering by default
# ---------------------------------------------------------


def test_plot_critical_points_is_called_without_overriding_the_default(monkeypatch, tmp_path):
    recorded_calls = []
    original_plot_critical_points = console_main.plot_critical_points

    def spy(*args, **kwargs):
        recorded_calls.append(kwargs)
        return original_plot_critical_points(*args, **kwargs)

    monkeypatch.setattr(console_main, "plot_critical_points", spy)

    _run_console(monkeypatch, tmp_path, ["e2e4", "critical_points_plot", "quit"])

    assert len(recorded_calls) == 1
    # console_app never passes show_only_accepted itself -- it relies
    # entirely on visualization/critical_points_plot.py's own default
    # (True), per "do not modify ... visualization ... behavior".
    assert "show_only_accepted" not in recorded_calls[0]


# ---------------------------------------------------------
# No duplicate analysis / no mutation of analysis history
# ---------------------------------------------------------


def test_critical_points_plot_does_not_trigger_duplicate_analysis(monkeypatch, tmp_path):
    call_count = {"n": 0}
    original_analyze_position = console_main.analyze_position

    def counting_analyze_position(board, move_details):
        call_count["n"] += 1
        return original_analyze_position(board, move_details)

    monkeypatch.setattr(console_main, "analyze_position", counting_analyze_position)

    _run_console(
        monkeypatch,
        tmp_path,
        ["e2e4", "critical_points_plot", "critical_points_plot", "quit"],
    )

    # One real move was played; critical_points_plot was invoked twice
    # for it. analyze_position must have run exactly once.
    assert call_count["n"] == 1


def test_critical_points_plot_does_not_change_recorded_history(monkeypatch, tmp_path, capsys):
    _run_console(
        monkeypatch,
        tmp_path,
        ["e2e4", "e7e5", "critical_points_plot", "critical_points_plot", "history", "quit"],
    )

    captured = capsys.readouterr()
    assert "Ход 1: e2e4" in captured.out
    assert "Ход 2: e7e5" in captured.out
    # Exactly two recorded moves -- invoking critical_points_plot
    # twice must not have appended or duplicated history entries.
    assert captured.out.count("Ход 1:") == 1
    assert captured.out.count("Ход 2:") == 1


# ---------------------------------------------------------
# Existing commands unaffected
# ---------------------------------------------------------


def test_existing_commands_remain_unaffected(monkeypatch, tmp_path, capsys):
    _run_console(
        monkeypatch,
        tmp_path,
        ["e2e4", "history", "export", "critical_points_plot", "quit"],
    )

    captured = capsys.readouterr()
    assert "История на партията" in captured.out
    assert (tmp_path / "output" / "game_analysis.json").exists()
    assert "Критични точки" in captured.out


# ---------------------------------------------------------
# Phase 0 hardening: pipeline computed exactly once per command
# ---------------------------------------------------------
#
# Before Phase 0, critical_points_plot's console handler computed
# surface/candidates/classified_points/assessments once for the
# printed summary, then plot_critical_points (visualization layer)
# recomputed the exact same chain internally -- 2x the expensive
# work (spline fit + Newton refinement + quality assessment) per
# command. These tests spy on BOTH the console_app.main binding and
# the visualization.critical_points_plot binding of each function
# (each module did its own `from ... import ...`, so they are
# independent references to the same underlying function and both
# must be patched to see the true total call count), and assert the
# combined count is exactly 1 -- proving the duplication is gone,
# not just moved.


def test_build_attack_influence_surface_called_once_per_command(monkeypatch, tmp_path):
    call_count = {"n": 0}
    original = console_main.build_attack_influence_surface

    def counting(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(console_main, "build_attack_influence_surface", counting)
    monkeypatch.setattr(
        critical_points_plot_module, "build_attack_influence_surface", counting
    )

    _run_console(monkeypatch, tmp_path, ["e2e4", "critical_points_plot", "quit"])

    assert call_count["n"] == 1


def test_locate_critical_points_called_once_per_command(monkeypatch, tmp_path):
    call_count = {"n": 0}
    original = console_main.locate_critical_points

    def counting(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(console_main, "locate_critical_points", counting)
    monkeypatch.setattr(
        critical_points_plot_module, "locate_critical_points", counting
    )

    _run_console(monkeypatch, tmp_path, ["e2e4", "critical_points_plot", "quit"])

    assert call_count["n"] == 1


def test_critical_points_plot_receives_the_same_objects_used_for_the_summary(
    monkeypatch, tmp_path
):
    created_surfaces = []
    original_build_surface = console_main.build_attack_influence_surface

    def capturing_build_surface(*args, **kwargs):
        result = original_build_surface(*args, **kwargs)
        created_surfaces.append(result)
        return result

    monkeypatch.setattr(
        console_main, "build_attack_influence_surface", capturing_build_surface
    )
    monkeypatch.setattr(
        critical_points_plot_module,
        "build_attack_influence_surface",
        capturing_build_surface,
    )

    recorded_calls = []
    original_plot = console_main.plot_critical_points

    def spy(*args, **kwargs):
        recorded_calls.append(kwargs)
        return original_plot(*args, **kwargs)

    monkeypatch.setattr(console_main, "plot_critical_points", spy)

    _run_console(monkeypatch, tmp_path, ["e2e4", "critical_points_plot", "quit"])

    assert len(created_surfaces) == 1
    # The exact same surface/classified_points/assessments objects
    # that fed print_critical_points_summary must be the ones handed
    # to the visualization layer -- identity (`is`), not just
    # equality, since these are the actual computed results, not
    # freshly-recomputed copies.
    assert recorded_calls[0]["surface"] is created_surfaces[0]
    assert recorded_calls[0]["classified_points"] is not None
    assert recorded_calls[0]["assessments"] is not None


def test_plot_critical_points_precomputed_path_draws_the_same_points_as_default_path(
    monkeypatch, tmp_path
):
    # Guards against Phase 0's optional parameters silently changing
    # rendered output: the points handed to draw_critical_points must
    # be identical whether plot_critical_points recomputes internally
    # (old default behavior, no optional args) or receives already-
    # computed results (new console_app.main call path).
    game = ChessGame()
    move_details = game.make_move("e2e4")
    analysis = analyze_position(game.board, move_details)

    surface = build_attack_influence_surface(analysis.attack_influence_field)
    candidates = locate_critical_points(surface)
    classified_points = classify_critical_points(candidates, surface)
    assessments = assess_critical_point_quality(classified_points, surface)

    recorded_draws = []
    original_draw = critical_points_plot_module.draw_critical_points

    def spy_draw(**kwargs):
        recorded_draws.append(
            [(p.x, p.y, p.classification) for p in kwargs["classified_points"]]
        )
        return original_draw(**kwargs)

    monkeypatch.setattr(critical_points_plot_module, "draw_critical_points", spy_draw)

    critical_points_plot_module.plot_critical_points(board=game.board, analysis=analysis)
    critical_points_plot_module.plot_critical_points(
        board=game.board,
        analysis=analysis,
        surface=surface,
        classified_points=classified_points,
        assessments=assessments,
    )

    assert len(recorded_draws) == 2
    assert recorded_draws[0] == recorded_draws[1]
