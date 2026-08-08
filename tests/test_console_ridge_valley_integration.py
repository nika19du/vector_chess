import dataclasses

import matplotlib

matplotlib.use("Agg")

import console_app.main as console_main
import visualization.ridge_valley_plot as ridge_valley_plot_module
from analysis.attack_influence_surface import build_attack_influence_surface
from analysis.critical_point_quality import assess_critical_point_quality
from analysis.critical_points import classify_critical_points, locate_critical_points
from analysis.ridge_valley import assess_ridge_valley_quality, locate_ridge_valley_chains
from chess_engine.analyzer import analyze_position
from chess_engine.board import ChessGame
from chess_engine.models import RidgeValleyChain, RidgeValleyPoint, RidgeValleyQualityAssessment


def _run_console(monkeypatch, tmp_path, commands: list[str]) -> None:
    monkeypatch.chdir(tmp_path)

    commands_iter = iter(commands)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(commands_iter))

    console_main.main()


def _real_pipeline_ridge_valley(moves: list[str]):
    game = ChessGame()
    move_details = None
    for move in moves:
        move_details = game.make_move(move)
    analysis = analyze_position(game.board, move_details)

    surface = build_attack_influence_surface(analysis.attack_influence_field)
    candidates = locate_critical_points(surface)
    classified = classify_critical_points(candidates, surface)
    critical_point_assessments = assess_critical_point_quality(classified, surface)
    accepted_critical_points = [
        a.point for a in critical_point_assessments if a.is_accepted
    ]

    ridge_chains = locate_ridge_valley_chains(surface, accepted_critical_points, kind="ridge")
    valley_chains = locate_ridge_valley_chains(surface, accepted_critical_points, kind="valley")

    ridge_assessments = assess_ridge_valley_quality(ridge_chains, surface)
    valley_assessments = assess_ridge_valley_quality(valley_chains, surface)

    return ridge_chains, valley_chains, ridge_assessments, valley_assessments


def _make_assessment(kind: str, is_accepted: bool, reason: str | None = None):
    point = RidgeValleyPoint(
        x=4.0, y=4.0, value=0.0, eigenvalue_cross=-1.0, alignment_residual=0.0, kind=kind,
    )
    chain = RidgeValleyChain(
        kind=kind,
        points=[point],
        anchor=None,
        start_status="left_domain",
        end_status="left_domain",
    )
    return RidgeValleyQualityAssessment(
        chain=chain,
        is_accepted=is_accepted,
        rejection_reasons=[] if is_accepted else [reason or "forced for test"],
    )


# ---------------------------------------------------------
# Command before any move
# ---------------------------------------------------------


def test_ridge_valley_plot_before_any_move_prints_guard(monkeypatch, tmp_path, capsys):
    _run_console(monkeypatch, tmp_path, ["ridge_valley_plot", "quit"])

    captured = capsys.readouterr()
    assert "Все още няма позиция" in captured.out


# ---------------------------------------------------------
# Command after one valid move / headless rendering succeeds
# ---------------------------------------------------------


def test_ridge_valley_plot_after_one_move_succeeds_headless(monkeypatch, tmp_path, capsys):
    # Must not raise; matplotlib.use("Agg") makes plt.show() a no-op.
    _run_console(monkeypatch, tmp_path, ["e2e4", "ridge_valley_plot", "quit"])

    captured = capsys.readouterr()
    assert "Ridge / Valley" in captured.out


# ---------------------------------------------------------
# Summary counts match the real pipeline
# ---------------------------------------------------------


def test_summary_counts_match_the_real_pipeline(monkeypatch, tmp_path, capsys):
    ridge_chains, valley_chains, ridge_assessments, valley_assessments = (
        _real_pipeline_ridge_valley(["e2e4"])
    )
    expected_accepted_ridge = sum(1 for a in ridge_assessments if a.is_accepted)
    expected_accepted_valley = sum(1 for a in valley_assessments if a.is_accepted)
    expected_rejected = (
        len(ridge_assessments) + len(valley_assessments)
        - expected_accepted_ridge - expected_accepted_valley
    )

    _run_console(monkeypatch, tmp_path, ["e2e4", "ridge_valley_plot", "quit"])

    captured = capsys.readouterr()
    assert f"Открити ridge вериги: {len(ridge_chains)}" in captured.out
    assert f"Открити valley вериги: {len(valley_chains)}" in captured.out
    assert f"Приети ridge вериги: {expected_accepted_ridge}" in captured.out
    assert f"Приети valley вериги: {expected_accepted_valley}" in captured.out
    assert f"Отхвърлени вериги: {expected_rejected}" in captured.out


# ---------------------------------------------------------
# Zero-chain case
# ---------------------------------------------------------


def test_print_ridge_valley_summary_handles_zero_chains(capsys):
    console_main.print_ridge_valley_summary(
        ridge_chains=[],
        valley_chains=[],
        ridge_assessments=[],
        valley_assessments=[],
    )

    captured = capsys.readouterr()
    assert "Открити ridge вериги: 0" in captured.out
    assert "Открити valley вериги: 0" in captured.out
    assert "Приети ridge вериги: 0" in captured.out
    assert "Приети valley вериги: 0" in captured.out
    assert "Отхвърлени вериги: 0" in captured.out
    assert "Причини за отхвърляне:" not in captured.out  # nothing to group


# ---------------------------------------------------------
# Rejection-reason grouping
# ---------------------------------------------------------


def test_print_ridge_valley_summary_groups_rejection_reasons_by_category(capsys):
    ridge_assessments = [
        _make_assessment("ridge", is_accepted=True),
        _make_assessment(
            "ridge", is_accepted=False,
            reason="chain too short (points=1 < min_points=3)",
        ),
    ]
    valley_assessments = [
        _make_assessment(
            "valley", is_accepted=False,
            reason=(
                "chain spends most of its length near the board boundary "
                "(fraction=0.80 > 0.5, boundary_margin=0.3000)"
            ),
        ),
        _make_assessment(
            "valley", is_accepted=False,
            reason=(
                "average cross-direction curvature too weak "
                "(|eigenvalue_cross| avg=0.010000 < min_curvature_magnitude=0.050000)"
            ),
        ),
    ]

    console_main.print_ridge_valley_summary(
        ridge_chains=[a.chain for a in ridge_assessments],
        valley_chains=[a.chain for a in valley_assessments],
        ridge_assessments=ridge_assessments,
        valley_assessments=valley_assessments,
    )

    captured = capsys.readouterr()
    assert "Приети ridge вериги: 1" in captured.out
    assert "Приети valley вериги: 0" in captured.out
    assert "Отхвърлени вериги: 3" in captured.out
    assert "Причини за отхвърляне:" in captured.out
    assert "веригата е твърде къса: 1" in captured.out
    assert "веригата е предимно близо до ръба на дъската: 1" in captured.out
    assert "цялостно твърде слаба кросова кривина: 1" in captured.out


def test_print_ridge_valley_summary_handles_zero_rejected_chains(capsys):
    ridge_assessments = [_make_assessment("ridge", is_accepted=True)]
    valley_assessments = [_make_assessment("valley", is_accepted=True)]

    console_main.print_ridge_valley_summary(
        ridge_chains=[a.chain for a in ridge_assessments],
        valley_chains=[a.chain for a in valley_assessments],
        ridge_assessments=ridge_assessments,
        valley_assessments=valley_assessments,
    )

    captured = capsys.readouterr()
    assert "Отхвърлени вериги: 0" in captured.out
    assert "Причини за отхвърляне:" not in captured.out


# ---------------------------------------------------------
# Accepted-only rendering by default
# ---------------------------------------------------------


def test_plot_ridge_valley_is_called_without_overriding_the_default(monkeypatch, tmp_path):
    recorded_calls = []
    original_plot_ridge_valley = console_main.plot_ridge_valley

    def spy(*args, **kwargs):
        recorded_calls.append(kwargs)
        return original_plot_ridge_valley(*args, **kwargs)

    monkeypatch.setattr(console_main, "plot_ridge_valley", spy)

    _run_console(monkeypatch, tmp_path, ["e2e4", "ridge_valley_plot", "quit"])

    assert len(recorded_calls) == 1
    # console_app never passes show_only_accepted itself -- it relies
    # entirely on visualization/ridge_valley_plot.py's own default
    # (True), per "do not modify tracing rules, thresholds, or
    # visualization semantics".
    assert "show_only_accepted" not in recorded_calls[0]


# ---------------------------------------------------------
# No duplicate analysis / no mutation of recorded history
# ---------------------------------------------------------


def test_ridge_valley_plot_does_not_trigger_duplicate_analysis(monkeypatch, tmp_path):
    call_count = {"n": 0}
    original_analyze_position = console_main.analyze_position

    def counting_analyze_position(board, move_details):
        call_count["n"] += 1
        return original_analyze_position(board, move_details)

    monkeypatch.setattr(console_main, "analyze_position", counting_analyze_position)

    _run_console(
        monkeypatch,
        tmp_path,
        ["e2e4", "ridge_valley_plot", "ridge_valley_plot", "quit"],
    )

    # One real move was played; ridge_valley_plot was invoked twice for
    # it. analyze_position must have run exactly once.
    assert call_count["n"] == 1


def test_ridge_valley_plot_does_not_change_recorded_history(monkeypatch, tmp_path, capsys):
    _run_console(
        monkeypatch,
        tmp_path,
        ["e2e4", "e7e5", "ridge_valley_plot", "ridge_valley_plot", "history", "quit"],
    )

    captured = capsys.readouterr()
    assert "Ход 1: e2e4" in captured.out
    assert "Ход 2: e7e5" in captured.out
    # Exactly two recorded moves -- invoking ridge_valley_plot twice
    # must not have appended or duplicated history entries.
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
            "audio",
            "ridge_valley_plot",
            "quit",
        ],
    )

    captured = capsys.readouterr()
    assert "История на партията" in captured.out
    assert (tmp_path / "output" / "game_analysis.json").exists()
    assert "Критични точки" in captured.out
    assert "Аудио клипът беше записан" in captured.out
    assert "Ridge / Valley" in captured.out
    assert (tmp_path / "output" / "audio").exists()


# ---------------------------------------------------------
# Phase 0 hardening: pipeline computed exactly once per command
# ---------------------------------------------------------
#
# Mirrors the critical-points version of this section in
# tests/test_console_critical_points_integration.py. Before Phase 0,
# ridge_valley_plot's console handler computed the entire
# surface -> critical points -> quality -> chains -> chain quality
# chain once for the printed summary, then plot_ridge_valley
# recomputed it all again internally. Each function is bound
# independently in console_app.main and visualization.ridge_valley_plot
# (separate `from ... import ...` statements pointing at the same
# underlying function), so both bindings must be patched to observe
# the true total call count across a single command.


def test_build_attack_influence_surface_called_once_per_command(monkeypatch, tmp_path):
    call_count = {"n": 0}
    original = console_main.build_attack_influence_surface

    def counting(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(console_main, "build_attack_influence_surface", counting)
    monkeypatch.setattr(
        ridge_valley_plot_module, "build_attack_influence_surface", counting
    )

    _run_console(monkeypatch, tmp_path, ["e2e4", "ridge_valley_plot", "quit"])

    assert call_count["n"] == 1


def test_locate_critical_points_called_once_per_command(monkeypatch, tmp_path):
    call_count = {"n": 0}
    original = console_main.locate_critical_points

    def counting(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(console_main, "locate_critical_points", counting)
    monkeypatch.setattr(
        ridge_valley_plot_module, "locate_critical_points", counting
    )

    _run_console(monkeypatch, tmp_path, ["e2e4", "ridge_valley_plot", "quit"])

    assert call_count["n"] == 1


def test_locate_ridge_valley_chains_called_once_per_kind_per_command(monkeypatch, tmp_path):
    call_count = {"n": 0}
    original = console_main.locate_ridge_valley_chains

    def counting(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(console_main, "locate_ridge_valley_chains", counting)
    monkeypatch.setattr(
        ridge_valley_plot_module, "locate_ridge_valley_chains", counting
    )

    _run_console(monkeypatch, tmp_path, ["e2e4", "ridge_valley_plot", "quit"])

    # One call for kind="ridge", one for kind="valley" -- exactly 2,
    # not 4 (which is what the pre-Phase-0 duplication would produce).
    assert call_count["n"] == 2


def test_ridge_valley_plot_receives_the_same_objects_used_for_the_summary(
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
        ridge_valley_plot_module,
        "build_attack_influence_surface",
        capturing_build_surface,
    )

    recorded_calls = []
    original_plot = console_main.plot_ridge_valley

    def spy(*args, **kwargs):
        recorded_calls.append(kwargs)
        return original_plot(*args, **kwargs)

    monkeypatch.setattr(console_main, "plot_ridge_valley", spy)

    _run_console(monkeypatch, tmp_path, ["e2e4", "ridge_valley_plot", "quit"])

    assert len(created_surfaces) == 1
    assert recorded_calls[0]["surface"] is created_surfaces[0]
    assert recorded_calls[0]["classified_points"] is not None
    assert recorded_calls[0]["critical_point_assessments"] is not None
    assert recorded_calls[0]["ridge_assessments"] is not None
    assert recorded_calls[0]["valley_assessments"] is not None


def test_plot_ridge_valley_precomputed_path_draws_the_same_chains_as_default_path(
    monkeypatch, tmp_path
):
    # Guards against Phase 0's optional parameters silently changing
    # rendered output: the chains handed to draw_ridge_valley_chains
    # must be identical whether plot_ridge_valley recomputes
    # internally (old default behavior, no optional args) or receives
    # already-computed results (new console_app.main call path).
    game = ChessGame()
    move_details = game.make_move("e2e4")
    analysis = analyze_position(game.board, move_details)

    surface = build_attack_influence_surface(analysis.attack_influence_field)
    candidates = locate_critical_points(surface)
    classified_points = classify_critical_points(candidates, surface)
    critical_point_assessments = assess_critical_point_quality(classified_points, surface)
    accepted_critical_points = [
        a.point for a in critical_point_assessments if a.is_accepted
    ]
    ridge_chains = locate_ridge_valley_chains(surface, accepted_critical_points, kind="ridge")
    valley_chains = locate_ridge_valley_chains(surface, accepted_critical_points, kind="valley")
    ridge_assessments = assess_ridge_valley_quality(ridge_chains, surface)
    valley_assessments = assess_ridge_valley_quality(valley_chains, surface)

    recorded_draws = []
    original_draw = ridge_valley_plot_module.draw_ridge_valley_chains

    def spy_draw(**kwargs):
        recorded_draws.append(
            [
                (
                    assessment.chain.kind,
                    assessment.is_accepted,
                    [(p.x, p.y) for p in assessment.chain.points],
                )
                for assessment in kwargs["assessments"]
            ]
        )
        return original_draw(**kwargs)

    monkeypatch.setattr(
        ridge_valley_plot_module, "draw_ridge_valley_chains", spy_draw
    )

    ridge_valley_plot_module.plot_ridge_valley(board=game.board, analysis=analysis)
    ridge_valley_plot_module.plot_ridge_valley(
        board=game.board,
        analysis=analysis,
        surface=surface,
        classified_points=classified_points,
        critical_point_assessments=critical_point_assessments,
        ridge_assessments=ridge_assessments,
        valley_assessments=valley_assessments,
    )

    assert len(recorded_draws) == 2
    assert recorded_draws[0] == recorded_draws[1]
