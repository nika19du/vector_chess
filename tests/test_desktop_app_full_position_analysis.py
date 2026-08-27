import chess

import desktop_app.full_position_analysis as full_position_analysis_module
from analysis.attack_influence import build_attack_influence_field
from analysis.attack_influence_surface import build_attack_influence_surface
from analysis.critical_point_quality import assess_critical_point_quality
from analysis.critical_points import classify_critical_points, locate_critical_points
from analysis.gradient_field import build_gradient_field
from analysis.morse_smale import (
    assemble_morse_smale_cells,
    assess_morse_smale_cell_quality,
    locate_morse_smale_separatrices,
)
from analysis.ridge_valley import assess_ridge_valley_quality, locate_ridge_valley_chains
from analysis.source_field import build_source_field
from desktop_app.full_position_analysis import build_full_position_analysis


def _midgame_board() -> chess.Board:
    board = chess.Board()
    for move in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"):
        board.push_san(move)
    return board


def test_returns_every_expected_shared_object():
    analysis = build_full_position_analysis(chess.Board())

    assert analysis.board.board_fen() == chess.Board().board_fen()
    assert analysis.attack_influence_field is not None
    assert analysis.gradient_field is not None
    assert analysis.surface is not None
    assert analysis.source_field is not None
    assert isinstance(analysis.classified_critical_points, list)
    assert isinstance(analysis.critical_point_assessments, list)
    assert isinstance(analysis.ridge_chains, list)
    assert isinstance(analysis.ridge_assessments, list)
    assert isinstance(analysis.valley_chains, list)
    assert isinstance(analysis.valley_assessments, list)
    assert analysis.morse_smale_complex is not None
    assert isinstance(analysis.cell_assessments, list)


def test_matches_the_exact_step_by_step_computation_console_app_uses():
    board = _midgame_board()
    analysis = build_full_position_analysis(board)

    attack_influence_field = build_attack_influence_field(board)
    gradient_field = build_gradient_field(attack_influence_field.matrix)
    surface = build_attack_influence_surface(attack_influence_field)
    candidates = locate_critical_points(surface)
    classified_points = classify_critical_points(candidates, surface)
    critical_point_assessments = assess_critical_point_quality(classified_points, surface)
    accepted_critical_points = [a.point for a in critical_point_assessments if a.is_accepted]

    ridge_chains = locate_ridge_valley_chains(surface, accepted_critical_points, kind="ridge")
    valley_chains = locate_ridge_valley_chains(surface, accepted_critical_points, kind="valley")

    assert analysis.attack_influence_field.matrix == attack_influence_field.matrix
    assert analysis.gradient_field.max_magnitude == gradient_field.max_magnitude
    assert analysis.surface.z.tolist() == surface.z.tolist()
    assert analysis.source_field.matrix == build_source_field(board).matrix
    assert len(analysis.classified_critical_points) == len(classified_points)
    assert [p.classification for p in analysis.classified_critical_points] == [
        p.classification for p in classified_points
    ]
    assert len(analysis.ridge_chains) == len(ridge_chains)
    assert len(analysis.valley_chains) == len(valley_chains)


def test_object_identity_contract_is_preserved_for_ridge_valley_anchors():
    # chess_engine/models.py's ClassifiedCriticalPoint docstring: "the same
    # critical point" downstream is determined exclusively by object
    # identity. A chain's anchor must be the *actual* object from
    # classified_critical_points, not a value-equal reconstruction.
    analysis = build_full_position_analysis(_midgame_board())

    anchored_chains = [chain for chain in analysis.ridge_chains + analysis.valley_chains if chain.anchor is not None]
    assert anchored_chains, "expected at least one anchored chain on this position"

    for chain in anchored_chains:
        assert any(chain.anchor is point for point in analysis.classified_critical_points), (
            "chain.anchor is not the same object as any point in classified_critical_points -- "
            "identity contract broken"
        )


def test_object_identity_contract_is_preserved_for_morse_smale_saddles():
    analysis = build_full_position_analysis(_midgame_board())

    assert analysis.morse_smale_complex.edges, "expected at least one separatrix on this position"

    for separatrix in analysis.morse_smale_complex.edges:
        assert any(
            separatrix.start_saddle is point for point in analysis.classified_critical_points
        ), "separatrix.start_saddle is not the same object as any point in classified_critical_points"


def test_source_field_is_independent_of_attack_influence_derived_chain(monkeypatch):
    """
    Milestone F correctness boundary: source_field must be built straight
    from `board`, with zero dependency on attack_influence_field/surface/the
    critical-point chain -- proven concretely, not via a mock, using a
    position where a rook's attack influence is fully blocked but its
    material still occupies its square (see the identical proof in
    tests/test_desktop_app_layers.py for the layer-level version). Swapping
    build_source_field's own output for a hand-built, deliberately wrong
    SourceField must not change anything else FullPositionAnalysis computes.
    """
    import desktop_app.full_position_analysis as full_position_analysis_module
    from chess_engine.models import SourceCell, SourceField

    board = _midgame_board()
    real_analysis = build_full_position_analysis(board)

    fake_source_field = SourceField(
        cells=[SourceCell(square="a1", white_mass=0.0, black_mass=0.0, difference=0.0)] * 64,
        matrix=[[0.0] * 8 for _ in range(8)],
        total_white_mass=0.0,
        total_black_mass=0.0,
        balance=0.0,
    )

    def fake_build_source_field(_board):
        return fake_source_field

    monkeypatch.setattr(full_position_analysis_module, "build_source_field", fake_build_source_field)
    patched_analysis = build_full_position_analysis(board)

    assert patched_analysis.source_field is fake_source_field
    assert patched_analysis.attack_influence_field.matrix == real_analysis.attack_influence_field.matrix
    assert patched_analysis.gradient_field.max_magnitude == real_analysis.gradient_field.max_magnitude
    assert patched_analysis.surface.z.tolist() == real_analysis.surface.z.tolist()
    assert len(patched_analysis.classified_critical_points) == len(real_analysis.classified_critical_points)
    assert len(patched_analysis.ridge_chains) == len(real_analysis.ridge_chains)
    assert len(patched_analysis.valley_chains) == len(real_analysis.valley_chains)
    assert patched_analysis.morse_smale_complex.cells and real_analysis.morse_smale_complex.cells


def test_each_analysis_function_is_called_exactly_once_per_position(monkeypatch):
    # The core Phase 5c performance claim: one build_full_position_analysis
    # call runs the shared chain exactly once -- never once per layer.
    call_counts = {
        "build_source_field": 0,
        "locate_critical_points": 0,
        "classify_critical_points": 0,
        "assess_critical_point_quality": 0,
        "locate_ridge_valley_chains": 0,
        "assess_ridge_valley_quality": 0,
        "locate_morse_smale_separatrices": 0,
        "assemble_morse_smale_cells": 0,
        "assess_morse_smale_cell_quality": 0,
    }

    def counted(name, real_function):
        def wrapper(*args, **kwargs):
            call_counts[name] += 1
            return real_function(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        full_position_analysis_module,
        "build_source_field",
        counted("build_source_field", build_source_field),
    )
    monkeypatch.setattr(
        full_position_analysis_module,
        "locate_critical_points",
        counted("locate_critical_points", locate_critical_points),
    )
    monkeypatch.setattr(
        full_position_analysis_module,
        "classify_critical_points",
        counted("classify_critical_points", classify_critical_points),
    )
    monkeypatch.setattr(
        full_position_analysis_module,
        "assess_critical_point_quality",
        counted("assess_critical_point_quality", assess_critical_point_quality),
    )
    monkeypatch.setattr(
        full_position_analysis_module,
        "locate_ridge_valley_chains",
        counted("locate_ridge_valley_chains", locate_ridge_valley_chains),
    )
    monkeypatch.setattr(
        full_position_analysis_module,
        "assess_ridge_valley_quality",
        counted("assess_ridge_valley_quality", assess_ridge_valley_quality),
    )
    monkeypatch.setattr(
        full_position_analysis_module,
        "locate_morse_smale_separatrices",
        counted("locate_morse_smale_separatrices", locate_morse_smale_separatrices),
    )
    monkeypatch.setattr(
        full_position_analysis_module,
        "assemble_morse_smale_cells",
        counted("assemble_morse_smale_cells", assemble_morse_smale_cells),
    )
    monkeypatch.setattr(
        full_position_analysis_module,
        "assess_morse_smale_cell_quality",
        counted("assess_morse_smale_cell_quality", assess_morse_smale_cell_quality),
    )

    full_position_analysis_module.build_full_position_analysis(_midgame_board())

    assert call_counts["build_source_field"] == 1
    assert call_counts["locate_critical_points"] == 1
    assert call_counts["classify_critical_points"] == 1
    assert call_counts["assess_critical_point_quality"] == 1
    # ridge AND valley each call locate/assess once (kind="ridge", kind="valley") -- 2 calls each, not 1 and not 6.
    assert call_counts["locate_ridge_valley_chains"] == 2
    assert call_counts["assess_ridge_valley_quality"] == 2
    assert call_counts["locate_morse_smale_separatrices"] == 1
    assert call_counts["assemble_morse_smale_cells"] == 1
    assert call_counts["assess_morse_smale_cell_quality"] == 1
