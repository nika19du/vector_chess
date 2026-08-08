from __future__ import annotations

from dataclasses import dataclass

import chess

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
from chess_engine.models import (
    AttackInfluenceField,
    AttackInfluenceSurface,
    ClassifiedCriticalPoint,
    CriticalPointQualityAssessment,
    GradientField,
    MorseSmaleCellQualityAssessment,
    MorseSmaleComplex,
    RidgeValleyChain,
    RidgeValleyQualityAssessment,
)


@dataclass(frozen=True)
class FullPositionAnalysis:
    """
    Every shared object the six Layer Registry layers (docs/interactive_ui.md
    Part 5) are built from, computed once per position by
    `build_full_position_analysis` -- never once per layer.

    This bundles the *exact*, already-tested call chain
    `console_app/main.py`'s `critical_points_plot`/`ridge_valley_plot`/
    `morse_smale_plot` commands each independently rebuild today (surface ->
    critical points -> quality -> ridge/valley -> Morse-Smale). No formula,
    threshold, or ordering here is new; this is composition only.

    Identity contract (see `chess_engine.models.ClassifiedCriticalPoint`'s
    docstring): `ridge_valley`/`morse_smale` identify "the same critical
    point" by Python object identity, not value equality. Every field below
    is a direct reference from the single call chain that produced it -- none
    is copied, reconstructed, or (de)serialized. Do not introduce
    `copy.deepcopy`/`dataclasses.replace` on any of these fields; doing so
    silently breaks the identity linkage the ridge/valley and Morse-Smale
    results depend on.
    """

    board: chess.Board

    attack_influence_field: AttackInfluenceField
    gradient_field: GradientField
    surface: AttackInfluenceSurface

    classified_critical_points: list[ClassifiedCriticalPoint]
    critical_point_assessments: list[CriticalPointQualityAssessment]

    ridge_chains: list[RidgeValleyChain]
    ridge_assessments: list[RidgeValleyQualityAssessment]
    valley_chains: list[RidgeValleyChain]
    valley_assessments: list[RidgeValleyQualityAssessment]

    morse_smale_complex: MorseSmaleComplex
    cell_assessments: list[MorseSmaleCellQualityAssessment]


def build_full_position_analysis(board: chess.Board) -> FullPositionAnalysis:
    """
    Runs the entire dependency chain for one position, once. Must remain a
    single, uninterrupted call -- see `FullPositionAnalysis`'s identity-
    contract note above for why this can never be split across a
    serialization boundary partway through.
    """

    attack_influence_field = build_attack_influence_field(board)
    gradient_field = build_gradient_field(attack_influence_field.matrix)
    surface = build_attack_influence_surface(attack_influence_field)

    candidates = locate_critical_points(surface)
    classified_critical_points = classify_critical_points(candidates, surface)
    critical_point_assessments = assess_critical_point_quality(
        classified_critical_points, surface
    )
    accepted_critical_points = [
        assessment.point
        for assessment in critical_point_assessments
        if assessment.is_accepted
    ]

    ridge_chains = locate_ridge_valley_chains(surface, accepted_critical_points, kind="ridge")
    valley_chains = locate_ridge_valley_chains(surface, accepted_critical_points, kind="valley")
    ridge_assessments = assess_ridge_valley_quality(ridge_chains, surface)
    valley_assessments = assess_ridge_valley_quality(valley_chains, surface)

    separatrices = locate_morse_smale_separatrices(surface, critical_point_assessments)
    morse_smale_complex = assemble_morse_smale_cells(separatrices)
    cell_assessments = assess_morse_smale_cell_quality(
        morse_smale_complex.cells, morse_smale_complex.topology_issues
    )

    return FullPositionAnalysis(
        board=board,
        attack_influence_field=attack_influence_field,
        gradient_field=gradient_field,
        surface=surface,
        classified_critical_points=classified_critical_points,
        critical_point_assessments=critical_point_assessments,
        ridge_chains=ridge_chains,
        ridge_assessments=ridge_assessments,
        valley_chains=valley_chains,
        valley_assessments=valley_assessments,
        morse_smale_complex=morse_smale_complex,
        cell_assessments=cell_assessments,
    )
