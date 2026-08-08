from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from analysis.morse_smale import compute_cell_geometry
from chess_engine.models import (
    ClassifiedCriticalPoint,
    MorseSmaleCell,
    MorseSmaleCellGeometry,
    RidgeValleyChain,
)
from desktop_app.position_cache import CacheEntry
from visualization.morse_smale_plot import MIN_POINTS_TO_DRAW as MIN_CELL_POINTS_TO_DRAW
from visualization.ridge_valley_plot import MIN_POINTS_TO_DRAW as MIN_CHAIN_POINTS_TO_DRAW

# Rendering constants (docs/interactive_ui.md Part 4.5: "The threshold is a
# rendering constant, not a mathematical one"), not re-derivations of
# anything in analysis/ or docs/mathematics.md.
CRITICAL_POINT_MATCH_MAX_DISTANCE = 2.5  # plot-space units (board extent 8x8)
CHAIN_MATCH_MAX_CENTROID_DISTANCE = 3.0  # plot-space units, fallback matching only
CELL_MATCH_MIN_OVERLAP_FRACTION = 0.5  # fraction of mapped boundary critical points shared


@dataclass(frozen=True)
class CriticalPointMatch:
    previous: ClassifiedCriticalPoint
    current: ClassifiedCriticalPoint


@dataclass(frozen=True)
class CriticalPointCorrespondence:
    matched: list[CriticalPointMatch]
    appeared: list[ClassifiedCriticalPoint]  # only in the current position
    disappeared: list[ClassifiedCriticalPoint]  # only in the previous position


@dataclass(frozen=True)
class ChainMatch:
    previous: RidgeValleyChain
    current: RidgeValleyChain


@dataclass(frozen=True)
class ChainCorrespondence:
    matched: list[ChainMatch]
    appeared: list[RidgeValleyChain]
    disappeared: list[RidgeValleyChain]


@dataclass(frozen=True)
class CellMatch:
    previous: MorseSmaleCell
    previous_geometry: MorseSmaleCellGeometry
    current: MorseSmaleCell
    current_geometry: MorseSmaleCellGeometry


@dataclass(frozen=True)
class UnmatchedCell:
    cell: MorseSmaleCell
    geometry: MorseSmaleCellGeometry


@dataclass(frozen=True)
class CellCorrespondence:
    matched: list[CellMatch]
    appeared: list[UnmatchedCell]
    disappeared: list[UnmatchedCell]


@dataclass(frozen=True)
class PositionCorrespondence:
    """
    Everything `desktop_app.transition_controller` needs to animate between
    two already-cached positions (docs/interactive_ui.md Part 4.5). Built
    once per (from_fen, to_fen) pair and memoized by `CorrespondenceCache`
    below -- never recomputed per animation frame.

    A "match" here is a rendering/UI judgment about which drawn object to
    morph into which, never stored back into `chess_engine.models` or any
    `analysis/*` output, and never a new mathematical claim about the
    position -- exactly Part 4.5's own framing.
    """

    critical_points: CriticalPointCorrespondence
    ridge_chains: ChainCorrespondence
    valley_chains: ChainCorrespondence
    morse_smale_cells: CellCorrespondence


def _accepted_critical_points(analysis) -> list[ClassifiedCriticalPoint]:
    return [assessment.point for assessment in analysis.critical_point_assessments if assessment.is_accepted]


def _drawable_chains(assessments) -> list[RidgeValleyChain]:
    return [
        assessment.chain
        for assessment in assessments
        if assessment.is_accepted and len(assessment.chain.points) >= MIN_CHAIN_POINTS_TO_DRAW
    ]


def _drawable_cells_with_geometry(analysis) -> list[tuple[MorseSmaleCell, MorseSmaleCellGeometry]]:
    assessment_by_cell_id = {assessment.cell.cell_id: assessment for assessment in analysis.cell_assessments}

    result: list[tuple[MorseSmaleCell, MorseSmaleCellGeometry]] = []
    for cell in analysis.morse_smale_complex.cells:
        if not cell.is_closed:
            continue
        assessment = assessment_by_cell_id.get(cell.cell_id)
        if assessment is None or not assessment.is_accepted:
            continue
        geometry = compute_cell_geometry(cell)
        if geometry is None or len(geometry.polygon) < MIN_CELL_POINTS_TO_DRAW:
            continue
        result.append((cell, geometry))
    return result


def _match_critical_points(
    previous_points: list[ClassifiedCriticalPoint],
    current_points: list[ClassifiedCriticalPoint],
    max_distance: float,
) -> CriticalPointCorrespondence:
    """
    Greedy nearest-neighbor, constrained to the same `classification`
    (docs/interactive_ui.md Part 4.5). Candidate pairs are sorted by
    `(distance, previous_index, current_index)` and consumed in that order,
    so the result is deterministic for a given pair of input lists
    regardless of dict/set iteration order anywhere else in the pipeline.
    """
    candidates: list[tuple[float, int, int]] = []
    for i, previous in enumerate(previous_points):
        for j, current in enumerate(current_points):
            if previous.classification != current.classification:
                continue
            distance = math.hypot(previous.x - current.x, previous.y - current.y)
            if distance <= max_distance:
                candidates.append((distance, i, j))
    candidates.sort()

    matched_previous: set[int] = set()
    matched_current: set[int] = set()
    matched: list[CriticalPointMatch] = []
    for _distance, i, j in candidates:
        if i in matched_previous or j in matched_current:
            continue
        matched_previous.add(i)
        matched_current.add(j)
        matched.append(CriticalPointMatch(previous=previous_points[i], current=current_points[j]))

    disappeared = [point for i, point in enumerate(previous_points) if i not in matched_previous]
    appeared = [point for j, point in enumerate(current_points) if j not in matched_current]
    return CriticalPointCorrespondence(matched=matched, appeared=appeared, disappeared=disappeared)


def _find_matched_current(matches: list[CriticalPointMatch], previous_point) -> ClassifiedCriticalPoint | None:
    for match in matches:
        if match.previous is previous_point:
            return match.current
    return None


def _chain_centroid(chain: RidgeValleyChain) -> tuple[float, float]:
    xs = [point.x for point in chain.points]
    ys = [point.y for point in chain.points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _match_chains(
    previous_chains: list[RidgeValleyChain],
    current_chains: list[RidgeValleyChain],
    critical_point_matches: list[CriticalPointMatch],
    max_centroid_distance: float,
) -> ChainCorrespondence:
    """
    Anchor-based first (docs/interactive_ui.md Part 4.5): a chain's `anchor`
    is the *same object* (by identity, per `ClassifiedCriticalPoint`'s own
    identity contract in chess_engine/models.py) as one of the accepted
    critical points for that position, so the already-computed critical
    point correspondence tells us which current chain a previous chain's
    anchor carries over to, if any. Falls back to centroid proximity, same
    `kind` only, for whatever's left.
    """
    used_previous: set[int] = set()
    used_current: set[int] = set()
    matched: list[ChainMatch] = []

    for i, previous_chain in enumerate(previous_chains):
        if previous_chain.anchor is None:
            continue
        mapped_anchor = _find_matched_current(critical_point_matches, previous_chain.anchor)
        if mapped_anchor is None:
            continue
        for j, current_chain in enumerate(current_chains):
            if j in used_current or current_chain.kind != previous_chain.kind:
                continue
            if current_chain.anchor is mapped_anchor:
                matched.append(ChainMatch(previous=previous_chain, current=current_chain))
                used_previous.add(i)
                used_current.add(j)
                break

    candidates: list[tuple[float, int, int]] = []
    for i, previous_chain in enumerate(previous_chains):
        if i in used_previous:
            continue
        for j, current_chain in enumerate(current_chains):
            if j in used_current or current_chain.kind != previous_chain.kind:
                continue
            distance = math.hypot(
                *(a - b for a, b in zip(_chain_centroid(previous_chain), _chain_centroid(current_chain)))
            )
            if distance <= max_centroid_distance:
                candidates.append((distance, i, j))
    candidates.sort()

    for _distance, i, j in candidates:
        if i in used_previous or j in used_current:
            continue
        used_previous.add(i)
        used_current.add(j)
        matched.append(ChainMatch(previous=previous_chains[i], current=current_chains[j]))

    disappeared = [chain for i, chain in enumerate(previous_chains) if i not in used_previous]
    appeared = [chain for j, chain in enumerate(current_chains) if j not in used_current]
    return ChainCorrespondence(matched=matched, appeared=appeared, disappeared=disappeared)


def _mapped_boundary_point_ids(cell: MorseSmaleCell, critical_point_matches: list[CriticalPointMatch]) -> set[int]:
    mapped_ids: set[int] = set()
    for point in cell.boundary_critical_points:
        current_point = _find_matched_current(critical_point_matches, point)
        if current_point is not None:
            mapped_ids.add(id(current_point))
    return mapped_ids


def _match_cells(
    previous_cells: list[tuple[MorseSmaleCell, MorseSmaleCellGeometry]],
    current_cells: list[tuple[MorseSmaleCell, MorseSmaleCellGeometry]],
    critical_point_matches: list[CriticalPointMatch],
    min_overlap_fraction: float,
) -> CellCorrespondence:
    """
    Matched by their bounding critical points' matches (docs/interactive_ui.md
    Part 4.5), never by `cell_id` -- `analysis.morse_smale.
    assemble_morse_smale_cells` assigns `cell_id` independently per position
    (`cell_id=len(cells)` at discovery time), so it carries no cross-position
    meaning. `ClassifiedCriticalPoint` is not hashable (a non-frozen
    dataclass), so overlap is computed over `id()` -- safe here since every
    object involved is kept alive for this function's entire, synchronous
    call by the `FullPositionAnalysis` each `CacheEntry` still references.
    """
    candidates: list[tuple[float, int, int]] = []
    for i, (previous_cell, _previous_geometry) in enumerate(previous_cells):
        mapped_ids = _mapped_boundary_point_ids(previous_cell, critical_point_matches)
        if not mapped_ids:
            continue
        for j, (current_cell, _current_geometry) in enumerate(current_cells):
            current_ids = {id(point) for point in current_cell.boundary_critical_points}
            overlap = len(mapped_ids & current_ids)
            denominator = max(len(mapped_ids), len(current_ids))
            score = overlap / denominator if denominator else 0.0
            if score >= min_overlap_fraction:
                candidates.append((-score, i, j))  # negative: highest score sorts first
    candidates.sort()

    used_previous: set[int] = set()
    used_current: set[int] = set()
    matched: list[CellMatch] = []
    for _negative_score, i, j in candidates:
        if i in used_previous or j in used_current:
            continue
        used_previous.add(i)
        used_current.add(j)
        previous_cell, previous_geometry = previous_cells[i]
        current_cell, current_geometry = current_cells[j]
        matched.append(
            CellMatch(
                previous=previous_cell,
                previous_geometry=previous_geometry,
                current=current_cell,
                current_geometry=current_geometry,
            )
        )

    disappeared = [
        UnmatchedCell(cell=cell, geometry=geometry)
        for i, (cell, geometry) in enumerate(previous_cells)
        if i not in used_previous
    ]
    appeared = [
        UnmatchedCell(cell=cell, geometry=geometry)
        for j, (cell, geometry) in enumerate(current_cells)
        if j not in used_current
    ]
    return CellCorrespondence(matched=matched, appeared=appeared, disappeared=disappeared)


def build_position_correspondence(entry_a: CacheEntry, entry_b: CacheEntry) -> PositionCorrespondence:
    """
    Builds the full correspondence between two already-`READY` cache
    entries. Never touches `analysis/*` -- every input here is already
    computed and cached; this only compares/matches already-existing
    objects (plus the one-time `compute_cell_geometry` derived-geometry
    call, itself a presentation-layer helper already invoked once per
    static render, see `desktop_app/layers/morse_smale_layer.py`).
    """
    if entry_a.analysis is None or entry_b.analysis is None:
        raise ValueError("both cache entries must be READY to build a correspondence")

    previous_points = _accepted_critical_points(entry_a.analysis)
    current_points = _accepted_critical_points(entry_b.analysis)
    critical_points = _match_critical_points(previous_points, current_points, CRITICAL_POINT_MATCH_MAX_DISTANCE)

    ridge_chains = _match_chains(
        _drawable_chains(entry_a.analysis.ridge_assessments),
        _drawable_chains(entry_b.analysis.ridge_assessments),
        critical_points.matched,
        CHAIN_MATCH_MAX_CENTROID_DISTANCE,
    )
    valley_chains = _match_chains(
        _drawable_chains(entry_a.analysis.valley_assessments),
        _drawable_chains(entry_b.analysis.valley_assessments),
        critical_points.matched,
        CHAIN_MATCH_MAX_CENTROID_DISTANCE,
    )

    morse_smale_cells = _match_cells(
        _drawable_cells_with_geometry(entry_a.analysis),
        _drawable_cells_with_geometry(entry_b.analysis),
        critical_points.matched,
        CELL_MATCH_MIN_OVERLAP_FRACTION,
    )

    return PositionCorrespondence(
        critical_points=critical_points,
        ridge_chains=ridge_chains,
        valley_chains=valley_chains,
        morse_smale_cells=morse_smale_cells,
    )


class CorrespondenceCache:
    """
    Memoizes `PositionCorrespondence` by `(from_fen, to_fen)`
    (docs/interactive_ui.md Part 4.5: "computed once, when the pair is
    first needed"). Synchronous, unlike `PositionCache` -- matching already-
    cached, already-small point/chain/cell lists is cheap geometry, not the
    `scipy`-heavy work Part 4.3 reserves a background worker for.

    `builder` is injectable, mirroring `PositionCache`'s own `_builder`
    swap-for-testing convention.
    """

    def __init__(self, builder: Callable[[CacheEntry, CacheEntry], PositionCorrespondence] = build_position_correspondence) -> None:
        self._builder = builder
        self._entries: dict[tuple[str, str], PositionCorrespondence] = {}

    def get_or_compute(self, from_fen: str, entry_a: CacheEntry, to_fen: str, entry_b: CacheEntry) -> PositionCorrespondence:
        key = (from_fen, to_fen)
        if key not in self._entries:
            self._entries[key] = self._builder(entry_a, entry_b)
        return self._entries[key]
