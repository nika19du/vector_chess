import chess
import numpy as np

from desktop_app.correspondence import (
    CellCorrespondence,
    CellMatch,
    ChainCorrespondence,
    ChainMatch,
    CriticalPointCorrespondence,
    CriticalPointMatch,
    UnmatchedCell,
    build_position_correspondence,
)
from desktop_app.layers.attack_influence_layer import build_attack_influence_frame, interpolate_attack_influence_frame
from desktop_app.layers.critical_points_layer import build_critical_points_frame, interpolate_critical_points_frame
from desktop_app.layers.morse_smale_layer import CELL_RESAMPLE_COUNT, interpolate_morse_smale_frame
from desktop_app.layers.ridge_valley_layer import CHAIN_RESAMPLE_COUNT, interpolate_ridge_valley_frame
from tests.conftest import make_cell_geometry, make_critical_point, make_morse_smale_cell, make_ridge_valley_chain, ready_cache_entry

_EMPTY_CHAIN_CORRESPONDENCE = ChainCorrespondence(matched=[], appeared=[], disappeared=[])


# ---------------------------------------------------------
# Attack Influence grid
# ---------------------------------------------------------


def test_attack_influence_t0_matches_previous_position_exactly():
    board_a = chess.Board()
    board_b = chess.Board()
    board_b.push_san("e4")
    entry_a = ready_cache_entry(board_a)
    entry_b = ready_cache_entry(board_b)

    interpolated = interpolate_attack_influence_frame(entry_a, entry_b, 0.0)
    expected = build_attack_influence_frame(entry_a)

    assert np.array_equal(interpolated.colors, expected.colors)


def test_attack_influence_t1_matches_next_position_exactly():
    board_a = chess.Board()
    board_b = chess.Board()
    board_b.push_san("e4")
    entry_a = ready_cache_entry(board_a)
    entry_b = ready_cache_entry(board_b)

    interpolated = interpolate_attack_influence_frame(entry_a, entry_b, 1.0)
    expected = build_attack_influence_frame(entry_b)

    assert np.array_equal(interpolated.colors, expected.colors)


def test_attack_influence_midpoint_differs_from_both_endpoints():
    board_a = chess.Board()
    board_b = chess.Board()
    board_b.push_san("e4")
    entry_a = ready_cache_entry(board_a)
    entry_b = ready_cache_entry(board_b)

    midpoint = interpolate_attack_influence_frame(entry_a, entry_b, 0.5)
    start = build_attack_influence_frame(entry_a)
    end = build_attack_influence_frame(entry_b)

    assert not np.array_equal(midpoint.colors, start.colors)
    assert not np.array_equal(midpoint.colors, end.colors)


# ---------------------------------------------------------
# Critical Points
# ---------------------------------------------------------


def _critical_point_correspondence():
    previous_matched = make_critical_point(1.0, 1.0, "maximum")
    current_matched = make_critical_point(3.0, 1.0, "maximum")
    disappeared = make_critical_point(5.0, 5.0, "saddle")
    appeared = make_critical_point(6.0, 6.0, "minimum")
    return CriticalPointCorrespondence(
        matched=[CriticalPointMatch(previous=previous_matched, current=current_matched)],
        appeared=[appeared],
        disappeared=[disappeared],
    )


def test_critical_points_t0_omits_appeared_and_keeps_disappeared_at_full_alpha():
    correspondence = _critical_point_correspondence()

    frame = interpolate_critical_points_frame(correspondence, 0.0)

    positions = {marker.xy for marker in frame.markers}
    assert (1.0, 1.0) in positions  # matched, at its previous position
    assert (5.0, 5.0) in positions  # disappeared, still fully present
    assert (6.0, 6.0) not in positions  # appeared, not yet born
    for marker in frame.markers:
        assert marker.color[3] == 1.0


def test_critical_points_t1_omits_disappeared_and_keeps_appeared_at_full_alpha():
    correspondence = _critical_point_correspondence()

    frame = interpolate_critical_points_frame(correspondence, 1.0)

    positions = {marker.xy for marker in frame.markers}
    assert (3.0, 1.0) in positions  # matched, at its current position
    assert (6.0, 6.0) in positions  # appeared, fully arrived
    assert (5.0, 5.0) not in positions  # disappeared, fully gone
    for marker in frame.markers:
        assert marker.color[3] == 1.0


def test_critical_points_midpoint_interpolates_matched_position_and_fades_the_rest():
    correspondence = _critical_point_correspondence()

    frame = interpolate_critical_points_frame(correspondence, 0.5)

    by_position = {marker.xy: marker for marker in frame.markers}
    assert (2.0, 1.0) in by_position  # matched, halfway between (1,1) and (3,1)
    assert by_position[(2.0, 1.0)].color[3] == 1.0  # matched points never fade

    disappeared_marker = by_position[(5.0, 5.0)]
    assert disappeared_marker.color[3] == 0.5

    appeared_marker = by_position[(6.0, 6.0)]
    assert appeared_marker.color[3] == 0.5


def test_critical_points_endpoints_match_static_build_on_real_data():
    board_a = chess.Board()
    board_b = chess.Board()
    board_b.push_san("e4")
    entry_a = ready_cache_entry(board_a)
    entry_b = ready_cache_entry(board_b)

    correspondence = build_position_correspondence(entry_a, entry_b)

    t0_positions = {marker.xy for marker in interpolate_critical_points_frame(correspondence.critical_points, 0.0).markers}
    expected_a_positions = {marker.xy for marker in build_critical_points_frame(entry_a).markers}
    assert t0_positions == expected_a_positions

    t1_positions = {marker.xy for marker in interpolate_critical_points_frame(correspondence.critical_points, 1.0).markers}
    expected_b_positions = {marker.xy for marker in build_critical_points_frame(entry_b).markers}
    assert t1_positions == expected_b_positions


# ---------------------------------------------------------
# Ridge / Valley
# ---------------------------------------------------------


def test_ridge_valley_matched_chain_endpoints_are_exact_and_unresampled():
    previous_chain = make_ridge_valley_chain("ridge", [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    current_chain = make_ridge_valley_chain("ridge", [(0.0, 2.0), (1.0, 2.0)])
    ridge_correspondence = ChainCorrespondence(matched=[ChainMatch(previous=previous_chain, current=current_chain)], appeared=[], disappeared=[])

    at_start = interpolate_ridge_valley_frame(ridge_correspondence, _EMPTY_CHAIN_CORRESPONDENCE, 0.0)
    at_end = interpolate_ridge_valley_frame(ridge_correspondence, _EMPTY_CHAIN_CORRESPONDENCE, 1.0)

    assert at_start.chains[0].points_xy == [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    assert at_end.chains[0].points_xy == [(0.0, 2.0), (1.0, 2.0)]


def test_ridge_valley_midpoint_resamples_to_the_configured_count():
    previous_chain = make_ridge_valley_chain("ridge", [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    current_chain = make_ridge_valley_chain("ridge", [(0.0, 2.0), (1.0, 2.0)])
    ridge_correspondence = ChainCorrespondence(matched=[ChainMatch(previous=previous_chain, current=current_chain)], appeared=[], disappeared=[])

    midpoint = interpolate_ridge_valley_frame(ridge_correspondence, _EMPTY_CHAIN_CORRESPONDENCE, 0.5)

    assert len(midpoint.chains[0].points_xy) == CHAIN_RESAMPLE_COUNT


def test_ridge_valley_disappeared_and_appeared_fade():
    disappeared_chain = make_ridge_valley_chain("valley", [(0.0, 0.0), (1.0, 0.0)])
    appeared_chain = make_ridge_valley_chain("valley", [(5.0, 5.0), (6.0, 5.0)])
    valley_correspondence = ChainCorrespondence(matched=[], appeared=[appeared_chain], disappeared=[disappeared_chain])

    at_start = interpolate_ridge_valley_frame(_EMPTY_CHAIN_CORRESPONDENCE, valley_correspondence, 0.0)
    at_end = interpolate_ridge_valley_frame(_EMPTY_CHAIN_CORRESPONDENCE, valley_correspondence, 1.0)

    assert len(at_start.chains) == 1  # only the disappeared chain, at t=0
    assert len(at_end.chains) == 1  # only the appeared chain, at t=1


# ---------------------------------------------------------
# Morse-Smale
# ---------------------------------------------------------


def _cell_correspondence():
    previous_geometry = make_cell_geometry([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)])
    current_geometry = make_cell_geometry([(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0), (4.0, 4.0)])
    disappeared_geometry = make_cell_geometry([(-2.0, -2.0), (-1.0, -2.0), (-1.0, -1.0), (-2.0, -1.0), (-2.0, -2.0)])
    appeared_geometry = make_cell_geometry([(8.0, 8.0), (9.0, 8.0), (9.0, 9.0), (8.0, 9.0), (8.0, 8.0)])

    match = CellMatch(
        previous=make_morse_smale_cell(0, []),
        previous_geometry=previous_geometry,
        current=make_morse_smale_cell(0, []),
        current_geometry=current_geometry,
    )
    return CellCorrespondence(
        matched=[match],
        appeared=[UnmatchedCell(cell=make_morse_smale_cell(1, []), geometry=appeared_geometry)],
        disappeared=[UnmatchedCell(cell=make_morse_smale_cell(2, []), geometry=disappeared_geometry)],
    )


def test_morse_smale_matched_cell_polygon_matches_original_at_endpoints():
    correspondence = _cell_correspondence()
    previous_polygon = correspondence.matched[0].previous_geometry.polygon
    current_polygon = correspondence.matched[0].current_geometry.polygon

    at_start = interpolate_morse_smale_frame(correspondence, 0.0)
    at_end = interpolate_morse_smale_frame(correspondence, 1.0)

    # The matched cell's polygon is always first in the output (matched
    # cells are appended before appeared/disappeared -- see
    # interpolate_morse_smale_frame's own construction order).
    assert at_start.polygons[0] == previous_polygon
    assert at_end.polygons[0] == current_polygon


def test_morse_smale_midpoint_resamples_boundary_to_the_configured_count():
    correspondence = _cell_correspondence()

    midpoint = interpolate_morse_smale_frame(correspondence, 0.5)

    assert len(midpoint.polygons[0]) == CELL_RESAMPLE_COUNT
    assert len(midpoint.centroids) == len(midpoint.polygons)


def test_morse_smale_appeared_and_disappeared_are_included_only_on_their_own_side():
    correspondence = _cell_correspondence()

    at_start = interpolate_morse_smale_frame(correspondence, 0.0)
    at_end = interpolate_morse_smale_frame(correspondence, 1.0)

    # 1 matched (previous shape) + 1 disappeared at t=0; appeared omitted.
    assert len(at_start.polygons) == 2
    # 1 matched (current shape) + 1 appeared at t=1; disappeared omitted.
    assert len(at_end.polygons) == 2


def test_morse_smale_centroid_interpolates_at_midpoint():
    correspondence = _cell_correspondence()
    previous_centroid = correspondence.matched[0].previous_geometry.centroid
    current_centroid = correspondence.matched[0].current_geometry.centroid

    midpoint = interpolate_morse_smale_frame(correspondence, 0.5)

    expected = ((previous_centroid[0] + current_centroid[0]) / 2, (previous_centroid[1] + current_centroid[1]) / 2)
    assert midpoint.centroids[0] == expected
