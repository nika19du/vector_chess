import chess

from desktop_app.correspondence import (
    CorrespondenceCache,
    _match_cells,
    _match_chains,
    _match_critical_points,
    build_position_correspondence,
)
from desktop_app.layers._arc_length import resample_by_arc_length
from tests.conftest import make_cell_geometry, make_critical_point, make_morse_smale_cell, make_ridge_valley_chain, ready_cache_entry

MAX_DISTANCE = 2.5


# ---------------------------------------------------------
# critical point matching
# ---------------------------------------------------------


def test_deterministic_correspondence_across_repeated_runs():
    previous = [make_critical_point(1.0, 1.0, "maximum"), make_critical_point(5.0, 5.0, "minimum")]
    current = [make_critical_point(1.2, 1.1, "maximum"), make_critical_point(5.1, 4.9, "minimum")]

    first = _match_critical_points(previous, current, MAX_DISTANCE)
    second = _match_critical_points(previous, current, MAX_DISTANCE)

    assert [(m.previous, m.current) for m in first.matched] == [(m.previous, m.current) for m in second.matched]
    assert first.appeared == second.appeared
    assert first.disappeared == second.disappeared


def test_matched_pairs_always_share_classification():
    previous = [make_critical_point(1.0, 1.0, "maximum"), make_critical_point(2.0, 2.0, "saddle")]
    current = [make_critical_point(1.1, 1.1, "maximum"), make_critical_point(2.1, 2.1, "saddle")]

    correspondence = _match_critical_points(previous, current, MAX_DISTANCE)

    assert len(correspondence.matched) == 2
    for match in correspondence.matched:
        assert match.previous.classification == match.current.classification


def test_no_cross_classification_match_even_at_zero_distance():
    # Same exact (x, y), different classification -- must never match.
    previous = [make_critical_point(3.0, 3.0, "maximum")]
    current = [make_critical_point(3.0, 3.0, "minimum")]

    correspondence = _match_critical_points(previous, current, MAX_DISTANCE)

    assert correspondence.matched == []
    assert correspondence.disappeared == previous
    assert correspondence.appeared == current


def test_births_and_deaths_beyond_the_match_threshold():
    previous = [make_critical_point(0.0, 0.0, "maximum")]
    current = [make_critical_point(0.0, 0.0, "maximum"), make_critical_point(7.9, 7.9, "minimum")]

    correspondence = _match_critical_points(previous, current, MAX_DISTANCE)

    assert len(correspondence.matched) == 1
    assert correspondence.matched[0].previous is previous[0]
    assert correspondence.matched[0].current is current[0]
    assert correspondence.disappeared == []
    assert correspondence.appeared == [current[1]]  # too far to match -- a birth, not a move


def test_ambiguous_match_prefers_the_nearer_point_deterministically():
    # Two previous maxima, one current maximum -- only the nearer one may
    # claim it; the other becomes "disappeared", never a duplicate match.
    near = make_critical_point(1.0, 1.0, "maximum")
    far = make_critical_point(1.4, 1.4, "maximum")
    current_point = make_critical_point(1.0, 1.0, "maximum")

    correspondence = _match_critical_points([near, far], [current_point], MAX_DISTANCE)

    assert len(correspondence.matched) == 1
    assert correspondence.matched[0].previous is near
    assert correspondence.disappeared == [far]
    assert correspondence.appeared == []


def test_exact_distance_ties_break_by_input_order():
    # Two previous points exactly equidistant from one current point --
    # deterministic tie-break picks the earlier (lower-index) previous point.
    a = make_critical_point(0.0, 0.0, "maximum")
    b = make_critical_point(2.0, 0.0, "maximum")
    current_point = make_critical_point(1.0, 0.0, "maximum")

    correspondence = _match_critical_points([a, b], [current_point], MAX_DISTANCE)

    assert correspondence.matched[0].previous is a
    assert correspondence.disappeared == [b]


# ---------------------------------------------------------
# arc-length resampling
# ---------------------------------------------------------


def test_resample_returns_requested_sample_count():
    points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]

    samples = resample_by_arc_length(points, 10)

    assert len(samples) == 10


def test_resample_preserves_endpoints():
    points = [(0.0, 0.0), (1.0, 2.0), (3.0, 3.0), (5.0, 0.0)]

    samples = resample_by_arc_length(points, 8)

    assert samples[0] == points[0]
    assert samples[-1] == points[-1]


def test_resample_is_evenly_spaced_for_a_straight_line():
    points = [(0.0, 0.0), (10.0, 0.0)]

    samples = resample_by_arc_length(points, 5)

    xs = [x for x, _ in samples]
    assert xs == [0.0, 2.5, 5.0, 7.5, 10.0]


def test_resample_single_point_repeats_it():
    samples = resample_by_arc_length([(3.0, 4.0)], 6)

    assert samples == [(3.0, 4.0)] * 6


def test_resample_closed_polygon_stays_closed():
    # A closed loop (first == last), same convention MorseSmaleCellGeometry
    # uses -- resampling should reproduce that closure.
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]

    samples = resample_by_arc_length(square, 12)

    assert samples[0] == samples[-1] == (0.0, 0.0)


# ---------------------------------------------------------
# chain matching
# ---------------------------------------------------------


def test_chain_matching_prefers_anchor_identity_over_centroid_proximity():
    previous_anchor = make_critical_point(1.0, 1.0, "maximum")
    current_anchor = make_critical_point(1.1, 1.1, "maximum")
    point_matches = _match_critical_points([previous_anchor], [current_anchor], MAX_DISTANCE).matched

    previous_chain = make_ridge_valley_chain("ridge", [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)], anchor=previous_anchor)
    # A decoy current chain that's centroid-closer to the previous chain but
    # anchored at an unrelated point -- must lose to the anchor-based match.
    decoy_chain = make_ridge_valley_chain("ridge", [(0.0, 1.05), (1.0, 1.05), (2.0, 1.05)], anchor=None)
    correct_chain = make_ridge_valley_chain("ridge", [(5.0, 5.0), (6.0, 5.0)], anchor=current_anchor)

    correspondence = _match_chains([previous_chain], [decoy_chain, correct_chain], point_matches, max_centroid_distance=10.0)

    assert len(correspondence.matched) == 1
    assert correspondence.matched[0].current is correct_chain
    assert correspondence.appeared == [decoy_chain]


def test_chain_matching_falls_back_to_centroid_proximity_without_anchor_carryover():
    previous_chain = make_ridge_valley_chain("valley", [(0.0, 0.0), (1.0, 0.0)], anchor=None)
    nearby_chain = make_ridge_valley_chain("valley", [(0.1, 0.1), (1.1, 0.1)], anchor=None)
    far_chain = make_ridge_valley_chain("valley", [(7.0, 7.0), (7.5, 7.0)], anchor=None)

    correspondence = _match_chains([previous_chain], [nearby_chain, far_chain], [], max_centroid_distance=1.0)

    assert len(correspondence.matched) == 1
    assert correspondence.matched[0].current is nearby_chain
    assert correspondence.appeared == [far_chain]


def test_chain_matching_never_crosses_kind():
    ridge = make_ridge_valley_chain("ridge", [(0.0, 0.0), (1.0, 0.0)], anchor=None)
    valley_same_spot = make_ridge_valley_chain("valley", [(0.0, 0.0), (1.0, 0.0)], anchor=None)

    correspondence = _match_chains([ridge], [valley_same_spot], [], max_centroid_distance=10.0)

    assert correspondence.matched == []
    assert correspondence.disappeared == [ridge]
    assert correspondence.appeared == [valley_same_spot]


# ---------------------------------------------------------
# cell matching
# ---------------------------------------------------------


def test_cell_matching_by_boundary_critical_point_overlap():
    previous_a, previous_b, previous_c = (
        make_critical_point(0.0, 0.0, "saddle"),
        make_critical_point(2.0, 0.0, "maximum"),
        make_critical_point(0.0, 2.0, "minimum"),
    )
    current_a, current_b, current_c = (
        make_critical_point(0.1, 0.1, "saddle"),
        make_critical_point(2.1, 0.1, "maximum"),
        make_critical_point(0.1, 2.1, "minimum"),
    )
    point_matches = _match_critical_points(
        [previous_a, previous_b, previous_c], [current_a, current_b, current_c], MAX_DISTANCE
    ).matched

    previous_cell = make_morse_smale_cell(0, [previous_a, previous_b, previous_c])
    previous_geometry = make_cell_geometry([(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.0, 0.0)])
    current_cell = make_morse_smale_cell(0, [current_a, current_b, current_c])  # same cell_id -- must not matter
    current_geometry = make_cell_geometry([(0.1, 0.1), (2.1, 0.1), (0.1, 2.1), (0.1, 0.1)])

    correspondence = _match_cells(
        [(previous_cell, previous_geometry)], [(current_cell, current_geometry)], point_matches, min_overlap_fraction=0.5
    )

    assert len(correspondence.matched) == 1
    assert correspondence.matched[0].current is current_cell
    assert correspondence.disappeared == []
    assert correspondence.appeared == []


def test_cell_matching_treats_low_overlap_as_birth_and_death():
    previous_point = make_critical_point(0.0, 0.0, "saddle")
    current_point = make_critical_point(9.0, 9.0, "saddle")  # too far to match at all

    previous_cell = make_morse_smale_cell(0, [previous_point])
    previous_geometry = make_cell_geometry([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)])
    current_cell = make_morse_smale_cell(0, [current_point])
    current_geometry = make_cell_geometry([(9.0, 9.0), (10.0, 9.0), (9.0, 10.0), (9.0, 9.0)])

    correspondence = _match_cells(
        [(previous_cell, previous_geometry)], [(current_cell, current_geometry)], [], min_overlap_fraction=0.5
    )

    assert correspondence.matched == []
    assert [unmatched.cell for unmatched in correspondence.disappeared] == [previous_cell]
    assert [unmatched.cell for unmatched in correspondence.appeared] == [current_cell]


# ---------------------------------------------------------
# memoization
# ---------------------------------------------------------


def test_correspondence_cache_memoizes_by_fen_pair():
    call_count = {"n": 0}

    def counting_builder(entry_a, entry_b):
        call_count["n"] += 1
        return build_position_correspondence(entry_a, entry_b)

    board_a = chess.Board()
    board_b = chess.Board()
    board_b.push_san("e4")
    entry_a = ready_cache_entry(board_a)
    entry_b = ready_cache_entry(board_b)

    cache = CorrespondenceCache(builder=counting_builder)

    first = cache.get_or_compute(board_a.board_fen(), entry_a, board_b.board_fen(), entry_b)
    second = cache.get_or_compute(board_a.board_fen(), entry_a, board_b.board_fen(), entry_b)

    assert first is second
    assert call_count["n"] == 1


def test_correspondence_cache_distinguishes_direction():
    call_count = {"n": 0}

    def counting_builder(entry_a, entry_b):
        call_count["n"] += 1
        return build_position_correspondence(entry_a, entry_b)

    board_a = chess.Board()
    board_b = chess.Board()
    board_b.push_san("e4")
    entry_a = ready_cache_entry(board_a)
    entry_b = ready_cache_entry(board_b)

    cache = CorrespondenceCache(builder=counting_builder)

    cache.get_or_compute(board_a.board_fen(), entry_a, board_b.board_fen(), entry_b)
    cache.get_or_compute(board_b.board_fen(), entry_b, board_a.board_fen(), entry_a)

    assert call_count["n"] == 2


# ---------------------------------------------------------
# end-to-end smoke test on real analysis data
# ---------------------------------------------------------


def test_build_position_correspondence_runs_end_to_end_on_real_positions():
    board_a = chess.Board()
    board_b = chess.Board()
    for move in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"):
        board_b.push_san(move)

    correspondence = build_position_correspondence(ready_cache_entry(board_a), ready_cache_entry(board_b))

    assert correspondence.critical_points is not None
    assert correspondence.ridge_chains is not None
    assert correspondence.valley_chains is not None
    assert correspondence.morse_smale_cells is not None
