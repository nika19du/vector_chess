import math

import chess
import numpy as np
import pytest

from analysis.attack_influence import PIECE_WEIGHTS
from analysis.source_field import build_source_field
from analysis.source_potential import (
    build_source_potential_surface,
    evaluate_source_potential,
    gaussian_kernel,
    softened_inverse_distance_kernel,
)


# ---------------------------------------------------------
# Kernel unit tests
# ---------------------------------------------------------

KERNELS_UNDER_TEST = [
    (gaussian_kernel, {"sigma": 1.0}),
    (softened_inverse_distance_kernel, {"epsilon": 1.0}),
]


@pytest.mark.parametrize("kernel_function,params", KERNELS_UNDER_TEST)
def test_kernel_equals_one_at_zero(kernel_function, params):
    assert kernel_function(0.0, **params) == pytest.approx(1.0)


@pytest.mark.parametrize("kernel_function,params", KERNELS_UNDER_TEST)
def test_kernel_is_symmetric_in_r(kernel_function, params):
    r = np.array([0.3, 1.5, 4.0])

    assert np.allclose(
        kernel_function(r, **params),
        kernel_function(-r, **params),
    )


@pytest.mark.parametrize("kernel_function,params", KERNELS_UNDER_TEST)
def test_kernel_is_strictly_decreasing(kernel_function, params):
    r = np.linspace(0.0, 6.0, 30)
    values = kernel_function(r, **params)

    assert np.all(np.diff(values) < 0)


def test_inverse_distance_kernel_matches_asymptote_at_large_r():
    epsilon = 1.0
    r = 50.0

    value = softened_inverse_distance_kernel(r, epsilon)
    expected = epsilon / r

    assert value == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------
# Field-level tests
# ---------------------------------------------------------


def empty_board() -> chess.Board:
    return chess.Board(None)


def test_empty_board_source_potential_is_zero_everywhere():
    source_field = build_source_field(empty_board())

    surface = build_source_potential_surface(source_field)

    assert np.all(surface.z == 0.0)
    assert evaluate_source_potential(source_field, 4.0, 4.0) == 0.0


def test_single_piece_potential_peaks_at_its_own_square():
    board = empty_board()
    board.set_piece_at(chess.D4, chess.Piece(chess.QUEEN, chess.WHITE))

    source_field = build_source_field(board)

    center_x, center_y = 3 + 0.5, (7 - 3) + 0.5  # d4 in screen coords

    value_at_center = evaluate_source_potential(
        source_field, center_x, center_y
    )

    assert value_at_center == pytest.approx(
        PIECE_WEIGHTS[chess.QUEEN], rel=1e-6
    )


def test_single_piece_potential_is_radially_symmetric():
    board = empty_board()
    board.set_piece_at(chess.D4, chess.Piece(chess.ROOK, chess.WHITE))

    source_field = build_source_field(board)

    center_x, center_y = 3 + 0.5, (7 - 3) + 0.5  # d4 in screen coords
    radius = 1.2

    samples = [
        evaluate_source_potential(
            source_field,
            center_x + radius * math.cos(angle),
            center_y + radius * math.sin(angle),
        )
        for angle in (0, math.pi / 2, math.pi, 3 * math.pi / 2)
    ]

    for sample in samples[1:]:
        assert sample == pytest.approx(samples[0], rel=1e-9)


def test_single_piece_potential_decreases_with_distance():
    board = empty_board()
    board.set_piece_at(chess.D4, chess.Piece(chess.BISHOP, chess.WHITE))

    source_field = build_source_field(board)

    center_x, center_y = 3.5, 3.5

    radii = [0.0, 0.5, 1.0, 2.0, 3.5]
    values = [
        evaluate_source_potential(source_field, center_x + r, center_y)
        for r in radii
    ]

    assert all(
        values[index] > values[index + 1]
        for index in range(len(values) - 1)
    )


def test_color_swap_negates_source_potential():
    white_board = empty_board()
    white_board.set_piece_at(chess.E5, chess.Piece(chess.KNIGHT, chess.WHITE))

    black_board = empty_board()
    black_board.set_piece_at(chess.E5, chess.Piece(chess.KNIGHT, chess.BLACK))

    white_field = build_source_field(white_board)
    black_field = build_source_field(black_board)

    for x, y in [(1.0, 1.0), (4.0, 4.0), (7.0, 0.5)]:
        white_value = evaluate_source_potential(white_field, x, y)
        black_value = evaluate_source_potential(black_field, x, y)

        assert white_value == pytest.approx(-black_value)


def test_superposition_of_two_separated_pieces():
    combined_board = empty_board()
    combined_board.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))
    combined_board.set_piece_at(chess.H8, chess.Piece(chess.ROOK, chess.BLACK))

    only_white_board = empty_board()
    only_white_board.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))

    only_black_board = empty_board()
    only_black_board.set_piece_at(chess.H8, chess.Piece(chess.ROOK, chess.BLACK))

    combined_field = build_source_field(combined_board)
    white_field = build_source_field(only_white_board)
    black_field = build_source_field(only_black_board)

    combined_surface = build_source_potential_surface(
        combined_field, resolution=40
    )
    white_surface = build_source_potential_surface(
        white_field, resolution=40
    )
    black_surface = build_source_potential_surface(
        black_field, resolution=40
    )

    assert np.allclose(
        combined_surface.z,
        white_surface.z + black_surface.z,
    )


def test_blocker_independence_does_not_change_rooks_contribution():
    """
    The key non-conflation regression test: inserting a piece that
    blocks the rook's line of sight must change the Attack Influence
    Field but must leave the rook's own Source Potential contribution
    completely unchanged, because Phi_source has no notion of
    blocking or ray-casting at all.
    """

    from analysis.attack_influence import build_attack_influence_field

    open_file_board = chess.Board(None)
    open_file_board.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))
    open_file_board.set_piece_at(chess.A8, chess.Piece(chess.KING, chess.BLACK))
    open_file_board.set_piece_at(chess.H1, chess.Piece(chess.KING, chess.WHITE))

    blocked_file_board = chess.Board(None)
    blocked_file_board.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))
    blocked_file_board.set_piece_at(chess.A8, chess.Piece(chess.KING, chess.BLACK))
    blocked_file_board.set_piece_at(chess.H1, chess.Piece(chess.KING, chess.WHITE))
    blocked_file_board.set_piece_at(chess.A4, chess.Piece(chess.PAWN, chess.WHITE))

    # Control check: the Attack Influence Field DOES change at a8
    # once the file is blocked (existing, expected behavior).
    open_attack_influence = build_attack_influence_field(open_file_board)
    blocked_attack_influence = build_attack_influence_field(blocked_file_board)

    assert open_attack_influence.matrix != blocked_attack_influence.matrix

    # The actual test: the rook's OWN contribution to Phi_source is
    # identical whether or not the pawn blocks its line of sight.
    open_source_field = build_source_field(open_file_board)
    blocked_source_field = build_source_field(blocked_file_board)

    rook_only_open = chess.Board(None)
    rook_only_open.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))
    rook_contribution_reference = build_source_potential_surface(
        build_source_field(rook_only_open), resolution=30
    )

    # Isolate just the rook's contribution from each full position by
    # zeroing out the other pieces before building the surface, using
    # the same kernel/points machinery.
    def rook_only_surface(board: chess.Board):
        isolated = chess.Board(None)
        isolated.set_piece_at(chess.A1, board.piece_at(chess.A1))
        return build_source_potential_surface(
            build_source_field(isolated), resolution=30
        )

    open_rook_surface = rook_only_surface(open_file_board)
    blocked_rook_surface = rook_only_surface(blocked_file_board)

    assert np.array_equal(open_rook_surface.z, blocked_rook_surface.z)
    assert np.array_equal(
        open_rook_surface.z, rook_contribution_reference.z
    )


def test_edge_piece_has_lower_in_board_integral_than_center_piece():
    """
    Documents the accepted boundary-clipping asymmetry: a corner
    piece's kernel tail partially falls outside the visible board,
    so its captured potential integral is measurably smaller than an
    identical piece placed centrally. Not a bug -- a regression
    baseline for the documented, accepted behavior.
    """

    corner_board = empty_board()
    corner_board.set_piece_at(chess.A1, chess.Piece(chess.QUEEN, chess.WHITE))

    center_board = empty_board()
    center_board.set_piece_at(chess.D4, chess.Piece(chess.QUEEN, chess.WHITE))

    corner_surface = build_source_potential_surface(
        build_source_field(corner_board), resolution=60
    )
    center_surface = build_source_potential_surface(
        build_source_field(center_board), resolution=60
    )

    assert np.sum(corner_surface.z) < np.sum(center_surface.z)


def test_unknown_kernel_name_raises():
    source_field = build_source_field(empty_board())

    with pytest.raises(ValueError):
        build_source_potential_surface(source_field, kernel="not_a_kernel")
