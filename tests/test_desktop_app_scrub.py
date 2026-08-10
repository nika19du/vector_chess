import chess
import chess.pgn

from desktop_app.scrub import (
    ScrubBounds,
    ScrubPosition,
    clamp_scrub_fraction,
    snap_scrub_position_to_node,
)


def _path(length: int) -> tuple[chess.pgn.GameNode, ...]:
    """A real, linear chain of GameNodes of the given length (root + moves)."""
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]
    root = chess.pgn.Game()
    nodes = [root]
    node = root
    for move in moves[: length - 1]:
        node = node.add_variation(chess.Move.from_uci(move))
        nodes.append(node)
    assert len(nodes) == length
    return tuple(nodes)


# ---------------------------------------------------------
# clamp_scrub_fraction
# ---------------------------------------------------------


def test_clamp_scrub_fraction_at_exact_integer_positions():
    assert clamp_scrub_fraction(0.0, path_length=4) == ScrubPosition(path_index=0, t=0.0)
    assert clamp_scrub_fraction(1.0, path_length=4) == ScrubPosition(path_index=1, t=0.0)
    assert clamp_scrub_fraction(2.0, path_length=4) == ScrubPosition(path_index=2, t=0.0)


def test_clamp_scrub_fraction_midpoint():
    position = clamp_scrub_fraction(1.5, path_length=4)
    assert position.path_index == 1
    assert position.t == 0.5


def test_clamp_scrub_fraction_below_zero_clamps_to_start():
    assert clamp_scrub_fraction(-5.0, path_length=4) == ScrubPosition(path_index=0, t=0.0)


def test_clamp_scrub_fraction_past_the_end_clamps_to_tip_with_t_zero():
    position = clamp_scrub_fraction(100.0, path_length=4)
    assert position.path_index == 3  # last valid index (path_length - 1)
    assert position.t == 0.0


def test_clamp_scrub_fraction_exactly_at_the_tip_forces_t_zero():
    # path_length=4 -> valid indices 0..3; raw_position == 3.0 is the tip.
    position = clamp_scrub_fraction(3.0, path_length=4)
    assert position.path_index == 3
    assert position.t == 0.0


def test_clamp_scrub_fraction_just_below_the_tip_is_not_forced_to_the_tip():
    position = clamp_scrub_fraction(2.9, path_length=4)
    assert position.path_index == 2
    assert abs(position.t - 0.9) < 1e-9


def test_clamp_scrub_fraction_single_node_path_always_resolves_to_index_zero():
    assert clamp_scrub_fraction(0.0, path_length=1) == ScrubPosition(path_index=0, t=0.0)
    assert clamp_scrub_fraction(5.0, path_length=1) == ScrubPosition(path_index=0, t=0.0)
    assert clamp_scrub_fraction(-5.0, path_length=1) == ScrubPosition(path_index=0, t=0.0)


# ---------------------------------------------------------
# snap_scrub_position_to_node -- the approved deterministic tie-break rule
# ---------------------------------------------------------


def test_snap_just_below_half_selects_the_left_node():
    path = _path(4)
    bounds = ScrubBounds(path=path)
    position = ScrubPosition(path_index=1, t=0.49)

    assert snap_scrub_position_to_node(position, bounds) is path[1]


def test_snap_at_exactly_half_selects_the_right_node():
    # Approved rule: t == 0.5 selects the right/next real GameNode, not the left.
    path = _path(4)
    bounds = ScrubBounds(path=path)
    position = ScrubPosition(path_index=1, t=0.5)

    assert snap_scrub_position_to_node(position, bounds) is path[2]


def test_snap_just_above_half_selects_the_right_node():
    path = _path(4)
    bounds = ScrubBounds(path=path)
    position = ScrubPosition(path_index=1, t=0.51)

    assert snap_scrub_position_to_node(position, bounds) is path[2]


def test_snap_at_t_zero_selects_the_left_node():
    path = _path(4)
    bounds = ScrubBounds(path=path)
    position = ScrubPosition(path_index=0, t=0.0)

    assert snap_scrub_position_to_node(position, bounds) is path[0]


def test_snap_at_the_tip_always_selects_the_tip_node():
    path = _path(4)
    bounds = ScrubBounds(path=path)
    position = ScrubPosition(path_index=3, t=0.0)

    assert snap_scrub_position_to_node(position, bounds) is path[3]


def test_snap_and_clamp_agree_on_a_full_sweep_across_a_path():
    # End-to-end sanity: sweeping raw_position from 0 to path_length-1 and
    # snapping at each point only ever yields path nodes, never an index error.
    path = _path(5)
    bounds = ScrubBounds(path=path)

    for hundredth in range(0, (len(path) - 1) * 100 + 1):
        raw_position = hundredth / 100.0
        position = clamp_scrub_fraction(raw_position, path_length=len(path))
        node = snap_scrub_position_to_node(position, bounds)
        assert node in path
