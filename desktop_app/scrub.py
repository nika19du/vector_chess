from __future__ import annotations

from dataclasses import dataclass

import chess.pgn


@dataclass(frozen=True)
class ScrubPosition:
    """
    A position along `SessionState.active_path()` (Phase 5e.2), analogous to
    `TransitionState` but positionally- rather than time-driven: `t` is
    exactly where the user's mouse is, not a fraction of elapsed time.

    path_index: index into `active_path()` of the node at/just-before this
        scrub position. 0 == root. Always a valid index into whatever
        `active_path()` returned when the scrub began
        (0 <= path_index <= len(path) - 1).
    t: fraction in [0.0, 1.0) toward `active_path()[path_index + 1]`.
        Clamped to 0.0 when `path_index` is the last index (the tip) --
        there is no "next" node to interpolate toward there, mirroring
        `TransitionState`'s own "progress == 1.0 means settled, no from_fen"
        convention.
    """

    path_index: int
    t: float


@dataclass(frozen=True)
class ScrubBounds:
    """
    Snapshot of the path a `ScrubController` is scrubbing over, captured
    once at drag-start (`SessionState.active_path()`). A tuple, not a list,
    so it is hashable/immutable like every other frozen value type in this
    module -- a drag never mutates the path it started with; if the
    underlying tree changes mid-drag (an externally-driven navigation), the
    scrub is cancelled rather than re-targeted (see `ScrubController.cancel`).
    """

    path: tuple[chess.pgn.GameNode, ...]


def clamp_scrub_fraction(raw_position: float, path_length: int) -> ScrubPosition:
    """
    Converts a raw, unbounded continuous position along the path (e.g.
    pixel_x / segment_width, computed by the caller -- `desktop_app.
    timeline_panel`'s scrub strip) into a valid `ScrubPosition`.

    Clamps `raw_position` into `[0, path_length - 1]` first (dragging past
    either end of the strip pins to that end), then splits the clamped value
    into `path_index`/`t`. At the tip (`path_index == path_length - 1`),
    `t` is forced to 0.0 -- there's no node beyond the tip to interpolate
    toward.
    """
    if path_length <= 1:
        return ScrubPosition(path_index=0, t=0.0)

    max_index = path_length - 1
    clamped = max(0.0, min(float(max_index), raw_position))
    path_index = int(clamped)

    if path_index >= max_index:
        return ScrubPosition(path_index=max_index, t=0.0)

    return ScrubPosition(path_index=path_index, t=clamped - path_index)


def snap_scrub_position_to_node(position: ScrubPosition, bounds: ScrubBounds) -> chess.pgn.GameNode:
    """
    The single, deterministic "nearest real endpoint" rule, reused for both
    the live board preview during a drag and the final commit target on
    release (one function, two call sites -- guarantees the commit always
    lands on whatever node was last shown as the preview).

    Approved rule: `t < 0.5` snaps to the left/lower node
    (`path[path_index]`); `t >= 0.5` snaps to the right/next node
    (`path[path_index + 1]`) -- exactly `t == 0.5` selects the right/next
    node, not the left one. At the tip, `t` is always 0.0 (see
    `clamp_scrub_fraction`), so this always resolves to `path[path_index]`
    there regardless of the `t >= 0.5` branch.
    """
    path = bounds.path
    if position.path_index >= len(path) - 1:
        return path[-1]
    if position.t < 0.5:
        return path[position.path_index]
    return path[position.path_index + 1]
