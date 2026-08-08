from __future__ import annotations

from dataclasses import dataclass

# docs/interactive_ui.md Part 6: "~500ms eased" for a move-to-move
# transition. The frame interval is a rendering constant (~60fps tick rate
# for the QTimer in desktop_app/transition_controller.py), not a
# mathematical one.
TRANSITION_DURATION_MS = 500.0
TRANSITION_FRAME_INTERVAL_MS = 16


@dataclass(frozen=True)
class TransitionState:
    """
    SessionState's record of the in-flight move-to-move animation
    (docs/interactive_ui.md Part 6), following the same pattern as every
    other SessionState slice (Part 4.1): a plain, immutable value type the
    owner (desktop_app/transition_controller.py) replaces wholesale on every
    change, never mutated in place.

    from_fen:
        The position the canvas is animating away from. None only for the
        initial, pre-any-transition state (nothing has ever been displayed
        yet) and for the settled state right after a transition completes
        (there is no "away from" once settled -- see `progress`).

    to_fen:
        The position the canvas is animating towards, or -- once settled
        (`progress == 1.0`) -- the position currently displayed.

    progress:
        The RAW (pre-easing) linear fraction of `TRANSITION_DURATION_MS`
        elapsed, in [0.0, 1.0]. 1.0 means settled: no transition is
        in-flight and the canvas exactly matches `to_fen`'s static render.
        Deliberately raw, not eased -- easing is a rendering detail applied
        by whoever consumes this state to interpolate a frame, not baked
        into the state itself.
    """

    from_fen: str | None
    to_fen: str
    progress: float
