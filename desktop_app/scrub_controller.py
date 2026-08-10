from __future__ import annotations

import chess.pgn
from PySide6.QtCore import QObject, QTimer

from desktop_app.correspondence import CorrespondenceCache
from desktop_app.gl_canvas import MathCanvas
from desktop_app.layers.attack_influence_layer import (
    interpolate_attack_influence_frame,
    render_attack_influence_frame,
)
from desktop_app.layers.critical_points_layer import interpolate_critical_points_frame, render_critical_points_frame
from desktop_app.layers.morse_smale_layer import interpolate_morse_smale_frame, render_morse_smale_frame
from desktop_app.layers.ridge_valley_layer import interpolate_ridge_valley_frame, render_ridge_valley_frame
from desktop_app.position_cache import CacheEntryState, PositionCache
from desktop_app.scrub import ScrubBounds, ScrubPosition, snap_scrub_position_to_node


class ScrubController(QObject):
    """
    Drives continuous Timeline scrubbing (Phase 5e.2, docs/interactive_ui.md
    Part 8's deferred slider). Deliberately mirrors `desktop_app.
    transition_controller.TransitionController`'s `_apply_frame` closely --
    same `PositionCache`/`CorrespondenceCache` reuse, same 4-of-6-layer
    scope (Attack Influence, Critical Points, Ridge/Valley, Morse-Smale;
    Equipotential/Gradient stay frozen, an inherited limitation, not a new
    one), same "never triggers new computation, only reads already-cached
    endpoints" contract -- but is driven by a mouse position instead of a
    `QTimer` tick, which changes two things:

      * Uses RAW `t`, never `ease_in_out(t)`. `t` here *is* the mouse's
        fractional position, not a time budget being spent; easing it would
        desync the rendered state from where the cursor actually is -- bad
        UX for a drag gesture.
      * Coalesces rapid mousemove-driven updates into at most one canvas
        write per Qt event-loop turn, via the exact same guard-flag +
        `QTimer.singleShot(0, ...)` idiom `desktop_app.timeline_panel.
        TimelinePanel` already uses for its own deferred rebuild (the
        proven fix for a real Qt reentrancy crash in this codebase --
        reused here, not reinvented).

    Never mutates `SessionState.current_node` -- that happens exactly once,
    from `TimelinePanel`'s `mouseReleaseEvent` handler, using `end()`'s
    return value, after this controller has already returned to idle. Scrub
    position is not a `SessionState` slice; it lives only here, exactly as
    transient/derived as `TransitionState` already is relative to
    `SessionState`.

    Operates entirely on real `chess.pgn.GameNode`s drawn from a snapshot of
    `SessionState.active_path()` -- every `from_fen`/`to_fen` this class
    ever computes is a real, already-visited position's `board_fen()`. No
    fractional/fake position is ever constructed.
    """

    def __init__(
        self,
        canvas: MathCanvas,
        position_cache: PositionCache,
        correspondence_cache: CorrespondenceCache | None = None,
    ) -> None:
        super().__init__()
        self._canvas = canvas
        self._position_cache = position_cache
        self.correspondence_cache = correspondence_cache or CorrespondenceCache()

        self._bounds: ScrubBounds | None = None  # None == idle
        self._pending: ScrubPosition | None = None
        # Guards against queuing a second QTimer.singleShot before the first
        # one fires -- same coalescing idiom as TimelinePanel's own
        # _rebuild_scheduled: several update() calls arriving before the
        # event loop's next turn collapse into one applied frame, which
        # always re-reads self._pending fresh at fire time rather than
        # using a captured value.
        self._frame_scheduled = False

        self._position_cache.position_ready.connect(self._on_position_ready)
        self._connected_to_position_ready = True

    @property
    def is_scrubbing(self) -> bool:
        return self._bounds is not None

    @property
    def path_length(self) -> int | None:
        """
        Length of the snapshotted path, or `None` if idle. Lets a caller
        (`desktop_app.timeline_panel`'s scrub strip) convert a raw pixel
        position into a `ScrubPosition` without reaching into this
        controller's private `_bounds` state.
        """
        return len(self._bounds.path) if self._bounds is not None else None

    def begin(self, active_path: list[chess.pgn.GameNode]) -> None:
        """
        Starts a scrub session over a snapshot of `active_path` (expected to
        be `SessionState.active_path()`, taken by the caller at press time).
        The snapshot is never re-walked mid-drag: if the underlying tree
        changes from an external source while scrubbing, the scrub is
        cancelled (`cancel()`), not re-targeted.
        """
        self._bounds = ScrubBounds(path=tuple(active_path))
        self._pending = None

    def update(self, position: ScrubPosition) -> None:
        """
        Records the latest mouse-driven position and schedules at most one
        coalesced render. Safe to call at mousemove flood rate: the
        prefetch request below is cheap/idempotent (`PositionCache.request`
        no-ops for an already COMPUTING/READY position), and the actual
        render is deferred and coalesced.
        """
        if not self.is_scrubbing:
            return
        self._pending = position
        self._prefetch_segment(position)
        self._schedule_frame()

    def snap_to_nearest_node(self, position: ScrubPosition) -> chess.pgn.GameNode:
        assert self._bounds is not None
        return snap_scrub_position_to_node(position, self._bounds)

    def end(self, position: ScrubPosition) -> chess.pgn.GameNode:
        """Returns the commit target (the exact same snap rule used for the
        live preview) and resets to idle."""
        node = self.snap_to_nearest_node(position)
        self._bounds = None
        self._pending = None
        return node

    def cancel(self) -> None:
        """
        Resets to idle without producing a commit target -- for an
        externally-driven `current_node_changed` arriving mid-drag (a
        branch switch, keyboard navigation, etc.), whose target this
        controller's snapshotted path no longer reflects. Cheap and
        idempotent, so (unlike the rebuild it may trigger elsewhere) it can
        run synchronously from inside a signal handler.
        """
        self._bounds = None
        self._pending = None

    def shutdown(self) -> None:
        """
        Stops reacting to `position_ready` and resets to idle. Unlike
        `PositionCache.shutdown()`, there is no worker thread of its own to
        join here -- this only needs to stop delivering into a `MainWindow`
        that may be mid-teardown. Safe to call more than once.
        """
        if self._connected_to_position_ready:
            self._position_cache.position_ready.disconnect(self._on_position_ready)
            self._connected_to_position_ready = False
        self._bounds = None
        self._pending = None

    def _segment_endpoints(self, position: ScrubPosition) -> tuple[chess.pgn.GameNode, chess.pgn.GameNode | None]:
        assert self._bounds is not None
        path = self._bounds.path
        lower = path[position.path_index]
        upper = path[position.path_index + 1] if position.path_index < len(path) - 1 else None
        return lower, upper

    def _segment_fens(self, position: ScrubPosition) -> tuple[str, str]:
        """
        `(from_fen, to_fen)` for the segment `position` falls in. At the
        branch tip (no "next" node), `from_fen == to_fen` -- interpolating a
        position against itself at `t=0` is an identity operation that
        reproduces that position's exact static frame through the same
        interpolate_*/render_* calls every other segment uses, rather than
        needing a special-cased "just show it statically" branch here.
        """
        lower, upper = self._segment_endpoints(position)
        from_fen = lower.board().board_fen()
        to_fen = upper.board().board_fen() if upper is not None else from_fen
        return from_fen, to_fen

    def _prefetch_segment(self, position: ScrubPosition) -> None:
        lower, upper = self._segment_endpoints(position)
        self._position_cache.request(lower.board())
        if upper is not None:
            self._position_cache.request(upper.board())

    def _schedule_frame(self) -> None:
        if self._frame_scheduled:
            return
        self._frame_scheduled = True
        QTimer.singleShot(0, self._run_scheduled_scrub_frame)

    def _run_scheduled_scrub_frame(self) -> None:
        self._frame_scheduled = False
        if not self.is_scrubbing or self._pending is None:
            return
        self._apply_frame(self._pending)

    def _on_position_ready(self, fen: str) -> None:
        """
        Stale-result discard: only re-schedules a render if `fen` is one of
        the *current* pending segment's endpoints, re-derived fresh from
        `self._pending` rather than anything captured when the request was
        issued. If the user has since scrubbed to a different segment, this
        is a no-op -- the result for the abandoned segment is simply never
        applied.
        """
        if not self.is_scrubbing or self._pending is None:
            return
        if fen in self._segment_fens(self._pending):
            self._schedule_frame()

    def _apply_frame(self, position: ScrubPosition) -> None:
        from_fen, to_fen = self._segment_fens(position)
        # At the tip, from_fen == to_fen and t is always 0.0 (clamp_scrub_
        # fraction's contract) -- t is not otherwise meaningful there, but
        # passing it through unchanged rather than hardcoding 0.0 keeps this
        # function honest about which value it actually interpolates with.
        t = position.t

        entry_a = self._position_cache.get(from_fen)
        entry_b = self._position_cache.get(to_fen)
        if entry_a.state != CacheEntryState.READY or entry_b.state != CacheEntryState.READY:
            return  # clamp: leave whatever was last uploaded, never fabricate a blend

        correspondence = self.correspondence_cache.get_or_compute(from_fen, entry_a, to_fen, entry_b)

        overlay_frame = interpolate_attack_influence_frame(entry_a, entry_b, t)
        self._canvas.set_overlay_colors(render_attack_influence_frame(overlay_frame))

        critical_points_frame = interpolate_critical_points_frame(correspondence.critical_points, t)
        self._canvas.set_layer_geometry("critical_points", render_critical_points_frame(critical_points_frame))

        ridge_valley_frame = interpolate_ridge_valley_frame(
            correspondence.ridge_chains, correspondence.valley_chains, t
        )
        self._canvas.set_layer_geometry("ridge_valley", render_ridge_valley_frame(ridge_valley_frame))

        morse_smale_frame = interpolate_morse_smale_frame(correspondence.morse_smale_cells, t)
        self._canvas.set_layer_geometry("morse_smale", render_morse_smale_frame(morse_smale_frame))
