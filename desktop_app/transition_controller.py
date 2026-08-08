from __future__ import annotations

import time
from typing import Callable

from PySide6.QtCore import QObject, QTimer

from desktop_app.correspondence import CorrespondenceCache
from desktop_app.gl_canvas import MathCanvas
from desktop_app.layers._lerp import ease_in_out
from desktop_app.layers.attack_influence_layer import (
    interpolate_attack_influence_frame,
    render_attack_influence_frame,
)
from desktop_app.layers.critical_points_layer import interpolate_critical_points_frame, render_critical_points_frame
from desktop_app.layers.morse_smale_layer import interpolate_morse_smale_frame, render_morse_smale_frame
from desktop_app.layers.ridge_valley_layer import interpolate_ridge_valley_frame, render_ridge_valley_frame
from desktop_app.position_cache import PositionCache
from desktop_app.session_state import SessionState
from desktop_app.transition import TRANSITION_DURATION_MS, TRANSITION_FRAME_INTERVAL_MS, TransitionState


class TransitionController(QObject):
    """
    Drives move-to-move animation (docs/interactive_ui.md Part 4.5, Part 6).

    Owns the one QTimer that ticks the transition; on each tick it reads two
    already-`READY` `CacheEntry`s from `PositionCache` (never requests new
    work from it -- `MainWindow` only calls `start_transition` once `to_fen`
    is already READY) and the memoized `PositionCorrespondence` for that
    pair (`CorrespondenceCache`, computed once per pair, reused every tick),
    then pushes interpolated geometry straight into `MathCanvas`. Never
    calls `analysis/*` or `build_full_position_analysis` -- only
    interpolates already-cached endpoint data.

    Only four of the six registered layers animate here (Attack Influence,
    Critical Points, Ridge/Valley, Morse-Smale) -- this milestone's explicit
    scope. Equipotential and Gradient simply keep whatever geometry was
    last uploaded for them until `on_settled` runs its full, ordinary
    static render at the end of the transition; a known, deliberate
    limitation, not an oversight (see the phase report).

    `now_fn` is injectable so tests can drive `_on_tick` with a fake clock
    instead of real elapsed wall-clock time, avoiding timing-flakiness in
    the correspondence/animation test suite while production code still
    uses real `time.monotonic`.
    """

    def __init__(
        self,
        canvas: MathCanvas,
        session_state: SessionState,
        position_cache: PositionCache,
        on_settled: Callable[[str], None],
        correspondence_cache: CorrespondenceCache | None = None,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self._canvas = canvas
        self._session_state = session_state
        self._position_cache = position_cache
        self._on_settled = on_settled
        self.correspondence_cache = correspondence_cache or CorrespondenceCache()
        self._now_fn = now_fn

        self._settled_fen: str | None = None
        self._from_fen: str | None = None
        self._to_fen: str | None = None
        self._start_time: float = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(TRANSITION_FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)

    @property
    def is_animating(self) -> bool:
        return self._timer.isActive()

    def start_transition(self, to_fen: str) -> None:
        """
        The only entry point `MainWindow` calls, once `to_fen`'s cache entry
        is already READY. Safe to call again while a transition is already
        in flight (rapid successive moves/undo/redo): cancels the running
        timer and retargets cleanly to the new `to_fen`, animating from the
        last position that was *actually* fully settled -- never from
        whatever the abandoned transition happened to be interpolating
        mid-flight. This means several rapid moves collapse into one
        animation straight to the latest position rather than queuing or
        replaying every intermediate step.
        """
        self._timer.stop()

        if self._settled_fen is None or self._settled_fen == to_fen:
            self._settle(to_fen)
            return

        self._from_fen = self._settled_fen
        self._to_fen = to_fen
        self._start_time = self._now_fn()
        self._apply_frame(0.0)
        self._timer.start()

    def _on_tick(self) -> None:
        elapsed_ms = (self._now_fn() - self._start_time) * 1000.0
        raw_t = min(1.0, elapsed_ms / TRANSITION_DURATION_MS)
        self._apply_frame(raw_t)
        if raw_t >= 1.0:
            self._timer.stop()
            self._settle(self._to_fen)

    def _settle(self, fen: str) -> None:
        """
        Delegates the final frame to `MainWindow._render_all_layers` --
        the exact same function the non-animated path already used before
        this milestone -- rather than trusting the interpolated `t=1`
        branch to reproduce it. This is what guarantees the settled frame
        is bit-for-bit identical to a plain static render, and it's the one
        place Equipotential/Gradient (untouched during the transition ticks
        themselves) get caught back up to `fen`.
        """
        self._settled_fen = fen
        self._from_fen = None
        self._to_fen = None
        self._on_settled(fen)
        self._session_state.set_transition_state(TransitionState(from_fen=None, to_fen=fen, progress=1.0))

    def _apply_frame(self, raw_t: float) -> None:
        entry_a = self._position_cache.get(self._from_fen)
        entry_b = self._position_cache.get(self._to_fen)
        correspondence = self.correspondence_cache.get_or_compute(self._from_fen, entry_a, self._to_fen, entry_b)
        eased_t = ease_in_out(raw_t)

        overlay_frame = interpolate_attack_influence_frame(entry_a, entry_b, eased_t)
        self._canvas.set_overlay_colors(render_attack_influence_frame(overlay_frame))

        critical_points_frame = interpolate_critical_points_frame(correspondence.critical_points, eased_t)
        self._canvas.set_layer_geometry("critical_points", render_critical_points_frame(critical_points_frame))

        ridge_valley_frame = interpolate_ridge_valley_frame(
            correspondence.ridge_chains, correspondence.valley_chains, eased_t
        )
        self._canvas.set_layer_geometry("ridge_valley", render_ridge_valley_frame(ridge_valley_frame))

        morse_smale_frame = interpolate_morse_smale_frame(correspondence.morse_smale_cells, eased_t)
        self._canvas.set_layer_geometry("morse_smale", render_morse_smale_frame(morse_smale_frame))

        self._session_state.set_transition_state(
            TransitionState(from_fen=self._from_fen, to_fen=self._to_fen, progress=raw_t)
        )
