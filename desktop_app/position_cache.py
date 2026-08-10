from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import chess
from PySide6.QtCore import QObject, Signal

from desktop_app.full_position_analysis import FullPositionAnalysis, build_full_position_analysis


class CacheEntryState(Enum):
    MISSING = auto()
    COMPUTING = auto()
    READY = auto()


@dataclass(frozen=True)
class CacheEntry:
    """
    Immutable snapshot of one position's cached analysis (docs/interactive_ui.md
    Part 4.3). A state transition always constructs a *new* `CacheEntry` and
    replaces the dict slot wholesale -- never mutates an existing entry's fields
    -- so a single-key dict assignment (atomic under CPython's GIL) is the whole
    "atomic swap" the architecture asks for; no lock is needed for readers.

    Phase 5c: `attack_influence_field` (the single object Phase 5a/5b cached)
    is replaced by `analysis: FullPositionAnalysis`, which bundles it alongside
    every other shared object the five new layers need (surface, classified
    critical points + quality, ridge/valley chains + quality, the Morse-Smale
    complex + cell quality) -- one cache entry, one computation, six layers
    reading from it.
    """

    state: CacheEntryState
    analysis: FullPositionAnalysis | None = None


_MISSING_ENTRY = CacheEntry(state=CacheEntryState.MISSING)


class PositionCache(QObject):
    """
    FEN-keyed cache of per-position analysis results (docs/interactive_ui.md
    Part 4.3). Keyed by `chess.Board.board_fen()` (piece placement only, not the
    full FEN and not ply index), so two move orders reaching the same placement
    share one entry, and undo/branching to an already-visited position never
    serves data for the wrong position.

    Executor note -- a deliberate, considered Phase 5c decision, not a change
    to the frozen architecture's observable contract: Part 4.3 names a process
    pool for genuinely `scipy`-heavy work, and `build_full_position_analysis`
    (spline fit, Newton iteration, marching, separatrix tracing) qualifies.
    `ThreadPoolExecutor` is kept anyway for two concrete reasons: (1)
    `AttackInfluenceSurface.spline` is a `scipy.interpolate.RectBivariateSpline`
    -- crossing a process boundary would require it (and everything built on
    top of it) to pickle correctly, unverified and risky to introduce in the
    same phase as five new layers; (2) numpy/scipy's C-implemented inner loops
    release the GIL during their heavy work, so a thread pool already gets
    real parallelism for this workload, not the pure-Python GIL-bound case a
    process pool exists to fix. Entirely hidden behind this class's public
    interface (`request`, `get`, `position_ready`); revisited only if a
    measured stall shows it's actually a problem, not swapped preemptively.

    No LRU eviction and no prefetch yet: eviction sizing (Part 4.3's 200-entry
    default) and prefetch are both relative to a scrub position that doesn't
    exist until the timeline (Phase 5e). Deferred to the phase that needs them.

    Shutdown contract (stability investigation, see `scripts/
    repro_position_cache_crash.py` and its evidence): a `PositionCache`
    that is simply dropped without calling `shutdown()` leaks its
    `ThreadPoolExecutor`'s worker thread(s) -- they keep running whatever
    `build_full_position_analysis` call they were mid-flight on for as long
    as it takes, unjoined. A leaked worker still executing scipy-heavy
    analysis code, combined with a *later*, unrelated `position_ready.emit()`
    (from this instance or any other) being delivered to a connected slot
    while that worker is still running, is a confirmed, reproducible cause of
    an intermittent native (Windows access-violation) crash -- confirmed via
    a standalone harness (54% crash rate with a connected slot vs. 0% with no
    slot connected, otherwise identical leak shape, N=600 each) and via the
    real full-test-suite reproduction (a leaked `ThreadPoolExecutor` worker
    caught mid-`analysis/morse_smale.py` while the main thread pumped Qt
    events for a later, unrelated test). `shutdown()` exists specifically to
    make this impossible: called before this cache (and the Qt objects that
    own it) are torn down, it (1) stops `request()` from scheduling any more
    work, (2) cancels not-yet-started queued work, (3) lets any
    already-running analysis finish (there is no safe way to abort native
    scipy work mid-call), (4) blocks until every worker thread it owns has
    actually exited, and (5) guarantees no `position_ready` reaches a slot
    once shutdown has begun -- so by the time `shutdown()` returns, no worker
    can still be racing whatever Qt teardown happens next. See
    `MainWindow.closeEvent` for the production call site.
    """

    position_ready = Signal(str)  # emits the fen that just became READY

    def __init__(
        self,
        builder: Callable[[chess.Board], FullPositionAnalysis] = build_full_position_analysis,
        max_workers: int = 2,
    ) -> None:
        super().__init__()
        self._builder = builder
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._entries: dict[str, CacheEntry] = {}
        # Set by shutdown() -- request() checks it to stop scheduling new
        # work, and _on_computed() checks it to stop publishing into Qt
        # objects that may already be torn down. A threading.Event (not a
        # plain bool) because _on_computed reads it from a worker thread
        # while shutdown() sets it from the GUI thread.
        self._shutdown_requested = threading.Event()

    def get(self, fen: str) -> CacheEntry:
        return self._entries.get(fen, _MISSING_ENTRY)

    def request(self, board: chess.Board) -> str:
        """
        Ensure a computation for `board`'s position is in flight or already
        done. Returns the FEN key. A second call for a position already
        COMPUTING or READY returns immediately without submitting another
        computation -- duplicate requests never trigger duplicate work.

        A no-op (beyond computing and returning the FEN) once `shutdown()`
        has been called: no entry is created and no work is scheduled. This
        is the deterministic behavior the shutdown contract requires for any
        `request()` call that arrives after teardown has begun, rather than
        whatever `ThreadPoolExecutor.submit` on a shut-down executor happens
        to raise.
        """
        fen = board.board_fen()
        if self._shutdown_requested.is_set():
            return fen
        if fen in self._entries:
            return fen

        self._entries[fen] = CacheEntry(state=CacheEntryState.COMPUTING)
        # A copy crosses the thread boundary, not the live board -- board is
        # mutable, and the interactive board keeps mutating it in place
        # (moves) while a background computation for an earlier position may
        # still be in flight.
        future = self._executor.submit(self._builder, board.copy())
        future.add_done_callback(lambda done_future, fen=fen: self._on_computed(fen, done_future))
        return fen

    def _on_computed(self, fen: str, future: Future) -> None:
        analysis = future.result()
        self._entries[fen] = CacheEntry(
            state=CacheEntryState.READY,
            analysis=analysis,
        )
        if self._shutdown_requested.is_set():
            # Shutdown contract point 6: never deliver a signal once teardown
            # has begun -- the Qt objects a queued cross-thread emit would
            # marshal into may already be destroyed/destructing by the time
            # this callback runs.
            return
        # Thread-safe regardless of which thread calls emit(): PySide6 queues
        # delivery to slots living on a different thread than the emitter --
        # this is the "safe publication back to the UI" the architecture asks
        # for, via Qt's own mechanism rather than a custom lock.
        self.position_ready.emit(fen)

    def shutdown(self, wait: bool = True) -> None:
        """
        Stops accepting new requests, cancels queued-but-not-started work,
        lets any already-running analysis finish safely, and (with the
        default `wait=True`) blocks until every worker thread this cache
        owns has actually exited -- see the class docstring's "Shutdown
        contract" section. Safe to call more than once.
        """
        self._shutdown_requested.set()
        self._executor.shutdown(wait=wait, cancel_futures=True)
