from __future__ import annotations

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

    def get(self, fen: str) -> CacheEntry:
        return self._entries.get(fen, _MISSING_ENTRY)

    def request(self, board: chess.Board) -> str:
        """
        Ensure a computation for `board`'s position is in flight or already
        done. Returns the FEN key. A second call for a position already
        COMPUTING or READY returns immediately without submitting another
        computation -- duplicate requests never trigger duplicate work.
        """
        fen = board.board_fen()
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
        # Thread-safe regardless of which thread calls emit(): PySide6 queues
        # delivery to slots living on a different thread than the emitter --
        # this is the "safe publication back to the UI" the architecture asks
        # for, via Qt's own mechanism rather than a custom lock.
        self.position_ready.emit(fen)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
