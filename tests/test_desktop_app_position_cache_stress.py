"""
Regression/stress coverage for the PositionCache shutdown contract
(`desktop_app/position_cache.py`'s "Shutdown contract" docstring), added as
part of the stability investigation into the intermittent Windows
access-violation crash documented in `tests/conftest.py`.

Exercises the SUPPORTED lifecycle repeatedly -- create, submit, shutdown,
join, verify -- not "a cache with no shutdown must never crash." That was
never a supported contract: production ownership requires explicit cleanup.
The original leaked/never-shut-down failure mode is reproduced and compared
before/after the fix in `scripts/repro_position_cache_crash.py` (its `E1`
config crashed at 54.2% over N=600; `E1-fixed`, identical except for calling
the new `shutdown()` contract every repeat, crashed 0/600) -- that evidence
lives with the harness, not here, since an access violation kills the
interpreter and can never be observed by a pytest assertion.

Needs the real `ThreadPoolExecutor`, not `tests/conftest.py`'s synchronous
test stand-in, since the whole point is asserting correct behavior under
real background threading -- see `_REAL_THREADING_EXEMPT_FILES` there.
"""

from __future__ import annotations

import threading

import chess

from desktop_app.full_position_analysis import build_full_position_analysis
from desktop_app.position_cache import CacheEntryState, PositionCache
from scripts.chess_position_pool import build_position_pool

POSITIONS = build_position_pool()

# 100 repeats: the original bug's standalone repro (scripts/
# repro_position_cache_crash.py's E1 config) crashed at 54.2% with only 25
# leaked, never-shut-down repeats per batch -- 100 repeats through the
# *correct* lifecycle is a comfortably large regression gate without making
# the suite slow (each repeat runs one real, full analysis pipeline call).
LIFECYCLE_REPEATS = 100


def test_repeated_lifecycle_cycle_leaves_no_worker_alive(qtbot):
    """
    The core regression gate: create -> request -> shutdown (per contract)
    -> join -> verify no worker survives, repeated many times. This is the
    original bug's instance-churn shape (tests/test_desktop_app_position_cache.py
    already showed real instances get created repeatedly across a session),
    but run through the *supported* lifecycle instead of leaking every
    instance -- the leaked-instance shape itself belongs in the standalone
    harness (scripts/repro_position_cache_crash.py's A3/A4/E1 configs), not
    as a permanent pytest contract.
    """
    for i in range(LIFECYCLE_REPEATS):
        board = POSITIONS[i % len(POSITIONS)]
        worker_threads: list[threading.Thread] = []

        def recording_builder(b, _threads=worker_threads):
            _threads.append(threading.current_thread())
            return build_full_position_analysis(b)

        cache = PositionCache(builder=recording_builder, max_workers=2)
        fen = cache.request(board)
        qtbot.waitUntil(lambda: cache.get(fen).state == CacheEntryState.READY, timeout=5000)

        cache.shutdown()

        assert worker_threads, f"repeat {i}: builder never ran"
        for thread in worker_threads:
            assert not thread.is_alive(), f"repeat {i}: worker thread still alive after shutdown()"


def test_concurrent_requests_on_one_instance_with_max_workers_two_do_not_crash(qtbot):
    """
    Mirrors the standalone harness's B2 config (clean instance,
    max_workers=2, two distinct positions submitted concurrently) -- stayed
    at 0/600 in the harness, and this asserts correct *content*, not just
    absence of a crash, then shuts down cleanly every repeat.
    """
    for i in range(30):
        board_a = POSITIONS[(2 * i) % len(POSITIONS)]
        board_b = POSITIONS[(2 * i + 1) % len(POSITIONS)]
        cache = PositionCache(max_workers=2)

        fen_a = cache.request(board_a)
        fen_b = cache.request(board_b)
        qtbot.waitUntil(
            lambda: cache.get(fen_a).state == CacheEntryState.READY
            and cache.get(fen_b).state == CacheEntryState.READY,
            timeout=5000,
        )

        assert cache.get(fen_a).analysis.board.board_fen() == board_a.board_fen()
        assert cache.get(fen_b).analysis.board.board_fen() == board_b.board_fen()

        cache.shutdown()


def test_shutdown_stops_accepting_new_requests():
    """Contract point 2: request() after shutdown() is a deterministic no-op
    -- no entry created, no work scheduled -- rather than whatever
    ThreadPoolExecutor.submit on a shut-down executor happens to raise."""
    call_count = 0

    def counting_builder(board):
        nonlocal call_count
        call_count += 1
        return build_full_position_analysis(board)

    cache = PositionCache(builder=counting_builder)
    cache.shutdown()

    board = chess.Board()
    fen = cache.request(board)

    assert fen == board.board_fen()
    assert call_count == 0
    assert cache.get(fen).state == CacheEntryState.MISSING


def test_shutdown_joins_worker_threads_before_returning(qtbot):
    """Contract point 5: shutdown() blocks until every worker thread it owns
    has actually exited, not merely until the method returns."""
    worker_threads: list[threading.Thread] = []

    def recording_builder(board, _threads=worker_threads):
        _threads.append(threading.current_thread())
        return build_full_position_analysis(board)

    cache = PositionCache(builder=recording_builder)
    cache.request(chess.Board())
    qtbot.waitUntil(lambda: len(worker_threads) == 1, timeout=5000)

    cache.shutdown()

    assert not worker_threads[0].is_alive()


def test_shutdown_is_safe_to_call_more_than_once():
    cache = PositionCache()
    cache.shutdown()
    cache.shutdown()  # must not raise


def test_no_signal_fires_after_shutdown_into_a_destroyed_cache():
    """
    Contract point 6: once shutdown has begun, a worker's completion must
    not publish into a slot -- the Qt objects a queued cross-thread emit
    would marshal into may already be destroyed/destructing by the time the
    callback runs. Uses a builder gated by two threading.Events so the test
    controls the exact ordering: the worker is confirmed running *before*
    shutdown() is requested, and is only allowed to finish *after*.
    """
    started = threading.Event()
    release = threading.Event()
    worker_threads: list[threading.Thread] = []

    def slow_builder(board):
        worker_threads.append(threading.current_thread())
        started.set()
        release.wait(timeout=5)
        return build_full_position_analysis(board)

    cache = PositionCache(builder=slow_builder, max_workers=1)
    received: list[str] = []
    cache.position_ready.connect(lambda fen: received.append(fen))

    cache.request(chess.Board())
    assert started.wait(timeout=5), "worker never started"

    # Non-blocking: marks shutdown requested and cancels queued-but-not-
    # started work, without waiting for the still-running worker (which is
    # deliberately blocked on `release` right now).
    cache.shutdown(wait=False)
    release.set()

    worker_threads[0].join(timeout=5)
    assert not worker_threads[0].is_alive()
    assert received == [], "position_ready fired for a request that completed after shutdown() began"
