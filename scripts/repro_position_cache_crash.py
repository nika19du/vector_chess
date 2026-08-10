"""
Standalone (non-pytest) reproduction harness for the intermittent Windows
access-violation crash documented in `tests/conftest.py`
(`_run_position_cache_synchronously_in_tests`'s docstring): a native crash
inside scipy (`RectBivariateSpline.__call__` in one occurrence,
`analysis/ridge_valley.py::check_self_intersection` in another -- the latter
has zero scipy/numpy calls, so that crash site is almost certainly heap
corruption manifesting elsewhere, not a local bug) on a `PositionCache`
`ThreadPoolExecutor` worker thread while the main thread pumps the Qt event
loop.

Not a test file: an access violation kills the interpreter outright, so a
pytest assertion can never observe it -- this script isolates each hypothesis
axis from the stability investigation plan as a runnable configuration, and
is meant to be invoked repeatedly via `--supervise` so a crash on repeat N
does not lose evidence about repeats 1..N-1 (the supervisor spawns one
subprocess per batch and tallies exit codes; a nonzero/abnormal exit code is
the crash signal itself).

Usage (run from the repo root, using the project's real .venv):

    .venv\\Scripts\\python.exe scripts\\repro_position_cache_crash.py \\
        --config B2 --repeats 200 --batch-size 20 --supervise

Worker mode (what `--supervise` invokes per batch; not normally run by hand):

    .venv\\Scripts\\python.exe scripts\\repro_position_cache_crash.py \\
        --config B2 --batch-size 20 --seed-offset 3
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Matches tests/conftest.py's rationale: on this Windows environment the
# offscreen QPA platform does not provide a real OpenGL context, but it does
# provide a real Qt event loop -- which is the only Qt behavior this harness
# needs, and it's the same platform the original crash was reproduced under.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import RectBivariateSpline  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from desktop_app.full_position_analysis import build_full_position_analysis  # noqa: E402
from desktop_app.position_cache import CacheEntryState, PositionCache  # noqa: E402
from scripts.chess_position_pool import build_position_pool  # noqa: E402


def _heartbeat(label: str) -> None:
    # Printed before every risky operation so the last line before an
    # abnormal process exit identifies the in-flight stage even without a
    # debugger attached to a crashed interpreter.
    print(f"HEARTBEAT {time.monotonic():.3f} {label}", flush=True)


# Reused identically by every configuration here AND by
# tests/test_desktop_app_position_cache_stress.py (both import
# scripts.chess_position_pool), so crash rates measured inside pytest and
# outside pytest are directly comparable.
POSITION_POOL = build_position_pool()


# ---------------------------------------------------------------------------
# Qt plumbing shared by every configuration that needs a real event loop.
# ---------------------------------------------------------------------------


def _make_qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _pump_events(app, duration_seconds: float) -> None:
    deadline = time.monotonic() + duration_seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.001)


def _wait_until_done(app, futures, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not all(f.done() for f in futures) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.001)


def _make_cache_with_slot(max_workers: int, connect_slot: bool = True) -> PositionCache:
    # Production always has a real slot connected (MainWindow._on_position_ready)
    # -- connecting a trivial counting slot here mirrors that shape faithfully
    # rather than testing emit() into an empty receiver list, which is cheaper
    # and not representative.
    cache = PositionCache(max_workers=max_workers)
    if connect_slot:
        counter = {"n": 0}
        cache.position_ready.connect(lambda fen: counter.__setitem__("n", counter["n"] + 1))
    return cache


def _submit_concurrent(cache: PositionCache, boards: list[chess.Board]):
    # Bypasses PositionCache.request() to get direct Future references for
    # polling running()/done() -- request() itself doesn't expose the Future,
    # and this harness needs it. Diagnostic-only; not a pattern for production
    # code, which should keep using request()/get().
    futures = []
    for board in boards:
        fen = board.board_fen()
        future = cache._executor.submit(cache._builder, board.copy())
        future.add_done_callback(lambda f, fen=fen, c=cache: c._on_computed(fen, f))
        futures.append(future)
    return futures


# ---------------------------------------------------------------------------
# Axis A -- leaked-worker-thread hypothesis
# ---------------------------------------------------------------------------


def run_config_A1(positions, batch_size, rng) -> None:
    """One instance, max_workers=1, never shut down, reused for the whole
    batch -- mirrors production's current single, never-explicitly-shutdown
    instance."""
    app = _make_qapp()
    cache = _make_cache_with_slot(max_workers=1)
    for i in range(batch_size):
        board = rng.choice(positions)
        fen = board.board_fen()
        _heartbeat(f"A1 repeat={i} submit")
        cache.request(board)
        deadline = time.monotonic() + 10.0
        while cache.get(fen).state != CacheEntryState.READY and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.001)
        _heartbeat(f"A1 repeat={i} state={cache.get(fen).state}")
    # Deliberately no shutdown() call -- this config is the leak itself.


def run_config_A2(positions, batch_size, rng) -> None:
    """One instance per repeat, max_workers=1, explicit shutdown(wait=True)
    and effectively joined before the next repeat starts -- expected-safe
    floor: no overlap is possible by construction."""
    app = _make_qapp()
    for i in range(batch_size):
        board = rng.choice(positions)
        fen = board.board_fen()
        cache = _make_cache_with_slot(max_workers=1)
        _heartbeat(f"A2 repeat={i} submit")
        cache.request(board)
        deadline = time.monotonic() + 10.0
        while cache.get(fen).state != CacheEntryState.READY and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.001)
        cache._executor.shutdown(wait=True, cancel_futures=False)
        _heartbeat(f"A2 repeat={i} done+shutdown+joined")


def run_config_A3(positions, batch_size, rng) -> None:
    """New PositionCache per repeat, max_workers=1, never shut down --
    mirrors tests/test_desktop_app_position_cache.py's actual shape (~8 real
    instances across its test functions, none ever call .shutdown())."""
    app = _make_qapp()
    leaked: list[PositionCache] = []  # keep references so GC can't help us
    for i in range(batch_size):
        board = rng.choice(positions)
        cache = _make_cache_with_slot(max_workers=1)
        leaked.append(cache)
        _heartbeat(f"A3 repeat={i} new instance, leaked_count={len(leaked)}")
        cache.request(board)
        # No wait for completion -- move straight to the next instance so the
        # previous one's worker is very likely still running.
    _pump_events(app, 0.2)
    _heartbeat(f"A3 batch complete, {len(leaked)} instances leaked")


def run_config_E1_no_slot(positions, batch_size, rng) -> None:
    """Same shape as E1 (never wait, never shut down, tight per-repeat event
    pump) but with NO slot connected to position_ready -- ablation to test
    whether cross-thread signal delivery (queued to the main thread by Qt's
    auto-connection, processed during the per-repeat processEvents pump) is a
    necessary co-factor alongside the concurrent native SciPy execution, or
    whether the raw leak+pump shape alone is sufficient without any signal
    ever being delivered."""
    app = _make_qapp()
    for i in range(batch_size):
        cache = _make_cache_with_slot(max_workers=1, connect_slot=False)
        board = rng.choice(positions)
        _heartbeat(f"E1-no-slot repeat={i} submit, will not wait")
        cache.request(board)
        _pump_events(app, 0.01)
        _heartbeat(f"E1-no-slot repeat={i} moving on with future in flight")


def run_config_E1_fixed(positions, batch_size, rng) -> None:
    """Post-fix comparison: identical shape to E1 (new instance per repeat,
    slot connected, tight per-repeat event pump) but each repeat now calls
    the new PositionCache.shutdown() contract (stops accepting requests,
    cancels queued work, joins running workers, suppresses post-shutdown
    emit) before moving to the next repeat -- the corrected usage pattern.
    If the shutdown contract is what actually closes the gap, this should
    reproduce at ~0% where plain E1 reproduced at ~54%."""
    app = _make_qapp()
    for i in range(batch_size):
        cache = _make_cache_with_slot(max_workers=1)
        board = rng.choice(positions)
        _heartbeat(f"E1-fixed repeat={i} submit")
        cache.request(board)
        _pump_events(app, 0.01)
        cache.shutdown()  # wait=True by default -- blocks until joined
        _heartbeat(f"E1-fixed repeat={i} shutdown+joined")


def run_config_A5(positions, batch_size, rng) -> None:
    """Mass-leak accumulation within a SINGLE process: many PositionCache
    instances (production's real max_workers=2 default), never shut down,
    created back-to-back with no wait, so dozens of worker threads are
    concurrently alive and competing for the GIL by the end of the batch --
    tests whether sheer leaked-thread VOLUME (not a real MainWindow, not
    pytest's own complexity) is sufficient to reproduce the real full-suite
    crash observed at tests/test_desktop_app_canvas.py (a leaked
    ThreadPoolExecutor-0_0 worker still inside
    analysis/morse_smale.py::_trace_gradient_flow_direction while the main
    thread pumped Qt events for an unrelated, much later test)."""
    app = _make_qapp()
    leaked: list[PositionCache] = []
    for i in range(batch_size):
        board = rng.choice(positions)
        cache = _make_cache_with_slot(max_workers=2)
        leaked.append(cache)
        cache.request(board)
        app.processEvents()
        if i % 25 == 0:
            _heartbeat(f"A5 repeat={i} leaked_count={len(leaked)}")
    # Hold the main thread pumping events for a while at the end, mirroring
    # the real crash's "main thread pumping Qt events while old workers are
    # still mid-flight" shape, instead of exiting immediately.
    _pump_events(app, 2.0)
    _heartbeat(f"A5 batch complete, {len(leaked)} instances leaked, still pumping done")


def run_config_A4(positions, batch_size, rng) -> None:
    """Same shape as A3 but with a *guaranteed* overlap window: poll the
    previous repeat's Future.running() before submitting the next instance's
    request, instead of hoping timing creates overlap."""
    app = _make_qapp()
    leaked: list[PositionCache] = []
    prev_future = None
    for i in range(batch_size):
        board = rng.choice(positions)
        cache = _make_cache_with_slot(max_workers=1)
        leaked.append(cache)
        if prev_future is not None:
            waited = 0.0
            while not prev_future.running() and not prev_future.done() and waited < 1.0:
                app.processEvents()
                time.sleep(0.002)
                waited += 0.002
            _heartbeat(
                f"A4 repeat={i} prev_future.running={prev_future.running()} "
                f"done={prev_future.done()}"
            )
        futures = _submit_concurrent(cache, [board])
        prev_future = futures[0]
    _pump_events(app, 0.2)
    _heartbeat(f"A4 batch complete, {len(leaked)} instances leaked")


# ---------------------------------------------------------------------------
# Axis B -- same-instance concurrency hypothesis, independent of any leak
# ---------------------------------------------------------------------------


def run_config_B1(positions, batch_size, rng) -> None:
    """Control: identical shape to A2 (clean instance per repeat,
    max_workers=1, shutdown+joined between repeats) -- should never crash if
    the cause really is concurrency, not mere volume of scipy calls."""
    run_config_A2(positions, batch_size, rng)


def run_config_B2(positions, batch_size, rng, max_workers: int = 2) -> None:
    """One clean instance, max_workers=N, N distinct positions submitted
    concurrently with no wait between submissions -- provably concurrent
    native calls on a single, cleanly-managed instance (no leak involved)."""
    app = _make_qapp()
    for i in range(batch_size):
        cache = _make_cache_with_slot(max_workers=max_workers)
        boards = rng.sample(positions, k=min(max_workers, len(positions)))
        _heartbeat(f"B{max_workers} repeat={i} submit {len(boards)} concurrently")
        futures = _submit_concurrent(cache, boards)
        _wait_until_done(app, futures)
        cache._executor.shutdown(wait=True)
        _heartbeat(f"B{max_workers} repeat={i} all_done={[f.done() for f in futures]}")


def run_config_B3(positions, batch_size, rng) -> None:
    run_config_B2(positions, batch_size, rng, max_workers=4)


def run_config_B2_spline_only(positions, batch_size, rng, workers: int = 2) -> None:
    """Two+ threads doing nothing but RectBivariateSpline construction and
    repeated evaluation on INDEPENDENTLY built spline objects -- isolates
    whether the unsafety needs a shared spline object or is a global/static
    state issue in the compiled FITPACK routine that hits independent objects
    too. No chess/Qt involved at all."""

    def worker(seed: int) -> None:
        rng_np = np.random.default_rng(seed)
        row = np.arange(8) + 0.5
        col = np.arange(8) + 0.5
        fine = np.linspace(0.5, 7.5, 200)
        for _ in range(batch_size):
            matrix = rng_np.random((8, 8)) * 4 - 2
            spline = RectBivariateSpline(row, col, matrix, kx=3, ky=3)
            spline(fine, fine)
            spline(fine, fine, dx=1, dy=0)
            spline(fine, fine, dx=0, dy=1)

    _heartbeat(f"B2-spline-only start workers={workers} batch_size={batch_size}")
    threads = [threading.Thread(target=worker, args=(seed,)) for seed in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    _heartbeat("B2-spline-only complete")


def run_config_B2_gaussian_only(positions, batch_size, rng, workers: int = 2) -> None:
    """Same shape as B2_spline_only but isolating scipy.ndimage.gaussian_filter
    (the other native call in analysis/attack_influence_surface.py, preceding
    the spline fit) as the potentially-unsafe call instead of the spline."""

    def worker(seed: int) -> None:
        rng_np = np.random.default_rng(seed)
        for _ in range(batch_size):
            matrix = rng_np.random((8, 8)) * 4 - 2
            gaussian_filter(matrix, sigma=0.6, mode="nearest")

    _heartbeat(f"B2-gaussian-only start workers={workers} batch_size={batch_size}")
    threads = [threading.Thread(target=worker, args=(seed,)) for seed in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    _heartbeat("B2-gaussian-only complete")


# ---------------------------------------------------------------------------
# Axis C -- is Qt event-loop pumping a necessary co-factor
# ---------------------------------------------------------------------------


def run_config_C1(positions, batch_size, rng) -> None:
    """Same shape as B2 (Qt pump active, slot connected) -- kept as a named
    alias so the config table reads in the same order as the investigation
    plan's axis list."""
    run_config_B2(positions, batch_size, rng, max_workers=2)


def run_config_C2(positions, batch_size, rng) -> None:
    """Identical request pattern to B2/C1 but with NO QApplication/event loop
    at all -- raw ThreadPoolExecutor + build_full_position_analysis calls
    only. If this still crashes at a similar rate, Qt is incidental and the
    cause is bare thread+SciPy concurrency."""
    for i in range(batch_size):
        executor = ThreadPoolExecutor(max_workers=2)
        boards = rng.sample(positions, k=2)
        _heartbeat(f"C2 repeat={i} submit (no Qt)")
        futures = [executor.submit(build_full_position_analysis, b.copy()) for b in boards]
        for f in futures:
            f.result(timeout=15.0)
        executor.shutdown(wait=True)
        _heartbeat(f"C2 repeat={i} done (no Qt)")


def run_config_C3(positions, batch_size, rng) -> None:
    """Same as B2/C1 (Qt pump active) but with NO slot connected to
    position_ready -- isolates the emit-and-cross-thread-marshal-into-a-slot
    step from the raw computation. Compare against B2/C1's crash rate: if C3
    stays clean while C1 doesn't, the emit/marshal step is implicated, not
    just the SciPy computation."""
    app = _make_qapp()
    for i in range(batch_size):
        cache = _make_cache_with_slot(max_workers=2, connect_slot=False)
        boards = rng.sample(positions, k=2)
        _heartbeat(f"C3 repeat={i} submit (no slot connected)")
        futures = _submit_concurrent(cache, boards)
        _wait_until_done(app, futures)
        cache._executor.shutdown(wait=True)
        _heartbeat(f"C3 repeat={i} done (no slot connected)")


# ---------------------------------------------------------------------------
# Axis D -- request()'s check-then-insert dict race (a different failure
# mode than the AV: RuntimeError / duplicate-submission / inconsistent
# state, not a crash -- reported separately)
# ---------------------------------------------------------------------------


def run_config_D1(positions, batch_size, rng) -> None:
    app = _make_qapp()
    inconsistencies = 0
    board = positions[0]
    for i in range(batch_size):
        cache = _make_cache_with_slot(max_workers=2)
        fens = set()
        try:
            for _ in range(20):
                fens.add(cache.request(board))
        except Exception as exc:  # noqa: BLE001 -- diagnostic harness, must not hide any exception type
            inconsistencies += 1
            _heartbeat(f"D1 repeat={i} EXCEPTION {exc!r}")
        if len(fens) != 1:
            inconsistencies += 1
            _heartbeat(f"D1 repeat={i} INCONSISTENT fens={fens}")
        fen = board.board_fen()
        deadline = time.monotonic() + 10.0
        while cache.get(fen).state != CacheEntryState.READY and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.001)
        cache._executor.shutdown(wait=True)
    _heartbeat(f"D1 complete inconsistencies={inconsistencies}/{batch_size}")


# ---------------------------------------------------------------------------
# Axis E -- interpreter/Qt shutdown ordering
# ---------------------------------------------------------------------------


def run_config_E1(positions, batch_size, rng) -> None:
    """Exit each repeat while its Future is still not-yet-done -- checks
    whether tearing down (dropping the reference / process exit) mid-native-
    call is itself a crash trigger, independent of any concurrent-access
    race. Deliberately never waits for completion or calls shutdown()."""
    app = _make_qapp()
    for i in range(batch_size):
        cache = _make_cache_with_slot(max_workers=1)
        board = rng.choice(positions)
        _heartbeat(f"E1 repeat={i} submit, will not wait")
        cache.request(board)
        _pump_events(app, 0.01)
        _heartbeat(f"E1 repeat={i} moving on with future in flight")


# ---------------------------------------------------------------------------
# Dispatch + runner
# ---------------------------------------------------------------------------

CONFIGS = {
    "A1": run_config_A1,
    "A2": run_config_A2,
    "A3": run_config_A3,
    "A4": run_config_A4,
    "A5": run_config_A5,
    "B1": run_config_B1,
    "B2": run_config_B2,
    "B3": run_config_B3,
    "B2-spline-only": run_config_B2_spline_only,
    "B2-gaussian-only": run_config_B2_gaussian_only,
    "C1": run_config_C1,
    "C2": run_config_C2,
    "C3": run_config_C3,
    "D1": run_config_D1,
    "E1": run_config_E1,
    "E1-no-slot": run_config_E1_no_slot,
    "E1-fixed": run_config_E1_fixed,
}


def worker_main(config_name: str, batch_size: int, seed_offset: int) -> None:
    faulthandler.enable()
    rng = random.Random(seed_offset)
    _heartbeat(f"worker start config={config_name} batch_size={batch_size} seed_offset={seed_offset}")
    CONFIGS[config_name](POSITION_POOL, batch_size, rng)
    _heartbeat(f"worker complete config={config_name}")


def supervise(config_name: str, repeats: int, batch_size: int) -> None:
    n_batches = (repeats + batch_size - 1) // batch_size
    crashes = 0
    completed_repeats = 0
    stage_histogram: dict[str, int] = {}
    for batch_index in range(n_batches):
        this_batch = min(batch_size, repeats - batch_index * batch_size)
        cmd = [
            sys.executable,
            __file__,
            "--config",
            config_name,
            "--batch-size",
            str(this_batch),
            "--seed-offset",
            str(batch_index),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            crashes += 1
            print(f"[{config_name}] batch {batch_index}: TIMEOUT (treated as failure)")
            continue

        heartbeats = [line for line in result.stdout.splitlines() if line.startswith("HEARTBEAT")]
        last_heartbeat = heartbeats[-1] if heartbeats else None
        if result.returncode != 0:
            crashes += 1
            print(
                f"[{config_name}] batch {batch_index}: CRASH exit={result.returncode} "
                f"last_heartbeat={last_heartbeat!r}"
            )
            if last_heartbeat:
                stage_histogram[last_heartbeat] = stage_histogram.get(last_heartbeat, 0) + 1
            tail = result.stderr[-2000:]
            if tail:
                print(tail)
        else:
            completed_repeats += this_batch

    print(
        f"\n=== {config_name}: {crashes}/{n_batches} batches crashed "
        f"({crashes / n_batches * 100:.1f}%), completed_repeats={completed_repeats}/{repeats} ==="
    )
    if stage_histogram:
        print("Last-heartbeat-before-crash histogram:")
        for stage, count in sorted(stage_histogram.items(), key=lambda kv: -kv[1]):
            print(f"  {count:3d}  {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--repeats", type=int, default=None, help="total repeats (--supervise mode only)")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--supervise", action="store_true", help="spawn one subprocess per batch and tally exit codes")
    args = parser.parse_args()

    if args.supervise:
        if args.repeats is None:
            parser.error("--supervise requires --repeats")
        supervise(args.config, args.repeats, args.batch_size)
    else:
        worker_main(args.config, args.batch_size, args.seed_offset)


if __name__ == "__main__":
    main()
