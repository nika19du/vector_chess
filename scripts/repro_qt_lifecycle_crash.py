"""
Standalone (non-pytest) reproduction harness for the Qt/native lifecycle
access-violation crash observed during Phase 5e's closeout -- distinct from
the already-fixed `PositionCache` ThreadPoolExecutor-leak crash reproduced by
`scripts/repro_position_cache_crash.py` (that one's stack trace always shows
a background worker thread inside scipy code; this one's stack trace shows
the CURRENT thread stuck inside Qt's own event loop, `pytestqt.qt_compat.py`
`exec()`, with no background worker listed).

Investigation plan: see the approved plan file (Qt/Native Lifecycle Crash
Investigation). This script isolates each experiment as a runnable
configuration and is meant to be invoked repeatedly via `--supervise`, for
the same reason as the PositionCache harness: an access violation kills the
interpreter outright, so a pytest assertion (or a plain Python try/except)
can never observe it. The supervisor spawns one subprocess per attempt and
tallies exit codes; a nonzero/abnormal exit is the crash signal itself. Each
worker process loops up to `--cap` iterations, heartbeating every iteration,
so the LAST heartbeat printed before an abnormal exit tells us the
approximate iteration count ("N threshold") at which that attempt died.

Configurations:
  baseline      -- construct MainWindow(), never .show()'d (matches every
                    crash-producing test's exact shape: `window = MainWindow()`
                    + a settle-wait, nothing else), never closed, never
                    deleteLater'd. The smallest possible reproducer.
  cleanup       -- same as baseline, but close() + deleteLater() +
                    processEvents() after each instance (Experiment B).
  stub-canvas   -- same as baseline, but desktop_app.main_window.MathCanvas
                    is monkeypatched (for this process only) to a trivial
                    QWidget with no-op versions of MathCanvas's public
                    methods -- isolates whether QOpenGLWidget specifically is
                    required, or whether plain QWidget churn is sufficient
                    (Experiment C).
  disciplined   -- construct, explicit .show() (actually engages real
                    widget/paint realization, unlike the other configs),
                    settle-wait, close(), deleteLater(), processEvents()
                    (Experiment H).

Usage (run from the repo root, using the project's real .venv):

    .venv\\Scripts\\python.exe scripts\\repro_qt_lifecycle_crash.py \\
        --config baseline --cap 150 --attempts 5 --supervise

Worker mode (what --supervise invokes per attempt; not normally run by hand):

    .venv\\Scripts\\python.exe scripts\\repro_qt_lifecycle_crash.py \\
        --config baseline --cap 150 --seed-offset 0
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import subprocess
import sys
import time
from pathlib import Path

# Same rationale as scripts/repro_position_cache_crash.py and
# tests/conftest.py: the offscreen QPA platform is what every crash-producing
# test run actually used, and it's the platform this investigation is about.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess  # noqa: E402


def _heartbeat(label: str) -> None:
    print(f"HEARTBEAT {time.monotonic():.3f} {label}", flush=True)


def _make_qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _pump_until_settled(app, window, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while window.canvas._overlay_colors is None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.001)
    return window.canvas._overlay_colors is not None


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


def run_config_baseline(app, cap: int, seed_offset: int) -> None:
    from desktop_app.main_window import MainWindow

    for i in range(cap):
        _heartbeat(f"baseline iter={i} construct")
        window = MainWindow()
        settled = _pump_until_settled(app, window)
        _heartbeat(f"baseline iter={i} settled={settled} discarding (no close/deleteLater)")
        # Deliberately: no close(), no deleteLater(), no qtbot.addWidget --
        # `window` just goes out of scope at the next loop iteration, exactly
        # matching every crash-producing test's own shape.
    _heartbeat(f"baseline complete cap={cap}")


def run_config_cleanup(app, cap: int, seed_offset: int) -> None:
    from desktop_app.main_window import MainWindow

    for i in range(cap):
        _heartbeat(f"cleanup iter={i} construct")
        window = MainWindow()
        settled = _pump_until_settled(app, window)
        _heartbeat(f"cleanup iter={i} settled={settled} closing")
        window.close()
        window.deleteLater()
        app.processEvents()
        _heartbeat(f"cleanup iter={i} closed+deleteLater+drained")
    _heartbeat(f"cleanup complete cap={cap}")


class _StubCanvas:
    """
    A trivial QWidget replacement for MathCanvas, exposing the same public
    method names as no-ops, so MainWindow's own code (which calls these
    unconditionally) needs no changes to run against it. Not a QOpenGLWidget
    at all -- isolates whether GL specifically is required for the crash.
    """

    def __init__(self, parent=None):
        from PySide6.QtWidgets import QWidget

        self._widget = QWidget(parent)
        self._overlay_colors = None

    # Delegate the handful of attributes MainWindow/tests actually touch.
    def parent(self):
        return self._widget.parent()

    def set_layer_order(self, layer_ids):
        pass

    def set_overlay_colors(self, colors):
        self._overlay_colors = colors

    def set_overlay_enabled(self, enabled):
        pass

    def set_overlay_opacity(self, opacity):
        pass

    def set_layer_geometry(self, layer_id, geometries):
        pass

    def set_layer_enabled(self, layer_id, enabled):
        pass

    def set_layer_opacity(self, layer_id, opacity):
        pass

    # QWidget-ish pass-throughs some tests/layout code may touch.
    def __getattr__(self, name):
        return getattr(self._widget, name)


def run_config_stub_canvas(app, cap: int, seed_offset: int) -> None:
    import desktop_app.main_window as main_window_module

    original_math_canvas = main_window_module.MathCanvas
    main_window_module.MathCanvas = _StubCanvas
    try:
        for i in range(cap):
            _heartbeat(f"stub-canvas iter={i} construct")
            window = main_window_module.MainWindow()
            # No real overlay colors will ever be set by a GL-backed canvas
            # (there is none), but PositionCache/TransitionController still
            # run for real -- settle on a short, bounded pump instead of
            # waiting on a condition that can never become true.
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.001)
            _heartbeat(f"stub-canvas iter={i} discarding (no close/deleteLater)")
        _heartbeat(f"stub-canvas complete cap={cap}")
    finally:
        main_window_module.MathCanvas = original_math_canvas


def run_config_disciplined(app, cap: int, seed_offset: int) -> None:
    from desktop_app.main_window import MainWindow

    for i in range(cap):
        _heartbeat(f"disciplined iter={i} construct")
        window = MainWindow()
        window.show()
        settled = _pump_until_settled(app, window)
        _heartbeat(f"disciplined iter={i} settled={settled} shown, closing")
        window.close()
        window.deleteLater()
        app.processEvents()
        _heartbeat(f"disciplined iter={i} closed+deleteLater+drained")
    _heartbeat(f"disciplined complete cap={cap}")


def run_config_timelinepanel_churn(app, cap: int, seed_offset: int) -> None:
    """
    Ad-hoc config added mid-investigation: pure MainWindow/MathCanvas volume
    (baseline, up to 1600 across two attempts) and pure widget-count churn
    from pre-existing files (session_state.py + layer_panel.py + board_panel.py,
    67 tests, more total churn than the crashing combo) both failed to
    reproduce the crash. session_state.py + timeline_panel.py + main_window.py
    (52 tests) DID reproduce it, 3/3. This isolates the one concrete
    structural difference found by inspection: TimelinePanel._rebuild()
    deletes and immediately reconstructs its QPushButton children on every
    navigation (via deleteLater(), a DEFERRED delete -- not immediate), and
    its own tests are the only ones in the suite that call .click() (real
    synthetic event dispatch) rather than direct property setters. This
    config repeatedly drives that exact rebuild path via .click() on a bare
    TimelinePanel (no MainWindow, no MathCanvas at all) `cap` times, then
    constructs ONE MainWindow at the end to match the exact trigger shape
    already observed (crash always occurs on/soon after a MainWindow
    construction that follows TimelinePanel churn, never within
    TimelinePanel churn alone in the original pytest runs).
    """
    import chess as chess_module
    from desktop_app.session_state import SessionState
    from desktop_app.timeline_panel import TimelinePanel

    for i in range(cap):
        _heartbeat(f"timelinepanel-churn iter={i} construct")
        session_state = SessionState(chess_module.pgn.Game())
        panel = TimelinePanel(session_state, None)
        session_state.make_move(chess_module.Move.from_uci("e2e4"))
        session_state.make_move(chess_module.Move.from_uci("e7e5"))
        panel._first_button.click()
        panel._next_button.click()
        session_state.make_move(chess_module.Move.from_uci("g1f3"))
        # Undo back into a branch point and click a branch badge, mirroring
        # test_desktop_app_timeline_panel.py's own branch-switch tests --
        # each badge click triggers a full _rebuild() (deleteLater on every
        # existing button + immediate reconstruction).
        session_state.undo()
        session_state.undo()
        session_state.make_move(chess_module.Move.from_uci("d2d4"))
        new_branch_node = session_state.current_node
        if new_branch_node in panel._branch_badges:
            panel._branch_badges[new_branch_node].click()
            panel._branch_badges[new_branch_node].click()
        app.processEvents()
        # No close/deleteLater on `panel`/`session_state` -- matches the
        # real tests' own shape (nothing ever tears these down either).
    _heartbeat(f"timelinepanel-churn phase 1 complete cap={cap}, constructing MainWindow")

    from desktop_app.main_window import MainWindow

    window = MainWindow()
    settled = _pump_until_settled(app, window)
    _heartbeat(f"timelinepanel-churn MainWindow settled={settled}")


CONFIGS = {
    "baseline": run_config_baseline,
    "cleanup": run_config_cleanup,
    "stub-canvas": run_config_stub_canvas,
    "disciplined": run_config_disciplined,
    "timelinepanel-churn": run_config_timelinepanel_churn,
}


def worker_main(config_name: str, cap: int, seed_offset: int) -> None:
    faulthandler.enable()
    _heartbeat(f"worker start config={config_name} cap={cap} seed_offset={seed_offset}")
    app = _make_qapp()
    CONFIGS[config_name](app, cap, seed_offset)
    _heartbeat(f"worker complete config={config_name}")


def supervise(config_name: str, attempts: int, cap: int) -> None:
    crashes = 0
    last_iters_before_crash: list[int] = []
    for attempt_index in range(attempts):
        cmd = [
            sys.executable,
            __file__,
            "--config",
            config_name,
            "--cap",
            str(cap),
            "--seed-offset",
            str(attempt_index),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            crashes += 1
            print(f"[{config_name}] attempt {attempt_index}: TIMEOUT (treated as crash)")
            continue

        heartbeats = [line for line in result.stdout.splitlines() if line.startswith("HEARTBEAT")]
        last_heartbeat = heartbeats[-1] if heartbeats else None
        if result.returncode != 0:
            crashes += 1
            print(
                f"[{config_name}] attempt {attempt_index}: CRASH exit={result.returncode} "
                f"last_heartbeat={last_heartbeat!r}"
            )
            if last_heartbeat is not None:
                import re

                match = re.search(r"iter=(\d+)", last_heartbeat)
                if match:
                    last_iters_before_crash.append(int(match.group(1)))
            tail = result.stderr[-2000:]
            if tail:
                print(tail)
        else:
            print(f"[{config_name}] attempt {attempt_index}: clean (cap={cap} reached)")

    print(
        f"\n=== {config_name}: {crashes}/{attempts} attempts crashed "
        f"({crashes / attempts * 100:.1f}%) at cap={cap} ==="
    )
    if last_iters_before_crash:
        print(f"Iteration counts reached before each crash: {sorted(last_iters_before_crash)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--cap", type=int, default=150, help="max loop iterations per attempt")
    parser.add_argument("--attempts", type=int, default=None, help="independent subprocess attempts (--supervise mode only)")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--supervise", action="store_true", help="spawn one subprocess per attempt and tally exit codes")
    args = parser.parse_args()

    if args.supervise:
        if args.attempts is None:
            parser.error("--supervise requires --attempts")
        supervise(args.config, args.attempts, args.cap)
    else:
        worker_main(args.config, args.cap, args.seed_offset)


if __name__ == "__main__":
    main()
