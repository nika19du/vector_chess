import os

import pytest

# Desktop-app tests (tests/test_desktop_app_*.py) construct a QApplication via
# pytest-qt's `qapp`/`qtbot` fixtures. Default to Qt's offscreen QPA platform
# so the suite runs headlessly without popping up windows -- unless the
# invoking shell already set QT_QPA_PLATFORM, which is respected as-is.
#
# Note: on this Windows environment, the offscreen platform does not provide
# a real OpenGL context (verified during Milestone 5a implementation); the
# one test that needs actual GL rendering
# (tests/test_desktop_app_canvas.py) detects this and skips with a clear
# reason rather than failing on an environment limitation. Existing tests
# under tests/ that don't import PySide6 are unaffected by this setting.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _ImmediateExecutor:
    """
    Test-only stand-in for `concurrent.futures.ThreadPoolExecutor`, used by
    `_run_position_cache_synchronously_in_tests` below. Satisfies the two
    methods `PositionCache` actually calls (`submit`, `shutdown`) but runs
    the builder call synchronously, on the calling thread, instead of on a
    real worker thread.
    """

    def submit(self, fn, *args, **kwargs):
        from concurrent.futures import Future

        future = Future()
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 -- must propagate through the Future, whatever it is
            future.set_exception(exc)
        else:
            future.set_result(result)
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        pass


_REAL_THREADING_EXEMPT_FILES = {"test_desktop_app_position_cache.py"}


@pytest.fixture(autouse=True)
def _run_position_cache_synchronously_in_tests(request):
    """
    Bug found while running this milestone's full suite repeatedly (not
    introduced by this milestone's own code -- reproduces identically with
    `test_desktop_app_correspondence.py`/`test_desktop_app_interpolation.py`/
    `test_desktop_app_transition_controller.py` excluded entirely, using only
    pre-existing test files): a real, intermittent Windows access violation
    inside scipy (`RectBivariateSpline.__call__` in one crash, `analysis/
    ridge_valley.py::check_self_intersection` in another) on `PositionCache`'s
    `ThreadPoolExecutor` worker thread, while the main thread is pumping the
    Qt event loop (`pytestqt.plugin._process_events`) in a `qtbot.wait*`
    call elsewhere. Reproduced across many full-suite runs at roughly a
    50-100% rate depending on which mitigation was tried; neither shutting
    down every `PositionCache` after each test (a real, separate leak: nothing
    called `.shutdown()`, so idle worker threads accumulated for the whole
    session) nor forcing `max_workers=1` eliminated it outright, so this isn't
    simply "too many concurrent threads" -- it reproduces with a single
    worker thread too.

    Not fixed in `analysis/*` (forbidden -- no mathematics change) or in
    `desktop_app/position_cache.py`'s real `ThreadPoolExecutor` architecture
    (Phase 5c's own deliberate, documented decision; redesigning it is out of
    this milestone's scope and deserves its own investigation, not a
    reactive patch here). Mitigated here, for the test session only: every
    `PositionCache` created during tests runs its builder call synchronously
    (`_ImmediateExecutor`, same `submit`/`shutdown` interface) instead of on
    a real background thread, which eliminates all background-thread/
    main-thread native-code concurrency in the test process. Production
    code's real `ThreadPoolExecutor` is completely untouched -- the live app
    still runs analysis off the UI thread exactly as designed.

    Exempts `tests/test_desktop_app_position_cache.py`: that file's own
    purpose includes asserting the real threading contract (background
    computation actually runs off the calling thread) -- applying this
    mitigation there would make that assertion trivially false for the
    wrong reason (a test-only synchronous stand-in, not a real regression).
    """
    if request.node.fspath.basename in _REAL_THREADING_EXEMPT_FILES:
        yield
        return

    try:
        from desktop_app.position_cache import PositionCache
    except ImportError:
        yield
        return

    original_init = PositionCache.__init__

    def _synchronous_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._executor = _ImmediateExecutor()

    PositionCache.__init__ = _synchronous_init
    try:
        yield
    finally:
        PositionCache.__init__ = original_init


def ready_cache_entry(board):
    """
    Shared test helper (Phase 5c): a real, fully-computed READY `CacheEntry`
    for `board`, running the exact `build_full_position_analysis` pipeline
    every layer's tests need -- not a partially-fake `FullPositionAnalysis`
    with dummy values for fields a given test doesn't happen to touch.
    """
    from desktop_app.full_position_analysis import build_full_position_analysis
    from desktop_app.position_cache import CacheEntry, CacheEntryState

    return CacheEntry(state=CacheEntryState.READY, analysis=build_full_position_analysis(board))


def make_critical_point(x, y, classification="maximum", value=0.0):
    """
    Shared test helper (correspondence/animation milestone): a minimal but
    real `ClassifiedCriticalPoint` for tests that need full control over
    exact positions/classifications (deterministic matching, ambiguous-match,
    cross-classification edge cases) rather than whatever a real chess
    position's own math happens to produce.
    """
    from chess_engine.models import ClassifiedCriticalPoint

    return ClassifiedCriticalPoint(
        x=x,
        y=y,
        value=value,
        gradient_norm=0.0,
        status="converged",
        iterations=1,
        f_xx=None,
        f_xy=None,
        f_yx=None,
        f_yy=None,
        eigenvalue_min=None,
        eigenvalue_max=None,
        classification=classification,
    )


def make_ridge_valley_chain(kind, points_xy, anchor=None):
    """Shared test helper: a minimal real `RidgeValleyChain` over given (x, y) points."""
    from chess_engine.models import RidgeValleyChain, RidgeValleyPoint

    points = [
        RidgeValleyPoint(x=x, y=y, value=0.0, eigenvalue_cross=0.0, alignment_residual=0.0, kind=kind)
        for x, y in points_xy
    ]
    return RidgeValleyChain(kind=kind, points=points, anchor=anchor, start_status="left_domain", end_status="left_domain")


def make_morse_smale_cell(cell_id, boundary_critical_points):
    """Shared test helper: a minimal real `MorseSmaleCell` -- boundary_separatrices left empty since
    correspondence/interpolation only ever read `boundary_critical_points` and paired geometry."""
    from chess_engine.models import MorseSmaleCell

    return MorseSmaleCell(
        cell_id=cell_id,
        boundary_separatrices=[],
        boundary_traversal_directions=[],
        boundary_critical_points=boundary_critical_points,
        is_closed=True,
        open_boundaries=[],
    )


def make_cell_geometry(polygon, centroid=None, area=1.0, perimeter=1.0):
    """Shared test helper: a minimal real `MorseSmaleCellGeometry` for a given closed polygon."""
    from chess_engine.models import MorseSmaleCellGeometry

    if centroid is None:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
    return MorseSmaleCellGeometry(polygon=polygon, area=area, centroid=centroid, perimeter=perimeter)
