import threading

import chess
import chess.pgn
import pytest

from analysis.attack_influence import build_attack_influence_field
from analysis.source_field import build_source_field
from desktop_app.position_cache import CacheEntryState, PositionCache


@pytest.fixture(autouse=True)
def _shutdown_every_real_cache_after_each_test():
    """
    Stability investigation (see `desktop_app/position_cache.py`'s "Shutdown
    contract" docstring): this file is exempt from `tests/conftest.py`'s
    synchronous-executor mitigation specifically so its tests exercise a
    *real* `ThreadPoolExecutor` -- which means every `PositionCache()`
    constructed here previously leaked its worker thread for the rest of the
    pytest session (nothing called `.shutdown()`). Confirmed as a real,
    residual contributor to the original crash: with a second real-threading
    file added during this investigation, a full-suite run still crashed at
    a low rate (1/15) with two idle-but-never-joined worker threads from
    this file's own earlier tests still alive at the time. Tracking every
    real instance here and shutting it down at teardown -- purely cleanup,
    after each test's own assertions have already run -- closes that gap
    without touching what any test actually verifies.
    """
    instances: list[PositionCache] = []
    original_init = PositionCache.__init__

    def _tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        instances.append(self)

    PositionCache.__init__ = _tracking_init
    try:
        yield
    finally:
        PositionCache.__init__ = original_init
        for cache in instances:
            cache.shutdown()


def test_get_on_an_unrequested_position_is_missing():
    cache = PositionCache()

    entry = cache.get(chess.Board().board_fen())

    assert entry.state == CacheEntryState.MISSING
    assert entry.analysis is None


def test_request_transitions_missing_to_computing_immediately(qtbot):
    cache = PositionCache()
    board = chess.Board()

    fen = cache.request(board)

    # Synchronous with request() itself -- the caller must never observe
    # MISSING again once request() has returned, regardless of how fast the
    # background computation finishes.
    assert cache.get(fen).state in (CacheEntryState.COMPUTING, CacheEntryState.READY)

    # Polling, not qtbot.waitSignal: this trivial workload can finish (and
    # emit position_ready) before a signal listener attached after request()
    # would start observing it -- waitUntil tolerates that race, a fixed
    # waitSignal(...) block after the fact would not.
    qtbot.waitUntil(lambda: cache.get(fen).state == CacheEntryState.READY, timeout=2000)


def test_cache_key_is_derived_from_board_fen_not_ply_or_move_count():
    cache = PositionCache()

    # Two different move orders reaching the identical piece placement --
    # a transposition. The architecture (docs/interactive_ui.md Part 4.3)
    # keys on board_fen() (piece placement only) specifically so these share
    # one cache entry instead of two.
    board_a = chess.Board()
    board_a.push_san("Nf3")
    board_a.push_san("Nf6")
    board_a.push_san("Nc3")
    board_a.push_san("Nc6")

    board_b = chess.Board()
    board_b.push_san("Nc3")
    board_b.push_san("Nc6")
    board_b.push_san("Nf3")
    board_b.push_san("Nf6")

    assert board_a.board_fen() == board_b.board_fen()

    fen_a = cache.request(board_a)
    fen_b = cache.request(board_b)

    assert fen_a == fen_b


def test_cache_key_differs_for_different_positions():
    cache = PositionCache()

    board_a = chess.Board()
    board_b = chess.Board()
    board_b.push_san("e4")

    assert cache.request(board_a) != cache.request(board_b)


def test_two_branch_nodes_that_transpose_to_the_same_fen_share_one_cache_entry():
    """
    Branch Exploration V1 (approved plan, Section 10): two different
    GameNode branches -- reached via different move orders -- that transpose
    to the identical piece placement correctly and intentionally share one
    PositionCache entry, exactly like two move orders in a single line
    already do (test_cache_key_is_derived_from_board_fen_not_ply_or_move_count).
    Documents this as accepted, deliberate behavior for branching, not a bug
    branching newly introduces -- PositionCache stays FEN-keyed, with no
    node-identity component, per the approved plan.
    """
    cache = PositionCache()

    root = chess.pgn.Game()
    branch_a = root.add_variation(chess.Move.from_uci("g1f3"))
    branch_a = branch_a.add_variation(chess.Move.from_uci("g8f6"))
    branch_a = branch_a.add_variation(chess.Move.from_uci("b1c3"))
    branch_a = branch_a.add_variation(chess.Move.from_uci("b8c6"))

    branch_b = root.add_variation(chess.Move.from_uci("b1c3"))
    branch_b = branch_b.add_variation(chess.Move.from_uci("b8c6"))
    branch_b = branch_b.add_variation(chess.Move.from_uci("g1f3"))
    branch_b = branch_b.add_variation(chess.Move.from_uci("g8f6"))

    assert branch_a is not branch_b
    assert branch_a.board().board_fen() == branch_b.board().board_fen()

    fen_a = cache.request(branch_a.board())
    fen_b = cache.request(branch_b.board())

    assert fen_a == fen_b
    assert cache.get(fen_a) is cache.get(fen_b)


def test_duplicate_requests_do_not_trigger_duplicate_computation(qtbot):
    call_count = 0
    call_count_lock = threading.Lock()

    def counting_builder(board: chess.Board):
        nonlocal call_count
        with call_count_lock:
            call_count += 1
        return build_attack_influence_field(board)

    cache = PositionCache(builder=counting_builder)
    board = chess.Board()

    fen_1 = cache.request(board)
    fen_2 = cache.request(board)
    fen_3 = cache.request(board)

    qtbot.waitUntil(lambda: cache.get(fen_1).state == CacheEntryState.READY, timeout=2000)
    qtbot.wait(50)  # let any (incorrect) duplicate submissions also finish

    assert fen_1 == fen_2 == fen_3
    assert call_count == 1


def test_background_completion_publishes_the_correct_fen_and_matches_direct_call(qtbot):
    cache = PositionCache()
    board = chess.Board()
    expected = build_attack_influence_field(board)

    with qtbot.waitSignal(cache.position_ready, timeout=2000) as blocker:
        fen = cache.request(board)

    assert blocker.args[0] == fen

    entry = cache.get(fen)
    assert entry.state == CacheEntryState.READY
    assert entry.analysis.attack_influence_field.matrix == expected.matrix
    assert entry.analysis.attack_influence_field.balance == expected.balance
    assert entry.analysis.attack_influence_field.strongest_white_square == expected.strongest_white_square
    assert entry.analysis.attack_influence_field.strongest_black_square == expected.strongest_black_square


def test_cache_entry_carries_source_field(qtbot):
    """Milestone F: FullPositionAnalysis.source_field must survive the real
    background-thread request/publish path, not just a direct
    build_full_position_analysis call."""
    cache = PositionCache()
    board = chess.Board()
    expected = build_source_field(board)

    with qtbot.waitSignal(cache.position_ready, timeout=2000):
        fen = cache.request(board)

    entry = cache.get(fen)
    assert entry.state == CacheEntryState.READY
    assert entry.analysis.source_field.matrix == expected.matrix
    assert entry.analysis.source_field.balance == expected.balance


def test_background_computation_runs_off_the_calling_thread(qtbot):
    calling_thread = threading.current_thread()
    worker_threads = []

    def recording_builder(board: chess.Board):
        worker_threads.append(threading.current_thread())
        return build_attack_influence_field(board)

    cache = PositionCache(builder=recording_builder)

    with qtbot.waitSignal(cache.position_ready, timeout=2000):
        cache.request(chess.Board())

    assert len(worker_threads) == 1
    assert worker_threads[0] is not calling_thread


def test_state_sequence_never_regresses_from_ready_back_to_computing(qtbot):
    cache = PositionCache()
    board = chess.Board()

    observed_states = [cache.get(board.board_fen()).state]

    def record_state(_fen):
        observed_states.append(cache.get(board.board_fen()).state)

    cache.position_ready.connect(record_state)

    with qtbot.waitSignal(cache.position_ready, timeout=2000):
        cache.request(board)

    assert observed_states == [
        CacheEntryState.MISSING,
        CacheEntryState.READY,
    ]
