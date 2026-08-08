import threading

import chess

from analysis.attack_influence import build_attack_influence_field
from desktop_app.position_cache import CacheEntryState, PositionCache


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
