"""
Phase 5f.3's required verification of the one-trigger-per-block engine
policy (built in 5f.2) against REALISTIC AudioController-driven move
sequences, not just synthetic AccentTrigger objects in isolation. Goal:
decide whether TriggerBuffer needs redesigning for real committed-move
traffic. Finding: it does not -- see the two scenarios below.
"""

import chess
import chess.pgn

from audio.backend import FakeAudioBackend
from audio.engine import DEFAULT_BLOCKSIZE, AccentTrigger, AudioEngine
from audio.voices import build_default_voice_registry
from desktop_app.audio_controller import AudioController
from desktop_app.session_state import SessionState


def _real_engine():
    backend = FakeAudioBackend()
    registry = build_default_voice_registry()
    engine = AudioEngine(backend, registry, blocksize=DEFAULT_BLOCKSIZE)
    engine.start()
    stream = backend.streams[-1]
    return engine, stream


# ---------------------------------------------------------
# Realistic play: audio playback runs continuously between moves (a
# human takes seconds between moves; the audio thread drains blocks
# every ~20ms regardless) -- no trigger is ever lost or misrepresented.
# ---------------------------------------------------------


def test_normal_paced_play_delivers_every_capture_and_check_trigger(qapp):
    session_state = SessionState(chess.pgn.Game())
    engine, stream = _real_engine()
    controller = AudioController(session_state, engine)

    # 1.e4 d5 2.exd5 Qxd5 3.Nc3 Qa5 4.Bb5+ -- 2 captures, 1 check,
    # verified move-by-move earlier in this phase's investigation.
    moves_and_expect_trigger = [
        ("e2e4", False),
        ("d7d5", False),
        ("e4d5", True),
        ("d8d5", True),
        ("b1c3", False),
        ("d5a5", False),
        ("f1b5", True),
    ]

    for move, expect_trigger in moves_and_expect_trigger:
        details = session_state.make_move(chess.Move.from_uci(move))
        assert details is not None, move

        # The very next audio block after a committed move drains at
        # most one pending trigger -- this is where a real capture/check
        # accent would actually start sounding.
        stream.render_block(64)
        if expect_trigger:
            assert engine._accent_state.active is True, f"{move} should have triggered an accent"

        # Simulate realistic spacing -- many blocks elapse before the
        # next move commits, exactly as real playback time would.
        for _ in range(50):
            stream.render_block(64)

        assert len(engine._trigger_buffer) == 0  # fully drained before the next move


def test_undo_redo_navigation_retriggers_correctly_without_backlog(qapp):
    session_state = SessionState(chess.pgn.Game())
    engine, stream = _real_engine()
    controller = AudioController(session_state, engine)

    session_state.make_move(chess.Move.from_uci("e2e4"))
    session_state.make_move(chess.Move.from_uci("d7d5"))
    session_state.make_move(chess.Move.from_uci("e4d5"))  # capture
    for _ in range(20):
        stream.render_block(64)
    assert len(engine._trigger_buffer) == 0

    session_state.undo()  # back to the quiet d7d5 node -- no new trigger
    stream.render_block(64)
    assert len(engine._trigger_buffer) == 0

    session_state.redo()  # forward again onto the capturing node -- retriggers
    stream.render_block(64)
    assert engine._accent_state.active is True


# ---------------------------------------------------------
# Pathological worst case: moves committed strictly faster than any
# block is ever drained (not realistic for human or even scripted
# real-time play, since the audio thread runs independently of the UI --
# included because the phase brief explicitly asks for it).
# ---------------------------------------------------------


def test_synthetic_burst_past_capacity_drops_oldest_not_newest_deterministically():
    engine, stream = _real_engine()

    for i in range(6):
        engine.push_event(AccentTrigger(kind=f"move-{i}", loudness=0.5))

    assert len(engine._trigger_buffer) == engine._trigger_buffer.capacity == 4

    drained = [engine._trigger_buffer.pop().kind for _ in range(4)]
    assert drained == ["move-2", "move-3", "move-4", "move-5"]  # 0 and 1 dropped, FIFO preserved
    assert engine._trigger_buffer.pop() is None


def test_rapid_committed_captures_with_zero_draining_stay_within_capacity(qapp):
    """
    Several real, committed, capturing moves with literally zero audio
    blocks rendered between them -- the buffer must never grow past its
    fixed capacity and must never raise.
    """

    session_state = SessionState(chess.pgn.Game())
    engine, stream = _real_engine()
    controller = AudioController(session_state, engine)

    moves = ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"]  # 2 captures, well under capacity
    for move in moves:
        details = session_state.make_move(chess.Move.from_uci(move))
        assert details is not None, move

    assert len(engine._trigger_buffer) == 2  # both captures preserved -- 2 < capacity=4
    assert len(engine._trigger_buffer) <= engine._trigger_buffer.capacity
