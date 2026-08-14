import chess
import chess.pgn
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from audio.backend import FakeAudioBackend
from audio.engine import AudioEngine
from audio.voices import build_default_voice_registry
from desktop_app.audio_controller import AudioController
from desktop_app.board_panel import BoardPanel
from desktop_app.gl_canvas import MathCanvas
from desktop_app.position_cache import PositionCache
from desktop_app.scrub_controller import ScrubController
from desktop_app.session_state import SessionState
from desktop_app.timeline_panel import TimelinePanel
from desktop_app.transition_controller import TransitionController

STRIP_WIDTH = 400

# Every test in this file drains pending Qt timers with qtbot.wait(10)
# before returning -- same reason as tests/test_desktop_app_timeline_scrub.py:
# none of the objects built here have a Qt parent, so a still-pending
# QTimer.singleShot left scheduled when the test function returns can fire
# against an already-GC'd object during pytest-qt's teardown pump.


class _FakeTransitionController:
    """
    Minimal stand-in exposing only what `_ScrubStrip`'s Rule 6 check reads
    (`.is_animating`) -- avoids needing to trigger and time out a real
    ~500ms QTimer-driven animation just to prove the audio gate follows the
    same boolean the visual gate already reads.
    """

    def __init__(self) -> None:
        self.is_animating = False


def _wired_panel(moves: list[str], transition_controller=None):
    session_state = SessionState(chess.pgn.Game())
    for uci in moves:
        session_state.make_move(chess.Move.from_uci(uci))

    position_cache = PositionCache()
    canvas = MathCanvas()
    board_panel = BoardPanel(session_state)
    if transition_controller is None:
        transition_controller = TransitionController(
            canvas=canvas,
            session_state=session_state,
            position_cache=position_cache,
            on_settled=lambda fen: None,
        )
    scrub_controller = ScrubController(canvas=canvas, position_cache=position_cache)

    backend = FakeAudioBackend()
    audio_engine = AudioEngine(backend, build_default_voice_registry(), blocksize=64)
    audio_engine.start()
    audio_controller = AudioController(session_state, audio_engine)

    panel = TimelinePanel(
        session_state,
        None,
        board_panel=board_panel,
        transition_controller=transition_controller,
        scrub_controller=scrub_controller,
        audio_controller=audio_controller,
    )
    panel._scrub_strip.resize(STRIP_WIDTH, panel._scrub_strip.FIXED_HEIGHT)

    return panel, session_state, board_panel, transition_controller, scrub_controller, audio_controller, audio_engine


def _mouse_event(kind, x: float, y: float, button, buttons) -> QMouseEvent:
    local = QPointF(x, y)
    return QMouseEvent(kind, local, local, button, buttons, Qt.KeyboardModifier.NoModifier)


def _press(strip, x: float, y: float = 5.0) -> None:
    strip.mousePressEvent(
        _mouse_event(QEvent.Type.MouseButtonPress, x, y, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    )


def _move(strip, x: float, y: float = 5.0) -> None:
    strip.mouseMoveEvent(
        _mouse_event(QEvent.Type.MouseMove, x, y, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    )


def _release(strip, x: float, y: float = 5.0) -> None:
    strip.mouseReleaseEvent(
        _mouse_event(QEvent.Type.MouseButtonRelease, x, y, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    )


# ---------------------------------------------------------
# visual/audio scrub stay synchronized
# ---------------------------------------------------------


def test_press_begins_both_visual_and_audio_scrub_together(qapp, qtbot):
    panel, session_state, board_panel, transition_controller, scrub_controller, audio_controller, audio_engine = (
        _wired_panel(["e2e4", "e7e5"])
    )
    strip = panel._scrub_strip

    _press(strip, 0)

    assert scrub_controller.is_scrubbing is True
    assert audio_controller.is_scrubbing is True
    qtbot.wait(10)


def test_release_ends_both_visual_and_audio_scrub_together(qapp, qtbot):
    panel, session_state, board_panel, transition_controller, scrub_controller, audio_controller, audio_engine = (
        _wired_panel(["e2e4", "e7e5"])
    )
    strip = panel._scrub_strip

    _press(strip, 0)
    _move(strip, 200)
    _release(strip, 200)

    assert scrub_controller.is_scrubbing is False
    assert audio_controller.is_scrubbing is False
    qtbot.wait(10)


def test_transition_animating_begins_neither_visual_nor_audio_scrub(qapp, qtbot):
    fake_transition_controller = _FakeTransitionController()
    panel, session_state, board_panel, transition_controller, scrub_controller, audio_controller, audio_engine = (
        _wired_panel(["e2e4", "e7e5"], transition_controller=fake_transition_controller)
    )
    strip = panel._scrub_strip
    fake_transition_controller.is_animating = True

    _press(strip, 0)

    assert scrub_controller.is_scrubbing is False
    assert audio_controller.is_scrubbing is False
    qtbot.wait(10)


def test_external_navigation_mid_drag_cancels_both_visual_and_audio_scrub(qapp, qtbot):
    panel, session_state, board_panel, transition_controller, scrub_controller, audio_controller, audio_engine = (
        _wired_panel(["e2e4", "e7e5", "g1f3"])
    )
    strip = panel._scrub_strip

    _press(strip, 0)
    _move(strip, 100)
    qtbot.wait(10)
    assert scrub_controller.is_scrubbing is True
    assert audio_controller.is_scrubbing is True

    session_state.undo()  # an external navigation, not the strip's own release

    assert scrub_controller.is_scrubbing is False
    assert audio_controller.is_scrubbing is False
    qtbot.wait(10)


# ---------------------------------------------------------
# scrub preview publishes without committing chess state
# ---------------------------------------------------------


def test_dragging_publishes_a_preview_state_without_moving_current_node(qapp, qtbot):
    panel, session_state, board_panel, transition_controller, scrub_controller, audio_controller, audio_engine = (
        _wired_panel(["e2e4", "e7e5"])
    )
    strip = panel._scrub_strip
    original_node = session_state.current_node

    _press(strip, 0)
    _move(strip, 150)
    qtbot.wait(10)

    assert session_state.current_node is original_node  # chess state untouched
    assert audio_engine._latest_state is not None  # but a preview was published

    _release(strip, 150)
    qtbot.wait(10)


# ---------------------------------------------------------
# capture/check crossed mid-drag: no trigger until a real committed release
# ---------------------------------------------------------


def test_capture_crossed_mid_drag_does_not_trigger_until_release(qapp, qtbot):
    panel, session_state, board_panel, transition_controller, scrub_controller, audio_controller, audio_engine = (
        _wired_panel(["e2e4", "d7d5"])
    )
    strip = panel._scrub_strip
    # active_path() only extends through SessionState's own `_active_child`
    # map (set by set_current_node), not through any child added via a raw
    # add_variation -- so the capture must actually be committed once (to
    # register it as the active child) and then undone, leaving it reachable
    # by scrub without being the current node.
    details = session_state.make_move(chess.Move.from_uci("e4d5"))
    assert details is not None
    capture_node = session_state.current_node
    session_state.undo()
    audio_engine._trigger_buffer.pop()  # discard the trigger the real commit above fired

    assert len(audio_engine._trigger_buffer) == 0

    active_path = session_state.active_path()
    assert capture_node in active_path  # sanity: the capture is reachable by scrub

    _press(strip, 0)
    for x in (50, 150, 250, 350, STRIP_WIDTH):
        _move(strip, x)
        qtbot.wait(1)
    qtbot.wait(10)

    assert len(audio_engine._trigger_buffer) == 0  # nothing queued from preview alone

    _release(strip, STRIP_WIDTH)  # release at the far end -- commits to the capture
    qtbot.wait(10)

    assert session_state.current_node is capture_node
    assert len(audio_engine._trigger_buffer) == 1  # exactly one trigger, from the real commit


# ---------------------------------------------------------
# branch-aware scrub through the real panel wiring
# ---------------------------------------------------------


def test_scrubbing_after_switching_branches_through_the_panel(qapp, qtbot):
    panel, session_state, board_panel, transition_controller, scrub_controller, audio_controller, audio_engine = (
        _wired_panel(["e2e4", "e7e5"])
    )
    strip = panel._scrub_strip

    session_state.undo()
    branch_node = session_state.make_move(chess.Move.from_uci("b8c6"))
    assert branch_node is not None
    qtbot.wait(10)

    _press(strip, 0)
    _move(strip, STRIP_WIDTH)  # drag to the far end, into the new branch
    qtbot.wait(10)

    assert audio_engine._latest_state is not None
    assert audio_engine._latest_state.segment_key[1] == session_state.current_node.board().board_fen()

    _release(strip, STRIP_WIDTH)
    qtbot.wait(10)


# ---------------------------------------------------------
# variation selector (Branch Exploration V1): no duplicate/stale audio
# ---------------------------------------------------------


def test_variation_selector_click_fires_exactly_one_audio_trigger(qapp, qtbot):
    panel, session_state, board_panel, transition_controller, scrub_controller, audio_controller, audio_engine = (
        _wired_panel(["e2e4"])
    )
    e4_node = session_state.current_node
    session_state.undo()
    session_state.make_move(chess.Move.from_uci("d2d4"))
    d4_node = session_state.current_node
    qtbot.wait(10)

    session_state.set_current_node(e4_node)
    qtbot.wait(10)
    # Drain the note-trigger buffer (fires on EVERY committed move, quiet or
    # not -- unlike _trigger_buffer, which is capture/check-only and stays
    # empty for e4/d4) so only the selector click below is being measured.
    while audio_engine._note_trigger_buffer.pop() is not None:
        pass

    panel._variation_next_buttons[e4_node].click()  # e4 -> d4, via the selector -- same
    # set_current_node() path as any other jump, so no separate audio wiring
    # for this control to duplicate or miss a trigger through.

    assert session_state.current_node is d4_node
    assert len(audio_engine._note_trigger_buffer) == 1  # exactly one trigger, no backlog/duplicate
    qtbot.wait(10)


# ---------------------------------------------------------
# close / shutdown mid-scrub: no crash, no stray publication, no thread leak
# ---------------------------------------------------------


def test_shutdown_mid_drag_stops_the_scrub_and_the_engine_cleanly(qapp, qtbot):
    panel, session_state, board_panel, transition_controller, scrub_controller, audio_controller, audio_engine = (
        _wired_panel(["e2e4", "e7e5"])
    )
    strip = panel._scrub_strip

    _press(strip, 0)
    _move(strip, 150)
    qtbot.wait(10)
    assert audio_controller.is_scrubbing is True

    # Mirrors MainWindow.closeEvent's ordering: scrub controller first, then
    # audio controller, before the position cache.
    scrub_controller.shutdown()
    audio_controller.shutdown()

    assert audio_controller.is_scrubbing is False
    assert audio_engine.is_shut_down is True
    assert audio_engine.is_running is False

    # A stray late mousemove after shutdown must not raise and must not
    # resurrect the scrub or the engine.
    _move(strip, 300)
    assert audio_controller.is_scrubbing is False
    assert audio_engine.is_running is False

    qtbot.wait(10)
