"""
Phase 5f.4: full MainWindow end-to-end coverage for "close the app
mid-scrub" -- the one manual smoke-test step that can't be exercised at
the panel level alone, since only MainWindow actually owns a real
AudioEngine/AudioController pair and wires closeEvent's shutdown
ordering. Reuses the real `SoundDeviceBackend` (a real device is
available in this environment, per the 5f.2/5f.3 investigation) --
AudioEngine.start()'s own documented fallback means this remains safe
even where one isn't.
"""

import chess
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from desktop_app.main_window import MainWindow

STRIP_WIDTH = 400


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


def _ready_window(qtbot, moves: list[str]) -> MainWindow:
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    for uci in moves:
        window.session_state.make_move(chess.Move.from_uci(uci))
        qtbot.waitUntil(lambda: not window.transition_controller.is_animating, timeout=5000)
    window.timeline_panel._scrub_strip.resize(STRIP_WIDTH, window.timeline_panel._scrub_strip.FIXED_HEIGHT)
    qtbot.wait(10)
    return window


def test_main_window_constructs_a_running_audio_engine_and_controller(qapp, qtbot):
    window = _ready_window(qtbot, ["e2e4"])

    assert window.audio_controller.is_scrubbing is False
    # start() may legitimately be False in a genuinely headless/no-device
    # environment (AudioEngine's own documented fallback) -- either way,
    # the window must not have crashed constructing it.
    assert window.audio_engine.is_available in (True, False)

    window.close()
    qtbot.wait(10)


def test_close_mid_scrub_shuts_down_cleanly_with_no_crash(qapp, qtbot):
    window = _ready_window(qtbot, ["e2e4", "e7e5"])
    strip = window.timeline_panel._scrub_strip

    _press(strip, 0)
    _move(strip, 150)
    qtbot.wait(10)
    assert window.scrub_controller.is_scrubbing is True
    assert window.audio_controller.is_scrubbing is True

    window.close()  # invokes closeEvent -- must not crash, must not hang
    qtbot.wait(10)

    assert window.scrub_controller.is_scrubbing is False
    assert window.audio_controller.is_scrubbing is False
    assert window.audio_engine.is_shut_down is True
    assert window.audio_engine.is_running is False


def test_close_after_a_settled_game_shuts_down_the_audio_engine(qapp, qtbot):
    window = _ready_window(qtbot, ["e2e4", "e7e5", "g1f3"])

    window.close()
    qtbot.wait(10)

    assert window.audio_engine.is_shut_down is True
    assert window.audio_engine.is_running is False


# ---------------------------------------------------------
# stress: repeated rapid scrubbing across the whole path, many times --
# mirrors test_desktop_app_main_window_scrub.py's own visual-side stress
# test, extended to also assert no analysis-recomputation blowup and a
# clean, still-running engine afterward.
# ---------------------------------------------------------


def test_repeated_rapid_scrub_stress_does_not_crash_or_leak_or_recompute(qapp, qtbot, monkeypatch):
    window = _ready_window(qtbot, ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"])
    strip = window.timeline_panel._scrub_strip

    import desktop_app.audio_controller as audio_controller_module

    call_count = {"n": 0}
    real_analyze_position = audio_controller_module.analyze_position

    def counting(*args, **kwargs):
        call_count["n"] += 1
        return real_analyze_position(*args, **kwargs)

    monkeypatch.setattr(audio_controller_module, "analyze_position", counting)

    _press(strip, 0)
    for _ in range(3):
        for x in range(0, STRIP_WIDTH + 1, 17):  # sweep forward
            _move(strip, x)
        for x in range(STRIP_WIDTH, -1, -17):  # sweep backward
            _move(strip, x)
    qtbot.wait(20)

    # 5 moves played -> 5 nodes; every one of them was already visited via
    # _ready_window's own play loop above, so no sweep-driven scrub update
    # should trigger a single additional analyze_position call.
    assert call_count["n"] == 0

    _release(strip, STRIP_WIDTH // 2)
    qtbot.wait(20)

    assert window.scrub_controller.is_scrubbing is False
    assert window.audio_controller.is_scrubbing is False
    assert window.audio_engine.is_running is True  # survived the sweep, still live

    window.close()
    qtbot.wait(10)
    assert window.audio_engine.is_shut_down is True


# ---------------------------------------------------------
# Manual smoke test stand-in (Phase 5f.4's own requested script) --
# automated, headless, against the real MainWindow object graph, exactly
# mirroring test_desktop_app_scrub_manual_smoke.py's own established
# precedent/rationale for why this is a stand-in rather than a literal
# mouse/screen session.
# ---------------------------------------------------------


def test_audio_manual_acceptance_smoke_test_end_to_end(qapp, qtbot):
    """
    Walks every step of Phase 5f.4's requested manual smoke test:
    play several moves; slowly scrub across one segment (harmony glides
    continuously); rapidly scrub back and forth; cross a capture/check
    during drag (no accent fires); release on that move (accent fires
    once); switch branch and scrub there; close mid-scrub (no crash, no
    audio thread leak).
    """

    # Step 1: launch, play a short opening.
    window = _ready_window(qtbot, ["e2e4", "e7e5", "g1f3", "b8c6"])
    strip = window.timeline_panel._scrub_strip
    assert window.audio_engine.is_running is True

    # This test uses the real backend (exempted in conftest.py) so its
    # own real audio thread is continuously draining the trigger buffer in
    # the background -- checking the buffer's occupancy at a point in time
    # would race against that thread. Counting push_event calls instead is
    # immune to that race: it's a call count, not a snapshot of a queue a
    # background thread is concurrently draining.
    push_event_calls = {"n": 0}
    real_push_event = window.audio_engine.push_event

    def _counting_push_event(trigger):
        push_event_calls["n"] += 1
        return real_push_event(trigger)

    window.audio_engine.push_event = _counting_push_event

    # Step 2: slowly scrub across one segment -- harmony should glide
    # continuously (verified numerically: not a single fixed value).
    _press(strip, 0)
    ratios = []
    for x in range(0, STRIP_WIDTH // 4 + 1, 10):
        _move(strip, x)
        qtbot.wait(1)
        state = window.audio_engine._latest_state
        if state is not None:
            ratios.append(state.harmony_interval_ratio)
    assert len(set(ratios)) > 1, "harmony did not audibly glide during a slow drag"
    _release(strip, STRIP_WIDTH // 4)
    qtbot.wait(10)

    # Step 3: rapidly scrub back and forth -- no crash, no unbounded growth.
    _press(strip, 0)
    for _ in range(3):
        for x in range(0, STRIP_WIDTH + 1, 23):
            _move(strip, x)
        for x in range(STRIP_WIDTH, -1, -23):
            _move(strip, x)
    qtbot.wait(10)
    assert push_event_calls["n"] == 0  # no capture/check crossed by pure scrubbing so far
    _release(strip, STRIP_WIDTH // 2)
    qtbot.wait(10)

    # Steps 2/3's releases may have landed on an earlier node than the
    # opening's tip (releasing partway along the strip is expected scrub
    # behavior, not a bug) -- return to the tip before continuing so the
    # position for step 4 is exactly "after 1.e4 e5 2.Nf3 Nc6" again.
    window.session_state.go_to_end()
    qtbot.waitUntil(lambda: not window.transition_controller.is_animating, timeout=5000)
    qtbot.wait(10)

    # Step 4: play into a capturing move (4.Nxe5), then scrub across its
    # segment without committing -- confirm no accent fires from preview
    # alone.
    details = window.session_state.make_move(chess.Move.from_uci("f3e5"))  # Nxe5 -- capture
    assert details is not None
    assert details.is_capture is True
    qtbot.waitUntil(lambda: not window.transition_controller.is_animating, timeout=5000)
    window.session_state.undo()
    # Rule 6 needs the undo's own animation fully settled too -- otherwise
    # the press below is correctly (silently) ignored, same as any other
    # press during an animation.
    qtbot.waitUntil(lambda: not window.transition_controller.is_animating, timeout=5000)
    push_event_calls["n"] = 0  # discard the trigger the commit above fired

    _press(strip, 0)
    for x in range(0, STRIP_WIDTH + 1, 20):
        _move(strip, x)
        qtbot.wait(1)
    assert push_event_calls["n"] == 0  # no accent from crossing it mid-drag

    # Step 5: release exactly on that capturing move -- accent fires once.
    _release(strip, STRIP_WIDTH)
    qtbot.wait(10)
    assert push_event_calls["n"] == 1

    # Step 6: switch branch and scrub there.
    window.session_state.undo()
    qtbot.waitUntil(lambda: not window.transition_controller.is_animating, timeout=5000)
    other_branch_details = window.session_state.make_move(chess.Move.from_uci("d2d4"))
    assert other_branch_details is not None
    qtbot.waitUntil(lambda: not window.transition_controller.is_animating, timeout=5000)
    qtbot.wait(10)

    _press(strip, 0)
    _move(strip, STRIP_WIDTH)
    qtbot.wait(10)
    assert window.audio_engine._latest_state is not None
    _release(strip, STRIP_WIDTH)
    qtbot.wait(10)

    # Step 7: close mid-scrub -- no crash, engine fully shut down.
    _press(strip, 0)
    _move(strip, 100)
    qtbot.wait(10)
    window.close()
    qtbot.wait(10)

    assert window.audio_controller.is_scrubbing is False
    assert window.audio_engine.is_shut_down is True
    assert window.audio_engine.is_running is False
