"""
Phase 5f.5: MainWindow-level coverage for the Live Audio mixer panel --
construction, usability during a real scrub gesture, a repeated
ON/OFF + mute/solo + scrub lifecycle stress pass, and clean shutdown.

Uses the FakeAudioBackend MainWindow gets by default in tests
(tests/conftest.py's autouse `_default_main_window_to_a_fake_audio_
backend_in_tests` fixture) -- this file is not in the real-hardware
exemption set, matching every other MainWindow test file except
test_desktop_app_main_window_audio_scrub.py.
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


def test_main_window_constructs_with_a_mixer_panel(qapp, qtbot):
    window = _ready_window(qtbot, ["e2e4"])

    assert window.audio_mixer_panel is not None
    assert set(window.audio_mixer_panel._mute_checkboxes) == {
        "harmony",
        "melody",
        "accent",
        "pulse",
        "drone",
        "space",
    }

    window.close()
    qtbot.wait(10)


def test_mixer_controls_remain_usable_during_a_scrub_and_do_not_cancel_it(qapp, qtbot):
    window = _ready_window(qtbot, ["e2e4", "e7e5"])
    strip = window.timeline_panel._scrub_strip
    mixer = window.audio_mixer_panel

    _press(strip, 0)
    _move(strip, 150)
    qtbot.wait(10)
    assert window.audio_controller.is_scrubbing is True

    mixer._mute_checkboxes["melody"].setChecked(True)
    mixer._master_slider.setValue(50)

    assert window.audio_controller.is_scrubbing is True  # unaffected

    _release(strip, 150)
    qtbot.wait(10)
    window.close()
    qtbot.wait(10)


def test_mixer_change_during_scrub_does_not_mutate_current_node(qapp, qtbot):
    window = _ready_window(qtbot, ["e2e4", "e7e5"])
    strip = window.timeline_panel._scrub_strip
    mixer = window.audio_mixer_panel
    original_node = window.session_state.current_node

    _press(strip, 0)
    _move(strip, 150)
    qtbot.wait(10)

    mixer._solo_checkboxes["harmony"].setChecked(True)

    assert window.session_state.current_node is original_node

    _release(strip, 150)
    qtbot.wait(10)
    window.close()
    qtbot.wait(10)


def test_mixer_change_during_scrub_does_not_recompute_analysis(qapp, qtbot, monkeypatch):
    window = _ready_window(qtbot, ["e2e4", "e7e5"])
    strip = window.timeline_panel._scrub_strip
    mixer = window.audio_mixer_panel

    import desktop_app.audio_controller as audio_controller_module

    call_count = {"n": 0}
    real_analyze_position = audio_controller_module.analyze_position

    def counting(*args, **kwargs):
        call_count["n"] += 1
        return real_analyze_position(*args, **kwargs)

    monkeypatch.setattr(audio_controller_module, "analyze_position", counting)

    _press(strip, 0)
    _move(strip, 150)
    qtbot.wait(10)
    baseline = call_count["n"]

    mixer._mute_checkboxes["melody"].setChecked(True)
    mixer._solo_checkboxes["harmony"].setChecked(True)
    mixer._master_slider.setValue(20)

    assert call_count["n"] == baseline  # unchanged -- mixer changes never recompute

    _release(strip, 150)
    qtbot.wait(10)
    window.close()
    qtbot.wait(10)


def test_repeated_on_off_mute_solo_scrub_lifecycle_stress_does_not_crash_or_leak(qapp, qtbot):
    window = _ready_window(qtbot, ["e2e4", "e7e5", "g1f3"])
    strip = window.timeline_panel._scrub_strip
    mixer = window.audio_mixer_panel

    for cycle in range(5):
        mixer._enabled_checkbox.setChecked(False)
        mixer._enabled_checkbox.setChecked(True)
        mixer._mute_checkboxes["melody"].setChecked(cycle % 2 == 0)
        mixer._solo_checkboxes["harmony"].setChecked(cycle % 2 == 1)
        mixer._master_slider.setValue(30 + cycle * 10)

        _press(strip, 0)
        _move(strip, 100 + cycle * 20)
        qtbot.wait(1)
        _release(strip, 100 + cycle * 20)
        qtbot.wait(1)

    assert window.audio_engine.is_running is True
    assert len(window.audio_engine._backend.streams) == 1  # never reopened
    assert window.audio_controller.is_scrubbing is False

    window.close()
    qtbot.wait(10)
    assert window.audio_engine.is_shut_down is True


def test_close_while_audio_active_shuts_down_cleanly(qapp, qtbot):
    window = _ready_window(qtbot, ["e2e4"])
    mixer = window.audio_mixer_panel
    assert mixer._enabled_checkbox.isChecked() is True

    window.close()
    qtbot.wait(10)

    assert window.audio_engine.is_shut_down is True
    assert window.audio_engine.is_running is False


def test_close_with_audio_off_still_shuts_down_the_engine_exactly_once(qapp, qtbot, monkeypatch):
    window = _ready_window(qtbot, ["e2e4"])
    window.audio_mixer_panel._enabled_checkbox.setChecked(False)
    assert window.audio_engine.is_running is False

    shutdown_calls = {"n": 0}
    real_shutdown = window.audio_engine.shutdown

    def counting_shutdown():
        shutdown_calls["n"] += 1
        return real_shutdown()

    monkeypatch.setattr(window.audio_engine, "shutdown", counting_shutdown)

    window.close()
    qtbot.wait(10)

    assert shutdown_calls["n"] == 1
    assert window.audio_engine.is_shut_down is True
