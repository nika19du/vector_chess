"""
Phase 5f.5: AudioMixerPanel widget tests -- construction against the
Voice Registry, master gain slider, mute/solo checkboxes, and Audio
ON/OFF, all asserted as calls into AudioController/AudioEngine's own
authoritative API (never a second copy of mixer state). Uses a real
AudioEngine + FakeAudioBackend + real AudioController, matching
tests/test_desktop_app_audio_scrub_integration.py's own "wire the real
objects, no hardware" convention.
"""

import chess
import chess.pgn
import pytest

from audio.backend import FakeAudioBackend
from audio.engine import AudioEngine
from audio.voices import build_default_voice_registry
from desktop_app.audio_controller import AudioController
from desktop_app.audio_mixer_panel import UNAVAILABLE_STATUS_TEXT, AudioMixerPanel
from desktop_app.session_state import SessionState


def _wired_panel(qapp, *, fail_on_open: bool = False):
    session_state = SessionState(chess.pgn.Game())
    backend = FakeAudioBackend(fail_on_open=fail_on_open)
    registry = build_default_voice_registry()
    engine = AudioEngine(backend, registry, blocksize=64)
    engine.start()
    controller = AudioController(session_state, engine)
    panel = AudioMixerPanel(controller, engine, registry, None)
    return panel, session_state, controller, engine, registry


# ---------------------------------------------------------
# Construction / rows -- registry-driven, exactly five voices
# ---------------------------------------------------------


def test_panel_has_exactly_five_voice_rows(qapp):
    panel, *_ = _wired_panel(qapp)

    assert set(panel._mute_checkboxes) == {"harmony", "melody", "accent", "drone", "space"}
    assert set(panel._solo_checkboxes) == {"harmony", "melody", "accent", "drone", "space"}
    assert len(panel._mute_checkboxes) == 5
    assert len(panel._solo_checkboxes) == 5


def test_melody_harmony_accent_controls_are_enabled(qapp):
    panel, *_ = _wired_panel(qapp)

    for voice_id in ("melody", "harmony", "accent"):
        assert panel._mute_checkboxes[voice_id].isEnabled() is True
        assert panel._solo_checkboxes[voice_id].isEnabled() is True


def test_drone_and_space_controls_are_visible_but_disabled(qapp):
    panel, *_ = _wired_panel(qapp)

    for voice_id in ("drone", "space"):
        assert panel._mute_checkboxes[voice_id].isEnabled() is False
        assert panel._solo_checkboxes[voice_id].isEnabled() is False


def test_inactive_voice_checkboxes_are_not_wired_to_the_controller(qapp):
    panel, session_state, controller, engine, registry = _wired_panel(qapp)

    panel._mute_checkboxes["drone"].setChecked(True)
    panel._solo_checkboxes["space"].setChecked(True)

    assert "drone" not in dict(controller.voice_mute)
    assert "space" not in dict(controller.voice_solo)


# ---------------------------------------------------------
# Master gain -- sensible range, deterministic mapping, no amplification
# ---------------------------------------------------------


def test_master_slider_initial_value_reflects_controller_gain(qapp):
    panel, session_state, controller, engine, registry = _wired_panel(qapp)

    assert panel._master_slider.value() == 100  # controller default gain=1.0


def test_master_slider_range_never_amplifies_beyond_unity(qapp):
    panel, *_ = _wired_panel(qapp)

    assert panel._master_slider.minimum() == 0
    assert panel._master_slider.maximum() == 100  # slider max 100 -> gain 1.0, never more


def test_moving_master_slider_updates_controller_gain_deterministically(qapp):
    panel, session_state, controller, engine, registry = _wired_panel(qapp)

    panel._master_slider.setValue(40)

    assert controller.master_gain == pytest.approx(0.4)


def test_master_slider_at_zero_and_max_map_exactly(qapp):
    panel, session_state, controller, engine, registry = _wired_panel(qapp)

    panel._master_slider.setValue(0)
    assert controller.master_gain == pytest.approx(0.0)

    panel._master_slider.setValue(100)
    assert controller.master_gain == pytest.approx(1.0)


# ---------------------------------------------------------
# Mute / Solo
# ---------------------------------------------------------


def test_toggling_mute_checkbox_calls_controller_set_voice_mute(qapp):
    panel, session_state, controller, engine, registry = _wired_panel(qapp)

    panel._mute_checkboxes["melody"].setChecked(True)

    assert dict(controller.voice_mute) == {"melody": True}


def test_unchecking_mute_checkbox_clears_it(qapp):
    panel, session_state, controller, engine, registry = _wired_panel(qapp)
    panel._mute_checkboxes["melody"].setChecked(True)

    panel._mute_checkboxes["melody"].setChecked(False)

    assert dict(controller.voice_mute) == {"melody": False}


def test_toggling_solo_checkbox_calls_controller_set_voice_solo(qapp):
    panel, session_state, controller, engine, registry = _wired_panel(qapp)

    panel._solo_checkboxes["harmony"].setChecked(True)

    assert dict(controller.voice_solo) == {"harmony": True}


def test_multiple_solo_checkboxes_can_be_checked_simultaneously(qapp):
    panel, session_state, controller, engine, registry = _wired_panel(qapp)

    panel._solo_checkboxes["melody"].setChecked(True)
    panel._solo_checkboxes["harmony"].setChecked(True)

    assert dict(controller.voice_solo) == {"melody": True, "harmony": True}


def test_mute_and_solo_on_the_same_voice_both_reach_the_controller(qapp):
    """
    Panel-level plumbing only -- MUTE-wins precedence itself is an
    AudioEngine playback concern (see tests/test_audio_engine.py), not
    something this widget decides; it must simply forward both flags
    faithfully.
    """

    panel, session_state, controller, engine, registry = _wired_panel(qapp)

    panel._mute_checkboxes["melody"].setChecked(True)
    panel._solo_checkboxes["melody"].setChecked(True)

    assert dict(controller.voice_mute) == {"melody": True}
    assert dict(controller.voice_solo) == {"melody": True}


# ---------------------------------------------------------
# Audio ON/OFF
# ---------------------------------------------------------


def test_enabled_checkbox_initial_state_reflects_engine_is_running(qapp):
    panel, session_state, controller, engine, registry = _wired_panel(qapp)

    assert panel._enabled_checkbox.isChecked() == engine.is_running


def test_unchecking_enabled_checkbox_stops_the_engine(qapp):
    panel, session_state, controller, engine, registry = _wired_panel(qapp)
    assert engine.is_running is True

    panel._enabled_checkbox.setChecked(False)

    assert engine.is_running is False


def test_rechecking_enabled_checkbox_resumes_the_engine(qapp):
    panel, session_state, controller, engine, registry = _wired_panel(qapp)
    panel._enabled_checkbox.setChecked(False)

    panel._enabled_checkbox.setChecked(True)

    assert engine.is_running is True


def test_repeated_on_off_cycling_never_opens_a_second_stream(qapp):
    panel, session_state, controller, engine, registry = _wired_panel(qapp)
    backend = engine._backend

    for _ in range(10):
        panel._enabled_checkbox.setChecked(False)
        panel._enabled_checkbox.setChecked(True)

    assert len(backend.streams) == 1


def test_repeated_on_off_cycling_preserves_audio_controller_mixer_state(qapp):
    panel, session_state, controller, engine, registry = _wired_panel(qapp)
    controller.set_voice_mute("melody", True)
    controller.set_master_gain(0.4)

    for _ in range(5):
        panel._enabled_checkbox.setChecked(False)
        panel._enabled_checkbox.setChecked(True)

    assert dict(controller.voice_mute) == {"melody": True}
    assert controller.master_gain == pytest.approx(0.4)


def test_off_freezes_engine_state_rather_than_clearing_it(qapp):
    """
    OFF is defined as AudioEngine.stop() (a real pause) -- not a
    master-gain-to-zero hack -- so it must not touch the controller's
    own mixer state or the engine's published state at all.
    """

    panel, session_state, controller, engine, registry = _wired_panel(qapp)
    session_state.make_move(chess.Move.from_uci("e2e4"))
    latest_before = engine._latest_state

    panel._enabled_checkbox.setChecked(False)

    assert engine._latest_state is latest_before  # unchanged, not cleared


# ---------------------------------------------------------
# Device unavailable
# ---------------------------------------------------------


def test_unavailable_backend_disables_all_controls_at_construction(qapp, qtbot):
    panel, session_state, controller, engine, registry = _wired_panel(qapp, fail_on_open=True)
    qtbot.addWidget(panel)
    panel.show()  # isVisible() below reflects real Qt show-state, not just an internal flag

    assert engine.is_available is False
    assert panel._enabled_checkbox.isEnabled() is False
    assert panel._enabled_checkbox.isChecked() is False
    assert panel._master_slider.isEnabled() is False
    for voice_id in ("melody", "harmony", "accent"):
        assert panel._mute_checkboxes[voice_id].isEnabled() is False
        assert panel._solo_checkboxes[voice_id].isEnabled() is False
    assert panel._status_label.text() == UNAVAILABLE_STATUS_TEXT
    assert panel._status_label.isVisible() is True


def test_status_label_hidden_when_device_available(qapp):
    panel, *_ = _wired_panel(qapp)

    assert panel._status_label.isVisible() is False


def test_unavailable_panel_construction_does_not_raise(qapp):
    # The construction itself (in _wired_panel) is the assertion here --
    # this test exists to name that expectation explicitly.
    panel, *_ = _wired_panel(qapp, fail_on_open=True)
    assert panel is not None


def test_a_start_failure_via_the_toggle_marks_the_panel_unavailable(qapp, qtbot, monkeypatch):
    panel, session_state, controller, engine, registry = _wired_panel(qapp)
    qtbot.addWidget(panel)
    panel.show()
    panel._enabled_checkbox.setChecked(False)  # go OFF first
    monkeypatch.setattr(engine, "start", lambda: False)  # simulate a later failure

    panel._enabled_checkbox.setChecked(True)  # attempt to turn back ON

    assert panel._enabled_checkbox.isChecked() is False
    assert panel._enabled_checkbox.isEnabled() is False
    assert panel._status_label.isVisible() is True
