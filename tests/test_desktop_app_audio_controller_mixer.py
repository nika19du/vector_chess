"""
Phase 5f.5: AudioController's mixer-state ownership (master_gain/
voice_mute/voice_solo) -- the authoritative API the Mixer UI drives.
Uses the same `_SpyAudioEngine` shape as
tests/test_desktop_app_audio_controller.py so the controller's own
translation/sequencing logic is asserted independent of AudioEngine's
internal synthesis.
"""

import chess
import chess.pgn
import pytest

from audio.engine import AccentTrigger
from desktop_app.audio_controller import AudioController
from desktop_app.scrub import ScrubPosition
from desktop_app.session_state import SessionState


class _SpyAudioEngine:
    def __init__(self) -> None:
        self.published: list = []
        self.triggers: list[AccentTrigger] = []
        self.note_trigger_count = 0
        self.scrub_active_calls: list[bool] = []
        self.shutdown_calls = 0

    def publish(self, state) -> None:
        self.published.append(state)

    def push_event(self, trigger: AccentTrigger) -> None:
        self.triggers.append(trigger)

    def push_note_trigger(self) -> None:
        self.note_trigger_count += 1

    def set_scrub_active(self, active: bool) -> None:
        self.scrub_active_calls.append(active)

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _controller(qapp) -> tuple[SessionState, AudioController, _SpyAudioEngine]:
    session_state = SessionState(chess.pgn.Game())
    engine = _SpyAudioEngine()
    controller = AudioController(session_state, engine)
    return session_state, controller, engine


def _play(session_state: SessionState, moves: list[str]) -> None:
    for move_text in moves:
        details = session_state.make_move(chess.Move.from_uci(move_text))
        assert details is not None, f"expected {move_text} to be legal"


# ---------------------------------------------------------
# Defaults
# ---------------------------------------------------------


def test_default_mixer_state_is_unity_gain_no_mute_no_solo(qapp):
    _, controller, _ = _controller(qapp)

    assert controller.master_gain == 1.0
    assert dict(controller.voice_mute) == {}
    assert dict(controller.voice_solo) == {}


# ---------------------------------------------------------
# Master gain
# ---------------------------------------------------------


def test_set_master_gain_updates_the_property(qapp):
    _, controller, _ = _controller(qapp)

    controller.set_master_gain(0.5)

    assert controller.master_gain == pytest.approx(0.5)


def test_set_master_gain_before_any_move_does_not_publish(qapp):
    _, controller, engine = _controller(qapp)

    controller.set_master_gain(0.5)

    assert engine.published == []


def test_set_master_gain_after_a_move_republishes_with_the_new_gain(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4"])
    assert len(engine.published) == 1

    controller.set_master_gain(0.4)

    assert len(engine.published) == 2
    assert engine.published[-1].master_gain == pytest.approx(0.4)
    # Everything else about the state is unchanged -- a mixer tweak is
    # not a new note.
    assert engine.published[-1].pitch_hz == engine.published[0].pitch_hz
    assert engine.published[-1].segment_key == engine.published[0].segment_key


def test_set_master_gain_does_not_fire_a_note_trigger_or_accent(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4"])
    engine.note_trigger_count = 0
    engine.triggers.clear()

    controller.set_master_gain(0.3)

    assert engine.note_trigger_count == 0
    assert engine.triggers == []


# ---------------------------------------------------------
# Mute
# ---------------------------------------------------------


def test_set_voice_mute_republishes_with_the_new_mute_map(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4"])

    controller.set_voice_mute("melody", True)

    assert dict(engine.published[-1].voice_mute) == {"melody": True}


def test_set_voice_mute_preserves_other_mute_entries(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4"])

    controller.set_voice_mute("melody", True)
    controller.set_voice_mute("harmony", True)

    assert dict(controller.voice_mute) == {"melody": True, "harmony": True}
    assert dict(engine.published[-1].voice_mute) == {"melody": True, "harmony": True}


def test_unmuting_republishes_with_mute_cleared(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4"])
    controller.set_voice_mute("melody", True)

    controller.set_voice_mute("melody", False)

    assert dict(controller.voice_mute) == {"melody": False}
    assert dict(engine.published[-1].voice_mute) == {"melody": False}


def test_set_voice_mute_does_not_fire_a_note_trigger_or_accent(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4"])
    engine.note_trigger_count = 0
    engine.triggers.clear()

    controller.set_voice_mute("melody", True)

    assert engine.note_trigger_count == 0
    assert engine.triggers == []


# ---------------------------------------------------------
# Solo
# ---------------------------------------------------------


def test_set_voice_solo_republishes_with_the_new_solo_map(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4"])

    controller.set_voice_solo("harmony", True)

    assert dict(engine.published[-1].voice_solo) == {"harmony": True}


def test_multiple_voices_can_be_soloed_simultaneously(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4"])

    controller.set_voice_solo("harmony", True)
    controller.set_voice_solo("melody", True)

    assert dict(controller.voice_solo) == {"harmony": True, "melody": True}
    assert dict(engine.published[-1].voice_solo) == {"harmony": True, "melody": True}


def test_set_voice_solo_does_not_fire_a_note_trigger_or_accent(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4"])
    engine.note_trigger_count = 0
    engine.triggers.clear()

    controller.set_voice_solo("melody", True)

    assert engine.note_trigger_count == 0
    assert engine.triggers == []


# ---------------------------------------------------------
# State ownership -- no SessionState mutation, no analysis recomputation
# ---------------------------------------------------------


def test_mixer_changes_never_mutate_session_state_current_node(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])
    node_before = session_state.current_node

    controller.set_master_gain(0.2)
    controller.set_voice_mute("melody", True)
    controller.set_voice_solo("harmony", True)

    assert session_state.current_node is node_before


def test_mixer_changes_do_not_recompute_analysis(qapp, monkeypatch):
    import desktop_app.audio_controller as audio_controller_module

    call_count = {"n": 0}
    real_analyze_position = audio_controller_module.analyze_position

    def counting(*args, **kwargs):
        call_count["n"] += 1
        return real_analyze_position(*args, **kwargs)

    monkeypatch.setattr(audio_controller_module, "analyze_position", counting)

    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4"])
    assert call_count["n"] == 1

    controller.set_master_gain(0.5)
    controller.set_voice_mute("melody", True)
    controller.set_voice_solo("harmony", True)

    assert call_count["n"] == 1  # unchanged -- cache hit, no recomputation


# ---------------------------------------------------------
# Scrub interaction
# ---------------------------------------------------------


def test_mixer_change_during_scrub_republishes_the_current_scrub_position(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])

    path = session_state.active_path()
    controller.begin_scrub(path)
    controller.update_scrub(ScrubPosition(path_index=0, t=0.5))
    published_before = len(engine.published)

    controller.set_voice_mute("melody", True)

    assert len(engine.published) == published_before + 1
    assert dict(engine.published[-1].voice_mute) == {"melody": True}
    assert controller.is_scrubbing is True  # the scrub itself was not cancelled


def test_mixer_change_during_scrub_does_not_fire_a_trigger(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])

    path = session_state.active_path()
    controller.begin_scrub(path)
    controller.update_scrub(ScrubPosition(path_index=0, t=0.5))
    engine.note_trigger_count = 0
    engine.triggers.clear()

    controller.set_master_gain(0.6)

    assert engine.note_trigger_count == 0
    assert engine.triggers == []


def test_mixer_change_during_scrub_does_not_mutate_session_state(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "e7e5"])
    node_before = session_state.current_node

    path = session_state.active_path()
    controller.begin_scrub(path)
    controller.update_scrub(ScrubPosition(path_index=0, t=0.5))

    controller.set_voice_solo("harmony", True)

    assert session_state.current_node is node_before


def test_mixer_change_immediately_after_scrub_release_republishes_the_settled_node(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4", "d7d5"])

    path = session_state.active_path()  # [root, e4, d5]
    controller.begin_scrub(path)
    controller.update_scrub(ScrubPosition(path_index=1, t=0.5))
    controller.end_scrub()
    node = session_state.make_move(chess.Move.from_uci("e4d5"))  # the real release commit
    assert node is not None

    published_before = len(engine.published)
    controller.set_voice_mute("harmony", True)

    assert len(engine.published) == published_before + 1
    # Republished the settled node, not a scrub preview.
    assert engine.published[-1].segment_key == engine.published[-2].segment_key
    assert dict(engine.published[-1].voice_mute) == {"harmony": True}


def test_mixer_change_before_any_move_is_a_safe_no_op(qapp):
    _, controller, engine = _controller(qapp)

    controller.set_master_gain(0.5)
    controller.set_voice_mute("melody", True)
    controller.set_voice_solo("harmony", True)

    assert engine.published == []


# ---------------------------------------------------------
# Shutdown
# ---------------------------------------------------------


def test_mixer_change_after_shutdown_is_a_safe_no_op(qapp):
    session_state, controller, engine = _controller(qapp)
    _play(session_state, ["e2e4"])
    controller.shutdown()

    controller.set_master_gain(0.5)
    controller.set_voice_mute("melody", True)
    controller.set_voice_solo("harmony", True)

    assert len(engine.published) == 1  # unchanged
