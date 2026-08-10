import pytest

from audio.voices import Voice, VoiceRegistry, build_default_voice_registry


# ---------------------------------------------------------
# VoiceRegistry -- registration and lookup
# ---------------------------------------------------------


def test_register_and_get_round_trips():
    registry = VoiceRegistry()
    voice = Voice("melody", "Melody", active=True)

    registry.register(voice)

    assert registry.get("melody") is voice
    assert "melody" in registry
    assert "harmony" not in registry
    assert len(registry) == 1


def test_registering_a_duplicate_voice_id_raises():
    registry = VoiceRegistry()
    registry.register(Voice("melody", "Melody", active=True))

    with pytest.raises(ValueError):
        registry.register(Voice("melody", "Melody Again", active=True))


def test_iteration_yields_every_registered_voice():
    registry = VoiceRegistry()
    registry.register(Voice("melody", "Melody", active=True))
    registry.register(Voice("drone", "Drone", active=False))

    assert {voice.voice_id for voice in registry} == {"melody", "drone"}


# ---------------------------------------------------------
# active vs. silent voice behavior
# ---------------------------------------------------------


def test_active_voices_excludes_silent_ones():
    registry = VoiceRegistry()
    registry.register(Voice("melody", "Melody", active=True))
    registry.register(Voice("drone", "Drone", active=False))

    assert [v.voice_id for v in registry.active_voices()] == ["melody"]


def test_silent_voices_excludes_active_ones():
    registry = VoiceRegistry()
    registry.register(Voice("melody", "Melody", active=True))
    registry.register(Voice("drone", "Drone", active=False))

    assert [v.voice_id for v in registry.silent_voices()] == ["drone"]


# ---------------------------------------------------------
# build_default_voice_registry -- exactly the five planned voices
# ---------------------------------------------------------


def test_default_registry_contains_exactly_the_five_planned_voices():
    registry = build_default_voice_registry()

    assert {voice.voice_id for voice in registry} == {
        "harmony",
        "melody",
        "accent",
        "drone",
        "space",
    }
    assert len(registry) == 5


def test_default_registry_marks_melody_harmony_accent_active():
    registry = build_default_voice_registry()

    assert registry.get("melody").active is True
    assert registry.get("harmony").active is True
    assert registry.get("accent").active is True


def test_default_registry_marks_drone_and_space_silent():
    registry = build_default_voice_registry()

    assert registry.get("drone").active is False
    assert registry.get("space").active is False


def test_default_registry_active_and_silent_partition_correctly():
    registry = build_default_voice_registry()

    active_ids = {v.voice_id for v in registry.active_voices()}
    silent_ids = {v.voice_id for v in registry.silent_voices()}

    assert active_ids == {"harmony", "melody", "accent"}
    assert silent_ids == {"drone", "space"}
    assert active_ids.isdisjoint(silent_ids)


def test_voice_is_frozen():
    voice = Voice("melody", "Melody", active=True)

    with pytest.raises(Exception):
        voice.active = False
