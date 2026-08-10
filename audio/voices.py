"""
Voice Registry -- the five named voices from docs/audio.md's
vocabulary, as Part 5 of docs/interactive_ui.md's mixer design already
names them (Harmony, Melody, Accent, Drone, Space). This registry did
not exist anywhere in the codebase before Phase 5f; Part 13's phase
table assigns "mixer UI built against the Voice Registry" to 5f itself,
so this is greenfield infrastructure, not a trim-down of something
pre-existing.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Voice:
    """
    One registered mixer voice.

    `active=False` means the voice has no real signal source yet -- it
    exists so the mixer UI's shape matches the frozen five-voice design
    now, without inventing a sound mapping ahead of the math that would
    justify one. This is a registry-entry decision, not a
    sound-parameter mapping, so it does not conflict with
    docs/audio.md's "no arbitrary mappings" principle -- that principle
    governs what a sound parameter *represents*, not which UI slots
    exist.
    """

    voice_id: str
    label: str
    active: bool


class VoiceRegistry:
    """
    Voices are looked up by string id (not object identity), matching
    docs/interactive_ui.md Part 4.1's stated reason: Registry code may
    read SessionState-adjacent code; SessionState-adjacent code should
    never need to import Registry types.
    """

    def __init__(self) -> None:
        self._voices: dict[str, Voice] = {}

    def register(self, voice: Voice) -> None:
        if voice.voice_id in self._voices:
            raise ValueError(f"voice already registered: {voice.voice_id!r}")

        self._voices[voice.voice_id] = voice

    def get(self, voice_id: str) -> Voice:
        return self._voices[voice_id]

    def __contains__(self, voice_id: str) -> bool:
        return voice_id in self._voices

    def __iter__(self):
        return iter(self._voices.values())

    def __len__(self) -> int:
        return len(self._voices)

    def active_voices(self) -> list[Voice]:
        return [voice for voice in self._voices.values() if voice.active]

    def silent_voices(self) -> list[Voice]:
        return [voice for voice in self._voices.values() if not voice.active]


def build_default_voice_registry() -> VoiceRegistry:
    """
    The five voices from docs/audio.md's vocabulary, in the same order
    as docs/interactive_ui.md Part 5's mixer strip ("Harmony [M][S]
    Melody [M][S] Accent [M][S] Drone [M][S] Space [M][S]").

    Melody, Harmony and Accent have real signal sources today --
    audio/mapping.py's six MVP signals (destination-square pitch,
    Attack Influence balance, and capture) already drive them. Drone
    (Source Field) and Space (Source Potential / Ridge-Valley pan,
    still colliding -- see docs/interactive_ui.md Review Disposition
    #17) have no mathematical content until Milestone 4b is scoped, so
    they are registered but silent: no synthesis code path exists for
    them anywhere in audio/engine.py, not merely a mapping forced to
    zero.
    """

    registry = VoiceRegistry()
    registry.register(Voice("harmony", "Harmony", active=True))
    registry.register(Voice("melody", "Melody", active=True))
    registry.register(Voice("accent", "Accent", active=True))
    registry.register(Voice("drone", "Drone", active=False))
    registry.register(Voice("space", "Space", active=False))
    return registry
