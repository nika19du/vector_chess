"""
Audio Layer 2 -- Rhythmic Layer: offline A/B listening deliverable.

Deliberately isolated here (not wired into console_app or any production
path), same as every prior experiments/ phase (audio_musicality,
phase_b1_mapping, audio_organic, audio_b3a_audit) -- this script exists
purely to render comparison WAVs for a human listening pass, using the
exact same live AudioEngine/AudioController pipeline production uses
(FakeAudioBackend swapped in for sounddevice, matching every existing
engine test's own convention), never a second, parallel synthesis path.

Renders:
  1. output/ab_no_pulse.wav / output/ab_with_pulse.wav -- the same
     7-move tactical sequence (1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6?? 4.Qxf7+,
     already used elsewhere in this codebase's own test suite), once
     with the Pulse voice off (pulse_density forced to 0.0) and once on,
     peak-normalized to the same level so the comparison isn't confounded
     by differing overall loudness.
  2. output/calm.wav / active.wav / tense.wav / chaotic.wav -- the same
     real position, paired with a synthetic DynamicsAnalysis at an
     intensity squarely inside each of analysis.dynamics's own four
     label bands (calm<5, active<12, tense<20, chaotic>=20), so the
     Dynamics.intensity -> pulse_density gradient is isolated and
     audible on its own, without a real game's incidental variation.

Run: python -m experiments.audio_pulse_mvp.render_ab
"""

import dataclasses
from pathlib import Path

import chess
import numpy as np

from audio.backend import FakeAudioBackend
from audio.engine import AccentTrigger, AudioEngine
from audio.export import write_wav
from audio.live_state import sonification_state_from_mapping
from audio.mapping import build_audio_mapping
from audio.models import AudioClip, AudioMapping
from audio.voices import build_default_voice_registry
from chess_engine.analyzer import analyze_position
from chess_engine.models import DynamicsAnalysis
from chess_engine.moves import execute_move

OUTPUT_DIR = Path(__file__).parent / "output"
SAMPLE_RATE = 44100
BLOCKSIZE = 64
TARGET_PEAK = 0.9

TACTICAL_SEQUENCE = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]


def _mappings_for_sequence(moves: list[str]) -> list[AudioMapping]:
    from analysis.dynamics import analyze_dynamics

    board = chess.Board()
    previous_analysis = None
    mappings = []

    for move_text in moves:
        details = execute_move(board, move_text)
        assert details is not None, f"expected {move_text} to be legal"
        analysis = analyze_position(board, details)
        dynamics = (
            None if previous_analysis is None else analyze_dynamics(previous=previous_analysis, current=analysis)
        )
        mappings.append(build_audio_mapping(analysis, dynamics))
        previous_analysis = analysis

    return mappings


def render_mapping_sequence(mappings: list[AudioMapping], *, pulse_enabled: bool, seconds_per_move: float) -> AudioClip:
    """
    Renders `mappings` through a real AudioEngine (FakeAudioBackend, no
    hardware) exactly the way AudioController drives it for a committed
    move: publish the settled state, push exactly one NoteTrigger, and
    push an AccentTrigger for a capturing/checking move -- then lets the
    engine ring for `seconds_per_move` before moving to the next mapping.
    """

    backend = FakeAudioBackend()
    registry = build_default_voice_registry()
    engine = AudioEngine(backend, registry, sample_rate=SAMPLE_RATE, blocksize=BLOCKSIZE)
    assert engine.start() is True
    stream = backend.streams[-1]

    blocks_per_move = int(round(seconds_per_move * SAMPLE_RATE / BLOCKSIZE))
    rendered_blocks = []

    for index, mapping in enumerate(mappings):
        segment_key = (f"move-{index}", f"move-{index + 1}")
        state = sonification_state_from_mapping(mapping, segment_key=segment_key)
        if not pulse_enabled:
            state = dataclasses.replace(state, pulse_density=0.0)

        engine.publish(state)
        engine.push_note_trigger()
        if mapping.is_capture or mapping.is_check:
            if mapping.is_capture and mapping.is_check:
                kind = "capture+check"
            elif mapping.is_capture:
                kind = "capture"
            else:
                kind = "check"
            engine.push_event(AccentTrigger(kind=kind, loudness=mapping.loudness))

        for _ in range(blocks_per_move):
            rendered_blocks.append(stream.render_block(BLOCKSIZE).flatten().copy())

    engine.shutdown()
    samples = np.concatenate(rendered_blocks) if rendered_blocks else np.zeros(0)
    return AudioClip(samples=samples, sample_rate=SAMPLE_RATE)


def render_single_mapping(mapping: AudioMapping, *, seconds: float) -> AudioClip:
    return render_mapping_sequence([mapping], pulse_enabled=True, seconds_per_move=seconds)


def normalize_peak(clip: AudioClip, target_peak: float = TARGET_PEAK) -> AudioClip:
    """
    A single global scalar per clip (never per-sample/per-segment) so the
    comparison is loudness-*matched*, not loudness-*distorted* -- the
    music's own internal dynamics (quiet opening vs. the capture+check
    finale) are untouched, only the whole clip's overall level is
    brought to a common reference peak.
    """

    peak = float(np.max(np.abs(clip.samples))) if clip.samples.size else 0.0
    if peak <= 0.0:
        return clip
    return AudioClip(samples=clip.samples * (target_peak / peak), sample_rate=clip.sample_rate)


def _synthetic_dynamics(intensity: float, label: str) -> DynamicsAnalysis:
    return DynamicsAnalysis(
        previous_force=0,
        current_force=0,
        delta_force=0,
        white_mobility_delta=0,
        black_mobility_delta=0,
        attack_vectors_delta=0,
        heatmap_change=0,
        intensity=intensity,
        label=label,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Main A/B: same tactical sequence, Pulse off vs. on.
    mappings = _mappings_for_sequence(TACTICAL_SEQUENCE)
    clip_a = normalize_peak(render_mapping_sequence(mappings, pulse_enabled=False, seconds_per_move=2.5))
    clip_b = normalize_peak(render_mapping_sequence(mappings, pulse_enabled=True, seconds_per_move=2.5))
    write_wav(clip_a, OUTPUT_DIR / "ab_no_pulse.wav")
    write_wav(clip_b, OUTPUT_DIR / "ab_with_pulse.wav")

    # 2. Intensity-gradient examples: same reference position/move, four
    # different synthetic Dynamics.intensity values, one squarely inside
    # each of analysis.dynamics's own four label bands.
    board = chess.Board()
    details = execute_move(board, "g1f3")
    analysis = analyze_position(board, details)

    examples = {
        "calm": _synthetic_dynamics(2.0, "calm"),
        "active": _synthetic_dynamics(8.0, "active"),
        "tense": _synthetic_dynamics(16.0, "tense"),
        "chaotic": _synthetic_dynamics(25.0, "chaotic"),
    }

    for name, dynamics in examples.items():
        mapping = build_audio_mapping(analysis, dynamics)
        clip = normalize_peak(render_single_mapping(mapping, seconds=4.0))
        write_wav(clip, OUTPUT_DIR / f"{name}.wav")
        print(f"{name}: intensity={dynamics.intensity}, pulse_density={mapping.pulse_density:.3f}")

    print(f"Wrote WAVs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
