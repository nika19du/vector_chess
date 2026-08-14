"""
Rhythmic Layer v2 -- Phase C listening artifact: old (v1, continuous-
loop "metronome") Pulse MVP vs. new (v2, finite phrase) version, same
7-move tactical sequence, one file: old -> short silence -> new.

The v1 half is NOT regenerated -- audio/pulse_pattern.py and its engine
mechanism no longer exist in production (deliberately removed, not
merely retuned). Reused as-is: experiments/audio_pulse_mvp/output/
ab_with_pulse.wav, the actual recording the user listened to when they
reported the metronome problem. The v2 half is rendered fresh through
the real, current production AudioEngine (FakeAudioBackend swapped in
for hardware) -- the same publish/push_note_trigger/push_event sequence
AudioController drives for a real committed move, not a hand-rolled
reimplementation.

Both halves are independently peak-normalized to the same target level
before concatenation, so the comparison is loudness-matched.
"""

import wave
from pathlib import Path

import chess
import numpy as np

from analysis.dynamics import analyze_dynamics
from audio.backend import FakeAudioBackend
from audio.engine import AccentTrigger, AudioEngine
from audio.export import write_wav
from audio.live_state import sonification_state_from_mapping
from audio.mapping import build_audio_mapping
from audio.models import AudioClip
from audio.voices import build_default_voice_registry
from chess_engine.analyzer import analyze_position
from chess_engine.moves import execute_move

OLD_MVP_WAV = Path(__file__).parent.parent / "audio_pulse_mvp" / "output" / "ab_with_pulse.wav"
OUTPUT_DIR = Path(__file__).parent / "output"
SAMPLE_RATE = 44100
BLOCKSIZE = 64
SECONDS_PER_MOVE = 3.0
SILENCE_GAP_SECONDS = 1.5
TARGET_PEAK = 0.9

TACTICAL_SEQUENCE = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]


def _read_wav_as_float(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        assert wav_file.getsampwidth() == 2
        frames = wav_file.readframes(wav_file.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32767.0
    return samples


def render_v2_sequence(moves: list[str]) -> np.ndarray:
    backend = FakeAudioBackend()
    registry = build_default_voice_registry()
    engine = AudioEngine(backend, registry, sample_rate=SAMPLE_RATE, blocksize=BLOCKSIZE)
    assert engine.start() is True
    stream = backend.streams[-1]

    board = chess.Board()
    previous_analysis = None
    blocks_per_move = int(round(SECONDS_PER_MOVE * SAMPLE_RATE / BLOCKSIZE))
    rendered = []

    for index, move_text in enumerate(moves):
        details = execute_move(board, move_text)
        assert details is not None
        analysis = analyze_position(board, details)
        dynamics = (
            None if previous_analysis is None else analyze_dynamics(previous=previous_analysis, current=analysis)
        )
        mapping = build_audio_mapping(analysis, dynamics)
        previous_analysis = analysis

        state = sonification_state_from_mapping(mapping, segment_key=(f"m{index}", f"m{index + 1}"))
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
            rendered.append(stream.render_block(BLOCKSIZE).flatten().copy())

    engine.shutdown()
    return np.concatenate(rendered) if rendered else np.zeros(0)


def normalize_peak(samples: np.ndarray, target_peak: float = TARGET_PEAK) -> np.ndarray:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 0.0:
        return samples
    return samples * (target_peak / peak)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    old_samples = normalize_peak(_read_wav_as_float(OLD_MVP_WAV))
    new_samples = normalize_peak(render_v2_sequence(TACTICAL_SEQUENCE))

    silence = np.zeros(int(round(SILENCE_GAP_SECONDS * SAMPLE_RATE)))
    combined = np.concatenate([old_samples, silence, new_samples])

    clip = AudioClip(samples=np.clip(combined, -1.0, 1.0), sample_rate=SAMPLE_RATE)
    out_path = OUTPUT_DIR / "v1_metronome_vs_v2_phrase.wav"
    write_wav(clip, out_path)

    print(f"old duration: {len(old_samples) / SAMPLE_RATE:.2f}s")
    print(f"new duration: {len(new_samples) / SAMPLE_RATE:.2f}s")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
