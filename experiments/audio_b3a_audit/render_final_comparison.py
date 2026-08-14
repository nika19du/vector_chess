"""
B3a Phase 4 -- final A/B/C comparison WAVs, for listening approval.

Renders the exact same Scholar's-mate sequence used by every prior
phase (B1/B2/B3) through three paths:

  A) b2_reference.wav   -- OrganicAudioRenderer (B2 offline, unchanged, the
                            accepted subjective reference)
  B) b3_current.wav     -- the live engine as it existed just before this
                            phase (reconstructed -- see legacy_b3_reference.py
                            module docstring for why, and its own verified
                            match against the real captured Phase 1 audio)
  C) b3a_candidate.wav  -- the current, actually-live AudioEngine (B3a)

Plus isolated White/Black/Harmony/capture-Accent for each path.

Read-only against production code; writes only under this module's own
output/ directory.

Run from the repo root:
    python -m experiments.audio_b3a_audit.render_final_comparison
"""

from pathlib import Path

import numpy as np

from audio.backend import FakeAudioBackend
from audio.engine import DEFAULT_SAMPLE_RATE, AccentTrigger, AudioEngine
from audio.live_state import sonification_state_from_mapping
from audio.models import RenderConfig
from audio.organic_renderer import ACCENT_BASE_FREQUENCY_HZ, HARMONY_GAIN, OrganicAudioRenderer, SUSTAINED_PEAK_HEADROOM, _harmony_frequency, _melody_params
from audio.organic_synthesis import ACCENT_MODAL_PARAMS, HARMONY_MODAL_PARAMS, modal_resonance
from audio.voices import build_default_voice_registry
from experiments.audio_musicality.render_candidates import build_scholars_mate_mappings
from experiments.audio_musicality.synth_common import peak, write_clip
from experiments.audio_b3a_audit.legacy_b3_reference import render_legacy_b3_harmony, render_legacy_b3_melody

OUTPUT_DIR = Path(__file__).parent / "output" / "audio_final"
SAMPLE_RATE = DEFAULT_SAMPLE_RATE
MOVE_DURATION_SECONDS = 1.6
CONFIG = RenderConfig(sample_rate=SAMPLE_RATE, clip_duration_seconds=MOVE_DURATION_SECONDS)
TARGET_PEAK = 0.9

WHITE_INDEX = 0
BLACK_INDEX = 1
CAPTURE_CHECK_INDEX = 6


def normalize_to_peak(samples: np.ndarray, target_peak: float = TARGET_PEAK) -> np.ndarray:
    current_peak = peak(samples)
    if current_peak <= 0.0:
        return samples
    return np.clip(samples * (target_peak / current_peak), -1.0, 1.0)


# ---------------------------------------------------------
# A) B2 offline reference (unchanged code path)
# ---------------------------------------------------------


def render_b2_move(mapping) -> np.ndarray:
    return OrganicAudioRenderer(CONFIG).render(mapping).samples


def render_b2_isolated_melody(mapping) -> np.ndarray:
    return modal_resonance(
        mapping.pitch_hz, _melody_params(mapping.color), MOVE_DURATION_SECONDS, SAMPLE_RATE,
        amplitude=mapping.loudness, attack_seconds=CONFIG.attack_seconds,
    )


def render_b2_isolated_harmony(mapping) -> np.ndarray:
    return modal_resonance(
        _harmony_frequency(mapping), HARMONY_MODAL_PARAMS, MOVE_DURATION_SECONDS, SAMPLE_RATE,
        amplitude=mapping.loudness * HARMONY_GAIN, attack_seconds=CONFIG.attack_seconds,
    )


def render_b2_isolated_accent(mapping) -> np.ndarray:
    return modal_resonance(
        ACCENT_BASE_FREQUENCY_HZ, ACCENT_MODAL_PARAMS, 0.2, SAMPLE_RATE,
        amplitude=mapping.loudness, attack_seconds=0.0,
    )


# ---------------------------------------------------------
# B) B3 (pre-B3a) reconstructed reference -- see legacy_b3_reference.py
# ---------------------------------------------------------


def render_legacy_b3_move(mapping) -> np.ndarray:
    melody = render_legacy_b3_melody(mapping.pitch_hz, mapping.color, MOVE_DURATION_SECONDS, SAMPLE_RATE, mapping.loudness)
    harmony_freq = _harmony_frequency(mapping)
    harmony = render_legacy_b3_harmony(harmony_freq, MOVE_DURATION_SECONDS, SAMPLE_RATE, mapping.loudness * HARMONY_GAIN)

    sustained = melody + harmony
    peak_bound = mapping.loudness + mapping.loudness * HARMONY_GAIN  # matches the old engine's own analytic bound
    if peak_bound > SUSTAINED_PEAK_HEADROOM:
        sustained = sustained * (SUSTAINED_PEAK_HEADROOM / peak_bound)

    if mapping.is_capture or mapping.is_check:
        accent = modal_resonance(
            ACCENT_BASE_FREQUENCY_HZ, ACCENT_MODAL_PARAMS, 0.2, SAMPLE_RATE,
            amplitude=mapping.loudness, attack_seconds=0.0,
        )
        n = max(len(sustained), len(accent))
        combined = np.zeros(n)
        combined[:len(sustained)] += sustained
        combined[:len(accent)] += accent
    else:
        combined = sustained

    return np.clip(combined, -1.0, 1.0)


# ---------------------------------------------------------
# C) B3a -- the actual current live AudioEngine
# ---------------------------------------------------------


def render_live_move(mapping, *, voice_mute=None, with_accent: bool = False) -> np.ndarray:
    backend = FakeAudioBackend()
    registry = build_default_voice_registry()
    engine = AudioEngine(backend, registry, sample_rate=SAMPLE_RATE, blocksize=64)
    engine.start()
    stream = backend.streams[-1]

    engine.publish(sonification_state_from_mapping(mapping, ("a", "b"), voice_mute=voice_mute or {}))
    engine.push_note_trigger()
    if with_accent and (mapping.is_capture or mapping.is_check):
        kind = "capture+check" if (mapping.is_capture and mapping.is_check) else ("capture" if mapping.is_capture else "check")
        engine.push_event(AccentTrigger(kind=kind, loudness=mapping.loudness))

    n = int(round(MOVE_DURATION_SECONDS * SAMPLE_RATE))
    blocks = []
    rendered = 0
    while rendered < n:
        frames = min(64, n - rendered)
        blocks.append(stream.render_block(frames).flatten())
        rendered += frames
    return np.concatenate(blocks)


def render_b3a_isolated_accent(mapping) -> np.ndarray:
    return render_live_move(mapping, voice_mute={"melody": True, "harmony": True}, with_accent=True)[:int(0.2 * SAMPLE_RATE)]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mappings = build_scholars_mate_mappings()
    white_mapping, black_mapping = mappings[WHITE_INDEX], mappings[BLACK_INDEX]
    capture_mapping = mappings[CAPTURE_CHECK_INDEX]
    assert capture_mapping.is_capture and capture_mapping.is_check

    print(f"Rendering {len(mappings)} moves ({' '.join(m.move for m in mappings)}) through A) B2, B) B3(pre-B3a), C) B3a...\n")

    # -- Full sequences ----------------------------------------------
    b2_full = np.concatenate([render_b2_move(m) for m in mappings])
    b3_full = np.concatenate([render_legacy_b3_move(m) for m in mappings])
    b3a_full = np.concatenate([render_live_move(m, with_accent=True) for m in mappings])

    write_clip(normalize_to_peak(b2_full), OUTPUT_DIR / "b2_reference.wav", sample_rate=SAMPLE_RATE)
    write_clip(normalize_to_peak(b3_full), OUTPUT_DIR / "b3_current.wav", sample_rate=SAMPLE_RATE)
    write_clip(normalize_to_peak(b3a_full), OUTPUT_DIR / "b3a_candidate.wav", sample_rate=SAMPLE_RATE)

    print(f"full sequence peaks (pre-norm): B2={peak(b2_full):.4f} B3={peak(b3_full):.4f} B3a={peak(b3a_full):.4f}")

    # -- Isolated voices, all three paths ------------------------------
    isolated = {
        "white": (render_b2_isolated_melody(white_mapping), render_legacy_b3_melody(white_mapping.pitch_hz, "white", MOVE_DURATION_SECONDS, SAMPLE_RATE, white_mapping.loudness), render_live_move(white_mapping, voice_mute={"harmony": True, "accent": True})),
        "black": (render_b2_isolated_melody(black_mapping), render_legacy_b3_melody(black_mapping.pitch_hz, "black", MOVE_DURATION_SECONDS, SAMPLE_RATE, black_mapping.loudness), render_live_move(black_mapping, voice_mute={"harmony": True, "accent": True})),
        "harmony": (render_b2_isolated_harmony(white_mapping), render_legacy_b3_harmony(_harmony_frequency(white_mapping), MOVE_DURATION_SECONDS, SAMPLE_RATE, white_mapping.loudness * HARMONY_GAIN), render_live_move(white_mapping, voice_mute={"melody": True, "accent": True})),
        "accent": (render_b2_isolated_accent(capture_mapping), modal_resonance(ACCENT_BASE_FREQUENCY_HZ, ACCENT_MODAL_PARAMS, 0.2, SAMPLE_RATE, amplitude=capture_mapping.loudness, attack_seconds=0.0), render_b3a_isolated_accent(capture_mapping)),
    }

    for label, (b2_clip, b3_clip, b3a_clip) in isolated.items():
        write_clip(normalize_to_peak(b2_clip), OUTPUT_DIR / f"b2_{label}.wav", sample_rate=SAMPLE_RATE)
        write_clip(normalize_to_peak(b3_clip), OUTPUT_DIR / f"b3_{label}.wav", sample_rate=SAMPLE_RATE)
        write_clip(normalize_to_peak(b3a_clip), OUTPUT_DIR / f"b3a_{label}.wav", sample_rate=SAMPLE_RATE)
        print(f"{label}: wrote b2_{label}.wav, b3_{label}.wav, b3a_{label}.wav")

    print(f"\nAll comparison WAVs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
