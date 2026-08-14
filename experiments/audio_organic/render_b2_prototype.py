"""
B2 listening prototype: renders the same Scholar's-mate sequence used
by Phase A/B1 (experiments/audio_musicality) through the new
OrganicAudioRenderer (audio/organic_renderer.py), producing the full
sequence plus isolated diagnostic clips for each voice.

Read-only against production code; writes only under this module's own
output/ directory. Mirrors experiments/audio_musicality/render_candidates.py's
structure and normalization methodology (RMS-normalize each clip to a
common target before writing) so the two phases are directly comparable
by ear.

Run from the repo root:
    python -m experiments.audio_organic.render_b2_prototype
"""

from pathlib import Path

import numpy as np

from audio.models import AudioMapping, RenderConfig
from audio.organic_renderer import (
    ACCENT_MODAL_PARAMS as _RENDERER_ACCENT_PARAMS,  # re-exported for isolated-clip helpers below
)
from audio.organic_renderer import (
    CAPTURE_ACCENT_DURATION_SECONDS,
    HARMONY_GAIN,
    OrganicAudioRenderer,
    _harmony_frequency,
    _melody_params,
)
from audio.organic_synthesis import (
    ACCENT_BASE_FREQUENCY_HZ,
    HARMONY_MODAL_PARAMS,
    modal_resonance,
)
from experiments.audio_musicality.render_candidates import build_scholars_mate_mappings
from experiments.audio_musicality.synth_common import DEFAULT_SAMPLE_RATE, peak, rms, write_clip

OUTPUT_DIR = Path(__file__).parent / "output" / "audio"
DIAGNOSTICS_DIR = Path(__file__).parent / "output" / "diagnostics"

# Phase A's candidates (experiments/audio_musicality) used RMS-target
# normalization -- appropriate there because their linear attack/release
# envelopes sustained energy across most of the clip. Damped modal
# resonance is far more front-loaded (most energy in the first fraction
# of the clip, then a long near-silent tail from the exponential decay),
# so matching RMS to the same target used there would inflate these
# clips' peaks past clipping. Peak-normalizing to a common ceiling
# instead gives a fair, non-clipping loudness comparison that doesn't
# fight the shape of the physical decay being evaluated.
TARGET_PEAK = 0.9


def normalize_to_peak(samples: np.ndarray, target_peak: float = TARGET_PEAK) -> np.ndarray:
    current_peak = peak(samples)
    if current_peak <= 0.0:
        return samples
    return samples * (target_peak / current_peak)

MOVE_DURATION_SECONDS = 1.6
RENDER_CONFIG = RenderConfig(sample_rate=DEFAULT_SAMPLE_RATE, clip_duration_seconds=MOVE_DURATION_SECONDS)

# Index into the Scholar's-mate mapping list (e2e4 e7e5 d1h5 b8c6 f1c4
# g8f6 h5f7): 0 = White's opening move, 1 = Black's reply, 6 = the
# final move, which is both a capture and a check.
WHITE_DEMO_INDEX = 0
BLACK_DEMO_INDEX = 1
CAPTURE_CHECK_INDEX = 6


def render_isolated_melody(mapping: AudioMapping) -> np.ndarray:
    return modal_resonance(
        fundamental_hz=mapping.pitch_hz,
        params=_melody_params(mapping.color),
        duration_seconds=MOVE_DURATION_SECONDS,
        sample_rate=DEFAULT_SAMPLE_RATE,
        amplitude=mapping.loudness,
        attack_seconds=RENDER_CONFIG.attack_seconds,
    )


def render_isolated_harmony(mapping: AudioMapping) -> np.ndarray:
    return modal_resonance(
        fundamental_hz=_harmony_frequency(mapping),
        params=HARMONY_MODAL_PARAMS,
        duration_seconds=MOVE_DURATION_SECONDS,
        sample_rate=DEFAULT_SAMPLE_RATE,
        amplitude=mapping.loudness * HARMONY_GAIN,
        attack_seconds=RENDER_CONFIG.attack_seconds,
    )


def render_isolated_accent(mapping: AudioMapping) -> np.ndarray:
    return modal_resonance(
        fundamental_hz=ACCENT_BASE_FREQUENCY_HZ,
        params=_RENDERER_ACCENT_PARAMS,
        duration_seconds=CAPTURE_ACCENT_DURATION_SECONDS,
        sample_rate=DEFAULT_SAMPLE_RATE,
        amplitude=mapping.loudness,
        attack_seconds=0.0,
    )


def render_full_sequence(mappings: list[AudioMapping]) -> np.ndarray:
    renderer = OrganicAudioRenderer(RENDER_CONFIG)
    clips = [renderer.render(mapping).samples for mapping in mappings]
    return np.concatenate(clips)


def _write_normalized(samples: np.ndarray, path: Path, log_lines: list[str], label: str) -> None:
    pre_rms, pre_peak = rms(samples), peak(samples)
    normalized = np.clip(normalize_to_peak(samples), -1.0, 1.0)
    post_rms, post_peak = rms(normalized), peak(normalized)

    write_clip(normalized, path, sample_rate=DEFAULT_SAMPLE_RATE)

    print(f"{label}: pre-norm RMS={pre_rms:.4f} peak={pre_peak:.4f} "
          f"-> post-norm RMS={post_rms:.4f} peak={post_peak:.4f} -> {path}")
    log_lines.append(f"| {label} | {pre_rms:.4f} | {pre_peak:.4f} | {post_rms:.4f} | {post_peak:.4f} |\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    mappings = build_scholars_mate_mappings()
    print(f"Rendering {len(mappings)} moves ({' '.join(m.move for m in mappings)}) "
          f"through OrganicAudioRenderer (B2 damped modal resonance synthesis)...\n")

    log_lines = [
        "# B2 Organic Synthesis -- Loudness / Normalization Log\n",
        f"Target peak: {TARGET_PEAK} (peak-normalized, not RMS -- see module docstring)\n\n",
        "| clip | pre-norm RMS | pre-norm peak | post-norm RMS | post-norm peak |\n",
        "|---|---|---|---|---|\n",
    ]

    _write_normalized(render_full_sequence(mappings), OUTPUT_DIR / "b2_full_sequence.wav", log_lines, "full_sequence")

    white_mapping = mappings[WHITE_DEMO_INDEX]
    black_mapping = mappings[BLACK_DEMO_INDEX]
    capture_check_mapping = mappings[CAPTURE_CHECK_INDEX]
    assert capture_check_mapping.is_capture and capture_check_mapping.is_check

    _write_normalized(render_isolated_melody(white_mapping), OUTPUT_DIR / "b2_white_note.wav", log_lines, "white_note")
    _write_normalized(render_isolated_melody(black_mapping), OUTPUT_DIR / "b2_black_note.wav", log_lines, "black_note")
    _write_normalized(render_isolated_harmony(white_mapping), OUTPUT_DIR / "b2_harmony.wav", log_lines, "harmony")
    _write_normalized(render_isolated_accent(capture_check_mapping), OUTPUT_DIR / "b2_accent.wav", log_lines, "accent")
    _write_normalized(
        OrganicAudioRenderer(RENDER_CONFIG).render(capture_check_mapping).samples,
        OUTPUT_DIR / "b2_capture_check.wav", log_lines, "capture_check_move",
    )

    (DIAGNOSTICS_DIR / "loudness_log.md").write_text("".join(log_lines), encoding="utf-8")
    print(f"\nWrote loudness log to {DIAGNOSTICS_DIR / 'loudness_log.md'}")
    print(f"All B2 prototype WAVs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
