"""
B3a Phase 1 -- audit / A-B measurement, before any implementation.

Traces identical AudioMapping inputs through:
  A) OrganicAudioRenderer (B2 offline reference, unchanged)
  B) the current live AudioEngine (B3, unchanged by this script)

and measures, over time: RMS envelope, spectral centroid, and (for
melody) the analytic per-mode weight each path is actually using --
to determine, with numbers rather than assumption, whether B3's
missing per-mode exp(-damping*t) evolution explains the audible
regression the user reported.

Read-only against production code (audio/, desktop_app/ untouched).
Writes only under this module's own output/ directory.

Run from the repo root:
    python -m experiments.audio_b3a_audit.measure_b2_vs_b3
"""

from pathlib import Path

import numpy as np

from audio.backend import FakeAudioBackend
from audio.engine import DEFAULT_SAMPLE_RATE, AccentTrigger, AudioEngine
from audio.live_state import sonification_state_from_mapping
from audio.mapping import build_audio_mapping
from audio.models import AudioMapping, RenderConfig
from audio.organic_renderer import HARMONY_GAIN, OrganicAudioRenderer, _harmony_frequency, _melody_params
from audio.organic_synthesis import ACCENT_BASE_FREQUENCY_HZ, ACCENT_MODES, HARMONY_MODES, modal_resonance
from audio.voices import build_default_voice_registry
from chess_engine.analyzer import analyze_position
from chess_engine.moves import execute_move

OUTPUT_AUDIO_DIR = Path(__file__).parent / "output" / "audio"
OUTPUT_DIAG_DIR = Path(__file__).parent / "output" / "diagnostics"

SAMPLE_RATE = DEFAULT_SAMPLE_RATE
DURATION_SECONDS = 1.6
LIVE_BLOCKSIZE = 64  # fine time-resolution for measurement, not the production blocksize
WINDOW_SECONDS = 0.04
CONFIG = RenderConfig(sample_rate=SAMPLE_RATE, clip_duration_seconds=DURATION_SECONDS)


def _mapping_after(moves: list[str]) -> AudioMapping:
    import chess

    board = chess.Board()
    analysis = None
    for move_text in moves:
        details = execute_move(board, move_text)
        assert details is not None, move_text
        analysis = analyze_position(board, details)
    return build_audio_mapping(analysis, dynamics=None)


def render_live(mapping: AudioMapping, *, voice_mute=None, with_accent: bool = False) -> np.ndarray:
    """
    Drives the real AudioEngine (production envelope constants, not a
    test override) through a single committed-note trigger, capturing
    exactly DURATION_SECONDS of audio at fine block granularity so the
    measurement below has good time resolution. This is not a test
    double of the DSP -- it is the actual _render_block callback.
    """

    backend = FakeAudioBackend()
    registry = build_default_voice_registry()
    engine = AudioEngine(backend, registry, sample_rate=SAMPLE_RATE, blocksize=LIVE_BLOCKSIZE)
    engine.start()
    stream = backend.streams[-1]

    engine.publish(sonification_state_from_mapping(mapping, ("a", "b"), voice_mute=voice_mute or {}))
    engine.push_note_trigger()
    if with_accent and (mapping.is_capture or mapping.is_check):
        kind = "capture+check" if (mapping.is_capture and mapping.is_check) else ("capture" if mapping.is_capture else "check")
        engine.push_event(AccentTrigger(kind=kind, loudness=mapping.loudness))

    n = int(round(DURATION_SECONDS * SAMPLE_RATE))
    blocks = []
    rendered = 0
    while rendered < n:
        frames = min(LIVE_BLOCKSIZE, n - rendered)
        blocks.append(stream.render_block(frames).flatten())
        rendered += frames
    return np.concatenate(blocks)


def windowed_rms(samples: np.ndarray, sample_rate: int, window_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    window = int(window_seconds * sample_rate)
    n_windows = len(samples) // window
    times = (np.arange(n_windows) + 0.5) * window_seconds
    rms = np.array([
        np.sqrt(np.mean(samples[i * window:(i + 1) * window] ** 2))
        for i in range(n_windows)
    ])
    return times, rms


def windowed_centroid(samples: np.ndarray, sample_rate: int, window_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    window = int(window_seconds * sample_rate)
    n_windows = len(samples) // window
    times = (np.arange(n_windows) + 0.5) * window_seconds
    centroids = []
    for i in range(n_windows):
        chunk = samples[i * window:(i + 1) * window]
        spectrum = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), d=1.0 / sample_rate)
        total = spectrum.sum()
        centroids.append(float((spectrum * freqs).sum() / total) if total > 1e-12 else 0.0)
    return times, np.array(centroids)


def analytic_mode_weights_over_time(modes, t: np.ndarray, *, apply_damping: bool) -> np.ndarray:
    """weight_k(t) for each mode k, normalized by weight_sum -- what
    fraction of the note's total (weight-normalized) amplitude budget
    mode k contributes at time t, under each path's own formula."""

    if apply_damping:
        return np.array([
            (w / modes.weight_sum) * np.exp(-d * t) for w, d in zip(modes.weights, modes.dampings)
        ])
    return np.array([
        np.full_like(t, w / modes.weight_sum) for w in modes.weights
    ])


def write_wav(samples: np.ndarray, path: Path) -> None:
    from audio.export import write_wav as _write_wav
    from audio.models import AudioClip

    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    normalized = samples * (0.9 / peak) if peak > 0.0 else samples
    _write_wav(AudioClip(samples=np.clip(normalized, -1.0, 1.0), sample_rate=SAMPLE_RATE), path)


def main() -> None:
    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIAG_DIR.mkdir(parents=True, exist_ok=True)

    white_mapping = _mapping_after(["e2e4"])
    black_mapping = _mapping_after(["e2e4", "e7e5"])
    full_move_mapping = _mapping_after(["e2e4", "e7e5", "g1f3"])  # a quiet non-capture move with real harmony content
    capture_mapping = _mapping_after(["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"])
    assert capture_mapping.is_capture and capture_mapping.is_check

    report_lines: list[str] = ["# B3a Phase 1 -- B2 (offline) vs B3 (live) A/B Measurement\n\n"]

    # -----------------------------------------------------------------
    # 1. White / Black melody -- isolated voice, both paths
    # -----------------------------------------------------------------
    for label, mapping in (("white", white_mapping), ("black", black_mapping)):
        offline_melody = modal_resonance(
            mapping.pitch_hz, _melody_params(mapping.color), DURATION_SECONDS, SAMPLE_RATE,
            amplitude=mapping.loudness, attack_seconds=CONFIG.attack_seconds,
        )
        live_melody = render_live(mapping, voice_mute={"harmony": True, "accent": True})

        write_wav(offline_melody, OUTPUT_AUDIO_DIR / f"b2_{label}_melody.wav")
        write_wav(live_melody, OUTPUT_AUDIO_DIR / f"b3_{label}_melody.wav")

        t_off, rms_off = windowed_rms(offline_melody, SAMPLE_RATE, WINDOW_SECONDS)
        t_live, rms_live = windowed_rms(live_melody, SAMPLE_RATE, WINDOW_SECONDS)
        _, cen_off = windowed_centroid(offline_melody, SAMPLE_RATE, WINDOW_SECONDS)
        _, cen_live = windowed_centroid(live_melody, SAMPLE_RATE, WINDOW_SECONDS)

        report_lines.append(f"## Melody -- {label}\n\n")
        report_lines.append("| t(s) | B2 RMS | B3 RMS | B2 centroid(Hz) | B3 centroid(Hz) |\n")
        report_lines.append("|---|---|---|---|---|\n")
        for i in range(0, len(t_off), 2):  # every other window to keep the table short
            report_lines.append(
                f"| {t_off[i]:.2f} | {rms_off[i]:.4f} | {rms_live[i]:.4f} | "
                f"{cen_off[i]:.1f} | {cen_live[i]:.1f} |\n"
            )

        # Analytic per-mode weight trace -- the direct mechanism check.
        modes = _melody_params(mapping.color)
        from audio.organic_synthesis import build_mode_arrays
        mode_arrays = build_mode_arrays(modes)
        t_fine = np.linspace(0.0, DURATION_SECONDS, 9)
        offline_mode_weights = analytic_mode_weights_over_time(mode_arrays, t_fine, apply_damping=True)
        live_mode_weights = analytic_mode_weights_over_time(mode_arrays, t_fine, apply_damping=False)

        report_lines.append(f"\n**Per-mode normalized weight over time ({label}, analytic, not measured from audio):**\n\n")
        report_lines.append("B2 (per-mode exp(-damping*t) applied):\n\n")
        report_lines.append("| mode | ratio | " + " | ".join(f"t={tt:.2f}s" for tt in t_fine) + " |\n")
        report_lines.append("|---|---|" + "---|" * len(t_fine) + "\n")
        for k, ratio in enumerate(mode_arrays.ratios):
            report_lines.append(f"| {k} | {ratio} | " + " | ".join(f"{v:.4f}" for v in offline_mode_weights[k]) + " |\n")
        report_lines.append("\nB3 (no per-mode damping -- constant weight, only the outer envelope changes total amplitude):\n\n")
        report_lines.append("| mode | ratio | " + " | ".join(f"t={tt:.2f}s" for tt in t_fine) + " |\n")
        report_lines.append("|---|---|" + "---|" * len(t_fine) + "\n")
        for k, ratio in enumerate(mode_arrays.ratios):
            report_lines.append(f"| {k} | {ratio} | " + " | ".join(f"{v:.4f}" for v in live_mode_weights[k]) + " |\n")
        report_lines.append("\n")

    # -----------------------------------------------------------------
    # 2. Harmony -- isolated voice, both paths
    # -----------------------------------------------------------------
    harmony_mapping = full_move_mapping
    harmony_freq = _harmony_frequency(harmony_mapping)
    from audio.organic_synthesis import HARMONY_MODAL_PARAMS

    offline_harmony = modal_resonance(
        harmony_freq, HARMONY_MODAL_PARAMS, DURATION_SECONDS, SAMPLE_RATE,
        amplitude=harmony_mapping.loudness * HARMONY_GAIN, attack_seconds=CONFIG.attack_seconds,
    )
    live_harmony = render_live(harmony_mapping, voice_mute={"melody": True, "accent": True})

    write_wav(offline_harmony, OUTPUT_AUDIO_DIR / "b2_harmony.wav")
    write_wav(live_harmony, OUTPUT_AUDIO_DIR / "b3_harmony.wav")

    t_off, rms_off = windowed_rms(offline_harmony, SAMPLE_RATE, WINDOW_SECONDS)
    _, rms_live = windowed_rms(live_harmony, SAMPLE_RATE, WINDOW_SECONDS)
    _, cen_off = windowed_centroid(offline_harmony, SAMPLE_RATE, WINDOW_SECONDS)
    _, cen_live = windowed_centroid(live_harmony, SAMPLE_RATE, WINDOW_SECONDS)

    report_lines.append("## Harmony\n\n")
    report_lines.append("| t(s) | B2 RMS | B3 RMS | B2 centroid(Hz) | B3 centroid(Hz) |\n")
    report_lines.append("|---|---|---|---|---|\n")
    for i in range(0, len(t_off), 2):
        report_lines.append(
            f"| {t_off[i]:.2f} | {rms_off[i]:.4f} | {rms_live[i]:.4f} | {cen_off[i]:.1f} | {cen_live[i]:.1f} |\n"
        )
    report_lines.append("\n")

    # -----------------------------------------------------------------
    # 3. Full non-capture move, and capture+check move (with Accent)
    # -----------------------------------------------------------------
    offline_full = OrganicAudioRenderer(CONFIG).render(full_move_mapping).samples
    live_full = render_live(full_move_mapping)
    write_wav(offline_full, OUTPUT_AUDIO_DIR / "b2_full_move.wav")
    write_wav(live_full, OUTPUT_AUDIO_DIR / "b3_full_move.wav")

    offline_capture = OrganicAudioRenderer(CONFIG).render(capture_mapping).samples
    live_capture = render_live(capture_mapping, with_accent=True)
    write_wav(offline_capture, OUTPUT_AUDIO_DIR / "b2_capture_move.wav")
    write_wav(live_capture, OUTPUT_AUDIO_DIR / "b3_capture_move.wav")

    t_off, rms_off = windowed_rms(offline_full, SAMPLE_RATE, WINDOW_SECONDS)
    _, rms_live = windowed_rms(live_full, SAMPLE_RATE, WINDOW_SECONDS)
    report_lines.append("## Full non-capture move (melody+harmony mix)\n\n")
    report_lines.append("| t(s) | B2 RMS | B3 RMS |\n|---|---|---|\n")
    for i in range(0, len(t_off), 2):
        report_lines.append(f"| {t_off[i]:.2f} | {rms_off[i]:.4f} | {rms_live[i]:.4f} |\n")

    (OUTPUT_DIAG_DIR / "measurements.md").write_text("".join(report_lines), encoding="utf-8")
    print("".join(report_lines))
    print(f"\nWrote measurements to {OUTPUT_DIAG_DIR / 'measurements.md'}")
    print(f"Wrote comparison WAVs to {OUTPUT_AUDIO_DIR}")


if __name__ == "__main__":
    main()
