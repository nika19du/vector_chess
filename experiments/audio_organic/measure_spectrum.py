"""
B2 spectral/quantitative measurement -- companion to render_b2_prototype.py.

Replays the same Scholar's-mate sequence through OrganicAudioRenderer's
own voice-construction logic (audio/organic_renderer.py, audio/organic_synthesis.py),
measuring each move's isolated Melody voice: fundamental, resonant mode
peaks, spectral centroid, energy above 4kHz/8kHz, peak/RMS -- then
summarizes White vs Black brightness. Mirrors
experiments/audio_musicality/measure.py's methodology so B1's and B2's
numbers are directly comparable.

Read-only against production code; writes only under this module's own
output/diagnostics/ directory.

Run from the repo root:
    python -m experiments.audio_organic.measure_spectrum
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio.models import AudioMapping
from audio.organic_synthesis import BLACK_MODAL_PARAMS, WHITE_MODAL_PARAMS, modal_resonance
from experiments.audio_musicality.render_candidates import build_scholars_mate_mappings
from experiments.audio_musicality.synth_common import peak, rms
from experiments.audio_organic.render_b2_prototype import (
    DEFAULT_SAMPLE_RATE,
    render_isolated_accent,
    render_isolated_harmony,
    render_isolated_melody,
)

OUTPUT_PATH = Path(__file__).parent / "output" / "diagnostics" / "measurements.md"

HIGH_FREQUENCY_THRESHOLDS_HZ = (4000.0, 8000.0)
PEAK_PICK_COUNT = 4


@dataclass(frozen=True)
class VoiceMeasurement:
    label: str
    color: str | None
    fundamental_hz: float
    resonant_peaks_hz: list[float]
    spectral_centroid_hz: float
    energy_fraction_above: dict[float, float]
    peak_amplitude: float
    rms_amplitude: float


def _spectral_measurements(samples: np.ndarray, sample_rate: int) -> tuple[float, list[float], dict[float, float]]:
    magnitude = np.abs(np.fft.rfft(samples))
    power = magnitude ** 2
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)

    total_power = power.sum()
    centroid = float((magnitude * freqs).sum() / magnitude.sum()) if magnitude.sum() else 0.0

    # Resonant peaks: local maxima in the magnitude spectrum, sorted by
    # magnitude descending -- the modal frequencies actually present,
    # not just the theoretical mode_ratios table.
    is_local_max = np.r_[False, (magnitude[1:-1] > magnitude[:-2]) & (magnitude[1:-1] > magnitude[2:]), False]
    peak_indices = np.argsort(magnitude[is_local_max])[::-1]
    peak_freqs_sorted = freqs[is_local_max][peak_indices]
    resonant_peaks = [float(f) for f in peak_freqs_sorted[:PEAK_PICK_COUNT]]

    energy_fraction_above = {}
    for threshold in HIGH_FREQUENCY_THRESHOLDS_HZ:
        energy_fraction_above[threshold] = float(power[freqs > threshold].sum() / total_power) if total_power else 0.0

    return centroid, resonant_peaks, energy_fraction_above


def measure_voice(label: str, color: str | None, fundamental_hz: float, samples: np.ndarray) -> VoiceMeasurement:
    centroid, resonant_peaks, energy_fraction_above = _spectral_measurements(samples, DEFAULT_SAMPLE_RATE)

    return VoiceMeasurement(
        label=label,
        color=color,
        fundamental_hz=fundamental_hz,
        resonant_peaks_hz=resonant_peaks,
        spectral_centroid_hz=centroid,
        energy_fraction_above=energy_fraction_above,
        peak_amplitude=peak(samples),
        rms_amplitude=rms(samples),
    )


def measure_melody_across_game(mappings: list[AudioMapping]) -> list[VoiceMeasurement]:
    return [
        measure_voice(f"{mapping.move} ({mapping.color})", mapping.color, mapping.pitch_hz, render_isolated_melody(mapping))
        for mapping in mappings
    ]


def summarize(measurements: list[VoiceMeasurement]) -> str:
    white = [m for m in measurements if m.color == "white"]
    black = [m for m in measurements if m.color == "black"]

    def stats(label: str, values: list[float]) -> str:
        if not values:
            return f"- **{label}**: no data\n"
        return (
            f"- **{label}**: min={min(values):.1f}, median={sorted(values)[len(values) // 2]:.1f}, "
            f"max={max(values):.1f}, n={len(values)}\n"
        )

    lines: list[str] = ["# B2 Organic Synthesis -- Spectral Measurements\n\n"]

    lines.append("## Spectral centroid (Hz), White vs Black, same pipeline\n\n")
    lines.append(stats("White centroid", [m.spectral_centroid_hz for m in white]))
    lines.append(stats("Black centroid", [m.spectral_centroid_hz for m in black]))
    if white and black:
        white_median = sorted(m.spectral_centroid_hz for m in white)[len(white) // 2]
        black_median = sorted(m.spectral_centroid_hz for m in black)[len(black) // 2]
        lines.append(f"\nBlack/White median centroid ratio: {black_median / white_median:.3f} "
                      f"(B1's Phase A measurement of the old additive-synthesis pipeline was 3.2x higher for "
                      f"Black; here Black is *lower*, by design -- see audio/organic_synthesis.py's "
                      f"BLACK_MODAL_PARAMS docstring for why).\n")

    for threshold in HIGH_FREQUENCY_THRESHOLDS_HZ:
        white_frac = [m.energy_fraction_above[threshold] for m in white]
        black_frac = [m.energy_fraction_above[threshold] for m in black]
        lines.append(f"\n## Energy fraction above {threshold:.0f} Hz\n\n")
        lines.append(stats(f"White > {threshold:.0f}Hz", white_frac))
        lines.append(stats(f"Black > {threshold:.0f}Hz", black_frac))

    lines.append("\n## Peak / RMS (isolated Melody voice, pre-normalization)\n\n")
    lines.append(stats("White peak", [m.peak_amplitude for m in white]))
    lines.append(stats("Black peak", [m.peak_amplitude for m in black]))
    lines.append(stats("White RMS", [m.rms_amplitude for m in white]))
    lines.append(stats("Black RMS", [m.rms_amplitude for m in black]))

    lines.append("\n## Per-move detail\n\n")
    lines.append("| move | color | fundamental_hz | resonant_peaks_hz | centroid_hz | >4kHz frac | >8kHz frac | peak | rms |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for m in measurements:
        peaks_str = ", ".join(f"{f:.0f}" for f in m.resonant_peaks_hz)
        lines.append(
            f"| {m.label} | {m.color} | {m.fundamental_hz:.1f} | {peaks_str} | {m.spectral_centroid_hz:.1f} | "
            f"{m.energy_fraction_above[4000.0]:.4f} | {m.energy_fraction_above[8000.0]:.4f} | "
            f"{m.peak_amplitude:.3f} | {m.rms_amplitude:.3f} |\n"
        )

    return "".join(lines)


def measure_performance() -> str:
    """
    B3 feasibility preview (design only -- no live engine work here).
    Measures whole-note render cost (this module's actual offline path)
    and a simulated single-block cost at AudioEngine's own block size,
    to estimate real-time headroom for the B3 port described in the B2
    plan's live-state design.
    """

    duration_seconds = 1.6
    repeats = 50

    modal_resonance(440.0, WHITE_MODAL_PARAMS, duration_seconds, DEFAULT_SAMPLE_RATE, amplitude=1.0)  # warm-up

    lines = ["\n## Performance preview (B3 feasibility, not implemented)\n\n"]

    for label, params in (("White (4 modes)", WHITE_MODAL_PARAMS), ("Black (5 modes)", BLACK_MODAL_PARAMS)):
        start = time.perf_counter()
        for _ in range(repeats):
            modal_resonance(440.0, params, duration_seconds, DEFAULT_SAMPLE_RATE, amplitude=1.0)
        per_note_ms = (time.perf_counter() - start) / repeats * 1000
        lines.append(f"- Whole-note offline render, {label}: {per_note_ms:.3f} ms per {duration_seconds}s note "
                      f"({per_note_ms / duration_seconds:.3f} ms per second of audio, no per-sample recursion).\n")

    # Simulated single audio-engine block (audio/engine.py's own
    # DEFAULT_BLOCKSIZE) for one 5-mode voice -- the actual B3 live
    # cost per voice per callback, evaluated the same closed-form way
    # modal_resonance already does, just windowed to one block.
    blocksize = 256
    t = np.arange(blocksize) / DEFAULT_SAMPLE_RATE
    weights = np.array([1.0 / k for k in range(1, 6)])
    dampings = np.array([3.0 * (1 + 0.5 * (k - 1)) for k in range(1, 6)])
    freqs = np.array([440.0 * r for r in (0.5, 1.0, 2.0, 3.0, 4.0)])

    block_repeats = 3000
    start = time.perf_counter()
    for _ in range(block_repeats):
        total = np.zeros(blocksize)
        for w, d, f in zip(weights, dampings, freqs):
            total += w * np.exp(-d * t) * np.sin(2 * np.pi * f * t)
    per_block_seconds = (time.perf_counter() - start) / block_repeats
    block_audio_seconds = blocksize / DEFAULT_SAMPLE_RATE
    headroom_ratio = per_block_seconds / block_audio_seconds

    lines.append(
        f"- Simulated single-voice (5 modes) block render at blocksize={blocksize}: "
        f"{per_block_seconds * 1e6:.1f} us compute for {block_audio_seconds * 1000:.2f} ms of audio "
        f"-> {headroom_ratio * 100:.3f}% of the block period (~{1 / headroom_ratio:.0f}x real-time headroom "
        f"for one voice; three simultaneous voices -- Melody+Harmony+Accent -- would still use only "
        f"~{headroom_ratio * 300:.2f}% of the block period).\n"
    )
    lines.append(
        "- State per active voice for the B3 port: 3 float64 arrays of length <=5 "
        "(frequencies, weights, dampings) plus a scalar onset_time -- no delay-line buffer, "
        "no per-sample recursion (contrast Karplus-Strong's O(sample_rate/frequency) delay line "
        "per note, which does not vectorize across a block). See the B2 plan's ModalVoiceState design.\n"
    )

    return "".join(lines)


def main() -> None:
    mappings = build_scholars_mate_mappings()
    measurements = measure_melody_across_game(mappings)

    harmony_measurement = measure_voice(
        "harmony (move 1)", None, mappings[0].pitch_hz, render_isolated_harmony(mappings[0])
    )
    accent_measurement = measure_voice(
        "accent (capture+check move)", None, 0.0, render_isolated_accent(mappings[6])
    )

    summary = summarize(measurements)
    summary += measure_performance()
    summary += "\n## Harmony / Accent (single-move samples, for reference)\n\n"
    summary += (
        f"- Harmony: centroid={harmony_measurement.spectral_centroid_hz:.1f} Hz, "
        f"peak={harmony_measurement.peak_amplitude:.3f}, rms={harmony_measurement.rms_amplitude:.3f}\n"
    )
    summary += (
        f"- Accent: centroid={accent_measurement.spectral_centroid_hz:.1f} Hz, "
        f"peak={accent_measurement.peak_amplitude:.3f}, rms={accent_measurement.rms_amplitude:.3f}\n"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(summary, encoding="utf-8")

    print(summary)
    print(f"\nWrote measurements to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
