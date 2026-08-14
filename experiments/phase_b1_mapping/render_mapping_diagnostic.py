"""
Phase B1 mapping-only diagnostic render.

Renders the Scholar's-mate sequence through the actual PRODUCTION
pipeline exactly as console_app/export.py::export_game_audio does
(analyze_position -> analyze_dynamics -> build_audio_mapping ->
AudioRenderer -> write_wav), with B1's new mapping (2-octave register
clamp, scale-tone pitch quantization, quantized committed harmony,
spectral-safety harmonic ceiling) now live in audio/mapping.py, but
the SYNTHESIS itself (audio/synthesis.py, audio/renderer.py) is
UNCHANGED -- still the old sine/additive-sine palette. B2 is what
replaces the synthesis; this file exists only to let the new musical
*grammar* (register/pitch/harmony) be heard on its own, isolated from
any timbre change, per the user's explicit request:

    "one offline comparison WAV using the NEW B1 mapping with the
    current synthesis, explicitly labeled as a mapping-only diagnostic"

Expect this to still sound synthetic/oscillator-like -- that is
correct and expected for B1. Only the register/pitch/harmony grammar
changed.

Run from the repo root:
    python -m experiments.phase_b1_mapping.render_mapping_diagnostic
"""

from pathlib import Path

from audio.export import write_wav
from audio.mapping import build_audio_mapping
from audio.models import AudioClip
from audio.renderer import AudioRenderer
from analysis.dynamics import analyze_dynamics
from chess_engine.analyzer import analyze_position
from chess_engine.board import ChessGame
from chess_engine.models import DynamicsAnalysis, MoveAnalysis

import numpy as np

OUTPUT_DIR = Path(__file__).parent / "output"

SCHOLARS_MATE = [
    "e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7",
]


def main() -> None:
    game = ChessGame()
    analysis_history: list[MoveAnalysis] = []
    dynamics_history: list[DynamicsAnalysis] = []
    previous_analysis: MoveAnalysis | None = None

    for move_text in SCHOLARS_MATE:
        move_details = game.make_move(move_text)
        if move_details is None:
            raise ValueError(f"Illegal move in fixture: {move_text}")

        analysis = analyze_position(game.board, move_details)
        analysis_history.append(analysis)

        if previous_analysis is not None:
            dynamics_history.append(analyze_dynamics(previous=previous_analysis, current=analysis))

        previous_analysis = analysis

    renderer = AudioRenderer()
    clips = []

    for index, analysis in enumerate(analysis_history):
        dynamics = dynamics_history[index - 1] if index > 0 else None
        mapping = build_audio_mapping(analysis, dynamics)
        clip = renderer.render(mapping)
        clips.append(clip.samples)

        print(
            f"{index + 1}. {mapping.move} ({mapping.color}): "
            f"pitch={mapping.pitch_hz:.1f}Hz richness={mapping.harmonic_richness} "
            f"harmony_ratio={mapping.harmony_interval_ratio:.4f} "
            f"capture={mapping.is_capture} check={mapping.is_check}"
        )

    combined = np.concatenate(clips)
    out_path = OUTPUT_DIR / "b1_mapping_diagnostic.wav"
    write_wav(AudioClip(samples=combined, sample_rate=44100), out_path)

    print(f"\nMAPPING-ONLY DIAGNOSTIC -- new B1 register/pitch/harmony grammar,")
    print(f"unchanged (still old/synthetic) synthesis. Wrote {out_path}")


if __name__ == "__main__":
    main()
