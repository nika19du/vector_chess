"""
Renders the Scholar's-mate move sequence through all three candidate
palettes, normalizes each to a common RMS target so no candidate wins
the A/B/C comparison by simply being louder, and writes the resulting
WAVs (plus per-candidate isolated diagnostic clips) via production's
own audio.export.write_wav.

Run from the repo root:
    python -m experiments.audio_musicality.render_candidates
"""

from pathlib import Path

import numpy as np

from audio.mapping import build_audio_mapping
from chess_engine.analyzer import analyze_position
from chess_engine.board import ChessGame
from chess_engine.models import DynamicsAnalysis, MoveAnalysis

from experiments.audio_musicality import candidate_a, candidate_b, candidate_c
from experiments.audio_musicality.synth_common import (
    DEFAULT_SAMPLE_RATE,
    TARGET_RMS,
    normalize_to_rms,
    peak,
    rms,
    write_clip,
)
from analysis.dynamics import analyze_dynamics

OUTPUT_DIR = Path(__file__).parent / "output" / "audio"

# Same Scholar's-mate line used by measure.py and console_app's own
# regression tests -- reused so the WAV comparison is grounded in the
# same real game the quantitative diagnosis measured.
SCHOLARS_MATE = [
    "e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7",
]

CANDIDATES = {
    "A": candidate_a,
    "B": candidate_b,
    "C": candidate_c,
}

MOVE_DURATION_SECONDS = 1.6


def build_scholars_mate_mappings() -> list:
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

    mappings = []
    for index, analysis in enumerate(analysis_history):
        dynamics = dynamics_history[index - 1] if index > 0 else None
        mappings.append(build_audio_mapping(analysis, dynamics))

    return mappings


def render_candidate_track(candidate_module, mappings) -> np.ndarray:
    clips = [
        candidate_module.render_move(mapping, sample_rate=DEFAULT_SAMPLE_RATE, duration_seconds=MOVE_DURATION_SECONDS)
        for mapping in mappings
    ]
    return np.concatenate(clips)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mappings = build_scholars_mate_mappings()

    print(f"Rendering {len(mappings)} moves ({' '.join(m.move for m in mappings)}) through 3 candidates...\n")

    loudness_report_lines = [
        "# Candidate Loudness / Normalization Log\n",
        f"Target RMS: {TARGET_RMS}\n\n",
        "| candidate | pre-norm RMS | pre-norm peak | post-norm RMS | post-norm peak |\n",
        "|---|---|---|---|---|\n",
    ]

    for name, module in CANDIDATES.items():
        track = render_candidate_track(module, mappings)

        pre_rms, pre_peak = rms(track), peak(track)
        normalized = np.clip(normalize_to_rms(track, TARGET_RMS), -1.0, 1.0)
        post_rms, post_peak = rms(normalized), peak(normalized)

        out_path = OUTPUT_DIR / f"candidate_{name}.wav"
        write_clip(normalized, out_path)

        print(f"Candidate {name}: pre-norm RMS={pre_rms:.4f} peak={pre_peak:.4f} "
              f"-> post-norm RMS={post_rms:.4f} peak={post_peak:.4f} -> {out_path}")

        loudness_report_lines.append(
            f"| {name} | {pre_rms:.4f} | {pre_peak:.4f} | {post_rms:.4f} | {post_peak:.4f} |\n"
        )

        # Isolated diagnostic clips -- also normalized to the same target
        # so cross-candidate comparison of isolated voices is fair too.
        for voice_name, render_fn in (
            ("white_note", module.render_isolated_white_note),
            ("black_note", module.render_isolated_black_note),
            ("harmony", module.render_isolated_harmony),
            ("accent", module.render_isolated_accent),
        ):
            clip = render_fn(sample_rate=DEFAULT_SAMPLE_RATE)
            normalized_clip = np.clip(normalize_to_rms(clip, TARGET_RMS), -1.0, 1.0)
            write_clip(normalized_clip, OUTPUT_DIR / f"candidate_{name}_{voice_name}.wav")

    diagnostics_dir = Path(__file__).parent / "output" / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    (diagnostics_dir / "loudness_log.md").write_text("".join(loudness_report_lines), encoding="utf-8")

    print(f"\nWrote loudness log to {diagnostics_dir / 'loudness_log.md'}")
    print(f"All candidate WAVs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
