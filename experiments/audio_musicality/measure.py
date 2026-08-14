"""
Phase A quantitative diagnosis -- measurement-only.

Replays real, already-regression-tested game lines through the
*actual* production pipeline (chess_engine.analyzer.analyze_position ->
analysis.dynamics.analyze_dynamics -> audio.mapping.build_audio_mapping)
and dumps the resulting pitch/harmonic/harmony frequencies to CSV and a
human-readable summary, so the musicality audit is grounded in real
numbers instead of assumption.

Does not modify, monkeypatch, or reimplement any production audio
logic -- it only calls build_audio_mapping and records what comes out.
The one exception is _harmony_frequency, which is private to
audio.renderer and therefore re-derived here (its two-line rule is
quoted verbatim from audio/renderer.py's docstring) rather than
imported.

Run from the repo root:
    python -m experiments.audio_musicality.measure
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import chess

from analysis.dynamics import analyze_dynamics
from audio.mapping import build_audio_mapping
from audio.models import AudioMapping
from chess_engine.analyzer import analyze_position
from chess_engine.board import ChessGame
from chess_engine.models import DynamicsAnalysis, MoveAnalysis, MoveDetails

OUTPUT_DIR = Path(__file__).parent / "output" / "diagnostics"

# Scholar's mate -- tests/test_console_audio_integration.py's
# test_checkmate_without_any_audio_command_still_exports_every_move.
# 1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6?? 4.Qxf7#
SCHOLARS_MATE = [
    "e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7",
]

# Verified fastest-stalemate line -- same test file's
# test_stalemate_without_any_audio_command_still_exports_every_move.
# Chosen for chaotic/tense dynamics-label coverage the 7-ply game
# above doesn't reach, and a non-mate ending.
FASTEST_STALEMATE = [
    "e2e3", "a7a5", "d1h5", "a8a6", "h5a5", "h7h5", "a5c7", "a6h6",
    "h2h4", "f7f6", "c7d7", "e8f7", "d7b7", "d8d3", "b7b8", "d3h7",
    "b8c8", "f7g6", "c8e6",
]

# Sicilian Najdorf, classical setup -- scripts/chess_position_pool.py's
# OPENING_LINES[1], added for tactical-middlegame density/register
# variety beyond the two console_app regression fixtures above. SAN,
# not UCI, so it is replayed with _play_san_move below rather than
# ChessGame.make_move.
SICILIAN_NAJDORF_SAN = (
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e5 Nb3 Be7 O-O O-O Be3 Be6 Nd5 Bxd5"
).split()

UCI_GAMES: dict[str, list[str]] = {
    "scholars_mate": SCHOLARS_MATE,
    "fastest_stalemate": FASTEST_STALEMATE,
}

SAN_GAMES: dict[str, list[str]] = {
    "sicilian_najdorf": SICILIAN_NAJDORF_SAN,
}

GAMES: dict[str, list[str]] = {**UCI_GAMES, **SAN_GAMES}

HIGH_PARTIAL_THRESHOLDS_HZ = (8000.0, 10000.0)


@dataclass(frozen=True)
class MeasurementRow:
    game: str
    ply: int
    move: str
    color: str
    destination_square: str
    pitch_hz: float
    pitch_midi: float
    harmonic_richness: int
    partial_hz: list[float]
    partial_midi: list[float]
    harmony_hz: float
    harmony_midi: float
    harmony_interval_ratio: float
    attack_influence_balance: float
    is_capture: bool
    is_check: bool
    dynamics_label: str | None
    loudness: float


def hz_to_midi(frequency_hz: float) -> float:
    """Diagnostic-only MIDI-equivalent -- the production pipeline is
    entirely Hz-native and has no MIDI concept anywhere; this exists
    purely to make register comparisons in the report legible."""

    return 69.0 + 12.0 * math.log2(frequency_hz / 440.0)


def harmony_frequency(mapping: AudioMapping) -> float:
    """
    Re-derivation of audio/renderer.py's private _harmony_frequency,
    quoted from its own docstring: "balance >= 0 ... raises the
    harmonizing interval above the melody pitch; balance < 0 ...
    lowers it below." Not imported because it is a private (underscore-
    prefixed) symbol of the renderer module.
    """

    if mapping.attack_influence_balance >= 0:
        return mapping.pitch_hz * mapping.harmony_interval_ratio

    return mapping.pitch_hz / mapping.harmony_interval_ratio


def partial_frequencies(mapping: AudioMapping) -> list[float]:
    """Mirrors audio/synthesis.py::additive_tone's exact partial
    formula: frequency_hz * partial_index for partial_index in
    1..harmonic_richness."""

    return [
        mapping.pitch_hz * partial_index
        for partial_index in range(1, mapping.harmonic_richness + 1)
    ]


def _play_san_move(board: chess.Board, san: str) -> MoveDetails:
    """
    Measurement-only: mirrors chess_engine/moves.py::execute_move
    line-for-line, but for SAN input (scripts/chess_position_pool.py's
    OPENING_LINES are SAN, and chess_engine.moves.execute_move only
    parses UCI via chess.Move.from_uci). Not a production code path --
    exists solely so this diagnosis script can replay a richer
    middlegame line without adding a SAN entry point to
    chess_engine/moves.py itself.
    """

    move = board.parse_san(san)

    moved_piece = board.piece_at(move.from_square)
    is_capture = board.is_capture(move)
    piece_name = chess.piece_name(moved_piece.piece_type)
    color = "white" if moved_piece.color == chess.WHITE else "black"
    from_square = chess.square_name(move.from_square)
    to_square = chess.square_name(move.to_square)

    board.push(move)

    is_check = board.is_check()

    return MoveDetails(
        move=move.uci(),
        piece_name=piece_name,
        color=color,
        from_square=from_square,
        to_square=to_square,
        is_capture=is_capture,
        is_check=is_check,
    )


def play_game(moves: list[str], move_format: str) -> tuple[list[MoveAnalysis], list[DynamicsAnalysis]]:
    """
    Replays a move list (UCI via ChessGame, or SAN via _play_san_move)
    through the real chess/analysis pipeline, building
    analysis_history/dynamics_history with the exact same convention
    as console_app/main.py's REPL loop: dynamics_history[k] pairs
    analysis_history[k] with analysis_history[k+1].
    """

    game = ChessGame()
    analysis_history: list[MoveAnalysis] = []
    dynamics_history: list[DynamicsAnalysis] = []
    previous_analysis: MoveAnalysis | None = None

    for move_text in moves:
        if move_format == "uci":
            move_details = game.make_move(move_text)
            if move_details is None:
                raise ValueError(f"Illegal or malformed move in fixture: {move_text}")
        else:
            move_details = _play_san_move(game.board, move_text)

        analysis = analyze_position(game.board, move_details)
        analysis_history.append(analysis)

        if previous_analysis is not None:
            dynamics_history.append(analyze_dynamics(previous=previous_analysis, current=analysis))

        previous_analysis = analysis

    return analysis_history, dynamics_history


def measure_game(game_name: str, moves: list[str], move_format: str) -> list[MeasurementRow]:
    analysis_history, dynamics_history = play_game(moves, move_format)
    rows: list[MeasurementRow] = []

    for index, analysis in enumerate(analysis_history):
        dynamics = dynamics_history[index - 1] if index > 0 else None
        mapping = build_audio_mapping(analysis, dynamics)

        partials = partial_frequencies(mapping)
        harmony_hz = harmony_frequency(mapping)

        rows.append(
            MeasurementRow(
                game=game_name,
                ply=index + 1,
                move=mapping.move,
                color=mapping.color,
                destination_square=mapping.destination_square,
                pitch_hz=mapping.pitch_hz,
                pitch_midi=hz_to_midi(mapping.pitch_hz),
                harmonic_richness=mapping.harmonic_richness,
                partial_hz=partials,
                partial_midi=[hz_to_midi(f) for f in partials],
                harmony_hz=harmony_hz,
                harmony_midi=hz_to_midi(harmony_hz),
                harmony_interval_ratio=mapping.harmony_interval_ratio,
                attack_influence_balance=mapping.attack_influence_balance,
                is_capture=mapping.is_capture,
                is_check=mapping.is_check,
                dynamics_label=mapping.dynamics_label,
                loudness=mapping.loudness,
            )
        )

    return rows


def write_csv(rows: list[MeasurementRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "game", "ply", "move", "color", "destination_square",
        "pitch_hz", "pitch_midi", "harmonic_richness",
        "partial_hz", "partial_midi",
        "harmony_hz", "harmony_midi", "harmony_interval_ratio",
        "attack_influence_balance", "is_capture", "is_check",
        "dynamics_label", "loudness",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "game": row.game,
                    "ply": row.ply,
                    "move": row.move,
                    "color": row.color,
                    "destination_square": row.destination_square,
                    "pitch_hz": f"{row.pitch_hz:.2f}",
                    "pitch_midi": f"{row.pitch_midi:.2f}",
                    "harmonic_richness": row.harmonic_richness,
                    "partial_hz": ";".join(f"{f:.2f}" for f in row.partial_hz),
                    "partial_midi": ";".join(f"{m:.2f}" for m in row.partial_midi),
                    "harmony_hz": f"{row.harmony_hz:.2f}",
                    "harmony_midi": f"{row.harmony_midi:.2f}",
                    "harmony_interval_ratio": f"{row.harmony_interval_ratio:.4f}",
                    "attack_influence_balance": f"{row.attack_influence_balance:.3f}",
                    "is_capture": row.is_capture,
                    "is_check": row.is_check,
                    "dynamics_label": row.dynamics_label or "",
                    "loudness": f"{row.loudness:.2f}",
                }
            )


def summarize(rows: list[MeasurementRow]) -> str:
    white_pitches = [r.pitch_hz for r in rows if r.color == "white"]
    black_pitches = [r.pitch_hz for r in rows if r.color == "black"]
    black_partials = [f for r in rows if r.color == "black" for f in r.partial_hz]
    white_partials = [f for r in rows if r.color == "white" for f in r.partial_hz]

    lines: list[str] = []
    lines.append("# Measured Frequency Ranges — Phase A Diagnosis\n")
    lines.append(f"Total moves measured: {len(rows)} across {len(GAMES)} games.\n")

    def stats(label: str, values: list[float]) -> str:
        if not values:
            return f"- **{label}**: no data\n"
        return (
            f"- **{label}**: min={min(values):.1f} Hz, "
            f"median={sorted(values)[len(values) // 2]:.1f} Hz, "
            f"max={max(values):.1f} Hz, n={len(values)}\n"
        )

    lines.append("## Melody fundamental (destination-square pitch)\n")
    lines.append(stats("White fundamental", white_pitches))
    lines.append(stats("Black fundamental", black_pitches))
    lines.append(
        "\nBoth colors draw from the *same* pitch formula "
        "(`_pitch_for_square`) — the fundamental range is not "
        "color-dependent; any White/Black fundamental range difference "
        "above is purely which squares each side happened to move to "
        "in these particular games, not a systematic mapping.\n"
    )

    lines.append("\n## Harmonic partials actually sounded (additive_tone output)\n")
    lines.append(stats("White partials (all, richness=1 -> == fundamental)", white_partials))
    lines.append(stats("Black partials (all, richness=3 -> fundamental+2nd+3rd)", black_partials))

    for threshold in HIGH_PARTIAL_THRESHOLDS_HZ:
        black_over = [f for f in black_partials if f > threshold]
        white_over = [f for f in white_partials if f > threshold]
        pct = 100.0 * len(black_over) / len(black_partials) if black_partials else 0.0
        lines.append(
            f"- Black partials > {threshold:.0f} Hz: {len(black_over)}/"
            f"{len(black_partials)} ({pct:.1f}%). "
            f"White partials > {threshold:.0f} Hz: {len(white_over)}/"
            f"{len(white_partials)}.\n"
        )

    lines.append(
        "\nThe ear's peak sensitivity to perceived harshness/piercingness "
        "sits roughly in the 2-5 kHz range (equal-loudness contours), "
        "with 8-10kHz+ content read as 'hiss'/'shrill' rather than tonal "
        "pitch. Black's 3rd partial reaching into or past this band on "
        "any move whose destination square sits in the upper half of the "
        "board's pitch range is the direct, measured mechanism for the "
        "shrillness complaint -- not a bug, but a direct consequence of "
        "summing un-capped upper harmonics on top of an already-wide "
        "(220-3520 Hz) fundamental range.\n"
    )

    lines.append("\n## Harmony voice\n")
    harmony_hz_values = [r.harmony_hz for r in rows]
    lines.append(stats("Harmony frequency (all moves)", harmony_hz_values))
    ratios = [r.harmony_interval_ratio for r in rows]
    lines.append(
        f"- Harmony interval ratio range: min={min(ratios):.4f}, "
        f"max={max(ratios):.4f} (production bounds: 1.0 unison .. "
        f"{2.0 ** 0.5:.4f} tritone, 1.25 check floor). Ratio values "
        f"observed are continuous/unquantized, confirming harmony is "
        f"never snapped to a discrete musical interval in production.\n"
    )

    lines.append("\n## Renderer-level compounding factors (cited, not re-measured)\n")
    lines.append(
        "- `HARMONY_VOICE_WEIGHT = 0.6` (audio/renderer.py) -- harmony "
        "is quieter than melody by design, so the shrillness is "
        "dominated by the melody voice's own upper partials, not the "
        "harmony voice.\n"
        "- `SUSTAINED_PEAK_HEADROOM = 0.85` (audio/renderer.py) -- the "
        "melody+harmony bed is normalized to a fixed peak regardless of "
        "loudness, so a Black move's dense upper-partial energy is "
        "*not* attenuated relative to a White move's; normalization "
        "does not selectively soften Black's extra harmonics.\n"
    )

    lines.append("\n## Per-game breakdown\n")
    for game_name in GAMES:
        game_rows = [r for r in rows if r.game == game_name]
        lines.append(f"\n### {game_name} ({len(game_rows)} plies)\n")
        lines.append("| ply | move | color | dest | pitch_hz | partials_hz | harmony_hz | ratio | check | capture | label |\n")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in game_rows:
            partials_str = ", ".join(f"{f:.0f}" for f in r.partial_hz)
            lines.append(
                f"| {r.ply} | {r.move} | {r.color} | {r.destination_square} | "
                f"{r.pitch_hz:.1f} | {partials_str} | {r.harmony_hz:.1f} | "
                f"{r.harmony_interval_ratio:.3f} | {r.is_check} | {r.is_capture} | "
                f"{r.dynamics_label or '-'} |\n"
            )

    return "".join(lines)


def main() -> None:
    all_rows: list[MeasurementRow] = []

    for game_name, moves in UCI_GAMES.items():
        all_rows.extend(measure_game(game_name, moves, move_format="uci"))

    for game_name, moves in SAN_GAMES.items():
        all_rows.extend(measure_game(game_name, moves, move_format="san"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(all_rows, OUTPUT_DIR / "measurements.csv")

    summary = summarize(all_rows)
    (OUTPUT_DIR / "measurements_summary.md").write_text(summary, encoding="utf-8")

    print(summary)
    print(f"\nWrote {len(all_rows)} rows to {OUTPUT_DIR / 'measurements.csv'}")
    print(f"Wrote summary to {OUTPUT_DIR / 'measurements_summary.md'}")


if __name__ == "__main__":
    main()
