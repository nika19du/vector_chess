import chess
import pytest

from audio.mapping import (
    BASE_FREQUENCY_HZ,
    BLACK_HARMONIC_RICHNESS,
    CHECK_DISSONANCE_FLOOR,
    CONSONANT_RATIO,
    DISSONANT_RATIO,
    LOUDNESS_BY_LABEL,
    NEUTRAL_LOUDNESS,
    OCTAVE_SPAN,
    RANK_STEPS,
    WHITE_HARMONIC_RICHNESS,
    _harmonic_richness_for_color,
    _harmony_interval_for_balance,
    _loudness_for_dynamics_label,
    _pitch_for_square,
    build_audio_mapping,
)
from chess_engine.analyzer import analyze_position
from chess_engine.models import DynamicsAnalysis, MoveDetails
from chess_engine.moves import execute_move


def _analysis_after(moves: list[str]):
    board = chess.Board()
    analysis = None

    for move_text in moves:
        move_details = execute_move(board, move_text)
        assert move_details is not None, f"expected {move_text} to be legal"
        analysis = analyze_position(board, move_details)

    return analysis


def _analysis_for_board(
    board: chess.Board,
    color: str = "white",
    to_square: str = "a1",
) -> object:
    """
    Builds a MoveAnalysis for a hand-placed board (not reached via
    legal play), the same way tests/test_source_potential.py places
    pieces directly -- lets balance-sign tests control the position
    exactly instead of hoping a real opening produces one.
    """

    move_details = MoveDetails(
        move="0000",
        piece_name="king",
        color=color,
        from_square="a1",
        to_square=to_square,
        is_capture=False,
        is_check=board.is_check(),
    )

    return analyze_position(board, move_details)


def _dynamics_with_label(label: str) -> DynamicsAnalysis:
    return DynamicsAnalysis(
        previous_force=0,
        current_force=0,
        delta_force=0,
        white_mobility_delta=0,
        black_mobility_delta=0,
        attack_vectors_delta=0,
        heatmap_change=0,
        intensity=0.0,
        label=label,
    )


# ---------------------------------------------------------
# Destination square -> pitch
# ---------------------------------------------------------


def test_pitch_at_a1_is_base_frequency():
    assert _pitch_for_square("a1") == pytest.approx(BASE_FREQUENCY_HZ)


def test_pitch_walks_a_fixed_ratio_between_consecutive_ranks():
    expected_ratio = 2.0 ** (OCTAVE_SPAN / RANK_STEPS)

    for file_letter in "abcdefgh":
        for rank in range(1, 8):
            lower = _pitch_for_square(f"{file_letter}{rank}")
            upper = _pitch_for_square(f"{file_letter}{rank + 1}")

            assert upper / lower == pytest.approx(expected_ratio)


def test_pitch_spans_exactly_one_octave_across_the_files():
    assert _pitch_for_square("h1") == pytest.approx(2 * _pitch_for_square("a1"))


# ---------------------------------------------------------
# Color -> timbre
# ---------------------------------------------------------


def test_harmonic_richness_distinguishes_colors():
    assert _harmonic_richness_for_color("white") == 1
    assert _harmonic_richness_for_color("black") > 1


# ---------------------------------------------------------
# Attack Influence balance -> harmony
# ---------------------------------------------------------


def test_harmony_is_most_consonant_at_zero_balance():
    assert _harmony_interval_for_balance(0.0, is_check=False) == pytest.approx(
        CONSONANT_RATIO
    )


def test_harmony_moves_monotonically_toward_dissonance():
    magnitudes = [0.0, 2.0, 5.0, 10.0, 20.0, 40.0]
    ratios = [_harmony_interval_for_balance(m, is_check=False) for m in magnitudes]

    assert all(ratios[i] <= ratios[i + 1] for i in range(len(ratios) - 1))
    assert ratios[-1] == pytest.approx(DISSONANT_RATIO)


def test_check_forces_dissonance_floor_even_at_zero_balance():
    assert _harmony_interval_for_balance(0.0, is_check=True) >= CHECK_DISSONANCE_FLOOR


def test_check_does_not_reduce_an_already_higher_dissonance():
    without_check = _harmony_interval_for_balance(20.0, is_check=False)
    with_check = _harmony_interval_for_balance(20.0, is_check=True)

    assert with_check == pytest.approx(without_check)


def test_harmony_dissonance_is_symmetric_for_positive_and_negative_balance():
    # Dissonance intensity reflects how lopsided the position is, not
    # which side is ahead -- direction/mode is a renderer concern
    # (see audio.renderer._harmony_frequency), not a mapping concern.
    assert _harmony_interval_for_balance(-8.0, is_check=False) == pytest.approx(
        _harmony_interval_for_balance(8.0, is_check=False)
    )


def test_harmony_near_zero_balance_stays_close_to_consonant():
    ratio = _harmony_interval_for_balance(0.5, is_check=False)

    assert CONSONANT_RATIO < ratio < DISSONANT_RATIO
    assert ratio == pytest.approx(CONSONANT_RATIO, abs=0.05)


# ---------------------------------------------------------
# Dynamics label -> loudness
# ---------------------------------------------------------


def test_loudness_is_neutral_when_dynamics_is_none():
    assert _loudness_for_dynamics_label(None) == NEUTRAL_LOUDNESS


def test_loudness_increases_with_dynamics_label_severity():
    labels_in_order = ["calm", "active", "tense", "chaotic"]
    values = [_loudness_for_dynamics_label(label) for label in labels_in_order]

    assert all(values[i] < values[i + 1] for i in range(len(values) - 1))


# ---------------------------------------------------------
# build_audio_mapping integration
# ---------------------------------------------------------


def test_first_move_has_no_dynamics_label_and_neutral_loudness():
    analysis = _analysis_after(["e2e4"])

    mapping = build_audio_mapping(analysis, dynamics=None)

    assert mapping.dynamics_label is None
    assert mapping.loudness == NEUTRAL_LOUDNESS


def test_capturing_check_sets_both_flags_and_forces_dissonance_floor():
    # 1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6?? 4.Qxf7+ -- a capture that is also check.
    analysis = _analysis_after(
        ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]
    )

    mapping = build_audio_mapping(analysis, dynamics=None)

    assert mapping.is_capture is True
    assert mapping.is_check is True
    assert mapping.harmony_interval_ratio >= CHECK_DISSONANCE_FLOOR


def test_pure_capture_sets_capture_flag_without_check():
    # 1.e4 d5 2.exd5 -- a capture that is not a check.
    analysis = _analysis_after(["e2e4", "d7d5", "e4d5"])

    mapping = build_audio_mapping(analysis, dynamics=None)

    assert mapping.is_capture is True
    assert mapping.is_check is False


def test_pure_check_sets_check_flag_without_capture():
    # 1.e4 d5 2.Bb5+ -- a check that is not a capture.
    analysis = _analysis_after(["e2e4", "d7d5", "f1b5"])

    mapping = build_audio_mapping(analysis, dynamics=None)

    assert mapping.is_capture is False
    assert mapping.is_check is True
    assert mapping.harmony_interval_ratio >= CHECK_DISSONANCE_FLOOR


def test_black_move_gets_the_richer_timbre_through_build_audio_mapping():
    analysis = _analysis_after(["e2e4", "e7e5"])

    mapping = build_audio_mapping(analysis, dynamics=None)

    assert mapping.color == "black"
    assert mapping.harmonic_richness == BLACK_HARMONIC_RICHNESS


def test_white_move_gets_the_pure_timbre_through_build_audio_mapping():
    analysis = _analysis_after(["e2e4"])

    mapping = build_audio_mapping(analysis, dynamics=None)

    assert mapping.color == "white"
    assert mapping.harmonic_richness == WHITE_HARMONIC_RICHNESS


def test_pitch_hz_in_the_mapping_matches_the_destination_square_formula():
    analysis = _analysis_after(["g1f3"])  # Nf3

    mapping = build_audio_mapping(analysis, dynamics=None)

    assert mapping.destination_square == "f3"
    assert mapping.pitch_hz == pytest.approx(_pitch_for_square("f3"))


def test_positive_balance_position_is_preserved_and_pushes_past_consonant():
    board = chess.Board(None)
    board.set_piece_at(chess.D4, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.A8, chess.Piece(chess.KING, chess.BLACK))

    analysis = _analysis_for_board(board, color="white", to_square="d4")
    assert analysis.attack_influence_field.balance > 0

    mapping = build_audio_mapping(analysis, dynamics=None)

    assert mapping.attack_influence_balance > 0
    assert mapping.harmony_interval_ratio > CONSONANT_RATIO


def test_negative_balance_position_is_preserved_and_pushes_past_consonant():
    board = chess.Board(None)
    board.set_piece_at(chess.D5, chess.Piece(chess.QUEEN, chess.BLACK))
    board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.A8, chess.Piece(chess.KING, chess.BLACK))

    analysis = _analysis_for_board(board, color="black", to_square="d5")
    assert analysis.attack_influence_field.balance < 0

    mapping = build_audio_mapping(analysis, dynamics=None)

    assert mapping.attack_influence_balance < 0
    assert mapping.harmony_interval_ratio > CONSONANT_RATIO


def test_near_zero_balance_position_stays_consonant():
    # Two lone kings in opposite corners attack the same number of
    # squares each -- balance is exactly zero, no material anywhere
    # else on the board.
    board = chess.Board(None)
    board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.A8, chess.Piece(chess.KING, chess.BLACK))

    analysis = _analysis_for_board(board, color="white", to_square="a1")
    assert analysis.attack_influence_field.balance == pytest.approx(0.0)

    mapping = build_audio_mapping(analysis, dynamics=None)

    assert mapping.harmony_interval_ratio == pytest.approx(CONSONANT_RATIO)


@pytest.mark.parametrize("label", ["calm", "active", "tense", "chaotic"])
def test_build_audio_mapping_uses_the_loudness_table_for_each_label(label):
    analysis = _analysis_after(["e2e4"])

    mapping = build_audio_mapping(analysis, _dynamics_with_label(label))

    assert mapping.dynamics_label == label
    assert mapping.loudness == LOUDNESS_BY_LABEL[label]


def test_mapping_is_deterministic_for_the_same_inputs():
    first_analysis = _analysis_after(["e2e4"])
    second_analysis = _analysis_after(["e2e4"])

    first_mapping = build_audio_mapping(first_analysis, dynamics=None)
    second_mapping = build_audio_mapping(second_analysis, dynamics=None)

    assert first_mapping == second_mapping


def test_mapping_traces_back_to_the_source_fields():
    analysis = _analysis_after(["e2e4"])
    mapping = build_audio_mapping(analysis, dynamics=None)

    assert mapping.move == analysis.move
    assert mapping.color == analysis.color
    assert mapping.destination_square == analysis.to_square
    assert mapping.attack_influence_balance == analysis.attack_influence_field.balance
