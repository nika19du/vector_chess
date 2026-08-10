import chess
import pytest

from audio.live_state import (
    SonificationState,
    interpolate_sonification_state,
    sonification_state_from_mapping,
)
from audio.mapping import (
    CHECK_DISSONANCE_FLOOR,
    CONSONANT_RATIO,
    build_audio_mapping,
    harmony_interval_for_balance,
)
from chess_engine.analyzer import analyze_position
from chess_engine.moves import execute_move


def _mapping_after(moves: list[str]):
    board = chess.Board()
    analysis = None

    for move_text in moves:
        move_details = execute_move(board, move_text)
        assert move_details is not None, f"expected {move_text} to be legal"
        analysis = analyze_position(board, move_details)

    return build_audio_mapping(analysis, dynamics=None)


SEGMENT_KEY = ("fen-a", "fen-b")


# ---------------------------------------------------------
# interpolate_sonification_state -- endpoint behavior
# ---------------------------------------------------------


def test_interpolate_at_t_zero_uses_balance_a_exactly():
    segment_mapping = _mapping_after(["e2e4"])

    state = interpolate_sonification_state(
        segment_mapping, balance_a=-6.0, balance_b=6.0, t=0.0, segment_key=SEGMENT_KEY
    )

    expected_at_a = harmony_interval_for_balance(-6.0, segment_mapping.is_check)
    assert state.harmony_interval_ratio == pytest.approx(expected_at_a)
    assert state.harmony_above_melody is False


def test_interpolate_at_t_one_matches_the_settled_state_for_that_segment():
    segment_mapping = _mapping_after(["e2e4"])

    interpolated = interpolate_sonification_state(
        segment_mapping,
        balance_a=0.0,
        balance_b=segment_mapping.attack_influence_balance,
        t=1.0,
        segment_key=SEGMENT_KEY,
    )
    settled = sonification_state_from_mapping(segment_mapping, segment_key=SEGMENT_KEY)

    assert interpolated == settled


def test_interpolate_past_one_and_before_zero_clamp_to_the_endpoints():
    segment_mapping = _mapping_after(["e2e4"])

    below = interpolate_sonification_state(
        segment_mapping, balance_a=1.0, balance_b=9.0, t=-5.0, segment_key=SEGMENT_KEY
    )
    at_zero = interpolate_sonification_state(
        segment_mapping, balance_a=1.0, balance_b=9.0, t=0.0, segment_key=SEGMENT_KEY
    )
    above = interpolate_sonification_state(
        segment_mapping, balance_a=1.0, balance_b=9.0, t=5.0, segment_key=SEGMENT_KEY
    )
    at_one = interpolate_sonification_state(
        segment_mapping, balance_a=1.0, balance_b=9.0, t=1.0, segment_key=SEGMENT_KEY
    )

    assert below == at_zero
    assert above == at_one


# ---------------------------------------------------------
# interpolate_sonification_state -- continuous harmony behavior
# ---------------------------------------------------------


def test_harmony_interval_is_the_linear_midpoint_at_t_half():
    segment_mapping = _mapping_after(["e2e4"])

    at_a = interpolate_sonification_state(
        segment_mapping, balance_a=0.0, balance_b=20.0, t=0.0, segment_key=SEGMENT_KEY
    )
    at_mid = interpolate_sonification_state(
        segment_mapping, balance_a=0.0, balance_b=20.0, t=0.5, segment_key=SEGMENT_KEY
    )
    at_b = interpolate_sonification_state(
        segment_mapping, balance_a=0.0, balance_b=20.0, t=1.0, segment_key=SEGMENT_KEY
    )

    assert at_a.harmony_interval_ratio < at_mid.harmony_interval_ratio < at_b.harmony_interval_ratio
    midpoint_expected = (at_a.harmony_interval_ratio + at_b.harmony_interval_ratio) / 2
    assert at_mid.harmony_interval_ratio == pytest.approx(midpoint_expected)


def test_harmony_above_melody_flips_sign_with_interpolated_balance():
    segment_mapping = _mapping_after(["e2e4"])

    negative_side = interpolate_sonification_state(
        segment_mapping, balance_a=-10.0, balance_b=-2.0, t=0.5, segment_key=SEGMENT_KEY
    )
    positive_side = interpolate_sonification_state(
        segment_mapping, balance_a=2.0, balance_b=10.0, t=0.5, segment_key=SEGMENT_KEY
    )

    assert negative_side.harmony_above_melody is False
    assert positive_side.harmony_above_melody is True


def test_check_dissonance_floor_holds_across_the_whole_segment_when_segment_is_a_check():
    # 1.e4 d5 2.Bb5+ -- the move that produced this segment is a check.
    segment_mapping = _mapping_after(["e2e4", "d7d5", "f1b5"])
    assert segment_mapping.is_check is True

    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        state = interpolate_sonification_state(
            segment_mapping, balance_a=0.0, balance_b=0.0, t=t, segment_key=SEGMENT_KEY
        )
        assert state.harmony_interval_ratio >= CHECK_DISSONANCE_FLOOR


def test_near_zero_balance_segment_stays_close_to_consonant_at_every_t():
    segment_mapping = _mapping_after(["e2e4"])
    assert segment_mapping.is_check is False

    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        state = interpolate_sonification_state(
            segment_mapping,
            balance_a=0.0,
            balance_b=0.0,
            t=t,
            segment_key=SEGMENT_KEY,
        )
        assert state.harmony_interval_ratio == pytest.approx(CONSONANT_RATIO)


# ---------------------------------------------------------
# interpolate_sonification_state -- discrete fields never move
# ---------------------------------------------------------


def test_discrete_identity_fields_are_constant_across_the_whole_segment():
    segment_mapping = _mapping_after(["e2e4", "e7e5"])  # black's move -> richer timbre

    states = [
        interpolate_sonification_state(
            segment_mapping, balance_a=-5.0, balance_b=5.0, t=t, segment_key=SEGMENT_KEY
        )
        for t in (0.0, 0.1, 0.5, 0.9, 1.0)
    ]

    assert all(s.pitch_hz == segment_mapping.pitch_hz for s in states)
    assert all(s.harmonic_richness == segment_mapping.harmonic_richness for s in states)
    assert all(s.loudness == segment_mapping.loudness for s in states)
    assert all(s.segment_key == SEGMENT_KEY for s in states)


# ---------------------------------------------------------
# sonification_state_from_mapping
# ---------------------------------------------------------


def test_sonification_state_from_mapping_traces_back_to_the_mapping_fields():
    mapping = _mapping_after(["g1f3"])  # Nf3

    state = sonification_state_from_mapping(mapping, segment_key=SEGMENT_KEY)

    assert state.harmony_interval_ratio == mapping.harmony_interval_ratio
    assert state.harmony_above_melody == (mapping.attack_influence_balance >= 0)
    assert state.pitch_hz == mapping.pitch_hz
    assert state.harmonic_richness == mapping.harmonic_richness
    assert state.loudness == mapping.loudness
    assert state.segment_key == SEGMENT_KEY


# ---------------------------------------------------------
# Determinism and mixer-field defaults
# ---------------------------------------------------------


def test_interpolation_is_deterministic_for_the_same_inputs():
    segment_mapping = _mapping_after(["e2e4"])

    first = interpolate_sonification_state(
        segment_mapping, balance_a=1.0, balance_b=7.0, t=0.3, segment_key=SEGMENT_KEY
    )
    second = interpolate_sonification_state(
        segment_mapping, balance_a=1.0, balance_b=7.0, t=0.3, segment_key=SEGMENT_KEY
    )

    assert first == second


def test_default_mixer_fields_are_neutral():
    segment_mapping = _mapping_after(["e2e4"])

    state = interpolate_sonification_state(
        segment_mapping, balance_a=0.0, balance_b=0.0, t=0.0, segment_key=SEGMENT_KEY
    )

    assert state.master_gain == 1.0
    assert state.voice_mute == {}
    assert state.voice_solo == {}


def test_mixer_field_dicts_are_copied_not_aliased():
    segment_mapping = _mapping_after(["e2e4"])
    mute = {"melody": True}

    state = interpolate_sonification_state(
        segment_mapping,
        balance_a=0.0,
        balance_b=0.0,
        t=0.0,
        segment_key=SEGMENT_KEY,
        voice_mute=mute,
    )
    mute["melody"] = False

    assert state.voice_mute == {"melody": True}


def test_sonification_state_is_frozen_and_hashable_by_equality():
    a = SonificationState(
        harmony_interval_ratio=1.0,
        harmony_above_melody=True,
        pitch_hz=220.0,
        harmonic_richness=1,
        loudness=0.5,
        segment_key=SEGMENT_KEY,
    )
    b = SonificationState(
        harmony_interval_ratio=1.0,
        harmony_above_melody=True,
        pitch_hz=220.0,
        harmonic_richness=1,
        loudness=0.5,
        segment_key=SEGMENT_KEY,
    )

    assert a == b
    with pytest.raises(Exception):
        a.pitch_hz = 440.0  # frozen dataclass -> FrozenInstanceError
