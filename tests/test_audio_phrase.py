"""
Audio Layer 2 -- Rhythmic Layer, v2: pure-math tests for audio/phrase.py.

Supersedes tests/test_audio_pulse_pattern.py (deleted). That file tested
v1's `pulse_pattern_for_density`/`PULSE_PERIOD_SECONDS` -- an infinite,
looping k-of-n pattern generator. Those functions no longer exist: v1's
entire scheduling model (a pattern repeated forever via `% PULSE_SLOTS_
PER_PHRASE`) was the diagnosed cause of the "sounds like a metronome"
regression (see experiments/rhythmic_layer_v2/PHASE_A_AUDIT.md) and has
been replaced, not merely retuned. `pulse_density_for_intensity` and
`euclidean_pulse_pattern` themselves were never the problem -- both are
carried over unchanged (same tests apply) -- only what they fed into
changed, from an unbounded loop to a bounded, decaying `PhraseDescription`.
"""

import pytest

from audio.phrase import (
    AMPLITUDE_DECAY_PER_HIT,
    INTENSITY_DENSITY_SCALE,
    MAX_PHRASE_EVENTS,
    PHRASE_DURATION_SECONDS,
    PHRASE_PITCH_RATIO,
    PHRASE_SLOTS,
    PULSE_GAIN,
    EMPTY_PHRASE,
    PhraseDescription,
    PhraseEvent,
    build_phrase,
    euclidean_pulse_pattern,
    event_count_for_density,
    pulse_density_for_intensity,
)


# ---------------------------------------------------------
# Dynamics.intensity -> pulse_density (carried over unchanged from v1)
# ---------------------------------------------------------


def test_density_is_zero_at_zero_intensity():
    assert pulse_density_for_intensity(0.0) == 0.0


def test_density_is_zero_below_zero_intensity_defensively():
    assert pulse_density_for_intensity(-5.0) == 0.0


def test_density_is_one_at_the_scale_threshold():
    assert pulse_density_for_intensity(INTENSITY_DENSITY_SCALE) == 1.0


def test_density_clamps_to_one_above_the_scale_threshold():
    assert pulse_density_for_intensity(INTENSITY_DENSITY_SCALE * 5) == 1.0


def test_density_is_the_linear_midpoint_at_half_the_scale():
    assert pulse_density_for_intensity(INTENSITY_DENSITY_SCALE / 2) == pytest.approx(0.5)


def test_density_is_monotonic_non_decreasing_with_intensity():
    intensities = [0.0, 1.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 25.0, 40.0]
    densities = [pulse_density_for_intensity(i) for i in intensities]
    assert all(densities[i] <= densities[i + 1] for i in range(len(densities) - 1))


def test_density_scale_matches_analyze_dynamics_chaotic_threshold():
    assert INTENSITY_DENSITY_SCALE == 20.0


# ---------------------------------------------------------
# density -> event count (renamed from v1's active_slot_count_for_density;
# same shape, new ceiling -- MAX_PHRASE_EVENTS=4, and a floor of 1, not 0)
# ---------------------------------------------------------


def test_event_count_is_one_at_zero_density():
    # Unlike v1 (0 active slots possible), a phrase always has at least
    # one event -- a move is always a musical event, even a calm one.
    assert event_count_for_density(0.0) == 1


def test_event_count_is_max_at_full_density():
    assert event_count_for_density(1.0) == MAX_PHRASE_EVENTS


def test_event_count_clamps_density_outside_zero_one():
    assert event_count_for_density(-0.5) == 1
    assert event_count_for_density(1.5) == MAX_PHRASE_EVENTS


def test_event_count_is_monotonic_non_decreasing_in_density():
    densities = [i / 20.0 for i in range(21)]
    counts = [event_count_for_density(d) for d in densities]
    assert all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1))
    assert min(counts) == 1
    assert max(counts) == MAX_PHRASE_EVENTS


# ---------------------------------------------------------
# k-of-n Euclidean placement (carried over unchanged from v1 -- this
# primitive was never the diagnosed problem)
# ---------------------------------------------------------


def test_pattern_has_all_false_slots_at_k_zero():
    assert euclidean_pulse_pattern(0, PHRASE_SLOTS) == (False,) * PHRASE_SLOTS


def test_pattern_has_all_true_slots_at_k_equals_n():
    assert euclidean_pulse_pattern(PHRASE_SLOTS, PHRASE_SLOTS) == (True,) * PHRASE_SLOTS


@pytest.mark.parametrize("k", range(0, PHRASE_SLOTS + 1))
def test_pattern_has_exactly_k_active_slots(k):
    pattern = euclidean_pulse_pattern(k, PHRASE_SLOTS)
    assert sum(pattern) == k
    assert len(pattern) == PHRASE_SLOTS


def test_pattern_is_deterministic():
    assert euclidean_pulse_pattern(3, 5) == euclidean_pulse_pattern(3, 5)


def test_pattern_never_fills_every_slot_for_a_real_phrase_event_count():
    # MAX_PHRASE_EVENTS (4) < PHRASE_SLOTS (5) by construction -- a real
    # phrase (event_count in [1, MAX_PHRASE_EVENTS]) always leaves at
    # least one slot silent, even at maximum density.
    pattern = euclidean_pulse_pattern(MAX_PHRASE_EVENTS, PHRASE_SLOTS)
    assert pattern.count(False) >= 1


# ---------------------------------------------------------
# build_phrase -- the new core: finite, deterministic, decaying
# ---------------------------------------------------------


def _phrase(density: float, *, pitch_hz: float = 440.0, harmony_interval_ratio: float = 1.25, color: str = "white") -> PhraseDescription:
    return build_phrase(
        pulse_density=density, pitch_hz=pitch_hz, harmony_interval_ratio=harmony_interval_ratio, color=color
    )


def test_phrase_at_zero_density_has_exactly_one_event_at_onset():
    phrase = _phrase(0.0)
    assert len(phrase.events) == 1
    assert phrase.events[0].onset_seconds == 0.0


def test_phrase_at_full_density_has_at_most_max_events():
    phrase = _phrase(1.0)
    assert len(phrase.events) <= MAX_PHRASE_EVENTS
    assert len(phrase.events) >= 1


def test_phrase_event_count_is_monotonic_non_decreasing_with_density():
    densities = [i / 10.0 for i in range(11)]
    counts = [len(_phrase(d).events) for d in densities]
    assert all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1))


def test_phrase_duration_is_fixed_regardless_of_density():
    # v1's period was already a fixed constant; v2 extends the same
    # discipline to the whole phrase envelope -- only internal event
    # count/placement/amplitude vary with density, never the phrase's
    # own length.
    assert _phrase(0.0).duration_seconds == PHRASE_DURATION_SECONDS
    assert _phrase(1.0).duration_seconds == PHRASE_DURATION_SECONDS


def test_all_events_land_within_the_phrase_window():
    for density in (0.0, 0.25, 0.5, 0.75, 1.0):
        phrase = _phrase(density)
        for event in phrase.events:
            assert 0.0 <= event.onset_seconds < PHRASE_DURATION_SECONDS


def test_phrase_generation_is_deterministic():
    first = _phrase(0.6, pitch_hz=330.0, harmony_interval_ratio=1.4, color="black")
    second = _phrase(0.6, pitch_hz=330.0, harmony_interval_ratio=1.4, color="black")
    assert first == second


def test_phrase_amplitude_decays_across_successive_events():
    phrase = _phrase(1.0)  # max events, so a real decay sequence exists
    amplitudes = [event.amplitude for event in phrase.events]
    assert all(amplitudes[i] > amplitudes[i + 1] for i in range(len(amplitudes) - 1))
    assert amplitudes[0] == pytest.approx(PULSE_GAIN)


def test_phrase_amplitude_decay_matches_the_documented_factor():
    phrase = _phrase(1.0)
    for i, event in enumerate(phrase.events):
        assert event.amplitude == pytest.approx(PULSE_GAIN * (AMPLITUDE_DECAY_PER_HIT**i))


def test_phrase_amplitude_never_exceeds_the_primary_gain():
    for density in (0.0, 0.3, 0.6, 1.0):
        for event in _phrase(density).events:
            assert 0.0 < event.amplitude <= PULSE_GAIN


# ---------------------------------------------------------
# Pitch: derives from Melody, not an independent fixed frequency
# ---------------------------------------------------------


def test_phrase_pitch_is_derived_from_melody_pitch():
    phrase = _phrase(1.0, pitch_hz=220.0, harmony_interval_ratio=1.5)
    call_pitch = 220.0 * PHRASE_PITCH_RATIO
    response_pitch = call_pitch * 1.5
    for event in phrase.events:
        assert event.pitch_hz == pytest.approx(call_pitch) or event.pitch_hz == pytest.approx(response_pitch)


def test_different_melody_pitches_produce_different_phrase_pitches():
    low = _phrase(1.0, pitch_hz=220.0)
    high = _phrase(1.0, pitch_hz=440.0)
    assert low.events[0].pitch_hz != high.events[0].pitch_hz


# ---------------------------------------------------------
# White/Black: role-order dialogue, never a register/pitch-height split
# ---------------------------------------------------------


def test_white_phrase_opens_on_the_call_role():
    phrase = _phrase(1.0, pitch_hz=220.0, harmony_interval_ratio=1.5, color="white")
    call_pitch = 220.0 * PHRASE_PITCH_RATIO
    assert phrase.events[0].pitch_hz == pytest.approx(call_pitch)


def test_black_phrase_opens_on_the_response_role():
    phrase = _phrase(1.0, pitch_hz=220.0, harmony_interval_ratio=1.5, color="black")
    call_pitch = 220.0 * PHRASE_PITCH_RATIO
    response_pitch = call_pitch * 1.5
    assert phrase.events[0].pitch_hz == pytest.approx(response_pitch)


def test_white_and_black_draw_from_the_identical_pitch_vocabulary():
    """
    The explicit regression this test guards: White/Black must NOT be
    distinguished by register/pitch height (the B1/B2 lesson this
    project already learned once with Melody). Both colors' phrases are
    built from exactly the same two pitches (call_pitch, response_pitch)
    -- only which one opens the phrase differs.
    """

    white = _phrase(1.0, pitch_hz=220.0, harmony_interval_ratio=1.5, color="white")
    black = _phrase(1.0, pitch_hz=220.0, harmony_interval_ratio=1.5, color="black")

    white_pitches = {round(e.pitch_hz, 6) for e in white.events}
    black_pitches = {round(e.pitch_hz, 6) for e in black.events}
    assert white_pitches == black_pitches


def test_white_and_black_use_the_same_amplitude_envelope():
    white = _phrase(1.0, color="white")
    black = _phrase(1.0, color="black")
    white_amplitudes = [e.amplitude for e in white.events]
    black_amplitudes = [e.amplitude for e in black.events]
    assert white_amplitudes == black_amplitudes


def test_white_and_black_roles_are_a_clean_swap_not_independent_designs():
    # Every even hit_index is "call" for White and "response" for Black,
    # and vice versa for odd -- a strict role swap, not two unrelated
    # rulesets that happen to sound different.
    call_pitch = 220.0 * PHRASE_PITCH_RATIO
    response_pitch = call_pitch * 1.5
    white = _phrase(1.0, pitch_hz=220.0, harmony_interval_ratio=1.5, color="white")
    black = _phrase(1.0, pitch_hz=220.0, harmony_interval_ratio=1.5, color="black")

    for i, (white_event, black_event) in enumerate(zip(white.events, black.events)):
        if white_event.pitch_hz == pytest.approx(call_pitch):
            assert black_event.pitch_hz == pytest.approx(response_pitch)
        else:
            assert black_event.pitch_hz == pytest.approx(call_pitch)


# ---------------------------------------------------------
# EMPTY_PHRASE -- the SonificationState default
# ---------------------------------------------------------


def test_empty_phrase_has_no_events():
    assert EMPTY_PHRASE.events == ()
    assert EMPTY_PHRASE.duration_seconds == 0.0
