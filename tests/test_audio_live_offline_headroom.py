"""
Headroom policy tests, plus live/offline modal parity where meaningful
(B3: Live Modal/Organic AudioEngine Port).

The live sustained mix follows the same SUSTAINED_PEAK_HEADROOM policy
as the offline organic renderer -- both import the identical constant
from audio/organic_renderer.py (the B2 offline modal renderer), not two
independently-defined values. See audio/engine.py's module docstring
for the exact causal, analytic-bound equivalent implemented (the
offline renderer can measure a whole clip's actual peak before scaling;
a real-time callback cannot look ahead, so this scales using the
currently-audible voices' known smoothed amplitudes instead --
intentionally approximate, not bit-identical, but equivalent in intent
and effect).

B3 note on what "parity" means here: live and OrganicAudioRenderer now
share the same modal DSP family (WHITE_MODES/BLACK_MODES/HARMONY_MODES/
ACCENT_MODES) and the same headroom/final-clip policy, but deliberately
NOT the same temporal envelope for Melody/Harmony -- offline applies
each mode's own exp(-damping*t) decay across a fixed 1.6s window; live
applies no per-mode decay and instead shapes the note via the
unchanged, retrigger-safe _ArticulationEnvelope (see engine.py's
docstring for why). Exact peak equality between the two paths is
therefore not a meaningful invariant to test for Melody/Harmony -- the
tests below assert the policy-level guarantees that DO hold on both
sides (never exceeds headroom/clip bounds, accent added on top
unscaled, muting only ever reduces the bound, White/Black stay
distinguishable) rather than bit-exact numeric matches.
"""

import chess
import numpy as np
import pytest

from audio.backend import FakeAudioBackend
from audio.engine import DEFAULT_BLOCKSIZE, DEFAULT_SAMPLE_RATE, AccentTrigger, AudioEngine
from audio.live_state import sonification_state_from_mapping
from audio.mapping import LOUDNESS_BY_LABEL, build_audio_mapping
from audio.organic_renderer import HARMONY_GAIN, SUSTAINED_PEAK_HEADROOM, OrganicAudioRenderer
from audio.voices import build_default_voice_registry
from chess_engine.analyzer import analyze_position
from chess_engine.models import DynamicsAnalysis
from chess_engine.moves import execute_move

CLIP_DURATION_SECONDS = 1.6  # matches RenderConfig's default offline clip duration


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


def _mapping_for(moves: list[str], label: str | None):
    board = chess.Board()
    analysis = None
    for move_text in moves:
        details = execute_move(board, move_text)
        assert details is not None, move_text
        analysis = analyze_position(board, details)
    dynamics = _dynamics_with_label(label) if label is not None else None
    return build_audio_mapping(analysis, dynamics)


def _quiet_move_mapping(label: str | None):
    return _mapping_for(["g1f3"], label)  # Nf3 -- no capture, no check


def _capturing_check_mapping(label: str | None):
    # 1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6?? 4.Qxf7+ -- a single move that is
    # both a capture and a check.
    return _mapping_for(["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"], label)


def _pure_capture_mapping(label: str | None):
    return _mapping_for(["e2e4", "d7d5", "e4d5"], label)  # 1.e4 d5 2.exd5


def _live_peak(mapping, *, voice_mute=None) -> float:
    backend = FakeAudioBackend()
    registry = build_default_voice_registry()
    engine = AudioEngine(backend, registry, sample_rate=DEFAULT_SAMPLE_RATE, blocksize=DEFAULT_BLOCKSIZE)
    engine.start()
    stream = backend.streams[-1]

    # Scrub mode exercises the headroom scaling formula in _render_block
    # via the continuous-smoothing code path, reaching a steady-state
    # peak in one block instead of needing to render through
    # _ArticulationEnvelope's attack/plateau/decay shape -- this file's
    # actual subject is the headroom SCALING FORMULA, not articulation
    # timing (covered separately in tests/test_audio_articulation.py).
    engine.set_scrub_active(True)

    engine.publish(sonification_state_from_mapping(mapping, ("a", "b"), voice_mute=voice_mute or {}))
    if mapping.is_capture or mapping.is_check:
        kind = "capture+check" if (mapping.is_capture and mapping.is_check) else ("capture" if mapping.is_capture else "check")
        engine.push_event(AccentTrigger(kind=kind, loudness=mapping.loudness))

    block_count = int(CLIP_DURATION_SECONDS * DEFAULT_SAMPLE_RATE / DEFAULT_BLOCKSIZE) + 1
    peak = 0.0
    for _ in range(block_count):
        block = stream.render_block(DEFAULT_BLOCKSIZE)
        peak = max(peak, float(np.abs(block).max()))
    return peak


ALL_LABELS = ["calm", "active", "tense", "chaotic", None]


# ---------------------------------------------------------
# Shared headroom constant -- a regression guard for the "one canonical
# headroom policy" claim itself: engine.py and organic_renderer.py must
# import the identical value, not two independently-defined ones that
# could drift apart.
# ---------------------------------------------------------


def test_engine_and_offline_organic_renderer_share_the_identical_headroom_constant():
    from audio.engine import SUSTAINED_PEAK_HEADROOM as engine_headroom

    assert engine_headroom == SUSTAINED_PEAK_HEADROOM


# ---------------------------------------------------------
# Quiet moves (sustained bed only, no accent): headroom scaling keeps
# the live peak within the target ceiling at every realistic loudness
# tier -- the exact regression 5f.3a fixed, still true under B3's
# modal timbre even though the exact numeric peak achieved is no
# longer bit-identical to the old additive-oscillator formula's own
# realized peak (see module docstring).
# ---------------------------------------------------------


@pytest.mark.parametrize("label", ALL_LABELS)
def test_live_peak_stays_within_headroom_for_quiet_moves(label):
    mapping = _quiet_move_mapping(label)

    peak = _live_peak(mapping)

    assert peak > 0.0  # audible
    assert peak <= SUSTAINED_PEAK_HEADROOM + 1e-3  # never exceeds the target ceiling


@pytest.mark.parametrize("label", ["tense", "chaotic"])
def test_live_no_longer_hard_clips_a_quiet_move_at_the_previously_clipping_tiers(label):
    """
    The exact regression Phase 5f.3a fixed: before it, "tense" and
    "chaotic" hit the hard clip bound (peak == 1.0) for a quiet move
    with no accent at all. They still don't, under B3's modal timbre.
    """

    assert LOUDNESS_BY_LABEL[label] > 1.0 / (1.0 + HARMONY_GAIN)  # would clip without headroom

    mapping = _quiet_move_mapping(label)
    peak = _live_peak(mapping)

    assert peak < 1.0
    assert peak <= SUSTAINED_PEAK_HEADROOM + 1e-3


# ---------------------------------------------------------
# With and without Harmony (mute) -- headroom only counts audible
# voices, matching the fact that a muted voice cannot clip anything.
# ---------------------------------------------------------


@pytest.mark.parametrize("label", ALL_LABELS)
def test_headroom_still_applies_with_only_melody_audible(label):
    mapping = _quiet_move_mapping(label)

    peak_with_harmony = _live_peak(mapping)
    peak_melody_only = _live_peak(mapping, voice_mute={"harmony": True})

    assert peak_melody_only <= SUSTAINED_PEAK_HEADROOM + 1e-4
    # Muting a voice can only ever reduce or match the peak, never raise it
    # (both converge to exactly SUSTAINED_PEAK_HEADROOM when either alone
    # already exceeds it, so a small floating-point tolerance is needed).
    assert peak_melody_only <= peak_with_harmony + 1e-4


# ---------------------------------------------------------
# Capture Accent -- must remain audible, and must NOT be scaled by the
# sustained-bed headroom (mixed in afterward, exactly like offline).
# ---------------------------------------------------------


@pytest.mark.parametrize("label", ALL_LABELS)
def test_capture_accent_remains_audible_alongside_sustained_headroom(label):
    mapping = _pure_capture_mapping(label)
    assert mapping.is_capture is True

    backend = FakeAudioBackend()
    registry = build_default_voice_registry()
    engine = AudioEngine(backend, registry, sample_rate=DEFAULT_SAMPLE_RATE, blocksize=DEFAULT_BLOCKSIZE)
    engine.start()
    stream = backend.streams[-1]

    engine.set_scrub_active(True)  # see _live_peak's own comment on why
    engine.publish(sonification_state_from_mapping(mapping, ("a", "b")))
    engine.push_event(AccentTrigger("capture", loudness=mapping.loudness))

    first_block = stream.render_block(DEFAULT_BLOCKSIZE)
    assert engine._accent_state.active is True
    assert np.abs(first_block).max() > 0.0  # the accent audibly contributes


def test_capture_accent_raises_the_live_peak_above_the_sustained_only_peak():
    """
    Modal-parity-where-meaningful: rather than comparing live to the old
    additive-oscillator offline renderer, this checks the accent's own
    documented contract directly -- it is layered on top of the
    sustained bed, unnormalized, so a capturing move's peak must be at
    least as high as the same move rendered without its accent.
    """

    for label in ALL_LABELS:
        capture_mapping = _pure_capture_mapping(label)
        non_capture_mapping = _mapping_for(["e2e4", "d7d5"], label)  # same position, no capture

        capture_peak = _live_peak(capture_mapping)
        non_capture_peak = _live_peak(non_capture_mapping)

        assert capture_peak >= non_capture_peak - 1e-3, f"label={label}"


# ---------------------------------------------------------
# Capture + check together -- both paths respect their own safety clip;
# offline's own already-accepted "chaotic capture" clipping (docs/audio.md's
# documented, accepted loudness-compression limitation) is unaffected by
# B3, since it is produced entirely by OrganicAudioRenderer, unchanged.
# ---------------------------------------------------------


@pytest.mark.parametrize("label", ALL_LABELS)
def test_capture_plus_check_stays_within_the_final_safety_bound_on_both_paths(label):
    mapping = _capturing_check_mapping(label)
    assert mapping.is_capture is True
    assert mapping.is_check is True

    offline_peak = float(np.abs(OrganicAudioRenderer().render(mapping).samples).max())
    live_peak = _live_peak(mapping)

    assert offline_peak <= 1.0 + 1e-6
    assert live_peak <= 1.0 + 1e-6


def test_capture_plus_check_at_chaotic_still_respects_offlines_own_accepted_clip_behavior():
    """
    The one already-accepted exception: docs/audio.md documents that a
    capturing move's accent can push OrganicAudioRenderer's own final
    hard clip at high loudness -- unchanged by B3, since B3 does not
    touch audio/organic_renderer.py at all. Live is not required to hit
    the same exact clip (its temporal envelope is deliberately
    different -- see module docstring); it must only stay within its
    own safety bound.
    """

    mapping = _capturing_check_mapping("chaotic")
    offline_peak = float(np.abs(OrganicAudioRenderer().render(mapping).samples).max())
    live_peak = _live_peak(mapping)

    assert offline_peak == pytest.approx(1.0, abs=1e-6)
    assert live_peak <= 1.0 + 1e-6


# ---------------------------------------------------------
# The final safety clip is a last resort, not the headroom mechanism --
# for a quiet move it should essentially never engage.
# ---------------------------------------------------------


@pytest.mark.parametrize("label", ALL_LABELS)
def test_final_safety_clip_does_not_engage_for_quiet_moves_at_any_tier(label):
    mapping = _quiet_move_mapping(label)
    peak = _live_peak(mapping)

    assert peak <= SUSTAINED_PEAK_HEADROOM + 1e-3  # never reaches the [-1, 1] clip bound


# ---------------------------------------------------------
# Live/offline modal parity where meaningful (B3 requirement): both
# paths draw Melody's timbre from the same WHITE_MODES/BLACK_MODES
# arrays, so color must remain audibly distinguishable on both, and
# neither should ever produce a non-finite sample.
# ---------------------------------------------------------


def test_white_and_black_are_distinguishable_on_both_live_and_offline_paths():
    white_mapping = _quiet_move_mapping(None)
    assert white_mapping.color == "white"
    from dataclasses import replace

    black_mapping = replace(white_mapping, color="black")

    offline_white = OrganicAudioRenderer().render(white_mapping).samples
    offline_black = OrganicAudioRenderer().render(black_mapping).samples
    assert not np.array_equal(offline_white, offline_black)

    def _live_block(mapping):
        backend = FakeAudioBackend()
        registry = build_default_voice_registry()
        engine = AudioEngine(backend, registry, sample_rate=DEFAULT_SAMPLE_RATE, blocksize=DEFAULT_BLOCKSIZE)
        engine.start()
        stream = backend.streams[-1]
        engine.set_scrub_active(True)
        engine.publish(sonification_state_from_mapping(mapping, ("a", "b")))
        return stream.render_block(DEFAULT_BLOCKSIZE)

    live_white = _live_block(white_mapping)
    live_black = _live_block(black_mapping)
    assert not np.array_equal(live_white, live_black)


def test_live_peak_never_produces_non_finite_samples_across_all_scenarios():
    for label in ALL_LABELS:
        for mapping in (_quiet_move_mapping(label), _pure_capture_mapping(label), _capturing_check_mapping(label)):
            backend = FakeAudioBackend()
            registry = build_default_voice_registry()
            engine = AudioEngine(backend, registry, sample_rate=DEFAULT_SAMPLE_RATE, blocksize=DEFAULT_BLOCKSIZE)
            engine.start()
            stream = backend.streams[-1]
            engine.publish(sonification_state_from_mapping(mapping, ("a", "b")))
            engine.push_note_trigger()
            if mapping.is_capture or mapping.is_check:
                engine.push_event(AccentTrigger("capture", loudness=mapping.loudness))
            block = stream.render_block(DEFAULT_BLOCKSIZE)
            assert np.all(np.isfinite(block)), f"label={label} move={mapping.move}"
