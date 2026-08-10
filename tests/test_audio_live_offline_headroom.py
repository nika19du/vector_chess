"""
Phase 5f.3a hardening: the live sustained mix now follows the offline
renderer's own SUSTAINED_PEAK_HEADROOM policy (imported, not
reinvented). See audio/engine.py's module docstring for the exact
causal, analytic-bound equivalent implemented (the offline renderer can
measure a whole clip's actual peak before scaling; a real-time callback
cannot look ahead, so this scales using the currently-audible voices'
known smoothed amplitudes instead -- intentionally approximate, not
bit-identical, but equivalent in intent and effect).

Measured result (see this phase's report for the full before/after
table): for every non-capturing, non-checking move at every realistic
MVP loudness tier, the live sustained-bed peak now matches the offline
renderer's peak EXACTLY. For a capturing/checking move (accent added on
top, unnormalized, exactly like the offline renderer's own
`mix(sustained, accent)`), live peaks are always <= the offline
renderer's own peak for the same mapping -- never worse, and the
already-accepted offline "chaotic capture" clipping (docs/audio.md's
documented, accepted loudness-compression limitation) is reproduced
identically at the one loudness tier where it was already occurring
offline, not newly introduced.
"""

import chess
import numpy as np
import pytest

from audio.backend import FakeAudioBackend
from audio.engine import DEFAULT_BLOCKSIZE, DEFAULT_SAMPLE_RATE, AccentTrigger, AudioEngine
from audio.live_state import sonification_state_from_mapping
from audio.mapping import LOUDNESS_BY_LABEL, build_audio_mapping
from audio.renderer import HARMONY_VOICE_WEIGHT, SUSTAINED_PEAK_HEADROOM, AudioRenderer
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

    # Phase 5f.4a: Melody/Harmony's amplitude in committed/idle mode now
    # follows a finite attack -> plateau -> decay envelope instead of
    # chasing its target forever -- correct for how the voices actually
    # sound (covered separately in tests/test_audio_articulation.py),
    # but it changes exactly *when* each voice reaches peak relative to
    # a capture/check accent's own click, which this file's exact-value
    # offline-parity assertions were never meant to re-derive. This
    # file's actual job is the headroom SCALING FORMULA in
    # _render_block, which is untouched by 5f.4a -- scrub mode exercises
    # that exact same formula via the original, still-unchanged
    # continuous-smoothing code path, so it reproduces these numbers
    # byte-for-byte.
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
# Quiet moves (sustained bed only, no accent): live now matches offline
# EXACTLY at every realistic loudness tier.
# ---------------------------------------------------------


@pytest.mark.parametrize("label", ALL_LABELS)
def test_live_sustained_bed_peak_matches_offline_exactly_for_quiet_moves(label):
    mapping = _quiet_move_mapping(label)

    offline_peak = float(np.abs(AudioRenderer().render(mapping).samples).max())
    live_peak = _live_peak(mapping)

    assert live_peak == pytest.approx(offline_peak, abs=1e-3)


@pytest.mark.parametrize("label", ["tense", "chaotic"])
def test_live_no_longer_hard_clips_a_quiet_move_at_the_previously_clipping_tiers(label):
    """
    The exact regression this phase fixes: before 5f.3a, "tense" and
    "chaotic" hit the hard clip bound (peak == 1.0) for a quiet move
    with no accent at all. They no longer do.
    """

    assert LOUDNESS_BY_LABEL[label] > 1.0 / (1.0 + HARMONY_VOICE_WEIGHT)  # would clip without headroom

    mapping = _quiet_move_mapping(label)
    peak = _live_peak(mapping)

    assert peak < 1.0
    assert peak == pytest.approx(SUSTAINED_PEAK_HEADROOM, abs=1e-3)


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


def test_capture_accent_peak_is_never_higher_live_than_offline():
    for label in ALL_LABELS:
        mapping = _pure_capture_mapping(label)
        offline_peak = float(np.abs(AudioRenderer().render(mapping).samples).max())
        live_peak = _live_peak(mapping)
        assert live_peak <= offline_peak + 1e-3, f"label={label}"


# ---------------------------------------------------------
# Capture + check together -- the one scenario where offline itself
# already clips (docs/audio.md's known, accepted loudness-compression
# limitation). Reproduced, not newly introduced.
# ---------------------------------------------------------


@pytest.mark.parametrize("label", ALL_LABELS)
def test_capture_plus_check_peak_is_never_higher_live_than_offline(label):
    mapping = _capturing_check_mapping(label)
    assert mapping.is_capture is True
    assert mapping.is_check is True

    offline_peak = float(np.abs(AudioRenderer().render(mapping).samples).max())
    live_peak = _live_peak(mapping)

    assert live_peak <= offline_peak + 1e-3


def test_capture_plus_check_at_chaotic_still_hits_the_final_safety_clip_on_both_paths():
    """
    The one already-accepted exception: docs/audio.md documents that a
    capturing move's accent can push the offline renderer's own final
    hard clip at high loudness. Both paths hit it identically here --
    this is the offline renderer's own pre-existing, accepted behavior,
    reproduced, not a new live-only regression.
    """

    mapping = _capturing_check_mapping("chaotic")
    offline_peak = float(np.abs(AudioRenderer().render(mapping).samples).max())
    live_peak = _live_peak(mapping)

    assert offline_peak == pytest.approx(1.0, abs=1e-6)
    assert live_peak == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------
# The final safety clip is a last resort, not the headroom mechanism --
# for a quiet move it should essentially never engage anymore.
# ---------------------------------------------------------


@pytest.mark.parametrize("label", ALL_LABELS)
def test_final_safety_clip_does_not_engage_for_quiet_moves_at_any_tier(label):
    mapping = _quiet_move_mapping(label)
    peak = _live_peak(mapping)

    assert peak <= SUSTAINED_PEAK_HEADROOM + 1e-3  # never reaches the [-1, 1] clip bound
