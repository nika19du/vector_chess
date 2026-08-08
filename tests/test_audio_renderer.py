from dataclasses import replace

import chess
import numpy as np

from audio.mapping import build_audio_mapping
from audio.models import RenderConfig
from audio.renderer import AudioRenderer, _harmony_frequency
from chess_engine.analyzer import analyze_position
from chess_engine.moves import execute_move

SHORT_CONFIG = RenderConfig(clip_duration_seconds=0.3)


def _mapping_after(moves: list[str]):
    board = chess.Board()
    analysis = None

    for move_text in moves:
        move_details = execute_move(board, move_text)
        assert move_details is not None, f"expected {move_text} to be legal"
        analysis = analyze_position(board, move_details)

    return build_audio_mapping(analysis, dynamics=None)


# ---------------------------------------------------------
# Configured shape
# ---------------------------------------------------------


def test_render_output_matches_configured_duration_and_rate():
    mapping = _mapping_after(["e2e4"])
    clip = AudioRenderer(SHORT_CONFIG).render(mapping)

    expected_samples = int(
        round(SHORT_CONFIG.clip_duration_seconds * SHORT_CONFIG.sample_rate)
    )

    assert clip.sample_rate == SHORT_CONFIG.sample_rate
    assert len(clip.samples) == expected_samples


def test_render_output_stays_within_pcm_bounds():
    mapping = _mapping_after(["e2e4"])
    clip = AudioRenderer(SHORT_CONFIG).render(mapping)

    assert np.max(np.abs(clip.samples)) <= 1.0


# ---------------------------------------------------------
# Determinism and purity
# ---------------------------------------------------------


def test_render_is_deterministic():
    mapping = _mapping_after(["e2e4"])
    renderer = AudioRenderer(SHORT_CONFIG)

    first = renderer.render(mapping)
    second = renderer.render(mapping)

    assert np.array_equal(first.samples, second.samples)


def test_render_does_not_mutate_mapping_or_config():
    mapping = _mapping_after(["e2e4"])
    config = RenderConfig(clip_duration_seconds=0.3)
    renderer = AudioRenderer(config)

    mapping_snapshot = replace(mapping)
    config_snapshot = replace(config)

    renderer.render(mapping)

    assert mapping == mapping_snapshot
    assert renderer.config == config_snapshot


# ---------------------------------------------------------
# Color -> timbre
# ---------------------------------------------------------


def test_color_changes_the_rendered_waveform():
    white_mapping = _mapping_after(["e2e4"])
    black_mapping = replace(white_mapping, color="black", harmonic_richness=3)

    renderer = AudioRenderer(SHORT_CONFIG)
    white_clip = renderer.render(white_mapping)
    black_clip = renderer.render(black_mapping)

    assert not np.array_equal(white_clip.samples, black_clip.samples)


# ---------------------------------------------------------
# Balance sign -> harmony direction
# ---------------------------------------------------------


def test_harmony_frequency_sits_above_pitch_for_positive_balance():
    mapping = replace(
        _mapping_after(["e2e4"]),
        attack_influence_balance=6.0,
        harmony_interval_ratio=1.2,
    )

    assert _harmony_frequency(mapping) > mapping.pitch_hz


def test_harmony_frequency_sits_below_pitch_for_negative_balance():
    mapping = replace(
        _mapping_after(["e2e4"]),
        attack_influence_balance=-6.0,
        harmony_interval_ratio=1.2,
    )

    assert _harmony_frequency(mapping) < mapping.pitch_hz


def test_render_reflects_the_balance_sign_as_a_different_waveform():
    positive_mapping = replace(
        _mapping_after(["e2e4"]),
        attack_influence_balance=6.0,
        harmony_interval_ratio=1.2,
    )
    negative_mapping = replace(positive_mapping, attack_influence_balance=-6.0)

    renderer = AudioRenderer(SHORT_CONFIG)
    positive_clip = renderer.render(positive_mapping)
    negative_clip = renderer.render(negative_mapping)

    assert not np.array_equal(positive_clip.samples, negative_clip.samples)


# ---------------------------------------------------------
# Capture -> independent transient
# ---------------------------------------------------------


def test_capture_raises_energy_in_the_opening_window():
    capture_mapping = _mapping_after(
        ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]
    )
    assert capture_mapping.is_capture

    non_capture_mapping = replace(capture_mapping, is_capture=False)

    renderer = AudioRenderer(SHORT_CONFIG)
    capture_clip = renderer.render(capture_mapping)
    non_capture_clip = renderer.render(non_capture_mapping)

    window = int(0.05 * SHORT_CONFIG.sample_rate)
    capture_energy = np.sum(capture_clip.samples[:window] ** 2)
    non_capture_energy = np.sum(non_capture_clip.samples[:window] ** 2)

    assert capture_energy > non_capture_energy


# ---------------------------------------------------------
# Check -> harmonic tension (already resolved in mapping, renderer must honor it)
# ---------------------------------------------------------


def test_check_adds_audible_harmonic_tension():
    calm_mapping = replace(
        _mapping_after(["e2e4"]),
        attack_influence_balance=0.0,
        harmony_interval_ratio=1.0,  # consonant, as if is_check were False
    )
    checked_mapping = replace(
        calm_mapping,
        is_check=True,
        harmony_interval_ratio=1.25,  # CHECK_DISSONANCE_FLOOR, as mapping.py would set it
    )

    renderer = AudioRenderer(SHORT_CONFIG)
    calm_clip = renderer.render(calm_mapping)
    checked_clip = renderer.render(checked_mapping)

    assert not np.array_equal(calm_clip.samples, checked_clip.samples)
    assert _harmony_frequency(checked_mapping) != _harmony_frequency(calm_mapping)


def test_checking_capture_combines_both_effects():
    # Each claim is isolated against a baseline that changes exactly
    # one thing, so global peak normalization (which shifts when the
    # overall clip content changes) can't confound either comparison.
    checking_capture_mapping = replace(
        _mapping_after(["e2e4"]),
        is_capture=True,
        is_check=True,
        attack_influence_balance=0.0,
        harmony_interval_ratio=1.25,
    )
    checking_only_mapping = replace(checking_capture_mapping, is_capture=False)
    calm_reference_mapping = replace(
        checking_capture_mapping,
        is_capture=False,
        is_check=False,
        harmony_interval_ratio=1.0,
    )

    # Dissonance effect: driven entirely by harmony_interval_ratio,
    # present whether or not the move is also a capture.
    assert _harmony_frequency(checking_capture_mapping) == _harmony_frequency(
        checking_only_mapping
    )
    assert _harmony_frequency(checking_capture_mapping) != _harmony_frequency(
        calm_reference_mapping
    )

    # Capture effect: extra transient energy layered on top of the same
    # dissonant harmony bed -- only is_capture toggles between these two.
    renderer = AudioRenderer(SHORT_CONFIG)
    combined_clip = renderer.render(checking_capture_mapping)
    checking_only_clip = renderer.render(checking_only_mapping)

    window = int(0.05 * SHORT_CONFIG.sample_rate)
    combined_energy = np.sum(combined_clip.samples[:window] ** 2)
    checking_only_energy = np.sum(checking_only_clip.samples[:window] ** 2)

    assert combined_energy > checking_only_energy


# ---------------------------------------------------------
# First move / neutral rendering
# ---------------------------------------------------------


def test_first_move_renders_cleanly_with_neutral_loudness():
    mapping = _mapping_after(["e2e4"])
    assert mapping.dynamics_label is None

    clip = AudioRenderer(SHORT_CONFIG).render(mapping)

    assert len(clip.samples) > 0
    assert np.max(np.abs(clip.samples)) <= 1.0
    assert np.max(np.abs(clip.samples)) > 0.0
