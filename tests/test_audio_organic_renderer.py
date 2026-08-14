from dataclasses import replace

import chess
import numpy as np

from audio.mapping import build_audio_mapping
from audio.models import RenderConfig
from audio.organic_renderer import HARMONY_GAIN, OrganicAudioRenderer, _harmony_frequency
from audio.organic_synthesis import HARMONY_MODAL_PARAMS, WHITE_MODAL_PARAMS, modal_resonance
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


CAPTURE_CHECK_MOVES = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]


# ---------------------------------------------------------
# Configured shape / determinism / bounds
# ---------------------------------------------------------


def test_render_output_matches_configured_duration_and_rate():
    mapping = _mapping_after(["e2e4"])
    clip = OrganicAudioRenderer(SHORT_CONFIG).render(mapping)

    expected_samples = int(round(SHORT_CONFIG.clip_duration_seconds * SHORT_CONFIG.sample_rate))

    assert clip.sample_rate == SHORT_CONFIG.sample_rate
    assert len(clip.samples) == expected_samples


def test_render_output_stays_within_pcm_bounds():
    for moves in (["e2e4"], ["e2e4", "d7d5", "e4d5"], CAPTURE_CHECK_MOVES):
        mapping = _mapping_after(moves)
        clip = OrganicAudioRenderer(SHORT_CONFIG).render(mapping)
        assert np.all(np.isfinite(clip.samples))
        assert np.max(np.abs(clip.samples)) <= 1.0


def test_render_is_deterministic():
    mapping = _mapping_after(["e2e4"])
    renderer = OrganicAudioRenderer(SHORT_CONFIG)

    first = renderer.render(mapping)
    second = renderer.render(mapping)

    assert np.array_equal(first.samples, second.samples)


def test_render_does_not_mutate_mapping_or_config():
    mapping = _mapping_after(["e2e4"])
    config = RenderConfig(clip_duration_seconds=0.3)
    renderer = OrganicAudioRenderer(config)

    mapping_snapshot = replace(mapping)
    config_snapshot = replace(config)

    renderer.render(mapping)

    assert mapping == mapping_snapshot
    assert renderer.config == config_snapshot


# ---------------------------------------------------------
# Color -> timbre
# ---------------------------------------------------------


def test_color_changes_the_rendered_waveform_at_the_same_pitch():
    white_mapping = _mapping_after(["e2e4"])
    black_mapping = replace(white_mapping, color="black")

    renderer = OrganicAudioRenderer(SHORT_CONFIG)
    white_clip = renderer.render(white_mapping)
    black_clip = renderer.render(black_mapping)

    assert not np.array_equal(white_clip.samples, black_clip.samples)


# ---------------------------------------------------------
# Balance sign -> harmony direction (same rule as AudioRenderer's)
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


# ---------------------------------------------------------
# Harmony must read as clearly softer/subordinate to Melody
# ---------------------------------------------------------


def test_harmony_voice_is_softer_than_melody_voice_in_isolation():
    """
    Isolates each voice the same way OrganicAudioRenderer.render builds
    them internally, rather than trying to parse the mixed clip back
    apart -- exercises the same code paths without duplicating the
    renderer's own mixing/headroom logic.
    """

    mapping = _mapping_after(["e2e4"])

    melody_samples = modal_resonance(
        mapping.pitch_hz, WHITE_MODAL_PARAMS, SHORT_CONFIG.clip_duration_seconds,
        SHORT_CONFIG.sample_rate, amplitude=mapping.loudness, attack_seconds=SHORT_CONFIG.attack_seconds,
    )
    harmony_samples = modal_resonance(
        _harmony_frequency(mapping), HARMONY_MODAL_PARAMS, SHORT_CONFIG.clip_duration_seconds,
        SHORT_CONFIG.sample_rate, amplitude=mapping.loudness * HARMONY_GAIN, attack_seconds=SHORT_CONFIG.attack_seconds,
    )

    melody_rms = np.sqrt(np.mean(melody_samples ** 2))
    harmony_rms = np.sqrt(np.mean(harmony_samples ** 2))

    assert harmony_rms < melody_rms
    assert np.max(np.abs(harmony_samples)) < np.max(np.abs(melody_samples))


# ---------------------------------------------------------
# Capture / check / combined
# ---------------------------------------------------------


def test_capture_raises_energy_in_the_opening_window():
    capture_mapping = _mapping_after(CAPTURE_CHECK_MOVES)
    assert capture_mapping.is_capture

    non_capture_mapping = replace(capture_mapping, is_capture=False)

    renderer = OrganicAudioRenderer(SHORT_CONFIG)
    capture_clip = renderer.render(capture_mapping)
    non_capture_clip = renderer.render(non_capture_mapping)

    window = int(0.05 * SHORT_CONFIG.sample_rate)
    capture_energy = np.sum(capture_clip.samples[:window] ** 2)
    non_capture_energy = np.sum(non_capture_clip.samples[:window] ** 2)

    assert capture_energy > non_capture_energy


def test_checking_capture_renders_validly_and_combines_both_effects():
    mapping = _mapping_after(CAPTURE_CHECK_MOVES)
    assert mapping.is_capture is True
    assert mapping.is_check is True

    clip = OrganicAudioRenderer(SHORT_CONFIG).render(mapping)

    assert np.all(np.isfinite(clip.samples))
    assert np.max(np.abs(clip.samples)) <= 1.0
    assert np.max(np.abs(clip.samples)) > 0.0


def test_first_move_renders_cleanly_with_neutral_loudness():
    mapping = _mapping_after(["e2e4"])
    assert mapping.dynamics_label is None

    clip = OrganicAudioRenderer(SHORT_CONFIG).render(mapping)

    assert len(clip.samples) > 0
    assert np.max(np.abs(clip.samples)) <= 1.0
    assert np.max(np.abs(clip.samples)) > 0.0
