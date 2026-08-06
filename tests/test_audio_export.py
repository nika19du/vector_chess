import wave

import numpy as np
import pytest

from audio.export import write_wav
from audio.models import AudioClip


# ---------------------------------------------------------
# File creation and format
# ---------------------------------------------------------


def test_write_wav_creates_the_file(tmp_path):
    clip = AudioClip(samples=np.linspace(-1.0, 1.0, 4410), sample_rate=44100)

    output_path = write_wav(clip, tmp_path / "clip.wav")

    assert output_path.exists()
    assert output_path.is_file()


def test_write_wav_is_valid_mono_uncompressed_pcm(tmp_path):
    clip = AudioClip(samples=np.linspace(-1.0, 1.0, 4410), sample_rate=44100)

    output_path = write_wav(clip, tmp_path / "clip.wav")

    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2  # 16-bit
        assert wav_file.getcomptype() == "NONE"  # uncompressed PCM


def test_write_wav_preserves_the_sample_rate(tmp_path):
    clip = AudioClip(samples=np.zeros(1000), sample_rate=22050)

    output_path = write_wav(clip, tmp_path / "clip.wav")

    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getframerate() == 22050


def test_write_wav_preserves_the_frame_count(tmp_path):
    samples = np.zeros(12345)
    clip = AudioClip(samples=samples, sample_rate=44100)

    output_path = write_wav(clip, tmp_path / "clip.wav")

    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getnframes() == len(samples)


# ---------------------------------------------------------
# int16 quantization
# ---------------------------------------------------------


def test_int16_conversion_stays_within_bounds_at_the_extremes(tmp_path):
    # Includes out-of-range inputs on purpose (a caller that forgot to
    # clip) to confirm the export boundary clamps rather than wraps.
    samples = np.array([-1.5, -1.0, 0.0, 1.0, 1.5])
    clip = AudioClip(samples=samples, sample_rate=44100)

    output_path = write_wav(clip, tmp_path / "clip.wav")

    with wave.open(str(output_path), "rb") as wav_file:
        raw = wav_file.readframes(wav_file.getnframes())

    quantized = np.frombuffer(raw, dtype=np.int16)

    assert quantized.tolist() == [-32767, -32767, 0, 32767, 32767]
    assert np.all(quantized >= -32768) and np.all(quantized <= 32767)


def test_round_trip_matches_within_int16_quantization_error(tmp_path):
    samples = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    clip = AudioClip(samples=samples, sample_rate=44100)

    output_path = write_wav(clip, tmp_path / "clip.wav")

    with wave.open(str(output_path), "rb") as wav_file:
        raw = wav_file.readframes(wav_file.getnframes())

    read_back = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32767
    quantization_step = 1.0 / 32767

    assert np.allclose(read_back, samples, atol=quantization_step * 2)


# ---------------------------------------------------------
# Determinism
# ---------------------------------------------------------


def test_write_wav_is_byte_identical_for_the_same_clip(tmp_path):
    samples = np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.5, 22050))
    clip = AudioClip(samples=samples, sample_rate=44100)

    first_path = write_wav(clip, tmp_path / "first.wav")
    second_path = write_wav(clip, tmp_path / "second.wav")

    assert first_path.read_bytes() == second_path.read_bytes()


# ---------------------------------------------------------
# Parent directory creation
# ---------------------------------------------------------


def test_write_wav_creates_missing_parent_directories(tmp_path):
    clip = AudioClip(samples=np.zeros(100), sample_rate=44100)

    target_dir = tmp_path / "does" / "not" / "exist"
    output_path = write_wav(clip, target_dir / "clip.wav")

    assert target_dir.exists()
    assert output_path.exists()


# ---------------------------------------------------------
# Invalid input
# ---------------------------------------------------------


def test_empty_clip_raises_a_clear_error(tmp_path):
    clip = AudioClip(samples=np.zeros(0), sample_rate=44100)

    with pytest.raises(ValueError, match="empty"):
        write_wav(clip, tmp_path / "clip.wav")


def test_non_positive_sample_rate_raises_a_clear_error(tmp_path):
    clip = AudioClip(samples=np.zeros(100), sample_rate=0)

    with pytest.raises(ValueError, match="sample_rate"):
        write_wav(clip, tmp_path / "clip.wav")


def test_negative_sample_rate_raises_a_clear_error(tmp_path):
    clip = AudioClip(samples=np.zeros(100), sample_rate=-44100)

    with pytest.raises(ValueError, match="sample_rate"):
        write_wav(clip, tmp_path / "clip.wav")


# ---------------------------------------------------------
# Purity
# ---------------------------------------------------------


def test_write_wav_does_not_mutate_the_clips_samples(tmp_path):
    samples = np.array([-1.5, -0.3, 0.0, 0.7, 1.5])
    original = samples.copy()
    clip = AudioClip(samples=samples, sample_rate=44100)

    write_wav(clip, tmp_path / "clip.wav")

    assert np.array_equal(clip.samples, original)
