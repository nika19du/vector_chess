import wave
from pathlib import Path

import numpy as np

from audio.models import AudioClip

PCM16_MAX_AMPLITUDE = 32767


def write_wav(clip: AudioClip, path: Path) -> Path:
    """
    Serializes an AudioClip to a mono 16-bit PCM .wav file, creating
    the parent directory if needed. Quantization to int16 happens only
    here, at the export boundary -- every earlier stage stays float64.

    This is the last stop before the filesystem, so it validates its
    input rather than trusting it: build_audio_mapping and
    AudioRenderer are the only intended callers today, but a future
    Event System or Interactive UI caller (see the approved plan's
    extension points) isn't guaranteed to only ever pass a
    renderer-produced clip.
    """

    if clip.samples.size == 0:
        raise ValueError("Cannot write an empty AudioClip to a .wav file.")

    if clip.sample_rate <= 0:
        raise ValueError(
            f"AudioClip.sample_rate must be positive, got {clip.sample_rate}."
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    clipped = np.clip(clip.samples, -1.0, 1.0)
    quantized = (clipped * PCM16_MAX_AMPLITUDE).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(clip.sample_rate)
        wav_file.writeframes(quantized.tobytes())

    return path
