"""
Thin injectable wrapper around the real-time audio device -- the only
place in audio/ that imports `sounddevice`. AudioEngine depends only on
the Stream/AudioBackend structural interfaces below, never on
`sounddevice` directly, so tests (and any future backend) never need to
touch real hardware. No Qt dependency anywhere in this module.
"""

from typing import Callable, Protocol

import numpy as np

AudioCallback = Callable[[np.ndarray, int, object, object], None]


class Stream(Protocol):
    """Structural interface a backend's opened stream must satisfy --
    matches the subset of `sounddevice.OutputStream` AudioEngine uses."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    @property
    def active(self) -> bool: ...


class AudioBackend(Protocol):
    def open_stream(
        self,
        *,
        samplerate: int,
        blocksize: int,
        channels: int,
        callback: AudioCallback,
    ) -> Stream: ...


class SoundDeviceBackend:
    """Real backend. Opens (but does not start) a `sounddevice.OutputStream`.
    `sounddevice` is imported lazily inside `open_stream`, not at module
    import time, so importing `audio.backend` never requires PortAudio
    to be installed/available."""

    def open_stream(
        self,
        *,
        samplerate: int,
        blocksize: int,
        channels: int,
        callback: AudioCallback,
    ) -> Stream:
        import sounddevice as sd

        return sd.OutputStream(
            samplerate=samplerate,
            blocksize=blocksize,
            channels=channels,
            callback=callback,
        )


class FakeStream:
    """
    Test double -- never touches real hardware. Tracks lifecycle calls
    so tests can assert on them, and exposes `render_block` so a test
    can drive the callback manually, exactly as `sounddevice` would
    invoke it on its own audio thread.
    """

    def __init__(self, callback: AudioCallback, channels: int) -> None:
        self._callback = callback
        self._channels = channels
        self._active = False
        self.start_calls = 0
        self.stop_calls = 0
        self.close_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        self._active = True

    def stop(self) -> None:
        self.stop_calls += 1
        self._active = False

    def close(self) -> None:
        self.close_calls += 1
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def render_block(self, frames: int, time_info: object = None, status: object = None) -> np.ndarray:
        outdata = np.zeros((frames, self._channels), dtype=np.float64)
        self._callback(outdata, frames, time_info, status)
        return outdata


class FakeAudioBackend:
    """
    Injectable test backend. Constructs `FakeStream` instances instead
    of opening real hardware. Set `fail_on_open=True` to simulate a
    device-unavailable failure (no PortAudio device, driver error,
    etc.) -- exercises AudioEngine's fallback path without needing to
    actually remove/break an audio device.
    """

    def __init__(self, fail_on_open: bool = False) -> None:
        self.fail_on_open = fail_on_open
        self.streams: list[FakeStream] = []

    def open_stream(
        self,
        *,
        samplerate: int,
        blocksize: int,
        channels: int,
        callback: AudioCallback,
    ) -> Stream:
        if self.fail_on_open:
            raise RuntimeError("simulated device-open failure")

        stream = FakeStream(callback, channels)
        self.streams.append(stream)
        return stream
