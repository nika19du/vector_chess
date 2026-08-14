import numpy as np
import pytest

from audio.backend import FakeAudioBackend, FakeStream


def _noop_callback(outdata, frames, time_info, status):
    outdata[:] = 0.0


# ---------------------------------------------------------
# FakeAudioBackend / FakeStream -- lifecycle tracking
# ---------------------------------------------------------


def test_open_stream_returns_a_fresh_stream_each_call():
    backend = FakeAudioBackend()

    first = backend.open_stream(samplerate=44100, blocksize=64, channels=1, callback=_noop_callback)
    second = backend.open_stream(samplerate=44100, blocksize=64, channels=1, callback=_noop_callback)

    assert first is not second
    assert backend.streams == [first, second]


def test_stream_starts_inactive_and_tracks_lifecycle_calls():
    backend = FakeAudioBackend()
    stream = backend.open_stream(samplerate=44100, blocksize=64, channels=1, callback=_noop_callback)

    assert stream.active is False

    stream.start()
    assert stream.active is True
    assert stream.start_calls == 1

    stream.stop()
    assert stream.active is False
    assert stream.stop_calls == 1

    stream.close()
    assert stream.close_calls == 1


def test_render_block_invokes_the_real_callback_with_the_right_shape():
    received = {}

    def callback(outdata, frames, time_info, status):
        received["shape"] = outdata.shape
        received["frames"] = frames
        outdata[:, 0] = 0.5

    backend = FakeAudioBackend()
    stream = backend.open_stream(samplerate=44100, blocksize=128, channels=1, callback=callback)

    block = stream.render_block(128)

    assert received["shape"] == (128, 1)
    assert received["frames"] == 128
    assert block.shape == (128, 1)
    assert np.all(block == 0.5)


def test_fail_on_open_raises_instead_of_returning_a_stream():
    backend = FakeAudioBackend(fail_on_open=True)

    with pytest.raises(RuntimeError):
        backend.open_stream(samplerate=44100, blocksize=64, channels=1, callback=_noop_callback)

    assert backend.streams == []


# ---------------------------------------------------------
# SoundDeviceBackend -- guarded real-hardware smoke test
# ---------------------------------------------------------


def test_sounddevice_backend_can_open_and_close_a_real_stream_when_hardware_is_available():
    """
    Not a hard requirement -- some environments (headless CI, no
    PortAudio device) cannot open a real stream at all. This is treated
    as an environment limitation, not a code defect: the test is
    skipped rather than failed when the real backend is unavailable,
    exactly as AudioEngine.start() itself handles the same failure at
    runtime (see test_audio_engine.py's unavailable-device tests, which
    use FakeAudioBackend(fail_on_open=True) to exercise that path
    deterministically without depending on real hardware).
    """

    from audio.backend import SoundDeviceBackend

    backend = SoundDeviceBackend()

    try:
        stream = backend.open_stream(samplerate=44100, blocksize=256, channels=1, callback=_noop_callback)
        stream.start()
    except Exception as error:
        pytest.skip(f"no real audio device available in this environment: {error}")

    try:
        assert stream.active is True
    finally:
        stream.stop()
        stream.close()
