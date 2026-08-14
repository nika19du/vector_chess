"""
Rhythmic Layer v2 -- Phase D measurement: real worst-case headroom peak
and real-time callback performance, using the actual production
AudioEngine (FakeAudioBackend swapped in for hardware, matching every
existing engine test's convention). Not a production code path -- a
one-off measurement script for the Phase C completion report.

Worst case exercised: a capturing+checking Black move (richest timbre),
maximum loudness, maximum-density phrase (4 events), rendered across a
rapid retrigger sequence so two phrase events can plausibly overlap
(primary+secondary of the shared _ModalVoice).
"""

import time

import numpy as np

from audio.backend import FakeAudioBackend
from audio.engine import DEFAULT_BLOCKSIZE, DEFAULT_SAMPLE_RATE, AccentTrigger, AudioEngine
from audio.live_state import SonificationState
from audio.phrase import build_phrase
from audio.voices import build_default_voice_registry

WORST_CASE_PHRASE = build_phrase(
    pulse_density=1.0, pitch_hz=440.0, harmony_interval_ratio=1.5, color="black"
)


def _worst_case_state() -> SonificationState:
    return SonificationState(
        harmony_interval_ratio=1.5,
        harmony_above_melody=True,
        pitch_hz=440.0,
        harmonic_richness=3,
        loudness=1.0,
        segment_key=("a", "b"),
        color="black",
        phrase=WORST_CASE_PHRASE,
        master_gain=1.0,
    )


def measure_headroom() -> float:
    backend = FakeAudioBackend()
    engine = AudioEngine(backend, build_default_voice_registry(), blocksize=DEFAULT_BLOCKSIZE)
    engine.start()
    stream = backend.streams[-1]

    engine.publish(_worst_case_state())
    engine.push_note_trigger()
    engine.push_event(AccentTrigger("capture+check", loudness=1.0))

    peak = 0.0
    for _ in range(120):  # ~2.4s -- covers the whole phrase window plus decay tails
        block = stream.render_block(DEFAULT_BLOCKSIZE)
        peak = max(peak, float(np.abs(block).max()))

    # Also probe a rapid-retrigger overlap: fire two phrases back to back,
    # close enough that _ModalVoice's primary+secondary can both be
    # ringing simultaneously.
    engine.push_note_trigger()
    for _ in range(3):
        stream.render_block(DEFAULT_BLOCKSIZE)
    engine.push_note_trigger()
    engine.push_event(AccentTrigger("capture+check", loudness=1.0))
    for _ in range(20):
        block = stream.render_block(DEFAULT_BLOCKSIZE)
        peak = max(peak, float(np.abs(block).max()))

    engine.shutdown()
    return peak


def measure_callback_performance() -> dict:
    backend = FakeAudioBackend()
    engine = AudioEngine(backend, build_default_voice_registry(), blocksize=DEFAULT_BLOCKSIZE)
    engine.start()
    stream = backend.streams[-1]

    engine.publish(_worst_case_state())
    engine.push_note_trigger()
    engine.push_event(AccentTrigger("capture+check", loudness=1.0))

    stream.render_block(DEFAULT_BLOCKSIZE)  # warm up

    block_seconds = DEFAULT_BLOCKSIZE / DEFAULT_SAMPLE_RATE
    n = 2000
    timings = np.zeros(n)
    for i in range(n):
        start = time.perf_counter()
        stream.render_block(DEFAULT_BLOCKSIZE)
        timings[i] = time.perf_counter() - start

    engine.shutdown()

    timings_ms = timings * 1000
    return {
        "block_ms": block_seconds * 1000,
        "mean_ms": float(np.mean(timings_ms)),
        "p99_ms": float(np.percentile(timings_ms, 99)),
        "max_ms": float(np.max(timings_ms)),
    }


if __name__ == "__main__":
    peak = measure_headroom()
    print(f"Worst-case measured peak: {peak:.4f} (clip bound is 1.0)")

    perf = measure_callback_performance()
    block_ms = perf["block_ms"]
    print(f"Block duration budget: {block_ms:.3f}ms")
    print(f"Mean callback time:    {perf['mean_ms']:.4f}ms ({perf['mean_ms'] / block_ms * 100:.2f}% of budget)")
    print(f"p99 callback time:     {perf['p99_ms']:.4f}ms ({perf['p99_ms'] / block_ms * 100:.2f}% of budget)")
    print(f"Max callback time:     {perf['max_ms']:.4f}ms ({perf['max_ms'] / block_ms * 100:.2f}% of budget)")
