import numpy as np

# Fixed click used for capture accents. A short, high, fast-decaying
# sine burst -- not noise -- so the accent stays fully deterministic
# (docs/audio.md principle 4: "same position, same sound, always").
PERCUSSIVE_CLICK_FREQUENCY_HZ = 900.0
PERCUSSIVE_DECAY_RATE = 40.0  # per second


def _sample_count(duration_seconds: float, sample_rate: int) -> int:
    return int(round(duration_seconds * sample_rate))


def sine_wave(
    frequency_hz: float,
    duration_seconds: float,
    sample_rate: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    """A pure sine tone -- the simplest member of the palette (White)."""

    n = _sample_count(duration_seconds, sample_rate)
    t = np.linspace(0.0, duration_seconds, n, endpoint=False)

    return amplitude * np.sin(2 * np.pi * frequency_hz * t)


def additive_tone(
    frequency_hz: float,
    harmonic_richness: int,
    duration_seconds: float,
    sample_rate: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Sums `harmonic_richness` weighted sine partials (weight 1/n for the
    n-th partial), normalized so the result stays comparable in scale
    to a single sine at the same amplitude.

    This mirrors, one layer removed, how Attack Influence is computed:
    a sum of weighted contributions. With harmonic_richness == 1 this
    is exactly sine_wave -- the pure-tone / White case.
    """

    n = _sample_count(duration_seconds, sample_rate)
    total = np.zeros(n)
    weight_sum = 0.0

    for partial_index in range(1, harmonic_richness + 1):
        weight = 1.0 / partial_index
        weight_sum += weight

        total += sine_wave(
            frequency_hz=frequency_hz * partial_index,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            amplitude=weight,
        )

    return total * (amplitude / weight_sum)


def apply_envelope(
    samples: np.ndarray,
    attack_seconds: float,
    release_seconds: float,
    sample_rate: int,
) -> np.ndarray:
    """
    Linear attack/release envelope so clips start and end at zero
    amplitude instead of clicking at the boundary.
    """

    n = len(samples)

    attack_n = min(int(round(attack_seconds * sample_rate)), n)
    release_n = min(int(round(release_seconds * sample_rate)), n - attack_n)

    envelope = np.ones(n)

    if attack_n > 0:
        envelope[:attack_n] = np.linspace(0.0, 1.0, attack_n, endpoint=False)

    if release_n > 0:
        envelope[n - release_n:] = np.linspace(1.0, 0.0, release_n, endpoint=True)

    return samples * envelope


def percussive_burst(
    duration_seconds: float,
    sample_rate: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    A short, fast-decaying click used for the capture accent.
    Deterministic by construction -- no randomness, no seed needed.
    """

    n = _sample_count(duration_seconds, sample_rate)
    t = np.linspace(0.0, duration_seconds, n, endpoint=False)

    decay_envelope = np.exp(-PERCUSSIVE_DECAY_RATE * t)

    return amplitude * decay_envelope * np.sin(
        2 * np.pi * PERCUSSIVE_CLICK_FREQUENCY_HZ * t
    )


def mix(*voices: np.ndarray) -> np.ndarray:
    """Sums voices of possibly different lengths, padding with zeros."""

    if not voices:
        return np.zeros(0)

    max_length = max(len(voice) for voice in voices)
    total = np.zeros(max_length)

    for voice in voices:
        total[: len(voice)] += voice

    return total
