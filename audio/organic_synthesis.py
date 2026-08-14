"""
B2: damped modal resonance synthesis -- the organic-timbre primitive.

A resonator is modeled as a small bank of sinusoidal modes, each with
its own frequency ratio to the fundamental, its own starting weight,
and its own exponential decay rate:

    y(t) = amplitude * attack_gate(t)
           * sum_k( w_k * exp(-d_k * t) * sin(2*pi * f_k * t) ) / sum_k(w_k)

    f_k = fundamental_hz * mode_ratios[k]
    w_k = 1 / k^brightness_exponent
    d_k = base_damping_per_second * (1 + (k-1) * damping_slope)

This is a direct generalization of audio/synthesis.py's own
additive_tone (a weighted partial sum): instead of static partials
under one note-level linear envelope, each mode decays independently,
so brightness evolves over the note's life the way a real
struck/plucked resonant body does -- higher modes physically damp
faster than the fundamental. No noise, no recursive per-sample state
(unlike Karplus-Strong): fully deterministic and fully vectorizable,
which is also what makes this model portable to a future real-time
block-callback engine (see the B2 plan's live-state design).

Kept fully isolated from audio/synthesis.py and audio/renderer.py --
see audio/organic_renderer.py's module docstring for why.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np

from audio.phrase import MAX_PHRASE_EVENTS
from audio.synthesis import PERCUSSIVE_CLICK_FREQUENCY_HZ

# Final numerical safety fade so the fixed-length clip buffer's last
# sample lands at exactly zero even though exp(-d*t) only approaches
# zero asymptotically -- independent of the physical decay itself.
END_FADE_SECONDS = 0.005


@dataclass(frozen=True)
class ModalVoiceParams:
    """
    One resonator's mode bank.

    mode_ratios: f_k / fundamental_hz for each mode, 1-indexed by
        position (mode_ratios[0] is mode k=1). (1.0, 2.0, 3.0, ...)
        for a harmonic resonator (Melody/Harmony); an inharmonic set
        for a struck-bar accent.
    brightness_exponent: controls each mode's starting weight
        (1 / k^brightness_exponent) -- higher exponent means a darker,
        more fundamental-dominated onset.
    base_damping_per_second: the fundamental's (k=1) own decay rate.
    damping_slope: how much faster each successive mode decays
        relative to the fundamental -- the mechanism that keeps a
        dense/bright onset from becoming a sustained shrill tail.
    """

    mode_ratios: tuple[float, ...]
    brightness_exponent: float
    base_damping_per_second: float
    damping_slope: float


# White: fewer modes, gentle brightness rolloff, slow/even damping --
# an open, plucked, warm resonance that rings comfortably.
WHITE_MODAL_PARAMS = ModalVoiceParams(
    mode_ratios=(1.0, 2.0, 3.0, 4.0),
    brightness_exponent=0.9,
    base_damping_per_second=3.0,
    damping_slope=0.35,
)

# Black: docs/audio.md originally described Black as "the same tone
# plus an added sub-octave / denser harmonic blend" -- a fix that was
# never actually implemented (audio/synthesis.py's additive_tone only
# ever added *upper* harmonics, which measurably caused the shrillness
# complaint; see experiments/audio_musicality/REPORT.md Section 2).
# This preset builds that original intent for real: mode_ratios opens
# with 0.5 (a genuine sub-octave below the fundamental, heavily
# weighted and slowly damped, so it persists through the note) plus
# four upper harmonics that are present at the strike (denser onset)
# but damp quickly (damping_slope). Measured result (see B2
# measurements): Black's whole-clip spectral centroid lands ~30% below
# White's at the same fundamental, at every register -- darker by
# construction, not shrill, without any arbitrary frequency cap.
BLACK_MODAL_PARAMS = ModalVoiceParams(
    mode_ratios=(0.5, 1.0, 2.0, 3.0, 4.0),
    brightness_exponent=1.0,
    base_damping_per_second=3.5,
    damping_slope=1.0,
)

# Harmony: fewest modes, softest brightness, fastest overall damping --
# a quiet, quickly-fading sympathetic resonance, clearly subordinate to
# Melody (further attenuated by OrganicAudioRenderer's HARMONY_GAIN).
HARMONY_MODAL_PARAMS = ModalVoiceParams(
    mode_ratios=(1.0, 2.0),
    brightness_exponent=1.3,
    base_damping_per_second=5.0,
    damping_slope=0.6,
)

# Accent (capture/check "knock"): fixed inharmonic ratios -- the first
# three modes of an idealized free-free bar (~1 : 2.756 : 5.404),  a
# textbook struck-bar mode set -- with heavy damping for a short,
# resonant knock built from the *same* modal_resonance primitive as
# Melody/Harmony, not a separate oscillator family.
ACCENT_MODAL_PARAMS = ModalVoiceParams(
    mode_ratios=(1.0, 2.756, 5.404),
    brightness_exponent=1.0,
    base_damping_per_second=25.0,
    damping_slope=0.5,
)

# Reuses production's own accent register (audio/synthesis.py) so the
# organic accent's pitch identity stays recognizable relative to
# today's percussive_burst, even though the timbre itself is now a
# modal knock instead of a single decaying sine.
ACCENT_BASE_FREQUENCY_HZ = PERCUSSIVE_CLICK_FREQUENCY_HZ

# Pulse (Audio Layer 2 -- Rhythmic Layer, v2: audio/phrase.py's finite
# per-move phrase): a short, soft, restrained modal tap -- deliberately
# the quietest and least harmonically busy voice in the family, so it
# never competes with Melody/Harmony for foreground attention or with
# Accent for "something structural just happened" meaning. Two purely
# harmonic modes (1x, 2x -- an octave, not Accent's inharmonic
# struck-bar ratios) keep it a member of this instrument's tonal family
# rather than an unrelated "drum machine" timbre; a steep brightness
# rolloff keeps the 2nd mode quiet at the strike (soft, not
# bright/clicky); moderate damping gives a brief tap noticeably shorter
# than a Melody/Harmony note but gentler and less percussive than
# Accent's hard knock. v2 dropped the fixed 440Hz register this preset
# used to pair with (see git history) -- phrase pitch is now derived
# from the move's own Melody pitch (audio/phrase.py::build_phrase), not
# an independent constant; this timbre preset itself is unchanged.
PULSE_MODAL_PARAMS = ModalVoiceParams(
    mode_ratios=(1.0, 2.0),
    brightness_exponent=1.6,
    base_damping_per_second=15.0,
    damping_slope=0.4,
)

# How much darker/more-damped each successive event within one phrase
# gets, relative to the first (audio/phrase.py's "hit index" -- a
# PhraseEvent's position in PhraseDescription.events). Purely an
# engine-side rendering detail: the phrase itself only decides *when*,
# *what pitch*, and *how loud* each event is (audio/phrase.py); how each
# one's timbre settles as the phrase progresses is decided here,
# alongside every other modal preset, so the timbre-defining numbers for
# every voice stay in exactly one place.
PULSE_DAMPING_STEP_PER_HIT = 2.0


@dataclass(frozen=True)
class ModalModeArrays:
    """
    A ModalVoiceParams preset's per-mode weights/dampings/ratios,
    precomputed once as numpy arrays (plus their weight sum, for
    normalization) -- the single canonical source both the offline
    renderer (audio/organic_renderer.py) and the live engine
    (audio/engine.py, B3) read the actual timbre-defining numbers from,
    so a voice's mode content can never be defined two different ways
    in two different places.
    """

    weights: np.ndarray
    dampings: np.ndarray
    ratios: np.ndarray
    weight_sum: float


def build_mode_arrays(params: ModalVoiceParams) -> ModalModeArrays:
    mode_count = len(params.mode_ratios)
    weights = np.array([1.0 / (k ** params.brightness_exponent) for k in range(1, mode_count + 1)])
    dampings = np.array(
        [
            params.base_damping_per_second * (1.0 + (k - 1) * params.damping_slope)
            for k in range(1, mode_count + 1)
        ]
    )
    ratios = np.array(params.mode_ratios, dtype=np.float64)
    return ModalModeArrays(weights=weights, dampings=dampings, ratios=ratios, weight_sum=float(weights.sum()))


# Precomputed once at import time -- pure functions of fixed preset
# constants above, so there is no reason to ever recompute these per
# block/per note. The live engine (B3) reads these directly.
WHITE_MODES = build_mode_arrays(WHITE_MODAL_PARAMS)
BLACK_MODES = build_mode_arrays(BLACK_MODAL_PARAMS)
HARMONY_MODES = build_mode_arrays(HARMONY_MODAL_PARAMS)
ACCENT_MODES = build_mode_arrays(ACCENT_MODAL_PARAMS)
PULSE_MODES = build_mode_arrays(PULSE_MODAL_PARAMS)

# One entry per possible "hit index" within a phrase (audio/phrase.py's
# PhraseDescription.events position, 0..MAX_PHRASE_EVENTS-1) -- each
# later event in the same phrase gets a progressively more damped
# variant of PULSE_MODAL_PARAMS, so a phrase's later events sound
# darker/more resonant, not just quieter (audio/phrase.py already
# handles "quieter" via each event's own amplitude). Precomputed once at
# import time, same discipline as WHITE_MODES/BLACK_MODES/etc above --
# AudioEngine only ever indexes into this fixed tuple, never builds a
# ModalModeArrays at render time.
PULSE_MODES_BY_HIT_INDEX = tuple(
    build_mode_arrays(
        dataclasses.replace(
            PULSE_MODAL_PARAMS,
            base_damping_per_second=PULSE_MODAL_PARAMS.base_damping_per_second + hit_index * PULSE_DAMPING_STEP_PER_HIT,
        )
    )
    for hit_index in range(MAX_PHRASE_EVENTS)
)


def evaluate_modal_modes(
    weights: np.ndarray, dampings: np.ndarray, frequencies_hz: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """
    Raw (not amplitude-normalized) sum of exponentially-decaying
    weighted sinusoids, evaluated at absolute time `t` (seconds since
    the shared onset every mode decays from). Shared primitive: the one
    place this summation is written -- modal_resonance below is just
    this function plus normalization/attack/fade shaping around a
    fixed-length buffer.
    """

    total = np.zeros_like(t)
    for weight, damping, frequency_hz in zip(weights, dampings, frequencies_hz):
        total += weight * np.exp(-damping * t) * np.sin(2 * np.pi * frequency_hz * t)
    return total


def linear_attack_gate(t: np.ndarray, attack_seconds: float) -> np.ndarray:
    """
    A 0->1 linear ramp reaching 1.0 at t=attack_seconds and staying
    there, evaluated at absolute time `t` (seconds since a note's own
    onset). Shared primitive: the live engine's per-block committed-note
    evaluator (audio/engine.py, B3a) uses this directly, so the same
    "smooth onset, no click" shape modal_resonance's own fixed-buffer
    attack ramp provides offline is not redefined a second, potentially
    drifting way. `attack_seconds <= 0` gates instantly to 1.0 (matches
    the "near-instant" test overrides used throughout the test suite).
    """

    if attack_seconds <= 0.0:
        return np.ones_like(t)
    return np.minimum(t / attack_seconds, 1.0)


def modal_resonance(
    fundamental_hz: float,
    params: ModalVoiceParams,
    duration_seconds: float,
    sample_rate: int,
    amplitude: float = 1.0,
    attack_seconds: float = 0.003,
) -> np.ndarray:
    """
    Renders one damped modal resonator -- see this module's docstring
    for the formula. Deterministic, fully vectorized (no per-sample
    recursion). The output's peak magnitude never exceeds `amplitude`:
    each mode's own peak magnitude is bounded by its weight (since
    exp(-d*t) <= 1 and |sin| <= 1), so the weight-normalized sum's peak
    is bounded by amplitude regardless of how the modes happen to align
    in phase.
    """

    n = int(round(duration_seconds * sample_rate))
    if n <= 0:
        return np.zeros(0)

    t = np.arange(n) / sample_rate

    modes = build_mode_arrays(params)
    frequencies_hz = fundamental_hz * modes.ratios

    total = evaluate_modal_modes(modes.weights, modes.dampings, frequencies_hz, t)
    total *= amplitude / modes.weight_sum

    attack_n = min(int(round(attack_seconds * sample_rate)), n)
    if attack_n > 0:
        attack_gate = np.ones(n)
        attack_gate[:attack_n] = np.linspace(0.0, 1.0, attack_n, endpoint=False)
        total *= attack_gate

    fade_n = min(int(round(END_FADE_SECONDS * sample_rate)), n)
    if fade_n > 0:
        fade_gate = np.ones(n)
        fade_gate[n - fade_n:] = np.linspace(1.0, 0.0, fade_n, endpoint=True)
        total *= fade_gate

    return total
