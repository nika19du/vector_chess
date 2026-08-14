"""
Rhythmic Layer v2 -- Phrase-Based Generative Rhythm: Phase B listening
candidates.

Deliberately isolated (not wired into audio/engine.py or any production
path) -- same experiments/ convention as every prior phase. This script
does NOT modify or import anything that changes production behavior; it
only reads the already-computed AudioMapping (pitch_hz, harmony_interval_
ratio, pulse_density -- all untouched, existing production fields) and
reuses existing, unmodified DSP primitives (audio/organic_synthesis.py's
modal_resonance, audio/organic_renderer.py's OrganicAudioRenderer,
audio/pulse_pattern.py's euclidean_pulse_pattern) to render three
candidate finite-phrase designs for a human listening pass.

Common decisions applied to ALL THREE candidates (so the comparison
isolates rhythmic GRAMMAR, the actual variable under test, not a second
confounding change):

- Phrase pitch is derived from the move's own Melody pitch
  (mapping.pitch_hz * PHRASE_PITCH_RATIO), not an independent fixed
  frequency -- ties the rhythmic layer audibly back to which square was
  played, addressing the "should Pulse remain an independent fixed-pitch
  voice" question directly (Option A from the request).
- Timbre reuses the existing, unmodified PULSE_MODAL_PARAMS (same modal
  family as Melody/Harmony/Accent, not an inharmonic drum sound).
- The phrase is FINITE (bounded duration, decays to true silence) and is
  generated once per committed move -- there is no continuous clock, no
  looping, no modulo-wraparound pattern that repeats indefinitely between
  moves. This is the core structural fix for the "metronome" diagnosis
  (see PHASE_A_AUDIT.md in this directory).
- event_count(density) = 1 + round(density * 3) -- 1 event (just the
  onset) at density 0, up to 4 events at density 1.0. Small, deliberately
  bounded -- "more internal events inside the phrase," never "more ticks
  on an infinite timeline."
- No RNG anywhere. Every placement/amplitude/pitch decision below is a
  pure function of already-existing, already-deterministic chess-derived
  values (mapping.pulse_density, mapping.pitch_hz,
  mapping.harmony_interval_ratio).

Each candidate differs only in HOW those events are placed and shaped
within the finite phrase window -- the actual question this listening
pass exists to answer.
"""

import dataclasses
from pathlib import Path

import chess
import numpy as np

from analysis.dynamics import analyze_dynamics
from audio.export import write_wav
from audio.mapping import build_audio_mapping
from audio.models import AudioClip, AudioMapping, RenderConfig
from audio.organic_renderer import OrganicAudioRenderer
from audio.organic_synthesis import PULSE_MODAL_PARAMS, ModalVoiceParams, modal_resonance
from audio.phrase import euclidean_pulse_pattern
from audio.synthesis import mix
from chess_engine.analyzer import analyze_position
from chess_engine.moves import execute_move

OUTPUT_DIR = Path(__file__).parent / "output"
SAMPLE_RATE = 44100
SLOT_DURATION_SECONDS = 3.0  # per-move allotment in the rendered comparison sequence
PHRASE_DURATION_SECONDS = 1.8  # the finite phrase itself -- well inside the slot, leaving guaranteed trailing silence
TARGET_PEAK = 0.9

MAX_EVENTS = 4
PHRASE_PITCH_RATIO = 2.0  # an octave above the move's own Melody pitch
PRIMARY_GAIN = 0.4

TACTICAL_SEQUENCE = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]


def _mappings_for_sequence(moves: list[str]) -> list[AudioMapping]:
    board = chess.Board()
    previous_analysis = None
    mappings = []

    for move_text in moves:
        details = execute_move(board, move_text)
        assert details is not None, f"expected {move_text} to be legal"
        analysis = analyze_position(board, details)
        dynamics = (
            None if previous_analysis is None else analyze_dynamics(previous=previous_analysis, current=analysis)
        )
        mappings.append(build_audio_mapping(analysis, dynamics))
        previous_analysis = analysis

    return mappings


def event_count_for_density(density: float, max_events: int = MAX_EVENTS) -> int:
    """1 event (onset only) at density 0.0, up to max_events at density 1.0. Deterministic, no RNG."""

    clamped = min(max(density, 0.0), 1.0)
    return 1 + round(clamped * (max_events - 1))


def _add_event(
    buf: np.ndarray,
    onset_seconds: float,
    pitch_hz: float,
    amplitude: float,
    sample_rate: int,
    params: ModalVoiceParams = PULSE_MODAL_PARAMS,
) -> None:
    """Renders one modal_resonance event and adds it into `buf` at the
    correct sample offset, bounded to buf's own length (never writes past
    the end)."""

    onset_sample = int(round(onset_seconds * sample_rate))
    remaining_seconds = len(buf) / sample_rate - onset_seconds
    if remaining_seconds <= 0.0 or amplitude <= 0.0:
        return

    event = modal_resonance(
        fundamental_hz=pitch_hz,
        params=params,
        duration_seconds=remaining_seconds,
        sample_rate=sample_rate,
        amplitude=amplitude,
        attack_seconds=0.005,
    )
    end_sample = min(onset_sample + len(event), len(buf))
    buf[onset_sample:end_sample] += event[: end_sample - onset_sample]


# ---------------------------------------------------------
# Candidate A -- Sparse Event-Resonance
#
# One primary strike at the move's onset, followed by 0-3 decelerating,
# progressively quieter echoes of the SAME event -- like a single struck
# object settling into silence, not a repeating grid. Echo k lands at
# phrase_duration * (1 - 0.5**k) (each echo halves its remaining
# distance to the phrase's end) and is quieter by a fixed factor per
# step (0.55**k). All echoes share the primary's pitch/timbre.
# ---------------------------------------------------------


def render_phrase_candidate_a(mapping: AudioMapping, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    buf = np.zeros(int(round(SLOT_DURATION_SECONDS * sample_rate)))
    pitch = mapping.pitch_hz * PHRASE_PITCH_RATIO
    n_events = event_count_for_density(mapping.pulse_density)

    _add_event(buf, 0.0, pitch, PRIMARY_GAIN, sample_rate)
    for k in range(1, n_events):
        t = PHRASE_DURATION_SECONDS * (1 - 0.5**k)
        amplitude = PRIMARY_GAIN * (0.55**k)
        _add_event(buf, t, pitch, amplitude, sample_rate)

    return buf


# ---------------------------------------------------------
# Candidate B -- Deterministic Call-Response
#
# Reuses the EXISTING production Euclidean placement algorithm
# (audio/pulse_pattern.py::euclidean_pulse_pattern, unmodified) to spread
# `event_count` hits across a small FIXED number of slots (not an
# infinite loop -- applied exactly once, within the finite phrase
# window). Alternates pitch between the move's own "call" pitch and a
# "response" pitch derived from the move's own already-computed harmony
# interval (mapping.harmony_interval_ratio) -- a legible relationship the
# instrument already establishes, not a new arbitrary interval. Hits
# decay mildly (0.85 per step) so the phrase still settles by its end.
# ---------------------------------------------------------

PHRASE_SLOTS = 5


def render_phrase_candidate_b(mapping: AudioMapping, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    buf = np.zeros(int(round(SLOT_DURATION_SECONDS * sample_rate)))
    call_pitch = mapping.pitch_hz * PHRASE_PITCH_RATIO
    response_pitch = call_pitch * mapping.harmony_interval_ratio
    n_events = event_count_for_density(mapping.pulse_density)

    pattern = euclidean_pulse_pattern(n_events, PHRASE_SLOTS)
    slot_duration = PHRASE_DURATION_SECONDS / PHRASE_SLOTS

    hit_index = 0
    for slot_index, active in enumerate(pattern):
        if not active:
            continue
        t = slot_index * slot_duration
        pitch = call_pitch if hit_index % 2 == 0 else response_pitch
        amplitude = PRIMARY_GAIN * (0.85**hit_index)
        _add_event(buf, t, pitch, amplitude, sample_rate)
        hit_index += 1

    return buf


# ---------------------------------------------------------
# Candidate C -- Hybrid: phrase with intensity subdivisions
#
# Combines B's deterministic Euclidean placement (where events land)
# with A's settling amplitude decay AND a per-hit brightness/damping
# increase (later hits are not just quieter but also more damped/darker
# -- "the phrase resonates and calms," not "the phrase repeats and
# stops"). The first hit is always the primary onset pitch; later hits
# use the harmony-derived response pitch, same legible relationship as B.
# ---------------------------------------------------------


def render_phrase_candidate_c(mapping: AudioMapping, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    buf = np.zeros(int(round(SLOT_DURATION_SECONDS * sample_rate)))
    call_pitch = mapping.pitch_hz * PHRASE_PITCH_RATIO
    response_pitch = call_pitch * mapping.harmony_interval_ratio
    n_events = event_count_for_density(mapping.pulse_density)

    pattern = euclidean_pulse_pattern(n_events, PHRASE_SLOTS)
    slot_duration = PHRASE_DURATION_SECONDS / PHRASE_SLOTS

    hit_index = 0
    for slot_index, active in enumerate(pattern):
        if not active:
            continue
        t = slot_index * slot_duration
        pitch = call_pitch if hit_index == 0 else response_pitch
        amplitude = PRIMARY_GAIN * (0.6**hit_index)
        params = dataclasses.replace(
            PULSE_MODAL_PARAMS,
            base_damping_per_second=PULSE_MODAL_PARAMS.base_damping_per_second + hit_index * 2.0,
        )
        _add_event(buf, t, pitch, amplitude, sample_rate, params=params)
        hit_index += 1

    return buf


CANDIDATES = {
    "candidate_a_sparse_resonance": render_phrase_candidate_a,
    "candidate_b_call_response": render_phrase_candidate_b,
    "candidate_c_hybrid_subdivision": render_phrase_candidate_c,
}


def normalize_peak(samples: np.ndarray, target_peak: float = TARGET_PEAK) -> np.ndarray:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 0.0:
        return samples
    return samples * (target_peak / peak)


def render_full_sequence(candidate_fn, mappings: list[AudioMapping]) -> np.ndarray:
    """
    Full instrument: Melody+Harmony(+capture Accent) via the existing,
    unmodified offline OrganicAudioRenderer, with the candidate's
    rhythmic phrase layer mixed on top -- so the listening comparison is
    heard exactly as it would sound in the finished instrument, not the
    rhythmic layer in isolation.
    """

    organic_renderer = OrganicAudioRenderer(RenderConfig(clip_duration_seconds=SLOT_DURATION_SECONDS))
    slots = []

    for mapping in mappings:
        instrument_clip = organic_renderer.render(mapping)
        phrase = candidate_fn(mapping)
        slot_samples = mix(instrument_clip.samples, phrase)
        slots.append(slot_samples)

    return np.concatenate(slots) if slots else np.zeros(0)


def measure(name: str, samples: np.ndarray, mappings: list[AudioMapping]) -> dict:
    """Peak, RMS, and a per-move trailing-silence check (last 0.4s of
    each move's slot must be numerically silent -- proves the phrase is
    finite, not a continuous clock)."""

    slot_samples = int(round(SLOT_DURATION_SECONDS * SAMPLE_RATE))
    tail_samples = int(round(0.4 * SAMPLE_RATE))

    tail_rms_values = []
    for i in range(len(mappings)):
        start = i * slot_samples
        end = start + slot_samples
        tail = samples[end - tail_samples : end]
        tail_rms_values.append(float(np.sqrt(np.mean(tail**2))) if tail.size else 0.0)

    return {
        "name": name,
        "peak": float(np.max(np.abs(samples))) if samples.size else 0.0,
        "rms": float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0,
        "max_tail_rms": max(tail_rms_values),
        "tail_rms_per_move": tail_rms_values,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mappings = _mappings_for_sequence(TACTICAL_SEQUENCE)

    print("densities per move:", [round(m.pulse_density, 3) for m in mappings])
    print("event counts per move:", [event_count_for_density(m.pulse_density) for m in mappings])
    print()

    reports = []
    for name, candidate_fn in CANDIDATES.items():
        first = render_full_sequence(candidate_fn, mappings)
        second = render_full_sequence(candidate_fn, mappings)
        assert np.array_equal(first, second), f"{name} is not deterministic!"

        normalized = normalize_peak(first)
        clip = AudioClip(samples=np.clip(normalized, -1.0, 1.0), sample_rate=SAMPLE_RATE)
        write_wav(clip, OUTPUT_DIR / f"{name}.wav")

        report = measure(name, normalized, mappings)
        reports.append(report)
        print(f"{name}: peak={report['peak']:.3f} rms={report['rms']:.4f} max_tail_rms={report['max_tail_rms']:.6f}")
        print(f"  tail_rms_per_move={[round(v, 6) for v in report['tail_rms_per_move']]}")

    print(f"\nWrote WAVs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
