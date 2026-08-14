"""
Audio Layer 2 -- Rhythmic Layer, v2: Phrase-Based Generative Rhythm.

Supersedes audio/pulse_pattern.py (deleted). v1's design was diagnosed
(by listening, then verified against the code -- see
experiments/rhythmic_layer_v2/PHASE_A_AUDIT.md) as sounding like a
metronome: `AudioEngine` computed `slot_index = (boundary // period) %
PULSE_SLOTS_PER_PHRASE` every block, a modulo wraparound that looped the
same fixed-period, fixed-pitch, fixed-timbre pattern forever once armed,
with nothing to ever silence it except the next move.

v2's model, chosen from a 3-candidate listening pass (Candidate C --
"Hybrid: phrase with intensity subdivisions" -- was the user's explicit
preference):

    committed move -> finite PhraseDescription -> a handful of
    deterministic events, decaying, then true silence -- until the next
    move.

This module is the ENTIRE pure musical grammar: given a move's
already-computed pitch/harmony/color, produce a bounded (<=
MAX_PHRASE_EVENTS), deterministic sequence of (onset, pitch, amplitude)
events. No RNG anywhere (docs/audio.md Design Principle 4: seeded
randomness is licensed only at the later Generative Music Engine stage,
not here). No Qt, no wall-clock time, no chess/analysis computation --
pure functions of already-existing values, safe to unit-test without
audio hardware and safe to call from the UI/analysis thread (this module
is never imported by anything that runs inside AudioEngine's real-time
callback; see audio/engine.py's own docstring for how the callback only
*consumes* an already-built PhraseDescription).
"""

from dataclasses import dataclass

# --- Dynamics.intensity -> pulse_density (unchanged from v1; this
# mapping was never the diagnosed problem -- only what "density" was
# applied to). Chosen to line up with analysis.dynamics.analyze_
# dynamics's own "chaotic" threshold (intensity >= 20).
INTENSITY_DENSITY_SCALE = 20.0


def pulse_density_for_intensity(intensity: float) -> float:
    """Dynamics.intensity -> [0, 1], clamped and linear."""

    if intensity <= 0.0:
        return 0.0
    if intensity >= INTENSITY_DENSITY_SCALE:
        return 1.0
    return intensity / INTENSITY_DENSITY_SCALE


# --- Phrase shape: fixed, not position-derived. Keeping duration and
# slot count fixed (only internal event count/placement/amplitude vary
# with density) isolates the one thing this milestone's listening pass
# was actually testing -- the same discipline v1 applied to
# PULSE_PERIOD_SECONDS, now applied to the whole phrase envelope.
PHRASE_DURATION_SECONDS = 1.8

# Events are placed on a 5-slot grid within the phrase, but at most
# MAX_PHRASE_EVENTS=4 of those 5 slots are ever filled -- at least one
# slot is always silent, by construction, so a phrase never fills every
# possible position. This is what keeps even the densest phrase feeling
# like "activity within a bounded gesture," not "a full grid."
PHRASE_SLOTS = 5
MAX_PHRASE_EVENTS = 4

# An octave above the move's own Melody pitch (audio/mapping.py's
# pitch_hz) -- ties the phrase audibly to which square was actually
# played, instead of v1's independent fixed 440Hz. Investigated and
# chosen from docs/audio.md's Option A ("Pulse derives its pitch from
# the current Melody: f_pulse = f_melody x musical_ratio").
PHRASE_PITCH_RATIO = 2.0

# Primary (first) event's amplitude; each subsequent event in the same
# phrase is quieter by this factor per step (0.6, 0.36, 0.216, ...) --
# Candidate C's "settling" decay, chosen over Candidate A's steeper
# 0.55**k (felt more like a delay effect at high density) and
# Candidate B's gentler 0.85**k (didn't read as "arriving at rest").
PULSE_GAIN = 0.4
AMPLITUDE_DECAY_PER_HIT = 0.6


@dataclass(frozen=True)
class PhraseEvent:
    """One event within a phrase. `onset_seconds` is relative to the
    phrase's own start (the committed move's NoteTrigger), never
    wall-clock time. Timbre/damping progression is NOT carried here --
    see audio/organic_synthesis.py's PULSE_MODES_BY_HIT_INDEX and this
    event's own position within PhraseDescription.events for why that's
    an engine-side rendering detail, not musical content."""

    onset_seconds: float
    pitch_hz: float
    amplitude: float


@dataclass(frozen=True)
class PhraseDescription:
    """A finite, deterministic, fully precomputed description of one
    move's rhythmic phrase -- at most MAX_PHRASE_EVENTS events, in onset
    order (a given event's index in `events` is also its "hit index",
    which audio/engine.py uses to select that event's timbral damping).
    `duration_seconds` is an informational bound (the phrase's own
    envelope length), not itself consulted by the engine's firing logic
    -- each event fires independently once its own onset_seconds has
    elapsed, and nothing fires past the last scheduled event."""

    events: tuple[PhraseEvent, ...]
    duration_seconds: float


EMPTY_PHRASE = PhraseDescription(events=(), duration_seconds=0.0)


def event_count_for_density(density: float, max_events: int = MAX_PHRASE_EVENTS) -> int:
    """
    density (0..1, clamped defensively) -> how many events this phrase
    contains, from 1 (density 0.0 -- just the move's own quiet onset;
    a move is always a musical event, even a calm one) up to max_events
    (density 1.0). Deterministic, monotonic non-decreasing in density.
    """

    clamped = min(max(density, 0.0), 1.0)
    return 1 + round(clamped * (max_events - 1))


def euclidean_pulse_pattern(active_slot_count: int, total_slots: int = PHRASE_SLOTS) -> tuple[bool, ...]:
    """
    Distributes `active_slot_count` active slots as evenly as possible
    across `total_slots`, using the classic Bresenham-line-style
    approximation to a Euclidean rhythm: deterministic, O(total_slots),
    no randomness. Exactly `active_slot_count` slots are True (clamped
    to [0, total_slots]) -- a placement rule, not a probability.

    Unchanged from v1 -- this primitive was never the diagnosed problem;
    only how its output was scheduled (forever, looping) was. Applied
    here exactly ONCE per phrase, never repeated.
    """

    if total_slots <= 0:
        return ()

    k = min(max(active_slot_count, 0), total_slots)

    if k == 0:
        return (False,) * total_slots
    if k == total_slots:
        return (True,) * total_slots

    pattern = [False] * total_slots
    bucket = 0
    for i in range(total_slots):
        bucket += k
        if bucket >= total_slots:
            bucket -= total_slots
            pattern[i] = True

    return tuple(pattern)


def build_phrase(
    *,
    pulse_density: float,
    pitch_hz: float,
    harmony_interval_ratio: float,
    color: str,
) -> PhraseDescription:
    """
    The one pure function that turns a move's already-computed state
    into a finite, deterministic PhraseDescription.

    Signal provenance (every mapping traces to an already-existing,
    already-justified value -- see the Phase C report's mapping table):
      - pulse_density (Dynamics.intensity, via pulse_density_for_intensity)
        -> event count (event_count_for_density) and placement
        (euclidean_pulse_pattern): "how much just changed" -> "how much
        internal activity this phrase has." The one genuinely new
        mapping this milestone introduces.
      - pitch_hz (the move's own Melody pitch, destination-square
        derived) -> this phrase's "call" pitch, one octave up
        (PHRASE_PITCH_RATIO): ties the phrase audibly to which square
        was played, instead of an independent fixed frequency.
      - harmony_interval_ratio (the move's own, already-quantized
        Attack Influence balance -> harmony interval -- the exact same
        number Harmony itself sounds) -> this phrase's "response"
        pitch: reuses Harmony's own already-established relationship
        rather than inventing a second, competing one.
      - color (white/black) -> which role (call or response) opens the
        phrase -- White's phrase opens on the call/tonic role, Black's
        opens on the response/harmonic role. A structural/dialogue
        difference, never a register or pitch-height difference (no
        shrillness, same absolute pitches available to both colors).

    Deliberately NOT used here (see the Phase C report's rejected-
    mappings section): is_capture/is_check (Accent's exclusive signal --
    using it here too would double-encode the same chess fact into two
    voices), Mobility (reserved for a future texture role, never
    rhythm), Dynamics.label (already spent on loudness), and Dynamics'
    own sub-components (attack_vectors_delta, heatmap_change -- already
    folded into the single composite `intensity`; decomposing them back
    out would fragment one designed signal into competing ones without
    new justification).
    """

    call_pitch = pitch_hz * PHRASE_PITCH_RATIO
    response_pitch = call_pitch * harmony_interval_ratio
    slot_duration = PHRASE_DURATION_SECONDS / PHRASE_SLOTS
    starts_with_response = color == "black"

    def _pitch_for_hit(hit_index: int) -> float:
        even_hit = hit_index % 2 == 0
        is_call_role = even_hit != starts_with_response
        return call_pitch if is_call_role else response_pitch

    def _amplitude_for_hit(hit_index: int) -> float:
        return PULSE_GAIN * (AMPLITUDE_DECAY_PER_HIT**hit_index)

    event_count = event_count_for_density(pulse_density)

    # The first event always marks the move itself, at the phrase's own
    # onset (t=0) -- the primary impulse a committed move causes, never
    # delayed regardless of density. Only ADDITIONAL events (when
    # density warrants more than one) are placed deterministically
    # across the remaining PHRASE_SLOTS-1 slots via the same Euclidean
    # bucket algorithm, so a calm move (event_count=1) is a single,
    # immediate, quiet acknowledgment -- not a delayed afterthought.
    events: list[PhraseEvent] = [
        PhraseEvent(onset_seconds=0.0, pitch_hz=_pitch_for_hit(0), amplitude=_amplitude_for_hit(0))
    ]

    remaining_slots = PHRASE_SLOTS - 1
    remaining_events = event_count - 1
    pattern = euclidean_pulse_pattern(remaining_events, remaining_slots)

    hit_index = 1
    for slot_offset, active in enumerate(pattern):
        if not active:
            continue

        onset_seconds = (slot_offset + 1) * slot_duration
        events.append(
            PhraseEvent(
                onset_seconds=onset_seconds,
                pitch_hz=_pitch_for_hit(hit_index),
                amplitude=_amplitude_for_hit(hit_index),
            )
        )
        hit_index += 1

    return PhraseDescription(events=tuple(events), duration_seconds=PHRASE_DURATION_SECONDS)
