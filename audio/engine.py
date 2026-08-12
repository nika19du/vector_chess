"""
Real-time audio runtime shell (Phase 5f.2).

Threading model (docs/interactive_ui.md Part 4.6, honestly restated --
see the module docstring notes below rather than overclaiming an exact
precedent match to PositionCache's dict-slot swap):

- `publish(state)` and `push_event(trigger)` are called from the UI
  thread (or, in these tests, the calling thread). `publish` replaces a
  single instance attribute -- a bare reference reassignment, safe
  under CPython's GIL as a single bytecode-level operation, but a new,
  explicitly-audited pattern for this codebase rather than a reuse of
  PositionCache's already-proven dict-slot-replace.
- `push_event` appends to a bounded `collections.deque` -- the standard
  library documents `append`/`popleft` on a deque as individually
  thread-safe for this exact single-producer/single-consumer shape.
- The real-time callback (`_render_block`) is invoked on the audio
  backend's own thread (PortAudio's callback thread for the real
  backend). It never touches Qt, chess/analysis code, PositionCache,
  disk I/O, or any lock. It reads `self._latest_state` once per block
  and drains at most one trigger per block.
- Mono output only (channels=1) for Phase 5f -- stereo/pan is
  deliberately out of scope; Finding #17 (Source Potential vs.
  Ridge/Valley both claiming stereo pan) stays deferred to whoever
  scopes Milestone 4b, untouched by this engine.

Sustained-bed headroom (Phase 5f.3a hardening -- see this module's own
`_render_block`): mirrors `audio.renderer`'s
`_normalize_to_peak(mix(melody, harmony), SUSTAINED_PEAK_HEADROOM)`
policy, reusing the same `SUSTAINED_PEAK_HEADROOM` constant rather than
inventing a live-only one. The offline renderer can measure the actual
peak of an entire pre-rendered clip before deciding a single scale
factor; a real-time callback cannot look ahead at samples it hasn't
generated yet, so this is an intentionally approximate, causal
equivalent: each block scales the sustained bed down (never up) using
the ANALYTIC worst-case peak bound of the currently-audible continuous
voices' amplitudes (melody_amplitude + harmony_amplitude, the maximum
two sinusoids summing could ever instantaneously reach), computed from
already-known per-block values -- no lookahead, no extra allocation, no
analysis. This is deliberately not bit-identical to the offline
renderer's own measured-peak result, only equivalent in intent and
effect. The Accent voice is mixed in AFTER this scaling, exactly like
the offline renderer's `mix(sustained, accent)` -- unnormalized,
protected only by the final safety clip, matching the offline
renderer's own already-accepted, unchanged behavior for a capturing
move's accent.

Real-time-safety note (see this module's honest limitation): output/mix
buffers are preallocated once and reused every block via numpy's `out=`
parameter wherever practical. The modal-synthesis inner loop for the
Melody/Harmony/Accent voices still allocates small, bounded numpy
temporaries per block (e.g. `ratio * phase`, `np.sin(...)`, one
`np.exp(...)` per mode) -- eliminating every one of those would require
a fully hand-unrolled buffer-management pass this phase does not
attempt. This does not violate the constraints actually required of the
hot path (no Qt, no analysis, no PositionCache, no locks, no disk I/O,
no *large* allocation) but it is not literally zero-allocation, and is
recorded here rather than silently assumed. Note-slot state itself
(`_ModalNoteState`) is fully preallocated (fixed-size arrays) and never
allocates on trigger -- see its own docstring.

B3a (restoring B2 timbre evolution live): B3 ported B2's mode
weights/ratios into the live engine but deliberately did NOT apply each
mode's own exp(-damping*t) decay for Melody/Harmony, instead keeping
the old, independently-configurable `_ArticulationEnvelope`
(attack -> plateau -> fixed-duration decay) as the note's sole temporal
shape. Phase 1 measurement (experiments/audio_b3a_audit/) confirmed
this was the audible regression's actual mechanism, not just a
plausible guess: `_ArticulationEnvelope`'s decay times (0.55s Melody,
1.1s Harmony) cut a note off roughly 3-5x sooner than B2's own slowest
(dominant) mode actually needs to reach silence, AND -- independent of
duration -- a note's relative mode balance never evolved at all while
audible (every mode stayed at its initial weight until the whole voice
was gated to zero together), so Black's core "denser at the strike,
darkens over time" identity was completely flattened live.

Resolution: Melody/Harmony's committed-note BODY, timbre evolution, and
final silence are now owned entirely by each mode's own accepted-B2
decay rate (`_ModalNoteState`, below) -- the same `WHITE_MODES`/
`BLACK_MODES`/`HARMONY_MODES` arrays the offline `OrganicAudioRenderer`
reads, evaluated via the same shared `evaluate_modal_modes` primitive
Accent already used since B3. `_ArticulationEnvelope` is retired for
Melody/Harmony (its class and its own direct unit tests are unaffected,
audio/organic_synthesis.py's `linear_attack_gate` now owns the one
remaining separate concern: a short, click-safe 0->1 onset ramp, the
same shape `modal_resonance`'s own fixed-buffer attack ramp uses
offline). One system, one clock, no envelope stacked on another.

Retrigger continuity (click-free, no discontinuity on note replacement)
is handled by giving each of Melody/Harmony two bounded note-slots
(`_ModalVoice`: `primary`/`secondary`) rather than by borrowing
`_ArticulationEnvelope`'s "continue from `_last_shape`" trick, which
does not generalize to per-mode decay (resetting an independent decay
clock on retrigger would itself reintroduce a click -- see the B3a
report). A retrigger demotes whatever is currently `primary` to
`secondary` (a reference swap, not an allocation) and starts a fresh
note in `primary`; `secondary` simply keeps decaying, untouched, via
its own already-established curve, until it independently crosses the
floor. Summing two independently-smooth curves is itself smooth --
no discontinuity is possible by construction. A third overlapping
retrigger (before `secondary` has finished) silently retires whatever
`secondary` currently holds -- a deliberate, bounded (fixed 2 slots)
simplification, not unbounded polyphony.

Scrub is unaffected by any of this: it never used `_ArticulationEnvelope`
or per-mode decay, and still doesn't -- a continuously undamped,
one-pole-smoothed weighted sum (still colored with B2's mode weights/
ratios), because a decaying voice would be wrong for a held preview
(requirement: scrub must not fade/retrigger like a plucked note).

Audio Layer 2 -- Rhythmic Layer (Pulse voice): a periodic, deterministic
k-of-n slot pattern (audio/pulse_pattern.py) derived purely from
`SonificationState.pulse_density`, using a running *integer* sample
counter (`_pulse_sample_counter`) reset only when a real committed
move's NoteTrigger lands -- never during scrub, never on a mixer-only
republish. Deliberately does NOT reuse the `TriggerBuffer` deque
mechanism AccentTrigger/NoteTrigger use: that buffer's drop-oldest,
at-most-one-pop-per-block semantics are correct for rare, independent
one-shot events (the newest matters, an occasional drop is fine) but
would be wrong for a *recurring* tick -- dropping the next-due tick
would be an audible rhythmic glitch, and popping at most once per ~20ms
block would quantize onsets to 20ms of jitter. Instead, Pulse timing is
derived fresh every block from the sample counter plus the currently
published `pulse_period_seconds`/`pulse_density` -- no cross-thread
queue for Pulse at all, so there is nothing to overflow or starve.
Playback reuses `_ModalVoice` unchanged (the same bounded 2-slot
polyphony Melody/Harmony already use) with a new, dedicated
`PULSE_MODES` timbre (audio/organic_synthesis.py) -- a short, soft,
harmonic tap, not an inharmonic drum sound.

`_pulse_armed` (not merely `_scrub_active`) gates whether Pulse may
trigger new ticks at all: it becomes True only when a NoteTrigger is
consumed (a real committed move), and False the instant
`set_scrub_active(True)` is called. This is what implements "resume
only from the real committed-node publish, not immediately from scrub
release": scrub_active flipping back to False (on `end_scrub`/
`cancel_scrub`) does not by itself re-arm Pulse -- only the following
NoteTrigger does, so a scrub cancelled without a subsequent commit
correctly leaves Pulse silent rather than resuming from a stale,
pre-scrub pattern.
"""

import collections
from dataclasses import dataclass

import numpy as np

from audio.backend import AudioBackend, Stream
from audio.live_state import SonificationState
from audio.organic_renderer import HARMONY_GAIN, SUSTAINED_PEAK_HEADROOM
from audio.organic_synthesis import (
    ACCENT_BASE_FREQUENCY_HZ,
    ACCENT_MODES,
    BLACK_MODES,
    HARMONY_MODES,
    PULSE_BASE_FREQUENCY_HZ,
    PULSE_MODES,
    WHITE_MODES,
    ModalModeArrays,
    evaluate_modal_modes,
    linear_attack_gate,
)
from audio.pulse_pattern import PULSE_SLOTS_PER_PHRASE, pulse_pattern_for_density
from audio.voices import VoiceRegistry

DEFAULT_SAMPLE_RATE = 44100
LATENCY_BUDGET_SECONDS = 0.02  # docs/interactive_ui.md Part 4.6's 20ms budget
DEFAULT_BLOCKSIZE = int(DEFAULT_SAMPLE_RATE * LATENCY_BUDGET_SECONDS)  # 882

# Upper bound on frames-per-callback this engine will ever preallocate
# scratch space for. Comfortably above any (sample_rate, blocksize)
# combination the 20ms budget implies at realistic sample rates.
MAX_BLOCK_FRAMES = 8192

# Minimal, block-rate (not per-sample) one-pole smoothing -- the
# pragmatic level already approved for Phase 5f.2, addressing
# docs/interactive_ui.md Finding M4 ("zipper noise") only enough for a
# proven-voice quality bar, not claimed as that finding's full
# resolution. Scrub-only (B3a): committed notes no longer use this.
DEFAULT_SMOOTHING_COEFFICIENT = 0.2

DEFAULT_TRIGGER_BUFFER_CAPACITY = 4

# Below this envelope amplitude a one-shot/committed modal voice
# (Accent, or -- B3a -- a Melody/Harmony note-slot) is considered
# finished and stops advancing/contributing. Shared by `_AccentState`
# and `_ModalNoteState` -- one floor, one meaning, not two independently
# tuned values that could quietly drift apart.
MODAL_ENVELOPE_FLOOR = 1e-4

# Fixed ceiling on modes-per-voice this engine ever preallocates
# fixed-size arrays for -- comfortably above WHITE_MODES (4), BLACK_MODES
# (5), HARMONY_MODES (2), ACCENT_MODES (3), so `_ModalNoteState.trigger`
# never needs to grow/reallocate its arrays.
MAX_MODE_COUNT = 5

# B3a -- the one remaining per-voice articulation knob: a short, linear,
# click-safe onset ramp (audio.organic_synthesis.linear_attack_gate).
# Note BODY/decay is no longer independently configurable -- it is
# whatever the accepted B2 mode dampings say it is (see module
# docstring). Melody is the short, foreground, "struck" voice; Harmony's
# onset is a touch softer, matching B2's own HARMONY_MODAL_PARAMS spirit.
MELODY_ATTACK_SECONDS = 0.012
HARMONY_ATTACK_SECONDS = 0.045

# Pulse (Audio Layer 2): a short, soft, restrained attack -- shorter
# than Melody's (this is a background tap, not a foreground note).
PULSE_ATTACK_SECONDS = 0.008

# Fixed, deliberately quiet -- Pulse's only signal is *how often* it
# ticks (pulse_density), not how loud any single tick is. Keeping
# per-tick amplitude constant (not scaled by density or loudness) keeps
# that one signal legible: density changes frequency of events, nothing
# else. Comparable in spirit to HARMONY_GAIN (subordinate to Melody),
# but quieter still -- Pulse is a texture layer, not a harmonic partner.
PULSE_GAIN = 0.35


@dataclass(frozen=True)
class AccentTrigger:
    """
    A one-shot accent event pushed onto the trigger buffer. Generic
    infrastructure only -- Phase 5f.2 does not decide which chess
    events produce one of these; that is Phase 5f.3's (AudioController)
    job. `kind` is carried through for the caller's own bookkeeping
    only; the accent voice's sound does not vary by kind.
    """

    kind: str
    loudness: float = 1.0


@dataclass(frozen=True)
class NoteTrigger:
    """
    A one-shot retrigger event for the Melody/Harmony note-slots
    (`_ModalVoice.trigger`). No payload needed: pitch, timbre, and
    loudness for the note already live in whatever `SonificationState`
    was just published via `publish()` -- this only says "start a new
    note now." Pushed exactly once per committed move
    (AudioController._on_current_node_changed), never during scrub
    preview.
    """


class TriggerBuffer:
    """
    Fixed-capacity FIFO for one-shot accent events. Bounded so a burst
    of events can never grow unbounded; overflow policy is deterministic
    drop-oldest (a `collections.deque(maxlen=...)` appended to on the
    right automatically discards from the left when full) -- the most
    recently issued event is the one most likely to still matter.
    """

    def __init__(self, capacity: int = DEFAULT_TRIGGER_BUFFER_CAPACITY) -> None:
        self.capacity = capacity
        self._items: collections.deque[AccentTrigger] = collections.deque(maxlen=capacity)

    def push(self, trigger: AccentTrigger) -> None:
        self._items.append(trigger)

    def pop(self) -> AccentTrigger | None:
        if not self._items:
            return None
        return self._items.popleft()

    def __len__(self) -> int:
        return len(self._items)


class _SmoothedParameter:
    """Block-rate one-pole smoothing toward a target value."""

    def __init__(self, initial: float = 0.0) -> None:
        self.value = initial

    def update(self, target: float, coefficient: float) -> float:
        self.value += (target - self.value) * coefficient
        return self.value


class _OscillatorState:
    """
    Mutable, audio-thread-owned state for one continuously-retuned
    voice. B3a: scrub-only -- committed Melody/Harmony notes use
    `_ModalVoice`/`_ModalNoteState` instead (see module docstring for
    why the two are genuinely different concerns, not the same
    mechanism reused). Never shared with the UI thread.
    """

    def __init__(self) -> None:
        self.phase = 0.0
        self.smoothed_frequency = _SmoothedParameter()
        self.smoothed_amplitude = _SmoothedParameter()


class _AccentState:
    """Mutable, audio-thread-owned state for the one-shot accent voice."""

    def __init__(self) -> None:
        self.active = False
        self.elapsed_seconds = 0.0
        self.loudness = 0.0

    def trigger(self, loudness: float) -> None:
        self.active = True
        self.elapsed_seconds = 0.0
        self.loudness = loudness


class _ModalNoteState:
    """
    Mutable, audio-thread-owned, bounded, audio-thread-owned state for
    one committed Melody/Harmony note (B3a). Mode arrays are fixed-size
    (`MAX_MODE_COUNT`), preallocated once in `__init__` and only ever
    overwritten in place by `trigger()` -- never resized, never
    reallocated, so retriggering a note is zero-allocation.

    Rendering reuses `evaluate_modal_modes` (audio/organic_synthesis.py)
    directly: each mode's frequency is fixed for this note's whole life
    (no continuous-glide concept here, unlike scrub's `_OscillatorState`),
    so absolute time-since-trigger is the natural, correct clock -- the
    same approach Accent already used since B3, and the same approach
    `modal_resonance` uses offline for a whole fixed-length clip.
    """

    def __init__(self) -> None:
        self.active = False
        self.elapsed_seconds = 0.0
        self.mode_count = 0
        self.weights = np.zeros(MAX_MODE_COUNT)
        self.dampings = np.zeros(MAX_MODE_COUNT)
        self.frequencies_hz = np.zeros(MAX_MODE_COUNT)
        self.weight_sum = 1.0
        self.amplitude = 0.0
        self.attack_seconds = 0.0

    def trigger(
        self,
        *,
        fundamental_hz: float,
        modes: ModalModeArrays,
        amplitude: float,
        attack_seconds: float,
    ) -> None:
        count = len(modes.weights)
        self.mode_count = count
        self.weights[:count] = modes.weights
        self.dampings[:count] = modes.dampings
        self.frequencies_hz[:count] = fundamental_hz * modes.ratios
        self.weight_sum = modes.weight_sum
        self.amplitude = amplitude
        self.attack_seconds = attack_seconds
        self.elapsed_seconds = 0.0
        self.active = True

    def render_into(self, buf: np.ndarray, frames: int, sample_rate: int, arange: np.ndarray) -> float:
        """
        Adds this note's contribution into `buf[:frames]` (does not
        clear it first -- callers mix multiple note-slots together).
        Returns an analytic, causal upper bound on this note's own peak
        contribution this block (for the sustained-bed headroom
        computation in `_render_block`) -- 0.0 if inactive.
        """

        if not self.active:
            return 0.0

        count = self.mode_count
        t = arange[:frames] / sample_rate + self.elapsed_seconds
        attack_gate = linear_attack_gate(t, self.attack_seconds)

        contribution = evaluate_modal_modes(
            self.weights[:count], self.dampings[:count], self.frequencies_hz[:count], t
        )
        contribution *= attack_gate * (self.amplitude / self.weight_sum)
        buf[:frames] += contribution

        self.elapsed_seconds += frames / sample_rate

        # Modes are constructed with non-decreasing damping by index
        # (see ModalVoiceParams's docstring), so index 0 is always the
        # slowest-decaying/dominant mode -- its own envelope (times the
        # attack gate) is both the peak bound this block and the
        # deactivation check.
        dominant_envelope = float(attack_gate[-1] * np.exp(-self.dampings[0] * t[-1]))
        if dominant_envelope < MODAL_ENVELOPE_FLOOR and t[-1] > self.attack_seconds:
            self.active = False

        return self.amplitude * dominant_envelope


class _ModalVoice:
    """
    Two bounded, preallocated note-slots (`primary` = most recently
    triggered note, `secondary` = the note it superseded, left to keep
    ringing down independently) for one committed Melody/Harmony voice
    -- see module docstring for why this replaces
    `_ArticulationEnvelope`'s single-shape retrigger-continuity trick.

    `trigger()` is a reference swap (`primary, secondary = secondary,
    primary`) followed by overwriting the now-`primary` slot's fixed
    arrays in place -- zero allocation, exactly like `_ModalNoteState`
    itself. A note that was already `secondary` when a third retrigger
    arrives is silently discarded (retired without ever being read
    again) -- a deliberate, bounded simplification: real committed
    moves are not expected to overlap three deep, and unbounded
    polyphony is explicitly out of scope for this phase.
    """

    def __init__(self) -> None:
        self._slot_a = _ModalNoteState()
        self._slot_b = _ModalNoteState()
        self.primary = self._slot_a
        self.secondary = self._slot_b

    def trigger(self, **kwargs) -> None:
        self.primary, self.secondary = self.secondary, self.primary
        self.primary.trigger(**kwargs)

    def render_into(self, buf: np.ndarray, frames: int, sample_rate: int, arange: np.ndarray) -> float:
        peak_bound = self.primary.render_into(buf, frames, sample_rate, arange)
        peak_bound += self.secondary.render_into(buf, frames, sample_rate, arange)
        return peak_bound


def _sample_boundaries_in_block(block_start_sample: int, frames: int, period_samples: int) -> list[int]:
    """
    Every integer multiple of `period_samples` in
    [block_start_sample, block_start_sample + frames) -- i.e. every
    Pulse-slot boundary this block's audio actually spans, computed with
    plain integer arithmetic (no float accumulation drift over a
    long-running session, unlike repeatedly adding frames/sample_rate).
    Includes `block_start_sample` itself when it lands exactly on a
    boundary, so a freshly-armed counter's slot 0 fires immediately
    rather than after one full period of silence. Handles multiple
    boundaries in one block correctly (relevant if period_samples is
    ever smaller than a block, e.g. in a test), not just the common
    real-world case of at most one.
    """

    if period_samples <= 0:
        return []

    block_end_sample = block_start_sample + frames
    first_boundary = (block_start_sample // period_samples) * period_samples
    if first_boundary < block_start_sample:
        first_boundary += period_samples

    boundaries = []
    boundary = first_boundary
    while boundary < block_end_sample:
        boundaries.append(boundary)
        boundary += period_samples
    return boundaries


class AudioEngine:
    """
    Owns the real-time output stream, the published-state slot, the
    trigger buffer, and all oscillator/note-slot state. Constructed with
    an injected `AudioBackend` so tests never touch real hardware.
    """

    def __init__(
        self,
        backend: AudioBackend,
        voice_registry: VoiceRegistry,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        blocksize: int = DEFAULT_BLOCKSIZE,
        channels: int = 1,
        trigger_capacity: int = DEFAULT_TRIGGER_BUFFER_CAPACITY,
        smoothing_coefficient: float = DEFAULT_SMOOTHING_COEFFICIENT,
        melody_attack_seconds: float = MELODY_ATTACK_SECONDS,
        harmony_attack_seconds: float = HARMONY_ATTACK_SECONDS,
    ) -> None:
        if channels != 1:
            raise ValueError(
                "AudioEngine only supports mono output in Phase 5f -- "
                "stereo/pan is deferred to whoever scopes Milestone 4b "
                "(docs/interactive_ui.md Finding #17)"
            )

        self._backend = backend
        self._voice_registry = voice_registry
        self._sample_rate = sample_rate
        self._blocksize = blocksize
        self._channels = channels
        self._smoothing_coefficient = smoothing_coefficient
        self._melody_attack_seconds = melody_attack_seconds
        self._harmony_attack_seconds = harmony_attack_seconds

        self._arange = np.arange(MAX_BLOCK_FRAMES, dtype=np.float64)
        self._scratch_melody = np.zeros(MAX_BLOCK_FRAMES, dtype=np.float64)
        self._scratch_harmony = np.zeros(MAX_BLOCK_FRAMES, dtype=np.float64)
        self._scratch_accent = np.zeros(MAX_BLOCK_FRAMES, dtype=np.float64)
        self._scratch_pulse = np.zeros(MAX_BLOCK_FRAMES, dtype=np.float64)
        self._scratch_mix = np.zeros(MAX_BLOCK_FRAMES, dtype=np.float64)

        self._latest_state: SonificationState | None = None
        self._trigger_buffer = TriggerBuffer(trigger_capacity)
        self._note_trigger_buffer = TriggerBuffer(trigger_capacity)

        self._melody_osc = _OscillatorState()  # scrub-only, see class docstring
        self._harmony_osc = _OscillatorState()  # scrub-only, see class docstring
        self._accent_state = _AccentState()

        self._melody_voice = _ModalVoice()
        self._harmony_voice = _ModalVoice()
        self._pulse_voice = _ModalVoice()

        # Audio Layer 2 -- Rhythmic Layer: integer sample counter (never
        # float-accumulated, so no drift over a long-running phrase) that
        # is Pulse's entire clock, plus the "may Pulse trigger new ticks
        # right now" gate -- see this module's docstring for why this is
        # a dedicated flag rather than reusing `_scrub_active` directly.
        self._pulse_sample_counter = 0
        self._pulse_armed = False

        # True only while AudioController's scrub gesture is active (set
        # via set_scrub_active) -- gates Melody/Harmony between the
        # always-on continuous-glide behavior (scrub preview) and the
        # note-slot-gated, silence-between-notes behavior (committed/idle
        # playback). Plain attribute, same GIL-atomic-reassignment safety
        # argument already documented for `publish()` above -- set from
        # the UI thread, read from the audio thread, no lock.
        self._scrub_active = False

        self._stream: Stream | None = None
        self._running = False
        self._shut_down = False
        self._unavailable = False
        self.last_open_error: Exception | None = None

    # -- lifecycle -----------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_available(self) -> bool:
        return not self._unavailable

    @property
    def is_shut_down(self) -> bool:
        return self._shut_down

    def start(self) -> bool:
        """
        Idempotent: returns True immediately if already running, False
        immediately (no-op) if already shut down. On a device-open
        failure, sets `is_available` False and returns False -- never
        raises, never crashes the caller.
        """

        if self._shut_down:
            return False
        if self._running:
            return True

        try:
            if self._stream is None:
                self._stream = self._backend.open_stream(
                    samplerate=self._sample_rate,
                    blocksize=self._blocksize,
                    channels=self._channels,
                    callback=self._render_block,
                )
            self._stream.start()
        except Exception as error:  # device/backend failure of any kind
            self._unavailable = True
            self.last_open_error = error
            self._stream = None
            return False

        self._running = True
        self._unavailable = False
        return True

    def stop(self) -> None:
        """Pauses the stream (idempotent no-op if not running). The
        stream is kept open so `start()` can resume it without
        reopening the device."""

        if not self._running or self._stream is None:
            return

        self._stream.stop()
        self._running = False

    def shutdown(self) -> None:
        """Final teardown -- idempotent. After shutdown, `start()`
        always returns False; the engine cannot be resurrected."""

        if self._shut_down:
            return

        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        self._running = False
        self._shut_down = True

    # -- publication (UI thread) ---------------------------------------

    def publish(self, state: SonificationState) -> None:
        self._latest_state = state

    def push_event(self, trigger: AccentTrigger) -> None:
        self._trigger_buffer.push(trigger)

    def push_note_trigger(self) -> None:
        """Starts a new committed Melody/Harmony note (B3a: `_ModalVoice.trigger`).
        Called once per committed move; never during scrub preview. A
        bare deque push, same thread-safety as push_event."""

        self._note_trigger_buffer.push(NoteTrigger())

    def set_scrub_active(self, active: bool) -> None:
        """
        True while a scrub gesture is in progress -- keeps Melody/
        Harmony on their original always-on, continuously-glided
        behavior for scrub preview. False (the default) uses the
        note-slot-gated, silence-between-notes behavior for ordinary
        committed-move playback. Bare attribute write; see this
        attribute's own docstring in __init__ for the thread-safety
        argument.

        Also disarms Pulse (`_pulse_armed = False`) whenever scrub
        begins -- Pulse only re-arms on the next real committed move's
        NoteTrigger (see module docstring), never merely because
        scrub_active later flips back to False.
        """

        self._scrub_active = active
        if active:
            self._pulse_armed = False

    # -- real-time callback (audio thread) ------------------------------

    def _render_block(self, outdata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        state = self._latest_state

        trigger = self._trigger_buffer.pop()
        if trigger is not None:
            self._accent_state.trigger(trigger.loudness)

        note_trigger = self._note_trigger_buffer.pop()
        if note_trigger is not None and state is not None:
            melody_modes = BLACK_MODES if state.color == "black" else WHITE_MODES
            self._melody_voice.trigger(
                fundamental_hz=state.pitch_hz,
                modes=melody_modes,
                amplitude=state.loudness,
                attack_seconds=self._melody_attack_seconds,
            )
            self._harmony_voice.trigger(
                fundamental_hz=_harmony_frequency_for_state(state),
                modes=HARMONY_MODES,
                amplitude=state.loudness * HARMONY_GAIN,
                attack_seconds=self._harmony_attack_seconds,
            )
            # Audio Layer 2: a real committed move is exactly the moment
            # Pulse is allowed to (re)start ticking -- see module
            # docstring on why this is a dedicated arm/reset, not merely
            # "scrub_active is False".
            self._pulse_sample_counter = 0
            self._pulse_armed = True

        melody, melody_amplitude = self._render_melody(state, frames)
        harmony, harmony_amplitude = self._render_harmony(state, frames)
        pulse, pulse_amplitude = self._render_pulse(state, frames)
        accent = self._render_accent(frames)

        sustained_buf = self._scratch_mix[:frames]
        sustained_buf.fill(0.0)

        if state is not None:
            mute = state.voice_mute
            solo = state.voice_solo
            any_solo = any(solo.values())

            def audible(voice_id: str) -> bool:
                # Phase 5f.5: mute always wins for that voice, even when
                # it is simultaneously soloed -- checked first and
                # unconditionally, before solo-mode gating ever applies.
                if mute.get(voice_id, False):
                    return False
                if any_solo:
                    return solo.get(voice_id, False)
                return True

            melody_audible = audible("melody")
            harmony_audible = audible("harmony")
            pulse_audible = audible("pulse")

            peak_bound = 0.0
            if melody_audible:
                sustained_buf += melody
                peak_bound += melody_amplitude
            if harmony_audible:
                sustained_buf += harmony
                peak_bound += harmony_amplitude
            if pulse_audible:
                # Pulse joins the headroom-managed sustained bed, not
                # Accent's raw/unnormalized-after-headroom mix: unlike
                # Accent (rare, one-shot per capture/check), Pulse ticks
                # recur regularly, so letting it bypass headroom the way
                # Accent does would risk routine, frequent clipping
                # rather than Accent's rare transient.
                sustained_buf += pulse
                peak_bound += pulse_amplitude

            # Sustained-bed headroom -- see this module's docstring.
            if peak_bound > SUSTAINED_PEAK_HEADROOM:
                sustained_buf *= SUSTAINED_PEAK_HEADROOM / peak_bound

            if audible("accent"):
                sustained_buf += accent

            gain = state.master_gain
        else:
            gain = 0.0

        np.multiply(sustained_buf, gain, out=sustained_buf)
        # Final safety clip -- a last resort, not the headroom policy
        # itself. With the headroom step above, this should only ever
        # engage for a capturing/checking move's accent transient
        # (the offline renderer's own already-accepted behavior),
        # never for the sustained bed alone.
        np.clip(sustained_buf, -1.0, 1.0, out=sustained_buf)

        outdata[:frames, 0] = sustained_buf

    def _render_melody(self, state: SonificationState | None, frames: int) -> tuple[np.ndarray, float]:
        osc = self._melody_osc
        buf = self._scratch_melody[:frames]
        buf.fill(0.0)

        if self._scrub_active:
            # Scrub preview (unchanged mechanism): one continuously
            # morphing voice -- both frequency and amplitude chase their
            # targets via the same one-pole smoother. Colored with the
            # accepted B2 mode weights/ratios (including Black's
            # sub-octave) but deliberately undamped -- see this module's
            # docstring for why a decaying voice would be wrong here.
            if state is None:
                target_frequency, target_amplitude, modes = 0.0, 0.0, WHITE_MODES
            else:
                target_frequency = state.pitch_hz
                target_amplitude = state.loudness
                modes = BLACK_MODES if state.color == "black" else WHITE_MODES

            frequency = osc.smoothed_frequency.update(target_frequency, self._smoothing_coefficient)
            amplitude = osc.smoothed_amplitude.update(target_amplitude, self._smoothing_coefficient)

            if frequency > 0.0 and amplitude > 0.0:
                t = self._arange[:frames] / self._sample_rate
                phase = osc.phase + 2 * np.pi * frequency * t
                for weight, ratio in zip(modes.weights, modes.ratios):
                    buf += weight * np.sin(ratio * phase)
                buf *= amplitude / modes.weight_sum

            osc.phase = (osc.phase + 2 * np.pi * frequency * frames / self._sample_rate) % (2 * np.pi)
            return buf, amplitude

        # Committed/idle mode (B3a): the note-slot pair owns attack,
        # body, timbre evolution, and final silence entirely -- see
        # module docstring. `_melody_osc` is not touched here; it is a
        # scrub-only concept.
        amplitude = self._melody_voice.render_into(buf, frames, self._sample_rate, self._arange)
        return buf, amplitude

    def _render_harmony(self, state: SonificationState | None, frames: int) -> tuple[np.ndarray, float]:
        osc = self._harmony_osc
        buf = self._scratch_harmony[:frames]
        buf.fill(0.0)
        modes = HARMONY_MODES

        if self._scrub_active:
            if state is None:
                target_frequency, target_amplitude = 0.0, 0.0
            else:
                target_frequency = _harmony_frequency_for_state(state)
                target_amplitude = state.loudness * HARMONY_GAIN

            frequency = osc.smoothed_frequency.update(target_frequency, self._smoothing_coefficient)
            amplitude = osc.smoothed_amplitude.update(target_amplitude, self._smoothing_coefficient)

            if frequency > 0.0 and amplitude > 0.0:
                t = self._arange[:frames] / self._sample_rate
                phase = osc.phase + 2 * np.pi * frequency * t
                for weight, ratio in zip(modes.weights, modes.ratios):
                    buf += weight * np.sin(ratio * phase)
                buf *= amplitude / modes.weight_sum

            osc.phase = (osc.phase + 2 * np.pi * frequency * frames / self._sample_rate) % (2 * np.pi)
            return buf, amplitude

        amplitude = self._harmony_voice.render_into(buf, frames, self._sample_rate, self._arange)
        return buf, amplitude

    def _render_pulse(self, state: SonificationState | None, frames: int) -> tuple[np.ndarray, float]:
        """
        Audio Layer 2 -- Rhythmic Layer. No scrub branch (unlike Melody/
        Harmony): Pulse has exactly one behavior, entirely frozen
        (`_pulse_armed` False) during scrub and while unarmed, and
        note-slot-gated (via `_ModalVoice`, same as committed Melody/
        Harmony) once armed. `_pulse_voice.render_into` is still called
        unconditionally so an already-ringing tick can finish decaying
        even in a block where no new tick is scheduled.
        """

        buf = self._scratch_pulse[:frames]
        buf.fill(0.0)

        if self._pulse_armed and not self._scrub_active and state is not None and state.pulse_period_seconds > 0.0:
            period_samples = max(1, round(state.pulse_period_seconds * self._sample_rate))
            pattern = pulse_pattern_for_density(state.pulse_density, PULSE_SLOTS_PER_PHRASE)

            for boundary in _sample_boundaries_in_block(self._pulse_sample_counter, frames, period_samples):
                slot_index = (boundary // period_samples) % PULSE_SLOTS_PER_PHRASE
                if pattern[slot_index]:
                    self._pulse_voice.trigger(
                        fundamental_hz=PULSE_BASE_FREQUENCY_HZ,
                        modes=PULSE_MODES,
                        amplitude=PULSE_GAIN,
                        attack_seconds=PULSE_ATTACK_SECONDS,
                    )

            self._pulse_sample_counter += frames

        amplitude = self._pulse_voice.render_into(buf, frames, self._sample_rate, self._arange)
        return buf, amplitude

    def _render_accent(self, frames: int) -> np.ndarray:
        """
        Accent has no competing outer envelope -- its own
        `_AccentState.elapsed_seconds` already *is* the note's whole
        temporal shape. A multi-mode "knock" (ACCENT_MODES), reusing
        evaluate_modal_modes directly -- unchanged since B3.
        """

        buf = self._scratch_accent[:frames]
        state = self._accent_state

        if not state.active:
            buf.fill(0.0)
            return buf

        t = self._arange[:frames] / self._sample_rate + state.elapsed_seconds
        modes = ACCENT_MODES
        frequencies_hz = ACCENT_BASE_FREQUENCY_HZ * modes.ratios

        buf[:] = evaluate_modal_modes(modes.weights, modes.dampings, frequencies_hz, t)
        buf *= state.loudness / modes.weight_sum

        state.elapsed_seconds += frames / self._sample_rate
        if np.exp(-modes.dampings[0] * t[-1]) < MODAL_ENVELOPE_FLOOR:
            state.active = False

        return buf


def _harmony_frequency_for_state(state: SonificationState) -> float:
    """Same balance-sign direction rule as audio/organic_renderer.py's
    own `_harmony_frequency` (balance >= 0 raises the interval above the
    melody pitch, balance < 0 lowers it below), expressed in terms of
    the live `SonificationState.harmony_above_melody` flag rather than
    the raw balance value -- `AudioController`/`audio.live_state` already
    reduce that sign to this boolean once, upstream."""

    if state.harmony_above_melody:
        return state.pitch_hz * state.harmony_interval_ratio
    return state.pitch_hz / state.harmony_interval_ratio
