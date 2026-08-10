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
voices' smoothed amplitudes (melody_amplitude + harmony_amplitude, the
maximum two sinusoids summing could ever instantaneously reach),
computed from already-known per-block values -- no lookahead, no extra
allocation, no analysis. This is deliberately not bit-identical to the
offline renderer's own measured-peak result, only equivalent in intent
and effect (see the Phase 5f.3a report for measured before/after
numbers). The Accent voice is mixed in AFTER this scaling, exactly like
the offline renderer's `mix(sustained, accent)` -- unnormalized,
protected only by the final safety clip, matching the offline
renderer's own already-accepted, unchanged behavior for a capturing
move's accent.

Real-time-safety note (see this module's honest limitation, reported
alongside the Phase 5f.2 test results): output/mix/phase/decay buffers
are preallocated once and reused every block via numpy's `out=`
parameter wherever practical. The additive-synthesis inner loop for the
Melody voice still allocates small, bounded numpy temporaries per
block (e.g. `k * phase`, `np.sin(...)`) -- eliminating every one of
those would require a fully hand-unrolled buffer-management pass this
phase does not attempt. This does not violate the constraints actually
required of the hot path (no Qt, no analysis, no PositionCache, no
locks, no disk I/O, no *large* allocation) but it is not literally
zero-allocation, and is recorded here rather than silently assumed.
"""

import collections
from dataclasses import dataclass

import numpy as np

from audio.backend import AudioBackend, Stream
from audio.live_state import SonificationState
from audio.renderer import HARMONY_VOICE_WEIGHT, SUSTAINED_PEAK_HEADROOM
from audio.synthesis import PERCUSSIVE_CLICK_FREQUENCY_HZ, PERCUSSIVE_DECAY_RATE
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
# resolution.
DEFAULT_SMOOTHING_COEFFICIENT = 0.2

DEFAULT_TRIGGER_BUFFER_CAPACITY = 4

# Below this envelope amplitude the accent voice is considered finished
# and stops advancing/contributing.
ACCENT_ENVELOPE_FLOOR = 1e-4

# Phase 5f.4a -- Live voice articulation.
#
# Melody and Harmony previously had no note concept at all: their
# oscillators ran forever, only ever retuned via the same one-pole
# smoother used for zipper-noise suppression, which is exactly what
# made a new move sound like a siren-style portamento sweep rather than
# a struck note. This phase adds a bounded, per-voice
# attack -> optional plateau -> exponential-decay-to-silence envelope
# (see `_ArticulationEnvelope` below), used whenever the engine is NOT
# in active scrub mode. While `_scrub_active` is True, Melody/Harmony
# keep their original always-on, continuously-retuned behavior
# unchanged -- that is the already-tested, intentional "one continuous
# morphing preview voice" scrub contract
# (tests/test_desktop_app_main_window_audio_scrub.py's own "harmony
# should glide continuously" assertion), not a bug this phase touches.
#
# These durations are deliberately named, separate constants rather
# than inlined literals: starting points for tuning after the manual
# listening protocol (docs/audio.md), not sacred values. Melody is the
# short, foreground, "struck" voice; Harmony is the softer, longer,
# receding support voice -- see this module's own report for the
# rationale.
MELODY_ATTACK_SECONDS = 0.012
MELODY_PLATEAU_SECONDS = 0.12
MELODY_DECAY_SECONDS = 0.55

HARMONY_ATTACK_SECONDS = 0.045
HARMONY_PLATEAU_SECONDS = 0.0
HARMONY_DECAY_SECONDS = 1.1

# Below this shape fraction (of peak), a Melody/Harmony note is
# considered finished and the voice goes fully silent until the next
# retrigger -- same order of magnitude and intent as ACCENT_ENVELOPE_FLOOR.
ARTICULATION_ENVELOPE_FLOOR = 1e-4


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
    A one-shot retrigger event for the Melody/Harmony articulation
    envelopes (Phase 5f.4a). No payload needed: pitch, timbre, and
    loudness for the note already live in whatever `SonificationState`
    was just published via `publish()` -- this only says "restart your
    envelope now." Pushed exactly once per committed move
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
    """Mutable, audio-thread-owned state for one continuous voice.
    Never shared with the UI thread."""

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


class _ArticulationEnvelope:
    """
    Mutable, audio-thread-owned attack -> optional plateau ->
    exponential-decay-to-silence shape (0..1 multiplier) for one
    continuous voice's amplitude in committed/idle (non-scrub) mode.
    Phase 5f.4a.

    Closed-form, evaluated as a function of elapsed time since the last
    `trigger()` call, exactly like `_AccentState`/`_render_accent`
    already do for the one-shot accent voice -- `render()` fills a
    per-sample array in one vectorized pass (never a single scalar per
    block), so the shape stays continuous even when a stage boundary
    (e.g. attack -> plateau) falls in the middle of a block.

    Retrigger-safe by construction: `trigger()` captures whatever shape
    value the envelope last produced (`_last_shape`) and uses it as the
    new attack ramp's starting point instead of forcing a hard drop to
    0 -- a rapid retrigger during decay ramps smoothly back up from
    wherever it currently is, never introducing an amplitude
    discontinuity (see this module's report on click/pop prevention).
    """

    def __init__(
        self,
        *,
        attack_seconds: float,
        plateau_seconds: float,
        decay_seconds: float,
        floor: float,
    ) -> None:
        self._attack_seconds = attack_seconds
        self._plateau_seconds = plateau_seconds
        self._plateau_end = attack_seconds + plateau_seconds
        self._floor = floor
        # Exponential decay rate solved so the shape reaches `floor`
        # exactly `decay_seconds` after decay begins (decay always
        # starts from shape == 1.0).
        self._decay_rate = -np.log(floor) / decay_seconds if decay_seconds > 0.0 else float("inf")

        self.active = False
        self.elapsed_seconds = 0.0
        self._attack_start_shape = 0.0
        self._last_shape = 0.0

    def trigger(self) -> None:
        self._attack_start_shape = self._last_shape
        self.elapsed_seconds = 0.0
        self.active = True

    def render(self, frames: int, sample_rate: int, arange: np.ndarray, out: np.ndarray) -> None:
        """Writes this block's 0..1 shape multiplier into `out[:frames]`."""

        segment = out[:frames]

        if not self.active:
            segment.fill(0.0)
            self._last_shape = 0.0
            return

        t = arange[:frames] / sample_rate + self.elapsed_seconds

        attack_mask = t < self._attack_seconds
        plateau_mask = (~attack_mask) & (t < self._plateau_end)
        decay_mask = ~attack_mask & ~plateau_mask

        segment[attack_mask] = self._attack_start_shape + (1.0 - self._attack_start_shape) * (
            t[attack_mask] / self._attack_seconds
        )
        segment[plateau_mask] = 1.0
        segment[decay_mask] = np.exp(-self._decay_rate * (t[decay_mask] - self._plateau_end))

        np.clip(segment, 0.0, 1.0, out=segment)

        self.elapsed_seconds += frames / sample_rate

        if frames > 0:
            self._last_shape = float(segment[-1])
            if self._last_shape < self._floor and t[-1] >= self._plateau_end:
                self.active = False
                self._last_shape = 0.0


class AudioEngine:
    """
    Owns the real-time output stream, the published-state slot, the
    trigger buffer, and all oscillator/envelope state. Constructed with
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
        melody_plateau_seconds: float = MELODY_PLATEAU_SECONDS,
        melody_decay_seconds: float = MELODY_DECAY_SECONDS,
        harmony_attack_seconds: float = HARMONY_ATTACK_SECONDS,
        harmony_plateau_seconds: float = HARMONY_PLATEAU_SECONDS,
        harmony_decay_seconds: float = HARMONY_DECAY_SECONDS,
        articulation_envelope_floor: float = ARTICULATION_ENVELOPE_FLOOR,
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

        self._arange = np.arange(MAX_BLOCK_FRAMES, dtype=np.float64)
        self._scratch_melody = np.zeros(MAX_BLOCK_FRAMES, dtype=np.float64)
        self._scratch_harmony = np.zeros(MAX_BLOCK_FRAMES, dtype=np.float64)
        self._scratch_accent = np.zeros(MAX_BLOCK_FRAMES, dtype=np.float64)
        self._scratch_mix = np.zeros(MAX_BLOCK_FRAMES, dtype=np.float64)
        self._scratch_melody_envelope = np.zeros(MAX_BLOCK_FRAMES, dtype=np.float64)
        self._scratch_harmony_envelope = np.zeros(MAX_BLOCK_FRAMES, dtype=np.float64)

        self._latest_state: SonificationState | None = None
        self._trigger_buffer = TriggerBuffer(trigger_capacity)
        self._note_trigger_buffer = TriggerBuffer(trigger_capacity)

        self._melody_osc = _OscillatorState()
        self._harmony_osc = _OscillatorState()
        self._accent_state = _AccentState()

        # Phase 5f.4a: True only while AudioController's scrub gesture is
        # active (set via set_scrub_active) -- gates Melody/Harmony
        # between the original always-on continuous-glide behavior
        # (scrub preview) and the new envelope-gated, silence-between-
        # notes behavior (committed/idle playback). Plain attribute,
        # same GIL-atomic-reassignment safety argument already
        # documented for `publish()` above -- set from the UI thread,
        # read from the audio thread, no lock.
        self._scrub_active = False

        self._melody_envelope = _ArticulationEnvelope(
            attack_seconds=melody_attack_seconds,
            plateau_seconds=melody_plateau_seconds,
            decay_seconds=melody_decay_seconds,
            floor=articulation_envelope_floor,
        )
        self._harmony_envelope = _ArticulationEnvelope(
            attack_seconds=harmony_attack_seconds,
            plateau_seconds=harmony_plateau_seconds,
            decay_seconds=harmony_decay_seconds,
            floor=articulation_envelope_floor,
        )

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
        """Retriggers the Melody/Harmony articulation envelopes (Phase
        5f.4a). Called once per committed move; never during scrub
        preview. A bare deque push, same thread-safety as push_event."""

        self._note_trigger_buffer.push(NoteTrigger())

    def set_scrub_active(self, active: bool) -> None:
        """
        Phase 5f.4a: True while a scrub gesture is in progress -- keeps
        Melody/Harmony on their original always-on, continuously-glided
        behavior for scrub preview. False (the default) uses the new
        envelope-gated, silence-between-notes behavior for ordinary
        committed-move playback. Bare attribute write; see this
        attribute's own docstring in __init__ for the thread-safety
        argument.
        """

        self._scrub_active = active

    # -- real-time callback (audio thread) ------------------------------

    def _render_block(self, outdata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        state = self._latest_state

        trigger = self._trigger_buffer.pop()
        if trigger is not None:
            self._accent_state.trigger(trigger.loudness)

        note_trigger = self._note_trigger_buffer.pop()
        if note_trigger is not None:
            self._melody_envelope.trigger()
            self._harmony_envelope.trigger()

        melody, melody_amplitude = self._render_melody(state, frames)
        harmony, harmony_amplitude = self._render_harmony(state, frames)
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
                # (Previously solo-mode ignored mute entirely for a
                # voice that was both muted and soloed; this is the
                # fix identified while wiring the Mixer UI to this
                # contract, not a Mixer-side workaround.)
                if mute.get(voice_id, False):
                    return False
                if any_solo:
                    return solo.get(voice_id, False)
                return True

            melody_audible = audible("melody")
            harmony_audible = audible("harmony")

            peak_bound = 0.0
            if melody_audible:
                sustained_buf += melody
                peak_bound += melody_amplitude
            if harmony_audible:
                sustained_buf += harmony
                peak_bound += harmony_amplitude

            # Sustained-bed headroom -- see this module's docstring.
            # Mirrors audio.renderer's `_normalize_to_peak(mix(melody,
            # harmony), SUSTAINED_PEAK_HEADROOM)`: scale DOWN only, never
            # up, and only the two continuous voices, never the accent.
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

        if state is None:
            target_frequency, target_amplitude, richness = 0.0, 0.0, 1
        else:
            target_frequency = state.pitch_hz
            target_amplitude = state.loudness
            richness = max(1, state.harmonic_richness)

        if self._scrub_active:
            # Scrub preview (Phase 5f.4, unchanged): one continuously
            # morphing voice -- both frequency and amplitude chase their
            # targets via the same one-pole smoother.
            frequency = osc.smoothed_frequency.update(target_frequency, self._smoothing_coefficient)
            amplitude = osc.smoothed_amplitude.update(target_amplitude, self._smoothing_coefficient)

            buf.fill(0.0)
            if frequency > 0.0 and amplitude > 0.0:
                t = self._arange[:frames] / self._sample_rate
                phase = osc.phase + 2 * np.pi * frequency * t
                weight_sum = 0.0
                for partial_index in range(1, richness + 1):
                    weight = 1.0 / partial_index
                    weight_sum += weight
                    buf += weight * np.sin(partial_index * phase)
                buf *= amplitude / weight_sum
        else:
            # Committed/idle mode (Phase 5f.4a): pitch snaps straight to
            # the target -- no glide, since the previous note has
            # already decayed toward silence by the time a normally
            # paced move arrives, so there is nothing to portamento
            # away from. Amplitude follows the articulation envelope,
            # not the one-pole smoother.
            frequency = target_frequency
            osc.smoothed_frequency.value = target_frequency

            envelope = self._scratch_melody_envelope[:frames]
            self._melody_envelope.render(frames, self._sample_rate, self._arange, envelope)

            buf.fill(0.0)
            if frequency > 0.0 and target_amplitude > 0.0:
                t = self._arange[:frames] / self._sample_rate
                phase = osc.phase + 2 * np.pi * frequency * t
                weight_sum = 0.0
                for partial_index in range(1, richness + 1):
                    weight = 1.0 / partial_index
                    weight_sum += weight
                    buf += weight * np.sin(partial_index * phase)
                buf *= (envelope * target_amplitude) / weight_sum

            amplitude = target_amplitude * (float(envelope.max()) if frames > 0 else 0.0)
            osc.smoothed_amplitude.value = amplitude

        osc.phase = (osc.phase + 2 * np.pi * frequency * frames / self._sample_rate) % (2 * np.pi)
        return buf, amplitude

    def _render_harmony(self, state: SonificationState | None, frames: int) -> tuple[np.ndarray, float]:
        osc = self._harmony_osc
        buf = self._scratch_harmony[:frames]

        if state is None:
            target_frequency, target_amplitude = 0.0, 0.0
        else:
            if state.harmony_above_melody:
                target_frequency = state.pitch_hz * state.harmony_interval_ratio
            else:
                target_frequency = state.pitch_hz / state.harmony_interval_ratio
            target_amplitude = state.loudness * HARMONY_VOICE_WEIGHT

        if self._scrub_active:
            frequency = osc.smoothed_frequency.update(target_frequency, self._smoothing_coefficient)
            amplitude = osc.smoothed_amplitude.update(target_amplitude, self._smoothing_coefficient)

            if frequency > 0.0 and amplitude > 0.0:
                t = self._arange[:frames] / self._sample_rate
                phase = osc.phase + 2 * np.pi * frequency * t
                np.sin(phase, out=buf)
                buf *= amplitude
            else:
                buf.fill(0.0)
        else:
            frequency = target_frequency
            osc.smoothed_frequency.value = target_frequency

            envelope = self._scratch_harmony_envelope[:frames]
            self._harmony_envelope.render(frames, self._sample_rate, self._arange, envelope)

            if frequency > 0.0 and target_amplitude > 0.0:
                t = self._arange[:frames] / self._sample_rate
                phase = osc.phase + 2 * np.pi * frequency * t
                np.sin(phase, out=buf)
                buf *= envelope * target_amplitude
            else:
                buf.fill(0.0)

            amplitude = target_amplitude * (float(envelope.max()) if frames > 0 else 0.0)
            osc.smoothed_amplitude.value = amplitude

        osc.phase = (osc.phase + 2 * np.pi * frequency * frames / self._sample_rate) % (2 * np.pi)
        return buf, amplitude

    def _render_accent(self, frames: int) -> np.ndarray:
        buf = self._scratch_accent[:frames]
        state = self._accent_state

        if not state.active:
            buf.fill(0.0)
            return buf

        t = self._arange[:frames] / self._sample_rate + state.elapsed_seconds
        decay = np.exp(-PERCUSSIVE_DECAY_RATE * t)
        np.sin(2 * np.pi * PERCUSSIVE_CLICK_FREQUENCY_HZ * t, out=buf)
        buf *= decay
        buf *= state.loudness

        state.elapsed_seconds += frames / self._sample_rate
        if decay[-1] < ACCENT_ENVELOPE_FLOOR:
            state.active = False

        return buf
