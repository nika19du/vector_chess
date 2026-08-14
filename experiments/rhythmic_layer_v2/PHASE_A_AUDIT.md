# Phase A — Audit of the Rhythmic Layer MVP ("Pulse")

Verified against the current code, not assumed. Every claim below cites the exact file/mechanism.

## Where each piece lives

| Question | Answer |
|---|---|
| 0.55s period source | `audio/pulse_pattern.py:29` — `PULSE_PERIOD_SECONDS = 0.55`, a fixed module constant, never derived from the move. |
| Pulse pitch/timbre | `audio/organic_synthesis.py` — `PULSE_BASE_FREQUENCY_HZ = 440.0` (fixed) and `PULSE_MODAL_PARAMS`/`PULSE_MODES` (fixed 2-mode harmonic preset). Neither varies with the move, position, or Melody's own pitch. |
| k-of-n density | `audio/pulse_pattern.py::pulse_pattern_for_density` → `active_slot_count_for_density` (density × 8 slots, rounded) → `euclidean_pulse_pattern` (Bresenham-bucket placement across those 8 slots). |
| Pulse phase/state storage | `AudioEngine.__init__`: `self._pulse_sample_counter = 0` (an integer sample clock) and `self._pulse_armed = False`. |
| SonificationState carries | `pulse_density: float`, `pulse_period_seconds: float` (`audio/live_state.py`) — both move/segment-identity-derived, set by `sonification_state_from_mapping`/`interpolate_sonification_state`. |
| Committed-move trigger | `AudioEngine._render_block`, inside the `note_trigger is not None` branch: `self._pulse_sample_counter = 0; self._pulse_armed = True`. |
| Scrub freeze/resume | `set_scrub_active(True)` sets `_pulse_armed = False` immediately; re-arming happens **only** on the next NoteTrigger, never merely because scrub ends. |
| Mute/solo | `audible("pulse")`, same closure Melody/Harmony/Accent already use. |
| Headroom | Pulse joins Melody+Harmony's `peak_bound`/`sustained_buf` headroom-managed mix, not Accent's raw-after-headroom path. |
| Tests locking this in | `tests/test_audio_pulse_pattern.py` (32), `tests/test_audio_engine_pulse_voice.py` (20), plus additions in `test_desktop_app_audio_controller.py`/`test_desktop_app_audio_scrub.py`. |

## The exact mechanism that produces the metronome

`AudioEngine._render_pulse` computes, every block, which slot is currently "playing":

```python
slot_index = (boundary // period_samples) % PULSE_SLOTS_PER_PHRASE
```

That `% PULSE_SLOTS_PER_PHRASE` is a **modulo wraparound** — it makes the 8-slot pattern loop forever. Once `_pulse_armed` becomes `True` on a committed move, there is **no condition anywhere that turns it back off** except another move or a scrub gesture. If the player just looks at the board for 30 seconds, the pattern keeps looping every `8 × 0.55s = 4.4s`, indefinitely, at the same fixed period, same fixed 440Hz pitch, same fixed timbre — literally a click track whose only variable is how many of the 8 slots are filled.

This confirms the user's diagnosis exactly. Three compounding causes, all verified above:

1. **No phrase boundary.** The "phrase" (`PULSE_SLOTS_PER_PHRASE = 8`) is not a finite unit — it's the loop period of an infinite clock.
2. **Pitch and timbre are constants, independent of the move.** A metronome's defining trait is that it doesn't respond to content — it just marks time. Pulse's fixed 440Hz/fixed-timbre design has that same property; density controls busyness, never identity.
3. **Architecturally modeled as a sustained voice, not a one-shot event.** Melody/Harmony are correctly "on until told otherwise" (they represent standing position state). Accent is correctly a bounded one-shot. Pulse was built like Melody/Harmony (continuously running, gated by `_pulse_armed`) when it should have been built like a *structured* one-shot — closer to Accent's model, but with internal structure instead of a single hit.

## What's worth keeping (per your instruction not to throw away working infrastructure)

- **The density mapping itself** (`Dynamics.intensity → pulse_density`, linear, clamped, tied to the existing "chaotic" threshold) — the *semantic* mapping was never the problem, only what "density" was applied to.
- **`euclidean_pulse_pattern`** — a genuinely useful, tested, deterministic placement primitive. Reused unmodified in two of the three candidates below.
- **The `_pulse_armed` / NoteTrigger-only-rearm gating** — this is exactly the right mechanism for "resume only on a real committed move," it just needs to gate a *finite* phrase instead of an infinite loop.
- **Mute/solo/headroom integration** — no change needed to the underlying mixer contract; only how the voice itself decides when to produce sound.
- **Determinism, sample-accurate scheduling, scrub-freeze contract** — all sound engineering, none of it is the cause of the metronome complaint, and both offline candidates below reuse the same "no RNG, pure function of density" discipline.

None of this needs to be thrown away — it needs to gate a *phrase generator*, not a *loop*.
