# Audio Layer 2 -- Rhythmic Layer: Listening Deliverable

Rendered by `render_ab.py`, isolated from production (same convention as every
prior `experiments/` phase). Uses the real live `AudioEngine`/`AudioController`
pipeline (`FakeAudioBackend` swapped in for hardware), never a second synthesis
path.

## Files

- `ab_no_pulse.wav` / `ab_with_pulse.wav` -- the same 7-move tactical sequence
  (1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6?? 4.Qxf7+), once with the Pulse voice off
  (`pulse_density` forced to 0.0, everything else identical) and once on.
  Peak-normalized to the same level (0.9) so the comparison isn't confounded
  by differing overall loudness -- the music's own internal dynamics are
  untouched, only the whole clip's level is matched.
- `calm.wav` / `active.wav` / `tense.wav` / `chaotic.wav` -- the same real
  position and move, paired with a synthetic `DynamicsAnalysis` at an
  intensity squarely inside each of `analysis.dynamics`'s own four label
  bands (calm<5, active<12, tense<20, chaotic>=20). Isolates the
  `Dynamics.intensity -> pulse_density` mapping on its own, without a real
  game's incidental variation confounding it.

## Measured mapping (this run)

| Example | intensity | pulse_density |
|---|---|---|
| calm | 2.0 | 0.100 |
| active | 8.0 | 0.400 |
| tense | 16.0 | 0.800 |
| chaotic | 25.0 | 1.000 |

Monotonic and linear by construction (`audio/pulse_pattern.py::pulse_density_for_intensity`).

## The listening question

Not "does `ab_with_pulse.wav` have more sounds than `ab_no_pulse.wav`."

**"Can you hear increasing positional intensity as increasing rhythmic
density, across `calm.wav` -> `active.wav` -> `tense.wav` -> `chaotic.wav`,
without it turning into generic dance music?"**

Each clip is 4 seconds (calm/active/tense/chaotic) or ~2.5s/move (the A/B
pair) at a fixed, narrow pulse period (0.55s) -- deliberately no BPM sweep, so
tempo is never a confounding variable; density (how many of 8 slots per
phrase are active) is the only thing changing.

## Known limitation

This is a synthetic offline harness built for this listening pass -- it is
not wired into `console_app`'s production export pipeline (`audio/export.py`
still only knows the six-signal MVP mapping). If Pulse is approved after this
listening pass, wiring it into the production offline export path is a
follow-up, not part of this MVP.
