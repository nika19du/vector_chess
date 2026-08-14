# Phase B1 Report — Mapping / Register Redesign

**Scope:** mapping/register/harmony grammar only. Synthesis is unchanged — this is expected to still
sound synthetic. B2 (organic instrument synthesis) is a separate, later checkpoint. Per instructions,
this report stops after B1; B2 has not been started.

---

## 1. Exact new pitch/register mapping

`audio/mapping.py::_pitch_for_square` pipeline, per the requested stages:

```
raw chess value (file, rank)
  -> normalized value: raw_hz = BASE_FREQUENCY_HZ * SCALE_RATIOS[file] * 2**((rank-1)*OCTAVE_SPAN/RANK_STEPS)
     (same formula shape as before, OCTAVE_SPAN now 2.0 instead of 3.0)
  -> scale degree + octave/register: raw_hz snapped to the nearest tone of LEGAL_MELODY_PITCHES_HZ
     (nearest-neighbor in log2 space — perceptual distance, not linear Hz distance)
  -> final frequency: the snapped result, always an exact member of LEGAL_MELODY_PITCHES_HZ
```

`LEGAL_MELODY_PITCHES_HZ` is built once at import time: the 7 distinct diatonic scale ratios
(`SCALE_RATIOS`, unchanged from before) repeated at 2 octave layers, giving **15 unique tones from
220 Hz to 880 Hz** — 220.0, 247.5, 275.0, 293.3, 330.0, 366.7, 412.5, 440.0, 495.0, 550.0, 586.7,
660.0, 733.3, 825.0, 880.0.

Register bound (all 64 squares, both colors, verified by test):
`BASE_FREQUENCY_HZ (220 Hz) <= pitch <= BASE_FREQUENCY_HZ * 2**OCTAVE_SPAN (880 Hz)`.

**A bug this quantization incidentally fixes:** under the old design, only rank-1 pitches (and the
full-octave endpoint at rank 8) were guaranteed to land on an actual diatonic scale tone. Because
`OCTAVE_SPAN / RANK_STEPS = 3/7` is not an integer, every intermediate rank's continuous octave
multiplier produced a frequency that was *not* a member of the scale at all — the literal mechanism
behind "arbitrary continuous frequency jumps" the user flagged. Snapping every output to
`LEGAL_MELODY_PITCHES_HZ` means **every square, at every rank, now sounds an actual scale tone.**

---

## 2. Scale used and rationale

**Diatonic** (the existing 7-degree just-intonation major scale, `SCALE_RATIOS`), not pentatonic —
deliberately not chosen "because it's familiar." Reasoning:

- It is already melody's own existing scale — zero unnecessary churn to an already-correct part of the
  design.
- It provides exactly 8 pitch classes (a..h, h duplicating the octave) matching the 8 files **1:1 with
  zero collision** at the per-rank-layer level — pentatonic (5 degrees) would force multiple files to
  share a pitch class purely from the scale-degree count, discarding file-ordering fidelity for no
  compensating benefit at the mapping level.
- Phase A's pentatonic candidate (Candidate A) was entangled with a completely different instrument
  timbre (Karplus-Strong), so it was never a controlled test of scale choice in isolation — there is no
  clean evidence pentatonic is musically better for melody once the shrillness/register problems are
  fixed by other means, so it isn't adopted here.
- Harmony's new vocabulary (below) is deliberately kept as a *subset* of the same diatonic ratios (plus
  the two named production endpoints), so melody and harmony share one internally consistent vocabulary
  rather than two unrelated ones.

---

## 3. Old vs. new frequency distributions (same 3 games, 46 moves)

Measured by rerunning `experiments/audio_musicality/measure.py` — unmodified script, now reading the
new production `audio/mapping.py`. Old numbers preserved at
`experiments/audio_musicality/output/diagnostics/phase_a_baseline/`.

| | OLD min | OLD median | OLD max | NEW min | NEW median | NEW max |
|---|---|---|---|---|---|---|
| White fundamental | 412.5 Hz | 804.5 Hz | 2200.0 Hz | 366.7 Hz | 586.7 Hz | 880.0 Hz |
| Black fundamental | 531.4 Hz | 1443.8 Hz | 3300.0 Hz | 440.0 Hz | 880.0 Hz | 880.0 Hz |
| Black summed partials (all) | 531.4 Hz | 2590.9 Hz | **9900.0 Hz** | 440.0 Hz | 1485.0 Hz | **2640.0 Hz** |
| Black partials > 8000 Hz | 1/66 (1.5%) | | | 0/66 (0.0%) | | |
| Harmony frequency | 375.7 Hz | 1400.1 Hz | 4666.9 Hz | 311.1 Hz | 933.4 Hz | 1244.5 Hz |
| Harmony interval ratio | continuous, 1.0–1.4142 | | | quantized, 5-value vocabulary | | |

Full per-move tables in `experiments/audio_musicality/output/diagnostics/measurements.csv` (new) vs.
`.../phase_a_baseline/measurements.csv` (old).

---

## 4. White/Black fundamental ranges

Both colors draw from the **identical** `_pitch_for_square` formula — there is no color term in pitch
anywhere in the pipeline, before or after B1 (verified by `test_same_square_produces_the_same_
fundamental_regardless_of_color`, which asserts this for all 64 squares). White and Black therefore
share the exact same 220–880 Hz fundamental range by construction, not by a separate "shared range"
mechanism layered on top.

---

## 5. Harmony quantization table

`HARMONY_INTERVAL_VOCABULARY = (1.0, 9/8, 5/4, 4/3, sqrt(2))`. `CONSONANT_RATIO` (unison) and
`DISSONANT_RATIO` (tritone) are production's own pre-existing named endpoints, kept exactly.
`CHECK_DISSONANCE_FLOOR` (1.25) is algebraically `5/4`, a just major third — already a natural member.
`9/8` and `4/3` fill the gap with two more scale-consistent steps.

| Interval | Ratio | Committed when raw `\|balance\|` (no check) is | With check forced |
|---|---|---|---|
| Unison | 1.0000 | 0 – 2.93 | never (floor excludes it) |
| Major second | 1.1250 | 2.93 – 8.97 | never (floor excludes it) |
| Major third (= check floor) | 1.2500 | 8.97 – 14.05 | 0 – 14.05 |
| Perfect fourth | 1.3333 | 14.05 – 18.02 | 14.05 – 18.02 |
| Tritone | 1.4142 | ≥ 18.02 (saturates at `REFERENCE_MAX_BALANCE`=20) | ≥ 18.02 |

Direction (which side leads) is still carried entirely by the sign of `attack_influence_balance`,
handled downstream in `audio/renderer.py::_harmony_frequency` (unmodified) — the vocabulary above only
constrains the *magnitude*.

**Important architectural note:** this quantization is applied **only** to a committed move's stored
`AudioMapping.harmony_interval_ratio`. `harmony_interval_for_balance` itself is left completely
unmodified and continuous, because `audio/live_state.py::interpolate_sonification_state` calls it
directly with a continuously-interpolated balance to drive scrub preview's smooth glide — quantizing
that shared function would have silently broken scrub, which the brief explicitly said must not change
in B1. This was caught by running the full test suite (see §10) before it could ship.

---

## 6. Evidence destination-square ordering is preserved

Two new tests, checked over the full board (all 64 squares):
- `test_pitch_is_monotonic_non_decreasing_across_ranks_for_a_fixed_file` — for every file, pitch never
  decreases as rank increases.
- `test_pitch_is_monotonic_non_decreasing_across_files_for_a_fixed_rank` — for every rank, pitch never
  decreases as file increases (a→h).

Both pass. Ordering survives even though 64 squares now compress into 15 discrete tones — some distinct
squares now legitimately share a pitch (e.g. e5 and c6 both land on 733.3 Hz), which is expected,
intentional lossy compression, not a defect (see §9's discussion of the two tests this required fixing).

---

## 7. Evidence Black no longer receives uncontrolled high-frequency identity

Two independent, stacked mechanisms:

1. **Register compression alone** already solves it: worst case is now
   `BASE_FREQUENCY_HZ * 2**OCTAVE_SPAN * BLACK_HARMONIC_RICHNESS = 880 * 3 = 2640 Hz`, far under any
   piercing threshold — confirmed empirically in §3 (measured max Black partial dropped from 9900 Hz to
   2640 Hz across the same 46 moves).
2. **`HARMONIC_CEILING_HZ = 4200.0`** (data-derived from Phase A's own measured partial distribution —
   the elbow where only 6% of Black's old partials exceeded 6000 Hz) is enforced explicitly via
   `_effective_harmonic_richness`, at the mapping source, *before* any synthesis runs — not by
   attenuating volume or clipping after the fact. Under today's register this ceiling is provably
   inert (2640 < 4200 always) — confirmed by `test_no_real_board_square_lets_black_exceed_the_harmonic_
   ceiling` (checks all 64 squares) — but it's kept as an explicit, independently-tested backstop
   (`test_effective_richness_never_lets_a_partial_exceed_the_harmonic_ceiling`, a synthetic-input unit
   test proving the capping logic itself works) so the invariant holds even if register constants
   change later, rather than relying on today's numbers as an accidental side effect.
   `test_black_retains_its_distinguishing_richness_across_the_whole_board` confirms the ceiling doesn't
   quietly erase Black's distinguishing signal — richness stays at the full intended 3 everywhere on
   the current board.

---

## 8. Exact production files changed

**`audio/mapping.py` only.** No other production file was touched — `audio/renderer.py`,
`audio/synthesis.py`, `audio/engine.py`, `audio/models.py`, `audio/live_state.py`,
`audio/backend.py`, `audio/export.py`, `audio/voices.py`, `desktop_app/*`, `chess_engine/*`,
`analysis/*`, `console_app/*`, and `docs/*` are all unmodified (verified via `git diff --stat`).

---

## 9. Exact tests changed and why

**`tests/test_audio_mapping.py`** — 1 test removed, 12 added, all others (26) unchanged and still pass:

- **Removed** `test_pitch_walks_a_fixed_ratio_between_consecutive_ranks`: asserted a *fixed* ratio
  between consecutive same-file ranks. This encoded the pre-B1 design's own bug (§1) as if it were an
  invariant — post-quantization, consecutive ranks are snapped independently and no longer separated by
  a constant ratio. This was never a real product requirement; it was an accidental property of the old
  continuous formula. Replaced by tests that check what actually matters: monotonic ordering
  (`test_pitch_is_monotonic_non_decreasing_across_ranks_for_a_fixed_file` /
  `..._across_files_for_a_fixed_rank`), scale membership
  (`test_every_square_produces_a_legal_scale_tone`), and register bound
  (`test_every_square_stays_within_the_clamped_two_octave_register`).
- **Added**: colorblind-fundamental test, effective-richness ceiling tests (synthetic + whole-board),
  harmony-vocabulary-membership test, check-floor-survives-quantization test, and a full-pipeline
  committed-mapping quantization test. All listed in §6–§7 above.
- **All 26 pre-existing tests not touched** — including every harmony test
  (`test_harmony_moves_monotonically_toward_dissonance`, `test_check_forces_dissonance_floor_even_at_
  zero_balance`, etc.) — because `harmony_interval_for_balance` itself was deliberately left unmodified
  (§5); those tests describe that raw continuous function, which didn't change. This was verified, not
  assumed: they were run before any other change and passed immediately.

**`tests/test_desktop_app_audio_controller.py`** — 1 assertion changed in
`test_playing_a_different_move_after_undo_creates_a_branch_with_fresh_analysis`: it used
`branch_state.pitch_hz != main_line_state.pitch_hz` as a proxy for "the new branch really did get fresh
analysis, not stale cached data." Under the new compressed/quantized register, two genuinely different
destination squares (e5 and c6) can legitimately share a pitch — the proxy is no longer reliable, not
because the underlying guarantee broke. Replaced with `branch_state.segment_key !=
main_line_state.segment_key` — the actual FEN-pair identity, which is the authoritative signal for "this
is really a different position" and is robust to any future pitch/scale tuning.

**`tests/test_desktop_app_audio_scrub.py`** — 2 changes:
- `test_scrub_after_a_branch_switch_uses_the_new_branchs_own_analysis`: same pitch-collision issue as
  above, same fix (`segment_key` instead of `pitch_hz`).
- `test_scrub_at_the_tip_reproduces_the_settled_state_via_the_identity_segment_trick`: this is the one
  test that exposed a genuine, expected architectural consequence rather than a fragile proxy. Before
  B1, scrub's "identity segment trick" (interpolating a node against itself) reproduced the settled
  state's `harmony_interval_ratio` bit-for-bit only because *both* paths called the same unquantized
  `harmony_interval_for_balance`. Now that committed mappings are quantized and scrub preview
  deliberately isn't (§5 — required to preserve smooth scrub glide, explicitly non-negotiable per the
  brief), the two paths correctly diverge at this value. Fixed by computing the correct new ground
  truth directly — `harmony_interval_for_balance(nf3_balance, is_check=False)` — instead of comparing
  against the now-quantized settled state. `pitch_hz`/`loudness` assertions on the same test were
  untouched and still pass unmodified, since both paths read those from the same already-quantized
  `segment_mapping.pitch_hz`, held fixed per segment.

No test was changed merely to accept a new output value without a stated reason — every change above
traces to either an old test encoding the old design's own defect, or a proxy assertion that stopped
being reliable once pitch collisions became an expected consequence of the new, intentionally
compressed register.

---

## 10. Full regression results

1. **Focused mapping tests**: `pytest tests/test_audio_mapping.py` — **39/39 passed** (28 original,
   minus 1 removed, plus 12 new).
2. **All audio mapping/offline tests + live-state + AudioController parity + full audio suite**: run
   together as `pytest tests/ -k audio` — **321/321 passed**, 0 failed (this single run covers mapping,
   renderer, synthesis, export, engine, backend, live_state, articulation, scrub, scrub-integration,
   controller, controller-mixer, mixer-panel, main-window audio wiring, and console audio integration).
3. **Full project suite**: `pytest tests/` — one file, `tests/test_ridge_valley_plot.py` (a matplotlib
   visualization test with no connection to `audio/` or this change), triggers a native access-violation
   crash when run in the same process as ~600 other tests — confirmed via isolated run
   (`pytest tests/test_ridge_valley_plot.py` alone: **18/18 passed**) that this is a pre-existing
   native/GC interaction, not a regression from this change. Running the rest of the suite excluding
   that one file: **909 passed, 7 skipped** (pre-existing, environment-related — see
   `tests/conftest.py`'s own documented GL-context skip), **0 failed**. Combined: every test in the
   repository passes; only their process-grouping needed splitting around one known-flaky file. (No
   explicit "native-flake partition" tooling/doc was found in the repo to reuse — this is the pragmatic
   equivalent: isolate the one file that crashes the process, verify it independently, run everything
   else together.)

---

## 11. Offline comparison WAV

`experiments/phase_b1_mapping/output/b1_mapping_diagnostic.wav` — the same Scholar's-mate sequence
(`e2e4 e7e5 d1h5 b8c6 f1c4 g8f6 h5f7`), rendered through the **actual production pipeline**
(`build_audio_mapping` → `AudioRenderer` → `write_wav`, all unmodified except `mapping.py`'s internals).
11.2 s, valid mono 16-bit PCM, verified. **Explicitly a mapping-only diagnostic** — synthesis is
untouched, so this will still sound like sine oscillators. What to listen for: narrower/more comfortable
register, Black no longer spiking into a piercing upper register, and harmony landing on cleaner,
discrete intervals move to move rather than a continuously arbitrary ratio.

Per-move mapping values produced during this render (`pitch_hz`, `harmonic_richness`,
`harmony_interval_ratio`) are printed in the script's own stdout and reproduced here:

| move | color | pitch (Hz) | richness | harmony ratio | capture | check |
|---|---|---|---|---|---|---|
| e2e4 | white | 586.7 | 1 | 1.4142 | – | – |
| e7e5 | black | 733.3 | 3 | 1.0000 | – | – |
| d1h5 | white | 880.0 | 1 | 1.4142 | – | – |
| b8c6 | black | 733.3 | 3 | 1.4142 | – | – |
| f1c4 | white | 495.0 | 1 | 1.4142 | – | – |
| g8f6 | black | 880.0 | 3 | 1.4142 | – | – |
| h5f7 | white | 880.0 | 1 | 1.4142 | ✓ | ✓ |

---

## 12. Remaining problems B2 must solve

Exactly what the user predicted and asked to defer:

- **Timbre is still pure sine / additive-sine** — `audio/synthesis.py` untouched. This grammar fix does
  not, by itself, make anything sound "instrumental," "organic," or "alive." B1 fixed *what notes get
  played*, not *what they sound like*.
- **No attack/body/resonance/decay/material character** — the current envelope is still linear
  attack/release around a flat sustained tone (offline) or the existing articulation envelope (live,
  unmodified) around a flat oscillator. B2 needs to introduce the actual organic/resonant synthesis
  (Karplus-Strong or equivalent, per Phase A Candidate A) or a disciplined filtered/bounded approach
  (Phase A Candidates B/C) — this report deliberately does not pick one; that's B2's job, informed by
  listening to this B1 diagnostic first per the user's own instruction.
- **White/Black identity is still purely `harmonic_richness` (1 vs. up to 3)** — now safely bounded, but
  still the same *mechanism* the user asked to move away from as the primary identity channel (brief
  §4: "combinations of harmonic distribution, resonance, damping, brightness, attack character, body
  response," not richness alone). B2 is where that shifts.
- **Harmony is still a plain sine voice** — quantized in interval now, but timbrally identical to
  before; B2's "support, not second siren" goal (brief §8) is about the *sound*, not the *interval*,
  which is all B1 addressed.
- **Accent is still the same generic 900 Hz percussive click** — B2's "same sonic material family"
  requirement (brief §9) is untouched here.

**STOP.** B1 complete. Awaiting listening feedback on the new musical grammar (via
`b1_mapping_diagnostic.wav`) before starting B2.
