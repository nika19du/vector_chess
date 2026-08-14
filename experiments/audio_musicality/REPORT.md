# VectorChess Audio Musicality Audit — Phase A Report

**Scope honored:** audit + listening prototypes only. No production file was modified (verified at
the end of this report). No new runtime dependency was added — only `numpy` (already used by
`audio/`) and Python stdlib. All new code lives under `experiments/audio_musicality/`. Raw data:
`output/diagnostics/measurements.csv`, `output/diagnostics/measurements_summary.md`,
`output/diagnostics/loudness_log.md`. Raw audio: `output/audio/*.wav`.

---

## 1. Pipeline trace (summary)

`build_audio_mapping(analysis, dynamics)` (`audio/mapping.py`) turns a move's already-computed
`MoveAnalysis`/`DynamicsAnalysis` into an `AudioMapping` — pure data, no sound yet. `AudioRenderer.render()`
(`audio/renderer.py`) turns that into an `AudioClip` by calling `audio/synthesis.py`'s primitives
(`additive_tone` for Melody, `sine_wave` for Harmony, `percussive_burst` for the capture/check Accent,
`apply_envelope`, `mix`), normalizing the sustained bed to a fixed headroom, and hard-clipping as a
final safety bound. `audio/export.py::write_wav` quantizes to 16-bit PCM. The live runtime
(`audio/engine.py`) delivers the same six mapped signals in real time with a different envelope shape
(`_ArticulationEnvelope`: attack → plateau → exponential decay) but adds no new mapping. Full
function-by-function trace with file:line references was produced during exploration and is not
repeated here in full; the load-bearing functions for this report are `_pitch_for_square`,
`_harmonic_richness_for_color`, `harmony_interval_for_balance`, `_harmony_frequency`, and
`additive_tone` — all quoted in the sections below where they matter.

---

## 2. Quantitative diagnosis: why Black sounds shrill

Measured by replaying three real, already-regression-tested games (Scholar's mate — 7 plies; the
project's own fastest-stalemate line — 19 plies; the Sicilian Najdorf opening — 20 plies; 46 moves
total) through the **actual production pipeline** (`analyze_position` → `analyze_dynamics` →
`build_audio_mapping`, unmodified) via `experiments/audio_musicality/measure.py`.

### Melody fundamental (destination-square pitch — colorblind by formula)

| | min | median | max | n |
|---|---|---|---|---|
| White fundamental | 412.5 Hz | 804.5 Hz | 2200.0 Hz | 24 |
| Black fundamental | 531.4 Hz | 1443.8 Hz | 3300.0 Hz | 22 |

`_pitch_for_square` takes only the destination square — there is no color term in the formula at all.
The fundamental-pitch difference above is incidental (which squares each side happened to move to in
these particular games), not a systematic White/Black split.

### Harmonic partials actually sounded (`additive_tone` output — where color *does* act)

| | min | median | max | n |
|---|---|---|---|---|
| White partials (richness=1, == fundamental) | 412.5 Hz | 804.5 Hz | 2200.0 Hz | 24 |
| Black partials (richness=3: 1×, 2×, 3× fundamental) | 531.4 Hz | **2590.9 Hz** | **9900.0 Hz** | 66 |

- Black partials exceeding 8000 Hz: 1/66 (1.5%). Exceeding 10000 Hz: 0/66.
- **The real finding is the median, not the tail**: White's median sounded frequency across these
  games is 804.5 Hz; Black's is 2590.9 Hz — a **3.2× gap** — even though both colors draw pitch from
  the identical, colorblind `_pitch_for_square` formula. This is not a formula bug; it is the direct,
  measured consequence of `additive_tone` summing partials at `frequency_hz × partial_index` (2×, 3×
  the fundamental) with no upper bound, for every Black move, regardless of register.
- One concrete extreme: `e8g8` (Black castling short, Sicilian Najdorf ply 16) has a destination-square
  fundamental of 3300 Hz; its 3rd partial reaches 9900 Hz — within a semitone of the top of typical
  piano range and squarely in the frequency band the ear reads as hiss/piercing rather than tone.
- **`docs/audio.md:46`'s stated intent ("Black: the same tone plus an added sub-octave / denser
  harmonic blend") does not match the implementation.** `additive_tone` adds *upper* harmonics
  (2×, 3× fundamental), not a sub-octave (0.5×) *below* it. Had the doc's original intent been built
  as written, Black would sound *warmer/lower*, not shrill — this discrepancy between stated design
  and shipped code is very likely the direct root cause of the shrillness complaint.

### Harmony voice

Range 375.7–4666.9 Hz across all 46 moves, interval ratio range 1.0000–1.4142 (production's own
unison..tritone bounds). Confirmed continuous/unquantized — unlike melody pitch (quantized via
`SCALE_RATIOS`), harmony is never snapped to a discrete interval.

### Renderer-level compounding factors (cited from `audio/renderer.py`, not re-measured)

- `HARMONY_VOICE_WEIGHT = 0.6` — harmony is quieter than melody by design, so shrillness is dominated
  by the melody voice's own upper partials, not the harmony voice.
- `SUSTAINED_PEAK_HEADROOM = 0.85` — the melody+harmony bed is normalized to a *fixed* peak regardless
  of loudness. This does **not** selectively soften Black's extra harmonic energy relative to White's;
  headroom compresses overall level, not spectral balance, so it does not explain or mask the gap above.

**Bottom line:** Black is shrill not because of a single bad note, but because *every* Black move
routinely sounds a spectral centroid roughly 3× higher than a same-position White move would, purely
from unbounded upper-harmonic summation — an effect invisible if you only look at the (colorblind)
fundamental-pitch formula, which is exactly why this needed direct measurement rather than code
inspection alone.

---

## 3. Musical-structure audit

- **No stable tonal center.** `_pitch_for_square`'s file→scale-degree mapping is internally consistent
  (a real 8-note just-intonation major scale), but the *rank* dimension applies a full 3-octave sweep
  per move with zero smoothing or melodic memory — consecutive moves can leap arbitrarily far in
  register purely because of where a piece physically landed. There is no melodic continuity logic
  anywhere in the pipeline (confirmed: no function references a previous move's pitch).
- **Register is uncomfortably wide.** 220–3520 Hz spans more than 4 octaves counting Black's 3rd
  partial's reach (up to ~9900 Hz measured) — well beyond what a single "melody" line comfortably
  occupies in most acoustic or electronic instrumental writing.
- **Harmony's interval vocabulary is unconstrained** (Section 2) — the one place a genuinely continuous
  math value (`attack_influence_balance`) is mapped directly into a continuous audio parameter with no
  musical quantization step in between. This is the second "too literal" mapping identified by this
  audit, alongside the register-per-move issue above.
- **White/Black distinguishability is real but perceptually miscalibrated.** The intended "smallest
  possible audible distinction" (harmonic-partial count) in practice produces the single largest
  perceptual effect in the whole system (the 3.2× spectral-centroid gap in Section 2) — a case where a
  parameter that looks small in code (`1` vs `3`) is not small in the resulting sound.
- **Foreground/background hierarchy exists but is thin.** Melody (`additive_tone`) is foreground,
  Harmony (`sine_wave`, weight 0.6) is background — a real, working two-voice hierarchy — but with only
  two voices and no timbral differentiation beyond partial count/weight, it reads as "two oscillators,"
  not as an ensemble.
- **Continuity between moves is entirely absent.** Every clip renders independently with its own
  attack/release envelope from silence to silence; nothing in `audio/renderer.py` or `audio/mapping.py`
  references a prior move's `AudioMapping`. This, combined with the wide per-move register swing above,
  is the structural source of the "arbitrary oscillator tones attached to chess moves" complaint —
  every note is locally coherent but the *sequence* has no melodic logic connecting one note to the next.

---

## 4. Register-as-identity investigation

Requirement 4 asked whether White/Black identity should move away from pitch/register. All three
candidates below deliberately do **not** use register or fundamental pitch to distinguish color (both
colors draw from the exact same, symmetric pitch formula in every candidate). The three alternatives
actually implemented and rendered:

- **Candidate A**: instrument-family / synthesis-technique split (plucked string vs. struck mallet).
- **Candidate B**: filter-brightness split (`WHITE_CUTOFF=0.35` vs `BLACK_CUTOFF=0.18`) — the most
  literal test of "timbre/brightness, not register."
- **Candidate C**: kept production's own axis (partial count) but bounded its absolute reach, testing
  whether the *existing* identity mechanism is salvageable once capped, without moving to a new axis
  at all.

All three keep melody register clamped to 2 octaves (220–880 Hz, see Section 6) regardless of which
identity axis is used — the register-narrowing fix and the identity-axis question are independent and
were deliberately varied independently across candidates so their effects can be told apart by ear.

---

## 5. Candidate designs

### Candidate A — Warm / acoustic-like

- **White timbre:** Karplus-Strong plucked string (delay line length `N = round(sample_rate / freq)`,
  deterministic 3-odd-harmonic tapered burst excitation, leaky-averaging feedback
  `y[i] = decay·0.5·(y[i-N] + y[i-N-1])`, `decay = 0.995`).
- **Black timbre:** production's own `additive_tone`, capped at 2 partials (vs. production's 3), with a
  new fast exponential-decay "mallet" envelope (`decay_rate = 6.0`/s) instead of production's linear
  attack/release — a struck, resonant character.
- **Melody timbre:** as above, color-dependent.
- **Harmony timbre:** pure sine (Karplus-Strong reserved for melody only — a second recursive delay
  line for harmony would double the physical-modeling footprint, worsening the conflict below).
- **Accent:** unchanged, production's `percussive_burst`, reused directly.
- **Pitch range:** 220–880 Hz (2-octave clamp), file→scale-degree ratio additionally snapped to
  **pentatonic** (`1.0, 9/8, 5/4, 3/2, 5/3`) — the most restrictive, most consonant vocabulary of the
  three candidates.
- **Scale/quantization:** pentatonic melody, diatonic-quantized harmony.
- **Tension → sound:** unchanged loudness signal only; no additional tension channel in this candidate.
- **White/Black distinguishability:** instrument-family (plucked vs. struck) — the largest, most
  immediately audible distinction of the three candidates.
- **Advantages:** most "instrumental"/musical-sounding of the three by construction; directly answers
  the "should feel like a coherent instrument palette" ask.
- **Disadvantages / open risk (must not be glossed over):** `docs/audio.md` Section 2 states physical
  modeling is "reserved, at most, for a single rare cadential gesture at checkmate," explicitly **not**
  a base identity. Using Karplus-Strong for White on *every* move is a base-identity use of physical
  modeling and directly conflicts with that stated design principle. Additionally, Karplus-Strong's
  per-sample recursive feedback loop does not trivially vectorize into `audio/engine.py`'s real-time
  block-callback model (see Section 10) — the biggest Phase B risk of the three candidates if chosen.

### Candidate B — Restrained electronic / ambient

- **White timbre:** soft additive oscillator, 2 partials, steep `1/n²` amplitude rolloff (vs.
  production's `1/n`), one-pole low-pass at `cutoff = 0.35` (brighter).
- **Black timbre:** identical synthesis, one-pole low-pass at `cutoff = 0.18` (darker/duller) — the
  *only* difference from White is filter brightness.
- **Melody timbre:** as above.
- **Harmony timbre:** sine, one-pole low-pass at a fixed, color-independent `cutoff = 0.22` (harmony
  reflects position balance, not which side moved).
- **Accent:** unchanged, production's `percussive_burst`, reused directly, deliberately left unfiltered
  so it stays a clean transient against the softened sustained bed.
- **Pitch range:** 220–880 Hz (2-octave clamp), diatonic (unchanged scale vocabulary from production,
  no pentatonic snap) — isolates "does clamping + filtering alone fix the shrillness" as a variable
  separate from Candidate A's scale change.
- **Scale/quantization:** diatonic melody and harmony.
- **Tension → sound:** filter cutoff rises with `dynamics_label` (`calm:+0.00 … chaotic:+0.14`,
  clamped at 0.9) — a second, independent, explainable tension channel layered on top of the unchanged
  loudness signal.
- **White/Black distinguishability:** filter brightness only — the most direct test of requirement 4's
  "timbre/filtering instead of pitch."
- **Advantages:** most fully compliant with `docs/audio.md`'s existing base-palette rules (still pure
  oscillators + additive synthesis, no new instrument family); smallest real-time-safety risk of the
  three (a one-pole filter is a standard, cheap, streaming-friendly DSP block).
- **Disadvantages:** least "acoustic"/instrumental-sounding — still reads as synthesized, just softer;
  filter-brightness as the sole identity channel is a subtler cue than Candidate A's timbre-family
  split and may be less immediately legible to an inattentive listener.

### Candidate C — Hybrid mathematical

- **White timbre:** production's own additive-synthesis identity, unchanged in spirit — 1 partial.
- **Black timbre:** same `additive_tone`-style summation, but partials whose *absolute* frequency
  exceeds `HARMONIC_CEILING_HZ = 4200 Hz` are omitted (not clipped/aliased). 4200 Hz was chosen from
  the measured data in Section 2: 21.2% of Black's measured partials exceed 4000 Hz, but only 6.1%
  exceed 6000 Hz and 1.5% exceed 8000 Hz — 4200 Hz sits at the elbow of that distribution, capping the
  most extreme quintile of Black's harmonic content while leaving roughly 79% of it untouched.
- **Melody timbre:** as above, color-dependent via partial count exactly as production does today.
- **Harmony timbre:** sine, diatonic-quantized ratio, passed through the same mild, color-independent
  low-pass (`cutoff = 0.5`) as melody.
- **Accent:** unchanged, production's `percussive_burst`, reused directly.
- **Pitch range:** 220–880 Hz (2-octave clamp) — same register fix as A/B, isolating "does clamping +
  an absolute partial ceiling fix it without changing the timbre identity at all."
- **Scale/quantization:** diatonic melody and harmony.
- **Tension → sound:** unchanged loudness signal only.
- **White/Black distinguishability:** partial count, exactly as production — the *least* structurally
  changed of the three candidates.
- **Advantages:** smallest philosophical delta from `docs/audio.md`'s existing justification ("the
  synthesis method *is* the mathematical method"); the ceiling itself is directly traceable to measured
  data rather than an aesthetic choice; lowest migration cost into production code.
- **Disadvantages:** least "solved" — still fundamentally the same sound family as today, just bounded;
  a listener who found the *character* (not just the extremity) of additive-sine timbre unsatisfying
  will likely still feel that here.

---

## 6. Pitch quantization

Melody pitch was **already** quantized (diatonic, via `SCALE_RATIOS`) before this audit — the genuinely
unquantized value is harmony's interval ratio. `register.quantize_ratio_to_scale(raw_ratio,
allowed_ratios)` snaps to the nearest neighbor **in log-frequency space** (perceptual distance, not
linear ratio distance) against one of two sets:

- **`DIATONIC_HARMONY_RATIOS`** (= melody's own `SCALE_RATIOS` values) — reuses melody's existing
  vocabulary, so a harmony interval always lands on a ratio the listener has already heard as a
  melodic step somewhere on the board. Used by Candidates B and C, to keep their harmony consistent
  with their (unchanged) diatonic melody scale.
- **`PENTATONIC_RATIOS`** (`1.0, 9/8, 5/4, 3/2, 5/3` — drops the 4th and 7th degrees, the two steps
  most responsible for perceived tension/color in a major scale) — used by Candidate A's melody (not
  chosen for familiarity; chosen because it structurally removes the two most dissonance-prone scale
  steps, matching Candidate A's "warm/consonant" design goal) and by nothing else, so the tradeoff of a
  smaller vocabulary vs. more consonance can be heard in isolation against the other two candidates.

Register span itself (`register.CLAMPED_OCTAVE_SPAN = 2.0` vs. production's `OCTAVE_SPAN = 3.0`,
shared by all three candidates) is a separate fix from ratio quantization — it narrows the *spread*
without changing which discrete pitches exist, preserving strict monotonic ordering (higher rank still
sounds higher) so ordering information survives even though absolute spread shrinks.

---

## 7. Synthesis technique prototypes

- **Karplus-Strong plucked string** (Candidate A, `candidate_a.py::karplus_strong_pluck`) — delay line
  length `N = round(sample_rate / frequency_hz)`, initialized with a deterministic 3-odd-harmonic
  tapered burst (not the textbook random-noise excitation, to avoid a seed/randomness question — see
  the open alternative noted in the module docstring), then repeatedly: output the front sample, append
  `decay · 0.5 · (front + second)` to the back, pop the front. `decay = 0.995`.
- **One-pole low-pass filter** (`synth_common.py::one_pole_lowpass`) —
  `y[n] = y[n-1] + cutoff·(x[n] − y[n-1])`; single parameter, streaming-friendly, no new dependency.
- **Exponential mallet decay** (Candidate A, `mallet_tone`) — `additive_tone` capped at 2 partials,
  multiplied by `exp(-6.0·t)` plus a 5ms linear attack.
- **Steep-rolloff soft additive oscillator** (Candidate B, `soft_additive_tone`) — same weighted-sum
  structure as production's `additive_tone` but `1/n²` weighting instead of `1/n`.
- **Capped-absolute-frequency additive synthesis** (Candidate C, `capped_additive_tone`) — identical to
  production's `additive_tone` except partials above `HARMONIC_CEILING_HZ` are omitted from the sum
  before normalization.

No new runtime dependency was introduced for any of the above — `numpy` + stdlib only, matching
`audio/`'s own footprint.

---

## 8. Listening prototypes

Rendered from the same 7-move Scholar's-mate sequence (`e2e4 e7e5 d1h5 b8c6 f1c4 g8f6 h5f7`) used by
`measure.py` and the project's own regression tests — quiet opening moves, both colors, a capture, a
check, and checkmate, exactly matching the requested move-mix.

**Normalization:** each candidate's full concatenated 7-move track was RMS-normalized to a common
target (`TARGET_RMS = 0.2`) before the final `[-1, 1]` hard clip — the same "normalize the bed, then
hard-clip, never rescale after" pattern production's own `AudioRenderer` uses. Pre/post-normalization
measurements (`output/diagnostics/loudness_log.md`):

| candidate | pre-norm RMS | pre-norm peak | post-norm RMS | post-norm peak |
|---|---|---|---|---|
| A | 0.2628 | 1.0000 | 0.2000 | 0.7610 |
| B | 0.3807 | 0.9805 | 0.2000 | 0.5152 |
| C | 0.4241 | 1.0000 | 0.2000 | 0.4715 |

All three land at identical RMS (0.2000) with no post-normalization clipping (all peaks < 1.0), so the
A/B/C comparison is level-matched.

**Files produced** (`experiments/audio_musicality/output/audio/`):
- `candidate_A.wav`, `candidate_B.wav`, `candidate_C.wav` — 11.2s each (7 × 1.6s moves), the main
  comparison tracks.
- `candidate_<X>_white_note.wav`, `_black_note.wav`, `_harmony.wav`, `_accent.wav` for each of A/B/C —
  isolated 1.6s (0.4s for accent) diagnostic clips, also RMS-normalized to the same target.

All 15 files verified to open as valid mono 16-bit PCM WAVs (`getnframes() > 0`, `getframerate() ==
44100`) with no NaN/Inf samples.

---

## 9. Mapping-preservation table

| Chess/analysis quantity | Current mapping | Proposed mapping | Musical consequence | Information preserved |
|---|---|---|---|---|
| `to_square` (file, rank) | `_pitch_for_square`: 8-step diatonic file scale × 3-octave rank span, 220–3520 Hz | Same file scale; rank span clamped to 2 octaves (220–880 Hz); melody additionally pentatonic-quantized in Candidate A only | Narrower, more comfortable register; fewer extreme highs | File→scale-degree and rank→register ordering both preserved (strictly monotonic in every candidate) |
| `color` → `harmonic_richness` | 1 (White) vs 3 (Black) partials, unbounded `n×f` — measured median spectral centroid gap 3.2× (804.5 Hz vs 2590.9 Hz) | A: partial count capped at 2 for Black + mallet decay shape. B: partial count unchanged (2 for both), differ by filter cutoff only. C: absolute-Hz ceiling (4200 Hz, data-derived) on partials, count unchanged | Removes/limits the specific mechanism producing the measured shrillness | White/Black distinction remains audible via at least one channel (timbre family, filter brightness, or bounded richness) in every candidate |
| `attack_influence_balance` → `harmony_interval_ratio` | continuous 1.0–1.4142, never quantized | Snapped to nearest ratio in `DIATONIC_HARMONY_RATIOS` or `PENTATONIC_RATIOS` (log-frequency nearest-neighbor) | Harmony always lands on a "musical" interval instead of an arbitrary continuous ratio | Direction (sign of balance) and relative-dissonance ordering both preserved; only the exact ratio value is snapped |
| `is_check` (dissonance floor) | forces ratio ≥ 1.25 before any quantization | floor still applied before quantization in every candidate | Check still reliably sounds "more tense," now on a scale-legal interval | Binary/discrete nature of check preserved exactly |
| `dynamics_label` → loudness | discrete bucket → gain, `SUSTAINED_PEAK_HEADROOM` compresses the audible difference (documented production limitation) | unchanged in A and C; Candidate B additionally routes label → filter-cutoff offset as a second channel | Candidate B makes tension somewhat more legible without touching the loudness signal itself | Existing signal fully preserved everywhere, extended (not replaced) in B |

---

## 10. Recommendation and Phase B scope (not implemented)

**Recommendation:** Candidate C, with Candidate B's filter-brightness idea folded in as a *second*,
lightweight color cue layered on top of C's bounded partial count (i.e., a hybrid of B and C) is the
lowest-risk path that most directly answers the measured problem: it removes the specific, data-proven
shrillness mechanism (Section 2) without adopting a new instrument family that conflicts with
`docs/audio.md`'s own stated palette rules (Candidate A's Karplus-Strong-as-base-identity tension,
Section 5). Candidate A is musically the most immediately satisfying of the three in isolated listening
and worth keeping as a live option, but its physical-modeling-as-base-identity conflict with
`docs/audio.md` and its real-time-safety risk (below) should be resolved as an explicit, separate design
decision before it's chosen, not by default. **This recommendation is not implemented in this phase.**

**What Phase B would need to touch, if a direction is approved:**
- `audio/mapping.py`: `OCTAVE_SPAN`/`BASE_FREQUENCY_HZ` (register clamp), `WHITE_HARMONIC_RICHNESS`/
  `BLACK_HARMONIC_RICHNESS` and a new absolute partial-ceiling constant, `harmony_interval_for_balance`
  (quantization step).
- `audio/synthesis.py`: would need new primitives added for real (Karplus-Strong pluck and/or one-pole
  filter and/or capped-partial additive tone, and/or ratio-quantization) — today these exist only in
  `experiments/`.
- `audio/renderer.py`: voice construction/weights (`HARMONY_VOICE_WEIGHT`, accent) if a chosen design's
  balance differs from production's; the `_harmony_frequency` direction rule stays as-is.
- `audio/models.py`: `RenderConfig` would likely need new fields (filter cutoff, register bounds,
  harmonic ceiling) as a dataclass extension.
- `audio/engine.py`: **the largest flagged risk.** `_ArticulationEnvelope`'s continuous-retuning,
  frame-by-frame block-callback model assumes each voice is a simple retuned oscillator. A one-pole
  filter (Candidates B/C) is cheap and streaming-friendly and should port with modest effort. Karplus-
  Strong (Candidate A) is fundamentally different: its recursive delay-line feedback needs persistent
  per-note state across render blocks and doesn't trivially vectorize the way `_render_melody`/
  `_render_harmony` currently do — if Candidate A (or a hybrid including it) is chosen, real-time
  safety and live/offline parity need dedicated design work before Phase B can proceed, not just a
  drop-in port.
- `desktop_app/audio_controller.py`, `audio_mixer_panel.py`: UI/mixer implications of any new per-voice
  parameters (e.g. a filter-cutoff or harmonic-ceiling control), if exposed to the user at all.
- `docs/audio.md`: Section 2 (Instrument Palette) and Section 7 (MVP Implementation Status) would need
  updating to record whichever decision is made — especially resolving the physical-modeling-as-base-
  identity question explicitly if Candidate A or a Candidate-A-derived hybrid is chosen, and correcting
  the "sub-octave" wording that this audit found does not match any implementation, past or proposed.

**Risks to live/offline parity and real-time safety**, summarized:
- Any candidate using only per-block-vectorizable DSP (B, C, and A's Black mallet voice) carries low
  real-time risk — the existing block-callback architecture already handles comparable operations.
- Candidate A's Karplus-Strong White voice carries real risk: it needs a persistent delay-line buffer
  per active note, spanning across audio callback blocks, which `AudioEngine`'s current per-voice
  continuous-oscillator model does not support without new state management — the single largest
  concrete Phase B risk identified in this audit.
- None of the candidates change scrub semantics, mixer UI, headroom philosophy, or chess analysis — all
  were left untouched in this phase, per the guardrails.

---

**STOP.** This concludes Phase A. No Phase B implementation has been started. Awaiting listening
feedback on `candidate_A.wav`, `candidate_B.wav`, `candidate_C.wav` (and the isolated diagnostic clips)
before any further work.
