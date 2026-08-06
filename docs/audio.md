# VectorChess Musical Language

This document defines how chess mathematics should sound.

It is a composition and sonification design document, not an implementation plan.

No software architecture or code decisions are made here — see the Audio Layer MVP implementation plan for that. This document exists first, and the implementation must answer to it, not the other way around.

---

# 1. Musical Style

## Recommendation: Generative Ambient, built from Minimalist process technique.

Chess positions evolve continuously and slowly relative to any single move. Ambient's slowly-evolving textures are the natural match for continuous fields (Attack Influence Surface, Source Potential) that barely change from one move to the next.

Generative is not a style choice on top of this — it is a requirement already stated elsewhere in this project: *"Music should emerge from mathematics. Never generate arbitrary sounds."* A generative system is, by definition, rule-driven and non-narrative. That is the opposite of a composer imposing a mood.

Minimalist process technique (phasing, gradual addition or removal of voices, drones that shift through small parameter changes) is the right vocabulary for *audible change*, because that is literally what the mathematics does — Attack Influence, Gradient, and Dynamics all describe small, continuous shifts punctuated by occasional sharp events (captures, checks). Minimalism has spent decades building a language for exactly that shape of change.

## Explicitly rejected as the base style

**Algorave / dance-functional electronic music** — imposes a fixed beat grid. Chess time is irregular, driven by decisions, not a metronome. Forcing moves onto a beat grid would be the first arbitrary mapping in the system.

**Full cinematic scoring** — cinematic music exists to *illustrate* a story that already has a predetermined emotional arc. VectorChess has no predetermined arc; the emotional shape must come from the mathematics of the actual position. Cinematic scoring works backwards from this project's own rule that visualization (and by extension, audio) must never invent data.

**Electroacoustic / noise-based texture** — powerful for atmosphere, but its parameter-to-sound relationships are frequently opaque even to the composer. That conflicts directly with "prefer explainability over complexity."

Cinematic and electroacoustic vocabulary are not banned forever — they are reserved as **rare accent gestures** for extreme, structurally significant moments (checkmate, and later, critical points), never as the base texture.

---

# 2. Instrument Palette

## Base palette: pure oscillators + additive synthesis.

A pure sine oscillator is the most mathematically legible sound that exists — a sine wave literally *is* a single frequency. Starting from pure oscillators keeps the sound honest to the project's own preference for explainability over complexity, the same reason the math layer prefers simple, documented formulas over opaque ones.

Additive synthesis (building a tone by summing weighted harmonic partials) is not an arbitrary choice of synthesis method — it is a structural echo of how **Attack Influence** is already computed: a sum of piece weights over every attacker of a square. Summing weighted sine partials to build a tone is the same operation, one layer removed. This is the strongest possible justification available: the synthesis method *is* the mathematical method.

## White and Black

Both colors are built from the same palette — this is a distinguishability problem, not a "who is good" problem. Chess colors are not moral opposites; they are a symmetry that has to be broken audibly so a listener can tell them apart without looking at the board.

- **White**: pure sine/triangle tone — the simplest member of the palette.
- **Black**: the same tone plus an added sub-octave / denser harmonic blend.

This is deliberately the *smallest possible* audible distinction — one binary parameter (harmonic richness), not a wholesale timbral identity swap. A listener should be able to learn "thinner = White, denser = Black" the same way they'd learn any consistent code, the way White/Black pieces are simply rendered in two fixed colors on the board with no further meaning attached.

## Explicitly deferred

**FM synthesis** — frequency modulation introduces sidebands, i.e. inharmonic complexity from a simple control (the modulation index). That is a genuinely good analog for *danger* or *tactical distortion*, but there is no implemented mathematical object yet whose values should drive it responsibly. Reserved for the Generative Music Engine stage, once Critical Points / Hessian exist to justify *when* distortion should appear.

**Physical modelling (plucked string, bowed string, etc.)** — evokes a specific real-world instrument with its own cultural baggage (a guitar, a cello) that competes with the field's own identity rather than expressing it. Reserved, at most, for a single rare cadential gesture at checkmate — a moment where the system briefly steps out of pure field-sonification and marks the return to human, narrative time. Optional, not required for any milestone.

**Sampled/real acoustic instruments (piano, strings, orchestral sections)** — permanently rejected as a base identity. Real instrument timbres carry pre-existing emotional and cultural connotations (piano = human performance, strings = orchestral drama) that impose a narrative on top of the mathematics rather than letting the mathematics speak. This is the audio equivalent of a visualization "inventing data."

---

# 3. Musical Philosophy

## What the listener should hear, in order of priority

1. **Balance and Motion** (primary, always audible) — the ongoing state of the position: who currently holds more weighted influence, and which direction that balance is currently moving. This is the drone/texture layer, present continuously.
2. **Tension and Conflict** (secondary, event-triggered) — sharp, discrete moments: captures, checks, swings in `DynamicsAnalysis.intensity`. These emerge from real computed intensity, not manufactured for effect.
3. **Topology** (future, deepest layer) — once critical points, ridges, valleys, and the Morse–Smale complex exist, the *shape* of the field itself becomes audible: charged regions, calm regions, structural transitions. This is intentionally the last layer to arrive, because it is the last layer to exist mathematically.

**Beauty** is treated as a byproduct of correctness, not a design target of its own. If a mapping is mathematically right, it should also sound coherent; if making something sound "nicer" requires bending the mapping, the mapping is what needs revisiting, not the math.

The intended emotional experience is closer to **observing a system than being told a story**: contemplative, not dramatic. A quiet, balanced position should sound quiet and balanced. A chaotic middlegame should sound chaotic because `DynamicsAnalysis` says it is chaotic, not because a composer decided the moment needed excitement.

---

# 4. Mathematical Mapping Review

Every mathematical object currently implemented, reviewed for what — if anything — it should control. Silence is a valid, intentional answer.

## Mobility (`MobilitySummary`)

**Texture density / stereo width.** Mobility measures the breadth of available space, not a single value at a single point — width and density express "how much room there is" far better than pitch or harmony, which are single-value dimensions. More reachable squares → wider stereo spread and more textural layers. Target: **Layer 2** (needs a texture-layer engine the static MVP clip doesn't have).

## Attacker Count Field

**Silent, for audio purposes.** It is the unweighted twin of Attack Influence. Sonifying both would double-encode nearly the same information, which is exactly the kind of redundancy this project's own "avoid duplicated coordinate transforms" instinct argues against for math — it applies equally to sound. Attack Influence supersedes it. Kept open as a *possible* future "density of contact" texture, distinct from "value of contact," but not promised.

## Attack Influence Field

**Harmony** (primary), **dynamics/volume** (secondary). The signed `balance` value is the clearest "who currently dominates" signal in the whole system, and harmonic color (major/minor, consonant/dissonant) is the most direct musical analog for a signed dominance scalar that exists. Total influence magnitude additionally drives overall loudness — a quiet position genuinely has less attack influence in play.

## Attack Influence Surface

**Not an independent audible signal.** Its role is to make the *transition* between one position and the next continuous rather than a hard cut — a glide/portamento between move N and move N+1, justified precisely because the surface itself is a continuous interpolation. It enables smoothness elsewhere; it does not own a mapping category of its own. Target: **Layer 2**.

## Source Field

**Timbre / texture, background layer.** Source Field is about *what material exists on the board*, independent of attack geometry — a different question from Attack Influence's "who is currently active." It drives the harmonic thickness of a background drone/pad, sitting underneath the Attack-Influence-driven foreground harmony. Foreground = activity, background = material. Target: **Layer 2**.

## Source Potential

**Stereo / spatial position.** Its signature property — being blocking-independent, verified by regression test in this codebase — means it represents *latent* mass distribution in space, not line-of-sight geometry. That makes it the natural candidate for literal spatial placement (panning), since it already has a real (x, y) spatial structure that has nothing to do with directional attack lines. Target: **Layer 2**.

## Gradient Field

**Pitch motion / melodic direction.** This mapping is already named in the project's own architecture: *Gradient → melodic direction*. Concretely: the direction of ∇F across a sequence of moves shapes a melodic contour (rising gradient → rising line), and its magnitude shapes the size and speed of that movement (steep → large, fast intervals; shallow → small, slow steps). This requires a time axis across multiple moves, so it cannot exist in a single static clip. Target: **Layer 2**, first feature that requires true sequential/generative behavior rather than one-shot rendering.

## Equipotential

**Not an independent field** — it is a visualization technique (a contour extracted from the Surface), not its own mathematical object. Its audio analog is a discrete "boundary-crossing" event: a soft chime whenever a move causes the position to cross a named equipotential band. Because equipotential lines are thresholds, they belong in the eventful/percussive domain, not the continuous-parameter domain. Target: **Layer 2**, as an accent, not a primary mapping.

## Dynamics (`DynamicsAnalysis`)

**Rhythm** (primary), **dynamics/volume and density** (secondary). This is the one object that is inherently *about* change over time between two positions, so it belongs in the temporal domain rather than the pitch/harmony domain. `label` already drives volume/density in the MVP; `intensity` can additionally modulate rhythmic subdivision (calm → sparse, chaotic → dense) once a rhythmic layer exists. Target: **Layer 2** for the rhythmic extension.

---

## Future mathematical objects — deferred, and why

**Critical Points** — reserved for **musical accents**, matching the project's own stated mapping (*Critical Points → musical accents*). Sonifying a critical point before the detection math exists and is tested would itself be an arbitrary/fake mapping — the exact failure mode this whole document is designed to prevent. Silent until `docs/mathematics.md`'s "Critical Points" section has a working, tested builder.

**Hessian** — once available, refines *how* a critical-point accent sounds rather than owning an independent role: a sharp peak could ring as a bright, short strike; a broad saddle could sound as a more diffuse cluster. It modulates the timbre of an event that Critical Points already triggers.

**Ridge / Valley Analysis** — candidate for continuous **spatial texture and timbral brightness**: a ridge (a connected chain of local maxima) as a sustained bright filament moving through the stereo field; a valley as a low, dark trough. Genuinely topological, and therefore silent until the topology itself exists.

**Morse–Smale Complex** — the deepest, most global structural object in the roadmap. Once available, it is the best candidate for driving overall **musical form** — the topological decomposition of the whole field could define macro-sections of a longer generative piece, each cell of the complex becoming a distinct textural region. Reserved explicitly for the Generative Music Engine stage.

**Temporal Derivative (dF/dt)** — the continuous counterpart to what `DynamicsAnalysis` already approximates discretely between two moves. Once implemented, it should smoothly drive continuous parameters (tempo, filter cutoff) in place of the current discrete label buckets. A refinement of the existing Dynamics → rhythm mapping, not a new category.

---

# 5. Growth Strategy

```
Audio MVP
   ↓
Audio Layer 2
   ↓
Generative Music Engine
   ↓
Live Performance Mode
```

## Audio MVP

One short, deterministic clip per move. Six signals: color → timbre, destination square → pitch, capture → accent, check → dissonance, Attack Influence balance → harmony, Dynamics label → volume/density. Proves the pipeline exists and is legible — nothing more.

**Implemented.** See Section 7 for architecture, phased implementation history, and known limitations.

## Audio Layer 2

Moves from one clip per move to a **continuous, evolving texture** that persists across the whole game — a generative process that reacts to each move rather than restarting from silence every time. Introduces every mapping marked "Target: Layer 2" above: Mobility → texture/width, Source Field → background drone, Source Potential → spatial position, Attack Influence Surface → glide between positions, Gradient → melodic contour, Dynamics.intensity → rhythmic density, Equipotential crossings → accent chimes. Still fully deterministic and renderable offline (e.g. a whole game as one continuous file) — no real-time constraint yet.

## Generative Music Engine

Unlocked incrementally, one mathematical object at a time, as Critical Points, Hessian, Ridge/Valley Analysis, and the Morse–Smale Complex are implemented and tested. This is where FM synthesis (tension/distortion) and the reserved instrumental cadence gesture (checkmate) enter the palette, because by this stage there is finally mathematical justification (real detected critical points, real topology) for when they should appear. Multiple simultaneous voices and longer musical memory become possible here — but every new capability is still gated by a corresponding, tested math milestone, never added speculatively.

## Live Performance Mode

Real-time synthesis driven by an actual live or streamed game, tied to the separately-planned Real-time Event System milestone. This is a **delivery and infrastructure** milestone, not a new sonification design — it plays the same rules established in this document, live instead of pre-rendered, matching the project's ultimate vision of a real-time audiovisual instrument.

**Governing rule for the whole roadmap: audio never outruns math.** Each stage above is unlocked strictly by its corresponding entry in `docs/roadmap.md` and `docs/mathematics.md` already existing and being tested. No stage sonifies placeholder or fake data.

---

# 6. Design Principles

Rules every future audio milestone must follow.

1. **No arbitrary mappings.** Every parameter mapping must name the exact field or value driving it and state, in one sentence, why that pairing is the natural analog. If it can't be explained in one sentence without hand-waving, it doesn't ship.

2. **Audio never leads math.** A field earns an audio mapping only after it has a working, tested builder, per this project's own rule that every mathematical concept needs a model, builder, visualization, and tests. Audio is downstream of the math, never ahead of it.

3. **No redundant double-encoding.** Each mathematical object gets at most one primary sonic role. When two objects would express nearly the same information (Attacker Count vs. Attack Influence), only the more complete one is sonified.

4. **Determinism first, randomness never — until it's earned.** Same position, same sound, always, through the Generative Music Engine stage. Any later controlled randomness must be seeded and reproducible, layered on top of the deterministic mapping, never replacing it.

5. **Match the domain to the mathematics.** Continuous fields get continuous musical parameters (pitch, harmony, texture, timbre); discrete events get discrete musical parameters (rhythm, accents). Never force a continuous field into a rhythmic trigger, and never smear a discrete event into a slowly-modulated parameter.

6. **Explainability survives translation.** A listener — or a reader of this document — must always be able to trace any sound back to the exact field and formula that produced it, the same way visualization must never invent data.

7. **Emotional honesty over manufactured drama.** Tension, dissonance, and intensity must reflect real computed values (`DynamicsAnalysis.intensity`, `AttackInfluenceField.balance`, etc.). A quiet position must sound quiet, even if a louder moment would be more exciting to listen to.

8. **The vocabulary only grows, it never gets redefined.** Once a mapping is established (e.g. "ascending gradient = ascending melody"), later layers may add richness on top of it but must not contradict it. A listener's learned associations must stay valid across milestones.

9. **The test of success is legibility without visuals.** Every mapping should be evaluated against one question: could an attentive listener, without seeing the board, learn to recognize a capture, a check, a dominant side, and the difference between a calm and a tense position? If not, the mapping needs revisiting before it ships.

---

# 7. MVP Implementation Status

The Audio MVP described in Section 5 is implemented. This section records what exists, how it was built, and what is not yet resolved — the same discipline this document asks of every mapping applied to the implementation itself.

## Architecture

A new `audio/` package, sibling to `analysis/` and `visualization/`:

- `audio/models.py` — `AudioMapping`, `AudioClip`, `RenderConfig` (frozen dataclasses).
- `audio/mapping.py` — `build_audio_mapping(analysis, dynamics)`: translates `MoveAnalysis` + optional `DynamicsAnalysis` into the six named signals below. Pure, deterministic, no sound generated.
- `audio/synthesis.py` — chess-agnostic DSP primitives (`sine_wave`, `additive_tone`, `apply_envelope`, `percussive_burst`, `mix`). No knowledge of chess at all.
- `audio/renderer.py` — `AudioRenderer.render(mapping)`: turns an `AudioMapping` into an `AudioClip` using only the primitives above.
- `audio/export.py` — `write_wav(clip, path)`: serializes to mono 16-bit PCM via the standard-library `wave` module. No new dependency was needed anywhere in this pipeline — `numpy` (already used by the math layer) covers synthesis, stdlib `wave` covers export.
- `console_app/export.py`'s `export_move_audio` and the `audio` REPL command in `console_app/main.py` wire the pipeline into the existing console application, reusing the already-computed `MoveAnalysis`/`DynamicsAnalysis` rather than re-analyzing the position.

Move → `MoveAnalysis` → `AudioMapping` → `AudioRenderer` → `AudioClip` → `write_wav` → `.wav`, exactly as specified in the approved implementation plan.

## Phases

Built and reviewed in five independently-tested phases, each gated on the previous one's approval:

1. **Foundations** — `audio/models.py` + `audio/synthesis.py`. No chess dependency; the DSP substrate Audio Layer 2 will also reuse.
2. **Mapping** — `audio/mapping.py`. All six signals implemented, each traceable in one sentence to a named `MoveAnalysis`/`DynamicsAnalysis` field; harmony dissonance intensity is symmetric in the sign of `balance` (direction is a renderer concern), check imposes a dissonance floor rather than replacing balance-driven harmony.
3. **Renderer** — `audio/renderer.py`. Surfaced and fixed a real defect: a single global peak-normalization pass could make a capturing move's opening window *quieter* than a non-capturing one, since the accent's own contribution could raise the divisor more than it raised the local energy. Fixed by normalizing the sustained melody+harmony bed to a fixed headroom independently of whether an accent exists, then hard-clipping (not rescaling) as the final `[-1.0, 1.0]` safety bound.
4. **Export** — `audio/export.py`. Quantization to int16 happens only at this boundary; added explicit `ValueError`s for an empty clip or non-positive sample rate, since this is the last stop before the filesystem and not guaranteed to only ever receive renderer-produced input once an Event System or UI exists.
5. **Console Integration** — verified (no production code changes were needed) that the `audio` command correctly reuses `previous_analysis`/`dynamics_history[-1]`, never re-triggers analysis, and writes deterministic, non-colliding filenames under `output/audio/`.

## Known limitation: loudness compression

`AudioRenderer`'s fixed sustained-bed headroom (`SUSTAINED_PEAK_HEADROOM`, added in Phase 3 to fix the normalization defect above) caps the melody+harmony mix at a constant peak regardless of `AudioMapping.loudness`. A mapping with low loudness (e.g. a neutral first move) may pass through this cap unscaled, while a high-loudness mapping (e.g. a `"chaotic"`-labeled move) gets pulled back down to the same cap — compressing away part of the difference Signal 6 (Dynamics label → volume/density) is meant to produce.

Measured on a real game: a neutral first move and a `"chaotic"`-labeled move with a 1.9× difference in the underlying `loudness` parameter produced only a ~7% difference in RMS energy. This is a real, measured tension between the Phase 3 correctness fix and the Phase 2 loudness signal, not a bug in either phase individually.

**Accepted as a known MVP limitation, not a blocker.** Not scheduled for change until a deliberate follow-up milestone — see the Growth Strategy in Section 5 and the project's `docs/roadmap.md`.

## Subjective validation is still open

Everything above was verified objectively: unit and integration tests, FFT spectral-peak comparisons (confirming Black's richer timbre shows a clean harmonic series a White clip doesn't), RMS energy measurements, and direct inspection of `AudioMapping` field values. None of this establishes principle 9 from Section 6 — *"could an attentive listener, without seeing the board, learn to recognize a capture, a check, a dominant side, and the difference between a calm and a tense position?"* — because verifying that requires actually listening, which the implementing process has no capability to do.

**A human listener must perform this validation before the MVP's legibility claim (Section 6, principle 9) is considered confirmed**, not just its correctness. Representative clips (a White move, a Black move, and a capturing check) were generated from a real game and handed off for this purpose; the result of that listening pass is not yet recorded here.

---

# Philosophy

The project already transforms

Chess

↓

Mathematics

↓

Geometry

↓

Visualization

↓

Music

This document adds one more line, underneath and governing all of Music:

Music

↓

**A language a listener can learn to trust — because every sound it makes, it can also justify.**
