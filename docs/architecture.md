# What VectorChess Is

VectorChess turns a chess position into a family of mathematical fields, reconstructs the
ones worth turning into continuous geometry, and renders the result as real-time
visualization and generative audio — so a game can be watched and heard changing shape, move
by move. It has no engine and makes no claim about which move is best; its subject is the
shape and sound of a position as it is, not a verdict on what should happen to it next. See
`VECTORCHESS_MATHEMATICAL_MODEL_V2.md` (`experiments/geometric_move_prediction/`) for which
of its mathematical layers are established, which are exploratory, and which are best
understood as artistic interpretation.

---

# Pipeline

Chess Position

↓

Mobility

↓

Attacker Count

↓

Attack Influence

↓

Surface

↓

Gradient

↓

Hessian

↓

Critical Points (localization → classification → quality assessment)

↓

Ridge / Valley (tracing → quality assessment)

↓

Morse-Smale Complex (separatrix tracing → cell assembly → cell quality assessment)

↓

Visualization

↓

Audio

---

# Consumers

The pipeline above is unchanged by any of this. Visualization and Audio each have two
consumers, not one:

- **Static / offline** (Milestones 1–4a) — `console_app/main.py` (REPL) drives
  `visualization/*.py` (matplotlib) and `audio/export.py` (`.wav` files), one command at
  a time, one file at a time.
- **Live / interactive** (Milestone 5) — a desktop application. A presentation layer
  (SessionState, Position Cache, Layer Registry, Voice Registry — see
  `docs/interactive_ui.md`) sits between this pipeline's output and a real-time
  renderer/audio engine. It caches, tracks cross-move identity, and delivers the same
  data continuously instead of one shot at a time — it does not add to or change what
  Visualization or Audio consume from the pipeline above.

The live audio consumer (Phase 5f, complete) is `audio/engine.py::AudioEngine` — a
real-time `sounddevice`-backed callback, fed by `desktop_app/audio_controller.py::
AudioController`, the sole bridge from `SessionState`/scrub gestures to it. Both sit
entirely downstream of this pipeline's existing `Audio` box: `AudioController` calls the
same `audio/mapping.py::build_audio_mapping` the offline console path already uses,
never a second mapping. See `docs/audio.md`'s "Live Audio Runtime" section for the full
live-audio architecture (articulation, scrub preview, mixer).

Full design for the live consumer: `docs/interactive_ui.md` (architecture frozen, v3).