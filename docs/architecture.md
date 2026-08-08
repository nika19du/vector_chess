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

Full design for the live consumer: `docs/interactive_ui.md` (architecture frozen, v3).