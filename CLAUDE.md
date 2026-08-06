# VectorChess

## Vision

VectorChess is not a chess engine.

It is not intended to compete with Stockfish or Leela.

The goal of the project is to transform a chess position into a living mathematical, geometric and musical system.

Every chess move should produce three synchronized layers:

1. Mathematical analysis
2. Scientific visualization
3. Generative music

The mathematics is the engine.

The visualization and music are the expression of the mathematics.

---

# Long-term Goal

Create a real-time interactive application where a chess game becomes

- geometry
- vector fields
- topology
- generative music
- scientific visualization

running simultaneously.

The final application should feel like an audiovisual mathematical instrument rather than a chess engine.

---

# Project Architecture

The project evolves in layers.

## Layer 1 — Chess

- python-chess
- legal move generation
- move history
- game state

---

## Layer 2 — Mathematical Representation

Current mathematical objects include:

- Mobility
- Attacker Count Field
- Attack Influence Field
- Attack Influence Surface
- Source Field
- Source Potential Field
- Gradient Field
- Equipotential Field

Future mathematical objects:

- Critical Points
- Hessian
- Ridge Analysis
- Valley Analysis
- Morse-Smale Complex
- Temporal Derivative (dF/dt)

This layer is the heart of the project.

---

## Layer 3 — Visualization

Visualizations never create information.

They only visualize existing mathematical objects.

Current visualizations:

- Board
- Attack Influence
- Equipotential
- Gradient
- Source Comparison

Future visualizations:

- Critical Points
- Ridge Network
- Morse Graph
- Force Network
- Temporal Animation

---

## Layer 4 — Audio

Status: Audio Layer MVP implemented — one deterministic clip per move, six signals (color, destination square, capture, check, Attack Influence balance, Dynamics label). See `docs/audio.md` for the sonification design and implementation status. Continuous, cross-move audio (Audio Layer 2) is future work.

Every mathematical object should eventually influence sound.

Examples:

Attack Influence
→ harmony

Gradient
→ melodic direction

Critical Points
→ musical accents

Capture
→ percussion

Check
→ tension

Checkmate
→ final cadence

Music should emerge from mathematics.

Never generate arbitrary sounds.

---

## Layer 5 — Interactive UI

The final application should contain

Left panel

• Chess board

Right panel

• Mathematical visualization

Bottom panel

• Musical state
• Position statistics
• Current field values

Everything updates in real time.

---

# Scientific Principles

Prefer mathematical correctness over visual effects.

Prefer explainability over complexity.

Every mathematical concept should have

- model
- builder
- visualization
- tests

Visualization must never invent data.

Every equation should be documented.

---

# Development Rules

One milestone = one mathematical concept.

Do not combine

- large refactors
- new mathematics
- UI redesign

inside the same milestone.

Each milestone should be independently testable.

Regression tests are mandatory.

---

# Coding Style

Prefer dataclasses.

Prefer immutable models when possible.

Keep mathematical functions pure.

Avoid duplicated coordinate transforms.

Document algorithms.

Explain equations.

---

# Terminology

Use consistent terminology.

Mobility

Unique reachable squares.

Attacker Count

Number of attacking pieces.

Attack Influence

Weighted attack field.

Source Field

Piece occupancy.

Source Potential

Continuous material field.

Gradient

Direction of maximum increase.

Equipotential

Lines of equal attack influence.

Critical Point

Gradient equals approximately zero.

---

# What This Project Is NOT

It is NOT

- another chess engine
- another Stockfish GUI
- another pygame chess clone

It is a research project exploring a new mathematical representation of chess positions and transforming that representation into geometry and music.

---

# Future Roadmap

Current

✔ Chess Engine

✔ Mathematical Fields

✔ Static Scientific Visualizations

✔ Audio Layer MVP

Next

□ Audio Layer 2

□ Event System

□ Interactive UI

□ Critical Point Detection

□ Hessian Analysis

□ Ridge Analysis

□ Morse-Smale Complex

□ Real-time Animation

□ Generative Music Engine

□ Machine Learning Features

---

# Project Philosophy

Technology serves curiosity.

Mathematics exists to reveal hidden structure.

Visualization exists to make that structure visible.

Music exists to make that structure audible.

Never introduce complexity unless it reveals something that was previously impossible to perceive.

---

# Final Objective

The final result should allow a user to play chess while simultaneously watching the mathematical structure of the position evolve and hearing that structure transformed into generative music in real time.