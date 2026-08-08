Milestone 1

Chess Engine

✔

Milestone 2

Mathematical Fields

✔

Milestone 3

Scientific Visualizations

✔

Milestone 4

Audio Layer MVP

✔

Milestone 4a

Audio MVP Maintenance (checkmate/game-end reachability, batch game audio export)

✔

Milestone 4b

Audio Layer 2

...

Milestone 5

Interactive UI. Full design in docs/interactive_ui.md (v3, architecture frozen -- ready
for implementation) -- no new mathematics, wires already-complete math (Milestones
1-4a, 6-8) to a real-time desktop application through a new presentation layer
(SessionState, Position Cache, Layer Registry, Voice Registry). Only phase 5f (live
audio) depends on Milestone 4b (Audio Layer 2); phases 5a-5e and 5g-5j proceed
independently of it. Phases: 5a rendering substrate (PySide6 + VisPy canvas; SessionState
with per-slice update signals; FEN-keyed Position Cache with asynchronous, background
cache-miss population), 5b interactive board (drag/drop, tree-aware undo/redo via
chess.pgn.GameNode, PGN load with variations), 5c Layer Registry and Voice Registry
stood up with all six field layers and five audio voices registered, 5d cross-move
correspondence matching (memoized) and move-to-move animation, 5e continuous timeline
scrubbing with branch switching, 5f live audio playback (sounddevice) at a 20ms latency
budget with a per-voice mixer built against the Voice Registry, 5g inspector panel, 5h
freeze/solo/compare interactions, 5i keyboard shortcuts and canvas zoom/pan, 5j offline
deterministic export (video via FFmpeg, MIDI via mido, WAV) isolated from live session
mutation via a snapshot taken at export start. Testing strategy (component tests,
concurrency stress test, cross-phase end-to-end test, standing performance regression)
in docs/interactive_ui.md Part 14.

...

Milestone 6

Critical Point Detection & Hessian Classification (merged from the originally separate "Critical Points" / "Hessian" entries -- classification via the second-derivative test is not a separable concept from detection). Phases: mathematical design note (docs/mathematics.md, Section 9), regression tests for AttackInfluenceSurface, Hessian evaluation, Newton localization, classification, post-classification quality filtering (analysis/critical_point_quality.py), visualization (visualization/critical_points_plot.py), console integration (critical_points_plot command).

✔

Milestone 7

Ridge / Valley Analysis. Phases: mathematical design note (docs/mathematics.md, Section 10), closed-form Hessian eigenvector derivation and eigenvector-following marching tracer (analysis/ridge_valley.py::locate_ridge_valley_chains), chain-level quality assessment (assess_ridge_valley_quality), visualization (visualization/ridge_valley_plot.py), console integration (ridge_valley_plot command).

✔

Milestone 8

Morse-Smale Complex. Phases: separatrix tracing (gradient-flow integral curves from quality-accepted saddles, analysis/morse_smale.py), cell assembly (half-edge/planar-face-tracing over closed separatrices, closed vs. open cell topology, structured TopologyIssue reporting), cell quality assessment (structured rejection-reason model, topology-invalid vs. topology-valid-but-low-quality distinction), visualization (visualization/morse_smale_plot.py), console integration (morse_smale_plot command).

✔