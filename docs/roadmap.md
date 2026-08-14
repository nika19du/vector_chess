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
(SessionState, Position Cache, Layer Registry, Voice Registry). Phase 5f (live audio) was
originally scoped to depend on Milestone 4b (Audio Layer 2); in practice it shipped
built entirely on the already-complete Milestone 4/4a MVP mappings and did not end up
needing 4b at all -- see docs/audio.md's "Live Audio Runtime" section and
docs/interactive_ui.md's implementation status note for the corrected record. Milestone
4b remains future work (richer audio content, layered onto the now-working runtime, not
a precondition for it). Phases: 5a rendering substrate (PySide6 + VisPy canvas; SessionState
with per-slice update signals; FEN-keyed Position Cache with asynchronous, background
cache-miss population), 5b interactive board (drag/drop, tree-aware undo/redo via
chess.pgn.GameNode, PGN load with variations), 5c Layer Registry and Voice Registry
stood up with all six field layers and five audio voices registered, 5d cross-move
correspondence matching (memoized) and move-to-move animation, 5e continuous timeline
scrubbing with branch switching, 5f live audio playback (sounddevice) at a 20ms latency
budget with finite per-voice articulation, scrub preview, and a mixer built against the
Voice Registry -- ✔ complete (5f.1-5f.6), 5g inspector panel, 5h
freeze/solo/compare interactions, 5i keyboard shortcuts and canvas zoom/pan, 5j offline
deterministic export (video via FFmpeg, MIDI via mido, WAV) isolated from live session
mutation via a snapshot taken at export start. Testing strategy (component tests,
concurrency stress test, cross-phase end-to-end test, standing performance regression)
in docs/interactive_ui.md Part 14.

... (5a-5f complete; 5g-5j remain)

Branch Exploration V1 (post-5e refinement, not a new lettered phase): the existing
`chess.pgn.GameNode` tree, per-branch `_active_child` redo memory, and Timeline branch
badges already gave 5e most of what branching needs (docs/interactive_ui.md Part 4.4);
V1 adds a compact "variation X/N" indicator with clamped ◀/▶ controls at the branch
point currently on screen and formalizes `redo()`'s existing most-recently-active-child
behavior as a stated contract -- no changes to chess logic, PositionCache, audio, or
transition animation (docs/interactive_ui.md Part 8).

Milestone 6

Critical Point Detection & Hessian Classification (merged from the originally separate "Critical Points" / "Hessian" entries -- classification via the second-derivative test is not a separable concept from detection). Phases: mathematical design note (docs/mathematics.md, Section 9), regression tests for AttackInfluenceSurface, Hessian evaluation, Newton localization, classification, post-classification quality filtering (analysis/critical_point_quality.py), visualization (visualization/critical_points_plot.py), console integration (critical_points_plot command).

✔

Milestone 7

Ridge / Valley Analysis. Phases: mathematical design note (docs/mathematics.md, Section 10), closed-form Hessian eigenvector derivation and eigenvector-following marching tracer (analysis/ridge_valley.py::locate_ridge_valley_chains), chain-level quality assessment (assess_ridge_valley_quality), visualization (visualization/ridge_valley_plot.py), console integration (ridge_valley_plot command).

✔

Milestone 8

Morse-Smale Complex. Phases: separatrix tracing (gradient-flow integral curves from quality-accepted saddles, analysis/morse_smale.py), cell assembly (half-edge/planar-face-tracing over closed separatrices, closed vs. open cell topology, structured TopologyIssue reporting), cell quality assessment (structured rejection-reason model, topology-invalid vs. topology-valid-but-low-quality distinction), visualization (visualization/morse_smale_plot.py), console integration (morse_smale_plot command).

✔