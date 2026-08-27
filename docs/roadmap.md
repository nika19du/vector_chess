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

Reconstruction-stability addendum (post-Experiment 007, see VECTORCHESS_MATHEMATICAL_MODEL_V2.md
in experiments/geometric_move_prediction/): Milestones 6-8's own quality-assessment phases
(post-classification/chain/cell quality filtering) already separate trustworthy detections from
noise at fixed reconstruction settings. Experiment 007 additionally measured stability *across*
reasonable reconstruction settings and found it is not uniform: dominant critical points hold up
comparatively well, ridge/valley chains inherit their anchor's stability, and the Morse-Smale
complex is the least stable of the three -- now treated as an exploratory/educational/artistic
layer (docs/mathematics.md Section 11) rather than a validated chess topology. No change to any
of the three milestones' code, detectors, or tests.

Model v2 Integration Audit lettered milestones (VECTORCHESS_MODEL_V2_INTEGRATION_AUDIT.md
Sec. 19, in experiments/geometric_move_prediction/): a small set of desktop-app-only
follow-ups proposed after the Integration Audit -- terminology/documentation alignment
(A), layer metadata (B), critical-point prominence tiering (C), Morse-Smale re-framing
(D), an educational pipeline walk-through (E, not yet started), and Source/Occupancy
desktop integration (F, below). None of these touch chess/mathematical logic; each is a
UI/documentation-only or pure-integration change to the already-complete Milestone 5
desktop app.

Milestone F

Source/Occupancy Desktop Integration (VECTORCHESS_MODEL_V2_INTEGRATION_AUDIT.md Sec. 19,
7, 15). Wires the existing, already-tested `analysis/source_field.py` /
`analysis/source_potential.py` (Model v2's second, non-attack-derived chess observable --
piece occupancy/material, ρ) into the desktop `LayerRegistry` as a seventh, optional
layer (`desktop_app/layers/source_potential_layer.py`, id `source_potential`, category
`Field`), closing the "only one observable is wired in" gap the Integration Audit
flagged. Samples the continuous Φ_source reconstruction (Gaussian kernel, production
defaults) at the 64 board-square centers via `analysis.source_potential.
evaluate_source_potential`; `FullPositionAnalysis`/`PositionCache` cache only the cheap
discrete `SourceField`, not the full 200x200 surface. Renders through the existing
generic `LayerGeometry` path, never Attack Influence's singleton overlay buffer.
Defaults OFF; absent from Overview/Influence/Flow/Topology, present only in "All
Layers". No transition/scrub interpolation added (follows the existing
Equipotential/Gradient settle-refresh precedent). No new mathematics, no change to
Attack Influence/Gradient/Critical Points/Ridge-Valley/Morse-Smale (still derived from
Attack Influence only), no audio change, Compare Mode not extended (see
docs/interactive_ui.md Part 5 for the full implementation note).

✔