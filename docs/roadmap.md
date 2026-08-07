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

Interactive UI

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