# VectorChess Mathematics

This document describes every mathematical object used in the project.

The purpose is not only to document formulas,
but to explain the intuition behind every concept.

---

# 1. Mobility

## Meaning

Mobility measures how many unique squares can be reached by each side.

It ignores piece values.

## Purpose

Represents available space.

---

# 2. Attacker Count Field

## Meaning

For every square

count how many pieces attack it.

Example

e4

White attackers = 3

Black attackers = 2

Difference = +1

## Purpose

Represents attack density.

---

# 3. Attack Influence Field

## Meaning

Each attacker contributes according to its weight.

Pawn = 1

Knight = 3

Bishop = 3.25

Rook = 5

Queen = 9

King = infinite (or configurable)

The resulting field describes weighted influence.

---

# 4. Source Field

## Meaning

Represents piece occupancy.

Every occupied square becomes a source.

White pieces

positive values.

Black pieces

negative values.

Empty squares

zero.

---

# 5. Source Potential

## Meaning

Transforms the discrete Source Field into a continuous surface.

A kernel distributes every source across nearby space.

Current kernels

Gaussian

Softened inverse distance

---

# 6. Attack Influence Surface

## Meaning

Continuous reconstruction of the Attack Influence Field.

Purpose

Allows differential analysis.

---

# 7. Gradient

## Meaning

Gradient shows the direction of maximum increase.

Mathematically

∇φ

Intuition

Imagine a mountain.

Gradient tells you

which direction is uphill.

---

# 8. Equipotential

## Meaning

Equipotential lines connect points having equal field value.

They are equivalent to contour lines on topographic maps.

---

# 9. Critical Points and the Hessian

## First principles

In one variable, a point `x₀` is a critical point of `f(x)` if `f'(x₀) = 0` — the tangent is flat. The sign of `f''(x₀)` then classifies it: positive curves upward (a minimum), negative curves downward (a maximum), zero is inconclusive.

In two variables, `AttackInfluenceSurface` is a function `φ(x, y)` over the board. A critical point is where both partial slopes vanish at once — the gradient itself is the zero vector:

```
∇φ(x₀, y₀) = (∂φ/∂x, ∂φ/∂y) = (0, 0)
```

This matches this document's own Terminology: Critical Point = "Gradient equals approximately zero."

Classifying the point needs more than one number, because curvature in two dimensions can differ in every direction (along files, along ranks, along every diagonal between). The object holding all of that at once is the **Hessian**, the matrix of second partial derivatives:

```
H(x, y) = [ φ_xx   φ_xy ]
          [ φ_xy   φ_yy ]
```

`φ_xx` is curvature along the file direction, `φ_yy` along the rank direction, and `φ_xy` (= `φ_yx`) is the "twist" term — how the x-slope changes as you move in y.

## Intuition

Picture `AttackInfluenceSurface` as terrain: White's influence pushes the ground up, Black's pushes it down. Standing exactly on a critical point, the ground feels level in every direction — but "level here" is compatible with three different landscapes:

- **Hilltop** — every direction goes downhill → **maximum**. Both Hessian eigenvalues negative.
- **Basin** — every direction goes uphill → **minimum**. Both eigenvalues positive.
- **Mountain pass** — uphill toward two peaks, downhill toward two valleys → **saddle**. Eigenvalues of opposite sign.

The classic **second-derivative test** reads this off two scalars without computing eigenvalues explicitly:

```
D = det(H) = φ_xx · φ_yy − (φ_xy)²
```

| Condition | Type |
|---|---|
| `D > 0`, `φ_xx < 0` | Maximum |
| `D > 0`, `φ_xx > 0` | Minimum |
| `D < 0` | Saddle |
| `D ≈ 0` | Degenerate / indeterminate — reported honestly, never guessed |

## Relation to chess

`AttackInfluenceSurface`'s value at a point is `white_attack_influence − black_attack_influence`, smoothed and fitted: positive means White dominates, negative means Black does.

- **A maximum** is a genuine local concentration of White dominance — every nearby direction is *less* White-dominant. This is a sharper claim than "high value": a broad plateau of uniformly high influence has no maximum on it at all, because nothing there peaks relative to its neighbors. Typically the convergence point of several attackers' lines. A minimum is the mirror image for Black.
- **A saddle** marks a contested corridor between two *different* concentrations — e.g. the file or diagonal connecting a White strongpoint and a Black strongpoint runs directly across it. It expresses tension *between* two hotspots, not a hotspot itself.

Critical points are the first mathematical object in this document that answers a question about local *shape* rather than local *magnitude*.

## Why now, and how it builds on Attack Influence Surface and Gradient

`analysis/attack_influence_surface.py` already fits a bicubic spline (`kx=ky=3`) over the Attack Influence field — its own docstring names critical points as a reason the surface exists. Bicubic splines support second derivatives (`dx=2`, `dy=2`, `dx=1,dy=1`) directly, with no new dependency beyond `scipy`, already in use.

```
Attack Influence Field (discrete, 8×8)
        ↓  build_attack_influence_surface()
Attack Influence Surface (continuous, fitted spline)
        ↓  sample_gradient()                          [existing]
Gradient  ∇φ = (φ_x, φ_y)             — where is it steepest?
        ↓
Hessian   H = [[φ_xx, φ_xy], [φ_xy, φ_yy]]   — how does the steepness change?
        ↓
Critical Points = {(x,y) : ∇φ(x,y) ≈ 0}, each classified by H
```

`sample_gradient` already evaluates the first partials via the fitted spline; the Hessian is the same mechanism one derivative order higher. Locating critical points means searching the same grid for where both gradient components vanish together, refined with a Newton step that uses the Hessian as its Jacobian.

## Visual examples

```
Maximum (White strongpoint)      Saddle (corridor between hotspots)     Minimum (Black strongpoint)
. . . . .                        + + . - -                             . . . . .
. + + + .                        + + . - -                             . - - - .
. + # + .                        . . X - -                             . - # - .
. + + + .                        - - . + +                             . - - - .
. . . . .                        - - . + +                             . . . . .
```

(`+` = White-dominant, `-` = Black-dominant, `.` ≈ neutral, `#`/`X` = the critical point itself.)

A reference case for testing: a synthetic 8×8 matrix shaped like a single centered Gaussian bump has one critical point — a maximum at the center — with a closed-form, negative-definite Hessian nearby. This is the case implementation should validate against before ever touching a real chess position.

## How Ridge, Valley, and Morse-Smale emerge from this

- **Ridge Analysis** and **Valley Analysis** — chains of points that are local maxima (ridge) or minima (valley) along a Hessian eigenvector direction: the mountain-range analogue of a single maximum or minimum. Undefined without the Hessian eigen-decomposition introduced here. Fully specified in Section 10.
- **Morse-Smale Complex** — once critical points are classified and gradient flow is computable (`sample_gradient`, existing), the complex is the partition of the surface into cells bounded by the flow lines connecting each maximum to each minimum through the saddles between them. It adds no new local object — it is the global bookkeeping of how this section's local objects connect across the board.

## A spline critical point is mathematically real — that does not make it chess-meaningful

Everything above establishes that a located, classified critical point is a genuine feature of `AttackInfluenceSurface`'s fitted spline: the gradient really does vanish there, and the second-derivative test really does hold. That is a fact about the *fitted surface*, not automatically a fact about *chess structure* — the surface is itself only an approximation, built by fitting a bicubic spline through just 64 known samples (the board's cells) and then reading it at arbitrary continuous points.

Two situations in particular can produce a mathematically legitimate critical point that is not a reliable chess signal:

- **Near the board boundary.** The spline is fit from a small, finite grid; its behavior right at the edges is the least constrained by real data and the most shaped by how the fit extrapolates past the last known sample. A critical point sitting almost on the edge of `[0.5, 7.5]²` is more likely to be an artifact of that extrapolation than a genuine structural feature of the position.
- **Where curvature is very weak.** A critical point whose Hessian eigenvalues are both extremely small is, numerically, barely distinguishable from the flat, near-zero-gradient noise inherent to fitting a smooth surface through a coarse 8×8 grid — the same undersampling behavior `smoothing_sigma` exists to soften (see `build_attack_influence_surface` above). A vanishingly shallow "peak" is not the same claim as a pronounced one.

Both were observed directly: Phase 2's localization tests documented spline ringing/overshoot from the 8×8 fit producing small extra critical points away from any intended feature, and Phase 4's first real-position render showed several `saddle`-classified points clustered right at the board edge.

This is why localization and classification (`locate_critical_points`, `classify_critical_points`) are kept strictly separate from a further quality-assessment step (`analysis/critical_point_quality.py`, `assess_critical_point_quality`): the math never lies about what the fitted surface does, and quality assessment never rewrites that math — it only adds an explicit, documented, always-inspectable judgment about which of the mathematically real critical points are trustworthy enough to present as chess-meaningful. A rejected point is not deleted or hidden from the data; it is retained with its reason recorded, exactly per this project's own "prefer explainability over complexity" principle.

---

# 10. Ridge and Valley Analysis

## First principles

Section 9 found points where the surface is flat in *every* direction at once (`∇φ = 0`). Ridge and Valley points relax that: a point where the surface is extremal in *one* direction only, while still possibly rising or falling in the perpendicular direction — the mountain-range analogue of a single peak. Standing on a ridge line, the ground drops away to either side, but you can still walk uphill or downhill *along* the ridge itself.

The Hessian's own eigenvectors are exactly the directions along which this "one direction only" test can be asked cleanly, because they are the directions in which curvature is *pure* — stepping along an eigenvector, to second order, `φ` bends only by that eigenvector's eigenvalue, with no coupling from the other direction. `compute_symmetric_eigenvalues` (`analysis/critical_points.py`) already returns `eigenvalue_min ≤ eigenvalue_max` for `H(x, y)`; call their (unit-length) eigenvectors `e_min` and `e_max`.

**Ridge point.** A point `p` is a ridge point if:

1. `eigenvalue_min(p) < 0` — curvature is concave-down in at least the `e_min` direction (the same sign a maximum requires, but only in one direction, not both).
2. `∇φ(p) · e_min(p) ≈ 0` — no slope in the `e_min` (cross-ridge) direction: `p` sits exactly at the crest of the local cross-section, not partway up its side. Equivalently, `∇φ(p)` is parallel to `e_max` — any remaining slope is only allowed to run *along* the ridge.

**Valley point** is the exact dual: `eigenvalue_max(p) > 0`, and `∇φ(p) · e_max(p) ≈ 0` (equivalently `∇φ(p)` parallel to `e_min`).

A ridge (valley) is then the connected chain of ridge (valley) points traced by walking along the *along*-crest eigenvector (`e_max` for a ridge, `e_min` for a valley) from a starting point.

This is the standard "height ridge" definition from ridge-detection literature (the same idea used for ridges in terrain and image analysis), specialized to a bicubic-spline surface with an already-available analytic Hessian — no new mathematical object beyond what Section 9 introduced, only a new *question* asked of it.

## Closed-form eigenvectors of the symmetric 2×2 Hessian

`compute_symmetric_eigenvalues` gives the eigenvalues but not the eigenvectors. For `H = [[f_xx, m], [m, f_yy]]` (`m` = the symmetrized mixed partial, exactly as `classify_critical_points` already computes it), the standard closed form is the Jacobi rotation angle:

```
θ = 0.5 * atan2(2m, f_xx - f_yy)

e_max = (cos θ, sin θ)     — eigenvector for eigenvalue_max
e_min = (-sin θ, cos θ)    — eigenvector for eigenvalue_min
```

This is robust (no division by a possibly-zero `m`) and reuses only `f_xx`, `f_yy`, `m`, already produced by `evaluate_hessian_at_points` — the same shared function `sample_hessian`, `refine_seed`, and `classify_critical_points` already call, unchanged.

**Degenerate case:** if `f_xx == f_yy` and `m == 0` at the same point (an isotropic point — curvature identical in every direction, e.g. exactly at the peak of a circular Gaussian bump), `atan2(0, 0)` is undefined and every direction is an eigenvector. No ridge/valley direction is well-defined there; implementation must detect this explicitly (rather than let `atan2` return an arbitrary angle silently) and treat it as a point with no defined cross-direction — a chain cannot be seeded or continued through it.

## The marching algorithm

1. **Seeding.** A quality-accepted maximum (Section 9's `CriticalPointQualityAssessment`) trivially satisfies the ridge conditions in the limit — `∇φ = 0` there, and both eigenvalues are negative, so `eigenvalue_min < 0` holds automatically. Maxima are therefore the natural seeds for ridge chains (marching outward along `±e_max`), symmetrically minima seed valley chains (marching along `±e_min`). This reuses Section 9's already-quality-filtered critical points rather than re-searching the whole grid from scratch.
2. **Step.** At the current point `p`, evaluate `H(p)` via `evaluate_hessian_at_points` (a single-point call, exactly as `refine_seed` already does), compute the along-crest eigenvector, and step `p_new = p ± step_size · e_along(p)`.
3. **Sign consistency.** An eigenvector and its negation are the same direction but not the same step — naively re-deriving `e_along` at every point risks the tracer flipping back on itself step to step. Each step picks whichever of `+e_along(p)` / `-e_along(p)` has positive dot product with the *previous* step's direction, so the walk always continues forward.
4. **Continuation check.** After each step, re-verify the ridge/valley condition at the new point within a tolerance (analogous in spirit to `CONVERGENCE_GRADIENT_THRESHOLD`): the cross-direction eigenvalue must keep the required sign, and `|∇φ(p) · e_cross(p)|` must stay small relative to `|∇φ(p)|`. A point that fails this has walked off the ridge/valley locus onto a differently-shaped part of the surface.
5. **Termination.** A chain stops, with an explicit recorded reason (mirroring `CriticalPointCandidate.status`'s vocabulary), when it: leaves the valid domain `[x_min, x_max] × [y_min, y_max]`; the cross-direction eigenvalue collapses toward zero (mirrors Section 9's `EIGENVALUE_ZERO_THRESHOLD`/quality-assessment `MIN_EIGENVALUE_MAGNITUDE` reasoning); the alignment condition in step 4 fails; it reaches another critical point (a legitimate Morse-theoretic boundary — two peaks joined by a ridge line generally pass through a saddle along the way, foreshadowing the Morse-Smale complex); or a maximum step budget is exhausted (mirrors `MAX_NEWTON_ITERATIONS`).

This is, in character, a simple explicit-Euler integral-curve tracer along a vector field — simpler than what Morse-Smale will eventually need (which integrates along the raw gradient field `∇φ` between critical points), but it introduces the one genuinely new piece of bookkeeping Morse-Smale will also depend on: following a *field of directions*, not a field of vectors, since eigenvectors (unlike gradients) carry no intrinsic sign.

## Relation to chess — working hypotheses, not established facts

Where Section 9's maxima and minima answer "where is the single sharpest concentration," ridges and valleys answer "what is the shape of the region around it" — the first object in this document describing an *extended* structure rather than a point.

**Everything in this section is a working hypothesis, not a proven chess fact.** The mathematics below is exact — it is a precise, checkable statement about the fitted spline. The chess reading attached to it is an interpretive claim about what that shape *tends* to correspond to over the board, and is not entitled to the same confidence merely because the math it rests on is rigorous. Each item below is therefore split into three explicitly separated parts, so the two kinds of claim — and the plan for closing the gap between them — are never merged into one sentence:

- **Mathematical definition** — what is actually computed, already exact and testable per this document's Section 10 machinery above.
- **Expected chess interpretation (hypothesis)** — what we currently believe this shape *tends* to mean over the board. Stated as a hypothesis on purpose.
- **Validation plan** — the concrete, specific way this hypothesis is intended to be checked against real positions before it is treated as established. Until that check happens, the hypothesis stays a hypothesis, exactly as Section 7 of `docs/audio.md` records its own MVP's listener-legibility claim as "still open" rather than assuming it.

### What a ridge represents

- **Mathematical definition.** A ridge is the connected chain of ridge points defined in Section 10 above: `eigenvalue_min(p) < 0` and `∇φ(p)` aligned with `e_max(p)`, traced by the marching algorithm from a quality-accepted maximum.
- **Expected chess interpretation (hypothesis).** A sustained spine of White-dominance concentration, where no single square is uniquely "the most" dominant but the whole line sits elevated above its surroundings — hypothesized to correlate with a **coordinated line of control**: an open file backed by doubled rooks, a long diagonal swept by a bishop-and-queen battery, or a set of pawns/pieces whose individual attack contributions overlap along a shared line. This is a *structural* hypothesis — a corridor held, not just a square contested — distinct from a lone maximum, which could equally be one overloaded square with no supporting structure around it.
- **Validation plan.** Curate a small reference set of real or hand-constructed positions with an independently-computable structural feature — e.g. a file with no pawns on it, occupied/attacked by ≥2 major pieces of one color, computed directly from `python-chess` board state, with no dependence on `AttackInfluenceSurface` at all. Run the ridge tracer on each and check, as an automatable Phase 6 regression test, whether the traced chain's coordinates fall along that independently-identified file or diagonal within a stated geometric tolerance. Only a chain that reliably lines up with an independently-verified structural feature earns the "coordinated line of control" reading; until that test exists and passes, this stays a hypothesis.

### What a valley represents

- **Mathematical definition.** The exact dual: the connected chain of valley points (`eigenvalue_max(p) > 0`, `∇φ(p)` aligned with `e_min(p)`), traced from a quality-accepted minimum.
- **Expected chess interpretation (hypothesis).** The same structural reading as a ridge, with White and Black exchanged: a coordinated line of Black control.
- **Validation plan.** The mirror of the ridge validation plan above, with the independently-computed structural feature checked for Black pieces instead of White.

### Long, continuous ridges vs. several short, disconnected ones

- **Mathematical definition.** Chain length and continuity as produced by the marching algorithm: an unbroken chain runs until it leaves the domain, reaches a critical point, or exhausts its step budget; a fragmented result is several separate chains, each terminated early by the continuation check in Section 10 (cross-direction eigenvalue collapsing, or the alignment condition failing).
- **Expected chess interpretation (hypothesis).** One long, unbroken ridge is hypothesized to read as a single, fully-held corridor (e.g. a completely open file or diagonal) — a stable structural asset rather than a momentary flare-up. Several short, disconnected fragments where one long ridge might otherwise be expected are hypothesized to read as *fragmented* influence, commonly because the line is physically interrupted (a blocking pawn or piece, or a diagonal broken by an exchange) — with the location of the break itself hypothesized to mark the obstruction, not a lack of attacking coordination.
- **Validation plan.** A paired synthetic-then-real test: (a) a position with a genuinely open, unobstructed diagonal/file, expecting one continuous ridge; (b) the identical position with a single piece inserted to block that line, expecting the ridge to fragment, with the break point's coordinates compared directly against the blocking piece's square. This "fragmentation coincides with a known obstruction" claim is directly automatable for Phase 6. The broader claim — that fragmentation in general (with no single obvious blocker) signals a "transitional or contested" position — is not mechanically checkable the same way, and is recorded here as an explicitly open interpretive question rather than assumed.

### Ridge intersections with saddle regions

- **Mathematical definition.** Section 9 already defines a saddle precisely (`D < 0` in the second-derivative test). A ridge/saddle intersection is either (a) a ridge chain terminating with reason `"reached_critical_point"` at a point classified `"saddle"`, or (b) a ridge chain passing within a small stated distance of a saddle without terminating there.
- **Expected chess interpretation (hypothesis).** Case (a) is hypothesized to mark a **front line**: an established White corridor running directly up against contested territory, where the settled structure ends and active contest begins — combining Section 9's existing saddle reading ("tension between two hotspots") with the ridge's corridor reading. Case (b) is hypothesized to mark a **key square**: a single contested pinch point along an otherwise White-favoring line, echoing the classical positional idea of a square where accurate play by either side can locally tip the balance without changing the broader structural advantage on either side of it.
- **Validation plan.** This is the hardest of the four to validate mechanically, because "key square" and "front line" are classical strategic concepts without one universally agreed computational ground truth. Two planned checks, neither sufficient alone: (1) a small curated set of textbook positions with a well-documented key square (a known idea in king-and-pawn endgame theory) or a well-documented open-file front line, checked for geometric coincidence with a detected ridge/saddle intersection; (2) a qualitative human review pass over the Phase 3 visualization on a sample of real games — explicitly modeled on `docs/audio.md` Section 7's own treatment of listener legibility as a distinct, tracked, and at-the-time-unresolved validation step, not something the implementation gets to mark done on its own. Until both are recorded, "front line" and "key square" remain hypotheses attached to a precisely-defined but not yet chess-validated geometric event.

## Boundary and weak-curvature caveats

The same caution Section 9 documents (lines above, "A spline critical point is mathematically real — that does not make it chess-meaningful") applies here, at the level of a whole chain rather than a single point: a chain traced mostly within the boundary margin of the fitted spline's domain, or one whose cross-sectional curvature stays only barely on the required side of zero throughout, is more likely a fitting artifact than real chess structure. This motivates a chain-level quality filter analogous to `analysis/critical_point_quality.py`, checked once the chain is fully traced rather than point by point.

## Reference case for testing

An anisotropic (unequal `sigma_x`, `sigma_y`) Gaussian bump, axis-aligned by construction — an 8×8 matrix built from a Gaussian stretched along one board axis. Its ridge is the closed-form straight line running along the long axis through the peak, with eigenvectors aligned to the coordinate axes by construction (avoiding the need to hand-verify the rotated closed form for a first correctness pass) and analytically known cross-sectional curvature everywhere along it. This is the case implementation should validate the eigenvector formula and the tracer against before ever touching a real chess position — the same role the single centered Gaussian bump played for Section 9.

## Data model

New dataclasses, following `CriticalPointCandidate` / `ClassifiedCriticalPoint` / `CriticalPointQualityAssessment`'s existing shape and naming convention in `chess_engine/models.py`:

- **`RidgeValleyPoint`** — one traced point along a chain: `x`, `y`, `value`, the cross-direction eigenvalue, the alignment residual `|∇φ · e_cross|`, `kind` (`"ridge"` / `"valley"`).
- **`RidgeValleyChain`** — an ordered list of `RidgeValleyPoint`, plus `kind`, the anchoring critical point it grew from (if any), and a termination reason per end, using the same status vocabulary as `CriticalPointCandidate.status` (`"left_domain"`, `"weak_curvature"`, `"misaligned"`, `"reached_critical_point"`, `"max_steps_reached"`).
- **`RidgeValleyQualityAssessment`** — wraps a `RidgeValleyChain` with `is_accepted` and `rejection_reasons`, mirroring `CriticalPointQualityAssessment` exactly.

---

# 11. Morse-Smale Complex

## First principles

Sections 9 and 10 answered "where is a single sharpest point" and "what is the shape of the region around it." The Morse-Smale complex answers the remaining question: how does the *whole* surface partition into regions, each belonging to exactly one maximum and one minimum? A 2-cell (basin) is the intersection of a maximum's descending manifold (the set of points that flow downhill, under `-∇φ`, to that maximum's own basin boundary) and a minimum's ascending manifold — bounded on every side by separatrices leaving the saddles between them. This adds no new local object beyond what Sections 9–10 already introduced (critical points, the Hessian, gradient flow); it is the global bookkeeping of how those local objects connect, exactly as foreshadowed in Section 9's "How Ridge, Valley, and Morse-Smale emerge from this."

## Separatrix tracing (`analysis/morse_smale.py::locate_morse_smale_separatrices`)

A separatrix is an integral curve of the gradient field leaving a saddle: two ascending branches (following `+∇φ`, seeded along the saddle's `eigenvalue_max` eigenvector, since the gradient itself is ~0 at the saddle) and two descending branches (following `-∇φ`, seeded along `eigenvalue_min`) — four per saddle. This is a simpler tracer than Section 10's ridge/valley marcher: gradient direction is unambiguous (no `_consistent_direction` sign-picking needed, unlike an eigenvector field), so each step after the first simply follows the local `∇φ`, evaluated pointwise on the fitted spline exactly as `refine_seed` and the ridge/valley tracer already do.

Anchors are restricted to quality-accepted saddles only (`CriticalPointQualityAssessment.is_accepted`, Section 9's own quality layer) — a mathematically real but untrustworthy saddle should not seed global topology. A branch terminates at: another accepted critical point (`"reached_critical_point"`), a quality-*rejected* one (`"reached_unreliable_point"` — stopped rather than silently passed through), the domain edge (`"left_domain"`), a near-zero gradient plateau (`"gradient_stagnation"`), a returning loop (`"self_intersection"`, reusing Section 10's `check_self_intersection` directly), or an exhausted step budget (`"max_steps_reached"`).

## Cell assembly (`assemble_morse_smale_cells`)

Cells are built *only* from already-traced separatrices — no gradient, Hessian, or surface evaluation happens in this step. Every closed separatrix (`termination_status == "reached_critical_point"`) contributes two directed half-edges (forward and backward); at each vertex, half-edges leaving it are sorted by the angle of their initial tangent direction. A cell's boundary is found by the standard planar-subdivision face trace: arriving at a vertex via one half-edge, continue via the *next* half-edge (by angle) at that vertex — the same "rotation system" construction used to enumerate faces of any embedded planar graph. Each half-edge belongs to exactly one face by this rule, so cells cannot be discovered twice.

Two outcomes per traced boundary:

- **Closed** — the walk returns to its starting half-edge. A genuine basin, with a well-defined polygon (Section 11's geometry, below).
- **Open / incomplete** — the walk reaches a saddle whose *next* direction is an unclosed separatrix (a "blocker," present in the angular ordering precisely so the walk cannot silently skip past it as if it weren't there). Recorded with the specific blocking separatrix in `open_boundaries`, never forced closed by inventing a boundary-rectangle edge that was never actually traced.

Anomalies that don't prevent assembly but are worth flagging — a separatrix reaching another saddle instead of an extremum, a saddle with other than its expected 4 branches, near-duplicate angular ordering, a face trace exceeding its step budget — are recorded as structured `TopologyIssue` entries (`kind`, the affected vertex/edge, a detail string), never silently dropped and never a reason to abort assembly.

## Cell geometry (derived, not stored)

`compute_cell_geometry` is a separate, on-demand function — `MorseSmaleCell` itself carries only topology (which separatrices, which direction, which vertices, closed or not). For a closed cell, the polygon is the actual traced curve (each boundary separatrix's own points, in its walked direction) between consecutive vertices, not a straight-line approximation — area via the shoelace formula, centroid via the standard polygon-centroid formula, perimeter by summing consecutive polygon-point distances. One correction worth recording here because it was wrong in an earlier draft of this document: perimeter is **not** simply the sum of each boundary separatrix's own `path_length`, because every separatrix stops short of its true target by design (Section 9's `REACHED_CRITICAL_POINT_DISTANCE`) — the polygon includes one additional closing segment per edge, from the last traced point to the true vertex, that `path_length` does not cover.

## Cell quality assessment (`assess_morse_smale_cell_quality`)

Mirrors Sections 9–10's own quality layers, applied one level up (whole cells, not points or chains): every cell is preserved and annotated, never deleted, with structured (enum-based) rejection reasons rather than free-text strings. Two genuinely different kinds of rejection:

- **Topology-invalid** — the cell itself is structurally suspect: not closed, or one of its own separatrices/vertices is independently named in a `TopologyIssue`.
- **Topology-valid but low quality** — a legitimately closed loop that simply doesn't clear an interpretive-reliability bar: area or perimeter near the numerical noise floor (calibrated the same way as Section 9's `MIN_EIGENVALUE_MAGNITUDE` — orders of magnitude above the noise a single marching step could produce, orders below a real measured cell), fewer than 3 distinct boundary vertices (rejects only the fully degenerate "bigon" case — a saddle whose two same-type branches happen to reach the same target), and two off-by-default checks (near-boundary participation, degenerate-vertex participation) mirroring Sections 9–10's own boundary-margin filters.

## Reference case for testing

The textbook "egg-crate" Morse function `f(x, y) = amplitude · cos(x) · cos(y)`: a checkerboard of maxima and minima at integer-period grid points with saddles at the half-period midpoints between them, entirely closed-form. On the board domain this produces a fully-tiled complex (every separatrix closes, no open cells) whose specific quads were hand-identified and cross-checked against an independent rhombus-diagonal area formula before being written into the test suite — the same role the single Gaussian bump played for Section 9 and the anisotropic bump for Section 10.

## Relation to chess — not yet a hypothesis, only a shape

Unlike Sections 9 and 10, this document does not yet propose a chess-semantic reading of a Morse-Smale cell (a "basin of local balance," a coordinated pocket of one side's dominance, or similar) as even a working hypothesis. The visualization (`visualization/morse_smale_plot.py`) and one real-position smoke render exist and are readable, but no validation plan comparable to Section 10's file/diagonal cross-checks has been designed or run. Any chess interpretation of cell shape, area, or count should be treated as entirely open until a future milestone proposes and validates one explicitly — consistent with this project's rule that a mathematically real object is not automatically a chess-meaningful one.

## Data model

New dataclasses in `chess_engine/models.py`, following the existing `CriticalPointCandidate`/`RidgeValleyChain` shape and naming convention:

- **`SeparatrixPoint`** / **`SeparatrixPath`** — one traced gradient-flow branch from a saddle (`flow_direction`: `"ascending"`/`"descending"`), with the same termination-status vocabulary style as `CriticalPointCandidate.status`.
- **`TopologyIssueKind`** / **`TopologyIssue`** — structured (enum-based) anomaly records produced during cell assembly.
- **`MorseSmaleCell`** — topology only: boundary separatrices, their traversal directions, boundary vertices, closed/open state, blocking separatrices for open cells.
- **`MorseSmaleComplex`** — the full result: vertices, edges (separatrices), cells, any half-edges left over from an abandoned trace, and topology issues.
- **`MorseSmaleCellGeometry`** — derived-only (polygon, area, centroid, perimeter), computed by `compute_cell_geometry`, never stored on a cell.
- **`MorseSmaleCellRejectionReasonKind`** / **`MorseSmaleCellRejectionReason`** / **`MorseSmaleCellQualityAssessment`** — the quality layer, mirroring `CriticalPointQualityAssessment` in shape.

---

# Philosophy

The project transforms

Chess

↓

Mathematics

↓

Geometry

↓

Visualization

↓

Music