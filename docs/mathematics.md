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

- **Ridge Analysis** — a ridge is the connected chain of points that are local maxima along the Hessian's minor-curvature eigenvector direction: the mountain-range analogue of a single maximum. Undefined without the Hessian eigen-decomposition introduced here.
- **Valley Analysis** — the exact dual, chains of local minima along the major-curvature eigenvector direction. Same machinery, opposite sign.
- **Morse-Smale Complex** — once critical points are classified and gradient flow is computable (`sample_gradient`, existing), the complex is the partition of the surface into cells bounded by the flow lines connecting each maximum to each minimum through the saddles between them. It adds no new local object — it is the global bookkeeping of how this section's local objects connect across the board.

## A spline critical point is mathematically real — that does not make it chess-meaningful

Everything above establishes that a located, classified critical point is a genuine feature of `AttackInfluenceSurface`'s fitted spline: the gradient really does vanish there, and the second-derivative test really does hold. That is a fact about the *fitted surface*, not automatically a fact about *chess structure* — the surface is itself only an approximation, built by fitting a bicubic spline through just 64 known samples (the board's cells) and then reading it at arbitrary continuous points.

Two situations in particular can produce a mathematically legitimate critical point that is not a reliable chess signal:

- **Near the board boundary.** The spline is fit from a small, finite grid; its behavior right at the edges is the least constrained by real data and the most shaped by how the fit extrapolates past the last known sample. A critical point sitting almost on the edge of `[0.5, 7.5]²` is more likely to be an artifact of that extrapolation than a genuine structural feature of the position.
- **Where curvature is very weak.** A critical point whose Hessian eigenvalues are both extremely small is, numerically, barely distinguishable from the flat, near-zero-gradient noise inherent to fitting a smooth surface through a coarse 8×8 grid — the same undersampling behavior `smoothing_sigma` exists to soften (see `build_attack_influence_surface` above). A vanishingly shallow "peak" is not the same claim as a pronounced one.

Both were observed directly: Phase 2's localization tests documented spline ringing/overshoot from the 8×8 fit producing small extra critical points away from any intended feature, and Phase 4's first real-position render showed several `saddle`-classified points clustered right at the board edge.

This is why localization and classification (`locate_critical_points`, `classify_critical_points`) are kept strictly separate from a further quality-assessment step (`analysis/critical_point_quality.py`, `assess_critical_point_quality`): the math never lies about what the fitted surface does, and quality assessment never rewrites that math — it only adds an explicit, documented, always-inspectable judgment about which of the mathematically real critical points are trustworthy enough to present as chess-meaningful. A rejected point is not deleted or hidden from the data; it is retained with its reason recorded, exactly per this project's own "prefer explainability over complexity" principle.

---

# Future Mathematics

## Ridge Analysis

Detects the main influence ridges.

---

## Valley Analysis

Detects influence valleys.

---

## Morse-Smale Complex

Topological decomposition of the continuous field.

Purpose

Understand the global structure of the position.

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