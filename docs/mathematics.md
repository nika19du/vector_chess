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

# Future Mathematics

## Hessian

Second derivatives.

Classifies local shape.

---

## Critical Points

Locations where

Gradient = 0

Possible types

Maximum

Minimum

Saddle

---

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