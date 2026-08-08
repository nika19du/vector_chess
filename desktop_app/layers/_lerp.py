from __future__ import annotations

import numpy as np


def lerp(a: float, b: float, t: float) -> float:
    """
    Linear interpolation with exact-endpoint branching: `t<=0` returns `a`
    and `t>=1` returns `b` verbatim, never via `a + (b - a) * t`, whose
    floating-point rounding is not guaranteed bit-exact at the boundary.
    Every animation invariant in this milestone ("t=0 is exactly the
    previous cached state, t=1 is exactly the next cached state") depends
    on this being a real branch, not an approximation.
    """
    if t <= 0.0:
        return a
    if t >= 1.0:
        return b
    return a + (b - a) * t


def lerp_point(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    return (lerp(a[0], b[0], t), lerp(a[1], b[1], t))


def lerp_array(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Array counterpart of `lerp`, same exact-endpoint branching."""
    if t <= 0.0:
        return np.array(a, dtype=np.float32)
    if t >= 1.0:
        return np.array(b, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return a + (b - a) * t


def ease_in_out(t: float) -> float:
    """
    Smoothstep -- a presentation-only timing curve applied to the transition
    controller's raw elapsed-time fraction (docs/interactive_ui.md Part 6:
    "~500ms eased"). Never applied to mathematical field values themselves,
    only to the interpolation parameter `t` fed into `lerp`/`lerp_array`.
    """
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)
