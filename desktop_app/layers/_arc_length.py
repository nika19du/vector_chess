from __future__ import annotations

import math


def resample_by_arc_length(points_xy: list[tuple[float, float]], sample_count: int) -> list[tuple[float, float]]:
    """
    Resamples an ordered polyline to `sample_count` points evenly spaced by
    NORMALIZED cumulative arc length -- a pure geometric resampling of
    already-traced points, used only so two independently-traced curves
    (e.g. the same ridge traced fresh at move 14 and again, independently,
    at move 15 -- possibly with a different point count and different exact
    positions) get a common parametrization to interpolate along, instead of
    naive index-for-index pairing (which would be meaningless once the two
    point counts differ).

    Shared by Ridge/Valley chains (an open curve) and Morse-Smale cell
    boundaries (a closed polygon, `polygon[0] == polygon[-1]` per
    `chess_engine.models.MorseSmaleCellGeometry`'s own contract) -- a closed
    polygon is just a curve whose first and last sample coincide, so no
    separate closed-curve code path is needed.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be >= 1")
    if len(points_xy) == 0:
        raise ValueError("points_xy must not be empty")
    if len(points_xy) == 1:
        return [points_xy[0]] * sample_count

    cumulative = [0.0]
    for previous, current in zip(points_xy, points_xy[1:]):
        distance = math.hypot(current[0] - previous[0], current[1] - previous[1])
        cumulative.append(cumulative[-1] + distance)
    total_length = cumulative[-1]

    if total_length == 0.0:
        return [points_xy[0]] * sample_count

    samples: list[tuple[float, float]] = []
    for step in range(sample_count):
        fraction = (step / (sample_count - 1)) if sample_count > 1 else 0.0
        samples.append(_point_at_arc_length(points_xy, cumulative, fraction * total_length))
    return samples


def _point_at_arc_length(
    points_xy: list[tuple[float, float]], cumulative: list[float], target: float
) -> tuple[float, float]:
    if target <= 0.0:
        return points_xy[0]
    if target >= cumulative[-1]:
        return points_xy[-1]

    for index in range(1, len(cumulative)):
        if cumulative[index] >= target:
            segment_length = cumulative[index] - cumulative[index - 1]
            local_t = 0.0 if segment_length == 0.0 else (target - cumulative[index - 1]) / segment_length
            x0, y0 = points_xy[index - 1]
            x1, y1 = points_xy[index]
            return (x0 + (x1 - x0) * local_t, y0 + (y1 - y0) * local_t)

    return points_xy[-1]  # pragma: no cover -- unreachable given the guards above
