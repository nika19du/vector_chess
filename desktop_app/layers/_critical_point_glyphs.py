from __future__ import annotations

import math

# V3: matches visualization/critical_points_plot.py's own hollow-circle
# degenerate marker approximated as a closed polygon of straight segments --
# a rendering-technique choice (GL has no native circle primitive), not a
# new visual concept.
RING_SEGMENT_COUNT = 16


def triangle_up_vertices(x: float, y: float, half_size: float) -> list[tuple[float, float]]:
    """
    Upward-pointing triangle (maximum) -- one GL_TRIANGLES primitive, apex
    visually at the top. Apex is at y - half_size, not y + half_size:
    plot-space y grows downward (screen/image convention, same as
    analysis/geometry.py's plot coordinates and gradient_layer.py's own
    "math Y grows up, plot Y grows down"), so the visually-higher point on
    screen is the *smaller* plot-y value.
    """
    return [(x, y - half_size), (x - half_size, y + half_size), (x + half_size, y + half_size)]


def triangle_down_vertices(x: float, y: float, half_size: float) -> list[tuple[float, float]]:
    """Downward-pointing triangle (minimum) -- triangle_up_vertices mirrored across y."""
    return [(x, y + half_size), (x - half_size, y - half_size), (x + half_size, y - half_size)]


def cross_line_segments(x: float, y: float, half_size: float) -> list[tuple[float, float]]:
    """
    Two diagonal line segments forming an X (saddle). Consecutive pairs are
    each one GL_LINES segment: (0,1) is one diagonal, (2,3) is the other.
    """
    return [
        (x - half_size, y - half_size),
        (x + half_size, y + half_size),
        (x - half_size, y + half_size),
        (x + half_size, y - half_size),
    ]


def ring_line_segments(
    x: float, y: float, radius: float, segment_count: int = RING_SEGMENT_COUNT
) -> list[tuple[float, float]]:
    """
    A closed circular outline (degenerate), approximated as `segment_count`
    consecutive GL_LINES segments -- consecutive pairs are each one segment,
    wrapping back to the first point to close the loop.
    """
    points = [
        (
            x + radius * math.cos(2 * math.pi * index / segment_count),
            y + radius * math.sin(2 * math.pi * index / segment_count),
        )
        for index in range(segment_count)
    ]
    segments: list[tuple[float, float]] = []
    for index in range(segment_count):
        segments.append(points[index])
        segments.append(points[(index + 1) % segment_count])
    return segments
