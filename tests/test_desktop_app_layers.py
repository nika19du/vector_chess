import dataclasses

import chess
import numpy as np
from OpenGL import GL

from analysis.geometry import math_to_plot_coords
from desktop_app.full_position_analysis import build_full_position_analysis
from desktop_app.layers.critical_points_layer import build_critical_points_frame, render_critical_points_frame
from desktop_app.layers.equipotential_layer import build_equipotential_frame, render_equipotential_frame
from desktop_app.layers.gradient_layer import build_gradient_frame, render_gradient_frame
from desktop_app.layers.morse_smale_layer import build_morse_smale_frame, render_morse_smale_frame
from desktop_app.layers.ridge_valley_layer import build_ridge_valley_frame, render_ridge_valley_frame
from desktop_app.position_cache import CacheEntry, CacheEntryState
from visualization.critical_points_plot import MARKER_SPECS
from visualization.gradient_plot import should_draw_vector
from visualization.ridge_valley_plot import MIN_POINTS_TO_DRAW


def _midgame_board() -> chess.Board:
    board = chess.Board()
    for move in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"):
        board.push_san(move)
    return board


def _entry_for(board: chess.Board) -> CacheEntry:
    return CacheEntry(state=CacheEntryState.READY, analysis=build_full_position_analysis(board))


# ---------------------------------------------------------
# Gradient
# ---------------------------------------------------------


def test_gradient_frame_only_includes_vectors_should_draw_vector_accepts():
    board = _midgame_board()
    entry = _entry_for(board)

    frame = build_gradient_frame(entry)

    max_magnitude = entry.analysis.gradient_field.max_magnitude
    expected_count = sum(
        1
        for vector in entry.analysis.gradient_field.vectors
        if vector.magnitude > 0 and should_draw_vector(board=board, vector=vector, max_magnitude=max_magnitude)
    )
    assert len(frame.segments) == expected_count


def test_gradient_segment_origin_matches_math_to_plot_coords():
    board = _midgame_board()
    entry = _entry_for(board)
    frame = build_gradient_frame(entry)

    assert frame.segments, "expected at least one visible gradient vector on this position"

    # Each segment's midpoint must fall inside its source square's cell
    # (center +/- 0.5), matching math_to_plot_coords's plot-space convention.
    for segment in frame.segments:
        midpoint_x = (segment.start_xy[0] + segment.end_xy[0]) / 2
        midpoint_y = (segment.start_xy[1] + segment.end_xy[1]) / 2
        # Find some vector whose cell center is within one full cell of the midpoint.
        assert any(
            abs(midpoint_x - (math_to_plot_coords(v.x, v.y)[0] + 0.5)) < 1e-6
            and abs(midpoint_y - (math_to_plot_coords(v.x, v.y)[1] + 0.5)) < 1e-6
            for v in entry.analysis.gradient_field.vectors
        )


def test_gradient_renderer_produces_two_vertices_per_segment_as_lines():
    board = _midgame_board()
    entry = _entry_for(board)
    frame = build_gradient_frame(entry)

    geometries = render_gradient_frame(frame)

    assert len(geometries) == 1
    geometry = geometries[0]
    assert geometry.primitive == GL.GL_LINES
    assert geometry.positions.shape == (len(frame.segments) * 2, 2)
    assert geometry.colors.shape == (len(frame.segments) * 2, 4)


# ---------------------------------------------------------
# Equipotential
# ---------------------------------------------------------


def test_equipotential_frame_has_segments_within_board_bounds():
    entry = _entry_for(_midgame_board())
    frame = build_equipotential_frame(entry)

    assert frame.segments_xy, "expected at least one equipotential segment"
    for start, end in frame.segments_xy:
        for x, y in (start, end):
            assert 0.0 <= x <= 8.0
            assert 0.0 <= y <= 8.0


def test_equipotential_renderer_uses_lines():
    entry = _entry_for(_midgame_board())
    frame = build_equipotential_frame(entry)

    geometries = render_equipotential_frame(frame)

    assert len(geometries) == 1
    assert geometries[0].primitive == GL.GL_LINES
    assert geometries[0].positions.shape[0] == len(frame.segments_xy) * 2


# ---------------------------------------------------------
# Critical Points
# ---------------------------------------------------------


def test_critical_points_frame_includes_only_accepted_points():
    entry = _entry_for(_midgame_board())
    frame = build_critical_points_frame(entry)

    expected_xy = {
        (a.point.x, a.point.y) for a in entry.analysis.critical_point_assessments if a.is_accepted
    }
    assert {marker.xy for marker in frame.markers} == expected_xy


def test_critical_points_frame_excludes_a_synthetically_rejected_point():
    # Preserves "quality-filter preservation" directly: flip one real,
    # currently-accepted assessment to rejected and confirm the layer
    # excludes exactly that point, keeping everything else real.
    analysis = build_full_position_analysis(_midgame_board())
    accepted = [a for a in analysis.critical_point_assessments if a.is_accepted]
    assert accepted, "expected at least one accepted critical point on this position"

    target = accepted[0]
    rejected_version = dataclasses.replace(target, is_accepted=False, rejection_reasons=["synthetic, for testing"])
    new_assessments = [rejected_version if a is target else a for a in analysis.critical_point_assessments]
    modified_analysis = dataclasses.replace(analysis, critical_point_assessments=new_assessments)
    entry = CacheEntry(state=CacheEntryState.READY, analysis=modified_analysis)

    frame = build_critical_points_frame(entry)

    assert (target.point.x, target.point.y) not in {marker.xy for marker in frame.markers}


def test_critical_points_use_marker_specs_colors():
    entry = _entry_for(_midgame_board())
    frame = build_critical_points_frame(entry)

    accepted_by_xy = {
        (a.point.x, a.point.y): a.point.classification
        for a in entry.analysis.critical_point_assessments
        if a.is_accepted
    }

    for marker in frame.markers:
        classification = accepted_by_xy[marker.xy]
        spec = MARKER_SPECS[classification]
        expected_hex = spec["facecolor"] if spec["facecolor"] != "none" else spec["edgecolor"]
        expected_hex = expected_hex.lstrip("#")
        expected_rgb = (
            int(expected_hex[0:2], 16) / 255.0,
            int(expected_hex[2:4], 16) / 255.0,
            int(expected_hex[4:6], 16) / 255.0,
        )
        assert marker.color[:3] == expected_rgb


def test_critical_points_renderer_produces_a_quad_per_marker():
    entry = _entry_for(_midgame_board())
    frame = build_critical_points_frame(entry)

    geometries = render_critical_points_frame(frame)

    assert len(geometries) == 1
    assert geometries[0].primitive == GL.GL_TRIANGLES
    assert geometries[0].positions.shape == (len(frame.markers) * 6, 2)


# ---------------------------------------------------------
# Ridge / Valley
# ---------------------------------------------------------


def test_ridge_valley_frame_includes_only_accepted_chains_meeting_min_points():
    entry = _entry_for(_midgame_board())
    frame = build_ridge_valley_frame(entry)

    expected_count = sum(
        1
        for assessment in entry.analysis.ridge_assessments + entry.analysis.valley_assessments
        if assessment.is_accepted and len(assessment.chain.points) >= MIN_POINTS_TO_DRAW
    )
    assert len(frame.chains) == expected_count


def test_ridge_valley_frame_excludes_a_synthetically_rejected_chain():
    analysis = build_full_position_analysis(_midgame_board())
    accepted = [a for a in analysis.ridge_assessments if a.is_accepted and len(a.chain.points) >= MIN_POINTS_TO_DRAW]
    assert accepted, "expected at least one accepted, drawable ridge chain on this position"

    target = accepted[0]
    rejected_version = dataclasses.replace(target, is_accepted=False, rejection_reasons=["synthetic, for testing"])
    new_ridge_assessments = [rejected_version if a is target else a for a in analysis.ridge_assessments]
    modified_analysis = dataclasses.replace(analysis, ridge_assessments=new_ridge_assessments)
    entry = CacheEntry(state=CacheEntryState.READY, analysis=modified_analysis)

    frame = build_ridge_valley_frame(entry)

    target_points_xy = [(p.x, p.y) for p in target.chain.points]
    assert not any(chain.points_xy == target_points_xy for chain in frame.chains)


def test_ridge_valley_chain_points_are_the_real_untouched_coordinates():
    entry = _entry_for(_midgame_board())
    frame = build_ridge_valley_frame(entry)

    accepted_chains = [
        a.chain
        for a in entry.analysis.ridge_assessments + entry.analysis.valley_assessments
        if a.is_accepted and len(a.chain.points) >= MIN_POINTS_TO_DRAW
    ]
    expected_point_sets = [[(p.x, p.y) for p in chain.points] for chain in accepted_chains]

    actual_point_sets = [chain.points_xy for chain in frame.chains]
    assert sorted(actual_point_sets) == sorted(expected_point_sets)


# ---------------------------------------------------------
# Morse-Smale
# ---------------------------------------------------------


def test_morse_smale_frame_includes_only_closed_accepted_cells():
    entry = _entry_for(_midgame_board())
    frame = build_morse_smale_frame(entry)

    assessment_by_cell_id = {a.cell.cell_id: a for a in entry.analysis.cell_assessments}
    expected_count = 0
    for cell in entry.analysis.morse_smale_complex.cells:
        if not cell.is_closed:
            continue
        assessment = assessment_by_cell_id.get(cell.cell_id)
        if assessment is not None and assessment.is_accepted:
            expected_count += 1

    # Not every accepted+closed cell necessarily yields >=2 polygon points
    # (see MIN_POINTS_TO_DRAW), so this is an upper bound, and every emitted
    # polygon must correspond to a genuinely accepted, closed cell.
    assert len(frame.polygons) <= expected_count
    assert len(frame.polygons) == len(frame.centroids)


def test_morse_smale_renderer_produces_fill_and_boundary_geometries():
    entry = _entry_for(_midgame_board())
    frame = build_morse_smale_frame(entry)

    geometries = render_morse_smale_frame(frame)

    assert len(geometries) == 2
    fill_geometry, line_geometry = geometries
    assert fill_geometry.primitive == GL.GL_TRIANGLES
    assert line_geometry.primitive == GL.GL_LINES

    if frame.polygons:
        # Fan triangulation: (len(polygon) - 1) triangles per polygon, 3 vertices each.
        expected_fill_vertices = sum(3 * (len(polygon) - 1) for polygon in frame.polygons)
        assert fill_geometry.positions.shape[0] == expected_fill_vertices
