import time

import chess
import pytest
from OpenGL import GL

from analysis.attack_influence import build_attack_influence_field
from desktop_app.gl_canvas import VERTEX_COUNT, MathCanvas
from desktop_app.layers.attack_influence_layer import ATTACK_INFLUENCE_LAYER
from desktop_app.layers.critical_points_layer import CRITICAL_POINTS_LAYER
from desktop_app.layers.equipotential_layer import EQUIPOTENTIAL_LAYER
from desktop_app.layers.gradient_layer import GRADIENT_LAYER
from desktop_app.layers.morse_smale_layer import MORSE_SMALE_LAYER
from desktop_app.layers.ridge_valley_layer import RIDGE_VALLEY_LAYER
from desktop_app.main_window import MainWindow
from tests.conftest import ready_cache_entry

REFERENCE_CANVAS_SIZE_PX = 896
BASELINE_FRAME_SAMPLES = 300

# docs/interactive_ui.md Part 11's budget: 60fps sustained with all six
# registered field layers visible simultaneously.
FRAME_BUDGET_MS = 1000.0 / 60.0

ALL_SIX_LAYERS = (
    ATTACK_INFLUENCE_LAYER,
    EQUIPOTENTIAL_LAYER,
    GRADIENT_LAYER,
    RIDGE_VALLEY_LAYER,
    MORSE_SMALE_LAYER,
    CRITICAL_POINTS_LAYER,
)


def _midgame_board() -> chess.Board:
    # An asymmetric, non-trivial position -- not the symmetric starting
    # position -- so every layer (gradient direction, ridge/valley,
    # critical points, Morse-Smale cells) has real, distinguishable content
    # to render, not a degenerate all-zero field.
    board = chess.Board()
    for move in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"):
        board.push_san(move)
    return board


def test_main_window_constructs_headlessly_without_error(qapp):
    # The requirement this exercises: the new UI can start in tests with no
    # display, no manual interaction, and no exception -- regardless of
    # whether the platform in use can back it with a real GL context (see
    # test_frame_time_baseline below for the GL-context-dependent case).
    window = MainWindow()

    assert window.canvas is not None
    # Phase 5b adds the board panel alongside the canvas, so the central
    # widget is now a container for both, not the canvas itself -- both must
    # still be real descendants of it.
    assert window.canvas.parent() is window.centralWidget()
    assert window.board_panel.parent() is window.centralWidget()
    assert window.windowTitle() == "VectorChess"


def test_starting_position_reaches_the_canvas_and_matches_direct_computation(qapp, qtbot):
    window = MainWindow()
    expected = build_attack_influence_field(chess.Board())

    # MainWindow.__init__ already called request() for the starting position
    # before this test could attach a signal listener, and this trivial
    # workload can finish before that -- poll instead of qtbot.waitSignal,
    # which would race a signal that may have already fired.
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    assert window.canvas._overlay_colors.shape == (VERTEX_COUNT, 4)

    # Re-derive the same colors directly from the existing, unmodified
    # analysis pipeline and confirm they match exactly -- the numeric ground
    # truth check the phase requires.
    entry = ready_cache_entry(chess.Board())
    expected_frame = ATTACK_INFLUENCE_LAYER.data_source(entry)
    expected_vertex_colors = ATTACK_INFLUENCE_LAYER.renderer(expected_frame)

    assert (window.canvas._overlay_colors == expected_vertex_colors).all()
    # entry was built independently of window's own cache entry -- confirm
    # both still agree with a third, fully independent direct computation.
    assert entry.analysis.attack_influence_field.matrix == expected.matrix


def test_disabling_the_overlay_actually_hides_it_on_screen(qapp):
    """
    Regression test for a bug found while wiring the layer-strip UI
    (this milestone): `paintGL` previously drew the Attack Influence
    overlay unconditionally, never checking `_overlay_enabled` -- so
    unchecking "Attack Influence" in the new layer panel correctly updated
    `SessionState`/`_overlay_enabled` but had no visible effect on the
    canvas. Fixed in `desktop_app/gl_canvas.py::paintGL`; this test renders
    both states with a real GL context and confirms the pixels differ.

    Skips (rather than fails) under the offscreen QPA platform, which does
    not provide a real GL context on this system -- same environment
    limitation as `test_frame_time_baseline` above.
    """
    canvas = MathCanvas()
    canvas.resize(64, 64)
    canvas.show()
    qapp.processEvents()

    if not canvas.isValid():
        pytest.skip(
            "No OpenGL context available under the current QPA platform "
            "(offscreen does not support real GL contexts on this system)."
        )

    entry = ready_cache_entry(_midgame_board())
    frame = ATTACK_INFLUENCE_LAYER.data_source(entry)
    rendered = ATTACK_INFLUENCE_LAYER.renderer(frame)
    canvas.set_overlay_colors(rendered)

    canvas.set_overlay_enabled(True)
    qapp.processEvents()
    enabled_image = canvas.grabFramebuffer()

    canvas.set_overlay_enabled(False)
    qapp.processEvents()
    disabled_image = canvas.grabFramebuffer()

    assert enabled_image != disabled_image


def test_frame_time_baseline(qapp):
    """
    Phase 5c multi-layer frame-time baseline (docs/interactive_ui.md Part
    11): all six registered layers rendered simultaneously, at the
    reference canvas size, compared honestly against the 60fps/16.67ms
    budget. Reports the measurement; does not fail the build if the budget
    isn't met -- per the task's own instruction not to optimize
    prematurely, this only measures and records.

    Skips (rather than fails) when no real OpenGL context is available --
    verified during Phase 5a implementation that Qt's `offscreen` QPA
    platform does not provide one on this Windows environment, which is
    what tests/conftest.py selects by default for headless runs. Run this
    file with QT_QPA_PLATFORM unset (or set to the platform default) on a
    machine with a display/GPU driver to exercise real GL rendering and see
    the measured number.
    """
    canvas = MathCanvas()
    canvas.resize(REFERENCE_CANVAS_SIZE_PX, REFERENCE_CANVAS_SIZE_PX)
    canvas.show()
    qapp.processEvents()

    if not canvas.isValid():
        pytest.skip(
            "No OpenGL context available under the current QPA platform "
            "(offscreen does not support real GL contexts on this system). "
            "Re-run without QT_QPA_PLATFORM=offscreen to measure the real "
            "frame-time baseline."
        )

    entry = ready_cache_entry(_midgame_board())
    canvas.set_layer_order([layer.id for layer in ALL_SIX_LAYERS])

    for layer in ALL_SIX_LAYERS:
        frame = layer.data_source(entry)
        rendered = layer.renderer(frame)
        if isinstance(rendered, list):
            canvas.set_layer_geometry(layer.id, rendered)
        else:
            canvas.set_overlay_colors(rendered)
        canvas.set_layer_enabled(layer.id, True)

    qapp.processEvents()

    canvas.makeCurrent()
    start = time.perf_counter()
    for _ in range(BASELINE_FRAME_SAMPLES):
        canvas.paintGL()
    GL.glFinish()
    elapsed_seconds = time.perf_counter() - start
    canvas.doneCurrent()

    average_frame_time_ms = (elapsed_seconds / BASELINE_FRAME_SAMPLES) * 1000
    fps = 1000 / average_frame_time_ms if average_frame_time_ms > 0 else float("inf")
    within_budget = average_frame_time_ms <= FRAME_BUDGET_MS

    print(
        f"\n[Phase 5c frame-time baseline] {average_frame_time_ms:.4f} ms/frame "
        f"(~{fps:.0f} fps) -- 6 layers, {REFERENCE_CANVAS_SIZE_PX}x{REFERENCE_CANVAS_SIZE_PX} "
        f"reference canvas. Budget: {FRAME_BUDGET_MS:.4f} ms/frame (60fps). "
        f"Within budget: {within_budget}."
    )

    assert average_frame_time_ms >= 0  # sanity only -- not a performance gate
