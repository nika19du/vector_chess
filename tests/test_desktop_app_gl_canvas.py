import numpy as np
from OpenGL import GL

from desktop_app.gl_canvas import LayerGeometry, MathCanvas

# These tests exercise the CPU-side state of the Phase 5c multi-layer
# mechanism only -- a bare, unshown MathCanvas() never triggers
# initializeGL() (Qt calls it lazily, only just before real painting), so
# `_vao` stays None and every set_* call here only touches plain Python
# dicts/lists, no real GL context required. Actual GPU-side rendering is
# covered by test_desktop_app_canvas.py::test_frame_time_baseline, which
# skips (rather than fakes a result) when no real GL context is available.
#
# `qapp` is still required on every test: constructing any QWidget (even one
# that's never shown) needs a live QApplication instance, which pytest-qt's
# `qapp` fixture provides -- without it, widget construction hangs rather
# than raising, since Qt is waiting on an event loop/platform integration
# that was never initialized.
#
# LayerGeometry instances are compared by identity (`is`), never `==`:
# it holds numpy array fields, and the dataclass-generated `__eq__`
# compares fields as a tuple, which raises ValueError ("ambiguous truth
# value") for a multi-element array's `==` result instead of giving a
# clean bool. Identity is also the more precise check here anyway -- it
# proves the exact object was stored, not merely an equal-looking copy.


def _geometry(vertex_count: int, primitive: int = GL.GL_LINES) -> LayerGeometry:
    return LayerGeometry(
        positions=np.zeros((vertex_count, 2), dtype=np.float32),
        colors=np.ones((vertex_count, 4), dtype=np.float32),
        primitive=primitive,
    )


def test_set_layer_order_stores_the_given_order_verbatim(qapp):
    canvas = MathCanvas()

    canvas.set_layer_order(["a", "b", "c"])

    assert canvas._layer_order == ["a", "b", "c"]


def test_set_layer_geometry_stores_geometry_without_a_gl_context(qapp):
    canvas = MathCanvas()
    geometry = _geometry(4)

    canvas.set_layer_geometry("gradient", [geometry])

    stored = canvas._layer_geometries["gradient"]
    assert len(stored) == 1 and stored[0] is geometry
    # No GL context was ever created, so nothing was uploaded.
    assert canvas._layer_gpu_buffers == {}


def test_layers_default_to_enabled(qapp):
    canvas = MathCanvas()

    # A layer never explicitly toggled defaults to drawn -- matches
    # SessionState.layer_visible's own "opt-out, not opt-in" default.
    assert canvas._layer_enabled.get("never_toggled", True) is True


def test_set_layer_enabled_stores_the_flag_without_touching_geometry(qapp):
    canvas = MathCanvas()
    geometry = _geometry(4)
    canvas.set_layer_geometry("gradient", [geometry])

    canvas.set_layer_enabled("gradient", False)

    assert canvas._layer_enabled["gradient"] is False
    # Disabling never clears or mutates the stored geometry -- it's still
    # there, just not drawn, so re-enabling needs no recomputation.
    stored = canvas._layer_geometries["gradient"]
    assert len(stored) == 1 and stored[0] is geometry


def test_set_layer_geometry_replaces_the_previous_geometry_for_that_layer(qapp):
    canvas = MathCanvas()
    canvas.set_layer_geometry("gradient", [_geometry(4)])

    canvas.set_layer_geometry("gradient", [_geometry(10)])

    assert len(canvas._layer_geometries["gradient"]) == 1
    assert canvas._layer_geometries["gradient"][0].positions.shape == (10, 2)


def test_multiple_layers_are_tracked_independently(qapp):
    canvas = MathCanvas()

    canvas.set_layer_geometry("gradient", [_geometry(2)])
    canvas.set_layer_geometry("critical_points", [_geometry(6, primitive=GL.GL_TRIANGLES)])
    canvas.set_layer_enabled("gradient", False)

    assert canvas._layer_enabled["gradient"] is False
    assert canvas._layer_enabled.get("critical_points", True) is True
    assert canvas._layer_geometries["gradient"][0].positions.shape == (2, 2)
    assert canvas._layer_geometries["critical_points"][0].positions.shape == (6, 2)


def test_overlay_defaults_to_enabled_and_fully_opaque(qapp):
    canvas = MathCanvas()

    assert canvas._overlay_enabled is True
    assert canvas._overlay_opacity == 1.0


def test_set_overlay_enabled_stores_the_flag(qapp):
    canvas = MathCanvas()

    canvas.set_overlay_enabled(False)

    assert canvas._overlay_enabled is False


def test_set_overlay_opacity_stores_the_value(qapp):
    canvas = MathCanvas()

    canvas.set_overlay_opacity(0.3)

    assert canvas._overlay_opacity == 0.3


def test_set_layer_opacity_stores_the_value_without_touching_geometry(qapp):
    canvas = MathCanvas()
    geometry = _geometry(4)
    canvas.set_layer_geometry("gradient", [geometry])

    canvas.set_layer_opacity("gradient", 0.5)

    assert canvas._layer_opacity["gradient"] == 0.5
    stored = canvas._layer_geometries["gradient"]
    assert len(stored) == 1 and stored[0] is geometry


def test_a_layer_can_have_multiple_geometries_with_different_primitives(qapp):
    # Morse-Smale needs exactly this: a filled-triangle interior and a
    # separate line boundary for the same layer.
    canvas = MathCanvas()
    fill = _geometry(3, primitive=GL.GL_TRIANGLES)
    boundary = _geometry(2, primitive=GL.GL_LINES)

    canvas.set_layer_geometry("morse_smale", [fill, boundary])

    stored = canvas._layer_geometries["morse_smale"]
    assert len(stored) == 2 and stored[0] is fill and stored[1] is boundary
