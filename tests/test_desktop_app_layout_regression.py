"""
V0 layout regression coverage: locks in the fix for the maximized-window
regression where MathCanvas was crushed into a distorted, non-square strip
while AudioMixerPanel's Mute/Solo controls drifted arbitrarily far from
their voice row. No test here exists before this milestone -- see
`desktop_app/gl_canvas.py::compute_square_viewport` and the row/column
stretch changes in `desktop_app/main_window.py` and
`desktop_app/audio_mixer_panel.py`.

Deliberately relational, not exact-pixel: Qt's layout arithmetic is not a
contract this suite should pin to the pixel, only the behavioral invariants
that actually matter (see each test's docstring).
"""

import pytest
from PySide6.QtWidgets import QGroupBox, QScrollArea

from desktop_app.board_panel import BOARD_PIXELS, MIN_BOARD_PIXELS
from desktop_app.gl_canvas import MathCanvas, compute_square_viewport
from desktop_app.main_window import MainWindow

RESOLUTIONS = [(1280, 720), (1600, 900), (1920, 1080)]

# Below every measured MathCanvas floor across all three target resolutions
# (and the pathological-shrink probe used while building this fix, which
# bottomed out at 184px per side) -- catches a real collapse regression
# without pinning to a specific Qt layout-arithmetic outcome.
MIN_USABLE_CANVAS_SIDE_PX = 150


# ---------------------------------------------------------
# compute_square_viewport: pure function, no Qt/GL needed
# ---------------------------------------------------------


def test_square_viewport_centers_a_square_in_a_wide_rect():
    x, y, w, h = compute_square_viewport(800, 300)
    assert (w, h) == (300, 300)
    assert x == 250  # (800 - 300) / 2
    assert y == 0


def test_square_viewport_centers_a_square_in_a_tall_rect():
    x, y, w, h = compute_square_viewport(300, 800)
    assert (w, h) == (300, 300)
    assert x == 0
    assert y == 250  # (800 - 300) / 2


def test_square_viewport_is_unchanged_for_an_already_square_rect():
    assert compute_square_viewport(400, 400) == (0, 0, 400, 400)


def test_square_viewport_handles_zero_dimensions_without_crashing():
    assert compute_square_viewport(0, 500) == (0, 0, 0, 0)
    assert compute_square_viewport(500, 0) == (0, 0, 0, 0)
    assert compute_square_viewport(0, 0) == (0, 0, 0, 0)


def test_square_viewport_handles_negative_input_defensively():
    # Qt should never hand a widget a negative size, but the pure function
    # must not raise if it ever does (e.g. a future caller passing an
    # unvalidated delta).
    assert compute_square_viewport(-10, 200) == (0, 0, 0, 0)


def test_paint_gl_actually_applies_the_letterboxed_square_viewport(qapp):
    """
    The pure-function tests above prove compute_square_viewport's own
    arithmetic; this proves paintGL actually calls it (every frame, not
    resizeGL -- see gl_canvas.py::resizeGL's own comment on why: Qt resets
    the real GL viewport before every paintGL call regardless of what a
    prior resizeGL set) and sets the real GL viewport accordingly, from
    true device-pixel dimensions, closing that gap.

    Verified via canvas._viewport (the value paintGL itself computed and
    passed to glViewport) rather than a later glGetIntegerv(GL_VIEWPORT)
    query -- Qt was observed to reset the live GL_VIEWPORT state again
    after paintGL returns (for its own present/swap bookkeeping), so
    querying it back after the fact reads that later reset, not what was
    actually bound during this frame's draw calls. canvas._viewport is the
    same value paintGL passed to glViewport() moments earlier; the pixel-
    content test below is the independent, ground-truth confirmation.

    Skips (rather than fails) under the offscreen QPA platform, which does
    not provide a real GL context on this system -- same environment
    limitation documented in test_desktop_app_canvas.py.
    """
    canvas = MathCanvas()
    canvas.resize(800, 300)
    canvas.show()
    qapp.processEvents()

    if not canvas.isValid():
        pytest.skip(
            "No OpenGL context available under the current QPA platform "
            "(offscreen does not support real GL contexts on this system)."
        )

    device_width = round(canvas.width() * canvas.devicePixelRatio())
    device_height = round(canvas.height() * canvas.devicePixelRatio())
    assert canvas._viewport == compute_square_viewport(device_width, device_height)


def test_paint_gl_viewport_honors_a_non_default_device_pixel_ratio_in_rendered_pixels(qapp, monkeypatch):
    """
    V5 (DPI audit): the specific bug this locks in -- paintGL must compute
    the viewport from *device* pixels (self.width() * devicePixelRatio()),
    not the logical pixel width/height alone. Reproduced directly (not
    merely suspected) on a real HiDPI display (devicePixelRatio 1.25) as a
    badly distorted, render stretched to fill the entire framebuffer
    instead of a small centered letterboxed square; invisible at ratio
    1.0, which is the only ratio the offscreen QPA platform this suite
    normally runs under can produce -- so devicePixelRatio is explicitly
    monkeypatched here to exercise a non-1.0 ratio deterministically,
    regardless of the real platform's own DPI setting.

    Checks actual rendered pixel content (grabFramebuffer), not a later GL
    state query -- see the sibling test's docstring for why a
    glGetIntegerv(GL_VIEWPORT) query after the fact is unreliable here.
    """
    canvas = MathCanvas()
    canvas.resize(800, 300)  # wide rect -> letterbox bars needed left/right
    canvas.show()
    qapp.processEvents()

    if not canvas.isValid():
        pytest.skip(
            "No OpenGL context available under the current QPA platform "
            "(offscreen does not support real GL contexts on this system)."
        )

    monkeypatch.setattr(type(canvas), "devicePixelRatio", lambda self: 2.0)
    canvas.update()
    qapp.processEvents()

    image = canvas.grabFramebuffer()
    # CHROME_BACKGROUND (#0f172a) fills anything outside the letterboxed
    # square; the board's own light/dark squares are both much lighter.
    # Sampling near the image's left edge -- well outside where a small
    # centered square could possibly reach -- must therefore still be the
    # background color if letterboxing is respecting the *device*-pixel
    # (not logical-pixel) frame this image was actually rendered at.
    corner_color = image.pixelColor(2, image.height() // 2)
    assert (corner_color.red(), corner_color.green(), corner_color.blue()) == (15, 23, 42)


# ---------------------------------------------------------
# MainWindow geometry invariants at the three target resolutions
# ---------------------------------------------------------


def _resize_and_settle(app, window, width, height):
    window.resize(width, height)
    for _ in range(5):
        app.processEvents()


def _legend_groupbox(window):
    return window.layer_panel.findChild(QGroupBox)


def test_board_stays_square_and_within_bounds_at_every_resolution(qapp, qtbot):
    """
    V5 (responsive layout): the board is no longer setFixedSize -- it's
    bounded between MIN_BOARD_PIXELS and BOARD_PIXELS (see
    desktop_app/board_panel.py) so it can shrink at 1280x720/1366x768
    instead of forcing MainWindow's minimum height past the requested
    size. Every size in that range is still square and still playable;
    see tests/test_desktop_app_responsive_layout.py for the full V5
    coverage of exactly which sizes it takes at which resolution.
    """
    window = MainWindow()
    window.show()
    for width, height in RESOLUTIONS:
        _resize_and_settle(qapp, window, width, height)
        board_view = window.board_panel.board_view
        assert board_view.width() == board_view.height()
        assert MIN_BOARD_PIXELS <= board_view.width() <= BOARD_PIXELS


def test_math_canvas_never_collapses_into_a_thin_strip(qapp, qtbot):
    window = MainWindow()
    window.show()
    for width, height in RESOLUTIONS:
        _resize_and_settle(qapp, window, width, height)
        canvas = window.canvas
        assert min(canvas.width(), canvas.height()) >= MIN_USABLE_CANVAS_SIDE_PX


def test_rendered_math_viewport_is_always_square(qapp, qtbot):
    """
    The widget rect itself need not be square (see class docstring on
    MathCanvas) -- what must always be square is the letterboxed viewport
    compute_square_viewport derives from whatever rect the widget actually
    has at each target resolution.
    """
    window = MainWindow()
    window.show()
    for width, height in RESOLUTIONS:
        _resize_and_settle(qapp, window, width, height)
        canvas = window.canvas
        _, _, viewport_w, viewport_h = compute_square_viewport(canvas.width(), canvas.height())
        assert viewport_w == viewport_h
        assert viewport_w > 0


def test_primary_content_gets_substantially_more_vertical_space_than_tertiary_strips(qapp, qtbot):
    """
    PRIMARY = the workspace row (board + canvas + layer_panel, all three
    siblings sharing one row's height), TERTIARY = timeline + mixer.
    board_panel.height() stands in for the whole workspace row's height --
    since V7 (desktop workspace layout), board_panel is simply an ordinary,
    non-spanning sibling in that row (no rowSpan involved at all; see
    main_window.py's workspace_row), so its height directly *is* the row's
    height, not a stand-in for a span.
    """
    window = MainWindow()
    window.show()
    for width, height in RESOLUTIONS:
        _resize_and_settle(qapp, window, width, height)
        primary_height = window.board_panel.height()
        tertiary_height = window.timeline_panel.height() + window.audio_mixer_panel.height()
        assert primary_height > tertiary_height


def test_timeline_and_mixer_stay_compact_and_dont_grow_with_window_size(qapp, qtbot):
    window = MainWindow()
    window.show()
    _resize_and_settle(qapp, window, *RESOLUTIONS[0])
    baseline_timeline_height = window.timeline_panel.height()
    baseline_mixer_height = window.audio_mixer_panel.height()

    for width, height in RESOLUTIONS[1:]:
        _resize_and_settle(qapp, window, width, height)
        assert window.timeline_panel.height() == baseline_timeline_height
        assert window.audio_mixer_panel.height() == baseline_mixer_height


def test_mixer_mute_solo_controls_stay_close_to_their_voice_row(qapp, qtbot):
    """
    Regression for the specific reported symptom: Mute/Solo scattered far
    apart from the voice row as the window widens, because the inner grid
    had no column stretch and Qt split leftover width evenly across all 3
    columns. The gap between a voice's Mute and Solo checkbox must stay
    small and, critically, must NOT grow as the window widens -- a growing
    gap is exactly what an (incorrectly) stretch-driven column looks like.
    """
    window = MainWindow()
    window.show()

    gaps = []
    for width, height in RESOLUTIONS:
        _resize_and_settle(qapp, window, width, height)
        mute = next(iter(window.audio_mixer_panel._mute_checkboxes.values()))
        solo = next(iter(window.audio_mixer_panel._solo_checkboxes.values()))
        gap = solo.geometry().x() - (mute.geometry().x() + mute.geometry().width())
        assert 0 <= gap <= 80
        gaps.append(gap)

    assert gaps[0] == gaps[1] == gaps[2]


def test_legend_stays_visible_at_every_resolution(qapp, qtbot):
    window = MainWindow()
    window.show()
    for width, height in RESOLUTIONS:
        _resize_and_settle(qapp, window, width, height)
        legend = _legend_groupbox(window)
        assert legend is not None
        assert legend.isVisible()


def test_layer_controls_are_never_inside_a_scroll_area_at_any_resolution(qapp, qtbot):
    """
    V7 (desktop workspace layout): the actual invariant the live screenshot
    that motivated this milestone exposed -- V6a's single scroll budget
    shared between controls and legend could land on a scroll position
    showing only the legend, with every checkbox/slider/preset combo
    scrolled out of view, even though QWidget.isVisible() still reported
    True for them (isVisible does not mean "within the scrolled
    viewport"). V7 moves the controls out of any QScrollArea entirely, so
    there is no scroll position that can hide them -- this asserts that
    structural guarantee directly, at every target resolution, rather than
    only checking isVisible() as the sibling test above does for the
    legend.
    """
    window = MainWindow()
    window.show()

    def _is_descendant_of_scroll_area(widget):
        ancestor = widget.parentWidget()
        while ancestor is not None and ancestor is not window.layer_panel:
            if isinstance(ancestor, QScrollArea):
                return True
            ancestor = ancestor.parentWidget()
        return False

    for width, height in RESOLUTIONS:
        _resize_and_settle(qapp, window, width, height)
        assert not _is_descendant_of_scroll_area(window.layer_panel._preset_combo)
        for checkbox in window.layer_panel._checkboxes.values():
            assert not _is_descendant_of_scroll_area(checkbox)
            assert checkbox.isVisible()


def test_repeated_resize_cycles_produce_stable_geometry(qapp, qtbot):
    """Cycling through the same resolutions twice must land on identical
    geometry the second time -- no drift/accumulation across resize events."""
    window = MainWindow()
    window.show()

    def _snapshot():
        return {
            "board": (window.board_panel.width(), window.board_panel.height()),
            "canvas": (window.canvas.width(), window.canvas.height()),
            "layer_panel": (window.layer_panel.width(), window.layer_panel.height()),
            "timeline": (window.timeline_panel.width(), window.timeline_panel.height()),
            "mixer": (window.audio_mixer_panel.width(), window.audio_mixer_panel.height()),
        }

    first_pass = []
    for width, height in RESOLUTIONS:
        _resize_and_settle(qapp, window, width, height)
        first_pass.append(_snapshot())

    second_pass = []
    for width, height in RESOLUTIONS:
        _resize_and_settle(qapp, window, width, height)
        second_pass.append(_snapshot())

    assert first_pass == second_pass
