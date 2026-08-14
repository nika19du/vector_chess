import chess

from desktop_app.main_window import MainWindow

RESOLUTIONS = [(1280, 720), (1600, 900), (1920, 1080)]


def _siblings(session_state):
    session_state.make_move(chess.Move.from_uci("e2e4"))
    e4_node = session_state.current_node
    session_state.undo()
    session_state.make_move(chess.Move.from_uci("d2d4"))
    d4_node = session_state.current_node
    return e4_node, d4_node


def _resize_and_settle(app, window, width, height):
    window.resize(width, height)
    for _ in range(5):
        app.processEvents()


# ---------------------------------------------------------
# workspace swap follows compare_state exactly
#
# `isHidden()` (not `isVisible()`) is used throughout: `isVisible()` also
# depends on the whole ancestor chain actually being shown on screen
# (`window.show()`), which most of these tests deliberately skip --
# `isHidden()` reflects only this widget's own last `setVisible()` call,
# exactly what MainWindow's `_on_compare_state_changed` controls.
# ---------------------------------------------------------


def test_compare_panel_starts_hidden_and_normal_workspace_starts_visible(qapp, qtbot):
    window = MainWindow()

    assert window.compare_panel.isHidden() is True
    assert window.board_panel.isHidden() is False
    assert window.canvas.isHidden() is False
    assert window.layer_panel.isHidden() is False


def test_entering_compare_swaps_the_workspace(qapp, qtbot):
    window = MainWindow()
    e4_node, d4_node = _siblings(window.session_state)
    qtbot.wait(10)

    window.session_state.enter_compare(d4_node, e4_node)

    assert window.compare_panel.isHidden() is False
    assert window.board_panel.isHidden() is True
    assert window.canvas.isHidden() is True
    assert window.layer_panel.isHidden() is True


def test_exiting_compare_restores_the_normal_workspace_byte_for_byte(qapp, qtbot):
    window = MainWindow()
    e4_node, d4_node = _siblings(window.session_state)
    qtbot.wait(10)
    window.session_state.enter_compare(d4_node, e4_node)

    window.session_state.exit_compare()

    assert window.compare_panel.isHidden() is True
    assert window.board_panel.isHidden() is False
    assert window.canvas.isHidden() is False
    assert window.layer_panel.isHidden() is False


def test_clicking_the_timeline_compare_button_opens_the_compare_workspace(qapp, qtbot):
    window = MainWindow()
    e4_node, d4_node = _siblings(window.session_state)
    qtbot.wait(10)

    window.timeline_panel._compare_buttons[d4_node].click()

    assert window.session_state.compare_state is not None
    assert window.compare_panel.isHidden() is False
    assert window.board_panel.isHidden() is True


def test_timeline_navigation_exits_compare_and_restores_the_normal_workspace(qapp, qtbot):
    window = MainWindow()
    e4_node, d4_node = _siblings(window.session_state)
    qtbot.wait(10)
    window.session_state.enter_compare(d4_node, e4_node)

    window.timeline_panel._first_button.click()
    qtbot.wait(10)

    assert window.session_state.compare_state is None
    assert window.compare_panel.isHidden() is True
    assert window.board_panel.isHidden() is False


def test_current_node_is_unchanged_by_entering_and_exiting_compare(qapp, qtbot):
    window = MainWindow()
    e4_node, d4_node = _siblings(window.session_state)
    qtbot.wait(10)
    before = window.session_state.current_node

    window.session_state.enter_compare(d4_node, e4_node)
    assert window.session_state.current_node is before

    window.session_state.exit_compare()
    assert window.session_state.current_node is before


# ---------------------------------------------------------
# TransitionController is never invoked by compare mode
# ---------------------------------------------------------


def test_transition_controller_does_not_animate_on_entering_or_exiting_compare(qapp, qtbot):
    window = MainWindow()
    e4_node, d4_node = _siblings(window.session_state)
    # Let the real move-to-move transition triggered by _siblings()'s own
    # navigation fully settle first -- compare mode must add no animation
    # of its own on top of that, but the ordinary one from making moves
    # needs to finish before it can be used as a "nothing changed" baseline.
    qtbot.waitUntil(lambda: not window.transition_controller.is_animating, timeout=2000)
    settled_fen_before = window.session_state.transition_state.to_fen

    window.session_state.enter_compare(d4_node, e4_node)
    window.session_state.exit_compare()

    assert window.session_state.transition_state.to_fen == settled_fen_before
    assert window.transition_controller.is_animating is False


# ---------------------------------------------------------
# audio is untouched by compare mode
# ---------------------------------------------------------


def test_entering_and_exiting_compare_never_publishes_to_the_audio_engine(qapp, qtbot):
    window = MainWindow()
    e4_node, d4_node = _siblings(window.session_state)
    qtbot.wait(10)

    publish_calls = []
    original_publish = window.audio_engine.publish

    def _counting_publish(state):
        publish_calls.append(state)
        return original_publish(state)

    window.audio_engine.publish = _counting_publish

    window.session_state.enter_compare(d4_node, e4_node)
    window.session_state.exit_compare()

    # AudioController only ever reacts to current_node_changed/scrub signals
    # (desktop_app/audio_controller.py) -- neither enter_compare nor
    # exit_compare emits current_node_changed, so this must be exactly zero,
    # not merely "no crash".
    assert publish_calls == []


# ---------------------------------------------------------
# rapid enter/exit leaves a clean, consistent state
# ---------------------------------------------------------


def test_rapid_enter_exit_leaves_the_normal_workspace_visible_and_consistent(qapp, qtbot):
    window = MainWindow()
    e4_node, d4_node = _siblings(window.session_state)
    qtbot.wait(10)

    for _ in range(20):
        window.session_state.enter_compare(d4_node, e4_node)
        window.session_state.exit_compare()

    assert window.session_state.compare_state is None
    assert window.compare_panel.isHidden() is True
    assert window.board_panel.isHidden() is False
    assert window.canvas.isHidden() is False
    assert window.layer_panel.isHidden() is False


# ---------------------------------------------------------
# layout stays within bounds at target resolutions while comparing
# ---------------------------------------------------------


def test_compare_workspace_never_exceeds_the_requested_size_at_target_resolutions(qapp, qtbot):
    window = MainWindow()
    window.show()
    e4_node, d4_node = _siblings(window.session_state)
    qtbot.wait(10)
    window.session_state.enter_compare(d4_node, e4_node)

    for width, height in RESOLUTIONS:
        _resize_and_settle(qapp, window, width, height)
        central = window.centralWidget()
        assert central.width() <= width
        assert central.height() <= height
