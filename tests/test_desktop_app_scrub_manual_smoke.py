import chess
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

STRIP_WIDTH = 400


def _mouse_event(kind, x, y, button, buttons):
    local = QPointF(x, y)
    return QMouseEvent(kind, local, local, button, buttons, Qt.KeyboardModifier.NoModifier)


def _press(strip, x, y=5.0):
    strip.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, x, y, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton))


def _move(strip, x, y=5.0):
    strip.mouseMoveEvent(_mouse_event(QEvent.Type.MouseMove, x, y, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton))


def _release(strip, x, y=5.0):
    strip.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, x, y, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton))


def test_manual_acceptance_smoke_test_end_to_end(qapp, qtbot):
    """
    Automated stand-in for the approved plan's manual desktop acceptance
    test (Section 13), run headlessly against the real MainWindow object
    graph (this environment has no real display -- see tests/conftest.py's
    QT_QPA_PLATFORM=offscreen setting) since a literal mouse/screen session
    isn't possible here. Walks the exact numbered steps from the plan.
    """
    from desktop_app.main_window import MainWindow

    # Step 1: launch, play a short opening.
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6"]:
        window.session_state.make_move(chess.Move.from_uci(uci))
        qtbot.waitUntil(lambda: not window.transition_controller.is_animating, timeout=5000)

    strip = window.timeline_panel._scrub_strip
    strip.resize(STRIP_WIDTH, strip.FIXED_HEIGHT)
    qtbot.wait(10)

    # Step 2: press and hold, drag slowly across the whole strip.
    initial_node = window.session_state.current_node
    initial_overlay = window.canvas._overlay_colors.copy()
    expected_label = "White to move" if initial_node.board().turn == chess.WHITE else "Black to move"

    _press(strip, 0)
    assert window.scrub_controller.is_scrubbing is True

    overlays_differed = False
    for x in range(0, STRIP_WIDTH + 1, 20):
        _move(strip, x)
        qtbot.wait(2)
        if window.canvas._overlay_colors is not None and not (window.canvas._overlay_colors == initial_overlay).all():
            overlays_differed = True

    assert overlays_differed  # canvas fields morphed continuously during the drag
    assert window.session_state.current_node is initial_node  # never committed mid-drag
    assert window.board_panel.board_view._preview_node is not None  # board snaps to a real position
    assert window.board_panel.side_to_move_label.text() == expected_label  # unaffected by the preview

    # Step 3: release mid-drag -> commits to the nearer real ply.
    with qtbot.waitSignal(window.session_state.current_node_changed, timeout=1000):
        _release(strip, STRIP_WIDTH // 2)

    assert window.board_panel.board_view._preview_node is None
    all_path_fens = {n.board().board_fen() for n in window.session_state.mainline_path()}
    assert window.session_state.current_node.board().board_fen() in all_path_fens  # a real position, never fabricated

    # Step 4: undo, play a different move (a branch), switch back, scrub
    # across the branch point -- scrubbing stays within the active branch.
    root = window.session_state.mainline_path()[0]
    window.session_state.go_to_start()
    qtbot.waitUntil(lambda: not window.transition_controller.is_animating, timeout=5000)

    with qtbot.waitSignal(window.session_state.current_node_changed, timeout=1000):
        window.session_state.make_move(chess.Move.from_uci("d2d4"))  # a genuinely new branch
    new_branch_tip = window.session_state.current_node
    qtbot.waitUntil(lambda: not window.transition_controller.is_animating, timeout=5000)
    assert len(root.variations) == 2  # both branches coexist -- nothing destroyed

    with qtbot.waitSignal(window.session_state.current_node_changed, timeout=1000):
        window.session_state.set_current_node(root)
    qtbot.waitUntil(lambda: not window.transition_controller.is_animating, timeout=5000)

    strip.resize(STRIP_WIDTH, strip.FIXED_HEIGHT)
    _press(strip, 0)
    active_path_len = window.scrub_controller.path_length
    assert active_path_len == len(window.session_state.active_path())

    with qtbot.waitSignal(window.session_state.current_node_changed, timeout=1000):
        _release(strip, STRIP_WIDTH)  # drag to the far end -> the active branch's own tip

    reached = window.session_state.current_node
    assert reached.board().board_fen() in all_path_fens or reached is new_branch_tip or reached is root
    qtbot.waitUntil(lambda: not window.transition_controller.is_animating, timeout=5000)

    # Step 5: make a fresh move, immediately scrub onto it before it's
    # necessarily settled -- no crash, no garbage frame. Return to a known
    # position first (step 4's scrub may have landed on either branch,
    # depending on which one was active) so this move is unambiguously legal.
    with qtbot.waitSignal(window.session_state.current_node_changed, timeout=1000):
        window.session_state.set_current_node(root)
    qtbot.waitUntil(lambda: not window.transition_controller.is_animating, timeout=5000)
    with qtbot.waitSignal(window.session_state.current_node_changed, timeout=1000):
        window.session_state.make_move(chess.Move.from_uci("g1f3"))
    strip.resize(STRIP_WIDTH, strip.FIXED_HEIGHT)
    _press(strip, 0)
    _move(strip, STRIP_WIDTH)
    qtbot.wait(10)
    _release(strip, STRIP_WIDTH)
    qtbot.waitUntil(lambda: not window.transition_controller.is_animating, timeout=5000)

    # Step 6: trigger a move animation, try to press-drag within its window.
    with qtbot.waitSignal(window.session_state.current_node_changed, timeout=1000):
        window.session_state.undo()
    if window.transition_controller.is_animating:
        _press(strip, 0)
        assert window.scrub_controller.is_scrubbing is False  # ignored, not stuck
    qtbot.waitUntil(lambda: not window.transition_controller.is_animating, timeout=5000)

    # Step 7: close the window mid-drag -- clean shutdown, no crash.
    _press(strip, 0)
    _move(strip, 100)
    window.close()

    assert window.position_cache._shutdown_requested.is_set()
    assert window.scrub_controller.is_scrubbing is False
    qtbot.wait(10)
