import chess
import chess.pgn

from desktop_app.full_position_analysis import build_full_position_analysis
from desktop_app.layers.attack_influence_layer import ATTACK_INFLUENCE_LAYER
from desktop_app.main_window import MainWindow
from desktop_app.session_state import DEFAULT_LAYER_VISIBILITY
from tests.conftest import ready_cache_entry


def _expected_vertex_colors(board: chess.Board):
    entry = ready_cache_entry(board)
    frame = ATTACK_INFLUENCE_LAYER.data_source(entry)
    return ATTACK_INFLUENCE_LAYER.renderer(frame)


# ---------------------------------------------------------
# the canvas stays synchronized with the current position
# ---------------------------------------------------------


def test_canvas_updates_to_the_new_position_after_a_move(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    starting_colors = window.canvas._overlay_colors.copy()

    window.session_state.make_move(chess.Move.from_uci("e2e4"))
    expected = _expected_vertex_colors(window.session_state.current_node.board())

    qtbot.waitUntil(
        lambda: window.canvas._overlay_colors is not None
        and (window.canvas._overlay_colors == expected).all(),
        timeout=5000,
    )
    # And it genuinely changed -- not a coincidental match with the start.
    assert not (window.canvas._overlay_colors == starting_colors).all()


def test_canvas_reflects_the_position_again_after_undo(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)

    window.session_state.make_move(chess.Move.from_uci("e2e4"))
    moved_colors = _expected_vertex_colors(window.session_state.current_node.board())
    qtbot.waitUntil(
        lambda: window.canvas._overlay_colors is not None
        and (window.canvas._overlay_colors == moved_colors).all(),
        timeout=5000,
    )

    window.session_state.undo()
    starting_colors = _expected_vertex_colors(chess.Board())

    qtbot.waitUntil(
        lambda: window.canvas._overlay_colors is not None
        and (window.canvas._overlay_colors == starting_colors).all(),
        timeout=5000,
    )


def test_all_six_layers_have_geometry_uploaded_after_the_startup_position_settles(qapp, qtbot):
    # "Changing chess position must update all enabled layers to the same
    # position snapshot" -- checked here for every registered layer, not
    # only Attack Influence's overlay colors. Checked against
    # `_layer_geometries` (CPU-side, always populated by set_layer_geometry
    # regardless of GL context) rather than `_layer_gpu_buffers` (GPU-side,
    # only populated once a real GL context exists via initializeGL -- which
    # this offscreen platform doesn't provide, per the same environment
    # limitation Phase 5a already documented).
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    # The five new layers upload through set_layer_geometry -- give the
    # (fast, but background-thread-driven) render pass a moment to run.
    qtbot.waitUntil(
        lambda: all(
            layer_id in window.canvas._layer_geometries
            for layer_id in ("equipotential", "gradient", "ridge_valley", "morse_smale", "critical_points")
        ),
        timeout=5000,
    )

    # Geometry is uploaded for every layer regardless of visibility -- an
    # already-uploaded buffer must exist the instant a hidden layer is
    # toggled back on, with no recomputation.
    for layer_id in ("equipotential", "gradient", "ridge_valley", "morse_smale", "critical_points"):
        assert layer_id in window.canvas._layer_geometries

    # But the *enabled* flag follows this milestone's sensible-default
    # visibility (DEFAULT_LAYER_VISIBILITY), not a blanket "everything on" --
    # that default is exactly what keeps the first view legible instead of
    # all six layers overlapping at once. Attack Influence is excluded here:
    # it draws through the overlay path (`_overlay_enabled`), never through
    # `_layer_enabled` -- checked separately below.
    for layer_id in ("equipotential", "gradient", "ridge_valley", "morse_smale", "critical_points"):
        assert window.canvas._layer_enabled.get(layer_id, True) == DEFAULT_LAYER_VISIBILITY[layer_id]
    assert window.canvas._overlay_enabled == DEFAULT_LAYER_VISIBILITY["attack_influence"]


# ---------------------------------------------------------
# Position Cache reuse / no duplicate computation
# ---------------------------------------------------------


def test_revisiting_an_already_computed_position_does_not_recompute(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)  # startup settled

    call_count = {"e2e4": 0}
    moved_fen = chess.Board()
    moved_fen.push_san("e4")
    moved_fen = moved_fen.board_fen()

    def counting_builder(board: chess.Board):
        if board.board_fen() == moved_fen:
            call_count["e2e4"] += 1
        return build_full_position_analysis(board)

    window.position_cache._builder = counting_builder

    window.session_state.make_move(chess.Move.from_uci("e2e4"))
    qtbot.waitUntil(lambda: call_count["e2e4"] == 1, timeout=5000)

    window.session_state.undo()
    window.session_state.redo()  # revisits the e2e4 position a second time
    qtbot.wait(100)  # give any (incorrect) recomputation a chance to happen

    assert call_count["e2e4"] == 1


def test_no_duplicate_computation_for_rapid_repeated_requests_of_the_same_position(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)  # startup settled

    call_count = {"n": 0}

    def counting_builder(board: chess.Board):
        call_count["n"] += 1
        return build_full_position_analysis(board)

    window.position_cache._builder = counting_builder

    window.session_state.make_move(chess.Move.from_uci("e2e4"))  # issues one request
    board_after_move = window.session_state.current_node.board()
    # Simulate redundant repeated requests for the same not-yet-visited
    # position -- must not resubmit work.
    window.position_cache.request(board_after_move)
    window.position_cache.request(board_after_move)
    window.position_cache.request(board_after_move)

    qtbot.waitUntil(lambda: call_count["n"] == 1, timeout=5000)
    qtbot.wait(100)  # give any incorrect extra submission a chance to run
    assert call_count["n"] == 1


def test_toggling_layer_visibility_never_triggers_recomputation(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)

    call_count = {"n": 0}

    def counting_builder(board: chess.Board):
        call_count["n"] += 1
        return build_full_position_analysis(board)

    window.position_cache._builder = counting_builder
    qtbot.wait(50)
    baseline_calls = call_count["n"]  # 0: nothing new requested yet at this fen

    # critical_points starts visible (True), morse_smale starts hidden
    # (False) under this milestone's default visibility -- chosen
    # specifically so every toggle below is a genuine state transition, not
    # a redundant write to a value already in place.
    window.session_state.set_layer_visible("critical_points", False)
    window.session_state.set_layer_visible("critical_points", True)
    window.session_state.set_layer_visible("morse_smale", True)
    qtbot.wait(100)

    assert call_count["n"] == baseline_calls
    # And the toggle actually took effect on the canvas's draw flags.
    assert window.canvas._layer_enabled["morse_smale"] is True
    assert window.canvas._layer_enabled["critical_points"] is True


# ---------------------------------------------------------
# LayerPanel is wired end to end through SessionState
# ---------------------------------------------------------


def test_layer_panel_checkbox_click_reaches_the_canvas(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    qtbot.waitUntil(lambda: "gradient" in window.canvas._layer_geometries, timeout=5000)

    assert window.canvas._layer_enabled.get("gradient") is False  # default-hidden

    window.layer_panel._checkboxes["gradient"].setChecked(True)

    assert window.session_state.layer_visible("gradient") is True
    assert window.canvas._layer_enabled["gradient"] is True


def test_layer_panel_checkbox_click_on_attack_influence_reaches_the_overlay(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)

    assert window.canvas._overlay_enabled is True  # default-visible

    window.layer_panel._checkboxes["attack_influence"].setChecked(False)

    assert window.session_state.layer_visible("attack_influence") is False
    assert window.canvas._overlay_enabled is False


def test_undo_redo_still_works_alongside_the_new_layer_panel(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)

    window.session_state.make_move(chess.Move.from_uci("e2e4"))
    moved_colors = _expected_vertex_colors(window.session_state.current_node.board())
    qtbot.waitUntil(
        lambda: window.canvas._overlay_colors is not None
        and (window.canvas._overlay_colors == moved_colors).all(),
        timeout=5000,
    )

    assert window.session_state.undo() is True
    assert window.session_state.redo() is True
    assert window.session_state.current_node.board().piece_at(chess.E4) is not None
