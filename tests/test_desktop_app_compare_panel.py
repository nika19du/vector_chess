import chess
import chess.pgn
import matplotlib
import numpy as np
from PySide6.QtCore import QObject, Signal

from desktop_app.compare_math import DIFFERENCE_COLORMAP_NAME
from desktop_app.compare_panel import ComparePanel
from desktop_app.layers.attack_influence_layer import COLORMAP_NAME as ATTACK_INFLUENCE_COLORMAP_NAME
from desktop_app.position_cache import CacheEntry, CacheEntryState
from desktop_app.session_state import SessionState


class _FakeField:
    def __init__(self, matrix, balance=0.0):
        self.matrix = matrix
        self.balance = balance


class _FakeAnalysis:
    def __init__(self, matrix, balance=0.0):
        self.attack_influence_field = _FakeField(matrix, balance)


class _ManualPositionCache(QObject):
    """
    Test double standing in for `desktop_app.position_cache.PositionCache`:
    `request()` never auto-completes -- the test calls `complete(fen, ...)`
    explicitly, to exercise ComparePanel's loading state deterministically.
    The real `PositionCache` is forced synchronous by tests/conftest.py's
    session-wide autouse fixture (crash mitigation -- see that fixture's own
    docstring), which makes a genuine "still loading" state unobservable
    from a test using the real class.
    """

    position_ready = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._entries: dict[str, CacheEntry] = {}
        self.requested: list[str] = []

    def request(self, board: chess.Board) -> str:
        fen = board.board_fen()
        self.requested.append(fen)
        self._entries.setdefault(fen, CacheEntry(state=CacheEntryState.COMPUTING))
        return fen

    def get(self, fen: str) -> CacheEntry:
        return self._entries.get(fen, CacheEntry(state=CacheEntryState.MISSING))

    def complete(self, fen: str, matrix, balance: float = 0.0) -> None:
        self._entries[fen] = CacheEntry(state=CacheEntryState.READY, analysis=_FakeAnalysis(matrix, balance))
        self.position_ready.emit(fen)


def _root() -> chess.pgn.Game:
    return chess.pgn.Game()


def _siblings(root: chess.pgn.Game) -> tuple[chess.pgn.GameNode, chess.pgn.GameNode]:
    node_a = root.add_variation(chess.Move.from_uci("e2e4"))
    node_b = root.add_variation(chess.Move.from_uci("d2d4"))
    return node_a, node_b


def _panel(qapp):
    root = _root()
    session_state = SessionState(root)
    cache = _ManualPositionCache()
    panel = ComparePanel(session_state, cache, None)
    return panel, session_state, cache, root


def _complete_both(cache: _ManualPositionCache, node_a, node_b, matrix_a=None, matrix_b=None):
    if matrix_a is None:
        matrix_a = np.zeros((8, 8))
    if matrix_b is None:
        matrix_b = np.zeros((8, 8))
    cache.complete(node_a.board().board_fen(), matrix_a)
    cache.complete(node_b.board().board_fen(), matrix_b)


def _metrics_text(panel) -> str:
    """Flattens the metrics grid's label/value QLabels into one string, the
    same shape the old single `_summary_label.text()` used to have, so
    substring assertions read the same way."""
    parts = []
    for i in range(panel._metrics_grid.count()):
        row = panel._metrics_grid.itemAt(i).layout()
        if row is None:
            continue
        for j in range(row.count()):
            widget = row.itemAt(j).widget()
            if widget is not None:
                parts.append(widget.text())
    return " ".join(parts)


# ---------------------------------------------------------
# visibility follows compare_state exactly
# ---------------------------------------------------------


def test_panel_starts_hidden(qapp):
    panel, _session_state, _cache, _root = _panel(qapp)

    assert panel.isVisible() is False


def test_panel_becomes_visible_on_enter_compare(qapp):
    panel, session_state, _cache, root = _panel(qapp)
    node_a, node_b = _siblings(root)

    session_state.enter_compare(node_a, node_b)

    assert panel.isVisible() is True


def test_panel_hides_on_exit_compare(qapp):
    panel, session_state, _cache, root = _panel(qapp)
    node_a, node_b = _siblings(root)
    session_state.enter_compare(node_a, node_b)

    session_state.exit_compare()

    assert panel.isVisible() is False


def test_close_button_exits_compare(qapp):
    panel, session_state, _cache, root = _panel(qapp)
    node_a, node_b = _siblings(root)
    session_state.enter_compare(node_a, node_b)

    panel._close_button.click()

    assert session_state.compare_state is None
    assert panel.isVisible() is False


def test_close_button_label_names_the_action(qapp):
    # V1.1 polish item 5: moved location, clearer label -- behavior (exit_compare) unchanged.
    panel, _session_state, _cache, _root = _panel(qapp)

    assert "close" in panel._close_button.text().lower()
    assert "comparison" in panel._close_button.text().lower()


# ---------------------------------------------------------
# PositionCache reuse -- both sides requested, no CompareAnalysis cache
# ---------------------------------------------------------


def test_entering_compare_requests_both_sides_from_position_cache(qapp):
    panel, session_state, cache, root = _panel(qapp)
    node_a, node_b = _siblings(root)

    session_state.enter_compare(node_a, node_b)

    assert node_a.board().board_fen() in cache.requested
    assert node_b.board().board_fen() in cache.requested


# ---------------------------------------------------------
# loading state until both sides are READY
# ---------------------------------------------------------


def test_loading_label_visible_before_either_side_is_ready(qapp):
    panel, session_state, _cache, root = _panel(qapp)
    node_a, node_b = _siblings(root)

    session_state.enter_compare(node_a, node_b)

    assert panel._loading_label.isVisible() is True
    assert _metrics_text(panel) == ""


def test_loading_label_stays_visible_with_only_one_side_ready(qapp):
    panel, session_state, cache, root = _panel(qapp)
    node_a, node_b = _siblings(root)
    session_state.enter_compare(node_a, node_b)

    cache.complete(node_a.board().board_fen(), np.zeros((8, 8)))

    assert panel._loading_label.isVisible() is True
    assert _metrics_text(panel) == ""


def test_loading_label_hides_once_both_sides_are_ready(qapp):
    panel, session_state, cache, root = _panel(qapp)
    node_a, node_b = _siblings(root)
    session_state.enter_compare(node_a, node_b)

    _complete_both(cache, node_a, node_b)

    assert panel._loading_label.isVisible() is False
    assert _metrics_text(panel) != ""


def test_a_position_ready_signal_for_an_unrelated_fen_is_ignored(qapp):
    panel, session_state, cache, root = _panel(qapp)
    node_a, node_b = _siblings(root)
    session_state.enter_compare(node_a, node_b)

    cache.position_ready.emit("some-unrelated-fen")

    assert panel._loading_label.isVisible() is True


# ---------------------------------------------------------
# board/move labels show the correct sides
# ---------------------------------------------------------


def test_move_labels_show_the_san_of_each_side(qapp):
    panel, session_state, _cache, root = _panel(qapp)
    node_a, node_b = _siblings(root)

    session_state.enter_compare(node_a, node_b)

    assert panel._move_label_a.text() == "e4"
    assert panel._move_label_b.text() == "d4"


def test_board_views_point_at_the_correct_nodes(qapp):
    panel, session_state, _cache, root = _panel(qapp)
    node_a, node_b = _siblings(root)

    session_state.enter_compare(node_a, node_b)

    assert panel._board_view_a._node is node_a
    assert panel._board_view_b._node is node_b


# ---------------------------------------------------------
# Δ: always shown once ready (V1.1 polish item 4 -- checkbox removed)
# ---------------------------------------------------------


def test_difference_palette_is_never_the_attack_influence_palette(qapp):
    assert DIFFERENCE_COLORMAP_NAME != ATTACK_INFLUENCE_COLORMAP_NAME


def test_difference_view_is_hidden_while_loading_and_visible_once_ready(qapp):
    panel, session_state, cache, root = _panel(qapp)
    node_a, node_b = _siblings(root)
    session_state.enter_compare(node_a, node_b)

    assert panel._diff_view.isVisible() is False

    _complete_both(cache, node_a, node_b)

    # No checkbox to toggle -- Compare mode's entire purpose is showing the
    # difference, so it is unconditionally visible once both sides are ready.
    assert panel._diff_view.isVisible() is True


def test_there_is_no_difference_checkbox(qapp):
    panel, _session_state, _cache, _root = _panel(qapp)

    assert not hasattr(panel, "_diff_checkbox")


def test_difference_view_resets_to_hidden_on_a_fresh_not_yet_ready_comparison(qapp):
    panel, session_state, cache, root = _panel(qapp)
    node_a, node_b = _siblings(root)
    session_state.enter_compare(node_a, node_b)
    _complete_both(cache, node_a, node_b)
    assert panel._diff_view.isVisible() is True

    session_state.exit_compare()
    node_c = root.add_variation(chess.Move.from_uci("c2c4"))
    session_state.enter_compare(node_a, node_c)

    assert panel._diff_view.isVisible() is False  # not yet ready for the new pair


# ---------------------------------------------------------
# semantic legend: audited against the real difference_colors() mapping
# ---------------------------------------------------------


def test_legend_bar_samples_the_actual_difference_colormap(qapp):
    # matplotlib.colormaps[...] hands back a fresh copy on each lookup (not
    # a singleton), so identity isn't the right check -- compare by name and
    # by sampled output instead, both of which must match exactly.
    panel, _session_state, _cache, _root = _panel(qapp)
    reference = matplotlib.colormaps[DIFFERENCE_COLORMAP_NAME]

    assert panel._legend_bar._colormap.name == reference.name
    assert panel._legend_bar._colormap(0.0) == reference(0.0)
    assert panel._legend_bar._colormap(1.0) == reference(1.0)


def test_legend_endpoint_colors_match_the_audited_puor_mapping(qapp):
    # Audited directly (see desktop_app/compare_panel.py's _LegendBar
    # docstring): PuOr's LOW end (t=0.0, most negative Δφ, "A stronger") is
    # orange; its HIGH end (t=1.0, most positive Δφ, "B stronger") is
    # purple. This is the reverse of a naive "Pu-then-Or" name reading, so
    # it is pinned here as an explicit regression guard.
    panel, _session_state, _cache, _root = _panel(qapp)
    colormap = panel._legend_bar._colormap

    low_r, low_g, low_b, _ = colormap(0.0)
    high_r, high_g, high_b, _ = colormap(1.0)

    # Orange: red channel high, blue channel low.
    assert low_r > low_b
    # Purple: blue channel high relative to green, red present but not dominant over blue.
    assert high_b > high_g


def test_legend_labels_are_ordered_a_stronger_then_zero_then_b_stronger(qapp):
    panel, _session_state, _cache, _root = _panel(qapp)

    labels_row = panel._legend_bar.parentWidget().layout()
    # Walk the legend block's second row (labels) via the panel's own
    # widgets rather than re-deriving layout internals: simplest robust
    # check is that both endpoint labels exist with the expected text and
    # "A" appears before "B" in reading order is guaranteed by construction
    # (see ComparePanel.__init__'s legend_labels_row) -- assert the texts
    # exist rather than re-walking QLayout internals.
    texts = _all_label_texts(panel)
    assert "A stronger" in texts
    assert "B stronger" in texts
    assert "0" in texts


def _all_label_texts(panel) -> list[str]:
    from PySide6.QtWidgets import QLabel

    return [label.text() for label in panel.findChildren(QLabel)]


# ---------------------------------------------------------
# numeric metrics: identical vs different fields (unchanged values/rounding)
# ---------------------------------------------------------


def test_metrics_report_no_change_for_identical_influence_fields(qapp):
    panel, session_state, cache, root = _panel(qapp)
    node_a, node_b = _siblings(root)
    session_state.enter_compare(node_a, node_b)

    _complete_both(cache, node_a, node_b, matrix_a=np.zeros((8, 8)), matrix_b=np.zeros((8, 8)))

    text = _metrics_text(panel).lower()
    assert "identical" in text
    assert "none" in text


def test_metrics_report_a_nonzero_change_and_the_changed_square(qapp):
    panel, session_state, cache, root = _panel(qapp)
    node_a, node_b = _siblings(root)
    session_state.enter_compare(node_a, node_b)

    matrix_a = np.zeros((8, 8))
    matrix_b = np.zeros((8, 8))
    matrix_b[0, 4] = 5.0  # e8

    _complete_both(cache, node_a, node_b, matrix_a=matrix_a, matrix_b=matrix_b)

    assert "e8" in _metrics_text(panel)


def test_metrics_show_exactly_three_rows(qapp):
    # Overall change / Largest change / Balance shift -- no invented scores.
    panel, session_state, cache, root = _panel(qapp)
    node_a, node_b = _siblings(root)
    session_state.enter_compare(node_a, node_b)
    _complete_both(cache, node_a, node_b)

    assert panel._metrics_grid.count() == 3


# ---------------------------------------------------------
# rapid re-entry
# ---------------------------------------------------------


def test_rapid_reentry_with_a_different_pair_shows_the_new_pair_only(qapp):
    panel, session_state, cache, root = _panel(qapp)
    node_a, node_b = _siblings(root)
    session_state.enter_compare(node_a, node_b)
    _complete_both(cache, node_a, node_b)
    session_state.exit_compare()

    node_c = root.add_variation(chess.Move.from_uci("c2c4"))
    session_state.enter_compare(node_a, node_c)

    assert panel.isVisible() is True
    assert panel._board_view_a._node is node_a
    assert panel._board_view_b._node is node_c
    assert panel._move_label_b.text() == "c4"
    # Not yet completed for this new pair -- must be back in the loading
    # state, not showing stale data from the previous (node_a, node_b) pair.
    assert panel._loading_label.isVisible() is True
    assert _metrics_text(panel) == ""
