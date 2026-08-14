import chess
import chess.pgn

from desktop_app.full_position_analysis import build_full_position_analysis
from desktop_app.gl_canvas import MathCanvas
from desktop_app.layers._lerp import ease_in_out
from desktop_app.layers.attack_influence_layer import (
    build_attack_influence_frame,
    interpolate_attack_influence_frame,
    render_attack_influence_frame,
)
from desktop_app.position_cache import CacheEntry, CacheEntryState, PositionCache
from desktop_app.scrub import ScrubPosition
from desktop_app.scrub_controller import ScrubController


def _chain(moves: list[str]) -> list[chess.pgn.GameNode]:
    root = chess.pgn.Game()
    nodes = [root]
    node = root
    for uci in moves:
        node = node.add_variation(chess.Move.from_uci(uci))
        nodes.append(node)
    return nodes


def _controller() -> tuple[ScrubController, PositionCache, MathCanvas]:
    cache = PositionCache()
    canvas = MathCanvas()
    controller = ScrubController(canvas=canvas, position_cache=cache)
    return controller, cache, canvas


def _request_all(cache: PositionCache, nodes: list[chess.pgn.GameNode]) -> None:
    for node in nodes:
        cache.request(node.board())


# ---------------------------------------------------------
# exact endpoint behavior
# ---------------------------------------------------------


def test_frame_at_a_node_index_with_t_zero_matches_that_node_exactly(qapp):
    controller, cache, canvas = _controller()
    nodes = _chain(["e2e4", "e7e5"])
    _request_all(cache, nodes)
    controller.begin(nodes)

    controller.update(ScrubPosition(path_index=0, t=0.0))
    controller._run_scheduled_scrub_frame()

    entry = cache.get(nodes[0].board().board_fen())
    expected = render_attack_influence_frame(build_attack_influence_frame(entry))
    assert (canvas._overlay_colors == expected).all()


def test_frame_at_the_next_node_index_with_t_zero_matches_that_node_exactly(qapp):
    # Landing exactly on path_index=1 (t=0) reproduces node[1]'s exact frame
    # -- the same guarantee as "t=1 of the previous segment", approached
    # from the other side (clamp_scrub_fraction never yields t==1.0 mid-segment).
    controller, cache, canvas = _controller()
    nodes = _chain(["e2e4", "e7e5"])
    _request_all(cache, nodes)
    controller.begin(nodes)

    controller.update(ScrubPosition(path_index=1, t=0.0))
    controller._run_scheduled_scrub_frame()

    entry = cache.get(nodes[1].board().board_fen())
    expected = render_attack_influence_frame(build_attack_influence_frame(entry))
    assert (canvas._overlay_colors == expected).all()


def test_frame_at_the_branch_tip_matches_the_tip_exactly(qapp):
    controller, cache, canvas = _controller()
    nodes = _chain(["e2e4", "e7e5", "g1f3"])
    _request_all(cache, nodes)
    controller.begin(nodes)
    tip_index = len(nodes) - 1

    controller.update(ScrubPosition(path_index=tip_index, t=0.0))
    controller._run_scheduled_scrub_frame()

    entry = cache.get(nodes[tip_index].board().board_fen())
    expected = render_attack_influence_frame(build_attack_influence_frame(entry))
    assert (canvas._overlay_colors == expected).all()


def test_midpoint_frame_differs_from_both_endpoints(qapp):
    controller, cache, canvas = _controller()
    nodes = _chain(["e2e4", "e7e5"])
    _request_all(cache, nodes)
    controller.begin(nodes)

    controller.update(ScrubPosition(path_index=0, t=0.5))
    controller._run_scheduled_scrub_frame()
    midpoint = canvas._overlay_colors.copy()

    entry_a = cache.get(nodes[0].board().board_fen())
    entry_b = cache.get(nodes[1].board().board_fen())
    frame_a = render_attack_influence_frame(build_attack_influence_frame(entry_a))
    frame_b = render_attack_influence_frame(build_attack_influence_frame(entry_b))

    assert not (midpoint == frame_a).all()
    assert not (midpoint == frame_b).all()


# ---------------------------------------------------------
# raw t, never eased -- a deliberate deviation from TransitionController
# ---------------------------------------------------------


def test_scrub_uses_raw_t_not_eased_t(qapp):
    controller, cache, canvas = _controller()
    nodes = _chain(["e2e4", "e7e5"])
    _request_all(cache, nodes)
    controller.begin(nodes)

    raw_t = 0.25
    assert ease_in_out(raw_t) != raw_t  # sanity: smoothstep genuinely differs from raw at 0.25

    controller.update(ScrubPosition(path_index=0, t=raw_t))
    controller._run_scheduled_scrub_frame()

    entry_a = cache.get(nodes[0].board().board_fen())
    entry_b = cache.get(nodes[1].board().board_fen())
    expected_raw = render_attack_influence_frame(interpolate_attack_influence_frame(entry_a, entry_b, raw_t))
    expected_eased = render_attack_influence_frame(
        interpolate_attack_influence_frame(entry_a, entry_b, ease_in_out(raw_t))
    )

    assert (canvas._overlay_colors == expected_raw).all()
    assert not (canvas._overlay_colors == expected_eased).all()


# ---------------------------------------------------------
# rapid scrubbing / reversal / coalescing
# ---------------------------------------------------------


def test_rapid_scrubbing_and_reversal_renders_only_the_latest_position(qapp):
    controller, cache, canvas = _controller()
    nodes = _chain(["e2e4", "e7e5"])
    _request_all(cache, nodes)
    controller.begin(nodes)

    for t in (0.1, 0.9, 0.2, 0.8, 0.05, 0.95, 0.3):
        controller.update(ScrubPosition(path_index=0, t=t))
    controller._run_scheduled_scrub_frame()  # only one drain -- proves coalescing, not per-update rendering

    entry_a = cache.get(nodes[0].board().board_fen())
    entry_b = cache.get(nodes[1].board().board_fen())
    expected = render_attack_influence_frame(interpolate_attack_influence_frame(entry_a, entry_b, 0.3))
    assert (canvas._overlay_colors == expected).all()


def test_coalescing_collapses_rapid_updates_into_one_canvas_write(qapp, qtbot):
    controller, cache, canvas = _controller()
    nodes = _chain(["e2e4", "e7e5"])
    _request_all(cache, nodes)
    controller.begin(nodes)

    call_count = {"n": 0}
    original = canvas.set_overlay_colors

    def counting_set_overlay_colors(colors):
        call_count["n"] += 1
        original(colors)

    canvas.set_overlay_colors = counting_set_overlay_colors

    for t in (0.1, 0.2, 0.3, 0.4, 0.5):
        controller.update(ScrubPosition(path_index=0, t=t))

    qtbot.wait(10)  # drain the single scheduled QTimer.singleShot

    assert call_count["n"] == 1


# ---------------------------------------------------------
# no recomputation, correspondence computed once per segment
# ---------------------------------------------------------


def test_no_analysis_recomputation_during_scrub(qapp):
    controller, cache, canvas = _controller()
    nodes = _chain(["e2e4", "e7e5"])
    _request_all(cache, nodes)  # both endpoints already READY
    controller.begin(nodes)

    call_count = {"n": 0}
    real_builder = cache._builder

    def counting_builder(board):
        call_count["n"] += 1
        return real_builder(board)

    cache._builder = counting_builder

    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        controller.update(ScrubPosition(path_index=0, t=t))
        controller._run_scheduled_scrub_frame()

    assert call_count["n"] == 0


def test_correspondence_computed_once_per_segment_not_per_frame(qapp):
    controller, cache, canvas = _controller()
    nodes = _chain(["e2e4", "e7e5"])
    _request_all(cache, nodes)
    controller.begin(nodes)

    call_count = {"n": 0}
    real_builder = controller.correspondence_cache._builder

    def counting_builder(entry_a, entry_b):
        call_count["n"] += 1
        return real_builder(entry_a, entry_b)

    controller.correspondence_cache._builder = counting_builder

    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        controller.update(ScrubPosition(path_index=0, t=t))
        controller._run_scheduled_scrub_frame()

    assert call_count["n"] == 1


# ---------------------------------------------------------
# cache miss during scrub: clamp to last-good frame, discard stale results
# ---------------------------------------------------------


def test_cache_miss_leaves_the_last_good_frame_untouched(qapp):
    controller, cache, canvas = _controller()
    nodes = _chain(["e2e4", "e7e5"])
    cache.request(nodes[0].board())
    cache.request(nodes[1].board())
    controller.begin(nodes)

    controller.update(ScrubPosition(path_index=0, t=0.4))
    controller._run_scheduled_scrub_frame()
    last_good = canvas._overlay_colors.copy()

    # Simulate the segment's upper endpoint still being computed (not READY).
    tip_fen = nodes[1].board().board_fen()
    cache._entries[tip_fen] = CacheEntry(state=CacheEntryState.COMPUTING)

    controller.update(ScrubPosition(path_index=0, t=0.8))
    controller._run_scheduled_scrub_frame()

    # No fabricated blend was written -- the canvas still shows the last
    # frame that was actually computed from two READY endpoints.
    assert (canvas._overlay_colors == last_good).all()


def test_stale_position_ready_for_an_abandoned_segment_is_discarded(qapp):
    controller, cache, canvas = _controller()
    nodes = _chain(["e2e4", "e7e5", "g1f3"])
    _request_all(cache, nodes)
    controller.begin(nodes)

    # Scrub into segment 0 (nodes[0]..nodes[1]) and render it.
    controller.update(ScrubPosition(path_index=0, t=0.5))
    controller._run_scheduled_scrub_frame()
    rendered_after_segment_0 = canvas._overlay_colors.copy()

    # The user has since moved on to segment 1 (nodes[1]..nodes[2]).
    controller.update(ScrubPosition(path_index=1, t=0.2))
    controller._run_scheduled_scrub_frame()

    # A stale position_ready for segment 0's endpoint arrives late -- must
    # not re-render (it belongs to an abandoned segment).
    call_count = {"n": 0}
    original = canvas.set_overlay_colors

    def counting_set_overlay_colors(colors):
        call_count["n"] += 1
        original(colors)

    canvas.set_overlay_colors = counting_set_overlay_colors
    controller._pending = ScrubPosition(path_index=1, t=0.2)  # re-affirm current segment
    controller._on_position_ready(nodes[0].board().board_fen())

    assert controller._frame_scheduled is False
    assert call_count["n"] == 0


def test_position_ready_for_the_current_segment_triggers_a_fresh_render(qapp):
    controller, cache, canvas = _controller()
    nodes = _chain(["e2e4", "e7e5"])
    cache.request(nodes[0].board())
    tip_fen = nodes[1].board().board_fen()
    cache._entries[tip_fen] = CacheEntry(state=CacheEntryState.COMPUTING)
    controller.begin(nodes)

    controller.update(ScrubPosition(path_index=0, t=0.6))
    controller._run_scheduled_scrub_frame()
    assert canvas._overlay_colors is None  # not ready yet -- nothing rendered at all so far

    # Position finishes computing; the real async path would call this via
    # PositionCache.position_ready -- invoked directly here for determinism.
    cache._entries[tip_fen] = CacheEntry(state=CacheEntryState.READY, analysis=build_full_position_analysis(nodes[1].board()))
    controller._on_position_ready(tip_fen)
    controller._run_scheduled_scrub_frame()

    assert canvas._overlay_colors is not None


# ---------------------------------------------------------
# no fake positions
# ---------------------------------------------------------


def test_only_real_path_fens_are_ever_requested_or_looked_up(qapp):
    controller, cache, canvas = _controller()
    nodes = _chain(["e2e4", "e7e5", "g1f3", "b8c6"])
    _request_all(cache, nodes)
    controller.begin(nodes)

    seen_fens = set()
    original_get = cache.get
    original_request = cache.request

    def spying_get(fen):
        seen_fens.add(fen)
        return original_get(fen)

    def spying_request(board):
        fen = original_request(board)
        seen_fens.add(fen)
        return fen

    cache.get = spying_get
    cache.request = spying_request

    for path_index in range(len(nodes)):
        for hundredth in range(0, 100, 10):
            controller.update(ScrubPosition(path_index=path_index, t=hundredth / 100.0))
            controller._run_scheduled_scrub_frame()

    real_fens = {node.board().board_fen() for node in nodes}
    assert seen_fens.issubset(real_fens)


# ---------------------------------------------------------
# begin / cancel / end / shutdown lifecycle
# ---------------------------------------------------------


def test_is_scrubbing_reflects_begin_and_cancel():
    controller, cache, canvas = _controller()
    nodes = _chain(["e2e4"])
    assert controller.is_scrubbing is False

    controller.begin(nodes)
    assert controller.is_scrubbing is True

    controller.cancel()
    assert controller.is_scrubbing is False


def test_end_returns_the_snapped_node_and_resets_to_idle():
    controller, cache, canvas = _controller()
    nodes = _chain(["e2e4"])
    controller.begin(nodes)

    node = controller.end(ScrubPosition(path_index=0, t=0.6))

    assert node is nodes[1]  # t=0.6 >= 0.5 -> right/next node, per the approved snap rule
    assert controller.is_scrubbing is False


def test_update_before_begin_is_a_safe_no_op():
    controller, cache, canvas = _controller()
    controller.update(ScrubPosition(path_index=0, t=0.5))  # never began -- must not raise
    assert controller.is_scrubbing is False


def test_shutdown_is_idempotent_and_resets_to_idle(qapp):
    controller, cache, canvas = _controller()
    nodes = _chain(["e2e4"])
    controller.begin(nodes)

    controller.shutdown()
    assert controller.is_scrubbing is False

    controller.shutdown()  # must not raise the second time
