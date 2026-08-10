from __future__ import annotations

import chess
import chess.pgn
import numpy as np
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QGridLayout, QLabel, QMainWindow, QWidget

from audio.backend import AudioBackend, SoundDeviceBackend
from audio.engine import AudioEngine
from audio.voices import build_default_voice_registry
from desktop_app.audio_controller import AudioController
from desktop_app.audio_mixer_panel import AudioMixerPanel
from desktop_app.board_panel import BoardPanel
from desktop_app.gl_canvas import MathCanvas
from desktop_app.layer_panel import LayerPanel
from desktop_app.layer_registry import LayerRegistry
from desktop_app.layers.attack_influence_layer import ATTACK_INFLUENCE_LAYER
from desktop_app.layers.critical_points_layer import CRITICAL_POINTS_LAYER
from desktop_app.layers.equipotential_layer import EQUIPOTENTIAL_LAYER
from desktop_app.layers.gradient_layer import GRADIENT_LAYER
from desktop_app.layers.morse_smale_layer import MORSE_SMALE_LAYER
from desktop_app.layers.ridge_valley_layer import RIDGE_VALLEY_LAYER
from desktop_app.position_cache import CacheEntryState, PositionCache
from desktop_app.scrub_controller import ScrubController
from desktop_app.session_state import SessionState
from desktop_app.timeline_panel import TimelinePanel
from desktop_app.transition_controller import TransitionController

# Registration order == draw order (docs/interactive_ui.md Part 5's "Layer
# Registry"; deterministic draw ordering is a Phase 5c requirement). Attack
# Influence (background) first, Critical Points (markers) always last/on
# top -- mirrors the exact zorder values the existing matplotlib modules
# already use where given (equipotential=3, ridge/valley=4,
# Morse-Smale cells=4, critical points=6 highest), not an invented ordering.
_LAYERS_IN_DRAW_ORDER = (
    ATTACK_INFLUENCE_LAYER,
    EQUIPOTENTIAL_LAYER,
    GRADIENT_LAYER,
    RIDGE_VALLEY_LAYER,
    MORSE_SMALE_LAYER,
    CRITICAL_POINTS_LAYER,
)

# Attack Influence draws through MathCanvas's fixed 8x8-quad overlay path
# (set_overlay_colors/set_overlay_enabled/set_overlay_opacity), not through
# the generic per-layer geometry path the other five layers use -- see the
# class docstring. This id is what tells the dispatch below which path a
# given layer's SessionState change belongs to.
_OVERLAY_LAYER_ID = ATTACK_INFLUENCE_LAYER.id


class MainWindow(QMainWindow):
    """
    Phase 5c: all six `docs/interactive_ui.md` Part 5 layers registered and
    rendered through `LayerRegistry`, sharing one `FullPositionAnalysis` per
    position (`desktop_app/full_position_analysis.py`) computed once by
    `PositionCache`, never once per layer.

    Render dispatch is by what a layer's `renderer` returns, not by layer
    name: Attack Influence keeps Phase 5a/5b's fixed 8x8-quad
    `MathCanvas.set_overlay_colors` path unchanged (its renderer still
    returns a plain `(VERTEX_COUNT, 4)` array); the five new layers return
    `list[LayerGeometry]` and go through `MathCanvas.set_layer_geometry`
    instead. No `if layer.id == "attack_influence"` branch exists anywhere
    here or in `gl_canvas.py` -- a future layer is routed correctly purely by
    which shape its own `renderer` produces.

    Layer-strip UI (Phase 5d, part 1): `self.layer_panel` (`desktop_app/
    layer_panel.py`) iterates the same `LayerRegistry` to build its
    checkbox/opacity rows and drives visibility/opacity exclusively through
    `SessionState` -- it never calls into `self.canvas` directly. Every
    change lands here first, through the single `_on_layer_state_changed`
    handler below, which is the only code that ever calls `MathCanvas`'s
    `set_layer_enabled`/`set_layer_opacity` (or their overlay equivalents).

    Move-to-move animation (Phase 5d, part 2): `self.transition_controller`
    (`desktop_app/transition_controller.py`) is now the sole caller of
    `_render_all_layers` -- both `_on_current_node_changed` and
    `_on_position_ready` hand it the target `fen` instead of rendering
    directly, and it either animates to it or settles instantly (no prior
    position, or already there), calling `_render_all_layers` back only
    once settled. `_render_all_layers` itself is unchanged.

    Timeline / history navigation (Phase 5e, discrete navigation): the
    discrete part of `self.timeline_panel` (`desktop_app/timeline_panel.py`)
    calls only `SessionState` navigation methods
    (`go_to_start`/`undo`/`redo`/`go_to_end`/`set_current_node`) and never
    touches `self.canvas`, `self.board_panel`, `self.position_cache`, or
    `self.transition_controller` directly -- discrete Timeline-driven
    navigation reaches all of those exactly the way a move/undo/redo already
    does, through `current_node_changed` and the two handlers below. No
    change to either handler was needed for this.

    Continuous Timeline scrubbing (Phase 5e.2): `self.scrub_controller`
    (`desktop_app/scrub_controller.py`) is a second, independent consumer of
    `self.canvas`/`self.position_cache`, driven by `TimelinePanel`'s scrub
    strip rather than by `current_node_changed`. It never mutates
    `SessionState.current_node` mid-drag -- only once, on release, via the
    same `set_current_node` call any other Timeline click already makes, at
    which point the two handlers below take back over exactly as usual.
    `self.timeline_panel` is constructed with `board_panel`/
    `transition_controller`/`scrub_controller` so its scrub strip can reach
    them directly (the one narrow exception to "Timeline never touches those
    objects directly" above, scoped entirely to the transient, non-
    committing scrub gesture).

    Live audio (Phase 5f.3 move-driven, Phase 5f.4 scrub-driven):
    `self.audio_engine` (`audio/engine.py`) is the real-time runtime;
    `self.audio_controller` (`desktop_app/audio_controller.py`) is its sole
    bridge to `SessionState` and to the scrub gesture, passed to
    `self.timeline_panel` alongside `scrub_controller` so its strip can
    forward the exact same `ScrubPosition` values to both. Like
    `scrub_controller`, `audio_controller` never mutates `SessionState.
    current_node` -- committed moves and their capture/check accents are
    published entirely through the ordinary `current_node_changed` path,
    unchanged by scrubbing existing.

    Live Audio mixer (Phase 5f.5): `self.audio_mixer_panel` (`desktop_app/
    audio_mixer_panel.py`) is the compact ON/OFF, master gain, and per-
    voice mute/solo strip. It never touches `self.audio_engine` beyond
    `is_running`/`is_available`/`start`/`stop` (the engine's own existing
    lifecycle contract -- OFF pauses the stream, it does not zero gain or
    tear anything down) and never touches `self.audio_controller` beyond
    its mixer-state API (`master_gain`/`voice_mute`/`voice_solo` and their
    setters) -- see both classes' own docstrings.

    `audio_backend` is an injection point (defaults to a real
    `SoundDeviceBackend`), added so tests that construct a `MainWindow` for
    unrelated reasons (canvas rendering, timeline navigation, ...) don't
    each also open a real hardware audio stream -- `tests/conftest.py`
    defaults it to a `FakeAudioBackend` session-wide for exactly that
    reason, the same precedent already established there for
    `PositionCache`'s executor. Production code (this class's own
    entry point) never passes it, so the real backend remains the default
    outside tests.
    """

    def __init__(
        self,
        initial_board: chess.Board | None = None,
        *,
        audio_backend: AudioBackend | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("VectorChess")

        game_root = chess.pgn.Game()
        if initial_board is not None:
            game_root.setup(initial_board)

        self.session_state = SessionState(game_root)
        self.position_cache = PositionCache()
        self.layer_registry = LayerRegistry()
        for layer in _LAYERS_IN_DRAW_ORDER:
            self.layer_registry.register(layer)

        self.board_panel = BoardPanel(self.session_state, self)
        self.canvas = MathCanvas(self)
        # Attack Influence is included here too -- see the class docstring:
        # it never actually populates `_layer_gpu_buffers` under this
        # mechanism (its renderer output goes through `set_overlay_colors`
        # instead), so the per-layer draw loop in `gl_canvas.paintGL` finds
        # nothing stored for it and silently skips it. No name filtering
        # needed to keep the two mechanisms from double-drawing it.
        self.canvas.set_layer_order([layer.id for layer in self.layer_registry.all()])
        self.layer_panel = LayerPanel(self.session_state, self.layer_registry, self)
        self.transition_controller = TransitionController(
            canvas=self.canvas,
            session_state=self.session_state,
            position_cache=self.position_cache,
            on_settled=self._render_all_layers,
        )
        self.scrub_controller = ScrubController(canvas=self.canvas, position_cache=self.position_cache)
        # Phase 5f.3/5f.4: live audio. AudioEngine owns the real-time output
        # stream (mono, sounddevice-backed -- see audio/engine.py; silently
        # falls back to an unavailable/no-op state if no device exists, per
        # its own start() contract, rather than raising here). AudioController
        # is the sole bridge from SessionState/scrub gestures to it -- no
        # other object in this window ever calls into audio/ directly.
        self.voice_registry = build_default_voice_registry()
        self.audio_engine = AudioEngine(audio_backend or SoundDeviceBackend(), self.voice_registry)
        self.audio_engine.start()
        self.audio_controller = AudioController(self.session_state, self.audio_engine)
        self.timeline_panel = TimelinePanel(
            self.session_state,
            self,
            board_panel=self.board_panel,
            transition_controller=self.transition_controller,
            scrub_controller=self.scrub_controller,
            audio_controller=self.audio_controller,
        )
        # Phase 5f.5: the Live Audio mixer strip -- reads/writes only
        # through audio_controller/audio_engine (see AudioMixerPanel's own
        # docstring), constructed after both exist.
        self.audio_mixer_panel = AudioMixerPanel(self.audio_controller, self.audio_engine, self.voice_registry, self)

        central = QWidget(self)
        layout = QGridLayout(central)
        # Requested layout: board spans both rows on the left; the canvas is
        # the top-right "protagonist" panel (Part 1); the layer strip sits
        # below it, right column only -- matching the frozen mockup's panel
        # roles (board = input, canvas = biggest panel, layer strip = its
        # own controls beneath it), not a redesign of Part 2's fuller
        # top-bar/inspector layout, which remains out of this milestone's
        # scope. The timeline panel is the one piece of Part 2's bottom
        # transport strip built so far (Phase 5e) -- it spans both columns,
        # beneath the board and canvas/layer-strip columns alike.
        header = QLabel("VectorChess")
        header.setStyleSheet("font-weight: bold; font-size: 16px; padding: 4px;")
        layout.addWidget(header, 0, 0, 1, 2)
        layout.addWidget(self.board_panel, 1, 0, 2, 1)
        layout.addWidget(self.canvas, 1, 1)
        layout.addWidget(self.layer_panel, 2, 1)
        layout.addWidget(self.timeline_panel, 3, 0, 1, 2)
        # Phase 5f.5: a new row beneath the timeline, spanning both
        # columns -- purely additive, no change to any existing widget's
        # row/column/span above. Visually secondary and compact (fixed
        # row stretch, like the timeline row above it), not a layout
        # redesign.
        layout.addWidget(self.audio_mixer_panel, 4, 0, 1, 2)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(1, 3)
        layout.setRowStretch(2, 1)
        layout.setRowStretch(3, 0)
        layout.setRowStretch(4, 0)
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.position_cache.position_ready.connect(self._on_position_ready)
        self.session_state.current_node_changed.connect(self._on_current_node_changed)
        self.session_state.layer_state_changed.connect(self._on_layer_state_changed)
        self.position_cache.request(self.session_state.current_node.board())

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Stability investigation (see `desktop_app/position_cache.py`'s
        "Shutdown contract" docstring): blocks until `self.position_cache`'s
        worker thread(s) have actually exited, *before* Qt starts tearing
        down this window and the `PositionCache` it owns -- closing this gap
        is what makes it impossible for a still-running analysis worker to
        try to publish a result into a `PositionCache`/`MainWindow` that's
        already being destroyed.

        `self.scrub_controller.shutdown()` runs first: it unsubscribes from
        `position_cache.position_ready` so a scrub still mid-drag when the
        window closes can't have a late-arriving cache result try to push a
        frame into `self.canvas` while this window is mid-teardown. Ordered
        before `position_cache.shutdown()` deliberately -- the controller
        disconnects itself while the cache is still fully alive, rather than
        racing the cache's own shutdown.

        `self.audio_controller.shutdown()` runs next, before `position_cache.
        shutdown()` too: it disconnects from `session_state.current_node_
        changed`, clears any in-progress scrub-preview state, and joins
        `AudioEngine`'s real-time thread -- so no late-arriving navigation or
        scrub update can try to publish into an engine that's mid-teardown,
        and no audio thread is left alive once this window closes.
        """
        self.scrub_controller.shutdown()
        self.audio_controller.shutdown()
        self.position_cache.shutdown()
        super().closeEvent(event)

    def _on_current_node_changed(self, node: chess.pgn.GameNode) -> None:
        fen = self.position_cache.request(node.board())
        # request() is a no-op (by design) for a position already
        # COMPUTING/READY -- e.g. undo/redo revisiting an earlier position --
        # so no position_ready signal will follow for an already-READY
        # entry. Start the transition immediately in that case (animating
        # from wherever the canvas last settled) rather than waiting forever
        # for a signal that was never going to fire.
        if self.position_cache.get(fen).state == CacheEntryState.READY:
            self.transition_controller.start_transition(fen)

    def _on_position_ready(self, fen: str) -> None:
        # Guards against a stale computation for a position that is no
        # longer current -- e.g. the user moved on again before an earlier
        # request finished.
        if fen != self.session_state.current_node.board().board_fen():
            return
        self.transition_controller.start_transition(fen)

    def _on_layer_state_changed(self, layer_id: str) -> None:
        """
        Fires for both a visibility toggle and an opacity change -- Part
        4.1's table treats them as one `layer_state` slice, and
        `SessionState` emits the same signal for either (see
        `SessionState.set_layer_opacity`'s docstring).

        Never touches `PositionCache` or any layer's `data_source`/
        `renderer` -- only `MathCanvas`'s per-frame draw flags/opacity for
        already-uploaded geometry. This is what makes "layer visibility/
        opacity changes must not trigger mathematical recomputation" true
        by construction: this handler's only calls are to
        `set_layer_enabled`/`set_layer_opacity` (or their overlay
        equivalents for Attack Influence, see `_OVERLAY_LAYER_ID`).
        """
        visible = self.session_state.layer_visible(layer_id)
        opacity = self.session_state.layer_opacity(layer_id)
        if layer_id == _OVERLAY_LAYER_ID:
            self.canvas.set_overlay_enabled(visible)
            self.canvas.set_overlay_opacity(opacity)
        else:
            self.canvas.set_layer_enabled(layer_id, visible)
            self.canvas.set_layer_opacity(layer_id, opacity)

    def _render_all_layers(self, fen: str) -> None:
        """
        Renders every registered layer's geometry for `fen` -- not only the
        currently-visible ones -- so an already-uploaded, current-position
        buffer is always available the instant a layer is toggled back on
        (via `_on_layer_state_changed` alone, no recomputation). This
        never re-touches the analysis pipeline: `entry.analysis` is read
        once from the cache and passed to every layer's cheap
        `data_source`/`renderer` pair -- "compute once, reuse six times."
        """
        entry = self.position_cache.get(fen)

        for layer in self.layer_registry.all():
            frame = layer.data_source(entry)
            rendered = layer.renderer(frame)
            visible = self.session_state.layer_visible(layer.id)
            opacity = self.session_state.layer_opacity(layer.id)

            if isinstance(rendered, np.ndarray):
                self.canvas.set_overlay_colors(rendered)
                self.canvas.set_overlay_enabled(visible)
                self.canvas.set_overlay_opacity(opacity)
            else:
                self.canvas.set_layer_geometry(layer.id, rendered)
                self.canvas.set_layer_enabled(layer.id, visible)
                self.canvas.set_layer_opacity(layer.id, opacity)
