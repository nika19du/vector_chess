from __future__ import annotations

from typing import Mapping

import chess
import chess.pgn
from PySide6.QtCore import QObject

from analysis.attack_influence import build_attack_influence_field
from analysis.dynamics import analyze_dynamics
from audio.engine import AccentTrigger, AudioEngine
from audio.live_state import interpolate_sonification_state, sonification_state_from_mapping
from audio.mapping import build_audio_mapping
from chess_engine.analyzer import analyze_position
from chess_engine.models import DynamicsAnalysis, MoveAnalysis, MoveDetails
from desktop_app.scrub import ScrubBounds, ScrubPosition
from desktop_app.session_state import SessionState


def _move_details_for_node(node: chess.pgn.GameNode) -> MoveDetails:
    """
    Reconstructs the same MoveDetails chess_engine.moves.execute_move would
    have produced for this node's move, without replaying/mutating any
    board -- node.parent.board() is already the exact pre-move position,
    node.board() the exact post-move position, both read-only.
    """

    parent_board = node.parent.board()
    move = node.move
    moved_piece = parent_board.piece_at(move.from_square)

    return MoveDetails(
        move=move.uci(),
        piece_name=chess.piece_name(moved_piece.piece_type),
        color="white" if moved_piece.color == chess.WHITE else "black",
        from_square=chess.square_name(move.from_square),
        to_square=chess.square_name(move.to_square),
        is_capture=parent_board.is_capture(move),
        is_check=node.board().is_check(),
    )


def _segment_key_for_node(node: chess.pgn.GameNode) -> tuple[str, str]:
    return (node.parent.board().board_fen(), node.board().board_fen())


class AudioController(QObject):
    """
    Translates committed SessionState navigation into live audio -- the
    smallest desktop integration that makes chess moves drive AudioEngine
    (Phase 5f.3). Depends only on SessionState and an already-constructed
    AudioEngine (dependency injection, matching ScrubController/
    TransitionController's existing style); does not construct or start the
    engine itself, and does not touch MainWindow, PositionCache, or any Qt
    widget.

    GameNode analysis strategy: every quantity build_audio_mapping needs
    (MoveAnalysis, DynamicsAnalysis) is derived purely from the selected
    node and its ancestors -- never from "whatever was computed last" --
    so undo, redo, branch switching, revisiting an old node, and a distant
    Timeline jump are all handled by the same code path with no special
    cases. A small node-keyed cache (chess.pgn.GameNode identity, not FEN --
    two different nodes can reach the same FEN by transposition without
    being the same tree node) avoids recomputing analysis for a node visited
    twice, without depending on call order.

    Scrub preview (Phase 5f.4): `begin_scrub`/`update_scrub`/`end_scrub`/
    `cancel_scrub` mirror `ScrubController`'s own state-machine shape
    (idle -> active via begin -> idle again via end/cancel) and operate on
    the exact same `ScrubPosition`/`ScrubBounds`/`SessionState.active_path()`
    model -- no fake GameNodes, no fake FENs, never writes into
    `SessionState.current_node`. Unlike the visual `ScrubController`,
    updates are NOT deferred/coalesced via `QTimer.singleShot`: computing
    a `SonificationState` is a handful of cheap pure-function calls (no
    GPU upload, no canvas repaint), so every mousemove can publish
    directly and the audio callback's own "read only the latest published
    reference" semantics already provide the coalescing for free (see
    Phase 5f.1's design notes) -- introducing a second coalescing
    mechanism here would be unneeded complexity, not a missing one.
    Only the one position-derived signal (Attack Influence balance ->
    harmony) is continuously interpolated; every move-identity-derived
    signal (pitch, timbre, loudness, capture/check) is held fixed at the
    active segment's own values and never triggers an AccentTrigger during
    preview -- only a real committed navigation (via `_on_current_node_
    changed`, unchanged from Phase 5f.3) ever fires one.

    Mixer state ownership (Phase 5f.5): `SonificationState.master_gain`/
    `voice_mute`/`voice_solo` are per-publish parameters, not persisted
    anywhere by `audio/live_state.py` -- so this class is the one place
    that remembers "what the Mixer UI currently has set" and carries it
    into every `SonificationState` it builds, settled or scrub-preview
    alike (`_master_gain`/`_voice_mute`/`_voice_solo` below, read back
    through the `master_gain`/`voice_mute`/`voice_solo` properties).
    `docs/interactive_ui.md` Part 4.1's original design proposed a
    `SessionState.mixer_state` slice for this (mirroring `layer_state`);
    this phase places it here instead, matching the Mixer's actual
    intended flow ("Mixer UI -> AudioController/AudioEngine API ->
    authoritative runtime state") and this class's own existing style of
    owning its caches directly -- audio mixer settings are not chess
    state, and `SessionState` has never had a mixer_state field in the
    real 5f.1-5f.4a implementation. `set_master_gain`/`set_voice_mute`/
    `set_voice_solo` immediately re-publish whatever is currently
    audible (the settled node, or the active scrub preview) with the
    new setting applied -- never a fake commit, never a new trigger
    (see `_republish_current_state`).
    """

    def __init__(self, session_state: SessionState, audio_engine: AudioEngine) -> None:
        super().__init__()
        self._session_state = session_state
        self._audio_engine = audio_engine
        self._analysis_cache: dict[chess.pgn.GameNode, MoveAnalysis] = {}
        # None is a valid cached result (a node with no "previous" move) --
        # containment (`in`), not `.get(...) is not None`, distinguishes
        # "not yet computed" from "computed, and the answer is None".
        self._dynamics_cache: dict[chess.pgn.GameNode, DynamicsAnalysis | None] = {}
        # The whole game tree shares exactly one root; memoized lazily the
        # first time a scrub segment touches it (_analysis_for cannot --
        # the root has no move, so no MoveDetails/MoveAnalysis exists for
        # it -- but its Attack Influence balance is still a well-defined,
        # cheap-to-compute position property).
        self._root_balance: float | None = None
        self._scrub_bounds: ScrubBounds | None = None  # None == not scrubbing
        # The most recent update_scrub() position, so a Mixer change
        # mid-drag can re-publish the same preview position with the new
        # mixer setting applied instead of guessing/re-deriving it.
        # Reset to None alongside _scrub_bounds (begin/end/cancel_scrub).
        self._last_scrub_position: ScrubPosition | None = None
        self._master_gain: float = 1.0
        self._voice_mute: dict[str, bool] = {}
        self._voice_solo: dict[str, bool] = {}
        self._shut_down = False

        session_state.current_node_changed.connect(self._on_current_node_changed)
        self._connected_to_current_node_changed = True

    # -- mixer state (Phase 5f.5) ---------------------------------------

    @property
    def master_gain(self) -> float:
        return self._master_gain

    @property
    def voice_mute(self) -> Mapping[str, bool]:
        return dict(self._voice_mute)

    @property
    def voice_solo(self) -> Mapping[str, bool]:
        return dict(self._voice_solo)

    def set_master_gain(self, gain: float) -> None:
        if self._shut_down:
            return
        self._master_gain = gain
        self._republish_current_state()

    def set_voice_mute(self, voice_id: str, muted: bool) -> None:
        if self._shut_down:
            return
        self._voice_mute = {**self._voice_mute, voice_id: muted}
        self._republish_current_state()

    def set_voice_solo(self, voice_id: str, soloed: bool) -> None:
        if self._shut_down:
            return
        self._voice_solo = {**self._voice_solo, voice_id: soloed}
        self._republish_current_state()

    def _republish_current_state(self) -> None:
        """
        Re-publishes whatever is currently audible with the mixer
        settings just changed -- the settled current node, or (if a
        scrub gesture is active) the last previewed scrub position.
        Never fires a NoteTrigger or AccentTrigger, never touches
        SessionState, never recomputes analysis (both `_analysis_for`/
        `_dynamics_for` and the scrub path below hit their existing
        node-keyed caches): a mixer tweak is not a chess event.
        """

        if self.is_scrubbing and self._last_scrub_position is not None:
            self._publish_scrub_state(self._last_scrub_position)
            return

        node = self._session_state.current_node
        if node.parent is None:
            return  # root -- nothing published yet to re-publish
        self._publish_settled_state(node, fire_triggers=False)

    def _analysis_for(self, node: chess.pgn.GameNode) -> MoveAnalysis:
        cached = self._analysis_cache.get(node)
        if cached is not None:
            return cached

        analysis = analyze_position(node.board(), _move_details_for_node(node))
        self._analysis_cache[node] = analysis
        return analysis

    def _dynamics_for(self, node: chess.pgn.GameNode) -> DynamicsAnalysis | None:
        if node in self._dynamics_cache:
            return self._dynamics_cache[node]

        parent = node.parent
        if parent is None or parent.parent is None:
            # node is the first move of the game (or, defensively, the root
            # itself) -- no previous move's MoveAnalysis exists to compare
            # against, exactly like the offline MVP's first move.
            result = None
        else:
            previous_analysis = self._analysis_for(parent)
            current_analysis = self._analysis_for(node)
            result = analyze_dynamics(previous=previous_analysis, current=current_analysis)

        self._dynamics_cache[node] = result
        return result

    def _balance_for_node(self, node: chess.pgn.GameNode) -> float:
        if node.parent is None:
            if self._root_balance is None:
                self._root_balance = build_attack_influence_field(node.board()).balance
            return self._root_balance
        return self._analysis_for(node).attack_influence_field.balance

    def _on_current_node_changed(self, node: chess.pgn.GameNode) -> None:
        if self._shut_down:
            return
        if node.parent is None:
            # The root position itself has no move -- nothing to sonify.
            return

        self._publish_settled_state(node, fire_triggers=True)

    def _publish_settled_state(self, node: chess.pgn.GameNode, *, fire_triggers: bool) -> None:
        """
        Builds and publishes the settled SonificationState for `node`,
        with the current mixer settings applied. `fire_triggers` is
        False only for a Mixer-driven re-publish (`_republish_current_
        state`): a mute/solo/gain change re-sounds the current note
        with its new mixing, but must never restart its envelope or
        fire a fresh capture/check accent -- those only ever come from
        a real committed navigation (`fire_triggers=True`, the only
        caller being `_on_current_node_changed` itself).
        """

        analysis = self._analysis_for(node)
        dynamics = self._dynamics_for(node)
        mapping = build_audio_mapping(analysis, dynamics)

        state = sonification_state_from_mapping(
            mapping,
            _segment_key_for_node(node),
            master_gain=self._master_gain,
            voice_mute=self._voice_mute,
            voice_solo=self._voice_solo,
        )
        self._audio_engine.publish(state)

        if not fire_triggers:
            return

        # Phase 5f.4a: every committed move retriggers the Melody/Harmony
        # articulation envelopes -- a note attack, distinct from the
        # capture/check accent below (which only fires on some moves).
        self._audio_engine.push_note_trigger()

        if mapping.is_capture or mapping.is_check:
            if mapping.is_capture and mapping.is_check:
                kind = "capture+check"
            elif mapping.is_capture:
                kind = "capture"
            else:
                kind = "check"
            self._audio_engine.push_event(AccentTrigger(kind=kind, loudness=mapping.loudness))

    # -- scrub preview (Phase 5f.4) -------------------------------------

    @property
    def is_scrubbing(self) -> bool:
        return self._scrub_bounds is not None

    def begin_scrub(self, active_path: list[chess.pgn.GameNode]) -> None:
        """
        Snapshots `active_path` (expected to be `SessionState.active_path()`,
        taken by the caller at press time -- exactly like `ScrubController.
        begin`). The gating decision ("is a scrub allowed to start at all
        right now") is made once, by the caller, before this and
        `ScrubController.begin` are both invoked from the same call site --
        this class does not re-derive or second-guess that decision, which
        is exactly what guarantees visual and audio scrub can never disagree
        about whether a scrub is active.
        """

        if self._shut_down:
            return
        self._scrub_bounds = ScrubBounds(path=tuple(active_path))
        # Phase 5f.4a: while a scrub gesture is active, Melody/Harmony
        # stay on their original continuous-glide behavior rather than
        # the new envelope-gated one -- one morphing preview voice, no
        # note retriggers on mouse movement.
        self._audio_engine.set_scrub_active(True)

    def update_scrub(self, position: ScrubPosition) -> None:
        """
        Publishes a scrub-preview SonificationState for `position`. Safe
        no-op if scrub hasn't begun (or was since cancelled) -- mirrors
        `ScrubController.update`'s own "before begin()" contract. Never
        touches `SessionState.current_node`, never pushes an AccentTrigger:
        only a real committed navigation does that, via `_on_current_node_
        changed`, unchanged from Phase 5f.3.
        """

        if self._shut_down or self._scrub_bounds is None:
            return

        self._last_scrub_position = position
        self._publish_scrub_state(position)

    def _publish_scrub_state(self, position: ScrubPosition) -> None:
        """
        Builds and publishes the scrub-preview SonificationState for
        `position` against the current `_scrub_bounds`, with the current
        mixer settings applied. Shared by `update_scrub` (a real mouse
        move) and `_republish_current_state` (a Mixer change re-applying
        the same position) so both go through identical segment/balance
        derivation -- no separate "mixer-during-scrub" code path to drift
        out of sync with the real one.
        """

        path = self._scrub_bounds.path
        lower = path[position.path_index]
        # At the tip, there is no "next" node -- mirrors ScrubController.
        # _segment_fens's own from_fen==to_fen identity trick: t is always
        # 0.0 there (clamp_scrub_fraction's contract), so interpolating
        # `lower` against itself at t=0 reproduces exactly the settled
        # state for `lower` through the same code path every other segment
        # uses, with no special-cased branch.
        upper = path[position.path_index + 1] if position.path_index < len(path) - 1 else lower

        if upper.parent is None:
            # Scrubbing before any move exists at all (a single-node path,
            # root only) -- nothing to sonify, mirrors _on_current_node_
            # changed's own root-skip.
            return

        segment_mapping = build_audio_mapping(self._analysis_for(upper), self._dynamics_for(upper))
        balance_a = self._balance_for_node(lower)
        balance_b = self._balance_for_node(upper)
        segment_key = (lower.board().board_fen(), upper.board().board_fen())

        state = interpolate_sonification_state(
            segment_mapping,
            balance_a,
            balance_b,
            position.t,
            segment_key,
            master_gain=self._master_gain,
            voice_mute=self._voice_mute,
            voice_solo=self._voice_solo,
        )
        self._audio_engine.publish(state)

    def end_scrub(self) -> None:
        """
        Resets to idle. Publishes nothing itself -- the existing committed
        navigation path (`SessionState.set_current_node`, called by the
        caller immediately after this, exactly like `ScrubController.end`'s
        own contract) remains the sole authority for the settled state and
        any capture/check trigger, via the unchanged `_on_current_node_
        changed` handler.
        """

        self._scrub_bounds = None
        self._last_scrub_position = None
        # Flip back to envelope-gated committed/idle mode before the
        # caller's own SessionState.set_current_node call fires the note
        # trigger that starts the settled note's attack -- see
        # AudioEngine.set_scrub_active's own docstring for why ordering
        # matters here.
        self._audio_engine.set_scrub_active(False)

    def cancel_scrub(self) -> None:
        """
        Resets to idle without a commit -- for an externally-driven
        `current_node_changed` arriving mid-drag (branch switch, keyboard
        navigation, etc.), mirroring `ScrubController.cancel`'s own
        contract and trigger. Any further `update_scrub` call becomes a
        safe no-op immediately (no queued/deferred work exists to cancel --
        see this class's docstring on why updates aren't coalesced via
        QTimer -- so there is nothing else that could let a stale scrub
        publication escape afterward).
        """

        self._scrub_bounds = None
        self._last_scrub_position = None
        self._audio_engine.set_scrub_active(False)

    def shutdown(self) -> None:
        """Idempotent: disconnects from SessionState, stops publishing, and
        shuts down the owned AudioEngine exactly once."""

        if self._shut_down:
            return
        self._shut_down = True
        self._scrub_bounds = None

        if self._connected_to_current_node_changed:
            self._session_state.current_node_changed.disconnect(self._on_current_node_changed)
            self._connected_to_current_node_changed = False

        self._audio_engine.shutdown()
