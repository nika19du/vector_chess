from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from audio.engine import AudioEngine
from audio.voices import VoiceRegistry
from desktop_app.audio_controller import AudioController

MASTER_GAIN_SLIDER_MIN = 0
MASTER_GAIN_SLIDER_MAX = 100

UNAVAILABLE_STATUS_TEXT = "Live Audio unavailable (no audio device)"

# V7 (desktop workspace layout): the voice grid's QScrollArea (see
# __init__) is now sized from the grid's own real sizeHint -- header row +
# one row per registered voice, measured after the actual widgets are built,
# under whatever font metrics are really active -- rather than a hardcoded
# pixel constant. The V6a-era constant this replaced (40px) could not fit
# even one full row, let alone a header plus 5-6 voice rows, under any font
# metrics, offscreen or real; that mismatch was the direct, confirmed cause
# of a live screenshot showing every voice row cut off below the visible
# window. This margin is only slack added on top of the measured sizeHint
# (border/rounding headroom), not the primary sizing mechanism.
VOICE_GRID_SCROLL_MARGIN_PX = 8


class AudioMixerPanel(QWidget):
    """
    The "Live Audio" mixer strip (`docs/interactive_ui.md` Part 5's
    bottom mixer strip / this phase's compact ASCII mockup). Phase 5f.5.

    Iterates `VoiceRegistry` generically -- one Mute/Solo row per
    registered voice, in registration order -- exactly like
    `LayerPanel` iterates `LayerRegistry` (`docs/interactive_ui.md`
    Part 12's extensibility claim: a future sixth voice needs no change
    here). Inactive voices (Drone, Space -- `voice.active is False`,
    no synthesis path exists for them in `audio/engine.py`) get a row
    too, but dimmed and disabled, never a silently-missing control.

    This widget owns no audio truth of its own. Every control reads its
    initial value from, and every interaction writes back through,
    `AudioController`'s mixer API (`master_gain`/`voice_mute`/
    `voice_solo` properties and `set_master_gain`/`set_voice_mute`/
    `set_voice_solo`) or `AudioEngine`'s own lifecycle
    (`is_running`/`is_available`/`start`/`stop`) -- never a second,
    independent copy of gain/mute/solo/on-off state. Nothing else in
    this codebase currently writes those values except this panel, so
    there is no external-change signal to listen for yet (unlike
    `LayerPanel`'s `layer_state_changed` sync-back, needed because
    keyboard shortcuts can also change layer state); if a future writer
    is added, this panel would need the same kind of sync-back.

    Audio ON/OFF (Phase 5f.5, `AudioEngine`'s existing lifecycle
    contract): OFF calls `AudioEngine.stop()` -- a real pause of the
    PortAudio callback itself (not a master-gain-to-zero hack), which
    freezes oscillator phase and envelope progression exactly where
    they were rather than continuing to run silently. ON calls `start()`,
    which resumes the *same* already-opened stream (see `AudioEngine.
    start`'s own docstring) -- repeated ON/OFF cycling never reopens a
    new `sounddevice` stream and never loses `AudioController`'s own
    caches/mixer state, which live entirely outside `AudioEngine` and
    are untouched by stop/start. If a `start()` attempt ever fails
    (device genuinely unavailable), the toggle is disabled outright
    rather than left clickable -- this is what keeps a bad device from
    turning into an uncontrolled open-retry loop; recovering requires a
    fresh `AudioEngine` (e.g. relaunching the app), not a button in
    this panel.
    """

    def __init__(
        self,
        audio_controller: AudioController,
        audio_engine: AudioEngine,
        voice_registry: VoiceRegistry,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = audio_controller
        self._engine = audio_engine

        self._mute_checkboxes: dict[str, QCheckBox] = {}
        self._solo_checkboxes: dict[str, QCheckBox] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        # V5 (responsive layout): pure spacing trim between title/status/
        # master-row/grid -- no change to any control itself.
        layout.setSpacing(2)

        title = QLabel("Live Audio")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #b00020;")
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)

        top_row = QHBoxLayout()
        self._enabled_checkbox = QCheckBox("Audio")
        self._enabled_checkbox.setChecked(audio_engine.is_running)
        self._enabled_checkbox.toggled.connect(self._on_enabled_toggled)
        top_row.addWidget(self._enabled_checkbox)

        top_row.addWidget(QLabel("Master"))
        self._master_slider = QSlider(Qt.Orientation.Horizontal)
        self._master_slider.setRange(MASTER_GAIN_SLIDER_MIN, MASTER_GAIN_SLIDER_MAX)
        self._master_slider.setValue(round(audio_controller.master_gain * 100))
        self._master_slider.setFixedWidth(120)
        self._master_slider.setToolTip("Master gain")
        self._master_slider.valueChanged.connect(self._on_master_gain_changed)
        top_row.addWidget(self._master_slider, stretch=1)
        layout.addLayout(top_row)

        # V6a (canvas-collapse fix), retained under V7 with a corrected
        # height source: the voice grid lives inside a fixed-height scroll
        # area instead of directly in this widget's own layout, so this
        # widget's own height stays compact/bounded (this panel is a
        # tertiary strip, not meant to grow with window size) regardless of
        # how many voices are registered, while every row stays reachable
        # by scrolling if it ever doesn't fit. The fixed height itself is
        # set below, from the grid's real content, after all voice rows are
        # built.
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        grid_content = QWidget()
        grid = QGridLayout(grid_content)
        # V7 (desktop workspace layout): trims Qt's default ~6px inter-row
        # spacing down to 1px -- pure spacing, zero content change, same
        # category of fix as V5's margin trims elsewhere. Now that this
        # grid's height is measured from real content (VOICE_GRID_SCROLL_
        # MARGIN_PX, see above) rather than clipped by a too-small fixed
        # constant, its 6 rows (header + 5 voices) at default Qt spacing
        # were the single largest contributor to a measured 26px overflow
        # of MainWindow's true minimum height at 1280x720.
        grid.setVerticalSpacing(0)
        # V0 layout fix: with no column ever given a nonzero stretch, Qt's
        # default is to split any leftover width evenly across all 3
        # columns (name/Mute/Solo) -- since this panel spans the full
        # window width, that scattered Mute/Solo far apart from their voice
        # row on a wide window. Giving only the name column stretch keeps
        # Mute/Solo pinned at their natural checkbox width, right next to
        # the row they belong to.
        grid.setColumnStretch(0, 1)
        grid.addWidget(QLabel(""), 0, 0)
        mute_header = QLabel("Mute")
        mute_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        solo_header = QLabel("Solo")
        solo_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(mute_header, 0, 1)
        grid.addWidget(solo_header, 0, 2)

        for row_index, voice in enumerate(voice_registry, start=1):
            self._build_voice_row(grid, row_index, voice.voice_id, voice.label, voice.active)

        # Sized from the grid's own real sizeHint (header row + every
        # registered voice row, all now actually built) instead of a
        # hardcoded constant -- see VOICE_GRID_SCROLL_MARGIN_PX's comment.
        scroll_area.setFixedHeight(grid.sizeHint().height() + VOICE_GRID_SCROLL_MARGIN_PX)
        scroll_area.setWidget(grid_content)
        layout.addWidget(scroll_area)

        if not audio_engine.is_available:
            self._mark_unavailable()

    def _build_voice_row(self, grid: QGridLayout, row: int, voice_id: str, label: str, active: bool) -> None:
        name_label = QLabel(label)
        mute_checkbox = QCheckBox()
        solo_checkbox = QCheckBox()
        mute_checkbox.setToolTip(f"Mute {label}")
        solo_checkbox.setToolTip(f"Solo {label}")

        if active:
            mute_checkbox.setChecked(self._controller.voice_mute.get(voice_id, False))
            solo_checkbox.setChecked(self._controller.voice_solo.get(voice_id, False))
            mute_checkbox.toggled.connect(
                lambda checked, voice_id=voice_id: self._controller.set_voice_mute(voice_id, checked)
            )
            solo_checkbox.toggled.connect(
                lambda checked, voice_id=voice_id: self._controller.set_voice_solo(voice_id, checked)
            )
        else:
            # Drone/Space: visible, clearly inactive -- no synthesis path
            # exists for them yet (audio/voices.py), so their controls
            # are disabled rather than wired to a mute/solo that would
            # have nothing to affect.
            name_label.setStyleSheet("color: gray;")
            mute_checkbox.setEnabled(False)
            solo_checkbox.setEnabled(False)

        self._mute_checkboxes[voice_id] = mute_checkbox
        self._solo_checkboxes[voice_id] = solo_checkbox

        grid.addWidget(name_label, row, 0)
        grid.addWidget(mute_checkbox, row, 1, Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(solo_checkbox, row, 2, Qt.AlignmentFlag.AlignCenter)

    def _on_enabled_toggled(self, checked: bool) -> None:
        if checked:
            if not self._engine.start():
                self._mark_unavailable()
        else:
            self._engine.stop()

    def _on_master_gain_changed(self, value: int) -> None:
        self._controller.set_master_gain(value / 100)

    def _mark_unavailable(self) -> None:
        """
        Disables every control so a genuinely unavailable device can
        never be retried into an open-stream loop from this panel, and
        makes that state clearly visible rather than a silently inert
        toggle -- see this class's own docstring on why ON/OFF has no
        retry affordance.
        """

        self._enabled_checkbox.blockSignals(True)
        self._enabled_checkbox.setChecked(False)
        self._enabled_checkbox.blockSignals(False)
        self._enabled_checkbox.setEnabled(False)

        self._master_slider.setEnabled(False)
        for checkbox in (*self._mute_checkboxes.values(), *self._solo_checkboxes.values()):
            checkbox.setEnabled(False)

        self._status_label.setText(UNAVAILABLE_STATUS_TEXT)
        self._status_label.setVisible(True)
