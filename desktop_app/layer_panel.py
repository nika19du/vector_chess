from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from desktop_app.layer_presets import CUSTOM_LABEL, PRESETS, apply_layer_preset, identify_current_preset
from desktop_app.layer_registry import LayerDefinition, LayerRegistry
from desktop_app.session_state import SessionState
from visualization.critical_points_plot import MARKER_SPECS
from visualization.equipotential_plot import CONTOUR_LINE_COLOR
from visualization.morse_smale_plot import ACCEPTED_CELL_FILL_COLOR
from visualization.ridge_valley_plot import RIDGE_LINE_COLOR, VALLEY_LINE_COLOR

OPACITY_SLIDER_MIN = 0
OPACITY_SLIDER_MAX = 100

# V7 (desktop workspace layout): the preset combo and the six layer rows --
# this panel's frequently-used controls -- are no longer inside any
# QScrollArea at all (see __init__): under V6a's old single shared scroll
# budget, a real-app scroll position could land on the legend and hide
# every checkbox/slider/preset combo entirely, which is exactly the bug a
# live screenshot exposed. LayerPanel now sits beside board_panel/canvas as
# an ordinary sibling with real row height (workspace_row in
# main_window.py), so the controls comfortably fit unscrolled at every
# target resolution without needing a font-metric-sensitive ceiling.
#
# Only the legend (reference material, read occasionally, safe to require a
# scroll gesture for) keeps a bounded QScrollArea -- this is just a minimum
# floor so it isn't crushed to zero height, not a hard ceiling.
LEGEND_SCROLL_MIN_HEIGHT_PX = 60

# Reuses the exact wording `visualization/attack_influence_plot.py` already
# puts on its own colorbar -- not a new claim about what the colors mean,
# just the same one carried into this panel's legend.
ATTACK_INFLUENCE_LEGEND_TEXT = "Black attack influence ← balance → White attack influence"

# Milestone F: reuses the exact wording visualization/source_potential_
# plot.py already puts on its own colorbar, same precedent as
# ATTACK_INFLUENCE_LEGEND_TEXT above. Deliberately says "material," not
# "attacks"/"control" -- Source Potential is occupancy/material-derived, a
# different observable from Attack Influence (see desktop_app/layers/
# source_potential_layer.py's short_caption).
SOURCE_POTENTIAL_LEGEND_TEXT = "Black material ← independent scale → White material"

# Reuses docs/mathematics.md Section 9's classification vocabulary and
# `visualization/critical_points_plot.py`'s own MARKER_SPECS colors --
# the same colors this layer's renderer already draws with
# (desktop_app/layers/critical_points_layer.py), not new ones invented here.
CRITICAL_POINT_CLASSIFICATION_ORDER = ("maximum", "minimum", "saddle", "degenerate")

# V4/V3 consistency: text analogs of the exact glyph shapes
# desktop_app/layers/critical_points_layer.py now actually draws
# (triangle-up/triangle-down/X/hollow-ring) -- not the pre-V3 filled square
# this legend used to show. Unicode glyphs are a legend-only rendering
# choice (this panel is plain Qt widgets, not the GL canvas); the real
# marker geometry lives entirely in _critical_point_glyphs.py, untouched.
CRITICAL_POINT_GLYPH_TEXT = {
    "maximum": "▲",
    "minimum": "▼",
    "saddle": "✕",
    "degenerate": "○",
}


class LayerPanel(QWidget):
    """
    The "Layers" controls (docs/interactive_ui.md Part 2's `[Layers ▾]`
    strip / the frozen mockup's checkbox list).

    Iterates `LayerRegistry.all()` generically -- one checkbox+opacity-slider
    row per registered layer, in registration order -- rather than
    hardcoding six checkbox handlers (Part 5's whole point: a seventh
    registered layer needs no change here, see
    tests/test_desktop_app_layer_panel.py).

    This widget owns no mathematical state. Every row is driven through
    `SessionState`: a checkbox click calls `set_layer_visible`, a slider
    move calls `set_layer_opacity`, and both are kept in sync with
    `layer_state_changed` so any other future writer of that slice (e.g. a
    later phase's keyboard shortcuts) updates this panel too -- it only
    mirrors `SessionState`, never mutates it directly outside of those two
    calls.
    """

    def __init__(
        self,
        session_state: SessionState,
        layer_registry: LayerRegistry,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._session_state = session_state
        self._checkboxes: dict[str, QCheckBox] = {}
        self._sliders: dict[str, QSlider] = {}

        # V7: this panel is now a direct sibling of board_panel/canvas in
        # workspace_row's QHBoxLayout, not a widget squeezed under the
        # canvas -- an explicit Expanding policy is what lets it actually
        # receive its stretch share of width and grow to the row's height,
        # matching the same rationale BoardPanel already documents for
        # itself (a layout only gives extra space to what a widget's own
        # sizePolicy asks for, not to whatever its internal children want).
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        outer_layout = QVBoxLayout(self)
        # V4: compact by construction (Part 5's own requirement) -- also a
        # direct, measured fix (see _build_legend's docstring) for the real
        # canvas-collapse regression the expanded legend caused at 1280x720.
        # V5 additionally trims the default 9px outer margins (pure
        # spacing, same rationale as main_window.py's grid).
        outer_layout.setSpacing(2)
        outer_layout.setContentsMargins(6, 4, 6, 4)

        title = QLabel("Layers")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        outer_layout.addWidget(title)

        # V7 (desktop workspace layout): the preset combo and every layer
        # row go directly into this panel's own layout, never inside a
        # QScrollArea -- always fully visible, never scrollable-out-of-view.
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset"))
        self._preset_combo = QComboBox()
        # Milestone D: one tooltip per preset item, wired the same minimal
        # way as the layer checkboxes' tooltips -- no new widget, no
        # layout change. CUSTOM_LABEL has no description (it isn't a real
        # preset) and so gets no tooltip.
        for preset in PRESETS:
            self._preset_combo.addItem(preset.name)
            if preset.description:
                self._preset_combo.setItemData(
                    self._preset_combo.count() - 1, preset.description, Qt.ItemDataRole.ToolTipRole
                )
        self._preset_combo.addItem(CUSTOM_LABEL)
        # `activated` fires only on genuine user interaction (mouse/keyboard
        # selection), never on the programmatic setCurrentText() calls
        # _sync_preset_combo makes below -- so applying a preset and merely
        # reflecting state never loop back into each other.
        self._preset_combo.activated.connect(self._on_preset_selected)
        preset_row.addWidget(self._preset_combo, stretch=1)
        outer_layout.addLayout(preset_row)

        for layer in layer_registry.all():
            outer_layout.addLayout(self._build_layer_row(layer))

        # Only the legend lives inside a scroll area -- reference material,
        # not a frequently-used control, so requiring a scroll gesture at
        # the tightest window sizes is an acceptable trade-off; the panel's
        # own Expanding size policy lets this area grow with whatever room
        # workspace_row actually has, with LEGEND_SCROLL_MIN_HEIGHT_PX only
        # as a floor against being crushed to zero.
        legend_scroll_area = QScrollArea()
        legend_scroll_area.setWidgetResizable(True)
        legend_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        legend_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        legend_scroll_area.setMinimumHeight(LEGEND_SCROLL_MIN_HEIGHT_PX)
        legend_scroll_area.setWidget(_build_legend())
        outer_layout.addWidget(legend_scroll_area, stretch=1)

        self.setLayout(outer_layout)

        session_state.layer_state_changed.connect(self._on_layer_state_changed)
        self._sync_preset_combo()

    def _build_layer_row(self, layer: LayerDefinition) -> QHBoxLayout:
        layer_id = layer.id
        row = QHBoxLayout()

        checkbox = QCheckBox(layer.display_name)
        # Milestone B (VECTORCHESS_MODEL_V2_INTEGRATION_AUDIT.md Sec. 5, 9):
        # the existing checkbox stays the one and only primary control --
        # its label is still just `display_name` -- the new category/caption
        # metadata rides along as a tooltip, the least intrusive Qt
        # mechanism available, with no new widget and no layout change.
        # Test-only placeholder LayerDefinitions (category="") get no
        # tooltip rather than a malformed "": ...".
        if layer.category and layer.short_caption:
            checkbox.setToolTip(f"{layer.category}: {layer.short_caption}")
        checkbox.setChecked(self._session_state.layer_visible(layer_id))
        checkbox.toggled.connect(
            lambda checked, layer_id=layer_id: self._session_state.set_layer_visible(layer_id, checked)
        )
        self._checkboxes[layer_id] = checkbox

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(OPACITY_SLIDER_MIN, OPACITY_SLIDER_MAX)
        slider.setValue(round(self._session_state.layer_opacity(layer_id) * 100))
        slider.setFixedWidth(90)
        slider.setToolTip("Opacity")
        slider.valueChanged.connect(
            lambda value, layer_id=layer_id: self._session_state.set_layer_opacity(layer_id, value / 100)
        )
        self._sliders[layer_id] = slider

        row.addWidget(checkbox, stretch=1)
        row.addWidget(slider)
        return row

    def _on_layer_state_changed(self, layer_id: str) -> None:
        checkbox = self._checkboxes.get(layer_id)
        if checkbox is not None and checkbox.isChecked() != self._session_state.layer_visible(layer_id):
            checkbox.blockSignals(True)
            checkbox.setChecked(self._session_state.layer_visible(layer_id))
            checkbox.blockSignals(False)

        slider = self._sliders.get(layer_id)
        if slider is not None:
            new_value = round(self._session_state.layer_opacity(layer_id) * 100)
            if slider.value() != new_value:
                slider.blockSignals(True)
                slider.setValue(new_value)
                slider.blockSignals(False)

        # Any layer_state change -- a preset just applied, a manual
        # checkbox/slider edit, or an external writer -- can change whether
        # the current state still exactly matches a preset, so the combo is
        # re-synced unconditionally rather than only from _on_preset_selected.
        self._sync_preset_combo()

    def _on_preset_selected(self, index: int) -> None:
        """
        Only fires on genuine user interaction (see the `activated` connect
        in __init__). Selecting CUSTOM_LABEL itself is a deliberate no-op --
        it isn't a real preset, nothing to apply, and per Part 3's
        "manual toggles remain available, don't force pretend-exactness",
        picking it must never mutate layer_state.
        """
        name = self._preset_combo.itemText(index)
        for preset in PRESETS:
            if preset.name == name:
                apply_layer_preset(self._session_state, preset)
                return

    def _sync_preset_combo(self) -> None:
        """
        Pure display sync, never a write: reflects whatever
        identify_current_preset (a read-only comparison against
        SessionState) currently reports, falling back to CUSTOM_LABEL when
        no preset matches exactly. setCurrentText alone does not emit
        `activated`, so this can never trigger _on_preset_selected.
        """
        matched_name = identify_current_preset(self._session_state)
        target_text = matched_name if matched_name is not None else CUSTOM_LABEL
        if self._preset_combo.currentText() != target_text:
            self._preset_combo.setCurrentText(target_text)


def _legend_swatch_row(layout: QVBoxLayout, swatch_widget: QWidget, text: str) -> None:
    row = QHBoxLayout()
    row.addWidget(swatch_widget)
    label = QLabel(text)
    label.setWordWrap(True)
    row.addWidget(label, stretch=1)
    layout.addLayout(row)


def _color_swatch(hex_color: str, width: int = 12, height: int = 12) -> QLabel:
    swatch = QLabel()
    swatch.setFixedSize(width, height)
    swatch.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #202124;")
    return swatch


def _glyph_swatch(glyph_text: str, hex_color: str) -> QLabel:
    glyph = QLabel(glyph_text)
    glyph.setFixedWidth(14)
    glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)
    glyph.setStyleSheet(f"color: {hex_color}; font-weight: bold;")
    return glyph


CRITICAL_POINT_GRID_COLUMNS = 2


def _build_legend() -> QGroupBox:
    """
    V4: covers all six registered layers -- the pre-V4 legend only ever
    explained Attack Influence's balance and the four critical-point
    classifications, leaving Equipotential/Gradient/Ridge/Valley/Morse-Smale
    completely unexplained despite all being simultaneously visible in the
    "All Layers" preset.

    Milestone F: now covers a seventh, Source Potential -- one further
    label row, same pattern as Attack Influence's own balance_label.

    Deliberately compact: the four critical-point classifications sit in a
    2x2 grid (not four stacked rows) and Ridge/Valley share one row (two
    small swatches, one label) -- both a direct, measured fix for a real
    layout regression this milestone introduced (a naive one-row-per-fact
    legend pushed MathCanvas below its usable floor at 1280x720, see
    tests/test_desktop_app_layout_regression.py) and simply the more
    compact presentation Part 6 itself asks for.
    """
    box = QGroupBox("Legend")
    layout = QVBoxLayout(box)
    layout.setSpacing(2)
    layout.setContentsMargins(6, 4, 6, 4)

    balance_label = QLabel(ATTACK_INFLUENCE_LEGEND_TEXT)
    balance_label.setWordWrap(True)
    layout.addWidget(balance_label)

    source_potential_label = QLabel(SOURCE_POTENTIAL_LEGEND_TEXT)
    source_potential_label.setWordWrap(True)
    layout.addWidget(source_potential_label)

    points_grid = QGridLayout()
    points_grid.setSpacing(2)
    for index, classification in enumerate(CRITICAL_POINT_CLASSIFICATION_ORDER):
        spec = MARKER_SPECS[classification]
        swatch_color = spec["facecolor"] if spec["facecolor"] != "none" else spec["edgecolor"]
        glyph = _glyph_swatch(CRITICAL_POINT_GLYPH_TEXT[classification], swatch_color)
        cell = QHBoxLayout()
        cell.addWidget(glyph)
        cell.addWidget(QLabel(spec["label"]))
        row, column = divmod(index, CRITICAL_POINT_GRID_COLUMNS)
        points_grid.addLayout(cell, row, column)
    layout.addLayout(points_grid)

    _legend_swatch_row(layout, _color_swatch(CONTOUR_LINE_COLOR), "Equipotential (contour)")
    _legend_swatch_row(layout, _glyph_swatch("→", "#202124"), "Gradient (steepest ascent)")

    ridge_valley_row = QHBoxLayout()
    ridge_valley_row.addWidget(_color_swatch(RIDGE_LINE_COLOR, width=14, height=4))
    ridge_valley_row.addWidget(_color_swatch(VALLEY_LINE_COLOR, width=14, height=4))
    ridge_valley_row.addWidget(QLabel("Ridge / Valley"), stretch=1)
    layout.addLayout(ridge_valley_row)

    # Milestone D: "exploratory" carries the same framing as this layer's
    # checkbox tooltip (desktop_app/layers/morse_smale_layer.py) into the
    # legend, without turning either into a paragraph.
    _legend_swatch_row(layout, _color_swatch(ACCEPTED_CELL_FILL_COLOR), "Morse-Smale (exploratory basin)")

    return box
