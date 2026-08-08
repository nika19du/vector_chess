from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from desktop_app.layer_registry import LayerRegistry
from desktop_app.session_state import SessionState
from visualization.critical_points_plot import MARKER_SPECS

OPACITY_SLIDER_MIN = 0
OPACITY_SLIDER_MAX = 100

# Reuses the exact wording `visualization/attack_influence_plot.py` already
# puts on its own colorbar -- not a new claim about what the colors mean,
# just the same one carried into this panel's legend.
ATTACK_INFLUENCE_LEGEND_TEXT = "Black attack influence ← balance → White attack influence"

# Reuses docs/mathematics.md Section 9's classification vocabulary and
# `visualization/critical_points_plot.py`'s own MARKER_SPECS colors --
# the same colors this layer's renderer already draws with
# (desktop_app/layers/critical_points_layer.py), not new ones invented here.
CRITICAL_POINT_CLASSIFICATION_ORDER = ("maximum", "minimum", "saddle", "degenerate")


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

        layout = QVBoxLayout(self)

        title = QLabel("Layers")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title)

        for layer in layer_registry.all():
            layout.addLayout(self._build_layer_row(layer.id, layer.display_name))

        layout.addWidget(_build_legend())
        layout.addStretch(1)
        self.setLayout(layout)

        session_state.layer_state_changed.connect(self._on_layer_state_changed)

    def _build_layer_row(self, layer_id: str, display_name: str) -> QHBoxLayout:
        row = QHBoxLayout()

        checkbox = QCheckBox(display_name)
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


def _build_legend() -> QGroupBox:
    box = QGroupBox("Legend")
    layout = QVBoxLayout(box)

    balance_label = QLabel(ATTACK_INFLUENCE_LEGEND_TEXT)
    balance_label.setWordWrap(True)
    layout.addWidget(balance_label)

    for classification in CRITICAL_POINT_CLASSIFICATION_ORDER:
        spec = MARKER_SPECS[classification]
        swatch_color = spec["facecolor"] if spec["facecolor"] != "none" else spec["edgecolor"]

        row = QHBoxLayout()
        swatch = QLabel()
        swatch.setFixedSize(12, 12)
        swatch.setStyleSheet(f"background-color: {swatch_color}; border: 1px solid #202124;")
        row.addWidget(swatch)
        row.addWidget(QLabel(spec["label"]))
        row.addStretch(1)
        layout.addLayout(row)

    return box
