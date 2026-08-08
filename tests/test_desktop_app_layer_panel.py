import chess.pgn

from desktop_app.layer_panel import LayerPanel
from desktop_app.layer_registry import LayerDefinition, LayerRegistry
from desktop_app.layers.attack_influence_layer import ATTACK_INFLUENCE_LAYER
from desktop_app.layers.critical_points_layer import CRITICAL_POINTS_LAYER
from desktop_app.layers.equipotential_layer import EQUIPOTENTIAL_LAYER
from desktop_app.layers.gradient_layer import GRADIENT_LAYER
from desktop_app.layers.morse_smale_layer import MORSE_SMALE_LAYER
from desktop_app.layers.ridge_valley_layer import RIDGE_VALLEY_LAYER
from desktop_app.session_state import DEFAULT_LAYER_VISIBILITY, SessionState

ALL_SIX_LAYERS = (
    ATTACK_INFLUENCE_LAYER,
    EQUIPOTENTIAL_LAYER,
    GRADIENT_LAYER,
    RIDGE_VALLEY_LAYER,
    MORSE_SMALE_LAYER,
    CRITICAL_POINTS_LAYER,
)


def _registry(layers=ALL_SIX_LAYERS) -> LayerRegistry:
    registry = LayerRegistry()
    for layer in layers:
        registry.register(layer)
    return registry


def _panel(qapp, registry=None) -> tuple[LayerPanel, SessionState]:
    session_state = SessionState(chess.pgn.Game())
    panel = LayerPanel(session_state, registry or _registry(), None)
    return panel, session_state


# ---------------------------------------------------------
# registry-driven, not hardcoded
# ---------------------------------------------------------


def test_one_checkbox_row_per_registered_layer(qapp):
    panel, _ = _panel(qapp)

    assert set(panel._checkboxes) == {layer.id for layer in ALL_SIX_LAYERS}
    assert set(panel._sliders) == {layer.id for layer in ALL_SIX_LAYERS}


def test_checkbox_labels_use_the_registry_display_name(qapp):
    panel, _ = _panel(qapp)

    for layer in ALL_SIX_LAYERS:
        assert panel._checkboxes[layer.id].text() == layer.display_name


def test_registering_a_seventh_layer_requires_no_change_to_layer_panel(qapp):
    # Directly exercises Part 12's extensibility claim against this widget:
    # a brand-new LayerDefinition gets a row with no LayerPanel code change.
    future_layer = LayerDefinition(
        id="temporal_derivative",
        display_name="Temporal Derivative",
        data_source=lambda entry: entry,
        renderer=lambda frame: frame,
    )
    registry = _registry(ALL_SIX_LAYERS + (future_layer,))

    panel, _ = _panel(qapp, registry)

    assert "temporal_derivative" in panel._checkboxes
    assert panel._checkboxes["temporal_derivative"].text() == "Temporal Derivative"
    assert len(panel._checkboxes) == 7


def test_rows_are_built_in_deterministic_registration_order(qapp):
    panel, _ = _panel(qapp)

    layout = panel.layout()
    # Row 0 is the "Layers" title; the six layer rows follow in registration
    # order; the legend group box is last.
    row_labels = []
    for index in range(1, 1 + len(ALL_SIX_LAYERS)):
        row_layout = layout.itemAt(index).layout()
        checkbox = row_layout.itemAt(0).widget()
        row_labels.append(checkbox.text())

    assert row_labels == [layer.display_name for layer in ALL_SIX_LAYERS]


# ---------------------------------------------------------
# correct initial visibility
# ---------------------------------------------------------


def test_initial_checkbox_state_matches_default_layer_visibility(qapp):
    panel, session_state = _panel(qapp)

    for layer_id, checkbox in panel._checkboxes.items():
        assert checkbox.isChecked() == session_state.layer_visible(layer_id)
        assert checkbox.isChecked() == DEFAULT_LAYER_VISIBILITY[layer_id]


def test_initial_slider_state_matches_default_layer_opacity(qapp):
    panel, session_state = _panel(qapp)

    for layer_id, slider in panel._sliders.items():
        assert slider.value() == round(session_state.layer_opacity(layer_id) * 100)


# ---------------------------------------------------------
# enable/disable every layer
# ---------------------------------------------------------


def test_unchecking_a_checkbox_hides_the_layer_through_session_state(qapp):
    panel, session_state = _panel(qapp)

    for layer in ALL_SIX_LAYERS:
        panel._checkboxes[layer.id].setChecked(False)
        assert session_state.layer_visible(layer.id) is False

        panel._checkboxes[layer.id].setChecked(True)
        assert session_state.layer_visible(layer.id) is True


def test_all_six_layers_can_still_be_enabled_together(qapp):
    panel, session_state = _panel(qapp)

    for checkbox in panel._checkboxes.values():
        checkbox.setChecked(True)

    assert all(session_state.layer_visible(layer.id) for layer in ALL_SIX_LAYERS)


# ---------------------------------------------------------
# SessionState propagation (both directions)
# ---------------------------------------------------------


def test_checkbox_toggle_calls_session_state_set_layer_visible(qapp):
    panel, session_state = _panel(qapp)

    panel._checkboxes["gradient"].setChecked(True)

    assert session_state.layer_visible("gradient") is True


def test_slider_move_calls_session_state_set_layer_opacity(qapp):
    panel, session_state = _panel(qapp)

    panel._sliders["gradient"].setValue(40)

    assert session_state.layer_opacity("gradient") == 0.4


def test_external_session_state_change_updates_the_checkbox(qapp):
    # The panel must not be the sole owner of layer state -- a change made
    # anywhere else through SessionState (e.g. a future keyboard shortcut)
    # must be reflected here too, via layer_state_changed.
    panel, session_state = _panel(qapp)

    session_state.set_layer_visible("morse_smale", True)

    assert panel._checkboxes["morse_smale"].isChecked() is True


def test_external_session_state_opacity_change_updates_the_slider(qapp):
    panel, session_state = _panel(qapp)

    session_state.set_layer_opacity("critical_points", 0.25)

    assert panel._sliders["critical_points"].value() == 25


# ---------------------------------------------------------
# legend
# ---------------------------------------------------------


def test_panel_includes_a_legend(qapp):
    panel, _ = _panel(qapp)

    layout = panel.layout()
    legend_index = layout.count() - 2  # last item before the trailing stretch
    legend = layout.itemAt(legend_index).widget()

    assert legend is not None
    assert legend.title() == "Legend"
