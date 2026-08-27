"""
V4 (Layer Presets + LayerPanel/Legend UX): named, reusable combinations of
the existing SessionState layer visibility/opacity slice -- no new
mathematical state, no renderer change. Covers three layers:

1. Pure preset definitions/semantics (desktop_app/layer_presets.py) -- no
   Qt needed at all.
2. LayerPanel's preset combo wiring -- selection applies a preset through
   SessionState exactly like a manual checkbox would; the combo reflects
   Custom whenever state drifts from every defined preset.
3. Non-interference with scrub/analysis/audio -- a preset change is
   *only* a visibility/opacity write, verified directly, not assumed.
"""

import chess
import chess.pgn

from desktop_app.full_position_analysis import build_full_position_analysis
from desktop_app.layer_panel import LayerPanel
from desktop_app.layer_presets import (
    ALL_LAYERS,
    CUSTOM_LABEL,
    FLOW,
    INFLUENCE,
    OVERVIEW,
    PRESETS,
    TOPOLOGY,
    apply_layer_preset,
    identify_current_preset,
)
from desktop_app.layer_registry import LayerDefinition, LayerRegistry
from desktop_app.layers.attack_influence_layer import ATTACK_INFLUENCE_LAYER
from desktop_app.layers.critical_points_layer import CRITICAL_POINTS_LAYER
from desktop_app.layers.equipotential_layer import EQUIPOTENTIAL_LAYER
from desktop_app.layers.gradient_layer import GRADIENT_LAYER
from desktop_app.layers.morse_smale_layer import MORSE_SMALE_LAYER
from desktop_app.layers.ridge_valley_layer import RIDGE_VALLEY_LAYER
from desktop_app.layers.source_potential_layer import SOURCE_POTENTIAL_LAYER
from desktop_app.main_window import MainWindow
from desktop_app.session_state import DEFAULT_LAYER_OPACITY, DEFAULT_LAYER_VISIBILITY, SessionState

ALL_SEVEN_LAYERS = (
    ATTACK_INFLUENCE_LAYER,
    SOURCE_POTENTIAL_LAYER,
    EQUIPOTENTIAL_LAYER,
    GRADIENT_LAYER,
    RIDGE_VALLEY_LAYER,
    MORSE_SMALE_LAYER,
    CRITICAL_POINTS_LAYER,
)
ALL_SEVEN_LAYER_IDS = {layer.id for layer in ALL_SEVEN_LAYERS}


def _registry() -> LayerRegistry:
    registry = LayerRegistry()
    for layer in ALL_SEVEN_LAYERS:
        registry.register(layer)
    return registry


def _panel(qapp) -> tuple[LayerPanel, SessionState]:
    session_state = SessionState(chess.pgn.Game())
    panel = LayerPanel(session_state, _registry(), None)
    return panel, session_state


# ---------------------------------------------------------
# 1. Pure preset definitions -- no Qt needed
# ---------------------------------------------------------


def test_every_preset_names_all_seven_registered_layers_explicitly():
    """Applying a preset must produce a reproducible view regardless of
    prior state -- every preset lists every layer id, never leaving one
    untouched (which would make the result depend on what was showing
    before). Milestone F: Source Potential is now a registered layer too,
    so every preset's dict must cover it explicitly (see
    identify_current_preset's exact-key-set comparison, tested directly by
    test_turning_on_source_potential_desyncs_from_overview_preset below) --
    not just the original six."""
    for preset in PRESETS:
        assert set(preset.visibility) == ALL_SEVEN_LAYER_IDS
        assert set(preset.opacity) == ALL_SEVEN_LAYER_IDS


def test_source_potential_is_off_in_every_curated_preset():
    """Source Potential is a genuinely different, non-comparable chess-
    derived observable (occupancy/material, not attack influence) -- none of
    Overview/Influence/Flow/Topology's own curated rationale needs it, and
    showing it by default would change the meaning of those existing views."""
    for preset in (OVERVIEW, INFLUENCE, FLOW, TOPOLOGY):
        assert preset.visibility["source_potential"] is False


def test_source_potential_is_on_only_in_all_layers():
    """"All Layers" means every registered layer, present tense -- Source
    Potential is included there, and only there, now that it has a real
    desktop layer."""
    assert ALL_LAYERS.visibility["source_potential"] is True
    for preset in (OVERVIEW, INFLUENCE, FLOW, TOPOLOGY):
        assert preset is ALL_LAYERS or preset.visibility["source_potential"] is False


def test_overview_is_exactly_the_existing_default_startup_state():
    """OVERVIEW is built directly from DEFAULT_LAYER_VISIBILITY/
    DEFAULT_LAYER_OPACITY, not a hand-copied duplicate -- this pins that
    relationship so the two can never silently drift apart."""
    assert OVERVIEW.visibility == DEFAULT_LAYER_VISIBILITY
    assert OVERVIEW.opacity == DEFAULT_LAYER_OPACITY


def test_influence_is_narrower_than_overview():
    """Distinct, not a near-duplicate: Influence shows strictly fewer
    layers than Overview."""
    influence_visible = {k for k, v in INFLUENCE.visibility.items() if v}
    overview_visible = {k for k, v in OVERVIEW.visibility.items() if v}
    assert influence_visible < overview_visible


def test_topology_omits_the_scalar_field_and_contours():
    assert TOPOLOGY.visibility["attack_influence"] is False
    assert TOPOLOGY.visibility["equipotential"] is False
    assert TOPOLOGY.visibility["ridge_valley"] is True
    assert TOPOLOGY.visibility["morse_smale"] is True
    assert TOPOLOGY.visibility["critical_points"] is True


def test_flow_pairs_equipotential_and_gradient_with_field_context():
    assert FLOW.visibility["equipotential"] is True
    assert FLOW.visibility["gradient"] is True
    assert FLOW.visibility["attack_influence"] is True  # field context, per equipotential_plot.py's own bundling


def test_all_layers_preset_shows_everything():
    assert all(is_visible for is_visible in ALL_LAYERS.visibility.values())


def test_apply_layer_preset_reaches_session_state():
    session_state = SessionState(chess.pgn.Game())
    apply_layer_preset(session_state, TOPOLOGY)

    for layer_id, expected in TOPOLOGY.visibility.items():
        assert session_state.layer_visible(layer_id) == expected
    for layer_id, expected in TOPOLOGY.opacity.items():
        assert session_state.layer_opacity(layer_id) == expected


def test_reapplying_the_same_preset_emits_no_redundant_signals():
    session_state = SessionState(chess.pgn.Game())
    apply_layer_preset(session_state, TOPOLOGY)

    emitted = []
    session_state.layer_state_changed.connect(emitted.append)
    apply_layer_preset(session_state, TOPOLOGY)

    assert emitted == []


def test_identify_current_preset_recognizes_the_default_state():
    session_state = SessionState(chess.pgn.Game())
    assert identify_current_preset(session_state) == "Overview"


def test_identify_current_preset_recognizes_each_applied_preset():
    session_state = SessionState(chess.pgn.Game())
    for preset in PRESETS:
        apply_layer_preset(session_state, preset)
        assert identify_current_preset(session_state) == preset.name


def test_identify_current_preset_returns_none_after_a_manual_toggle():
    """A manual per-layer change that departs from every preset's exact
    combination must not be silently reported as still matching one."""
    session_state = SessionState(chess.pgn.Game())
    apply_layer_preset(session_state, TOPOLOGY)

    session_state.set_layer_visible("gradient", True)  # not part of Topology

    assert identify_current_preset(session_state) is None


def test_identify_current_preset_returns_none_after_a_manual_opacity_change():
    session_state = SessionState(chess.pgn.Game())
    apply_layer_preset(session_state, OVERVIEW)

    session_state.set_layer_opacity("attack_influence", 0.5)

    assert identify_current_preset(session_state) is None


def test_turning_on_source_potential_desyncs_from_overview_preset():
    """Milestone F regression: identify_current_preset compares each
    preset's visibility dict key-by-key against SessionState. Before every
    preset's dict explicitly carried "source_potential": False, turning on
    Source Potential while "Overview" was selected would have left the combo
    silently still reporting "Overview" instead of "Custom" -- a preset dict
    that never mentioned the new layer's id would still "match" no matter
    its actual visibility. This is the single most important new test for
    this milestone: it fails loudly if that explicit-False requirement is
    ever dropped from a preset's definition."""
    session_state = SessionState(chess.pgn.Game())
    apply_layer_preset(session_state, OVERVIEW)
    assert identify_current_preset(session_state) == "Overview"

    session_state.set_layer_visible("source_potential", True)

    assert identify_current_preset(session_state) is None


def test_manually_recreating_a_presets_exact_state_is_recognized_again():
    """Returning to an exact preset combination by manual toggles alone
    (no apply_layer_preset call) must re-identify it -- this is a pure
    state comparison, not a memory of how the state was reached."""
    session_state = SessionState(chess.pgn.Game())
    apply_layer_preset(session_state, TOPOLOGY)
    session_state.set_layer_visible("gradient", True)
    assert identify_current_preset(session_state) is None

    session_state.set_layer_visible("gradient", False)  # back to exactly Topology
    assert identify_current_preset(session_state) == "Topology"


# ---------------------------------------------------------
# 2. LayerPanel preset combo wiring
# ---------------------------------------------------------


def test_combo_lists_every_preset_plus_custom_in_order(qapp):
    panel, _ = _panel(qapp)
    items = [panel._preset_combo.itemText(i) for i in range(panel._preset_combo.count())]
    assert items == [preset.name for preset in PRESETS] + [CUSTOM_LABEL]


# ---------------------------------------------------------
# Milestone D (VECTORCHESS_MODEL_V2_INTEGRATION_AUDIT.md Sec. 4): preset
# descriptions, wired as combo item tooltips.
# ---------------------------------------------------------


def test_every_real_preset_has_a_nonempty_description():
    for preset in PRESETS:
        assert preset.description.strip() != "", preset.name


def test_topology_preset_description_flags_morse_smale_as_exploratory_not_uniform():
    description = TOPOLOGY.description.lower()
    assert "exploratory" in description
    assert "morse-smale" in description
    for banned_word in ("unreliable", "invalid", "fake", "broken", "untrustworthy", "merely decorative"):
        assert banned_word not in description


def test_combo_item_tooltips_match_each_presets_description(qapp):
    from PySide6.QtCore import Qt

    panel, _ = _panel(qapp)
    for index, preset in enumerate(PRESETS):
        assert panel._preset_combo.itemText(index) == preset.name
        assert panel._preset_combo.itemData(index, Qt.ItemDataRole.ToolTipRole) == preset.description


def test_custom_label_has_no_tooltip(qapp):
    from PySide6.QtCore import Qt

    panel, _ = _panel(qapp)
    custom_index = panel._preset_combo.count() - 1
    assert panel._preset_combo.itemText(custom_index) == CUSTOM_LABEL
    assert panel._preset_combo.itemData(custom_index, Qt.ItemDataRole.ToolTipRole) is None


def test_combo_starts_on_overview(qapp):
    panel, _ = _panel(qapp)
    assert panel._preset_combo.currentText() == "Overview"


def test_selecting_a_preset_in_the_combo_applies_it_through_session_state(qapp):
    panel, session_state = _panel(qapp)

    topology_index = [panel._preset_combo.itemText(i) for i in range(panel._preset_combo.count())].index("Topology")
    panel._preset_combo.setCurrentIndex(topology_index)
    panel._on_preset_selected(topology_index)  # emulate the `activated` signal directly

    for layer_id, expected in TOPOLOGY.visibility.items():
        assert session_state.layer_visible(layer_id) == expected


def test_selecting_custom_in_the_combo_does_not_mutate_layer_state(qapp):
    panel, session_state = _panel(qapp)
    before = {layer_id: session_state.layer_visible(layer_id) for layer_id in ALL_SEVEN_LAYER_IDS}

    custom_index = [panel._preset_combo.itemText(i) for i in range(panel._preset_combo.count())].index(CUSTOM_LABEL)
    panel._on_preset_selected(custom_index)

    after = {layer_id: session_state.layer_visible(layer_id) for layer_id in ALL_SEVEN_LAYER_IDS}
    assert before == after


def test_manually_toggling_a_checkbox_switches_the_combo_to_custom(qapp):
    panel, session_state = _panel(qapp)
    panel._checkboxes["gradient"].setChecked(True)  # Overview has gradient off

    assert panel._preset_combo.currentText() == CUSTOM_LABEL


def test_external_session_state_change_updates_the_combo_too(qapp):
    """LayerPanel must reflect SessionState regardless of who wrote it --
    the same guarantee test_desktop_app_layer_panel.py already proves for
    checkboxes/sliders, extended here to the preset combo."""
    panel, session_state = _panel(qapp)
    session_state.set_layer_visible("ridge_valley", True)
    session_state.set_layer_visible("morse_smale", True)
    session_state.set_layer_visible("attack_influence", False)
    session_state.set_layer_visible("equipotential", False)
    # Now exactly Topology's combination.

    assert panel._preset_combo.currentText() == "Topology"


def test_layer_panel_holds_no_independent_preset_state():
    """
    No duplicate state ownership: LayerPanel's only persistent
    preset-related attribute is the combo widget itself, driven entirely by
    reading SessionState through identify_current_preset -- there is no
    second dict/flag tracking "the active preset" anywhere on the panel.

    Checks `self.<attr> =` assignment patterns specifically, not a bare
    substring -- LayerPanel legitimately calls the imported function
    `identify_current_preset(...)`, which itself contains "_current_preset"
    as a substring without being a stored attribute.
    """
    import inspect

    source = inspect.getsource(LayerPanel)
    assert "self._active_preset =" not in source
    assert "self._current_preset =" not in source
    assert "self._selected_preset =" not in source


# ---------------------------------------------------------
# 3. Non-interference with scrub / analysis / audio
# ---------------------------------------------------------


def test_applying_a_preset_triggers_no_analysis_recomputation(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)

    call_count = {"n": 0}

    def counting_builder(board: chess.Board):
        call_count["n"] += 1
        return build_full_position_analysis(board)

    window.position_cache._builder = counting_builder
    qtbot.wait(50)
    baseline_calls = call_count["n"]

    apply_layer_preset(window.session_state, TOPOLOGY)
    qtbot.wait(100)

    assert call_count["n"] == baseline_calls


def test_applying_a_preset_mid_scrub_does_not_cancel_the_scrub_or_move_current_node(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)
    window.session_state.make_move(chess.Move.from_uci("e2e4"))
    qtbot.wait(50)

    current_node_before = window.session_state.current_node
    window.scrub_controller.begin(window.session_state.active_path())
    assert window.scrub_controller.is_scrubbing

    apply_layer_preset(window.session_state, FLOW)

    assert window.scrub_controller.is_scrubbing  # not cancelled
    assert window.session_state.current_node is current_node_before  # GameNode untouched


def test_applying_a_preset_does_not_change_audio_state(qapp, qtbot):
    window = MainWindow()
    qtbot.waitUntil(lambda: window.canvas._overlay_colors is not None, timeout=5000)

    master_gain_before = window.audio_controller.master_gain
    voice_mute_before = dict(window.audio_controller.voice_mute)
    voice_solo_before = dict(window.audio_controller.voice_solo)

    apply_layer_preset(window.session_state, ALL_LAYERS)

    assert window.audio_controller.master_gain == master_gain_before
    assert dict(window.audio_controller.voice_mute) == voice_mute_before
    assert dict(window.audio_controller.voice_solo) == voice_solo_before


# ---------------------------------------------------------
# Legend completeness / V3 glyph consistency / Source Potential
# ---------------------------------------------------------


def test_legend_mentions_every_implemented_layer(qapp):
    from PySide6.QtWidgets import QGroupBox, QLabel

    panel, _ = _panel(qapp)
    legend = panel.findChild(QGroupBox)
    assert legend is not None

    all_text = " ".join(label.text() for label in legend.findChildren(QLabel))
    for expected_substring in (
        "attack influence",
        "material",  # Milestone F: Source Potential's legend row
        "equipotential",
        "gradient",
        "ridge",
        "valley",
        "morse-smale",
        "maximum",
        "minimum",
        "saddle",
        "degenerate",
    ):
        assert expected_substring in all_text.lower(), f"legend missing {expected_substring!r}"


def test_legend_frames_morse_smale_as_exploratory(qapp):
    # Milestone D: the legend swatch label for Morse-Smale carries the same
    # "exploratory" framing as the checkbox tooltip and the console plot,
    # without turning the legend into a paragraph.
    from PySide6.QtWidgets import QGroupBox, QLabel

    panel, _ = _panel(qapp)
    legend = panel.findChild(QGroupBox)
    all_text = " ".join(label.text() for label in legend.findChildren(QLabel))

    assert "morse-smale (exploratory basin)" in all_text.lower()


def test_legend_mentions_source_potential(qapp):
    """Milestone F: Source Potential now has a real desktop layer, so its
    legend row must exist -- reusing visualization/source_potential_plot.py's
    own colorbar wording exactly, same precedent as Attack Influence's
    balance_label."""
    from PySide6.QtWidgets import QGroupBox, QLabel

    panel, _ = _panel(qapp)
    legend = panel.findChild(QGroupBox)
    all_text = " ".join(label.text() for label in legend.findChildren(QLabel)).lower()
    assert "black material" in all_text
    assert "white material" in all_text


def test_legend_critical_point_glyphs_match_v3_shapes_not_obsolete_squares(qapp):
    """No obsolete plain-square swatch for critical points -- each
    classification's legend row uses a shape glyph consistent with
    desktop_app/layers/_critical_point_glyphs.py's actual rendered shapes."""
    from desktop_app.layer_panel import CRITICAL_POINT_GLYPH_TEXT

    assert CRITICAL_POINT_GLYPH_TEXT == {
        "maximum": "▲",
        "minimum": "▼",
        "saddle": "✕",
        "degenerate": "○",
    }
