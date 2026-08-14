from __future__ import annotations

from dataclasses import dataclass

from desktop_app.session_state import DEFAULT_LAYER_OPACITY, DEFAULT_LAYER_VISIBILITY, SessionState

# V4: named, reusable combinations of the existing SessionState layer_state
# slice (visibility + opacity) -- no new mathematical state, no
# recomputation, no renderer change. A preset is nothing more than "call
# set_layer_visible/set_layer_opacity for each of the six registered
# layers"; SessionState remains the single source of truth throughout (see
# apply_layer_preset/identify_current_preset below, both pure reads/writes
# through its existing API).
#
# Each preset explicitly lists all six desktop-registered layers (matching
# desktop_app/layer_registry.py's registration order) -- omitting a layer
# id here would leave its prior visibility untouched, which would make
# "apply preset" state-dependent on whatever was showing before it, not a
# clean, reproducible view. Source Potential is not one of the six: it has
# no desktop layer (see docs/interactive_ui.md Part 5's V1 note), so it
# cannot appear in any preset.


@dataclass(frozen=True)
class LayerPreset:
    name: str
    visibility: dict[str, bool]
    # Every preset normalizes opacity back to fully opaque for every layer
    # -- presets are about *which* layers form a coherent view, not *how
    # strong* an individual one is (that stays a manual, per-layer control,
    # untouched by preset selection except that selecting a preset resets
    # it back to a known, deterministic starting point).
    opacity: dict[str, float]


_ALL_LAYER_IDS = ("attack_influence", "equipotential", "gradient", "ridge_valley", "morse_smale", "critical_points")


def _preset(name: str, visible_layer_ids: set[str]) -> LayerPreset:
    return LayerPreset(
        name=name,
        visibility={layer_id: layer_id in visible_layer_ids for layer_id in _ALL_LAYER_IDS},
        opacity={layer_id: 1.0 for layer_id in _ALL_LAYER_IDS},
    )


# OVERVIEW: *is* the existing default startup view -- built directly from
# session_state.py's own DEFAULT_LAYER_VISIBILITY/DEFAULT_LAYER_OPACITY
# (unchanged by this milestone), not a hand-copied duplicate of those
# values, so the two can never drift apart. The scalar field, its
# equipotential structure, and the named discrete features (critical
# points), without either vector field or the cell partition. Communicates
# "who's better, where, and why" in one glance; the two heavier/denser
# layers (Ridge/Valley, Morse-Smale) and the vector field (Gradient) are one
# click away, not pre-loaded.
OVERVIEW = LayerPreset(
    name="Overview", visibility=dict(DEFAULT_LAYER_VISIBILITY), opacity=dict(DEFAULT_LAYER_OPACITY)
)

# INFLUENCE: the raw scalar field alone, nothing else -- for reading pure
# dominance without any overlay competing for attention. Deliberately
# narrower than OVERVIEW (no equipotential, no markers), so the two remain
# genuinely distinct choices rather than near-duplicates.
INFLUENCE = _preset("Influence", {"attack_influence"})

# FLOW: equipotential (constant-value contours) + gradient (direction of
# steepest change) -- a complementary pair the reference's own
# equipotential_plot.py already draws together with the scalar field
# (ATTACK_INFLUENCE_ALPHA is defined and used in that same module), so
# Attack Influence stays on here too rather than showing bare contour/
# vector geometry with no field context for what's "high" or "low".
FLOW = _preset("Flow", {"attack_influence", "equipotential", "gradient"})

# TOPOLOGY: the structural/topological triad -- critical points (the
# features), ridge/valley (the chains connecting them), and Morse-Smale
# (the basins they partition the field into). Attack Influence and
# Equipotential are off: verified visually (this milestone's manual check)
# that the raw scalar field competes with the translucent Morse-Smale cell
# fill for the same color attention, while the plain board chrome lets the
# topology read cleanly as its own "skeleton" view.
TOPOLOGY = _preset("Topology", {"critical_points", "ridge_valley", "morse_smale"})

# ALL: every implemented layer -- technically complete, verified visually
# to be noticeably more cluttered than any single-purpose preset above (V2/
# V3's hierarchy work keeps it *readable*, not *uncluttered*). Kept as the
# explicit debug/inspection option, not the default.
ALL_LAYERS = _preset("All Layers", set(_ALL_LAYER_IDS))

# Order matters: this is the exact order presets appear in the LayerPanel's
# combo box, Overview first since it's the default.
PRESETS: tuple[LayerPreset, ...] = (OVERVIEW, INFLUENCE, FLOW, TOPOLOGY, ALL_LAYERS)

# Shown in the combo box when the current layer_state exactly matches no
# defined preset -- never itself an applicable preset (see
# apply_layer_preset_by_name's explicit no-op for it).
CUSTOM_LABEL = "Custom"


def apply_layer_preset(session_state: SessionState, preset: LayerPreset) -> None:
    """
    The entire mechanism: existing SessionState.set_layer_visible/
    set_layer_opacity calls, nothing else. Both setters are already no-ops
    when the new value equals the current one (see session_state.py), so
    re-applying an already-active preset emits no redundant
    layer_state_changed signals.
    """
    for layer_id, visible in preset.visibility.items():
        session_state.set_layer_visible(layer_id, visible)
    for layer_id, opacity in preset.opacity.items():
        session_state.set_layer_opacity(layer_id, opacity)


def identify_current_preset(session_state: SessionState) -> str | None:
    """
    Pure comparison helper (Part 9's explicit allowance: "a small pure
    helper... for comparing current state against presets" is fine) --
    reads SessionState, never writes it, and holds no state of its own.
    Returns the matching preset's name if the current visibility+opacity
    state is an *exact* match for one of PRESETS, else None (meaning:
    show CUSTOM_LABEL). Deterministic: opacity is compared with ==, not a
    tolerance, since every preset's opacity values are exactly 1.0 and
    SessionState.set_layer_opacity's own clamping never produces anything
    else for an untouched slider.
    """
    for preset in PRESETS:
        visibility_matches = all(
            session_state.layer_visible(layer_id) == visible for layer_id, visible in preset.visibility.items()
        )
        opacity_matches = all(
            session_state.layer_opacity(layer_id) == opacity for layer_id, opacity in preset.opacity.items()
        )
        if visibility_matches and opacity_matches:
            return preset.name
    return None
