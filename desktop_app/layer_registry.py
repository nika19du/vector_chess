from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# Model v2 Integration Audit, Milestone B (VECTORCHESS_MODEL_V2_INTEGRATION_AUDIT.md
# Sec. 6, 19): the small, fixed set of conceptual categories a real,
# user-visible layer's metadata may declare. Deliberately three, not more --
# see VECTORCHESS_MATHEMATICAL_MODEL_V2.md's information hierarchy:
#
#   "Field"    -- a chess-derived observable read directly off the position
#                 (Model v2 Level 2-3), no reconstruction involved.
#   "Geometry" -- a deterministic differential/geometric readout of the
#                 reconstructed surface (Model v2 Level 4-5): contour lines,
#                 gradient vectors -- no detection or classification step.
#   "Topology" -- a detected, classified, and quality-filtered discrete
#                 structure traced on the reconstructed surface (Model v2
#                 Level 6-7): critical points, ridge/valley chains, the
#                 Morse-Smale complex. Matches `layer_presets.py`'s existing
#                 "Topology" preset grouping exactly the same three layers,
#                 not a new, competing taxonomy.
LAYER_CATEGORIES = ("Field", "Geometry", "Topology")


@dataclass(frozen=True)
class LayerDefinition:
    """
    A registered visualization layer (docs/interactive_ui.md Part 5).

    `animator` and `audio_voice` from the frozen design are deliberately not
    fields here yet: nothing animates or plays audio in Phase 5a, and adding
    unused optional fields now would be exactly the speculative abstraction
    this phase is scoped to avoid. Both are additive when they're actually
    needed (Phase 5d, Phase 5f/5c) -- adding a field with a default to a
    dataclass never breaks an existing registration.

    `category`/`short_caption` (Milestone B, VECTORCHESS_MODEL_V2_INTEGRATION
    _AUDIT.md Sec. 2, 19): the same "additive, defaulted field" pattern the
    note above already establishes -- semantic metadata answering "what am I
    looking at, and what kind of thing is it" for the Layer Panel's tooltip,
    not a research record. Deliberately NOT here: confidence scores,
    percentages, experiment IDs, reconstruction parameters, or long
    descriptions (see VECTORCHESS_MATHEMATICAL_MODEL_V2.md and the
    Integration Audit for where that evidence actually lives). Defaulted to
    "" rather than made required so the extensibility tests in
    tests/test_desktop_app_layer_registry.py and
    tests/test_desktop_app_layer_panel.py (a bare, minimal `LayerDefinition`
    registers and renders with no other change) keep working unchanged --
    every real, user-visible layer sets both explicitly at its own
    `desktop_app/layers/*_layer.py` definition site.

    category:
        one of `LAYER_CATEGORIES` for every real layer; "" only for
        test-only placeholder definitions.

    short_caption:
        one sentence, answering "what am I looking at" -- never "how was
        this computed" (that belongs in docs/mathematics.md).
    """

    id: str
    display_name: str
    data_source: Callable[[Any], Any]
    renderer: Callable[[Any], Any]
    category: str = ""
    short_caption: str = ""


class LayerRegistry:
    """
    Ordered registry of `LayerDefinition`s (docs/interactive_ui.md Part 5). The
    layer strip UI, the animation driver, and the canvas renderer all iterate
    this registry generically -- none of them hardcode a list of layer names.
    Registering an additional layer requires no change to this class or to any
    of its consumers (Part 12's extensibility claim, exercised directly by
    tests/test_desktop_app_layer_registry.py).
    """

    def __init__(self) -> None:
        self._layers: dict[str, LayerDefinition] = {}
        self._order: list[str] = []

    def register(self, layer: LayerDefinition) -> None:
        if layer.id in self._layers:
            raise ValueError(f"layer '{layer.id}' is already registered")
        self._layers[layer.id] = layer
        self._order.append(layer.id)

    def get(self, layer_id: str) -> LayerDefinition:
        return self._layers[layer_id]

    def all(self) -> list[LayerDefinition]:
        return [self._layers[layer_id] for layer_id in self._order]
