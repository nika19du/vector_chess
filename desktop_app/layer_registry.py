from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


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
    """

    id: str
    display_name: str
    data_source: Callable[[Any], Any]
    renderer: Callable[[Any], Any]


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
