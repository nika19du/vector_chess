import chess
import pytest

from desktop_app.layer_registry import LayerDefinition, LayerRegistry
from desktop_app.layers.attack_influence_layer import ATTACK_INFLUENCE_LAYER
from desktop_app.layers.critical_points_layer import CRITICAL_POINTS_LAYER
from desktop_app.layers.equipotential_layer import EQUIPOTENTIAL_LAYER
from desktop_app.layers.gradient_layer import GRADIENT_LAYER
from desktop_app.layers.morse_smale_layer import MORSE_SMALE_LAYER
from desktop_app.layers.ridge_valley_layer import RIDGE_VALLEY_LAYER
from tests.conftest import ready_cache_entry

ALL_SIX_LAYERS = (
    ATTACK_INFLUENCE_LAYER,
    EQUIPOTENTIAL_LAYER,
    GRADIENT_LAYER,
    RIDGE_VALLEY_LAYER,
    MORSE_SMALE_LAYER,
    CRITICAL_POINTS_LAYER,
)


def test_register_and_retrieve_the_attack_influence_layer():
    registry = LayerRegistry()

    registry.register(ATTACK_INFLUENCE_LAYER)

    layer = registry.get("attack_influence")
    assert layer.id == "attack_influence"
    assert layer.display_name == "Attack Influence"
    assert layer in registry.all()


def test_all_returns_layers_in_registration_order():
    registry = LayerRegistry()
    second_layer = LayerDefinition(
        id="dummy",
        display_name="Dummy",
        data_source=lambda entry: entry,
        renderer=lambda frame: frame,
    )

    registry.register(ATTACK_INFLUENCE_LAYER)
    registry.register(second_layer)

    assert [layer.id for layer in registry.all()] == ["attack_influence", "dummy"]


def test_registering_a_duplicate_id_raises():
    registry = LayerRegistry()
    registry.register(ATTACK_INFLUENCE_LAYER)

    with pytest.raises(ValueError):
        registry.register(ATTACK_INFLUENCE_LAYER)


def test_registering_an_additional_layer_requires_no_change_to_the_registry():
    # Directly exercises the Part 12 extensibility claim, scoped to what's
    # buildable in Phase 5a: a brand-new, unrelated LayerDefinition registers
    # and retrieves cleanly through the exact same two methods, with no
    # LayerRegistry code change and no special-casing of "attack_influence".
    registry = LayerRegistry()
    registry.register(ATTACK_INFLUENCE_LAYER)

    future_layer = LayerDefinition(
        id="temporal_derivative",  # CLAUDE.md's own named future object
        display_name="Temporal Derivative",
        data_source=lambda entry: entry,
        renderer=lambda frame: frame,
    )
    registry.register(future_layer)

    assert registry.get("temporal_derivative") is future_layer
    assert len(registry.all()) == 2


def test_attack_influence_layer_data_source_and_renderer_round_trip():
    entry = ready_cache_entry(chess.Board())

    frame = ATTACK_INFLUENCE_LAYER.data_source(entry)
    vertex_colors = ATTACK_INFLUENCE_LAYER.renderer(frame)

    assert frame.colors.shape == (8, 8, 4)
    assert vertex_colors.shape == (8 * 8 * 6, 4)


# ---------------------------------------------------------
# Phase 5c: registry completeness and deterministic ordering
# ---------------------------------------------------------


def test_all_six_layers_register_without_error():
    registry = LayerRegistry()
    for layer in ALL_SIX_LAYERS:
        registry.register(layer)

    registered_ids = {layer.id for layer in registry.all()}
    assert registered_ids == {
        "attack_influence",
        "equipotential",
        "gradient",
        "ridge_valley",
        "morse_smale",
        "critical_points",
    }


def test_six_layer_draw_order_is_deterministic_and_matches_registration():
    registry = LayerRegistry()
    for layer in ALL_SIX_LAYERS:
        registry.register(layer)

    expected_order = [layer.id for layer in ALL_SIX_LAYERS]

    # Calling all() repeatedly must never reorder -- the draw order is fixed
    # once, at registration time, never recomputed per frame.
    assert [layer.id for layer in registry.all()] == expected_order
    assert [layer.id for layer in registry.all()] == expected_order


def test_registering_a_seventh_dummy_layer_requires_no_change_to_the_registry():
    # The Phase 13/5c acceptance test named explicitly: registering a
    # seventh layer, alongside the six real ones, requires no change to
    # LayerRegistry or to any of the six existing layers.
    registry = LayerRegistry()
    for layer in ALL_SIX_LAYERS:
        registry.register(layer)

    dummy_layer = LayerDefinition(
        id="dummy_seventh",
        display_name="Dummy Seventh",
        data_source=lambda entry: entry,
        renderer=lambda frame: frame,
    )
    registry.register(dummy_layer)

    assert len(registry.all()) == 7
    assert registry.get("dummy_seventh") is dummy_layer


def test_each_new_layers_data_source_and_renderer_round_trip_on_a_real_position():
    # A smoke check that every new layer's pipeline runs end to end without
    # error on a real, non-trivial position -- full coordinate/quality
    # assertions live in tests/test_desktop_app_layers.py; this only proves
    # every layer is wired correctly through the registry itself.
    board = chess.Board()
    for move in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6"):
        board.push_san(move)
    entry = ready_cache_entry(board)

    for layer in ALL_SIX_LAYERS:
        frame = layer.data_source(entry)
        rendered = layer.renderer(frame)
        assert rendered is not None
