"""Tests for the pure room-adjacency graph and local reference helpers."""

from __future__ import annotations

import pytest

from custom_components.roommind.const import TOPOLOGY_DOOR_WEIGHT, TOPOLOGY_OPEN_WEIGHT
from custom_components.roommind.control.topology import (
    build_room_graph,
    reference_temperature,
    reference_weights,
)


def test_symmetrisation_from_one_side_only() -> None:
    """A declared edge is mirrored even when the other room declares nothing."""
    graph = build_room_graph(
        {
            "a": {"neighbour_rooms": ["b"]},
            "b": {},
        }
    )
    assert graph.weights["a"] == {"b": TOPOLOGY_DOOR_WEIGHT}
    assert graph.weights["b"] == {"a": TOPOLOGY_DOOR_WEIGHT}
    assert graph.configured is True


def test_open_weight_wins_over_door() -> None:
    """A pair declared as both gets the stronger OPEN weight on both sides."""
    graph = build_room_graph(
        {
            "a": {"neighbour_rooms": ["b"]},
            "b": {"open_rooms": ["a"]},
        }
    )
    assert graph.weights["a"] == {"b": TOPOLOGY_OPEN_WEIGHT}
    assert graph.weights["b"] == {"a": TOPOLOGY_OPEN_WEIGHT}


def test_open_declared_on_both_sides_same_pair() -> None:
    """Both sides declaring an open edge still yields one OPEN-weighted edge."""
    graph = build_room_graph(
        {
            "a": {"open_rooms": ["b"]},
            "b": {"open_rooms": ["a"]},
        }
    )
    assert graph.weights["a"] == {"b": TOPOLOGY_OPEN_WEIGHT}
    assert graph.weights["b"] == {"a": TOPOLOGY_OPEN_WEIGHT}


def test_open_implies_edge_without_door_declaration() -> None:
    """An open connection alone creates the edge."""
    graph = build_room_graph(
        {
            "a": {"open_rooms": ["b"]},
            "b": {},
        }
    )
    assert graph.weights["a"] == {"b": TOPOLOGY_OPEN_WEIGHT}
    assert graph.weights["b"] == {"a": TOPOLOGY_OPEN_WEIGHT}


def test_self_edge_ignored() -> None:
    """A room listing itself never produces an edge or marks the graph configured."""
    graph = build_room_graph({"a": {"neighbour_rooms": ["a"], "open_rooms": ["a"]}})
    assert graph.weights["a"] == {}
    assert graph.configured is False
    assert graph.mixing_group["a"] is None


def test_unknown_area_id_dropped() -> None:
    """Edges to deleted rooms or typos are silently dropped, never raised."""
    graph = build_room_graph(
        {
            "a": {"neighbour_rooms": ["ghost", "b"], "open_rooms": ["deleted"]},
            "b": {"neighbour_rooms": ["typo"]},
        }
    )
    assert graph.weights["a"] == {"b": TOPOLOGY_DOOR_WEIGHT}
    assert graph.weights["b"] == {"a": TOPOLOGY_DOOR_WEIGHT}
    assert all("ghost" not in neighbours for neighbours in graph.weights.values())
    assert all("deleted" not in neighbours for neighbours in graph.weights.values())
    assert all("typo" not in neighbours for neighbours in graph.weights.values())


def test_three_room_chain_one_component_and_fourth_undeclared() -> None:
    """A chain of three rooms forms one component; an isolated room gets None."""
    graph = build_room_graph(
        {
            "a": {"neighbour_rooms": ["b"]},
            "b": {"neighbour_rooms": ["c"]},
            "c": {},
            "d": {},
        }
    )
    assert graph.mixing_group["a"] == graph.mixing_group["b"] == graph.mixing_group["c"] == 0
    assert graph.mixing_group["d"] is None
    assert graph.configured is True


def test_deterministic_component_ids() -> None:
    """Component ids follow sorted room order, independent of declaration order."""
    rooms = {
        "z": {"neighbour_rooms": ["y"]},
        "y": {},
        "b": {"open_rooms": ["a"]},
        "a": {},
        "m": {},
    }
    graph = build_room_graph(rooms)
    # Sorted order: a,b,m,y,z -> component {a,b} gets 0, {y,z} gets 1, m is None.
    assert graph.mixing_group == {"a": 0, "b": 0, "m": None, "y": 1, "z": 1}

    shuffled = dict(reversed(list(rooms.items())))
    assert build_room_graph(shuffled).mixing_group == graph.mixing_group


def test_configured_false_for_empty_house() -> None:
    """An empty mapping yields an empty, unconfigured graph."""
    graph = build_room_graph({})
    assert graph.weights == {}
    assert graph.mixing_group == {}
    assert graph.configured is False


def test_configured_false_when_rooms_declare_nothing() -> None:
    """Rooms present but with no usable edges are unconfigured."""
    graph = build_room_graph({"a": {}, "b": {}})
    assert graph.configured is False
    assert graph.mixing_group == {"a": None, "b": None}


@pytest.mark.parametrize(
    "junk",
    [
        None,
        42,
        "a",
        {"a": 1},
        ["a", 42, None, b"a", True],
        ("a", 3.5),
    ],
)
def test_junk_declarations_are_coerced_defensively(junk: object) -> None:
    """Non-string or malformed declarations must not raise and must not create edges."""
    graph = build_room_graph({"a": {"neighbour_rooms": junk}, "b": {}})  # type: ignore[dict-item]
    assert graph.weights["a"].get("b") is None or graph.weights["a"]["b"] == TOPOLOGY_DOOR_WEIGHT


def test_non_mapping_room_value_is_tolerated() -> None:
    """A malformed room value is skipped without raising."""
    graph = build_room_graph({"a": "not a dict", "b": {}})  # type: ignore[dict-item]
    assert graph.weights["a"] == {}
    assert graph.configured is False


def test_reference_weights_excludes_self_and_non_included() -> None:
    """Only included neighbours are returned; the room itself never is."""
    graph = build_room_graph(
        {
            "a": {"neighbour_rooms": ["b"], "open_rooms": ["c"]},
            "b": {},
            "c": {},
            "d": {},
        }
    )
    assert reference_weights(graph, "a", {"b", "c", "d"}) == {
        "b": TOPOLOGY_DOOR_WEIGHT,
        "c": TOPOLOGY_OPEN_WEIGHT,
    }
    assert reference_weights(graph, "a", {"b"}) == {"b": TOPOLOGY_DOOR_WEIGHT}
    assert reference_weights(graph, "a", set()) == {}
    assert reference_weights(graph, "unknown", {"a"}) == {}


def test_reference_temperature_weighted_mean() -> None:
    """The weighted arithmetic is exact."""
    assert reference_temperature({"a": 1.0, "b": 3.0}, {"a": 20.0, "b": 24.0}, 0.0) == 23.0


def test_reference_temperature_empty_weights_uses_fallback() -> None:
    """No weights means no information: use the whole-house fallback."""
    assert reference_temperature({}, {"a": 20.0}, 19.5) == 19.5


def test_reference_temperature_missing_key_skipped() -> None:
    """A neighbour without a temperature is ignored, not an error."""
    assert reference_temperature({"a": 1.0, "b": 3.0}, {"a": 20.0}, 99.0) == 20.0


def test_reference_temperature_all_missing_uses_fallback() -> None:
    """When every neighbour lacks a temperature, fall back."""
    assert reference_temperature({"a": 1.0, "b": 3.0}, {}, 17.0) == 17.0


def test_reference_temperature_zero_weight_sum_uses_fallback() -> None:
    """A non-positive weight sum falls back rather than dividing by zero."""
    assert reference_temperature({"a": 0.0, "b": 0.0}, {"a": 20.0, "b": 24.0}, 18.0) == 18.0
    assert reference_temperature({"a": -1.0}, {"a": 20.0}, 18.0) == 18.0


def test_single_warm_neighbour_equals_its_temperature() -> None:
    """A closet with one warm neighbour gets that room's temperature, not the house mean.

    With only one usable neighbour the weighted mean collapses to that single
    temperature, which is the exact 'closet vs living room' behaviour this
    feature exists to provide.
    """
    graph = build_room_graph({"closet": {"neighbour_rooms": ["living"]}, "living": {}})
    weights = reference_weights(graph, "closet", {"closet", "living"})
    house_mean = (18.0 + 24.0) / 2
    assert weights == {"living": TOPOLOGY_DOOR_WEIGHT}
    assert reference_temperature(weights, {"closet": 18.0, "living": 24.0}, house_mean) == 24.0
