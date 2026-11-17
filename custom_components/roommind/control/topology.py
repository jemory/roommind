"""Pure room-adjacency graph for the building's local reference temperature.

A ducted forced-air house used to model every room's exchange against a single
scalar: the unweighted mean of *all* room air temperatures. That is wrong in
two distinct ways. It lets a room swap heat with rooms it does not touch, and
it weighs a 3 m2 closet the same as a 40 m2 living room.

The fix is a *local* reference built from the rooms a room actually connects
to. Two connection levels are deliberate:

* ``neighbour_rooms`` -- a normal doorway. Buoyancy exchange through an open
  door couples two rooms far more strongly than the wall between them: tens of
  W/K for a standard 0.9 x 2.0 m opening, against a few W/K for a partition and
  the ~13-15 W/K that is a small room's whole envelope plus infiltration.
  Weight :data:`TOPOLOGY_DOOR_WEIGHT`.
* ``open_rooms`` -- a cased/open-concept opening that shares a freely flowing
  air volume, so it dominates the reference. Weight :data:`TOPOLOGY_OPEN_WEIGHT`.

A room that declares no connections keeps the whole-house mean; the feature is
therefore opt-in. This module is deliberately pure: stdlib only, no Home
Assistant imports, no I/O, no clock, so it can be built and tested standalone.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from ..const import TOPOLOGY_DOOR_WEIGHT, TOPOLOGY_OPEN_WEIGHT


@dataclass(frozen=True)
class RoomGraph:
    """Immutable adjacency graph and mixing groups for the house.

    ``weights`` is a symmetric adjacency map: ``area_id -> {neighbour: weight}``.
    ``mixing_group`` maps each area_id to its connected-component id, or
    ``None`` for a room that declared no connections (callers treat ``None`` as
    "compatible with everything"). ``configured`` is true iff at least one
    edge exists anywhere in the graph.
    """

    weights: Mapping[str, Mapping[str, float]]
    mixing_group: Mapping[str, int | None]
    configured: bool


def _declared_neighbours(rooms: Mapping[str, Mapping[str, Any]], key: str) -> dict[str, set[str]]:
    """Collect, per area_id, the valid string entries declared under *key*.

    Stored data is arbitrary: a missing key, a non-iterable value or junk
    entries (including non-strings) are all tolerated. Only names that are
    known rooms are returned; unknown ids are dropped by the caller.
    """
    result: dict[str, set[str]] = {}
    for area_id, room in rooms.items():
        raw = room.get(key, []) if isinstance(room, Mapping) else []
        if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple, set, frozenset)):
            continue
        valid = {entry for entry in raw if isinstance(entry, str) and entry != area_id}
        if valid:
            result[area_id] = valid
    return result


def build_room_graph(rooms: Mapping[str, Mapping[str, Any]]) -> RoomGraph:
    """Build the symmetric adjacency graph and mixing groups for *rooms*.

    Rules, all deliberate:

    * An edge declared on **either** side is enough. Neighbour lists are
      symmetrised by union, so ``A`` listing ``B`` and ``B`` declaring nothing
      still produces the edge ``A-B``. This lets a user declare a connection
      once, from whichever room they happen to configure.
    * ``open_rooms`` implies a connection even without a ``neighbour_rooms``
      declaration, and when the same pair is declared both ways the OPEN
      weight wins (a freely shared air volume dominates a doorway).
    * Self-edges and edges naming an unknown area_id (a deleted room or a
      typo) are dropped silently; this function is total over stored data.

    Component ids are deterministic: rooms are visited in sorted area_id order
    and each newly discovered component takes the next integer id, starting at
    zero. A room with no edges gets ``None``.
    """
    known = set(rooms)
    doors = _declared_neighbours(rooms, "neighbour_rooms")
    opens = _declared_neighbours(rooms, "open_rooms")

    weights: dict[str, dict[str, float]] = {area_id: {} for area_id in known}

    def add_edge(source: str, targets: set[str], weight: float) -> None:
        for target in targets:
            if target not in known:
                continue
            weights[source][target] = weight
            weights[target][source] = weight

    # Door edges first; OPEN is applied afterwards so it wins on a shared pair.
    for source, targets in doors.items():
        add_edge(source, targets, TOPOLOGY_DOOR_WEIGHT)
    for source, targets in opens.items():
        add_edge(source, targets, TOPOLOGY_OPEN_WEIGHT)

    mixing_group: dict[str, int | None] = {area_id: None for area_id in known}
    next_group = 0
    for start in sorted(known):
        if mixing_group[start] is not None:
            continue
        # A room with no edges stays undeclared; only real components get an id.
        if not weights[start]:
            continue
        mixing_group[start] = next_group
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbour in weights[current]:
                if mixing_group[neighbour] is None and weights[neighbour]:
                    mixing_group[neighbour] = next_group
                    stack.append(neighbour)
        next_group += 1

    return RoomGraph(
        weights=weights,
        mixing_group=mixing_group,
        configured=any(neighbours for neighbours in weights.values()),
    )


def reference_weights(graph: RoomGraph, area_id: str, included: Collection[str]) -> dict[str, float]:
    """Return *area_id*'s neighbour weights restricted to *included* rooms.

    ``included`` is the caller's set of rooms with a usable temperature. The
    room itself is never included, even if it (defensively) appears in its own
    adjacency map. An empty result means the caller should fall back to the
    whole-house mean.
    """
    allowed = set(included)
    allowed.discard(area_id)
    neighbours = graph.weights.get(area_id, {})
    return {neighbour: weight for neighbour, weight in neighbours.items() if neighbour in allowed}


def reference_temperature(
    weights: Mapping[str, float],
    temperatures: Mapping[str, float],
    fallback: float,
) -> float:
    """Weighted mean of ``temperatures[n]`` over the (neighbour, weight) map.

    Neighbours missing from *temperatures* are skipped. If nothing usable
    remains, or the total weight is not positive, *fallback* is returned. This
    is the model's new regressor, so the arithmetic is kept explicit.
    """
    total = 0.0
    weighted_sum = 0.0
    for neighbour, weight in weights.items():
        if neighbour not in temperatures:
            continue
        total += weight
        weighted_sum += weight * temperatures[neighbour]
    if total <= 0.0:
        return fallback
    return weighted_sum / total
