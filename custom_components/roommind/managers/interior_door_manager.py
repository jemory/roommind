"""Interior door smoothing for RoomMind.

An interior door separates a Thermal Zone from the rest of the house.  When it
is open, the room exchanges air with the adjacent space and the central return
path, so its effective coupling to the House Reference Temperature is high.
When it is closed, the room is partly decoupled: less of the central air stream
reaches it and passive exchange with the rest of the house slows down.

The manager debounces raw door sensors symmetrically (opening and closing wait
the same delay) and hands the MPC a ``mix_factor`` and ``supply_factor``.  A
missing or unavailable sensor is open by fail-safe: an unknown door state must
not make the model believe a room is sealed off.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from homeassistant.core import HomeAssistant

DEFAULT_INTERIOR_DOOR_DELAY_S = 60
DEFAULT_DOOR_CLOSED_MIX_FACTOR = 0.40
DOOR_CLOSED_SUPPLY_FACTOR = 0.75

_MIN_CLOSED_MIX_FACTOR = 0.10
_MAX_CLOSED_MIX_FACTOR = 1.0


@dataclass
class InteriorDoorState:
    """Effective interior-door state for one room."""

    is_open: bool
    raw_open: bool
    has_sensors: bool
    mix_factor: float
    supply_factor: float


class InteriorDoorManager:
    """Debounce interior door sensors and derive room air-exchange factors."""

    def __init__(self) -> None:
        self._states: dict[str, InteriorDoorState] = {}
        self._effective: dict[str, bool] = {}
        self._pending_target: dict[str, bool] = {}
        self._pending_since: dict[str, float] = {}
        self._seen: set[str] = set()

    @staticmethod
    def has_sensors(room: dict) -> bool:
        """Return True when the room has at least one configured door sensor."""
        return bool(room.get("interior_door_sensors") or [])

    def _read_raw_open(self, hass: HomeAssistant, sensors: list[str]) -> bool:
        """Aggregate sensors: open if any is on/unavailable, closed only if all are off."""
        for entity_id in sensors:
            state = hass.states.get(entity_id)
            if state is None:
                # Missing (deleted, disabled or not started): open fail-safe.
                return True
            if state.state != "off":
                # "on", "unavailable", "unknown" and anything else unexpected.
                return True
        return False

    def resolve_and_update(
        self,
        hass: HomeAssistant,
        area_id: str,
        room: dict,
        now_ts: float | None = None,
    ) -> InteriorDoorState:
        """Read the room's door sensors, debounce them and return the effective state."""
        now = time.time() if now_ts is None else now_ts
        sensors = room.get("interior_door_sensors") or []

        if not sensors:
            self._seen.discard(area_id)
            self._pending_target.pop(area_id, None)
            self._pending_since.pop(area_id, None)
            self._effective[area_id] = True
            state = InteriorDoorState(
                is_open=True,
                raw_open=True,
                has_sensors=False,
                mix_factor=1.0,
                supply_factor=1.0,
            )
            self._states[area_id] = state
            return state

        raw_open = self._read_raw_open(hass, sensors)
        delay = int(room.get("interior_door_delay", DEFAULT_INTERIOR_DOOR_DELAY_S))

        if area_id not in self._seen:
            self._seen.add(area_id)
            effective = raw_open
            self._pending_target.pop(area_id, None)
            self._pending_since.pop(area_id, None)
        else:
            effective = self._effective.get(area_id, raw_open)
            if raw_open == effective:
                self._pending_target.pop(area_id, None)
                self._pending_since.pop(area_id, None)
            else:
                if self._pending_target.get(area_id) != raw_open:
                    self._pending_target[area_id] = raw_open
                    self._pending_since[area_id] = now
                since = self._pending_since[area_id]
                if delay <= 0 or now - since >= delay:
                    effective = raw_open
                    self._pending_target.pop(area_id, None)
                    self._pending_since.pop(area_id, None)

        self._effective[area_id] = effective
        if effective:
            mix_factor = 1.0
            supply_factor = 1.0
        else:
            configured = float(room.get("door_closed_mix_factor", DEFAULT_DOOR_CLOSED_MIX_FACTOR))
            mix_factor = max(_MIN_CLOSED_MIX_FACTOR, min(_MAX_CLOSED_MIX_FACTOR, configured))
            supply_factor = DOOR_CLOSED_SUPPLY_FACTOR

        state = InteriorDoorState(
            is_open=effective,
            raw_open=raw_open,
            has_sensors=True,
            mix_factor=mix_factor,
            supply_factor=supply_factor,
        )
        self._states[area_id] = state
        return state

    def is_door_open(self, area_id: str) -> bool:
        """Return the debounced door state; unknown rooms are open by fail-safe."""
        return self._effective.get(area_id, True)

    def state_for(self, area_id: str) -> InteriorDoorState | None:
        """Return the state from the last update, or None when never resolved."""
        return self._states.get(area_id)

    def remove_room(self, area_id: str) -> None:
        """Drop all door state for a removed room."""
        self._states.pop(area_id, None)
        self._effective.pop(area_id, None)
        self._pending_target.pop(area_id, None)
        self._pending_since.pop(area_id, None)
        self._seen.discard(area_id)
