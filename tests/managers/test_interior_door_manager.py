"""Tests for InteriorDoorManager: symmetric debounce and multi-sensor aggregation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.roommind.managers.interior_door_manager import (
    DEFAULT_DOOR_CLOSED_MIX_FACTOR,
    DEFAULT_INTERIOR_DOOR_DELAY_S,
    DOOR_CLOSED_SUPPLY_FACTOR,
    InteriorDoorManager,
    InteriorDoorState,
)

OPEN = "binary_sensor.hall_door"


def _hass_with_states(states: dict[str, str | None]) -> MagicMock:
    """Build a MagicMock hass whose state machine returns *states*."""
    hass = MagicMock()

    def _get(entity_id: str) -> MagicMock | None:
        value = states.get(entity_id)
        if value is None:
            return None
        state = MagicMock()
        state.entity_id = entity_id
        state.state = value
        return state

    hass.states.get = MagicMock(side_effect=_get)
    return hass


def _room(*entities: str, delay: int = DEFAULT_INTERIOR_DOOR_DELAY_S, mix: float | None = None) -> dict:
    room: dict = {"interior_door_sensors": list(entities), "interior_door_delay": delay}
    if mix is not None:
        room["door_closed_mix_factor"] = mix
    return room


def test_has_sensors() -> None:
    assert InteriorDoorManager.has_sensors({"interior_door_sensors": ["binary_sensor.a"]}) is True
    assert InteriorDoorManager.has_sensors({"interior_door_sensors": []}) is False
    assert InteriorDoorManager.has_sensors({}) is False


def test_no_sensors_is_fully_open_and_mixing() -> None:
    manager = InteriorDoorManager()
    state = manager.resolve_and_update(_hass_with_states({}), "living", {"interior_door_sensors": []})
    assert isinstance(state, InteriorDoorState)
    assert state.is_open is True
    assert state.raw_open is True
    assert state.has_sensors is False
    assert state.mix_factor == 1.0
    assert state.supply_factor == 1.0
    assert manager.is_door_open("living") is True


def test_unknown_room_is_open_by_failsafe() -> None:
    manager = InteriorDoorManager()
    assert manager.is_door_open("never-seen") is True
    assert manager.state_for("never-seen") is None


def test_first_observation_adopts_closed_without_delay() -> None:
    manager = InteriorDoorManager()
    hass = _hass_with_states({OPEN: "off"})
    state = manager.resolve_and_update(hass, "living", _room(OPEN, delay=300), now_ts=1000.0)
    assert state.raw_open is False
    assert state.is_open is False
    assert manager.is_door_open("living") is False


def test_first_observation_adopts_open_without_delay() -> None:
    manager = InteriorDoorManager()
    hass = _hass_with_states({OPEN: "on"})
    state = manager.resolve_and_update(hass, "living", _room(OPEN, delay=300), now_ts=1000.0)
    assert state.is_open is True
    assert state.mix_factor == 1.0
    assert state.supply_factor == 1.0


def test_open_debounce() -> None:
    manager = InteriorDoorManager()
    hass = _hass_with_states({OPEN: "off"})
    room = _room(OPEN, delay=60)
    manager.resolve_and_update(hass, "living", room, now_ts=1000.0)

    hass = _hass_with_states({OPEN: "on"})
    state = manager.resolve_and_update(hass, "living", room, now_ts=1010.0)
    assert state.raw_open is True
    assert state.is_open is False  # still inside the debounce

    state = manager.resolve_and_update(hass, "living", room, now_ts=1070.0)
    assert state.is_open is True
    assert state.mix_factor == 1.0
    assert state.supply_factor == 1.0


def test_close_debounce_is_symmetric() -> None:
    manager = InteriorDoorManager()
    room = _room(OPEN, delay=60)
    manager.resolve_and_update(_hass_with_states({OPEN: "on"}), "living", room, now_ts=1000.0)

    closed = _hass_with_states({OPEN: "off"})
    state = manager.resolve_and_update(closed, "living", room, now_ts=1020.0)
    assert state.raw_open is False
    assert state.is_open is True  # still inside the close debounce

    state = manager.resolve_and_update(closed, "living", room, now_ts=1079.0)
    assert state.is_open is True

    state = manager.resolve_and_update(closed, "living", room, now_ts=1080.0)
    assert state.is_open is False
    assert state.mix_factor == pytest.approx(DEFAULT_DOOR_CLOSED_MIX_FACTOR)
    assert state.supply_factor == DOOR_CLOSED_SUPPLY_FACTOR


def test_zero_delay_changes_immediately() -> None:
    manager = InteriorDoorManager()
    room = _room(OPEN, delay=0)
    manager.resolve_and_update(_hass_with_states({OPEN: "off"}), "living", room, now_ts=0.0)
    assert manager.resolve_and_update(_hass_with_states({OPEN: "on"}), "living", room, now_ts=1.0).is_open is True
    assert manager.resolve_and_update(_hass_with_states({OPEN: "off"}), "living", room, now_ts=2.0).is_open is False


def test_reopen_during_close_debounce_cancels_timer() -> None:
    manager = InteriorDoorManager()
    room = _room(OPEN, delay=60)
    manager.resolve_and_update(_hass_with_states({OPEN: "on"}), "living", room, now_ts=0.0)

    manager.resolve_and_update(_hass_with_states({OPEN: "off"}), "living", room, now_ts=10.0)
    assert manager.is_door_open("living") is True

    # Reopen before the close delay expires: still open and the pending close is gone.
    state = manager.resolve_and_update(_hass_with_states({OPEN: "on"}), "living", room, now_ts=30.0)
    assert state.is_open is True
    state = manager.resolve_and_update(_hass_with_states({OPEN: "on"}), "living", room, now_ts=600.0)
    assert state.is_open is True


def test_close_pending_then_raw_oscillates() -> None:
    """A raw flip back to open clears the close delay; a new close restarts it."""
    manager = InteriorDoorManager()
    room = _room(OPEN, delay=60)
    manager.resolve_and_update(_hass_with_states({OPEN: "on"}), "living", room, now_ts=0.0)

    manager.resolve_and_update(_hass_with_states({OPEN: "off"}), "living", room, now_ts=100.0)
    manager.resolve_and_update(_hass_with_states({OPEN: "on"}), "living", room, now_ts=120.0)
    # Closing again starts a fresh timer at 130, so at 170 it must still be open.
    manager.resolve_and_update(_hass_with_states({OPEN: "off"}), "living", room, now_ts=130.0)
    assert manager.is_door_open("living") is True
    assert manager.resolve_and_update(_hass_with_states({OPEN: "off"}), "living", room, now_ts=190.0).is_open is False


def test_multiple_sensors_any_open_wins() -> None:
    manager = InteriorDoorManager()
    room = _room("binary_sensor.a", "binary_sensor.b", delay=0)
    hass = _hass_with_states({"binary_sensor.a": "off", "binary_sensor.b": "on"})
    state = manager.resolve_and_update(hass, "living", room, now_ts=100.0)
    assert state.raw_open is True
    assert state.is_open is True


def test_multiple_sensors_all_closed_only_when_all_off() -> None:
    manager = InteriorDoorManager()
    room = _room("binary_sensor.a", "binary_sensor.b", delay=0)
    hass = _hass_with_states({"binary_sensor.a": "off", "binary_sensor.b": "off"})
    state = manager.resolve_and_update(hass, "living", room, now_ts=100.0)
    assert state.raw_open is False
    assert state.is_open is False


def test_unavailable_and_unknown_are_open_failsafe() -> None:
    manager = InteriorDoorManager()
    room = _room(OPEN, delay=0)
    for value in ("unavailable", "unknown", "weird"):
        state = manager.resolve_and_update(_hass_with_states({OPEN: value}), "living", room, now_ts=0.0)
        assert state.raw_open is True
        assert state.is_open is True


def test_missing_sensor_is_open_failsafe() -> None:
    manager = InteriorDoorManager()
    room = _room(OPEN, delay=0)
    state = manager.resolve_and_update(_hass_with_states({}), "living", room, now_ts=0.0)
    assert state.raw_open is True
    assert state.is_open is True


def test_closed_mix_factor_default_and_clamp() -> None:
    manager = InteriorDoorManager()
    default = manager.resolve_and_update(_hass_with_states({OPEN: "off"}), "a", _room(OPEN, delay=0), now_ts=0.0)
    assert default.mix_factor == pytest.approx(DEFAULT_DOOR_CLOSED_MIX_FACTOR)

    below = manager.resolve_and_update(
        _hass_with_states({OPEN: "off"}), "b", _room(OPEN, delay=0, mix=0.01), now_ts=0.0
    )
    assert below.mix_factor == 0.10

    above = manager.resolve_and_update(_hass_with_states({OPEN: "off"}), "c", _room(OPEN, delay=0, mix=2.0), now_ts=0.0)
    assert above.mix_factor == 1.0


def test_sensors_removed_resets_to_open() -> None:
    manager = InteriorDoorManager()
    room = _room(OPEN, delay=0)
    manager.resolve_and_update(_hass_with_states({OPEN: "off"}), "living", room, now_ts=0.0)
    assert manager.is_door_open("living") is False

    state = manager.resolve_and_update(_hass_with_states({}), "living", {"interior_door_sensors": []}, now_ts=10.0)
    assert state.has_sensors is False
    assert state.is_open is True
    assert manager.is_door_open("living") is True

    # Re-adding a sensor is a fresh start: it adopts its raw state immediately.
    re_added = manager.resolve_and_update(_hass_with_states({OPEN: "off"}), "living", room, now_ts=20.0)
    assert re_added.is_open is False


def test_default_delay_and_implicit_now() -> None:
    manager = InteriorDoorManager()
    room = {"interior_door_sensors": [OPEN]}  # no interior_door_delay -> default 60 s
    manager.resolve_and_update(_hass_with_states({OPEN: "off"}), "living", room, now_ts=0.0)
    state = manager.resolve_and_update(_hass_with_states({OPEN: "on"}), "living", room, now_ts=1.0)
    assert state.raw_open is True
    assert state.is_open is False

    # now_ts omitted: wall clock is far past the debounce, so the door opens.
    state = manager.resolve_and_update(_hass_with_states({OPEN: "on"}), "living", room)
    assert state.is_open is True


def test_remove_room() -> None:
    manager = InteriorDoorManager()
    manager.resolve_and_update(_hass_with_states({OPEN: "off"}), "living", _room(OPEN, delay=0), now_ts=0.0)
    assert manager.is_door_open("living") is False
    manager.remove_room("living")
    assert manager.is_door_open("living") is True
    assert manager.state_for("living") is None
