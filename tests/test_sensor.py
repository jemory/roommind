"""Tests for the sensor platform."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from custom_components.roommind.const import DOMAIN
from custom_components.roommind.sensor import (
    RoomMindCentralActionSensor,
    RoomMindHouseTemperatureSensor,
    RoomMindModeSensor,
    RoomMindTargetTemperatureSensor,
    _create_global_entities,
    _create_room_entities,
    async_setup_entry,
)


def _make_coordinator(rooms_data=None, plant=None):
    """Build a mock coordinator with data dict."""
    coordinator = MagicMock()
    coordinator.data = {"rooms": rooms_data or {}}
    if plant is not None:
        coordinator.data["central_plant"] = plant
    coordinator._entity_areas = set()
    return coordinator


@pytest.mark.asyncio
async def test_setup_entry_creates_entities(hass, mock_config_entry, store):
    """Entities are created for each existing room plus the global plant sensors."""
    await store.async_load()
    await store.async_save_room("room_a", {"temperature_sensor": "sensor.temp"})

    coordinator = _make_coordinator()
    hass.data[DOMAIN] = {
        mock_config_entry.entry_id: coordinator,
        "store": store,
    }
    add_entities = MagicMock()

    await async_setup_entry(hass, mock_config_entry, add_entities)

    # Callback stored on coordinator
    assert coordinator.async_add_entities is add_entities
    # 2 entities per room (target_temp + mode) plus the 2 Central Plant sensors
    add_entities.assert_called_once()
    entities = add_entities.call_args[0][0]
    assert len(entities) == 4
    assert any(isinstance(e, RoomMindCentralActionSensor) for e in entities)
    assert any(isinstance(e, RoomMindHouseTemperatureSensor) for e in entities)


@pytest.mark.asyncio
async def test_setup_entry_no_rooms(hass, mock_config_entry, store):
    """The global plant sensors exist even without any room."""
    await store.async_load()

    coordinator = _make_coordinator()
    hass.data[DOMAIN] = {
        mock_config_entry.entry_id: coordinator,
        "store": store,
    }
    add_entities = MagicMock()

    await async_setup_entry(hass, mock_config_entry, add_entities)

    assert coordinator.async_add_entities is add_entities
    add_entities.assert_called_once()
    entities = add_entities.call_args[0][0]
    assert [type(e) for e in entities] == [RoomMindCentralActionSensor, RoomMindHouseTemperatureSensor]


@pytest.mark.asyncio
async def test_setup_entry_multiple_rooms(hass, mock_config_entry, store):
    """Entities created for each room."""
    await store.async_load()
    await store.async_save_room("room_a", {"temperature_sensor": "sensor.temp_a"})
    await store.async_save_room("room_b", {"temperature_sensor": "sensor.temp_b"})

    coordinator = _make_coordinator()
    hass.data[DOMAIN] = {
        mock_config_entry.entry_id: coordinator,
        "store": store,
    }
    add_entities = MagicMock()

    await async_setup_entry(hass, mock_config_entry, add_entities)

    entities = add_entities.call_args[0][0]
    assert len(entities) == 6  # 2 per room + 2 global


def test_create_room_entities():
    """_create_room_entities returns target temp and mode sensors."""
    coordinator = _make_coordinator()
    entities = _create_room_entities(coordinator, "room_a")
    assert len(entities) == 2
    assert isinstance(entities[0], RoomMindTargetTemperatureSensor)
    assert isinstance(entities[1], RoomMindModeSensor)


def test_create_global_entities():
    """_create_global_entities returns the two Central Plant sensors, enabled by default."""
    sensors = _create_global_entities(_make_coordinator())
    assert [type(sensor) for sensor in sensors] == [RoomMindCentralActionSensor, RoomMindHouseTemperatureSensor]
    assert all(sensor.entity_registry_enabled_default is True for sensor in sensors)


def test_target_temp_sensor_value():
    """Target temperature sensor returns value from room data."""
    coordinator = _make_coordinator({"room_a": {"target_temp": 21.5}})
    sensor = RoomMindTargetTemperatureSensor(coordinator, "room_a")
    assert sensor.native_value == 21.5


def test_target_temp_sensor_missing_room():
    """Target temperature sensor returns None when room is missing."""
    coordinator = _make_coordinator({})
    sensor = RoomMindTargetTemperatureSensor(coordinator, "room_a")
    assert sensor.native_value is None


def test_target_temp_sensor_missing_key():
    """Target temperature sensor returns None when key is missing."""
    coordinator = _make_coordinator({"room_a": {"mode": "idle"}})
    sensor = RoomMindTargetTemperatureSensor(coordinator, "room_a")
    assert sensor.native_value is None


def test_target_temp_sensor_is_a_celsius_measurement():
    """The °C native unit plus device class makes Home Assistant convert the target on °F installs."""
    sensor = RoomMindTargetTemperatureSensor(_make_coordinator(), "room_a")
    assert sensor.device_class == "temperature"
    assert sensor.state_class == "measurement"
    assert sensor.native_unit_of_measurement == "°C"


def test_mode_sensor_value():
    """Mode sensor returns value from room data."""
    coordinator = _make_coordinator({"room_a": {"mode": "heating"}})
    sensor = RoomMindModeSensor(coordinator, "room_a")
    assert sensor.native_value == "heating"


def test_mode_sensor_defaults_to_idle():
    """Mode sensor defaults to 'idle' when key is missing."""
    coordinator = _make_coordinator({"room_a": {"target_temp": 21.0}})
    sensor = RoomMindModeSensor(coordinator, "room_a")
    assert sensor.native_value == "idle"


def test_mode_sensor_missing_room():
    """Mode sensor returns 'idle' when room is missing."""
    coordinator = _make_coordinator({})
    sensor = RoomMindModeSensor(coordinator, "room_a")
    assert sensor.native_value == "idle"


def test_sensor_unique_id():
    """Sensors have correct unique_id format."""
    coordinator = _make_coordinator()
    temp_sensor = RoomMindTargetTemperatureSensor(coordinator, "room_a")
    mode_sensor = RoomMindModeSensor(coordinator, "room_a")
    assert temp_sensor.unique_id == f"{DOMAIN}_room_a_target_temp"
    assert mode_sensor.unique_id == f"{DOMAIN}_room_a_mode"


def test_sensor_entity_id():
    """Sensors have correct entity_id format."""
    coordinator = _make_coordinator()
    temp_sensor = RoomMindTargetTemperatureSensor(coordinator, "room_a")
    mode_sensor = RoomMindModeSensor(coordinator, "room_a")
    assert temp_sensor.entity_id == f"sensor.{DOMAIN}_room_a_target_temp"
    assert mode_sensor.entity_id == f"sensor.{DOMAIN}_room_a_mode"


def test_deprecated_sensors_are_disabled_for_new_rooms():
    """New registry entries of the deprecated sensors start disabled (#238)."""
    coordinator = _make_coordinator()
    assert RoomMindTargetTemperatureSensor(coordinator, "room_a").entity_registry_enabled_default is False
    assert RoomMindModeSensor(coordinator, "room_a").entity_registry_enabled_default is False


# ---------------------------------------------------------------------------
# Central Plant sensors
# ---------------------------------------------------------------------------


def test_central_action_sensor_ids_and_unit():
    """The Central HVAC Action sensor is one enabled-by-default global entity."""
    sensor = RoomMindCentralActionSensor(_make_coordinator())
    assert sensor.unique_id == "roommind_central_action"
    assert sensor.entity_id == "sensor.roommind_central_action"
    # The display name comes from entity.sensor.central_action, so the entity stays
    # translatable instead of pinning an English name in the code.
    assert sensor.translation_key == "central_action"
    assert sensor.icon == "mdi:hvac"
    assert sensor.entity_registry_enabled_default is True


def test_central_action_sensor_value_and_attributes():
    """The action and its decision are read from the plant block."""
    coordinator = _make_coordinator(
        plant={
            "action": "cooling",
            "commanded_action": "cooling",
            "driving_rooms": ["office"],
            "decision_reason": "cooling_priority",
            "guard_active": True,
            "guard_reason": "min_run",
            "wet_coil_locked": False,
            "exterior_paused_rooms": ["kitchen"],
            "plant_override_active": True,
            "any_override": True,
            "legal_actions": ["circulating", "cooling", "idle"],
            "outdoor_blocked": [],
            "outdoor_temp": 24.5,
            "all_force_off": False,
        }
    )
    sensor = RoomMindCentralActionSensor(coordinator)
    assert sensor.native_value == "cooling"
    assert sensor.extra_state_attributes == {
        "commanded_action": "cooling",
        "driving_rooms": ["office"],
        "decision_reason": "cooling_priority",
        "guard_active": True,
        "guard_reason": "min_run",
        "wet_coil_locked": False,
        "exterior_paused_rooms": ["kitchen"],
        "plant_override_active": True,
        "any_override": True,
        "legal_actions": ["circulating", "cooling", "idle"],
        "outdoor_blocked": [],
        "outdoor_temp_c": 24.5,
        "all_force_off": False,
    }


def test_central_action_sensor_explains_an_idle_plant():
    """An idle plant publishes what withheld the action, so it is not a mystery."""
    coordinator = _make_coordinator(
        plant={
            "action": "idle",
            "commanded_action": "idle",
            "decision_reason": "bangbang",
            "legal_actions": ["circulating", "heating", "idle"],
            "outdoor_blocked": ["cooling"],
            "outdoor_temp": 10.5,
            "all_force_off": False,
        }
    )
    attrs = RoomMindCentralActionSensor(coordinator).extra_state_attributes
    assert attrs["legal_actions"] == ["circulating", "heating", "idle"]
    assert attrs["outdoor_blocked"] == ["cooling"]
    assert attrs["outdoor_temp_c"] == 10.5
    assert attrs["all_force_off"] is False


def test_central_action_sensor_maps_the_coordinator_plant_keys():
    """The coordinator names some decision keys differently; the entity still exposes the contract."""
    coordinator = _make_coordinator(plant={"action": "heating", "blocking_rooms": ["kitchen"]})
    attrs = RoomMindCentralActionSensor(coordinator).extra_state_attributes
    assert attrs["commanded_action"] == "heating"
    assert attrs["exterior_paused_rooms"] == ["kitchen"]


def test_a_room_override_is_not_reported_as_a_thermostat_override():
    """A temporary room temperature is the wider override, and the one that lifts the outdoor lockout.

    It used to feed the ``plant_override_active`` flag through a fallback, so the
    attribute read false while a room override was exactly what unblocked cooling.
    """
    coordinator = _make_coordinator(plant={"action": "cooling", "any_override": True})
    attrs = RoomMindCentralActionSensor(coordinator).extra_state_attributes
    assert attrs["any_override"] is True
    assert attrs["plant_override_active"] is False


def test_central_action_sensor_defaults_to_idle():
    """Before the first coordinator cycle the plant reads idle with empty attributes."""
    coordinator = _make_coordinator()
    sensor = RoomMindCentralActionSensor(coordinator)
    assert sensor.native_value == "idle"
    attrs = sensor.extra_state_attributes
    assert attrs["driving_rooms"] == []
    assert attrs["guard_active"] is False
    assert attrs["guard_reason"] is None
    assert attrs["plant_override_active"] is False

    coordinator.data = None
    assert sensor.native_value == "idle"
    # A coordinator whose data block is malformed must not crash the entity
    coordinator.data = {"central_plant": "not-a-dict"}
    assert sensor.native_value == "idle"


def test_house_temperature_sensor_ids_and_device_class():
    """The House Reference Temperature is a temperature measurement in Celsius."""
    sensor = RoomMindHouseTemperatureSensor(_make_coordinator())
    assert sensor.unique_id == "roommind_house_temperature"
    assert sensor.entity_id == "sensor.roommind_house_temperature"
    assert sensor.translation_key == "house_temperature"
    assert sensor.device_class == "temperature"
    assert sensor.state_class == "measurement"
    assert sensor.native_unit_of_measurement == "°C"
    assert sensor.entity_registry_enabled_default is True


def test_house_temperature_sensor_value():
    coordinator = _make_coordinator(plant={"house_ref_temp": 20.5})
    sensor = RoomMindHouseTemperatureSensor(coordinator)
    assert sensor.native_value == 20.5


def test_house_temperature_sensor_missing_value():
    """Unknown or malformed House Reference Temperatures read as None, not as a number."""
    coordinator = _make_coordinator()
    sensor = RoomMindHouseTemperatureSensor(coordinator)
    assert sensor.native_value is None

    coordinator.data = {"central_plant": {"house_ref_temp": "warm"}}
    assert sensor.native_value is None


# ---------------------------------------------------------------------------
# Entity translations
# ---------------------------------------------------------------------------

_STRINGS_PATH = Path(__file__).resolve().parent.parent / "custom_components" / "roommind" / "strings.json"
_TRANSLATIONS_DIR = _STRINGS_PATH.parent / "translations"


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _translation_keys(data: dict, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for key, value in data.items():
        path = f"{prefix}/{key}"
        keys.add(path)
        if isinstance(value, dict):
            keys |= _translation_keys(value, path)
    return keys


def test_the_central_entities_name_themselves_through_the_translations():
    """The three Central Plant entities resolve their English name from strings.json."""
    strings = _load_json(_STRINGS_PATH)
    assert strings == _load_json(_TRANSLATIONS_DIR / "en.json")
    entity = strings["entity"]
    assert entity["sensor"]["central_action"]["name"] == "Central HVAC Action"
    assert entity["sensor"]["house_temperature"]["name"] == "House Reference Temperature"
    assert entity["binary_sensor"]["wet_coil_lockout"]["name"] == "Wet Coil Circulation Lockout"


def test_every_translation_file_has_the_same_keys():
    """A missing key silently falls back to the entity ID, so the four files must stay in sync."""
    keys = _translation_keys(_load_json(_STRINGS_PATH))
    for language in ("en", "de", "fr", "it"):
        assert _translation_keys(_load_json(_TRANSLATIONS_DIR / f"{language}.json")) == keys, language


def test_the_hvac_action_state_uses_the_home_assistant_wire_value():
    """``HVACAction.FAN`` serializes as ``"fan"``; a ``"circulating"`` key never renders."""
    for language in ("en", "de", "fr", "it"):
        state = _load_json(_TRANSLATIONS_DIR / f"{language}.json")["entity"]["climate"]["room"]["state_attributes"][
            "hvac_action"
        ]["state"]
        assert "fan" in state, language
        assert "circulating" not in state, language
