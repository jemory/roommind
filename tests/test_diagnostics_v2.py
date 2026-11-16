"""Diagnostics schema v2: enough detail to rebuild a reporter's setup in a simulation."""

from __future__ import annotations

from datetime import time as dt_time
from unittest.mock import MagicMock, patch

import pytest

from custom_components.roommind.const import DOMAIN
from custom_components.roommind.diagnostics import (
    DIAGNOSTICS_SCHEMA_VERSION,
    HISTORY_48H_FIELDS,
    _build_entity_states,
    async_get_config_entry_diagnostics,
)
from tests.test_diagnostics import _make_coordinator, _make_estimator


def _state(state: str, **attrs):  # noqa: ANN003, ANN202
    obj = MagicMock()
    obj.state = state
    obj.attributes = attrs
    obj.last_changed.timestamp.return_value = 1000.0
    return obj


def _setup(hass, rooms, coordinator, states):  # noqa: ANN001, ANN202
    store = MagicMock()
    store.get_settings.return_value = {}
    store.get_rooms.return_value = rooms
    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}
    hass.states.get = MagicMock(side_effect=states.get)


@pytest.mark.asyncio
async def test_schema_version_and_location_rounded(hass, mock_config_entry):
    hass.config.latitude = 52.5163
    hass.config.longitude = 13.3777
    _setup(hass, {}, _make_coordinator(), {})
    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)
    assert result["schema_version"] == DIAGNOSTICS_SCHEMA_VERSION == 2
    integ = result["integration"]
    assert integ["latitude"] == 53
    assert integ["longitude"] == 13
    assert integ["time_zone"] == hass.config.time_zone
    assert integ["unit_system"] in ("metric", "us_customary")
    assert "ha_version" in integ


def test_entity_states_cover_every_referenced_entity(hass):
    states = {
        "sensor.t": _state("21.3", unit_of_measurement="°C", device_class="temperature"),
        "binary_sensor.win": _state("off", device_class="window"),
        "schedule.plan": _state("on"),
        "binary_sensor.motion": _state("on", device_class="occupancy"),
    }
    hass.states.get = MagicMock(side_effect=states.get)
    room = {
        "temperature_sensor": "sensor.t",
        "humidity_sensor": "",
        "window_sensors": ["binary_sensor.win"],
        "occupancy_sensors": ["binary_sensor.gone", "binary_sensor.motion"],
        "presence_persons": ["person.someone"],
        "schedules": [{"entity_id": "schedule.plan"}],
    }
    with patch("custom_components.roommind.diagnostics.time.time", return_value=1060.0):
        result = _build_entity_states(hass, room)
    assert result["sensor.t"] == {"state": "21.3", "unit": "°C", "device_class": "temperature", "age_s": 60}
    assert result["binary_sensor.win"]["device_class"] == "window"
    assert result["binary_sensor.gone"] == {"state": "not_found"}
    assert "schedule.plan" in result
    assert "" not in result
    assert "age_s" not in result["binary_sensor.motion"]  # no occupancy timing in public exports
    assert "person.someone" not in result


def test_entity_states_include_the_central_hvac_openings_and_covers(hass):
    """Exterior windows, interior doors and both cover lists are exported, privacy kept."""
    states = {
        "binary_sensor.ext": _state("on", device_class="window"),
        "binary_sensor.door": _state("off", device_class="door"),
        "cover.blind": _state("open", device_class="shade"),
        "cover.only_read": _state("closed"),
        "binary_sensor.presence": _state("on", device_class="presence"),
    }
    hass.states.get = MagicMock(side_effect=states.get)
    room = {
        "exterior_window_sensors": ["binary_sensor.ext"],
        "interior_door_sensors": ["binary_sensor.door"],
        "covers": ["cover.blind"],
        "read_only_covers": ["cover.only_read"],
        "occupancy_sensors": ["binary_sensor.presence"],
        "presence_persons": ["person.someone"],
    }
    with patch("custom_components.roommind.diagnostics.time.time", return_value=1060.0):
        result = _build_entity_states(hass, room)

    assert result["binary_sensor.ext"] == {
        "state": "on",
        "unit": None,
        "device_class": "window",
        "age_s": 60,
    }
    assert result["binary_sensor.door"]["device_class"] == "door"
    assert result["cover.blind"]["device_class"] == "shade"
    assert "cover.only_read" in result
    # Presence timing stays private, and person entities are never listed
    assert "age_s" not in result["binary_sensor.presence"]
    assert not any(entity.startswith("person.") for entity in result)


@pytest.mark.asyncio
async def test_room_has_full_model_schedule_blocks_and_entities(hass, mock_config_entry):
    est = _make_estimator()
    est.to_dict.return_value = {"x": [20.0, 0.3], "P": [[1.0]], "n_updates": 200}
    coordinator = _make_coordinator(estimators={"room_a": est})
    coordinator._schedule_blocks_cache = {
        "schedule.plan": {"monday": [{"from": dt_time(6, 0), "to": dt_time(22, 0), "data": {"temperature": 21}}]}
    }
    room = {"temperature_sensor": "sensor.t", "schedules": [{"entity_id": "schedule.plan"}]}
    _setup(hass, {"room_a": room}, coordinator, {"sensor.t": _state("20.5")})
    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)
    diag = result["rooms"]["room_a"]
    assert diag["model_state"] == {"x": [20.0, 0.3], "P": [[1.0]], "n_updates": 200}
    assert diag["schedule_blocks"]["schedule.plan"]["monday"][0] == {
        "from": "06:00:00",
        "to": "22:00:00",
        "data": {"temperature": 21},
    }
    assert diag["entities"]["sensor.t"]["state"] == "20.5"
    # The 9-state EKF and the plant figures are part of the v2 model block
    assert diag["model"]["alpha_out"] == 0.3
    assert diag["model"]["beta_fan"] == 0.8
    assert diag["model"]["n_circulating"] == 15


@pytest.mark.asyncio
async def test_central_plant_v2_section_has_everything_a_simulation_needs(hass, mock_config_entry):
    """Plant config, thermostat capabilities, guards and coil state in one export."""
    store = MagicMock()
    store.get_settings.return_value = {
        "central_plant": {"thermostat_entity": "climate.hall", "wet_coil_lockout_minutes": 20}
    }
    store.get_rooms.return_value = {}
    wet_coil = MagicMock()
    wet_coil.to_dict.return_value = {"charge": 0.35, "last_cooling_ts": 1234.0}
    plant_manager = MagicMock()
    plant_manager.to_dict.return_value = {
        "active_action": "cooling",
        "action_started_at": 1000.0,
        "last_heat_ended_at": None,
        "last_cool_ended_at": None,
        "state_dirty": True,
    }
    coordinator = _make_coordinator(
        central_plant={"action": "cooling", "guard_active": True, "guard_reason": "min_run", "wet_coil_locked": True},
        wet_coil=wet_coil,
        plant_manager=plant_manager,
    )
    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}
    thermostat = _state(
        "cool",
        hvac_mode="cool",
        hvac_modes=["off", "heat", "cool", "fan_only"],
        hvac_action="cooling",
        current_temperature=24.5,
        temperature=23.0,
        fan_mode="auto",
        fan_modes=["auto", "low", "high"],
        min_temp=10.0,
        max_temp=32.0,
        preset_modes=["none"],
        supported_features=385,
    )
    hass.states.get = MagicMock(side_effect=lambda eid: thermostat if eid == "climate.hall" else None)

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)

    plant = result["central_plant"]
    assert plant["settings"]["wet_coil_lockout_minutes"] == 20
    assert plant["thermostat_entity"] == "climate.hall"
    assert plant["thermostat"]["state"] == "cool"
    assert plant["thermostat"]["hvac_action"] == "cooling"
    assert plant["thermostat"]["current_temperature"] == 24.5
    assert plant["thermostat"]["min_temp"] == 10.0
    assert plant["thermostat"]["supported_features"] == 385
    assert plant["live"]["guard_active"] is True
    assert plant["live"]["guard_reason"] == "min_run"
    assert plant["live"]["wet_coil_locked"] is True
    assert plant["wet_coil"]["charge"] == 0.35
    assert plant["plant_manager"]["active_action"] == "cooling"
    assert plant["plant_manager"]["state_dirty"] is True


@pytest.mark.asyncio
async def test_history_48h_has_all_columns_and_is_capped(hass, mock_config_entry):
    rows = [
        {"timestamp": str(i), "room_temp": "20.0", "mode": "idle", "device_setpoint": "21", "occupancy": "True"}
        for i in range(1500)
    ]
    history = MagicMock()
    history.read_detail = MagicMock(return_value=rows)
    coordinator = _make_coordinator(history_store=history)
    _setup(hass, {"room_a": {}}, coordinator, {})
    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)
    hist = result["history_48h"]["room_a"]
    assert len(hist) == 1000
    assert hist[-1]["timestamp"] == "1499"
    assert hist[0]["device_setpoint"] == "21"
    assert "occupancy" not in hist[0]
    assert len(result["recent_history"]["room_a"]) == 240  # v1 field unchanged


@pytest.mark.asyncio
async def test_history_48h_carries_the_central_hvac_columns(hass, mock_config_entry):
    """The recorded control variables and model inputs travel with the export."""
    for field in (
        "circulate_starts",
        "effective_temp",
        "priority_weight",
        "door_open",
        "fan_circulating",
        "model_power",
        "model_solar",
        "model_occupancy",
        "model_house_temp",
        "model_fan",
        "model_door_mix",
        "model_door_supply",
    ):
        assert field in HISTORY_48H_FIELDS

    history = MagicMock()
    history.read_detail = MagicMock(
        return_value=[
            {
                "timestamp": "1000",
                "circulate_starts": "1",
                "effective_temp": "21.9",
                "priority_weight": "0.8",
                "door_open": "True",
                "fan_circulating": "True",
                "model_house_temp": "20.4",
            }
        ]
    )
    coordinator = _make_coordinator(history_store=history)
    _setup(hass, {"room_a": {}}, coordinator, {})
    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)

    row = result["history_48h"]["room_a"][0]
    assert row["effective_temp"] == "21.9"
    assert row["priority_weight"] == "0.8"
    assert row["door_open"] == "True"
    assert row["fan_circulating"] == "True"
    assert row["circulate_starts"] == "1"
    assert row["model_house_temp"] == "20.4"


@pytest.mark.asyncio
async def test_history_48h_read_error_is_empty(hass, mock_config_entry):
    history = MagicMock()
    history.read_detail = MagicMock(side_effect=OSError("disk"))
    coordinator = _make_coordinator(history_store=history)
    _setup(hass, {"room_a": {}}, coordinator, {})
    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)
    assert result["history_48h"]["room_a"] == []


@pytest.mark.asyncio
async def test_rooms_export_no_device_snapshots_anymore(hass, mock_config_entry):
    """A room with a legacy device list no longer exports per-device states."""
    room = {"devices": [{"entity_id": "climate.trv1", "type": "trv"}], "covers": ["cover.blind"]}
    _setup(hass, {"room_a": room}, _make_coordinator(), {"cover.blind": _state("open")})
    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)

    diag = result["rooms"]["room_a"]
    assert "device_states" not in diag
    assert diag["entities"]["cover.blind"]["state"] == "open"
