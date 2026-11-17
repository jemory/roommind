"""RoomMindStore create/update semantics for the Central HVAC room schema."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from custom_components.roommind.const import (
    DEFAULT_COMFORT_COOL,
    DEFAULT_COMFORT_HEAT,
    DEFAULT_COVERS_OUTDOOR_MIN_TEMP,
    DEFAULT_ECO_COOL,
    DEFAULT_ECO_HEAT,
    OCCUPANCY_MODE_FOLLOW_SCHEDULE,
    OCCUPANCY_MODE_REACTIVE_TREND,
)

pytestmark = pytest.mark.asyncio


async def test_create_a_minimal_room(store):
    await store.async_load()

    room = await store.async_save_room("living_room", {})

    assert room["area_id"] == "living_room"
    assert room["temperature_sensor"] == ""
    assert room["humidity_sensor"] == ""
    assert room["occupancy_sensors"] == []
    assert room["climate_mode"] == "auto"
    assert room["comfort_temp"] == DEFAULT_COMFORT_HEAT
    assert room["eco_temp"] == DEFAULT_ECO_HEAT
    assert room["covers"] == []
    assert room["covers_auto_enabled"] is False
    assert room["covers_outdoor_min_temp"] == DEFAULT_COVERS_OUTDOOR_MIN_TEMP
    assert room["is_outdoor"] is False
    assert room["climate_control_enabled"] is True


async def test_create_keeps_every_supplied_field(store):
    await store.async_load()
    config = {
        "temperature_sensor": "sensor.temp",
        "humidity_sensor": "sensor.humidity",
        "occupancy_sensors": ["binary_sensor.motion"],
        "occupancy_mode": "reactive_and_trend",
        "schedules": [{"entity_id": "schedule.living"}],
        "schedule_selector_entity": "input_boolean.pick",
        "window_sensors": ["binary_sensor.window"],
        "exterior_window_sensors": ["binary_sensor.window2"],
        "interior_door_sensors": ["binary_sensor.door"],
        "interior_door_delay": 30,
        "door_closed_mix_factor": 0.3,
        "presence_persons": ["person.alex"],
        "display_name": "Living Room",
        "climate_mode": "heat_only",
        "comfort_heat": 20.0,
        "comfort_cool": 25.0,
        "eco_heat": 16.0,
        "eco_cool": 28.0,
        "covers": ["cover.blind"],
        "covers_auto_enabled": True,
        "cover_schedules": [{"entity_id": "schedule.covers"}],
        "read_only_covers": ["sensor.blind"],
        "co2_sensor": "sensor.co2",
        "co2_threshold": 900,
        "voc_sensor": "sensor.voc",
        "voc_threshold": 300,
        "ventilation_entity": "fan.vent",
        "price_shift": 1.5,
        "preheat_enabled": False,
        "ignore_presence": True,
        "is_outdoor": True,
        "override_heat": 21.0,
        "override_cool": 25.0,
        "override_until": 1234.0,
        "override_type": "custom",
    }

    room = await store.async_save_room("living_room", config)

    for key, value in config.items():
        if key == "window_sensors":
            continue  # the canonical field wins and is mirrored onto the legacy one
        assert room[key] == value, key
    # The canonical window field wins over the legacy one and is mirrored back.
    assert room["window_sensors"] == ["binary_sensor.window2"]


async def test_create_and_merge_the_connectivity_keys(store):
    await store.async_load()
    config = {
        "solar_azimuths": [0, 90],
        "neighbour_rooms": ["hallway"],
        "open_rooms": ["dining"],
    }

    created = await store.async_save_room("living_room", config)
    assert created["solar_azimuths"] == [0, 90]
    assert created["neighbour_rooms"] == ["hallway"]
    assert created["open_rooms"] == ["dining"]

    updated = await store.async_save_room(
        "living_room",
        {"solar_azimuths": [180], "neighbour_rooms": ["kitchen"], "open_rooms": []},
    )
    assert updated["solar_azimuths"] == [180]
    assert updated["neighbour_rooms"] == ["kitchen"]
    assert updated["open_rooms"] == []


async def test_create_derives_the_occupancy_mode(store):
    await store.async_load()

    without_sensors = await store.async_save_room("a", {})
    with_sensors = await store.async_save_room("b", {"occupancy_sensors": ["binary_sensor.motion"]})
    reactive = await store.async_save_room(
        "c", {"occupancy_sensors": ["binary_sensor.motion"], "occupancy_eco_enabled": True}
    )

    assert without_sensors["occupancy_mode"] == OCCUPANCY_MODE_FOLLOW_SCHEDULE
    assert with_sensors["occupancy_mode"] == "schedule_assumed_when_on"
    assert reactive["occupancy_mode"] == OCCUPANCY_MODE_REACTIVE_TREND


async def test_create_syncs_comfort_heat_to_the_legacy_field(store):
    await store.async_load()

    room = await store.async_save_room("living_room", {"comfort_heat": 22.5, "eco_heat": 18.5})

    assert room["comfort_temp"] == 22.5
    assert room["eco_temp"] == 18.5


async def test_create_from_legacy_single_temps(store):
    await store.async_load()

    room = await store.async_save_room("living_room", {"comfort_temp": 23.0, "eco_temp": 19.0})

    assert room["comfort_heat"] == 23.0
    assert room["eco_heat"] == 19.0


async def test_create_a_cool_only_room_from_a_single_temp(store):
    await store.async_load()

    room = await store.async_save_room(
        "living_room", {"climate_mode": "cool_only", "comfort_temp": 22.0, "eco_temp": 26.0}
    )

    assert room["comfort_cool"] == 22.0
    assert room["eco_cool"] == 26.0
    assert room["comfort_heat"] == 22.0


async def test_create_ignores_the_device_keys(store):
    await store.async_load()

    room = await store.async_save_room(
        "living_room",
        {
            "devices": [{"entity_id": "climate.trv", "type": "trv"}],
            "thermostats": ["climate.trv"],
            "acs": ["climate.ac"],
            "heating_system_type": "radiator",
            "follow_device_setpoint": True,
            "valve_protection_exclude": ["climate.trv"],
            "heat_source_orchestration": True,
        },
    )

    for key in (
        "devices",
        "thermostats",
        "acs",
        "heating_system_type",
        "follow_device_setpoint",
        "valve_protection_exclude",
        "heat_source_orchestration",
    ):
        assert key not in room


async def test_update_merges_only_given_fields(store):
    await store.async_load()
    await store.async_save_room(
        "living_room",
        {"display_name": "Living", "comfort_heat": 21.0, "temperature_sensor": "sensor.temp"},
    )

    room = await store.async_save_room("living_room", {"display_name": "Lounge"})

    assert room["display_name"] == "Lounge"
    assert room["comfort_heat"] == 21.0
    assert room["temperature_sensor"] == "sensor.temp"


async def test_update_removes_a_list_when_sent_empty(store):
    await store.async_load()
    await store.async_save_room("living_room", {"covers": ["cover.blind"]})

    room = await store.async_save_room("living_room", {"covers": []})

    assert room["covers"] == []


async def test_update_comfort_heat_syncs_the_legacy_field(store):
    await store.async_load()
    await store.async_save_room("living_room", {})

    room = await store.async_save_room("living_room", {"comfort_heat": 22.0})

    assert room["comfort_temp"] == 22.0


async def test_update_legacy_comfort_temp_syncs_the_split_field(store):
    await store.async_load()
    await store.async_save_room("living_room", {})

    room = await store.async_save_room("living_room", {"comfort_temp": 23.0})

    assert room["comfort_heat"] == 23.0


async def test_update_comfort_heat_wins_over_comfort_temp(store):
    await store.async_load()
    await store.async_save_room("living_room", {})

    room = await store.async_save_room("living_room", {"comfort_heat": 22.0, "comfort_temp": 25.0})

    assert room["comfort_heat"] == 22.0
    assert room["comfort_temp"] == 22.0


async def test_update_eco_heat_and_temp_mirror(store):
    await store.async_load()
    await store.async_save_room("living_room", {})

    await store.async_save_room("living_room", {"eco_heat": 18.0})
    assert store.get_room("living_room")["eco_temp"] == 18.0

    await store.async_save_room("living_room", {"eco_temp": 19.0})
    assert store.get_room("living_room")["eco_heat"] == 19.0


async def test_update_a_single_temp_in_a_cool_only_room_targets_cooling(store):
    await store.async_load()
    await store.async_save_room("living_room", {"climate_mode": "cool_only"})

    room = await store.async_save_room("living_room", {"comfort_temp": 22.0, "eco_temp": 26.0})

    assert room["comfort_cool"] == 22.0
    assert room["eco_cool"] == 26.0


async def test_update_promotes_the_canonical_window_sensors(store):
    await store.async_load()
    await store.async_save_room("living_room", {})

    room = await store.async_save_room("living_room", {"exterior_window_sensors": ["binary_sensor.w"]})

    assert room["exterior_window_sensors"] == ["binary_sensor.w"]
    assert room["window_sensors"] == ["binary_sensor.w"]


async def test_update_from_the_legacy_window_field(store):
    await store.async_load()
    await store.async_save_room("living_room", {})

    room = await store.async_save_room("living_room", {"window_sensors": ["binary_sensor.old"]})

    assert room["exterior_window_sensors"] == ["binary_sensor.old"]


async def test_update_occupancy_mode_keeps_the_legacy_flag_in_sync(store):
    await store.async_load()
    await store.async_save_room("living_room", {})

    room = await store.async_save_room("living_room", {"occupancy_mode": "reactive_and_trend"})

    assert room["occupancy_eco_enabled"] is True

    await store.async_update_room("living_room", {"occupancy_mode": "follow_schedule"})

    assert store.get_room("living_room")["occupancy_eco_enabled"] is False


async def test_create_then_update_preserves_the_area_id(store):
    await store.async_load()
    await store.async_save_room("living_room", {"area_id": "ignored"})

    room = await store.async_save_room("living_room", {"area_id": "hijacked"})

    assert room["area_id"] == "living_room"


async def test_each_save_persists_to_storage(store):
    await store.async_load()

    await store.async_save_room("living_room", {})

    store._store.async_save.assert_awaited()
    payload = store._store.async_save.await_args.args[0]
    assert "living_room" in payload["rooms"]


async def test_the_central_plant_settings_are_completed(store):
    await store.async_load()
    await store.async_save_settings({"central_plant": {"thermostat_entity": "climate.hall"}})

    central = store.get_settings()["central_plant"]

    assert central["thermostat_entity"] == "climate.hall"
    assert central["min_run_minutes"] == 10
    assert central["exterior_open_policy"] == "pause_unless_door_closed"


async def test_created_defaults_match_the_constants(store):
    await store.async_load()

    a = await store.async_save_room("a", {"climate_mode": "auto"})
    b = await store.async_save_room("b", {"climate_mode": "auto"})

    assert a["comfort_cool"] == b["comfort_cool"] == DEFAULT_COMFORT_COOL
    assert a["eco_cool"] == b["eco_cool"] == DEFAULT_ECO_COOL


async def test_a_new_room_follows_the_house_settings_by_default(store):
    await store.async_load()

    room = await store.async_save_room("living_room", {})

    assert room["follow_house"] is True


async def test_an_explicit_follow_house_false_is_stored(store):
    await store.async_load()

    room = await store.async_save_room("living_room", {"follow_house": False})

    assert room["follow_house"] is False


async def test_an_old_room_without_the_flag_reads_back_as_following(store):
    store._store.async_load = AsyncMock(
        return_value={
            "rooms": {"living_room": {"area_id": "living_room"}},
            "settings": {},
            "thermal_data": {},
            "occupancy_trends": {},
        }
    )

    await store.async_load()

    assert store.get_room("living_room")["follow_house"] is True
