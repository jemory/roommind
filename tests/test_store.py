"""RoomMindStore: room CRUD, schema migration and the learned-data buckets."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from custom_components.roommind.const import (
    DEFAULT_CENTRAL_PLANT_SETTINGS,
    DEFAULT_CO2_THRESHOLD,
    DEFAULT_COMFORT_COOL,
    DEFAULT_COMFORT_HEAT,
    DEFAULT_ECO_COOL,
    DEFAULT_ECO_HEAT,
    OCCUPANCY_MODE_FOLLOW_SCHEDULE,
    OCCUPANCY_MODE_REACTIVE_TREND,
    OCCUPANCY_MODE_SCHEDULE_ASSUMED,
)
from custom_components.roommind.store import (
    RETIRED_ROOM_KEYS,
    _RoomMindHaStore,
    migrate_store_data,
    strip_retired_room_keys,
)

# ---------------------------------------------------------------------------
# Room CRUD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_room_creates_a_room_with_defaults(store):
    await store.async_load()

    room = await store.async_save_room("living_room", {})

    assert room["area_id"] == "living_room"
    assert room["comfort_heat"] == DEFAULT_COMFORT_HEAT
    assert room["comfort_cool"] == DEFAULT_COMFORT_COOL
    assert room["eco_heat"] == DEFAULT_ECO_HEAT
    assert room["eco_cool"] == DEFAULT_ECO_COOL
    assert room["climate_mode"] == "auto"
    assert room["occupancy_mode"] == OCCUPANCY_MODE_FOLLOW_SCHEDULE
    assert room["co2_threshold"] == DEFAULT_CO2_THRESHOLD
    assert room["interior_door_sensors"] == []
    assert room["read_only_covers"] == []
    assert store.get_room("living_room") is not None


@pytest.mark.asyncio
async def test_save_room_merges_into_an_existing_room(store):
    await store.async_load()
    await store.async_save_room("living_room", {"display_name": "Living"})

    room = await store.async_save_room("living_room", {"display_name": "Lounge", "comfort_heat": 22.0})

    assert room["display_name"] == "Lounge"
    assert room["comfort_heat"] == 22.0
    assert room["comfort_temp"] == 22.0


@pytest.mark.asyncio
async def test_get_rooms_and_get_room_return_copies(store):
    await store.async_load()
    await store.async_save_room("living_room", {"display_name": "Living"})

    rooms = store.get_rooms()
    room = store.get_room("living_room")
    assert room is not None
    room["display_name"] = "changed"
    rooms["living_room"]["display_name"] = "changed too"

    assert store.get_room("living_room")["display_name"] == "Living"
    assert store.get_room("missing") is None


@pytest.mark.asyncio
async def test_delete_room_removes_learned_data(store):
    await store.async_load()
    await store.async_save_room("living_room", {})
    await store.async_save_thermal_data({"living_room": {"x": 1}})
    await store.async_save_occupancy_trends({"living_room": {"trend": {}}})

    await store.async_delete_room("living_room")

    assert store.get_rooms() == {}
    assert store.get_thermal_data() == {}
    assert store.get_occupancy_trends() == {}


@pytest.mark.asyncio
async def test_deleting_a_missing_room_raises(store):
    await store.async_load()
    with pytest.raises(KeyError):
        await store.async_delete_room("missing")


@pytest.mark.asyncio
async def test_update_room_changes_only_given_fields(store):
    await store.async_load()
    await store.async_save_room("living_room", {"display_name": "Living", "comfort_heat": 21.0})

    room = await store.async_update_room("living_room", {"display_name": "Lounge", "area_id": "hijack"})

    assert room["display_name"] == "Lounge"
    assert room["comfort_heat"] == 21.0
    assert room["area_id"] == "living_room"


@pytest.mark.asyncio
async def test_update_room_missing_raises(store):
    await store.async_load()
    with pytest.raises(KeyError):
        await store.async_update_room("missing", {})


@pytest.mark.asyncio
async def test_the_persisted_payload_has_the_three_buckets(store):
    await store.async_load()
    await store.async_save_room("living_room", {})

    payload = store._store.async_save.await_args.args[0]

    assert set(payload) == {"rooms", "settings", "thermal_data", "occupancy_trends"}
    assert "living_room" in payload["rooms"]


# ---------------------------------------------------------------------------
# Retired per-room device configuration
# ---------------------------------------------------------------------------


def test_every_retired_room_key_is_stripped():
    room = {key: [] for key in RETIRED_ROOM_KEYS}
    room["display_name"] = "Living"

    strip_retired_room_keys(room)

    assert room == {"display_name": "Living"}


@pytest.mark.asyncio
async def test_a_saved_room_cannot_keep_device_configuration(store):
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
            "heat_source_primary_delta": 2.0,
            "heat_source_outdoor_threshold": 4.0,
            "heat_source_ac_min_outdoor": -10.0,
            "heat_source_price_aware": True,
            "display_name": "Living",
        },
    )

    for key in RETIRED_ROOM_KEYS:
        assert key not in room
    assert room["display_name"] == "Living"


@pytest.mark.asyncio
async def test_updating_a_room_ignores_retired_keys(store):
    await store.async_load()
    await store.async_save_room("living_room", {"display_name": "Living"})

    room = await store.async_update_room(
        "living_room",
        {"devices": [{"entity_id": "climate.trv"}], "display_name": "Lounge"},
    )

    assert "devices" not in room
    assert room["display_name"] == "Lounge"


@pytest.mark.asyncio
async def test_loading_an_old_room_strips_its_devices(store):
    store._store.async_load = AsyncMock(
        return_value={
            "rooms": {
                "living_room": {
                    "area_id": "living_room",
                    "devices": [{"entity_id": "climate.trv", "type": "trv"}],
                    "thermostats": ["climate.trv"],
                    "acs": [],
                    "heating_system_type": "radiator",
                }
            },
            "settings": {},
            "thermal_data": {},
        }
    )

    await store.async_load()

    room = store.get_room("living_room")
    for key in RETIRED_ROOM_KEYS:
        assert key not in room


# ---------------------------------------------------------------------------
# v1 -> v2 migration
# ---------------------------------------------------------------------------


def test_migrate_store_data_is_idempotent():
    raw = {
        "rooms": {"living_room": {"area_id": "living_room", "comfort_temp": 22.5, "override_temp": 23.0}},
        "settings": {"heating_threshold": 1.0, "compressor_groups": []},
    }

    once = migrate_store_data(raw)
    twice = migrate_store_data(once)

    assert once == twice


def test_migrate_store_data_does_not_mutate_its_input():
    raw = {
        "rooms": {"living_room": {"area_id": "living_room", "devices": [{"entity_id": "climate.trv"}]}},
        "settings": {},
    }

    migrate_store_data(raw)

    assert raw["rooms"]["living_room"]["devices"] == [{"entity_id": "climate.trv"}]


def test_migrate_store_data_handles_a_non_dict_payload():
    migrated = migrate_store_data({"rooms": [], "settings": "junk", "thermal_data": 5, "occupancy_trends": None})

    assert migrated["rooms"] == {}
    assert migrated["settings"]["central_plant"] == DEFAULT_CENTRAL_PLANT_SETTINGS
    assert migrated["thermal_data"] == {}
    assert migrated["occupancy_trends"] == {}


def test_migrate_store_data_skips_non_dict_rooms():
    migrated = migrate_store_data({"rooms": {"broken": "not a room"}, "settings": {}})

    assert migrated["rooms"] == {"broken": "not a room"}


def test_migrate_store_data_seeds_the_learned_buckets():
    migrated = migrate_store_data({"settings": {}})

    assert migrated["thermal_data"] == {}
    assert migrated["occupancy_trends"] == {}


def test_migrate_store_data_keeps_existing_learned_buckets():
    migrated = migrate_store_data(
        {"settings": {}, "thermal_data": {"living_room": {"x": 1}}, "occupancy_trends": {"living_room": {"y": 2}}}
    )

    assert migrated["thermal_data"] == {"living_room": {"x": 1}}
    assert migrated["occupancy_trends"] == {"living_room": {"y": 2}}


def test_migrate_store_data_promotes_a_single_shared_thermostat():
    migrated = migrate_store_data(
        {
            "rooms": {
                "living_room": {"area_id": "living_room", "devices": [{"entity_id": "climate.hall"}]},
                "bedroom": {"area_id": "bedroom", "thermostats": ["climate.hall"]},
            },
            "settings": {},
        }
    )

    assert migrated["settings"]["central_plant"]["thermostat_entity"] == "climate.hall"


def test_migrate_store_data_does_not_promote_ambiguous_thermostats():
    migrated = migrate_store_data(
        {
            "rooms": {
                "living_room": {"area_id": "living_room", "devices": [{"entity_id": "climate.living"}]},
                "bedroom": {"area_id": "bedroom", "devices": [{"entity_id": "climate.bedroom"}]},
            },
            "settings": {},
        }
    )

    assert migrated["settings"]["central_plant"]["thermostat_entity"] == ""


def test_migrate_store_data_keeps_a_configured_thermostat():
    migrated = migrate_store_data(
        {
            "rooms": {"living_room": {"area_id": "living_room", "devices": [{"entity_id": "climate.hall"}]}},
            "settings": {"central_plant": {"thermostat_entity": "climate.choosen"}},
        }
    )

    assert migrated["settings"]["central_plant"]["thermostat_entity"] == "climate.choosen"


def test_migrate_store_data_strips_retired_settings():
    migrated = migrate_store_data(
        {
            "settings": {
                "heating_threshold": 1.0,
                "compressor_groups": [{"id": "g"}],
                "valve_last_actuation": {},
                "ac_head_offsets": {},
                "keep_me": True,
            }
        }
    )

    for key in ("heating_threshold", "compressor_groups", "valve_last_actuation", "ac_head_offsets"):
        assert key not in migrated["settings"]
    assert migrated["settings"]["keep_me"] is True


def test_migrate_room_temps_adds_split_fields():
    migrated = migrate_store_data({"rooms": {"living_room": {"area_id": "living_room", "comfort_temp": 22.5}}})

    room = migrated["rooms"]["living_room"]
    assert room["comfort_heat"] == 22.5
    assert room["comfort_cool"] == DEFAULT_COMFORT_COOL


def test_migrate_room_temps_keeps_existing_split_fields():
    migrated = migrate_store_data(
        {
            "rooms": {
                "living_room": {
                    "area_id": "living_room",
                    "comfort_temp": 19.0,
                    "comfort_heat": 22.0,
                    "eco_heat": 18.0,
                    "eco_cool": 26.0,
                }
            }
        }
    )

    room = migrated["rooms"]["living_room"]
    assert room["comfort_heat"] == 22.0
    assert room["eco_heat"] == 18.0
    assert room["eco_cool"] == 26.0


@pytest.mark.parametrize(
    ("climate_mode", "expected"),
    [("cool_only", 22.0), ("heat_only", DEFAULT_COMFORT_COOL), ("auto", DEFAULT_COMFORT_COOL)],
)
def test_a_legacy_single_temp_is_the_cooling_target_in_a_cool_only_room(climate_mode, expected):
    migrated = migrate_store_data(
        {"rooms": {"living_room": {"area_id": "living_room", "climate_mode": climate_mode, "comfort_temp": 22.0}}}
    )

    assert migrated["rooms"]["living_room"]["comfort_cool"] == expected


def test_migrate_override_temp_auto_creates_a_dead_band():
    room = migrate_store_data({"rooms": {"living_room": {"area_id": "living_room", "override_temp": 22.0}}})["rooms"][
        "living_room"
    ]

    assert room["override_heat"] == 22.0
    assert room["override_cool"] == DEFAULT_COMFORT_COOL


def test_migrate_override_temp_heat_only_has_one_side():
    room = migrate_store_data(
        {"rooms": {"living_room": {"area_id": "living_room", "climate_mode": "heat_only", "override_temp": 22.0}}}
    )["rooms"]["living_room"]

    assert room["override_heat"] == 22.0
    assert room["override_cool"] is None


def test_migrate_override_temp_cool_only_has_one_side():
    room = migrate_store_data(
        {"rooms": {"living_room": {"area_id": "living_room", "climate_mode": "cool_only", "override_temp": 22.0}}}
    )["rooms"]["living_room"]

    assert room["override_heat"] is None
    assert room["override_cool"] == 22.0


def test_migrate_override_temp_none_is_dropped():
    room = migrate_store_data({"rooms": {"living_room": {"area_id": "living_room", "override_temp": None}}})["rooms"][
        "living_room"
    ]

    assert "override_temp" not in room
    assert room.get("override_heat") is None


def test_migrate_override_temp_keeps_explicit_split_fields():
    room = migrate_store_data(
        {
            "rooms": {
                "living_room": {
                    "area_id": "living_room",
                    "override_temp": 22.0,
                    "override_heat": 20.0,
                    "override_cool": 25.0,
                }
            }
        }
    )["rooms"]["living_room"]

    assert room["override_heat"] == 20.0
    assert room["override_cool"] == 25.0


def test_migrate_window_sensors_fills_the_canonical_field():
    room = migrate_store_data(
        {"rooms": {"living_room": {"area_id": "living_room", "window_sensors": ["binary_sensor.w"]}}}
    )["rooms"]["living_room"]

    assert room["exterior_window_sensors"] == ["binary_sensor.w"]
    assert room["window_sensors"] == ["binary_sensor.w"]


def test_migrate_window_sensors_prefers_the_canonical_field():
    room = migrate_store_data(
        {
            "rooms": {
                "living_room": {
                    "area_id": "living_room",
                    "window_sensors": ["binary_sensor.old"],
                    "exterior_window_sensors": ["binary_sensor.new"],
                }
            }
        }
    )["rooms"]["living_room"]

    assert room["exterior_window_sensors"] == ["binary_sensor.new"]
    assert room["window_sensors"] == ["binary_sensor.new"]


@pytest.mark.parametrize(
    ("sensors", "legacy_flag", "expected"),
    [
        ([], False, OCCUPANCY_MODE_FOLLOW_SCHEDULE),
        (["binary_sensor.motion"], True, OCCUPANCY_MODE_REACTIVE_TREND),
        (["binary_sensor.motion"], False, OCCUPANCY_MODE_SCHEDULE_ASSUMED),
    ],
)
def test_migrate_occupancy_mode_from_the_legacy_flags(sensors, legacy_flag, expected):
    room = migrate_store_data(
        {
            "rooms": {
                "living_room": {
                    "area_id": "living_room",
                    "occupancy_sensors": sensors,
                    "occupancy_eco_enabled": legacy_flag,
                }
            }
        }
    )["rooms"]["living_room"]

    assert room["occupancy_mode"] == expected
    assert room["occupancy_eco_enabled"] is (expected == OCCUPANCY_MODE_REACTIVE_TREND)


def test_migrate_store_data_adds_the_room_defaults():
    room = migrate_store_data({"rooms": {"living_room": {"area_id": "living_room"}}})["rooms"]["living_room"]

    assert room["interior_door_delay"] == 60
    assert room["door_closed_mix_factor"] == 0.4
    assert room["occupancy_dwell_minutes"] == 3.0


@pytest.mark.asyncio
async def test_async_migrate_func_returns_migrated_data(store):
    migrated = await _RoomMindHaStore._async_migrate_func(
        store._store, 1, 0, {"rooms": {}, "settings": {"heating_threshold": 1}}
    )

    assert "central_plant" in migrated["settings"]
    assert "heating_threshold" not in migrated["settings"]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_settings_default_to_empty(store):
    await store.async_load()

    assert store.get_settings() == {}


@pytest.mark.asyncio
async def test_save_settings_merges_and_persists(store):
    await store.async_load()

    await store.async_save_settings({"climate_control_active": False})
    await store.async_save_settings({"vacation_temp": 15.0})

    settings = store.get_settings()
    assert settings["climate_control_active"] is False
    assert settings["vacation_temp"] == 15.0


@pytest.mark.asyncio
async def test_save_settings_merges_a_partial_central_plant(store):
    await store.async_load()

    await store.async_save_settings({"central_plant": {"thermostat_entity": "climate.hall"}})
    await store.async_save_settings({"central_plant": {"nudge_delta_c": 1.0}})

    central = store.get_settings()["central_plant"]
    assert central["thermostat_entity"] == "climate.hall"
    assert central["nudge_delta_c"] == 1.0
    assert central["min_deadband_c"] == DEFAULT_CENTRAL_PLANT_SETTINGS["min_deadband_c"]


@pytest.mark.asyncio
async def test_save_settings_initialises_central_plant_with_defaults(store):
    await store.async_load()

    await store.async_save_settings({"central_plant": {"thermostat_entity": "climate.hall"}})

    assert store.get_settings()["central_plant"]["thermostat_entity"] == "climate.hall"
    assert "fallback_heat_c" in store.get_settings()["central_plant"]


@pytest.mark.asyncio
async def test_orphaned_settings_are_dropped_on_load(store):
    store._store.async_load = AsyncMock(
        return_value={"rooms": {}, "settings": {"heating_threshold": 1, "keep": True}, "thermal_data": {}}
    )

    await store.async_load()

    assert store.get_settings() == {"central_plant": DEFAULT_CENTRAL_PLANT_SETTINGS, "keep": True}


# ---------------------------------------------------------------------------
# Learned data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thermal_data_round_trips(store):
    await store.async_load()
    assert store.get_thermal_data() == {}

    await store.async_save_thermal_data({"living_room": {"x": 1}})

    assert store.get_thermal_data() == {"living_room": {"x": 1}}
    store.get_thermal_data()["living_room"]["x"] = 2
    assert store.get_thermal_data()["living_room"]["x"] == 1


@pytest.mark.asyncio
async def test_clear_thermal_data_room_and_all(store):
    await store.async_load()
    await store.async_save_thermal_data({"living_room": {"x": 1}, "bedroom": {"x": 2}})

    await store.async_clear_thermal_data_room("living_room")

    assert store.get_thermal_data() == {"bedroom": {"x": 2}}

    await store.async_clear_all_thermal_data()

    assert store.get_thermal_data() == {}


@pytest.mark.asyncio
async def test_occupancy_trends_round_trip(store):
    await store.async_load()
    assert store.get_occupancy_trends() == {}

    await store.async_save_occupancy_trends({"living_room": {"trend": {"days_observed": 4.0}}})

    trends = store.get_occupancy_trends()
    assert trends["living_room"]["trend"]["days_observed"] == 4.0
    trends["living_room"]["trend"]["days_observed"] = 0.0
    assert store.get_occupancy_trends()["living_room"]["trend"]["days_observed"] == 4.0


@pytest.mark.asyncio
async def test_occupancy_trends_are_cleared_per_room_and_all(store):
    await store.async_load()
    await store.async_save_occupancy_trends({"living_room": {"a": 1}, "bedroom": {"b": 2}})

    await store.async_clear_occupancy_trends_room("living_room")

    assert store.get_occupancy_trends() == {"bedroom": {"b": 2}}

    await store.async_clear_all_occupancy_trends()

    assert store.get_occupancy_trends() == {}


@pytest.mark.asyncio
async def test_an_empty_store_load_clears_everything(store):
    store._data = {"living_room": {}}
    store._settings = {"a": 1}
    store._thermal_data = {"living_room": {}}
    store._occupancy_trends = {"living_room": {}}
    store._store.async_load = AsyncMock(return_value=None)

    await store.async_load()

    assert store.get_rooms() == {}
    assert store.get_settings() == {}
    assert store.get_thermal_data() == {}
    assert store.get_occupancy_trends() == {}


@pytest.mark.asyncio
async def test_a_loaded_store_is_saved_when_migrated(store):
    store._store.async_load = AsyncMock(
        return_value={"rooms": {"living_room": {"area_id": "living_room", "comfort_temp": 22.0}}, "settings": {}}
    )

    await store.async_load()

    store._store.async_save.assert_awaited_once()
    assert store.get_room("living_room")["comfort_heat"] == 22.0


@pytest.mark.asyncio
async def test_a_current_store_is_not_saved_again_on_load(store):
    store._store.async_load = AsyncMock(return_value=migrate_store_data({"rooms": {}, "settings": {}}))

    await store.async_load()

    store._store.async_save.assert_not_awaited()
