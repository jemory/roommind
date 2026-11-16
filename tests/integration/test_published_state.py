"""The published cycle payload: what the entities, the panel and the websocket API read."""

from __future__ import annotations

import time

import pytest

from custom_components.roommind.const import MODE_HEATING

from .conftest import DEFAULT_THERMOSTAT_ATTRIBUTES, THERMOSTAT_ENTITY, Harness, make_room

pytestmark = pytest.mark.asyncio

# Keys ``roommind/rooms/list`` forwards per room (websocket_api._build_room_live).
ROOM_LIVE_KEYS = {
    "current_temp",
    "current_humidity",
    "effective_temp",
    "target_temp",
    "heat_target",
    "cool_target",
    "mode",
    "heating_power",
    "device_setpoint",
    "window_open",
    "override_active",
    "override_type",
    "override_heat",
    "override_cool",
    "override_until",
    "override_suppressed",
    "active_schedule_index",
    "confidence",
    "mpc_active",
    "presence_away",
    "occupancy_vacant",
    "priority_weight",
    "occupancy_probability",
    "occupancy_dwell",
    "interior_door_open",
    "exterior_window_open",
    "fan_circulating",
    "beta_fan_rate",
    "co2",
    "voc",
    "air_quality_bad",
    "air_quality_reason",
    "ventilation_active",
    "mold_risk_level",
    "mold_surface_rh",
    "mold_surface_temp",
    "mold_prevention_active",
    "mold_prevention_delta",
    "n_observations",
    "blind_position",
    "cover_auto_paused",
    "cover_override_until",
    "cover_forced_reason",
    "cover_reason",
    "active_cover_schedule_index",
    "schedule_temp_warnings",
    "price_level",
    "preheat",
    "price_shift_heat",
    "price_shift_cool",
}

# Keys the panel reads from ``central_plant`` (frontend/src/types/index.ts).
CENTRAL_PLANT_KEYS = {
    "configured",
    "thermostat_entity",
    "thermostat_area_id",
    "action",
    "commanded_action",
    "observed_hvac_action",
    "fan_circulating",
    "thermostat_temp",
    "house_ref_temp",
    "commanded_low",
    "commanded_high",
    "commanded_single",
    "driving_rooms",
    "decision_reason",
    "guard_active",
    "guard_reason",
    "guard_until",
    "wet_coil_charge",
    "wet_coil_locked",
    "exterior_paused_rooms",
    "plant_override_active",
    "plant_override_heat",
    "plant_override_cool",
    "plant_override_until",
}


async def test_every_room_live_key_is_published(harness: Harness):
    await harness.seed({"living_room": make_room()})

    data = await harness.cycle()

    missing = ROOM_LIVE_KEYS - set(data["rooms"]["living_room"])
    assert missing == set()


async def test_every_central_plant_key_is_published(harness: Harness):
    await harness.seed({"living_room": make_room()})

    data = await harness.cycle()

    missing = CENTRAL_PLANT_KEYS - set(data["central_plant"])
    assert missing == set()


async def test_the_commanded_setpoints_are_published(harness: Harness):
    await harness.seed({"living_room": make_room()})

    data = await harness.cycle()

    plant = data["central_plant"]
    assert plant["commanded_low"] is not None
    assert plant["commanded_high"] is not None
    assert plant["commanded_low"] < plant["commanded_high"]
    assert plant["configured"] is True


async def test_the_observed_action_is_published(harness: Harness, fake_states):
    fake_states.set(THERMOSTAT_ENTITY, "heat", {**DEFAULT_THERMOSTAT_ATTRIBUTES, "hvac_action": "heating"})
    await harness.seed({"living_room": make_room()})

    data = await harness.cycle()

    assert data["central_plant"]["observed_hvac_action"] == "heating"
    assert data["central_plant"]["commanded_action"] == MODE_HEATING
    # A running furnace moves air through the shared blower.
    assert data["central_plant"]["fan_circulating"] is True


async def test_the_plant_override_fields_are_published(harness: Harness):
    await harness.seed(
        {"living_room": make_room()},
        settings={
            "plant_override_heat": 22.0,
            "plant_override_cool": 25.0,
            "plant_override_until": time.time() + 3600,
        },
    )

    data = await harness.cycle()

    plant = data["central_plant"]
    assert plant["plant_override_active"] is True
    assert plant["plant_override_heat"] == 22.0
    assert plant["plant_override_cool"] == 25.0
    assert plant["plant_override_until"] is not None


async def test_an_exterior_pause_lists_the_rooms(harness: Harness, fake_states):
    fake_states.set("binary_sensor.window", "on")
    await harness.seed(
        {"living_room": make_room({"exterior_window_sensors": ["binary_sensor.window"], "window_open_delay": 0})},
        settings={"central_plant": {"thermostat_entity": THERMOSTAT_ENTITY, "exterior_open_policy": "pause_always"}},
    )

    data = await harness.cycle()

    assert data["central_plant"]["exterior_paused_rooms"] == ["living_room"]


async def test_the_room_state_carries_the_beta_fan_rate(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    await harness.cycle()

    rate = coordinator.rooms["living_room"]["beta_fan_rate"]
    assert rate is not None
    assert rate >= 0.0


async def test_an_unmodelled_room_publishes_no_beta_fan_rate(harness: Harness):
    await harness.seed({"living_room": make_room({"temperature_sensor": ""})})

    data = await harness.cycle()

    # No temperature, so no estimator and no fan mixing rate.
    assert data["rooms"]["living_room"]["beta_fan_rate"] is None


async def test_the_recent_temps_series_grows(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})

    await harness.cycle()
    series = coordinator.recent_temps.get("living_room")

    assert series
    _ts, temp, target = series[-1]
    assert temp == 18.0
    assert target == 21.0
