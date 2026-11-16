"""Features meeting each other: overrides, presence, vacation, windows, air, mold, covers."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock

import pytest

from custom_components.roommind.const import (
    MODE_IDLE,
    TARGET_REASON_AWAY,
    TARGET_REASON_OVERRIDE,
    TARGET_REASON_PLANT_OVERRIDE,
    TARGET_REASON_VACATION,
)

from .conftest import THERMOSTAT_ENTITY, Harness, make_room

pytestmark = pytest.mark.asyncio

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def schedule_service(blocks: dict) -> Any:
    async def _call(domain: str, service: str, data: dict, **kwargs: Any) -> Any:
        if domain == "schedule" and service == "get_schedule":
            return {data["entity_id"]: blocks}
        return {}

    return _call


def week_blocks(from_time: str, to_time: str, data: dict) -> dict:
    return {day: [{"from": from_time, "to": to_time, "data": dict(data)}] for day in WEEKDAYS}


async def test_a_room_override_beats_the_schedule_and_the_price(harness: Harness, fake_states):
    harness.hass.services.async_call = AsyncMock(
        side_effect=schedule_service(week_blocks("00:00:00", "23:59:59", {"heat_temperature": 19.0}))
    )
    fake_states.set("schedule.living", "on", {})
    await harness.seed(
        {
            "living_room": make_room(
                {
                    "schedules": [{"entity_id": "schedule.living"}],
                    "price_shift": 2.0,
                    "override_heat": 23.0,
                    "override_cool": 25.0,
                    "override_until": time.time() + 3600,
                    "override_type": "custom",
                }
            )
        }
    )

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["heat_target"] == 23.0
    assert room["target_reason"] == TARGET_REASON_OVERRIDE
    assert room["price_shift_heat"] == 0.0


async def test_presence_away_clears_a_room_override_when_opted_in(harness: Harness, fake_states):
    fake_states.set("person.alex", "not_home")
    await harness.seed(
        {
            "living_room": make_room(
                {
                    "presence_persons": ["person.alex"],
                    "override_heat": 23.0,
                    "override_cool": 25.0,
                    "override_until": time.time() + 3600,
                    "override_type": "custom",
                }
            )
        },
        settings={
            "presence_enabled": True,
            "presence_persons": ["person.alex"],
            "presence_clears_override": True,
        },
    )

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["target_reason"] == TARGET_REASON_AWAY
    assert room["heat_target"] == 17.0
    assert room["override_active"] is True
    assert room["override_suppressed"] is True


async def test_a_room_override_outranks_vacation(harness: Harness):
    await harness.seed(
        {
            "living_room": make_room(
                {
                    "override_heat": 22.0,
                    "override_cool": 24.0,
                    "override_until": None,
                    "override_type": "custom",
                }
            )
        },
        settings={"vacation_until": time.time() + 86400, "vacation_temp": 15.0},
    )

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["heat_target"] == 22.0
    assert room["target_reason"] == TARGET_REASON_OVERRIDE


async def test_vacation_without_an_override_holds_the_setback(harness: Harness):
    await harness.seed(
        {"living_room": make_room()},
        settings={"vacation_until": time.time() + 86400, "vacation_temp": 15.0},
    )

    data = await harness.cycle()

    assert data["rooms"]["living_room"]["target_reason"] == TARGET_REASON_VACATION
    assert data["central_plant"]["action"] == MODE_IDLE


async def test_a_room_override_lifts_the_window_pause_but_not_its_own_room(harness: Harness, fake_states):
    fake_states.set("binary_sensor.living_window", "on")
    fake_states.set("binary_sensor.office_window", "off")
    await harness.seed(
        {
            "living_room": make_room(
                {
                    "exterior_window_sensors": ["binary_sensor.living_window"],
                    "window_open_delay": 0,
                    "override_heat": 23.0,
                    "override_cool": 25.0,
                    "override_until": time.time() + 3600,
                    "override_type": "custom",
                }
            ),
            "office": make_room(
                {
                    "exterior_window_sensors": ["binary_sensor.office_window"],
                    "window_open_delay": 0,
                    "temperature_sensor": "sensor.office_temp",
                }
            ),
        },
        settings={"central_plant": {"thermostat_entity": THERMOSTAT_ENTITY, "exterior_open_policy": "pause_always"}},
    )
    fake_states.set("sensor.office_temp", "18.0", {"unit_of_measurement": "°C"})

    data = await harness.cycle()

    assert data["central_plant"]["any_override"] is True
    assert data["central_plant"]["paused"] is False
    # The window room still does not contribute; the office drives the plant.
    assert data["rooms"]["living_room"]["mode"] == MODE_IDLE
    assert data["central_plant"]["action"] == "heating"


async def test_the_plant_override_wins_over_the_schedule(harness: Harness):
    await harness.seed(
        {"living_room": make_room()},
        settings={
            "plant_override_heat": 22.5,
            "plant_override_cool": 25.5,
            "plant_override_until": time.time() + 3600,
        },
    )

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["target_reason"] == TARGET_REASON_PLANT_OVERRIDE
    assert room["heat_target"] == 22.5
    assert data["central_plant"]["plant_override_active"] is True
    assert data["central_plant"]["plant_override_heat"] == 22.5


async def test_two_rooms_share_one_ventilation_unit(harness: Harness, fake_states):
    fake_states.set("sensor.living_co2", "1400")
    fake_states.set("sensor.office_co2", "500")
    fake_states.set("fan.shared", "off")
    await harness.seed(
        {
            "living_room": make_room(
                {"co2_sensor": "sensor.living_co2", "co2_threshold": 1000, "ventilation_entity": "fan.shared"}
            ),
            "office": make_room(
                {
                    "co2_sensor": "sensor.office_co2",
                    "co2_threshold": 1000,
                    "ventilation_entity": "fan.shared",
                    "temperature_sensor": "sensor.office_temp",
                }
            ),
        }
    )
    fake_states.set("sensor.office_temp", "20.0", {"unit_of_measurement": "°C"})

    data = await harness.cycle()

    assert data["rooms"]["living_room"]["air_quality_bad"] is True
    assert data["rooms"]["office"]["air_quality_bad"] is False
    # The unit runs for the room that needs it, and both read it as running.
    assert data["rooms"]["living_room"]["ventilation_active"] is True
    assert data["rooms"]["office"]["ventilation_active"] is True
    assert data["central_plant"]["action"] == "heating"


async def test_a_moldy_wall_keeps_the_room_from_being_switched_off(harness: Harness, fake_states):
    fake_states.set("sensor.room_humidity", "92.0")
    fake_states.set("sensor.room_temp", "15.0", {"unit_of_measurement": "°C"})
    await harness.seed(
        {"living_room": make_room()},
        settings={
            "mold_prevention_enabled": True,
            "mold_prevention_intensity": "strong",
            "schedule_off_action": "off",
        },
    )

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["mold_prevention_active"] is True
    assert room["force_off"] is False
    assert room["heat_target"] > 20.0


async def test_covers_are_processed_with_the_house_demand(harness: Harness, fake_states):
    fake_states.set(
        "cover.living",
        "open",
        {"current_position": 100, "supported_features": 15},
    )
    await harness.seed(
        {
            "living_room": make_room(
                {
                    "covers": ["cover.living"],
                    "covers_auto_enabled": True,
                    "covers_deploy_threshold": 0.5,
                }
            )
        }
    )

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert isinstance(room["cover_reason"], str) and room["cover_reason"]
    assert room["cover_forced_reason"] == ""
    assert room["active_cover_schedule_index"] == -1


async def test_the_climate_mode_is_taken_from_the_rooms(harness: Harness, fake_states):
    await harness.seed(
        {
            "heat_room": make_room({"climate_mode": "heat_only", "temperature_sensor": "sensor.heat_temp"}),
            "cool_room": make_room({"climate_mode": "cool_only", "temperature_sensor": "sensor.cool_temp"}),
        }
    )
    fake_states.set("sensor.heat_temp", "18.0", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.cool_temp", "27.0", {"unit_of_measurement": "°C"})
    # Between the cooling minimum (16) and the heating maximum (22) the plant may do both.
    fake_states.set("sensor.outdoor_temp", "20.0", {"unit_of_measurement": "°C"})

    data = await harness.cycle()

    assert data["central_plant"]["climate_mode"] == "auto"
    assert "heating" in data["central_plant"]["legal_actions"]
    assert "cooling" in data["central_plant"]["legal_actions"]
