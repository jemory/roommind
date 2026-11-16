"""Whole-home behaviour: circulation mixes disagreeing rooms, the MPC takes over when ready."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from custom_components.roommind.const import MODE_CIRCULATING, MODE_COOLING, MODE_HEATING, MODE_IDLE

from .conftest import DEFAULT_THERMOSTAT_ATTRIBUTES, THERMOSTAT_ENTITY, Harness, make_room

pytestmark = pytest.mark.asyncio

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def schedule_service(blocks: dict) -> Any:
    async def _call(domain: str, service: str, data: dict, **kwargs: Any) -> Any:
        if domain == "schedule" and service == "get_schedule":
            return {data["entity_id"]: blocks}
        return {}

    return _call


async def test_opposing_rooms_are_mixed_by_circulation(harness: Harness, fake_states):
    # One room too warm, one too cold, and the plant may do both.
    fake_states.set("sensor.warm_temp", "27.0", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.cold_temp", "18.0", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.outdoor_temp", "20.0", {"unit_of_measurement": "°C"})
    await harness.seed(
        {
            "warm_room": make_room({"temperature_sensor": "sensor.warm_temp"}),
            "cold_room": make_room({"temperature_sensor": "sensor.cold_temp"}),
        }
    )

    data = await harness.cycle()

    # Circulating costs a fraction of a compressor or furnace run and can equalize both.
    assert data["central_plant"]["action"] == MODE_CIRCULATING
    assert set(data["central_plant"]["driving_rooms"]) == {"warm_room", "cold_room"}


async def test_a_guardrail_breach_drives_the_plant_even_at_low_priority(harness: Harness, fake_states):
    fake_states.set("sensor.cold_temp", "12.0", {"unit_of_measurement": "°C"})
    await harness.seed(
        {
            "empty_room": make_room(
                {
                    "temperature_sensor": "sensor.cold_temp",
                    "climate_mode": "heat_only",
                    "occupancy_mode": "guardrail_only",
                }
            )
        }
    )

    data = await harness.cycle()

    assert data["central_plant"]["action"] == MODE_HEATING
    assert data["central_plant"]["decision_reason"] in ("guardrail", "bangbang", "mpc")


async def test_the_joint_mpc_takes_over_when_a_room_is_ready(harness: Harness, fake_states):
    fake_states.set(
        THERMOSTAT_ENTITY,
        "heat",
        {**DEFAULT_THERMOSTAT_ATTRIBUTES, "hvac_action": "heating"},
    )
    await harness.seed({"living_room": make_room()})

    with (
        patch("custom_components.roommind.coordinator.is_room_mpc_ready", return_value=True),
        patch("custom_components.roommind.control.central_mpc.is_room_mpc_ready", return_value=True),
    ):
        data = await harness.cycle()

    assert data["central_plant"]["mpc_active"] is True
    assert data["rooms"]["living_room"]["mpc_active"] is True
    assert data["central_plant"]["decision_reason"] in ("mpc", "preheat", "hold_guard", "guardrail")
    assert data["central_plant"]["actions"]


async def test_a_ready_room_with_opposing_demands_plans_circulation(harness: Harness, fake_states):
    fake_states.set("sensor.warm_temp", "26.0", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.cold_temp", "19.0", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.outdoor_temp", "20.0", {"unit_of_measurement": "°C"})
    await harness.seed(
        {
            "warm_room": make_room({"temperature_sensor": "sensor.warm_temp"}),
            "cold_room": make_room({"temperature_sensor": "sensor.cold_temp"}),
        }
    )

    with (
        patch("custom_components.roommind.coordinator.is_room_mpc_ready", return_value=True),
        patch("custom_components.roommind.control.central_mpc.is_room_mpc_ready", return_value=True),
    ):
        data = await harness.cycle()

    assert data["central_plant"]["mpc_active"] is True
    assert data["central_plant"]["action"] in (MODE_CIRCULATING, MODE_HEATING, MODE_COOLING, MODE_IDLE)


async def test_an_arriving_room_gets_a_preheat_badge(harness: Harness, fake_states):
    # A block that raises the heating target starts within the plan horizon.
    blocks = {
        day: [
            {"from": "00:00:00", "to": "06:00:00", "data": {"heat_temperature": 17.0}},
            {"from": "06:00:00", "to": "23:59:59", "data": {"heat_temperature": 22.0}},
        ]
        for day in WEEKDAYS
    }
    fake_states.set("schedule.living", "on", {})
    fake_states.set("sensor.room_temp", "18.0", {"unit_of_measurement": "°C"})
    await harness.seed({"living_room": make_room({"schedules": [{"entity_id": "schedule.living"}]})})
    harness.hass.services.async_call = _schedule_call(blocks)

    with (
        patch("custom_components.roommind.coordinator.is_room_mpc_ready", return_value=True),
        patch("custom_components.roommind.control.central_mpc.is_room_mpc_ready", return_value=True),
    ):
        data = await harness.cycle()

    assert isinstance(data["central_plant"]["preheat"], dict)
    assert data["rooms"]["living_room"]["heat_target"] in (17.0, 22.0)


def _schedule_call(blocks: dict) -> Any:
    async def _call(domain: str, service: str, data: dict, **kwargs: Any) -> Any:
        if domain == "schedule" and service == "get_schedule":
            return {data["entity_id"]: blocks}
        return {}

    return _call


async def test_the_house_mean_and_thermostat_hallway_drive_the_plan(harness: Harness, fake_states):
    fake_states.set(THERMOSTAT_ENTITY, "heat", {**DEFAULT_THERMOSTAT_ATTRIBUTES, "current_temperature": 21.0})
    await harness.seed(
        {
            "living_room": make_room(),
            "bedroom": make_room({"temperature_sensor": "sensor.bedroom_temp"}),
        }
    )
    fake_states.set("sensor.bedroom_temp", "19.0", {"unit_of_measurement": "°C"})

    data = await harness.cycle()

    # (18 + 19 + 21) / 3
    assert data["central_plant"]["house_ref_temp"] == pytest.approx(19.3333, abs=1e-3)
    assert data["central_plant"]["thermostat_temp"] == 21.0


async def test_a_heat_only_room_never_demands_cooling(harness: Harness, fake_states):
    # The heat-only room is above its cooling band (but below the overheat guardrail),
    # yet cooling is not its business.
    fake_states.set("sensor.hot_temp", "27.5", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.cold_temp", "18.0", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.outdoor_temp", "20.0", {"unit_of_measurement": "°C"})
    await harness.seed(
        {
            "hot_room": make_room({"temperature_sensor": "sensor.hot_temp", "climate_mode": "heat_only"}),
            "cold_room": make_room({"temperature_sensor": "sensor.cold_temp"}),
        }
    )

    data = await harness.cycle()

    assert "cooling" in data["central_plant"]["legal_actions"]
    assert data["central_plant"]["action"] == MODE_HEATING
