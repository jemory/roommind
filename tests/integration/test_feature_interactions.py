"""Features meeting each other: overrides, presence, vacation, windows, air, mold, covers."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

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


# ---------------------------------------------------------------------------
# House layer (T4): MPC inputs, priority inputs and the live payload
# ---------------------------------------------------------------------------

HOUSE_ON = {
    "house": {
        "enabled": True,
        "heat_c": 22.0,
        "cool_c": 25.0,
        "eco_heat_c": 16.0,
        "eco_cool_c": 28.0,
        "climate_mode": "auto",
    }
}
HOUSE_HEAT_ONLY = {
    "house": {
        "enabled": True,
        "heat_c": 22.0,
        "cool_c": 25.0,
        "eco_heat_c": 16.0,
        "eco_cool_c": 28.0,
        "climate_mode": "heat_only",
    }
}
HOUSE_OFF = {"house": {"enabled": False}}


async def _capture_mpc_rooms(harness: Harness) -> list[Any]:
    """Cycle once and return the ``RoomHorizonInput`` list handed to the MPC."""
    coordinator = harness.coordinator
    assert coordinator is not None
    captured: list[Any] = []
    real_evaluate = coordinator._mpc_controller.evaluate

    def _spy(*args: Any, **kwargs: Any) -> Any:
        captured.extend(kwargs.get("rooms", []))
        return real_evaluate(*args, **kwargs)

    coordinator._mpc_controller.evaluate = MagicMock(side_effect=_spy)
    await harness.cycle()
    return captured


async def test_mpc_comfort_series_come_from_the_house_for_a_follower(harness: Harness, fake_states):
    fake_states.set("sensor.office_temp", "18.0", {"unit_of_measurement": "°C"})
    captured = await _capture_mpc_rooms(await _seed_with_house(harness, HOUSE_ON))

    office = next(room for room in captured if room.area_id == "office")
    assert office.comfort_heat_series[0] == 22.0
    assert office.comfort_cool_series[0] == 25.0


async def test_house_heat_only_mode_masks_the_followers_cool_series(harness: Harness, fake_states):
    fake_states.set("sensor.living_temp", "18.0", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.office_temp", "18.0", {"unit_of_measurement": "°C"})
    captured = await _capture_mpc_rooms(
        await _seed_with_house(
            harness,
            HOUSE_HEAT_ONLY,
            rooms={
                "living_room": make_room(),
                "office": make_room({"temperature_sensor": "sensor.office_temp"}),
            },
        )
    )

    assert captured
    for room in captured:
        assert all(heat == 22.0 for heat in room.comfort_heat_series)
        assert all(cool is None for cool in room.comfort_cool_series)


async def _seed_with_house(
    harness: Harness,
    settings: dict,
    rooms: dict | None = None,
) -> Harness:
    """Seed the harness with the given house settings (house layer under test)."""
    await harness.seed(
        rooms or {"living_room": make_room(), "office": make_room({"temperature_sensor": "sensor.office_temp"})},
        settings=settings,
    )
    return harness


async def test_priority_weight_of_a_follower_without_a_setting_is_its_occupancy_weight(harness: Harness):
    await harness.seed(
        {"living_room": make_room()},
        settings=HOUSE_ON,
    )

    data = await harness.cycle()

    # follow_schedule with no sensors and no schedule block: occupancy weight 1.0.
    assert data["rooms"]["living_room"]["priority_weight"] == pytest.approx(1.0)


async def test_priority_weight_adds_the_room_setting_bump(harness: Harness):
    harness.hass.services.async_call = AsyncMock(
        side_effect=schedule_service(week_blocks("00:00:00", "23:59:59", {"heat_temperature": 19.0}))
    )
    harness.hass.states.set("schedule.living", "on", {})
    await harness.seed(
        {"living_room": make_room({"schedules": [{"entity_id": "schedule.living"}]})},
        settings=HOUSE_ON,
    )

    data = await harness.cycle()

    # Base 1.0 + ROOM_SETTING_TIE_WEIGHT (0.02).
    assert data["rooms"]["living_room"]["priority_weight"] == pytest.approx(1.02)


async def test_priority_weight_adds_the_manual_override_bump(harness: Harness):
    await harness.seed(
        {
            "living_room": make_room(
                {
                    "override_heat": 23.0,
                    "override_cool": 26.0,
                    "override_until": time.time() + 3600,
                    "override_type": "custom",
                }
            )
        },
        settings=HOUSE_ON,
    )

    data = await harness.cycle()

    # Base 1.0 + MANUAL_OVERRIDE_TIE_WEIGHT (0.05).
    assert data["rooms"]["living_room"]["priority_weight"] == pytest.approx(1.05)


async def test_live_house_payload_carries_every_consumed_key(harness: Harness, fake_states):
    fake_states.set("schedule.house", "on", {"friendly_name": "Whole home"})
    harness.hass.services.async_call = AsyncMock(
        side_effect=schedule_service(week_blocks("00:00:00", "23:59:59", {"heat_temperature": 21.0}))
    )
    await harness.seed(
        {
            "living_room": make_room(),
            "office": make_room({"temperature_sensor": "sensor.office_temp"}),
        },
        settings={
            "house": {
                "enabled": True,
                "heat_c": 22.0,
                "cool_c": 25.0,
                "eco_heat_c": 16.0,
                "eco_cool_c": 28.0,
                "climate_mode": "auto",
                "schedules": [{"entity_id": "schedule.house"}],
            }
        },
    )

    data = await harness.cycle()

    house = data["house"]
    # Resolved settings keys are always present.
    assert house["enabled"] is True
    assert house["heat_c"] == 22.0
    assert house["eco_heat_c"] == 16.0
    assert house["climate_mode"] == "auto"
    assert house["schedules"] == [{"entity_id": "schedule.house"}]
    # Derived live fields the card / panel / entity read.
    assert house["active"] is True
    assert house["band"] == {"heat_c": 21.0, "cool_c": 25.0}
    assert house["source"] == "schedule"
    assert house["override_active"] is False
    assert house["schedule_entity"] == "schedule.house"
    assert house["schedule_name"] == "Whole home"
    assert house["schedule_block"] == {"from": "00:00:00", "to": "23:59:59"}
    assert "next_change" in house
    assert house["following_rooms"] == 2


async def test_live_house_payload_exists_before_and_without_an_active_house(harness: Harness):
    # No cycle yet: the coordinator always advertises a house payload.
    await harness.seed({"living_room": make_room()}, settings=HOUSE_OFF)

    data = await harness.cycle()

    house = data["house"]
    assert house["active"] is False
    assert house["enabled"] is False
    # Even inert, the payload carries the configured house targets as the band.
    assert house["band"] == {"heat_c": 21.0, "cool_c": 24.0}
    assert house["source"] is None
    assert house["override_active"] is False
    assert house["schedule_entity"] == ""
    assert house["schedule_name"] == ""
    assert house["schedule_block"] is None
    assert house["next_change"] is None
    assert house["following_rooms"] == 0
