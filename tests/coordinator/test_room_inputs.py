"""How each room's inputs reach the plant: sensors, openings, targets, covers, air."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.roommind.const import (
    MODE_COOLING,
    MODE_HEATING,
    MODE_IDLE,
    TARGET_REASON_AWAY,
    TARGET_REASON_HOUSE,
    TARGET_REASON_OVERRIDE,
    TARGET_REASON_PLANT_OVERRIDE,
    TARGET_REASON_VACANT,
    TARGET_REASON_VACATION,
)
from custom_components.roommind.utils.price_utils import LEVEL_CHEAP, PriceSignal, PriceSlot

from .conftest import DEFAULT_THERMOSTAT_ATTRIBUTES, THERMOSTAT_ENTITY, Harness, make_room

pytestmark = pytest.mark.asyncio

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def all_week_blocks(data: dict[str, Any]) -> dict:
    """The same schedule block on every weekday (the test does not pin the clock's day)."""
    return {day: [{"from": "00:00:00", "to": "23:59:59", "data": dict(data)}] for day in WEEKDAYS}


def schedule_service(blocks: dict) -> Any:
    """A ``hass.services.async_call`` that answers ``schedule.get_schedule``."""

    async def _call(domain: str, service: str, data: dict, **kwargs: Any) -> Any:
        if domain == "schedule" and service == "get_schedule":
            return {data["entity_id"]: blocks}
        return {}

    return _call


async def test_the_schedule_block_sets_the_targets(harness: Harness):
    harness.hass.services.async_call = AsyncMock(
        side_effect=schedule_service(all_week_blocks({"heat_temperature": 22.5, "cool_temperature": 25.0}))
    )
    await harness.seed(
        {"living_room": make_room({"schedules": [{"entity_id": "schedule.living"}]})},
    )
    harness.hass.states.set("schedule.living", "on", {})

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["heat_target"] == 22.5
    assert room["cool_target"] == 25.0
    assert room["target_reason"] == ""
    assert data["central_plant"]["action"] == MODE_HEATING


async def test_a_single_block_temperature_sets_both_sides(harness: Harness):
    harness.hass.services.async_call = AsyncMock(side_effect=schedule_service(all_week_blocks({"temperature": 20.0})))
    await harness.seed({"living_room": make_room({"schedules": [{"entity_id": "schedule.living"}]})})
    harness.hass.states.set("schedule.living", "on", {})

    data = await harness.cycle()

    assert data["rooms"]["living_room"]["heat_target"] == 20.0
    assert data["rooms"]["living_room"]["cool_target"] == 20.0


async def test_an_unreadable_block_temperature_falls_back_and_warns(harness: Harness):
    harness.hass.services.async_call = AsyncMock(side_effect=schedule_service(all_week_blocks({"temperature": 110.0})))
    await harness.seed({"living_room": make_room({"schedules": [{"entity_id": "schedule.living"}]})})
    harness.hass.states.set("schedule.living", "on", {})

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["heat_target"] == 21.0
    assert room["schedule_temp_warnings"]
    assert room["schedule_temp_warnings"][0]["value"] == "110.0"


async def test_a_schedule_that_is_off_uses_eco(harness: Harness):
    harness.hass.services.async_call = AsyncMock(side_effect=schedule_service({}))
    await harness.seed({"living_room": make_room({"schedules": [{"entity_id": "schedule.living"}]})})
    harness.hass.states.set("schedule.living", "off", {})

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["heat_target"] == 17.0
    assert room["cool_target"] == 27.0


async def test_a_room_override_outranks_the_schedule(harness: Harness):
    await harness.seed(
        {
            "living_room": make_room(
                {
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
    assert room["override_active"] is True


async def test_an_expired_override_is_ignored(harness: Harness):
    await harness.seed(
        {
            "living_room": make_room(
                {
                    "override_heat": 23.0,
                    "override_cool": 25.0,
                    "override_until": time.time() - 60,
                    "override_type": "custom",
                }
            )
        }
    )

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["heat_target"] == 21.0
    assert room["target_reason"] == ""


async def test_a_plant_override_outranks_the_schedule(harness: Harness):
    await harness.seed(
        {"living_room": make_room()},
        settings={
            "central_plant": {"thermostat_entity": THERMOSTAT_ENTITY},
            "plant_override_heat": 22.0,
            "plant_override_cool": None,
            "plant_override_until": time.time() + 3600,
        },
    )

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["heat_target"] == 22.0
    # A one-sided wall setpoint takes the other side from the schedule it replaces.
    assert room["cool_target"] == 24.0
    assert room["target_reason"] == TARGET_REASON_PLANT_OVERRIDE


async def test_vacation_holds_the_house_at_its_setback(harness: Harness):
    await harness.seed(
        {"living_room": make_room()},
        settings={"vacation_until": time.time() + 86400, "vacation_temp": 15.0},
    )

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["heat_target"] == 15.0
    assert room["target_reason"] == TARGET_REASON_VACATION


async def test_presence_away_drops_the_room_to_eco(harness: Harness, fake_states):
    fake_states.set("person.alex", "not_home")
    await harness.seed(
        {"living_room": make_room({"presence_persons": ["person.alex"]})},
        settings={"presence_enabled": True, "presence_persons": ["person.alex"]},
    )

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["heat_target"] == 17.0
    assert room["presence_away"] is True
    assert room["target_reason"] == TARGET_REASON_AWAY


async def test_presence_away_can_switch_a_room_off(harness: Harness, fake_states):
    fake_states.set("person.alex", "not_home")
    await harness.seed(
        {"living_room": make_room({"presence_persons": ["person.alex"]})},
        settings={
            "presence_enabled": True,
            "presence_persons": ["person.alex"],
            "presence_away_action": "off",
        },
    )

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["heat_target"] is None
    assert room["cool_target"] is None
    assert room["force_off"] is True
    assert room["target_reason"] == TARGET_REASON_AWAY


async def test_an_empty_room_is_capped_at_eco(harness: Harness, fake_states):
    fake_states.set("binary_sensor.motion", "off")
    await harness.seed(
        {
            "living_room": make_room(
                {
                    "occupancy_sensors": ["binary_sensor.motion"],
                    "occupancy_mode": "reactive_and_trend",
                    "occupancy_dwell_minutes": 0,
                    "comfort_heat": 22.0,
                    "eco_heat": 18.0,
                }
            )
        }
    )

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["occupancy_vacant"] is True
    assert room["heat_target"] == 18.0
    assert room["target_reason"] == TARGET_REASON_VACANT


async def test_a_cool_only_room_drops_its_heating_side(harness: Harness, fake_states):
    fake_states.set("sensor.room_temp", "27.0", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.outdoor_temp", "31.0", {"unit_of_measurement": "°C"})
    await harness.seed({"living_room": make_room({"climate_mode": "cool_only"})})

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    # The heat side is not used by the plant, but the cool side drives the room.
    assert room["cool_target"] == 24.0
    assert room["target_temp"] == 24.0
    assert data["central_plant"]["climate_mode"] == "cool_only"
    assert data["central_plant"]["action"] == MODE_COOLING


async def test_a_heat_only_room_drops_its_cooling_side(harness: Harness):
    await harness.seed({"living_room": make_room({"climate_mode": "heat_only"})})

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["heat_target"] == 21.0
    assert room["target_temp"] == 21.0
    assert data["central_plant"]["climate_mode"] == "heat_only"


async def test_mixed_room_modes_leave_the_plant_in_auto(harness: Harness):
    await harness.seed(
        {
            "living_room": make_room({"climate_mode": "heat_only"}),
            "office": make_room({"climate_mode": "cool_only", "temperature_sensor": "sensor.office_temp"}),
        }
    )
    harness.hass.states.set("sensor.office_temp", "20.0", {"unit_of_measurement": "°C"})

    data = await harness.cycle()

    assert data["central_plant"]["climate_mode"] == "auto"


async def test_an_interior_door_closed_reduces_the_house_coupling(harness: Harness, fake_states):
    # Interior door sensors report "on" while the door is open, like a window contact.
    fake_states.set("binary_sensor.study_door", "off")
    await harness.seed(
        {
            "study": make_room(
                {
                    "interior_door_sensors": ["binary_sensor.study_door"],
                    "interior_door_delay": 0,
                    "door_closed_mix_factor": 0.4,
                }
            )
        }
    )

    data = await harness.cycle()

    room = data["rooms"]["study"]
    assert room["interior_door_open"] is False
    # The closed door keeps the room out of the blower's mixing path.
    assert room["priority_weight"] > 0.0


async def test_cover_positions_feed_the_shading_factor(harness: Harness, fake_states):
    fake_states.set("cover.living", "open", {"current_position": 0, "supported_features": 15})
    await harness.seed({"living_room": make_room({"covers": ["cover.living"]})})

    data = await harness.cycle()

    assert data["rooms"]["living_room"]["shading_factor"] < 1.0
    assert data["rooms"]["living_room"]["blind_position"] == 0


async def test_automatic_covers_run_with_the_house_context(harness: Harness, fake_states):
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
    assert room["cover_reason"]
    assert room["active_cover_schedule_index"] == -1


async def test_air_quality_turns_the_ventilation_on(harness: Harness, fake_states):
    fake_states.set("sensor.co2", "1400")
    fake_states.set("fan.vent", "off")
    await harness.seed(
        {
            "living_room": make_room(
                {
                    "co2_sensor": "sensor.co2",
                    "co2_threshold": 1000,
                    "ventilation_entity": "fan.vent",
                }
            )
        }
    )

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["co2"] == 1400.0
    assert room["air_quality_bad"] is True
    assert room["air_quality_reason"] == "co2"
    assert room["ventilation_active"] is True


async def test_clean_air_keeps_the_ventilation_off(harness: Harness, fake_states):
    fake_states.set("sensor.co2", "700")
    fake_states.set("fan.vent", "off")
    await harness.seed(
        {
            "living_room": make_room(
                {"co2_sensor": "sensor.co2", "co2_threshold": 1000, "ventilation_entity": "fan.vent"}
            )
        }
    )

    data = await harness.cycle()

    assert data["rooms"]["living_room"]["air_quality_bad"] is False
    assert data["rooms"]["living_room"]["ventilation_active"] is False


async def test_mold_prevention_lifts_the_heating_target(harness: Harness, fake_states):
    fake_states.set("sensor.room_humidity", "90.0")
    fake_states.set("sensor.room_temp", "16.0", {"unit_of_measurement": "°C"})
    await harness.seed(
        {"living_room": make_room()},
        settings={
            "mold_prevention_enabled": True,
            "mold_prevention_intensity": "medium",
            "mold_f_rsi": 0.8,
        },
    )

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["mold_prevention_active"] is True
    assert room["mold_boost"] == pytest.approx(2.0)
    assert room["heat_target"] == pytest.approx(23.0)


async def test_a_mold_boost_rescues_a_switched_off_room(harness: Harness, fake_states):
    fake_states.set("sensor.room_humidity", "90.0")
    fake_states.set("sensor.room_temp", "16.0", {"unit_of_measurement": "°C"})
    await harness.seed(
        {"living_room": make_room()},
        settings={
            "mold_prevention_enabled": True,
            "mold_prevention_intensity": "medium",
            "schedule_off_action": "off",
        },
    )
    # An explicit "off" band is what the room resolves to without a schedule.
    coordinator = harness.coordinator
    assert coordinator is not None
    coordinator._price_manager.signal = None

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["mold_prevention_active"] is True
    assert room["force_off"] is False


async def test_the_price_shift_moves_an_opted_in_room(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room({"price_shift": 2.0})})
    now = time.time()
    coordinator._price_manager.signal = PriceSignal(
        current_level=LEVEL_CHEAP,
        slots=[
            PriceSlot(start=now - 3600, end=now + 3600, price=0.01),
            PriceSlot(start=now + 3600, end=now + 2 * 3600, price=0.9),
        ],
        thresholds=(0.1, 0.2),
        now=now,
    )
    coordinator._price_manager.async_update = AsyncMock(return_value=None)

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    # Cheap now, expensive ahead: the cooling target is pre-cooled below its band.
    assert room["price_shift_cool"] < 0
    assert room["price_shift_heat"] == 0.0
    assert room["cool_target"] < 24.0


async def test_a_room_without_a_price_opt_in_is_not_shifted(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    now = time.time()
    coordinator._price_manager.signal = PriceSignal(
        current_level=LEVEL_CHEAP,
        slots=[
            PriceSlot(start=now - 3600, end=now + 3600, price=0.01),
            PriceSlot(start=now + 3600, end=now + 2 * 3600, price=0.9),
        ],
        thresholds=(0.1, 0.2),
        now=now,
    )
    coordinator._price_manager.async_update = AsyncMock(return_value=None)

    data = await harness.cycle()

    assert data["rooms"]["living_room"]["cool_target"] == 24.0


async def test_a_missing_sensor_keeps_the_last_reading(harness: Harness, fake_states):
    await harness.seed({"living_room": make_room()})
    await harness.cycle()

    fake_states.remove("sensor.room_temp")
    data = await harness.cycle()

    assert data["rooms"]["living_room"]["current_temp"] == 18.0


async def test_a_stale_cached_reading_is_dropped(harness: Harness, fake_states, monkeypatch):
    coordinator = await harness.seed({"living_room": make_room()})
    await harness.cycle()

    fake_states.remove("sensor.room_temp")
    # Age the cached reading past the staleness cap.
    coordinator._last_valid_temps["living_room"] = (18.0, time.monotonic() - 10_000)
    data = await harness.cycle()

    assert data["rooms"]["living_room"]["current_temp"] is None


async def test_a_room_without_a_schedule_treats_the_whole_day_as_on(harness: Harness):
    await harness.seed(
        {"living_room": make_room({"occupancy_sensors": ["binary_sensor.motion"], "occupancy_mode": "follow_schedule"})}
    )
    harness.hass.states.set("binary_sensor.motion", "off")

    data = await harness.cycle()

    # follow_schedule without a schedule: the room is assumed occupied all day.
    assert data["rooms"]["living_room"]["priority_weight"] == pytest.approx(1.0)


async def test_the_occupancy_trend_feeds_the_priority(harness: Harness, fake_states):
    await harness.seed(
        {
            "living_room": make_room(
                {
                    "occupancy_sensors": ["binary_sensor.motion"],
                    "occupancy_mode": "reactive_and_trend",
                }
            )
        },
        occupancy_trends={
            "living_room": {
                "trend": {
                    "weekday_bins": [1.0] * 96,
                    "weekend_bins": [1.0] * 96,
                    "days_observed": 7.0,
                }
            }
        },
    )
    fake_states.set("binary_sensor.motion", "off")

    data = await harness.cycle()

    assert data["rooms"]["living_room"]["occupancy_probability"] > 0.5
    assert data["rooms"]["living_room"]["priority_weight"] > 0.5


async def test_startup_holds_the_plant_until_the_first_reading(harness: Harness):
    await harness.seed({"living_room": make_room({"temperature_sensor": "sensor.missing"})})

    data = await harness.cycle()

    assert data["central_plant"]["startup_waiting"] is True
    assert data["central_plant"]["action"] == MODE_IDLE
    assert harness.hass.services.async_call.await_count == 0


async def test_after_the_startup_grace_the_plant_runs(harness: Harness, monkeypatch):
    coordinator = await harness.seed({"living_room": make_room({"temperature_sensor": "sensor.missing"})})
    coordinator._startup_ts = time.monotonic() - 10_000

    data = await harness.cycle()

    assert data["central_plant"]["startup_waiting"] is False
    assert coordinator._startup_guard_warned == {"living_room"}


async def test_a_pending_window_holds_the_plant(harness: Harness, fake_states, fake_registry):
    fake_registry.entities["binary_sensor.window"] = MagicMock()
    await harness.seed({"living_room": make_room({"exterior_window_sensors": ["binary_sensor.window"]})})
    # Registered but without a state yet: not started, so the window is pending.
    data = await harness.cycle()

    assert data["central_plant"]["startup_waiting"] is True
    assert data["central_plant"]["action"] == MODE_IDLE


async def test_a_missing_thermostat_state_still_plans(harness: Harness, fake_states):
    fake_states.remove(THERMOSTAT_ENTITY)
    await harness.seed({"living_room": make_room()})

    data = await harness.cycle()

    # The plan is made from the room state; without a readable thermostat no command is sent.
    assert data["central_plant"]["action"] == MODE_HEATING
    assert harness.hass.services.async_call.await_count == 0


async def test_circulation_is_displayed_as_the_plant_state(harness: Harness, fake_states):
    attributes = dict(DEFAULT_THERMOSTAT_ATTRIBUTES)
    fake_states.set(THERMOSTAT_ENTITY, "fan_only", {**attributes, "hvac_action": "fan"})
    await harness.seed({"living_room": make_room()})

    data = await harness.cycle()

    assert data["central_plant"]["display_action"] == "circulating"
    assert data["rooms"]["living_room"]["mode"] == "circulating"


async def test_a_humid_air_room_still_gets_valid_targets(harness: Harness, fake_states):
    fake_states.set("sensor.room_humidity", "unavailable")
    await harness.seed({"living_room": make_room()})

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["effective_temp"] == pytest.approx(18.0)
    assert room["heat_target"] == 21.0


async def test_the_thermostat_reading_is_used_as_a_room_sensor(harness: Harness):
    # The room has no own sensor but the thermostat sits in it.
    await harness.seed(
        {"living_room": make_room({"temperature_sensor": ""})},
        settings={"central_plant": {"thermostat_entity": THERMOSTAT_ENTITY, "thermostat_area_id": "living_room"}},
    )

    data = await harness.cycle()

    assert data["rooms"]["living_room"]["current_temp"] == 20.0
    assert data["central_plant"]["house_ref_temp"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# House layer (T4)
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
HOUSE_OFF = {"house": {"enabled": False}}


async def test_a_following_room_uses_the_house_band_and_reports_house(harness: Harness):
    await harness.seed({"living_room": make_room()}, settings=HOUSE_ON)

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert (room["heat_target"], room["cool_target"]) == (22.0, 25.0)
    assert room["target_reason"] == TARGET_REASON_HOUSE
    assert room["house_following"] is True


async def test_a_disabled_house_reproduces_v3_parity(harness: Harness):
    # A schedule gap with no house reports the room's own band and an empty reason.
    await harness.seed({"living_room": make_room()}, settings=HOUSE_OFF)

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert (room["heat_target"], room["cool_target"]) == (21.0, 24.0)
    assert room["target_reason"] == ""
    assert room["house_following"] is False
    assert data["house"]["active"] is False


async def test_a_room_schedule_with_temps_beats_the_house(harness: Harness):
    harness.hass.services.async_call = AsyncMock(
        side_effect=schedule_service(all_week_blocks({"heat_temperature": 19.0, "cool_temperature": 23.0}))
    )
    await harness.seed(
        {"living_room": make_room({"schedules": [{"entity_id": "schedule.living"}]})},
        settings=HOUSE_ON,
    )
    harness.hass.states.set("schedule.living", "on", {})

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    # A block carrying temperatures outranks the house and keeps the empty reason.
    assert (room["heat_target"], room["cool_target"]) == (19.0, 23.0)
    assert room["target_reason"] == ""


async def test_a_room_schedule_gap_falls_back_to_the_house(harness: Harness):
    # A block without temperatures is a gap: the following room inherits the house.
    harness.hass.services.async_call = AsyncMock(side_effect=schedule_service(all_week_blocks({})))
    await harness.seed(
        {"living_room": make_room({"schedules": [{"entity_id": "schedule.living"}]})},
        settings=HOUSE_ON,
    )
    harness.hass.states.set("schedule.living", "on", {})

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert (room["heat_target"], room["cool_target"]) == (22.0, 25.0)
    assert room["target_reason"] == TARGET_REASON_HOUSE


async def test_an_independent_room_ignores_the_house(harness: Harness):
    await harness.seed(
        {"living_room": make_room({"follow_house": False})},
        settings=HOUSE_ON,
    )

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert (room["heat_target"], room["cool_target"]) == (21.0, 24.0)
    assert room["target_reason"] == ""
    assert room["house_following"] is False
    assert data["house"]["following_rooms"] == 0


async def test_a_manual_override_beats_the_house(harness: Harness):
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

    room = data["rooms"]["living_room"]
    assert (room["heat_target"], room["cool_target"]) == (23.0, 26.0)
    assert room["target_reason"] == TARGET_REASON_OVERRIDE


async def test_vacation_beats_the_house(harness: Harness):
    await harness.seed(
        {"living_room": make_room()},
        settings={**HOUSE_ON, "vacation_until": time.time() + 86400, "vacation_temp": 15.0},
    )

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["target_reason"] == TARGET_REASON_VACATION
    assert room["heat_target"] == 15.0


async def test_away_beats_the_house(harness: Harness, fake_states):
    fake_states.set("person.alex", "not_home")
    await harness.seed(
        {"living_room": make_room({"presence_persons": ["person.alex"]})},
        settings={**HOUSE_ON, "presence_enabled": True, "presence_persons": ["person.alex"]},
    )

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["target_reason"] == TARGET_REASON_AWAY
    assert room["heat_target"] == 17.0


async def test_a_vacant_following_room_is_capped_at_the_house_eco_pair(harness: Harness, fake_states):
    fake_states.set("binary_sensor.motion", "off")
    await harness.seed(
        {
            "living_room": make_room(
                {
                    "occupancy_sensors": ["binary_sensor.motion"],
                    "occupancy_mode": "reactive_and_trend",
                    "occupancy_dwell_minutes": 0,
                }
            )
        },
        settings=HOUSE_ON,
    )

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["occupancy_vacant"] is True
    # House band 22/25 capped to the house eco pair 16/28.
    assert (room["heat_target"], room["cool_target"]) == (16.0, 28.0)
    assert room["target_reason"] == TARGET_REASON_VACANT
