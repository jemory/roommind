"""The Central Plant control loop: one shared thermostat, many Thermal Zones."""

from __future__ import annotations

import pytest

from custom_components.roommind.const import DOMAIN, MODE_COOLING, MODE_HEATING, MODE_IDLE

from .conftest import DEFAULT_THERMOSTAT_ATTRIBUTES, THERMOSTAT_ENTITY, Harness, make_room

pytestmark = pytest.mark.asyncio


async def _climate_calls(harness: Harness) -> list[str]:
    """The climate service names the coordinator called this cycle."""
    return [
        call.args[1]
        for call in harness.hass.services.async_call.call_args_list
        if call.args and call.args[0] == "climate"
    ]


async def test_a_cycle_publishes_the_room_and_plant_state(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})

    data = await harness.cycle()

    assert set(data) == {"rooms", "central_plant"}
    assert set(data["rooms"]) == {"living_room"}
    assert coordinator.rooms == data["rooms"]
    assert coordinator.central_plant == data["central_plant"]
    room = data["rooms"]["living_room"]
    assert room["current_temp"] == 18.0
    assert room["heat_target"] == 21.0
    assert room["cool_target"] == 24.0
    assert room["target_temp"] == 21.0
    plant = data["central_plant"]
    assert plant["thermostat_entity"] == THERMOSTAT_ENTITY
    assert plant["house_ref_temp"] == pytest.approx(19.0)
    assert plant["climate_mode"] == "auto"
    # 5 °C outside is below the cooling gate, so cooling is not an option this cycle.
    assert set(plant["legal_actions"]) == {"idle", "heating", "circulating"}


async def test_heat_demand_nudges_the_central_thermostat(harness: Harness):
    await harness.seed({"living_room": make_room()})

    data = await harness.cycle()

    calls = await _climate_calls(harness)
    assert "set_hvac_mode" in calls
    assert "set_temperature" in calls
    plant = data["central_plant"]
    assert plant["action"] == MODE_HEATING
    assert plant["guard_active"] is False


async def test_a_room_at_target_leaves_the_plant_idle(harness: Harness, fake_states):
    fake_states.set("sensor.room_temp", "21.1", {"unit_of_measurement": "°C"})
    await harness.seed({"living_room": make_room()})

    data = await harness.cycle()

    assert data["central_plant"]["action"] == MODE_IDLE
    assert data["rooms"]["living_room"]["mode"] == MODE_IDLE
    # Idle parks the thermostat in the safe Fallback Band instead of switching it off.
    set_temperature = [
        call.args[2] for call in harness.hass.services.async_call.call_args_list if call.args[1] == "set_temperature"
    ]
    assert set_temperature
    assert set_temperature[-1]["target_temp_low"] == pytest.approx(16.5)
    assert set_temperature[-1]["target_temp_high"] == pytest.approx(26.5)


async def test_cooling_demand_runs_the_air_conditioner(harness: Harness, fake_states):
    fake_states.set("sensor.room_temp", "27.0", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.outdoor_temp", "31.0", {"unit_of_measurement": "°C"})
    await harness.seed({"living_room": make_room()})

    data = await harness.cycle()

    assert data["central_plant"]["action"] == MODE_COOLING
    assert data["rooms"]["living_room"]["cool_target"] == 24.0


async def test_without_a_central_thermostat_nothing_is_commanded(harness: Harness):
    await harness.seed(
        {"living_room": make_room()},
        settings={"central_plant": {"thermostat_entity": ""}},
    )

    data = await harness.cycle()

    assert data["central_plant"]["action"] == MODE_IDLE
    assert await _climate_calls(harness) == []


async def test_global_control_off_sends_nothing(harness: Harness):
    await harness.seed({"living_room": make_room()}, settings={"climate_control_active": False})

    data = await harness.cycle()

    assert data["central_plant"]["action"] == MODE_IDLE
    assert await _climate_calls(harness) == []


async def test_a_room_switched_off_is_forced_off(harness: Harness):
    await harness.seed({"living_room": make_room({"climate_control_enabled": False})})

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["force_off"] is True
    assert data["central_plant"]["action"] == MODE_IDLE
    assert data["central_plant"]["all_force_off"] is True


async def test_an_outdoor_area_is_reported_passively(harness: Harness):
    await harness.seed({"garden": make_room({"is_outdoor": True, "climate_control_enabled": False})})

    data = await harness.cycle()

    room = data["rooms"]["garden"]
    assert room["force_off"] is True
    assert room["heat_target"] is None
    assert room["cool_target"] is None
    assert room["mode"] == MODE_IDLE
    assert data["central_plant"]["action"] == MODE_IDLE


async def test_a_thermostat_without_hvac_action_shows_the_command(harness: Harness, fake_states):
    attributes = dict(DEFAULT_THERMOSTAT_ATTRIBUTES)
    attributes.pop("hvac_action")
    fake_states.set(THERMOSTAT_ENTITY, "heat", attributes)
    await harness.seed({"living_room": make_room()})

    data = await harness.cycle()

    assert data["central_plant"]["display_action"] == MODE_HEATING
    assert data["rooms"]["living_room"]["mode"] == MODE_HEATING


async def test_the_hvac_action_reported_by_the_thermostat_is_shown(harness: Harness, fake_states):
    attributes = dict(DEFAULT_THERMOSTAT_ATTRIBUTES)
    attributes["hvac_action"] = "heating"
    fake_states.set(THERMOSTAT_ENTITY, "heat", attributes)
    await harness.seed({"living_room": make_room()})

    data = await harness.cycle()

    assert data["central_plant"]["display_action"] == MODE_HEATING


async def test_house_reference_includes_the_thermostat_hallway(harness: Harness, fake_states):
    fake_states.set(THERMOSTAT_ENTITY, "heat", {**DEFAULT_THERMOSTAT_ATTRIBUTES, "current_temperature": 24.0})
    await harness.seed({"living_room": make_room(), "office": make_room({"temperature_sensor": "sensor.office_temp"})})
    fake_states.set("sensor.office_temp", "20.0", {"unit_of_measurement": "°C"})

    data = await harness.cycle()

    # (18 + 20 + 24) / 3
    assert data["central_plant"]["house_ref_temp"] == pytest.approx(20.6667, abs=1e-3)


async def test_house_reference_does_not_double_count_the_thermostat_area(harness: Harness, fake_states):
    fake_states.set(THERMOSTAT_ENTITY, "heat", {**DEFAULT_THERMOSTAT_ATTRIBUTES, "current_temperature": 24.0})
    await harness.seed(
        {"living_room": make_room()},
        settings={"central_plant": {"thermostat_entity": THERMOSTAT_ENTITY, "thermostat_area_id": "living_room"}},
    )

    data = await harness.cycle()

    assert data["central_plant"]["house_ref_temp"] == pytest.approx(18.0)


async def test_an_open_window_room_is_idle_and_left_out_of_the_house_mean(harness: Harness, fake_states):
    fake_states.set("binary_sensor.living_window", "on")
    await harness.seed(
        {
            "living_room": make_room(
                {"exterior_window_sensors": ["binary_sensor.living_window"], "window_open_delay": 0}
            ),
            "bedroom": make_room({"temperature_sensor": "sensor.bedroom_temp"}),
        }
    )
    fake_states.set("sensor.bedroom_temp", "20.0", {"unit_of_measurement": "°C"})

    data = await harness.cycle()

    assert data["rooms"]["living_room"]["window_open"] is True
    assert data["rooms"]["living_room"]["mode"] == MODE_IDLE
    # Only the bedroom and the hallway thermostat contribute.
    assert data["central_plant"]["house_ref_temp"] == pytest.approx(20.0)


async def test_a_humid_room_feels_warmer_than_its_sensor(harness: Harness, fake_states):
    fake_states.set("sensor.room_humidity", "95.0")
    await harness.seed({"living_room": make_room()})

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["effective_temp"] > room["current_temp"]


async def test_an_unavailable_thermostat_leaves_the_plant_idle(harness: Harness, fake_states):
    fake_states.set(THERMOSTAT_ENTITY, "unavailable", {})
    coordinator = await harness.seed({"living_room": make_room()})

    data = await harness.cycle()

    # The plan still exists, but an unavailable thermostat takes no command ...
    assert data["central_plant"]["action"] == MODE_HEATING
    assert await _climate_calls(harness) == []
    # ... and nothing is trained on a transient/unavailable equipment reading.
    assert coordinator._model_inputs == {}


async def test_an_open_window_holds_the_plant_of_the_other_rooms(harness: Harness, fake_states):
    fake_states.set("binary_sensor.living_window", "on")
    await harness.seed(
        {
            "living_room": make_room(
                {
                    "exterior_window_sensors": ["binary_sensor.living_window"],
                    "window_open_delay": 0,
                    "temperature_sensor": "sensor.living_temp",
                }
            ),
            "bedroom": make_room({"temperature_sensor": "sensor.bedroom_temp"}),
        },
        settings={
            "central_plant": {
                "thermostat_entity": THERMOSTAT_ENTITY,
                "exterior_open_policy": "pause_always",
            }
        },
    )
    fake_states.set("sensor.living_temp", "18.0", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.bedroom_temp", "18.0", {"unit_of_measurement": "°C"})

    data = await harness.cycle()

    assert data["central_plant"]["paused"] is True
    assert data["central_plant"]["blocking_rooms"] == ["living_room"]
    assert data["central_plant"]["action"] == MODE_IDLE
    assert data["central_plant"]["guard_reason"] == "exterior_open"
    # Paused means the fallback band, never a heating call for the other rooms.
    setpoints = [
        call.args[2] for call in harness.hass.services.async_call.call_args_list if call.args[1] == "set_temperature"
    ]
    assert setpoints
    assert all(setpoint.get("target_temp_low", 0) < 18 for setpoint in setpoints)


async def test_a_room_deleted_before_the_cycle_is_not_processed(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room(), "office": make_room()})
    store = coordinator.hass.data[DOMAIN]["store"]
    await store.async_delete_room("office")

    data = await harness.cycle()

    assert set(data["rooms"]) == {"living_room"}


async def test_a_room_without_sensor_still_counts_with_the_thermostat_reading(harness: Harness, fake_states):
    await harness.seed(
        {"living_room": make_room({"temperature_sensor": ""})},
        settings={"central_plant": {"thermostat_entity": THERMOSTAT_ENTITY, "thermostat_area_id": "living_room"}},
    )

    data = await harness.cycle()

    assert data["rooms"]["living_room"]["current_temp"] == 20.0


async def test_a_room_without_sensor_and_without_thermostat_area_has_no_temperature(harness: Harness):
    await harness.seed({"living_room": make_room({"temperature_sensor": ""})})

    data = await harness.cycle()

    assert data["rooms"]["living_room"]["current_temp"] is None
    assert data["central_plant"]["action"] == MODE_IDLE
