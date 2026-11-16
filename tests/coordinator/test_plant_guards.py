"""Equipment guards, the wet coil, the exterior gate and the wall-thermostat override."""

from __future__ import annotations

import time
from typing import Any

import pytest

from custom_components.roommind.const import MODE_COOLING, MODE_HEATING, MODE_IDLE

from .conftest import DEFAULT_THERMOSTAT_ATTRIBUTES, THERMOSTAT_ENTITY, Harness, make_room

pytestmark = pytest.mark.asyncio


def plant(coordinator: Any) -> Any:
    return coordinator._plant_manager


def skip_learned_load(coordinator: Any) -> None:
    """The first cycle restores saved sub-manager state; tests pre-set it instead."""
    coordinator._model_loaded = True


async def _saved_settings(coordinator: Any) -> list[dict]:
    store = coordinator.hass.data["roommind"]["store"]
    saved: list[dict] = []

    async def record(changes: dict) -> dict:
        saved.append(dict(changes))
        return store.get_settings().update(changes) or store.get_settings()

    store.async_save_settings = record
    return saved


async def test_a_running_furnace_finishes_its_minimum_run(harness: Harness, fake_states):
    coordinator = await harness.seed({"living_room": make_room({"comfort_heat": 19.0})})
    skip_learned_load(coordinator)
    fake_states.set("sensor.room_temp", "21.0", {"unit_of_measurement": "°C"})
    coordinator._plant_manager.active_action = MODE_HEATING
    coordinator._plant_manager.action_started_at = time.time()

    data = await harness.cycle()

    assert data["central_plant"]["action"] == MODE_HEATING
    assert data["central_plant"]["guard_reason"] == "min_run"
    assert data["central_plant"]["forced_action"] == MODE_HEATING


async def test_the_minimum_off_time_blocks_a_new_run(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    skip_learned_load(coordinator)
    coordinator._plant_manager.active_action = MODE_IDLE
    coordinator._plant_manager.last_heat_ended_at = time.time()

    data = await harness.cycle()

    assert data["central_plant"]["action"] == MODE_IDLE
    assert data["central_plant"]["guard_reason"] == "min_off"
    assert "heating" not in data["central_plant"]["legal_actions"]


async def test_changeover_keeps_heating_off_after_a_cooling_run(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    skip_learned_load(coordinator)
    coordinator._plant_manager.active_action = MODE_IDLE
    coordinator._plant_manager.last_cool_ended_at = time.time()

    data = await harness.cycle()

    assert "heating" not in data["central_plant"]["legal_actions"]
    assert data["central_plant"]["action"] == MODE_IDLE


async def test_a_wet_coil_locks_circulation_out(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    skip_learned_load(coordinator)
    coordinator._wet_coil.charge = 1.0
    coordinator._wet_coil.last_cooling_ts = time.time()

    data = await harness.cycle()

    assert data["central_plant"]["wet_coil_locked"] is True
    assert "circulating" not in data["central_plant"]["legal_actions"]
    assert coordinator._wet_coil.charge > 0.0


async def test_the_wet_coil_drains_over_time(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    skip_learned_load(coordinator)
    coordinator._wet_coil.charge = 1.0
    coordinator._wet_coil.last_cooling_ts = time.time()

    await harness.cycle()
    drained = coordinator._wet_coil.charge

    assert 0.0 <= drained < 1.0


async def test_the_wet_coil_state_is_persisted(harness: Harness, fake_states):
    coordinator = await harness.seed({"living_room": make_room()})
    skip_learned_load(coordinator)
    coordinator._wet_coil.charge = 0.8
    coordinator._wet_coil.last_cooling_ts = time.time()
    coordinator._wet_coil.state_dirty = True

    await harness.cycle()

    settings = coordinator.hass.data["roommind"]["store"].get_settings()
    assert "wet_coil_state" in settings
    assert settings["wet_coil_state"]["last_cooling_ts"] is not None


async def test_the_plant_state_is_persisted(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    await harness.cycle()

    settings = coordinator.hass.data["roommind"]["store"].get_settings()
    assert settings["central_plant_state"]["active_action"] == MODE_HEATING
    assert settings["central_plant_state"]["action_started_at"] is not None


async def test_an_exterior_opening_pauses_the_plant(harness: Harness, fake_states):
    fake_states.set("binary_sensor.window", "on")
    await harness.seed(
        {"living_room": make_room({"exterior_window_sensors": ["binary_sensor.window"], "window_open_delay": 0})},
        settings={
            "central_plant": {
                "thermostat_entity": THERMOSTAT_ENTITY,
                "exterior_open_policy": "pause_always",
            }
        },
    )

    data = await harness.cycle()

    assert data["central_plant"]["paused"] is True
    assert data["central_plant"]["guard_reason"] == "exterior_open"
    assert data["central_plant"]["action"] == MODE_IDLE


async def test_a_closed_interior_door_keeps_the_pause_policy_local(harness: Harness, fake_states):
    fake_states.set("binary_sensor.window", "on")
    fake_states.set("binary_sensor.door", "off")
    await harness.seed(
        {
            "living_room": make_room(
                {
                    "exterior_window_sensors": ["binary_sensor.window"],
                    "interior_door_sensors": ["binary_sensor.door"],
                    "interior_door_delay": 0,
                    "window_open_delay": 0,
                }
            )
        }
    )

    data = await harness.cycle()

    # The default policy leaves a room with a confirmed-closed door out of the pause.
    assert data["central_plant"]["paused"] is False
    assert data["central_plant"]["blocking_rooms"] == []


async def test_an_override_lifts_the_exterior_gate(harness: Harness, fake_states):
    fake_states.set("binary_sensor.window", "on")
    await harness.seed(
        {
            "living_room": make_room(
                {
                    "exterior_window_sensors": ["binary_sensor.window"],
                    "window_open_delay": 0,
                    "override_heat": 23.0,
                    "override_cool": 25.0,
                    "override_until": time.time() + 3600,
                    "override_type": "custom",
                }
            )
        },
        settings={"central_plant": {"thermostat_entity": THERMOSTAT_ENTITY, "exterior_open_policy": "pause_always"}},
    )

    data = await harness.cycle()

    assert data["central_plant"]["any_override"] is True
    assert data["central_plant"]["paused"] is False


async def test_exclude_room_only_never_pauses_the_house(harness: Harness, fake_states):
    fake_states.set("binary_sensor.window", "on")
    await harness.seed(
        {"living_room": make_room({"exterior_window_sensors": ["binary_sensor.window"], "window_open_delay": 0})},
        settings={
            "central_plant": {
                "thermostat_entity": THERMOSTAT_ENTITY,
                "exterior_open_policy": "exclude_room_only",
            }
        },
    )

    data = await harness.cycle()

    assert data["central_plant"]["paused"] is False


async def test_a_manual_wall_change_becomes_a_plant_override(harness: Harness, fake_states):
    coordinator = await harness.seed({"living_room": make_room()})
    attributes = dict(DEFAULT_THERMOSTAT_ATTRIBUTES)
    attributes["target_temp_low"] = 18.0
    attributes["target_temp_high"] = 24.0
    fake_states.set(THERMOSTAT_ENTITY, "heat", attributes)
    await harness.cycle()

    # Someone walked up to the wall thermostat and turned it up.
    fake_states.set(THERMOSTAT_ENTITY, "heat", {**attributes, "target_temp_low": 22.0, "target_temp_high": 26.0})
    data = await harness.cycle()

    assert data["central_plant"]["any_override"] is True
    settings = coordinator.hass.data["roommind"]["store"].get_settings()
    assert settings["plant_override_heat"] == pytest.approx(22.0)
    assert settings["plant_override_cool"] == pytest.approx(26.0)
    assert data["rooms"]["living_room"]["heat_target"] == pytest.approx(22.0)


async def test_the_wall_change_echo_is_not_an_override(harness: Harness, fake_states):
    coordinator = await harness.seed({"living_room": make_room()})
    await harness.cycle()
    # The actuator's own setpoint shows up in the state one cycle later.
    state = fake_states.get(THERMOSTAT_ENTITY)
    assert state is not None
    low = state.attributes.get("target_temp_low")
    high = state.attributes.get("target_temp_high")
    fake_states.set(THERMOSTAT_ENTITY, "heat", {**state.attributes, "target_temp_low": low, "target_temp_high": high})

    await harness.cycle()

    settings = coordinator.hass.data["roommind"]["store"].get_settings()
    assert "plant_override_heat" not in settings


async def test_an_expired_plant_override_is_cleared(harness: Harness, fake_states):
    coordinator = await harness.seed(
        {"living_room": make_room()},
        settings={
            "central_plant": {"thermostat_entity": THERMOSTAT_ENTITY},
            "plant_override_heat": 23.0,
            "plant_override_cool": 25.0,
            "plant_override_until": time.time() - 60,
        },
    )

    data = await harness.cycle()

    settings = coordinator.hass.data["roommind"]["store"].get_settings()
    assert settings["plant_override_heat"] is None
    assert settings["plant_override_until"] is None
    assert data["central_plant"]["any_override"] is False


async def test_the_startup_waiting_state_is_reported(harness: Harness):
    await harness.seed({"living_room": make_room({"temperature_sensor": "sensor.missing"})})

    data = await harness.cycle()

    assert data["central_plant"]["startup_waiting"] is True


async def test_the_guard_state_reports_why_it_holds(harness: Harness, fake_states):
    coordinator = await harness.seed({"living_room": make_room({"comfort_heat": 19.0})})
    skip_learned_load(coordinator)
    fake_states.set("sensor.room_temp", "21.0", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.outdoor_temp", "31.0", {"unit_of_measurement": "°C"})
    coordinator._plant_manager.active_action = MODE_COOLING
    coordinator._plant_manager.action_started_at = time.time()

    data = await harness.cycle()

    assert data["central_plant"]["guard_active"] is True
    assert data["central_plant"]["guard_until"] is not None
    assert data["central_plant"]["guard_until"] > time.time()


async def test_the_circulating_min_fan_guard_holds_the_blower(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    skip_learned_load(coordinator)
    coordinator._plant_manager.active_action = "circulating"
    coordinator._plant_manager.action_started_at = time.time()
    coordinator._plant_manager.last_heat_ended_at = time.time()

    data = await harness.cycle()

    assert data["central_plant"]["guard_reason"] == "min_fan"
    assert data["central_plant"]["action"] in ("circulating", "cooling", "heating")
