"""Several control cycles in a row: commands, guards, detections and learned state."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from custom_components.roommind.const import DOMAIN, MODE_COOLING, MODE_HEATING, MODE_IDLE
from custom_components.roommind.coordinator import RoomMindCoordinator

from .conftest import Harness, make_room

pytestmark = pytest.mark.asyncio


async def test_heating_runs_its_minimum_run_then_settles(harness: Harness, fake_states):
    coordinator = await harness.seed({"living_room": make_room()})

    first = await harness.cycle()
    assert first["central_plant"]["action"] == MODE_HEATING

    # The room reaches its target, but the furnace is still inside its minimum run.
    fake_states.set("sensor.room_temp", "21.2", {"unit_of_measurement": "°C"})
    second = await harness.cycle()
    assert second["central_plant"]["action"] == MODE_HEATING
    assert second["central_plant"]["guard_reason"] == "min_run"

    # Once the run is old enough the plant may idle.
    coordinator._plant_manager.action_started_at = time.time() - 3600
    third = await harness.cycle()
    assert third["central_plant"]["action"] == MODE_IDLE


async def test_the_plant_follows_a_cooling_day(harness: Harness, fake_states):
    fake_states.set("sensor.outdoor_temp", "30.0", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.room_temp", "26.0", {"unit_of_measurement": "°C"})
    await harness.seed({"living_room": make_room()})

    data = await harness.cycle()

    assert data["central_plant"]["action"] == MODE_COOLING
    # 30 °C outside is above the heating gate: only cooling and circulation are left.
    assert "heating" not in data["central_plant"]["legal_actions"]
    assert "cooling" in data["central_plant"]["legal_actions"]


async def test_a_window_pauses_and_releases_the_plant(harness: Harness, fake_states):
    fake_states.set("binary_sensor.window", "on")
    await harness.seed(
        {"living_room": make_room({"exterior_window_sensors": ["binary_sensor.window"], "window_open_delay": 0})}
    )

    paused = await harness.cycle()
    assert paused["central_plant"]["paused"] is True
    assert paused["rooms"]["living_room"]["window_open"] is True
    assert paused["central_plant"]["action"] == MODE_IDLE

    fake_states.set("binary_sensor.window", "off")
    released = await harness.cycle()
    assert released["central_plant"]["paused"] is False
    assert released["rooms"]["living_room"]["window_open"] is False
    assert released["central_plant"]["action"] == MODE_HEATING


async def test_the_learned_state_survives_a_restart(harness: Harness, fake_states):
    coordinator = await harness.seed({"living_room": make_room()})
    await harness.cycle()
    store = coordinator.hass.data[DOMAIN]["store"]
    await store.async_save_thermal_data(coordinator._model_manager.to_dict())
    await store.async_save_settings(
        {
            "central_plant_state": coordinator._plant_manager.to_dict(),
            "wet_coil_state": {"charge": 0.7, "last_cooling_ts": time.time()},
            "ventilation_state": {"fan.vent": time.time()},
        }
    )

    restarted = RoomMindCoordinator(harness.hass, MagicMock())
    restarted._history_store = MagicMock(
        read_detail=MagicMock(return_value=[]), read_history=MagicMock(return_value=[])
    )
    harness.hass.data[DOMAIN] = {"store": store}
    restarted._load_learned_state(store, store.get_settings())

    assert restarted._model_manager.get_estimator("living_room")._x[0] == pytest.approx(18.0)
    assert restarted._plant_manager.active_action == MODE_HEATING
    assert restarted._wet_coil.charge == pytest.approx(0.7)
    assert restarted._ventilation.is_running("fan.vent") is True


async def test_a_second_cycle_reuses_the_same_estimators(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})

    await harness.cycle()
    estimator = coordinator._model_manager.get_estimator("living_room")
    data = await harness.cycle()

    assert coordinator._model_manager.get_estimator("living_room") is estimator
    assert data["rooms"]["living_room"]["n_observations"] >= 0


async def test_the_thermostat_keeps_its_fallback_band_while_idle(harness: Harness, fake_states):
    fake_states.set("sensor.room_temp", "21.1", {"unit_of_measurement": "°C"})
    await harness.seed({"living_room": make_room()})

    await harness.cycle()
    await harness.cycle()

    setpoints = [
        call.args[2] for call in harness.hass.services.async_call.call_args_list if call.args[1] == "set_temperature"
    ]
    assert setpoints
    assert 5.0 <= setpoints[-1]["target_temp_low"] <= setpoints[-1]["target_temp_high"] <= 35.0


async def test_a_room_override_reaches_the_wall_thermostat(harness: Harness):
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

    assert data["rooms"]["living_room"]["heat_target"] == 23.0
    nudge = [
        call.args[2] for call in harness.hass.services.async_call.call_args_list if call.args[1] == "set_temperature"
    ]
    # The nudge sits just past the thermostat's own reading, not at the remote room target.
    assert nudge and nudge[-1]["target_temp_low"] < 23.0


async def test_the_cycle_is_idempotent_without_changes(harness: Harness, fake_states):
    await harness.seed({"living_room": make_room()})

    first = await harness.cycle()
    second = await harness.cycle()

    assert first["rooms"]["living_room"]["heat_target"] == second["rooms"]["living_room"]["heat_target"]
    assert first["central_plant"]["action"] == second["central_plant"]["action"] == MODE_HEATING
