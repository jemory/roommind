"""Learning, history and persistence around the control loop."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.roommind.const import (
    HISTORY_ROTATE_CYCLES,
    HISTORY_WRITE_CYCLES,
    MODE_COOLING,
    MODE_HEATING,
    MODE_IDLE,
    THERMAL_SAVE_CYCLES,
)
from custom_components.roommind.control.solar import compute_q_solar_norm, plane_of_array_irradiance
from custom_components.roommind.control.thermal_model import RoomModelManager, ThermalEKF, regressor_signature
from custom_components.roommind.coordinator import _RoomCycle
from custom_components.roommind.utils.history_store import MODEL_INPUT_FIELDS

from .conftest import DEFAULT_THERMOSTAT_ATTRIBUTES, THERMOSTAT_ENTITY, Harness, make_room

pytestmark = pytest.mark.asyncio

# Solar noon (UTC) at the harness coordinates, so the facade and the horizontal
# plane receive measurably different irradiance.
_FIXED_TS = datetime(2023, 6, 21, 11, 8, tzinfo=UTC).timestamp()


async def test_a_heating_cycle_charges_the_forced_air_rebound(harness: Harness, fake_states):
    fake_states.set(THERMOSTAT_ENTITY, "heat", {**DEFAULT_THERMOSTAT_ATTRIBUTES, "hvac_action": "heating"})
    coordinator = await harness.seed({"living_room": make_room()})

    await harness.cycle()

    assert coordinator._rebound_tracker.charge("living_room") > 0.0


async def test_a_cooling_cycle_charges_the_forced_air_rebound_negatively(harness: Harness, fake_states):
    fake_states.set("sensor.room_temp", "27.0", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.outdoor_temp", "31.0", {"unit_of_measurement": "°C"})
    fake_states.set(THERMOSTAT_ENTITY, "cool", {**DEFAULT_THERMOSTAT_ATTRIBUTES, "hvac_action": "cooling"})
    coordinator = await harness.seed({"living_room": make_room()})

    await harness.cycle()

    assert coordinator._rebound_tracker.charge("living_room") < 0.0


async def test_the_model_inputs_are_recorded_for_the_backtest(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})

    data = await harness.cycle()

    inputs = coordinator._model_inputs["living_room"]
    assert set(inputs) == set(MODEL_INPUT_FIELDS)
    assert inputs["model_fan"] == 0.0
    assert inputs["model_door_mix"] == 1.0
    room = data["rooms"]["living_room"]
    assert room["model_power"] == inputs["model_power"]


async def test_a_missing_house_reference_is_recorded_as_unknown(harness: Harness):
    """A missing house reference must stay empty for the backtest, never become a 0 °C sink (#audit)."""
    coordinator = await harness.seed({"living_room": make_room()})
    cycle = _RoomCycle(area_id="living_room", room=make_room())

    inputs = coordinator._model_input_row(cycle, MODE_IDLE, None)

    assert set(inputs) == set(MODEL_INPUT_FIELDS)
    assert inputs["model_house_temp"] is None
    assert inputs["model_house_temp"] != 0.0
    # A real reference is still rounded and passed through.
    assert coordinator._model_input_row(cycle, MODE_IDLE, 20.4567)["model_house_temp"] == 20.457


async def test_an_idle_cycle_trains_a_zero_power_label(harness: Harness, fake_states):
    fake_states.set("sensor.room_temp", "21.1", {"unit_of_measurement": "°C"})
    coordinator = await harness.seed({"living_room": make_room()})

    await harness.cycle()

    assert coordinator._model_inputs["living_room"]["model_power"] == 0.0


async def test_a_learning_disabled_room_is_not_trained(harness: Harness):
    coordinator = await harness.seed(
        {"living_room": make_room()},
        settings={"learning_disabled_rooms": ["living_room"]},
    )

    await harness.cycle()

    assert coordinator._model_inputs == {}
    assert coordinator._model_manager.get_n_observations("living_room") == 0


async def test_no_outdoor_temperature_pauses_learning(harness: Harness, fake_states):
    fake_states.remove("sensor.outdoor_temp")
    coordinator = await harness.seed({"living_room": make_room()})

    await harness.cycle()

    assert coordinator.outdoor_temp_effective is None
    assert coordinator._model_inputs == {}
    assert coordinator._model_manager.get_n_observations("living_room") == 0


async def test_the_outdoor_watchdog_notifies_once_and_clears(harness: Harness, fake_states):
    fake_states.remove("sensor.outdoor_temp")
    coordinator = await harness.seed({"living_room": make_room()})
    coordinator._outdoor_unavailable_cycles = 59

    with (
        patch("custom_components.roommind.coordinator.async_create_notification") as create,
        patch("custom_components.roommind.coordinator.async_dismiss_notification") as dismiss,
    ):
        await harness.cycle()
        create.assert_called_once()
        dismiss.assert_not_called()

        fake_states.set("sensor.outdoor_temp", "5.0", {"unit_of_measurement": "°C"})
        await harness.cycle()
        dismiss.assert_called_once()


async def test_the_outdoor_watchdog_can_be_switched_off(harness: Harness, fake_states):
    fake_states.remove("sensor.outdoor_temp")
    coordinator = await harness.seed(
        {"living_room": make_room()},
        settings={"outdoor_unavailable_notify": False},
    )
    coordinator._outdoor_unavailable_cycles = 59

    with patch("custom_components.roommind.coordinator.async_create_notification") as create:
        await harness.cycle()

    create.assert_not_called()


async def test_the_thermal_model_is_saved_periodically(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    coordinator._thermal_save_count = THERMAL_SAVE_CYCLES - 1
    store = coordinator.hass.data["roommind"]["store"]
    store.async_save_thermal_data = AsyncMock(return_value=None)
    store.async_save_occupancy_trends = AsyncMock(return_value=None)

    with patch.object(type(coordinator._model_manager.get_estimator("living_room")), "note_confidence_day") as note:
        await harness.cycle()

    store.async_save_thermal_data.assert_awaited_once()
    store.async_save_occupancy_trends.assert_awaited_once()
    note.assert_called_once()


async def test_the_history_row_is_written_every_six_cycles(harness: Harness, fake_states):
    fake_states.set(THERMOSTAT_ENTITY, "heat", {**DEFAULT_THERMOSTAT_ATTRIBUTES, "hvac_action": "heating"})
    coordinator = await harness.seed({"living_room": make_room()})
    coordinator._history_write_count = HISTORY_WRITE_CYCLES - 1
    history = coordinator._history_store
    assert history is not None

    await harness.cycle()

    history.record.assert_called_once()
    area_id, row, _timestamp = history.record.call_args.args
    assert area_id == "living_room"
    assert row["mode"] == MODE_HEATING
    assert row["heat_target"] == 21.0
    assert row["model_power"] == 1.0
    assert row["fan_circulating"] is False
    assert row["predicted_temp"] is not None


async def test_a_learning_disabled_room_writes_no_history(harness: Harness):
    coordinator = await harness.seed(
        {"living_room": make_room()},
        settings={"learning_disabled_rooms": ["living_room"]},
    )
    coordinator._history_write_count = HISTORY_WRITE_CYCLES - 1
    history = coordinator._history_store
    assert history is not None

    await harness.cycle()

    history.record.assert_not_called()


async def test_the_history_is_rotated_periodically(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    coordinator._history_rotate_count = HISTORY_ROTATE_CYCLES - 1
    history = coordinator._history_store
    assert history is not None

    await harness.cycle()

    history.rotate.assert_called_once_with("living_room")


async def test_a_history_write_failure_does_not_break_the_cycle(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    coordinator._history_write_count = HISTORY_WRITE_CYCLES - 1
    history = coordinator._history_store
    assert history is not None
    history.record.side_effect = OSError("disk full")

    data = await harness.cycle()

    assert "living_room" in data["rooms"]


async def test_the_learned_state_is_restored_on_the_first_cycle(harness: Harness, fake_states):
    model_manager = RoomModelManager()
    model_manager.get_estimator("living_room").sync_temperature(19.0)
    coordinator = await harness.seed(
        {"living_room": make_room()},
        thermal_data=model_manager.to_dict(),
        occupancy_trends={
            "living_room": {"trend": {"weekday_bins": [1.0] * 96, "weekend_bins": [1.0] * 96, "days_observed": 7.0}}
        },
        settings={
            "central_plant": {"thermostat_entity": THERMOSTAT_ENTITY},
            "central_plant_state": {"active_action": MODE_COOLING, "action_started_at": time.time()},
            "wet_coil_state": {"charge": 0.6, "last_cooling_ts": time.time()},
            "ventilation_state": {"fan.vent": time.time()},
        },
    )
    store = coordinator.hass.data["roommind"]["store"]
    fake_states.set("sensor.outdoor_temp", "31.0", {"unit_of_measurement": "°C"})

    coordinator._load_learned_state(store, store.get_settings())

    assert coordinator._model_loaded is True
    assert coordinator._model_manager.get_estimator("living_room")._x[0] == pytest.approx(19.0)
    assert coordinator._occupancy_manager.weighted_trend_probability("living_room", time.time()) > 0.0
    assert coordinator._plant_manager.active_action == MODE_COOLING
    assert coordinator._wet_coil.charge == pytest.approx(0.6)
    assert coordinator._ventilation.is_running("fan.vent") is True

    # The second call is a no-op: the state is only restored once.
    coordinator._load_learned_state(store, {"central_plant_state": {"active_action": MODE_IDLE}})
    assert coordinator._plant_manager.active_action == MODE_COOLING


async def test_a_circulating_cycle_records_the_fan_input(harness: Harness, fake_states):
    fake_states.set(
        THERMOSTAT_ENTITY,
        "fan_only",
        {
            "current_temperature": 20.0,
            "hvac_modes": ["off", "heat", "cool", "heat_cool", "fan_only"],
            "supported_features": 2,
            "fan_modes": ["auto", "on"],
            "fan_mode": "on",
            "hvac_action": "fan",
        },
    )
    coordinator = await harness.seed({"living_room": make_room()})

    await harness.cycle()

    inputs = coordinator._model_inputs["living_room"]
    assert inputs["model_fan"] == 1.0
    assert coordinator._rebound_tracker.charge("living_room") == 0.0


async def test_the_price_history_is_read_back_from_the_records(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    history = coordinator._history_store
    assert history is not None
    history.read_detail.return_value = [
        {"timestamp": "100.0", "price": "0.2"},
        {"timestamp": "200.0", "price": "junk"},
        {"timestamp": "300.0"},
    ]
    history.read_history.return_value = [{"timestamp": "150.0", "price": "0.3"}]

    prices = await coordinator._async_load_price_history()

    assert prices == [(100.0, 0.2), (150.0, 0.3)]


async def test_the_price_history_is_empty_without_records(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    history = coordinator._history_store
    assert history is not None
    history.read_detail.return_value = []
    history.read_history.return_value = []

    assert await coordinator._async_load_price_history() == []

    coordinator._history_store = None
    assert await coordinator._async_load_price_history() == []


async def test_releasing_ventilation_switches_off_what_roommind_started(harness: Harness, fake_states):
    fake_states.set("fan.vent", "on")
    coordinator = await harness.seed(
        {"living_room": make_room({"co2_sensor": "sensor.co2", "ventilation_entity": "fan.vent"})},
        settings={"ventilation_state": {"fan.vent": time.time()}},
    )
    coordinator._ventilation.load_state({"fan.vent": time.time()})
    coordinator._ventilation._last_state["fan.vent"] = "on"

    await coordinator.async_release_ventilation()

    assert coordinator._ventilation.is_running("fan.vent") is False
    off_calls = [
        call
        for call in harness.hass.services.async_call.call_args_list
        if call.args[0] == "fan" and call.args[1] == "turn_off"
    ]
    assert off_calls
    assert coordinator.hass.data["roommind"]["store"].get_settings()["ventilation_state"] == {}


async def test_releasing_ventilation_tolerates_a_failing_service(harness: Harness, fake_states):
    fake_states.set("fan.vent", "on")
    coordinator = await harness.seed({"living_room": make_room()})
    coordinator._ventilation.load_state({"fan.vent": time.time()})
    coordinator._ventilation._last_state["fan.vent"] = "on"
    harness.hass.services.async_call.side_effect = RuntimeError("service down")

    await coordinator.async_release_ventilation()

    # The unit stays owned because the off call failed, but the cycle does not raise.
    assert coordinator._ventilation.is_running("fan.vent") is True


# ── Per-room solar and regressor-signature wiring ───────────────────────


async def test_unconfigured_room_keeps_the_house_solar_value(harness: Harness):
    """A room with no solar_azimuths keeps today's exact GHI value."""
    coordinator = await harness.seed({"plain": make_room()})
    lat, lon = harness.hass.config.latitude, harness.hass.config.longitude
    ghi = compute_q_solar_norm(lat, lon, _FIXED_TS, None)
    cycles = [
        _RoomCycle(area_id="plain", room=make_room()),
        _RoomCycle(area_id="oriented", room=make_room({"solar_azimuths": [180]})),
    ]
    with patch("custom_components.roommind.coordinator.time.time", return_value=_FIXED_TS):
        coordinator._current_q_solar = ghi
        coordinator._refresh_room_q_solar(cycles, None)

    assert coordinator._current_q_solar_by_room["plain"] == ghi
    assert coordinator._room_q_solar(cycles[0]) == ghi
    # An unknown area id also falls back to the house-wide value.
    assert coordinator._room_q_solar(_RoomCycle(area_id="missing", room={})) == ghi
    expected_poa = plane_of_array_irradiance(lat, lon, _FIXED_TS, 180, cloud_coverage=None) / 1000.0
    assert coordinator._current_q_solar_by_room["oriented"] == pytest.approx(expected_poa)
    assert expected_poa != ghi  # the oriented value is genuinely different


async def test_model_input_row_records_oriented_solar_times_shading(harness: Harness):
    """The backtest replays the exact per-room regressor the EKF trained on."""
    coordinator = await harness.seed({"living_room": make_room({"solar_azimuths": [180]})})
    coordinator._current_q_solar = 0.8
    coordinator._current_q_solar_by_room = {"living_room": 0.5}
    cycle = _RoomCycle(area_id="living_room", room=make_room({"solar_azimuths": [180]}), shading_factor=0.6)

    row = coordinator._model_input_row(cycle, MODE_HEATING, 20.0)

    assert row["model_solar"] == round(0.5 * 0.6, 3)
    assert row["model_solar"] != round(0.8 * 0.6, 3)


async def test_history_solar_irradiance_remains_horizontal_ghi(harness: Harness):
    """The user-facing history field keeps the horizontal GHI, not the model input."""
    coordinator = await harness.seed({"living_room": make_room({"solar_azimuths": [180]})})
    await harness.cycle()
    coordinator._current_q_solar = 0.8
    coordinator._current_q_solar_by_room = {"living_room": 0.5}
    room = make_room({"solar_azimuths": [180]})
    cycle = _RoomCycle(area_id="living_room", room=room, shading_factor=0.6)
    state = harness.coordinator.rooms["living_room"]  # type: ignore[union-attr]
    coordinator._history_store.record.reset_mock()
    await coordinator._async_record_history_row("living_room", cycle, state, None, time.time())
    row = coordinator._history_store.record.call_args.args[1]

    assert row["solar_irradiance"] == 0.8
    assert coordinator._model_input_row(cycle, MODE_IDLE, 20.0)["model_solar"] == 0.3


async def test_regressor_signature_is_applied_even_when_room_is_not_trainable(harness: Harness):
    """A saved configuration change is picked up on the first cycle, before the training gate."""
    coordinator = await harness.seed(
        {
            "oriented": make_room({"solar_azimuths": [180]}),
            "connected": make_room({"neighbour_rooms": ["other"]}),
            "other": make_room(),
        },
        settings={"learning_disabled_rooms": ["oriented"]},
    )
    with patch.object(ThermalEKF, "apply_regressor_signature", autospec=True) as apply:
        await harness.cycle()

    oriented = coordinator._model_manager.get_estimator("oriented")
    connected = coordinator._model_manager.get_estimator("connected")
    by_estimator = {id(call.args[0]): call.args[1] for call in apply.call_args_list}
    assert by_estimator[id(oriented)] == regressor_signature([180], False)
    assert by_estimator[id(connected)] == regressor_signature((), True)
