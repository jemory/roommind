"""Tests for the diagnostics module."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from custom_components.roommind.const import DOMAIN
from custom_components.roommind.diagnostics import (
    _build_central_plant_snapshot,
    _build_model_info,
    async_get_config_entry_diagnostics,
)


def _make_estimator() -> MagicMock:
    """Build a mock 9-state ThermalEKF estimator.

    ``x = [T, alpha_out, alpha_house, beta_h, beta_c, beta_fan, beta_s, beta_o, beta_f]``
    """
    est = MagicMock()
    est._x = [20.0, 0.3, 0.2, 4.0, 3.5, 0.8, 0.5, 0.25, 0.35]
    est._n_updates = 200
    est._n_idle = 120
    est._n_heating = 60
    est._n_cooling = 20
    est._n_circulating = 15
    est._applicable_modes = {"idle", "heating", "circulating"}
    est._P = [[0.01 * (i == j) for j in range(9)] for i in range(9)]
    est.prediction_std.return_value = 0.25
    est.confidence = 0.85
    rc = MagicMock()
    rc.Q_heat = 100.0
    rc.to_dict.return_value = {"alpha": 0.3, "beta_h": 100.0}
    est.get_model.return_value = rc
    return est


def _make_coordinator(
    rooms: dict | None = None,
    outdoor_temp: float | None = 5.0,
    outdoor_humidity: float | None = 60,
    forecast: list | None = None,
    history_store: object | None = None,
    estimators: dict | None = None,
    window_paused: dict | None = None,
    window_open_since: dict | None = None,
    window_closed_since: dict | None = None,
    cover_states: dict | None = None,
    heat_source_states: dict | None = None,
    central_plant: object | None = None,
    wet_coil: object | None = None,
    plant_manager: object | None = None,
    rebound_charge: float | None = None,
) -> MagicMock:
    """Build a mock coordinator with the managers diagnostics read."""
    coordinator = MagicMock()
    coordinator.rooms = rooms if rooms is not None else {}
    coordinator.outdoor_temp = outdoor_temp
    coordinator.outdoor_humidity = outdoor_humidity
    coordinator._weather_manager._outdoor_forecast = forecast or []
    coordinator._history_store = history_store
    coordinator._model_manager._estimators = estimators or {}
    coordinator._schedule_blocks_cache = {}

    # Window manager
    coordinator._window_manager._paused = window_paused or {}
    coordinator._window_manager._open_since = window_open_since or {}
    coordinator._window_manager._closed_since = window_closed_since or {}

    # Cover manager
    coordinator._cover_orchestrator._cover_manager._states = cover_states or {}

    # Legacy heat source routing, still reported when the coordinator has it
    coordinator._heat_source_states = heat_source_states or {}

    # Central Plant: the live summary, the wet-coil charge and the equipment guards
    coordinator.central_plant = central_plant
    coordinator._wet_coil = wet_coil
    coordinator._plant_manager = plant_manager
    if rebound_charge is None:
        coordinator._rebound_tracker = None
    else:
        coordinator._rebound_tracker.charge.return_value = rebound_charge

    return coordinator


def _thermostat_state(**attrs: object) -> MagicMock:
    state = MagicMock()
    state.state = "heat"
    state.attributes = attrs
    return state


def test_build_model_info():
    """_build_model_info reads the 9-state Central Forced-Air EKF."""
    info = _build_model_info(_make_estimator())

    assert info["alpha_out"] == 0.3
    assert info["alpha"] == 0.3  # alias kept for consumers of the older export
    assert info["alpha_house"] == 0.2
    assert info["beta_h"] == 4.0
    assert info["beta_c"] == 3.5
    assert info["beta_fan"] == 0.8
    assert info["beta_s"] == 0.5
    assert info["beta_o"] == 0.25
    assert info["beta_f"] == 0.35
    assert info["n_updates"] == 200
    assert info["n_idle"] == 120
    assert info["n_heating"] == 60
    assert info["n_cooling"] == 20
    assert info["n_circulating"] == 15
    assert info["applicable_modes"] == ["circulating", "heating", "idle"]  # sorted
    assert len(info["P_diagonal"]) == 9
    assert info["confidence"] == 0.85
    assert "model_params" in info


def test_build_model_info_tolerates_an_estimator_without_circulation_count():
    est = _make_estimator()
    del est._n_circulating
    assert _build_model_info(est)["n_circulating"] == 0


# ---------------------------------------------------------------------------
# Central Plant snapshot
# ---------------------------------------------------------------------------


def test_central_plant_snapshot_has_the_resolved_settings(hass):
    hass.states.get = MagicMock(return_value=None)
    settings = {"central_plant": {"thermostat_entity": "climate.hall", "min_run_minutes": 7}}
    snapshot = _build_central_plant_snapshot(hass, settings, None)

    assert snapshot["thermostat_entity"] == "climate.hall"
    # Unspecified keys are the plant defaults, not missing
    assert snapshot["settings"]["min_run_minutes"] == 7
    assert snapshot["settings"]["min_off_minutes"] > 0
    assert snapshot["settings"]["thermostat_entity"] == "climate.hall"
    assert "live" not in snapshot
    assert "wet_coil" not in snapshot
    assert "plant_manager" not in snapshot


def test_central_plant_snapshot_without_a_thermostat(hass):
    hass.states.get = MagicMock(return_value=None)
    snapshot = _build_central_plant_snapshot(hass, {}, None)
    assert snapshot["thermostat_entity"] == ""
    assert "thermostat" not in snapshot
    hass.states.get.assert_not_called()


def test_central_plant_snapshot_exporters_the_thermostat_state_and_attributes(hass):
    hass.states.get = MagicMock(
        return_value=_thermostat_state(
            hvac_mode="heat",
            hvac_modes=["off", "heat", "cool", "fan_only"],
            current_temperature=20.5,
            temperature=21.0,
            fan_mode="low",
            target_temp_low=20.0,
            target_temp_high=24.0,
            assumed_state=False,
        )
    )
    snapshot = _build_central_plant_snapshot(hass, {"central_plant": {"thermostat_entity": "climate.hall"}}, None)

    thermostat = snapshot["thermostat"]
    assert thermostat["entity_id"] == "climate.hall"
    assert thermostat["state"] == "heat"
    assert thermostat["hvac_modes"] == ["off", "heat", "cool", "fan_only"]
    assert thermostat["current_temperature"] == 20.5
    assert thermostat["temperature"] == 21.0
    assert thermostat["target_temp_low"] == 20.0
    assert thermostat["target_temp_high"] == 24.0
    assert thermostat["fan_mode"] == "low"
    # Attributes the thermostat does not publish are left out instead of exported as null
    assert "assumed_state" not in thermostat


def test_central_plant_snapshot_marks_a_missing_thermostat(hass):
    hass.states.get = MagicMock(return_value=None)
    snapshot = _build_central_plant_snapshot(hass, {"central_plant": {"thermostat_entity": "climate.gone"}}, None)
    assert snapshot["thermostat"] == {"entity_id": "climate.gone", "state": "not_found"}


def test_central_plant_snapshot_includes_the_live_plant_state(hass):
    hass.states.get = MagicMock(return_value=None)
    wet_coil = MagicMock()
    wet_coil.to_dict.return_value = {"charge": 0.4, "last_cooling_ts": 1000.0}
    plant_manager = MagicMock()
    plant_manager.to_dict.return_value = {"active_action": "cooling", "action_started_at": 900.0}
    coordinator = _make_coordinator(
        central_plant={"action": "cooling", "driving_rooms": ["living_room"], "guard_active": False},
        wet_coil=wet_coil,
        plant_manager=plant_manager,
    )

    snapshot = _build_central_plant_snapshot(hass, {}, coordinator)

    assert snapshot["live"]["action"] == "cooling"
    assert snapshot["live"]["driving_rooms"] == ["living_room"]
    assert snapshot["wet_coil"] == {"charge": 0.4, "last_cooling_ts": 1000.0}
    assert snapshot["plant_manager"] == {"active_action": "cooling", "action_started_at": 900.0}


def test_central_plant_snapshot_accepts_a_manager_as_the_live_summary(hass):
    hass.states.get = MagicMock(return_value=None)
    live = MagicMock()
    live.to_dict.return_value = {"action": "idle"}
    coordinator = _make_coordinator(central_plant=live)

    assert _build_central_plant_snapshot(hass, {}, coordinator)["live"] == {"action": "idle"}


def test_central_plant_snapshot_skips_managers_without_a_state(hass):
    hass.states.get = MagicMock(return_value=None)
    coordinator = _make_coordinator(central_plant=object())
    snapshot = _build_central_plant_snapshot(hass, {}, coordinator)
    assert "live" not in snapshot


# ---------------------------------------------------------------------------
# config entry diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diagnostics_no_store(hass, mock_config_entry):
    """Returns error when store is not loaded."""
    hass.data = {}
    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)
    assert result == {"error": "Integration not loaded"}


@pytest.mark.asyncio
async def test_diagnostics_no_store_in_domain(hass, mock_config_entry):
    """Returns error when domain data exists but no store."""
    hass.data[DOMAIN] = {}
    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)
    assert result == {"error": "Integration not loaded"}


@pytest.mark.asyncio
async def test_diagnostics_no_coordinator(hass, mock_config_entry):
    """Works without coordinator (live states empty)."""
    store = MagicMock()
    store.get_settings.return_value = {"presence_enabled": False}
    store.get_rooms.return_value = {"room_a": {"window_sensors": ["binary_sensor.win"]}}
    hass.data[DOMAIN] = {"store": store}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)

    assert result["rooms"]["room_a"]["live"]["mode"] == "idle"
    assert result["rooms"]["room_a"]["live"]["current_temp"] is None
    assert "model" not in result["rooms"]["room_a"]
    assert result["outdoor"]["temp"] is None
    assert result["recent_history"] == {}
    # The plant settings are exported even before the coordinator exists
    assert result["central_plant"]["settings"]["min_run_minutes"] > 0


@pytest.mark.asyncio
async def test_diagnostics_with_coordinator_and_model(hass, mock_config_entry):
    """Full path with coordinator, rooms, and EKF model."""
    store = MagicMock()
    store.get_settings.return_value = {"presence_enabled": True, "presence_persons": ["person.alice"]}
    store.get_rooms.return_value = {"room_a": {}}

    est = _make_estimator()
    coordinator = _make_coordinator(
        rooms={
            "room_a": {
                "current_temp": 21.5,
                "current_humidity": 55,
                "target_temp": 21.0,
                "mode": "heating",
                "window_open": False,
                "mpc_active": True,
                "confidence": 0.9,
                "presence_away": False,
            }
        },
        estimators={"room_a": est},
    )

    alice_state = MagicMock()
    alice_state.state = "home"
    hass.states.get = MagicMock(return_value=alice_state)

    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)

    assert result["rooms"]["room_a"]["live"]["current_temp"] == 21.5
    assert result["rooms"]["room_a"]["live"]["mpc_active"] is True
    assert "model" in result["rooms"]["room_a"]
    assert result["rooms"]["room_a"]["model"]["beta_fan"] == 0.8
    assert result["outdoor"]["temp"] == 5.0
    assert result["outdoor"]["forecast_available"] is False
    assert result["presence"]["enabled"] is True
    assert result["presence"]["person_states"]["person.alice"] == "home"
    # control_mode was never saved: the export shows the effective default
    assert result["settings"]["control_mode"] == "mpc"


@pytest.mark.asyncio
async def test_diagnostics_room_without_estimator(hass, mock_config_entry):
    """Room without EKF estimator has no model key."""
    store = MagicMock()
    store.get_settings.return_value = {}
    store.get_rooms.return_value = {"room_a": {}}

    coordinator = _make_coordinator(rooms={})

    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)
    assert "model" not in result["rooms"]["room_a"]


@pytest.mark.asyncio
async def test_diagnostics_history_store_with_rows(hass, mock_config_entry):
    """History data is included when history store is available."""
    store = MagicMock()
    store.get_settings.return_value = {}
    store.get_rooms.return_value = {"room_a": {}}

    history_store = MagicMock()
    history_store.read_detail.return_value = [
        {
            "timestamp": "1000",
            "room_temp": "21.0",
            "outdoor_temp": "5.0",
            "target_temp": "21.0",
            "mode": "idle",
            "predicted_temp": "21.1",
        },
    ]

    coordinator = _make_coordinator(history_store=history_store)

    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)
    assert len(result["recent_history"]["room_a"]) == 1
    assert result["recent_history"]["room_a"][0]["ts"] == "1000"


@pytest.mark.asyncio
async def test_diagnostics_history_store_exception(hass, mock_config_entry):
    """History read exception returns empty list for that room."""
    store = MagicMock()
    store.get_settings.return_value = {}
    store.get_rooms.return_value = {"room_a": {}}

    history_store = MagicMock()
    history_store.read_detail.side_effect = RuntimeError("disk error")

    coordinator = _make_coordinator(history_store=history_store)

    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)
    assert result["recent_history"]["room_a"] == []


@pytest.mark.asyncio
async def test_diagnostics_presence_person_unavailable(hass, mock_config_entry):
    """Person entity not found shows 'unavailable'."""
    store = MagicMock()
    store.get_settings.return_value = {
        "presence_enabled": True,
        "presence_persons": ["person.ghost"],
    }
    store.get_rooms.return_value = {}

    hass.states.get = MagicMock(return_value=None)
    hass.data[DOMAIN] = {"store": store}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)
    assert result["presence"]["person_states"]["person.ghost"] == "unavailable"


@pytest.mark.asyncio
async def test_diagnostics_history_capped_at_240(hass, mock_config_entry):
    """History rows are capped at 240 per room."""
    store = MagicMock()
    store.get_settings.return_value = {}
    store.get_rooms.return_value = {"room_a": {}}

    rows = [
        {
            "timestamp": str(i),
            "room_temp": "21.0",
            "outdoor_temp": "5.0",
            "target_temp": "21.0",
            "mode": "idle",
            "predicted_temp": "21.0",
        }
        for i in range(300)
    ]
    history_store = MagicMock()
    history_store.read_detail.return_value = rows

    coordinator = _make_coordinator(history_store=history_store)

    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)
    assert len(result["recent_history"]["room_a"]) == 240


@pytest.mark.asyncio
async def test_diagnostics_window_state(hass, mock_config_entry):
    """Window manager state is included per room."""
    store = MagicMock()
    store.get_settings.return_value = {}
    store.get_rooms.return_value = {"room_a": {}}

    now = time.time()
    coordinator = _make_coordinator(
        window_paused={"room_a": True},
        window_open_since={"room_a": now - 120},
    )
    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)

    window = result["rooms"]["room_a"]["window"]
    assert window["paused"] is True
    assert 118 <= window["open_since"] <= 122


@pytest.mark.asyncio
async def test_diagnostics_cover_state(hass, mock_config_entry):
    """Cover manager state is included when cover exists."""
    store = MagicMock()
    store.get_settings.return_value = {}
    store.get_rooms.return_value = {"room_a": {}}

    now = time.time()
    cover_state = MagicMock()
    cover_state.current_position = 80
    cover_state.last_commanded_position = 80
    cover_state.last_was_forced = False
    cover_state.last_change_ts = now - 300
    cover_state.user_override_until = now + 600

    coordinator = _make_coordinator(cover_states={"room_a": cover_state})
    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)

    cover = result["rooms"]["room_a"]["cover"]
    assert cover["current_position"] == 80
    assert cover["last_commanded_position"] == 80
    assert cover["last_was_forced"] is False
    assert 298 <= cover["last_change_ago_s"] <= 302
    assert 598 <= cover["user_override_remaining_s"] <= 602


@pytest.mark.asyncio
async def test_diagnostics_cover_state_no_override(hass, mock_config_entry):
    """Cover state without active override omits remaining_s."""
    store = MagicMock()
    store.get_settings.return_value = {}
    store.get_rooms.return_value = {"room_a": {}}

    cover_state = MagicMock()
    cover_state.current_position = 100
    cover_state.last_commanded_position = None
    cover_state.last_was_forced = False
    cover_state.last_change_ts = 0
    cover_state.user_override_until = 0
    cover_state.owned = False
    cover_state.baseline_position = None

    coordinator = _make_coordinator(cover_states={"room_a": cover_state})
    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)

    cover = result["rooms"]["room_a"]["cover"]
    assert "user_override_remaining_s" not in cover
    assert "last_change_ago_s" not in cover
    assert cover["owned"] is False
    assert cover["baseline_position"] is None


@pytest.mark.asyncio
async def test_diagnostics_heat_source_routing(hass, mock_config_entry):
    """Heat source routing is still reported when the coordinator carries the legacy state."""
    store = MagicMock()
    store.get_settings.return_value = {}
    store.get_rooms.return_value = {"room_a": {}}

    coordinator = _make_coordinator(heat_source_states={"room_a": "ac_only"})
    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)
    assert result["rooms"]["room_a"]["heat_source_routing"] == "ac_only"


@pytest.mark.asyncio
async def test_diagnostics_without_heat_source_routing(hass, mock_config_entry):
    store = MagicMock()
    store.get_settings.return_value = {}
    store.get_rooms.return_value = {"room_a": {}}

    coordinator = _make_coordinator()
    del coordinator._heat_source_states
    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)
    assert "heat_source_routing" not in result["rooms"]["room_a"]


@pytest.mark.asyncio
async def test_diagnostics_window_closed_since(hass, mock_config_entry):
    store = MagicMock()
    store.get_settings.return_value = {}
    store.get_rooms.return_value = {"room_a": {}}

    now = time.time()
    coordinator = _make_coordinator(
        window_closed_since={"room_a": now - 60},
    )
    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)

    window = result["rooms"]["room_a"]["window"]
    assert 58 <= window["closed_since"] <= 62


@pytest.mark.asyncio
async def test_diagnostics_room_with_temperature_sensor(hass, mock_config_entry):
    store = MagicMock()
    store.get_settings.return_value = {}
    store.get_rooms.return_value = {
        "room_a": {"temperature_sensor": "sensor.temp_living"},
    }

    sensor_state = MagicMock()
    sensor_state.state = "21.3"
    hass.states.get = MagicMock(return_value=sensor_state)

    coordinator = _make_coordinator()
    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)

    assert result["rooms"]["room_a"]["live"]["sensor_state"] == "21.3"


@pytest.mark.asyncio
async def test_diagnostics_room_with_temperature_sensor_not_found(hass, mock_config_entry):
    store = MagicMock()
    store.get_settings.return_value = {}
    store.get_rooms.return_value = {
        "room_a": {"temperature_sensor": "sensor.missing"},
    }

    hass.states.get = MagicMock(return_value=None)

    coordinator = _make_coordinator()
    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)

    assert result["rooms"]["room_a"]["live"]["sensor_state"] == "not_found"


@pytest.mark.asyncio
async def test_diagnostics_room_with_active_schedule(hass, mock_config_entry):
    store = MagicMock()
    store.get_settings.return_value = {}
    store.get_rooms.return_value = {
        "room_a": {
            "schedules": [{"entity_id": "schedule.weekday"}],
        },
    }

    schedule_state = MagicMock()
    schedule_state.state = "on"
    hass.states.get = MagicMock(return_value=schedule_state)

    coordinator = _make_coordinator(
        rooms={"room_a": {"active_schedule_index": 0}},
    )
    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)

    live = result["rooms"]["room_a"]["live"]
    assert live["schedule_entity"] == "schedule.weekday"
    assert live["schedule_state"] == "on"


@pytest.mark.asyncio
async def test_diagnostics_room_schedule_entity_not_found(hass, mock_config_entry):
    store = MagicMock()
    store.get_settings.return_value = {}
    store.get_rooms.return_value = {
        "room_a": {
            "schedules": [{"entity_id": "schedule.missing"}],
        },
    }

    hass.states.get = MagicMock(return_value=None)

    coordinator = _make_coordinator(
        rooms={"room_a": {"active_schedule_index": 0}},
    )
    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)

    live = result["rooms"]["room_a"]["live"]
    assert live["schedule_entity"] == "schedule.missing"
    assert live["schedule_state"] == "not_found"


@pytest.mark.asyncio
async def test_diagnostics_cover_with_last_command_ts(hass, mock_config_entry):
    store = MagicMock()
    store.get_settings.return_value = {}
    store.get_rooms.return_value = {"room_a": {}}

    now = time.time()
    cover_state = MagicMock()
    cover_state.current_position = 50
    cover_state.last_commanded_position = 50
    cover_state.last_was_forced = True
    cover_state.last_change_ts = None
    cover_state.last_command_ts = now - 200
    cover_state.user_override_until = 0

    coordinator = _make_coordinator(cover_states={"room_a": cover_state})
    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)

    cover = result["rooms"]["room_a"]["cover"]
    assert 198 <= cover["last_command_ago_s"] <= 202
    assert "last_change_ago_s" not in cover


@pytest.mark.asyncio
async def test_diagnostics_reports_the_signed_rebound_charge(hass, mock_config_entry):
    """The air/mass charge of the forced-air run replaces the radiator afterglow figures."""
    store = MagicMock()
    store.get_settings.return_value = {}
    store.get_rooms.return_value = {"room_a": {}, "room_b": {}}
    coordinator = _make_coordinator(rebound_charge=-0.42655)
    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)

    assert result["rooms"]["room_a"]["live"]["q_rebound"] == -0.4265
    assert "q_residual" not in result["rooms"]["room_a"]["live"]
    assert "residual_scale" not in result["rooms"]["room_a"]["live"]


@pytest.mark.asyncio
async def test_diagnostics_without_a_rebound_tracker(hass, mock_config_entry):
    store = MagicMock()
    store.get_settings.return_value = {}
    store.get_rooms.return_value = {"room_a": {}}
    coordinator = _make_coordinator()
    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)

    assert "q_rebound" not in result["rooms"]["room_a"]["live"]


@pytest.mark.asyncio
async def test_diagnostics_export_the_central_plant(hass, mock_config_entry):
    """The plant section carries settings, thermostat state and the live managers."""
    store = MagicMock()
    store.get_settings.return_value = {"central_plant": {"thermostat_entity": "climate.hall", "min_run_minutes": 9}}
    store.get_rooms.return_value = {}

    wet_coil = MagicMock()
    wet_coil.to_dict.return_value = {"charge": 0.0, "last_cooling_ts": None}
    plant_manager = MagicMock()
    plant_manager.to_dict.return_value = {"active_action": "idle"}
    coordinator = _make_coordinator(
        central_plant={"action": "circulating", "fan_circulating": True},
        wet_coil=wet_coil,
        plant_manager=plant_manager,
    )
    hass.states.get = MagicMock(return_value=_thermostat_state(hvac_mode="fan_only", current_temperature=19.5))
    hass.data[DOMAIN] = {"store": store, "coordinator": coordinator}

    result = await async_get_config_entry_diagnostics(hass, mock_config_entry)

    plant = result["central_plant"]
    assert plant["settings"]["min_run_minutes"] == 9
    assert plant["thermostat"]["state"] == "heat"
    assert plant["thermostat"]["hvac_mode"] == "fan_only"
    assert plant["live"] == {"action": "circulating", "fan_circulating": True}
    assert plant["wet_coil"] == {"charge": 0.0, "last_cooling_ts": None}
    assert plant["plant_manager"] == {"active_action": "idle"}
    # The AC-era per-device snapshots are gone
    assert "compressor_groups" not in result
    assert "valve_protection" not in result
    assert "coil_dry_state" not in result
    assert "ac_head_offsets" not in result
