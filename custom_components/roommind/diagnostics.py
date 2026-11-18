"""Diagnostics support for RoomMind."""

from __future__ import annotations

import time
from datetime import time as dt_time
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import __version__ as HA_VERSION
from homeassistant.core import HomeAssistant

from .const import DEFAULT_CONTROL_MODE, DOMAIN, VERSION, get_central_plant_settings
from .utils.history_store import DETAIL_FIELDS

# v2 adds what a simulation needs to rebuild a setup (referenced entity states, the
# full 9-state EKF, the Central Plant, schedule blocks, 48 h history).
DIAGNOSTICS_SCHEMA_VERSION = 2
HISTORY_48H_MAX_ROWS = 1000
_ROOM_ENTITY_KEYS = (
    "temperature_sensor",
    "humidity_sensor",
    "schedule_selector_entity",
    "cover_schedule_selector_entity",
)
# Exports end up in public issues: no person entities, no timing of occupancy.
_ROOM_ENTITY_LIST_KEYS = (
    "window_sensors",
    "exterior_window_sensors",
    "interior_door_sensors",
    "occupancy_sensors",
    "covers",
    "read_only_covers",
)
# Climate attributes of the Central Thermostat worth exporting (a public issue needs these)
_THERMOSTAT_ATTRIBUTES = (
    "hvac_mode",
    "hvac_modes",
    "hvac_action",
    "current_temperature",
    "temperature",
    "target_temp_low",
    "target_temp_high",
    "min_temp",
    "max_temp",
    "target_temp_step",
    "preset_mode",
    "preset_modes",
    "fan_mode",
    "fan_modes",
    "supported_features",
)
_PRIVATE_DEVICE_CLASSES = ("occupancy", "presence", "motion")
HISTORY_48H_FIELDS = [f for f in DETAIL_FIELDS if f != "occupancy"]


def _build_model_info(estimator: Any) -> dict[str, Any]:
    """Build model diagnostics from a ThermalEKF estimator.

    The state vector is the 9-state Central Forced-Air model
    ``x = [T, alpha_out, alpha_house, beta_h, beta_c, beta_fan, beta_s, beta_o, beta_f]``;
    ``alpha`` stays as an alias of ``alpha_out`` for consumers of the older export.
    """
    rc = estimator.get_model()
    return {
        "alpha": round(estimator._x[1], 6),
        "alpha_out": round(estimator._x[1], 6),
        "alpha_house": round(estimator._x[2], 4),
        "beta_h": round(estimator._x[3], 4),
        "beta_c": round(estimator._x[4], 4),
        "beta_fan": round(estimator._x[5], 4),
        "beta_s": round(estimator._x[6], 4),
        "beta_o": round(estimator._x[7], 4),
        "beta_f": round(estimator._x[8], 4),
        "n_updates": estimator._n_updates,
        "n_idle": estimator._n_idle,
        "n_heating": estimator._n_heating,
        "n_cooling": estimator._n_cooling,
        "n_circulating": getattr(estimator, "_n_circulating", 0),
        "applicable_modes": sorted(estimator._applicable_modes),
        "P_diagonal": [round(estimator._P[i][i], 6) for i in range(len(estimator._x))],
        "prediction_std_idle": round(estimator.prediction_std(0.0, 20.0, 15.0, 5.0), 4),
        "prediction_std_heating": round(estimator.prediction_std(rc.Q_heat, 20.0, 10.0, 5.0), 4),
        "confidence": round(estimator.confidence, 4),
        "model_params": rc.to_dict(),
    }


def _thermostat_snapshot(hass: HomeAssistant, entity_id: str) -> dict[str, Any]:
    """Live state and climate attributes of the Central Thermostat (or "not_found")."""
    state = hass.states.get(entity_id)
    if state is None:
        return {"entity_id": entity_id, "state": "not_found"}
    attrs = state.attributes
    return {
        "entity_id": entity_id,
        "state": state.state,
        **{key: attrs.get(key) for key in _THERMOSTAT_ATTRIBUTES if key in attrs},
    }


def _live_snapshot(value: Any) -> Any:
    """A manager's live state as plain data: dicts pass through, objects use ``to_dict``."""
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    return to_dict() if callable(to_dict) else None


def _build_central_plant_snapshot(hass: HomeAssistant, settings: dict, coordinator: Any) -> dict[str, Any]:
    """Everything about the shared forced-air plant a public issue needs.

    The resolved plant settings, the Central Thermostat's live state, the coordinator's
    live plant summary and the two stateful helpers behind it (the wet-coil charge that
    blocks Circulation and the equipment guards that outlive a control cycle).
    """
    plant = get_central_plant_settings(settings)
    thermostat = str(plant.get("thermostat_entity") or "")
    snapshot: dict[str, Any] = {
        "settings": plant,
        "thermostat_entity": thermostat,
    }
    if thermostat:
        snapshot["thermostat"] = _thermostat_snapshot(hass, thermostat)
    if coordinator is None:
        return snapshot
    live = _live_snapshot(getattr(coordinator, "central_plant", None))
    if live is not None:
        snapshot["live"] = live
    wet_coil = _live_snapshot(getattr(coordinator, "_wet_coil", None))
    if wet_coil is not None:
        snapshot["wet_coil"] = wet_coil
    plant_manager = _live_snapshot(getattr(coordinator, "_plant_manager", None))
    if plant_manager is not None:
        snapshot["plant_manager"] = plant_manager
    return snapshot


def _rebound_charge(coordinator: Any, area_id: str) -> float | None:
    """Signed air/mass charge of the room's forced-air run, None without a tracker."""
    tracker = getattr(coordinator, "_rebound_tracker", None)
    charge = getattr(tracker, "charge", None)
    if not callable(charge):
        return None
    value = charge(area_id)
    return round(float(value), 4) if isinstance(value, int | float) else None


def _plain(value: Any) -> Any:
    """Schedule blocks hold datetime.time objects; export them as "HH:MM:SS" strings."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dt_time):
        return value.isoformat()
    return value


def _build_entity_states(hass: HomeAssistant, config: dict) -> dict[str, dict[str, Any]]:
    """State of every entity a room refers to (sensors, windows, schedules, ...)."""
    ids: list[str] = [config.get(k, "") for k in _ROOM_ENTITY_KEYS]
    for key in _ROOM_ENTITY_LIST_KEYS:
        ids.extend(config.get(key) or [])
    for key in ("schedules", "cover_schedules"):
        ids.extend(s.get("entity_id", "") for s in config.get(key) or [])
    now = time.time()
    result: dict[str, dict[str, Any]] = {}
    for eid in ids:
        if not eid or eid in result:
            continue
        state = hass.states.get(eid)
        if state is None:
            result[eid] = {"state": "not_found"}
            continue
        device_class = state.attributes.get("device_class")
        entry = {
            "state": state.state,
            "unit": state.attributes.get("unit_of_measurement"),
            "device_class": device_class,
        }
        if device_class not in _PRIVATE_DEVICE_CLASSES:
            entry["age_s"] = round(now - state.last_changed.timestamp())
        result[eid] = entry
    return result


def _build_window_state(coordinator: Any, area_id: str) -> dict[str, Any]:
    """Build window manager state for a room."""
    wm = coordinator._window_manager
    now = time.time()
    result: dict[str, Any] = {
        "paused": wm._paused.get(area_id, False),
    }
    open_since = wm._open_since.get(area_id)
    if open_since:
        result["open_since"] = round(now - open_since)
    closed_since = wm._closed_since.get(area_id)
    if closed_since:
        result["closed_since"] = round(now - closed_since)
    return result


def _build_cover_state(coordinator: Any, area_id: str) -> dict[str, Any] | None:
    """Build cover manager state for a room.

    The manager lives on the CoverOrchestrator, not on the coordinator. Reading the old
    attribute raised AttributeError inside the websocket handler, which HA reported as
    ``unknown_error`` — the whole export failed, so the panel's buttons did nothing.
    The chain is defensive so a future move costs this one section, not the export.
    """
    orchestrator = getattr(coordinator, "_cover_orchestrator", None)
    cm = getattr(orchestrator, "_cover_manager", None)
    states = getattr(cm, "_states", None)
    if not states or area_id not in states:
        return None
    cs = states[area_id]
    now = time.time()
    result: dict[str, Any] = {
        "current_position": cs.current_position,
        "last_commanded_position": cs.last_commanded_position,
        "last_was_forced": cs.last_was_forced,
        "owned": cs.owned,
        "baseline_position": cs.baseline_position,
    }
    if cs.last_change_ts:
        result["last_change_ago_s"] = round(now - cs.last_change_ts)
    if cs.last_command_ts:
        result["last_command_ago_s"] = round(now - cs.last_command_ts)
    if cs.user_override_until > now:
        result["user_override_remaining_s"] = round(cs.user_override_until - now)

    return result


async def async_get_config_entry_diagnostics(hass: HomeAssistant, config_entry: ConfigEntry) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    data = hass.data.get(DOMAIN, {})
    store = data.get("store")
    coordinator = data.get("coordinator")

    if not store:
        return {"error": "Integration not loaded"}

    settings = store.get_settings()
    rooms_config = store.get_rooms()
    live_states = coordinator.rooms if coordinator else {}

    # Build per-room diagnostics
    rooms_diag: dict[str, dict] = {}
    for area_id, config in rooms_config.items():
        live = live_states.get(area_id, {})
        # Expose all room state fields with sensible defaults
        live_diag: dict[str, Any] = {
            "current_temp": None,
            "current_humidity": None,
            "target_temp": None,
            "heat_target": None,
            "cool_target": None,
            "mode": "idle",
            "heating_power": 0,
            "device_setpoint": None,
            "window_open": False,
            "override_active": False,
            "mpc_active": False,
            "confidence": None,
            "presence_away": False,
            "force_off": False,
            "n_observations": 0,
        }
        live_diag.update({k: v for k, v in live.items() if k not in ("area_id", "current_temp_raw")})
        live_diag["ignore_presence"] = config.get("ignore_presence", False)

        # Sensor entity availability
        temp_sensor_id = config.get("temperature_sensor", "")
        if temp_sensor_id:
            ts = hass.states.get(temp_sensor_id)
            live_diag["sensor_state"] = ts.state if ts else "not_found"
        else:
            live_diag["sensor_state"] = "no_sensor"

        # Coordinator internal state for this room
        if coordinator:
            live_diag["previous_mode"] = coordinator._previous_modes.get(area_id, "idle")
            on_since = coordinator._mode_on_since.get(area_id)
            if on_since is not None:
                live_diag["mode_active_for_s"] = round(time.time() - on_since)
            cached = coordinator._last_valid_temps.get(area_id)
            if cached is not None:
                live_diag["cached_temp"] = cached[0]
                live_diag["cached_temp_age_s"] = round(time.monotonic() - cached[1])

        # Forced-air state: the signed air/mass charge of the run in progress
        if coordinator:
            charge = _rebound_charge(coordinator, area_id)
            if charge is not None:
                live_diag["q_rebound"] = charge

        # Schedule entity state
        schedules = config.get("schedules", [])
        if schedules:
            active_idx = live.get("active_schedule_index", -1)
            if 0 <= active_idx < len(schedules):
                sched_eid = schedules[active_idx].get("entity_id", "")
                if sched_eid:
                    ss = hass.states.get(sched_eid)
                    live_diag["schedule_entity"] = sched_eid
                    live_diag["schedule_state"] = ss.state if ss else "not_found"

        room_diag: dict[str, Any] = {
            "config": dict(config),
            "live": live_diag,
        }

        room_diag["entities"] = _build_entity_states(hass, config)

        # Model info from EKF estimator
        if coordinator:
            mgr = coordinator._model_manager
            if area_id in mgr._estimators:
                room_diag["model"] = _build_model_info(mgr._estimators[area_id])
                room_diag["model_state"] = mgr._estimators[area_id].to_dict()
            cache = getattr(coordinator, "_schedule_blocks_cache", None) or {}
            blocks = {
                s["entity_id"]: _plain(cache[s["entity_id"]])
                for s in config.get("schedules") or []
                if s.get("entity_id") in cache
            }
            if blocks:
                room_diag["schedule_blocks"] = blocks

        # Window manager state
        if coordinator:
            room_diag["window"] = _build_window_state(coordinator, area_id)

        # Cover manager state
        if coordinator:
            cover = _build_cover_state(coordinator, area_id)
            if cover:
                room_diag["cover"] = cover

        # Heat source routing of the room, when the orchestration still reports it
        states = getattr(coordinator, "_heat_source_states", {}) if coordinator else {}
        if area_id in states:
            room_diag["heat_source_routing"] = states[area_id]

        rooms_diag[area_id] = room_diag

    # Outdoor conditions
    outdoor: dict[str, Any] = {
        "temp": coordinator.outdoor_temp if coordinator else None,
        "humidity": coordinator.outdoor_humidity if coordinator else None,
    }
    if coordinator:
        forecast = coordinator._weather_manager._outdoor_forecast
        outdoor["forecast_available"] = bool(forecast)
        outdoor["forecast_points"] = len(forecast) if forecast else 0

    # Recent history (last 2 hours of detail data per room)
    recent_history: dict[str, list] = {}
    if coordinator and coordinator._history_store:
        for area_id in rooms_config:
            try:
                rows = await hass.async_add_executor_job(coordinator._history_store.read_detail, area_id, 7200)
                recent_history[area_id] = [
                    {
                        "ts": row.get("timestamp", ""),
                        "room_temp": row.get("room_temp", ""),
                        "outdoor_temp": row.get("outdoor_temp", ""),
                        "target_temp": row.get("target_temp", ""),
                        "mode": row.get("mode", ""),
                        "predicted_temp": row.get("predicted_temp", ""),
                    }
                    for row in rows[-240:]  # Cap at ~240 points
                ]
            except Exception:  # noqa: BLE001
                recent_history[area_id] = []

    history_48h: dict[str, list] = {}
    if coordinator and coordinator._history_store:
        for area_id in rooms_config:
            try:
                rows = await hass.async_add_executor_job(coordinator._history_store.read_detail, area_id, 48 * 3600)
                history_48h[area_id] = [
                    {field: row.get(field, "") for field in HISTORY_48H_FIELDS} for row in rows[-HISTORY_48H_MAX_ROWS:]
                ]
            except Exception:  # noqa: BLE001
                history_48h[area_id] = []

    # Central Plant: the shared forced-air system behind every room decision
    central_plant = _build_central_plant_snapshot(hass, settings, coordinator)

    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "integration": {
            "version": VERSION,
            "domain": DOMAIN,
            "ha_version": HA_VERSION,
            "ha_temp_unit": hass.config.units.temperature_unit,
            "unit_system": "us_customary" if hass.config.units.temperature_unit == "°F" else "metric",
            "time_zone": hass.config.time_zone,
            # Whole degrees: enough for the sun position, not a home address (exports are public).
            "latitude": round(hass.config.latitude),
            "longitude": round(hass.config.longitude),
        },
        "settings": {"control_mode": DEFAULT_CONTROL_MODE, **settings},
        "rooms": rooms_diag,
        "outdoor": outdoor,
        "recent_history": recent_history,
        "history_48h": history_48h,
        "central_plant": central_plant,
        "presence": {
            "enabled": settings.get("presence_enabled", False),
            "persons": settings.get("presence_persons", []),
            "person_states": {
                pid: (s.state if (s := hass.states.get(pid)) else "unavailable")
                for pid in settings.get("presence_persons", [])
            },
        },
    }
