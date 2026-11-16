"""WebSocket API for RoomMind room CRUD operations."""

from __future__ import annotations

import inspect
import time
from typing import TYPE_CHECKING

import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.core import HomeAssistant, callback

from .const import (
    CLIMATE_MODES,
    CONTROL_MODE_BANGBANG,
    CONTROL_MODE_MPC,
    DEFAULT_CONTROL_MODE,
    DOMAIN,
    EXTERIOR_OPEN_POLICIES,
    MAX_CO2_THRESHOLD,
    MAX_PRICE_SHIFT_K,
    MAX_TARGET_TEMP,
    MIN_CO2_THRESHOLD,
    MIN_OCCUPANCY_ECO_DELAY_MINUTES,
    MIN_TARGET_TEMP,
    OCCUPANCY_MODES,
    OVERRIDE_CUSTOM,
    OVERRIDE_TYPES,
    PRICE_SHIFT_MODES,
    VENTILATION_DOMAINS,
    build_override_live,
    get_central_plant_settings,
    get_house_settings,
    is_override_suppressed,
)
from .services.analytics_service import (
    _compute_target_forecast,  # noqa: F401 - re-exported for tests
    _csv_to_points,  # noqa: F401 - re-exported for tests
    _safe_float,  # noqa: F401 - re-exported for tests
    build_analytics_data,
)
from .utils.area_utils import REST_SWITCHED_OFF, rooms_with_area, rooms_without_area
from .utils.schedule_utils import mask_override_band, next_schedule_change, override_preset_band

if TYPE_CHECKING:
    from homeassistant.components.websocket_api import ActiveConnection

    from .coordinator import RoomMindCoordinator
    from .store import RoomMindStore


def _get_coordinator(hass: HomeAssistant) -> RoomMindCoordinator | None:
    """Return the RoomMindCoordinator from hass.data, or None."""
    coordinator: RoomMindCoordinator | None = hass.data.get(DOMAIN, {}).get("coordinator")
    return coordinator


_ROOM_SAVE_FIELDS = (
    "temperature_sensor",
    "humidity_sensor",
    "occupancy_sensors",
    "occupancy_mode",
    "occupancy_dwell_minutes",
    # Legacy alias for occupancy_mode = reactive_and_trend; still written for old cards
    "occupancy_eco_enabled",
    "occupancy_eco_delay",
    "preheat_enabled",
    "price_shift",
    "climate_mode",
    "schedules",
    "schedule_selector_entity",
    # Canonical exterior-opening field; the legacy window_sensors alias is mirrored onto it
    "exterior_window_sensors",
    "window_sensors",
    "window_open_delay",
    "window_close_delay",
    "interior_door_sensors",
    "interior_door_delay",
    "door_closed_mix_factor",
    "comfort_temp",
    "eco_temp",
    "comfort_heat",
    "comfort_cool",
    "eco_heat",
    "eco_cool",
    "presence_persons",
    "display_name",
    "covers",
    "read_only_covers",
    "covers_auto_enabled",
    "covers_deploy_threshold",
    "covers_min_position",
    "covers_outdoor_min_temp",
    "covers_override_minutes",
    "cover_schedules",
    "cover_schedule_selector_entity",
    "cover_orientations",
    "covers_night_close",
    "covers_night_close_elevation",
    "covers_night_close_offset_minutes",
    "covers_night_position",
    "covers_snap_deploy",
    "covers_window_hold",
    "cover_min_positions",
    "ignore_presence",
    "is_outdoor",
    "climate_control_enabled",
    "co2_sensor",
    "co2_threshold",
    "voc_sensor",
    "voc_threshold",
    "ventilation_entity",
    "follow_house",
)

# Same limits as the room climate entity; also turns away nan/inf
_TARGET_TEMP = vol.All(vol.Coerce(float), vol.Range(min=MIN_TARGET_TEMP, max=MAX_TARGET_TEMP))

# What the dashboard card saves; the only room fields a non-admin may send
_USER_ROOM_FIELDS = frozenset(
    {
        "comfort_temp",
        "eco_temp",
        "comfort_heat",
        "comfort_cool",
        "eco_heat",
        "eco_cool",
        "climate_mode",
        "climate_control_enabled",
        "follow_house",
    }
)

_SETTINGS_SAVE_FIELDS = (
    "outdoor_temp_sensor",
    "price_entity",
    "outdoor_humidity_sensor",
    "outdoor_cooling_min",
    "outdoor_heating_max",
    "control_mode",
    "comfort_weight",
    "weather_entity",
    "outdoor_unavailable_notify",
    "climate_control_active",
    "learning_disabled_rooms",
    "hidden_rooms",
    "vacation_temp",
    "vacation_until",
    "prediction_enabled",
    "presence_enabled",
    "presence_persons",
    "presence_away_action",
    "presence_clears_override",
    "schedule_off_action",
    "mold_detection_enabled",
    "mold_humidity_threshold",
    "mold_f_rsi",
    "mold_sustained_minutes",
    "mold_notification_cooldown",
    "mold_notifications_enabled",
    "mold_notification_targets",
    "mold_prevention_enabled",
    "mold_prevention_intensity",
    "mold_prevention_notify_enabled",
    "mold_prevention_notify_targets",
    "room_order",
    "group_by_floor",
    "central_plant",
    "house",
)


# _safe_float, _csv_to_points and _compute_target_forecast are imported from
# .services.analytics_service (see imports above) and re-exported so that
# existing callers (incl. tests) keep working.


def _compute_anyone_home(hass: HomeAssistant, settings: dict) -> bool:
    """Return True if at least one tracked person is home (or fail-safe)."""
    from .utils.presence_utils import is_presence_away

    return not is_presence_away(hass, {}, settings)  # all away


def _validate_no_own_entities(config: dict, own_prefix: str) -> str | None:
    """Check that no RoomMind-owned entities are assigned. Returns error message or None."""
    for field in (
        "exterior_window_sensors",
        "window_sensors",
        "covers",
        "read_only_covers",
        "interior_door_sensors",
        "occupancy_sensors",
    ):
        for eid in config.get(field) or []:
            if eid.split(".", 1)[-1].startswith(own_prefix):
                return f"Cannot assign RoomMind's own entity '{eid}' to a room"
    for field in ("schedules", "cover_schedules"):
        for entry in config.get(field) or []:
            eid = entry.get("entity_id", "") if isinstance(entry, dict) else ""
            if eid.split(".", 1)[-1].startswith(own_prefix):
                return f"Cannot assign RoomMind's own entity '{eid}' to a room"
    for field in (
        "temperature_sensor",
        "humidity_sensor",
        "co2_sensor",
        "voc_sensor",
        "ventilation_entity",
        "schedule_selector_entity",
        "cover_schedule_selector_entity",
    ):
        eid = config.get(field) or ""
        if eid and eid.split(".", 1)[-1].startswith(own_prefix):
            return f"Cannot assign RoomMind's own entity '{eid}' to a room"
    return None


def _read_only_cover_entity(value: object) -> str:
    """A ``cover.*`` entity mirrored read-only, or a ``sensor.*`` carrying a cover position."""
    if not isinstance(value, str) or not value.startswith(("cover.", "sensor.")):
        raise vol.Invalid("must be a cover or sensor entity")
    if value.split(".", 1)[-1].startswith(f"{DOMAIN}_"):
        raise vol.Invalid("Cannot assign RoomMind's own entity as a read-only cover")
    return value


def _plant_thermostat_entity(value: object) -> str:
    """A Central Thermostat ``climate.*`` entity, "" when unset; never one of RoomMind's own."""
    if value is None or value == "":
        return ""
    if not isinstance(value, str) or not value.startswith("climate."):
        raise vol.Invalid("thermostat_entity must be a climate entity")
    if value.split(".", 1)[-1].startswith(f"{DOMAIN}_"):
        raise vol.Invalid("Cannot assign RoomMind's own entity as the Central Thermostat")
    return value


# The Central Plant settings block; every key is optional so the panel can save one field at a time
_CENTRAL_PLANT_SCHEMA = {
    vol.Optional("thermostat_entity"): _plant_thermostat_entity,
    vol.Optional("thermostat_area_id"): str,
    vol.Optional("nudge_delta_c"): vol.All(vol.Coerce(float), vol.Range(min=0.5, max=3.5)),
    vol.Optional("max_nudge_headroom_c"): vol.All(vol.Coerce(float), vol.Range(min=0.5, max=6.0)),
    vol.Optional("fallback_heat_c"): vol.All(vol.Coerce(float), vol.Range(min=5.0, max=25.0)),
    vol.Optional("fallback_cool_c"): vol.All(vol.Coerce(float), vol.Range(min=18.0, max=35.0)),
    vol.Optional("min_deadband_c"): vol.All(vol.Coerce(float), vol.Range(min=1.0, max=5.0)),
    vol.Optional("circulate_fan_mode"): str,
    vol.Optional("idle_fan_mode"): str,
    vol.Optional("min_run_minutes"): vol.All(vol.Coerce(int), vol.Range(min=1, max=60)),
    vol.Optional("min_off_minutes"): vol.All(vol.Coerce(int), vol.Range(min=1, max=30)),
    vol.Optional("min_fan_minutes"): vol.All(vol.Coerce(int), vol.Range(min=1, max=60)),
    vol.Optional("changeover_minutes"): vol.All(vol.Coerce(int), vol.Range(min=5, max=180)),
    vol.Optional("follow_thermostat_setpoint"): bool,
    vol.Optional("thermostat_override_hours"): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=24.0)),
    vol.Optional("exterior_open_policy"): vol.In(EXTERIOR_OPEN_POLICIES),
    vol.Optional("humidity_comfort_enabled"): bool,
    vol.Optional("dehumidify_rh_threshold"): vol.All(vol.Coerce(float), vol.Range(min=40.0, max=80.0)),
    vol.Optional("max_dehumidify_overcool_c"): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.5)),
    vol.Optional("wet_coil_lockout_minutes"): vol.All(vol.Coerce(int), vol.Range(min=0, max=120)),
    vol.Optional("wet_coil_max_rh"): vol.All(vol.Coerce(float), vol.Range(min=35.0, max=80.0)),
    vol.Optional("price_shift_modes"): vol.In(PRICE_SHIFT_MODES),
}


# The whole-House settings block. Every key is optional so a panel can save one
# field at a time; extra keys are rejected (vol.Schema default: PREVENT_EXTRA).
_HOUSE_SCHEMA = vol.Schema(
    {
        vol.Optional("enabled"): bool,
        vol.Optional("heat_c"): vol.Any(_TARGET_TEMP, None),
        vol.Optional("cool_c"): vol.Any(_TARGET_TEMP, None),
        vol.Optional("override_heat"): vol.Any(_TARGET_TEMP, None),
        vol.Optional("override_cool"): vol.Any(_TARGET_TEMP, None),
        vol.Optional("eco_heat_c"): _TARGET_TEMP,
        vol.Optional("eco_cool_c"): _TARGET_TEMP,
        vol.Optional("climate_mode"): vol.In(CLIMATE_MODES),
        vol.Optional("schedules"): [{vol.Required("entity_id"): str}],
        vol.Optional("schedule_selector_entity"): str,
        vol.Optional("preheat_enabled"): bool,
        vol.Optional("price_shift"): vol.All(vol.Coerce(float), vol.Range(min=0, max=MAX_PRICE_SHIFT_K)),
        vol.Optional("override_until"): vol.Any(vol.Coerce(float), None),
        vol.Optional("override_type"): vol.Any(vol.In(OVERRIDE_TYPES), None),
    }
)

# Fields a non-admin may write through roommind/house/save: the band/override
# targets only. Structure (schedules, mode, enabled, ...) stays admin-only.
_USER_HOUSE_FIELDS = frozenset(
    {
        "heat_c",
        "cool_c",
        "eco_heat_c",
        "eco_cool_c",
        "override_heat",
        "override_cool",
        "override_until",
        "override_type",
    }
)


def _central_plant_live(coordinator: RoomMindCoordinator | None) -> dict:
    """Return the coordinator's live Central Plant state, or an empty dict."""
    if coordinator is None:
        return {}
    data = coordinator.data or {}
    plant = data.get("central_plant") if isinstance(data, dict) else None
    return plant if isinstance(plant, dict) else {}


def _house_live(coordinator: RoomMindCoordinator | None, store: RoomMindStore) -> dict:
    """Return the coordinator's live House block, falling back to the store.

    The coordinator does not publish ``data["house"]`` yet; until it does (and
    on the very first cycle before it has run) answer with the configured
    settings so the panel can always render. Never raises for a None or
    never-run coordinator.
    """
    if coordinator is not None:
        data = coordinator.data or {}
        house = data.get("house") if isinstance(data, dict) else None
        if isinstance(house, dict):
            return house
    return get_house_settings(store.get_settings())


def _validate_house_band(settings: dict, changes: dict) -> bool:
    """True when the merged house comfort/eco bands stay ordered."""
    resolved: dict = {**get_house_settings(settings), **changes}
    heat, cool = resolved.get("heat_c"), resolved.get("cool_c")
    if isinstance(heat, (int, float)) and isinstance(cool, (int, float)):
        if float(cool) <= float(heat):
            return False
    eco_heat, eco_cool = resolved.get("eco_heat_c"), resolved.get("eco_cool_c")
    if isinstance(eco_heat, (int, float)) and isinstance(eco_cool, (int, float)):
        if float(eco_cool) < float(eco_heat):
            return False
    return True


# ---------------------------------------------------------------------------
# List rooms
# ---------------------------------------------------------------------------


def _build_room_live(
    coordinator: RoomMindCoordinator | None,
    area_id: str,
    room_config: dict,
    settings: dict,
    learning_paused_reason: str | None,
) -> dict:
    """Build the live payload of one room from the coordinator's latest cycle.

    The coordinator only publishes what it evaluated this cycle; everything the
    dashboard reads has to be named here or it is silently dropped (#386).
    """
    live: dict = (coordinator.rooms.get(area_id, {}) if coordinator else {}) or {}
    learning_disabled = set(settings.get("learning_disabled_rooms", []) or [])
    boost_anchors = settings.get("boost_applied_at", {}) or {}
    return {
        "current_temp": live.get("current_temp"),
        "current_humidity": live.get("current_humidity"),
        "effective_temp": live.get("effective_temp"),
        "target_temp": live.get("target_temp"),
        "heat_target": live.get("heat_target"),
        "cool_target": live.get("cool_target"),
        "mode": live.get("mode", "idle"),
        "heating_power": live.get("heating_power", 0),
        "device_setpoint": live.get("device_setpoint"),
        "window_open": live.get("window_open", False),
        **build_override_live(
            room_config,
            suppressed=is_override_suppressed(room_config, settings, live.get("presence_away", False)),
        ),
        "active_schedule_index": live.get("active_schedule_index", -1),
        "confidence": live.get("confidence"),
        "mpc_active": live.get("mpc_active", False),
        "presence_away": live.get("presence_away", False),
        "occupancy_vacant": live.get("occupancy_vacant", False),
        # Room Priority and occupancy from the whole-home optimizer
        "priority_weight": live.get("priority_weight", 0.0),
        "occupancy_probability": live.get("occupancy_probability", 0.0),
        "occupancy_dwell": live.get("occupancy_dwell", 0.0),
        "interior_door_open": live.get("interior_door_open", False),
        "exterior_window_open": live.get("exterior_window_open", False),
        "fan_circulating": live.get("fan_circulating", False),
        "beta_fan_rate": live.get("beta_fan_rate", 0.0),
        "co2": live.get("co2"),
        "voc": live.get("voc"),
        "air_quality_bad": live.get("air_quality_bad", False),
        "air_quality_reason": live.get("air_quality_reason"),
        "ventilation_active": live.get("ventilation_active", False),
        "mold_risk_level": live.get("mold_risk_level", "ok"),
        "mold_surface_rh": live.get("mold_surface_rh"),
        "mold_surface_temp": live.get("mold_surface_temp"),
        "mold_prevention_active": live.get("mold_prevention_active", False),
        "mold_prevention_delta": live.get("mold_prevention_delta", 0),
        "n_observations": live.get("n_observations", 0),
        # A paused room does not fade the boost, so no countdown for it
        "boost_applied_at": None if area_id in learning_disabled else boost_anchors.get(area_id),
        "blind_position": live.get("blind_position"),
        "cover_auto_paused": live.get("cover_auto_paused", False),
        "cover_override_until": live.get("cover_override_until"),
        "cover_forced_reason": live.get("cover_forced_reason", ""),
        "cover_reason": live.get("cover_reason", ""),
        "active_cover_schedule_index": live.get("active_cover_schedule_index", -1),
        "learning_paused_reason": learning_paused_reason,
        "schedule_temp_warnings": live.get("schedule_temp_warnings", []),
        "price_level": live.get("price_level"),
        "preheat": live.get("preheat"),
        "price_shift_heat": live.get("price_shift_heat", 0.0),
        "price_shift_cool": live.get("price_shift_cool", 0.0),
        "recent_temps": coordinator.recent_temps.get(area_id) if coordinator else [],
    }


@websocket_api.websocket_command({vol.Required("type"): "roommind/rooms/list"})
@websocket_api.async_response
async def websocket_list_rooms(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: dict,
) -> None:
    """Return all rooms with current state."""
    store = hass.data[DOMAIN]["store"]
    stored = store.get_rooms()
    rooms = rooms_with_area(hass, stored)

    # Merge live state from coordinator
    coordinator = _get_coordinator(hass)
    # If coordinator has no data yet but rooms exist, trigger an immediate refresh
    if coordinator and rooms and not coordinator.rooms:
        await coordinator.async_request_refresh()

    # Outdoor temperature availability gates EKF training (#301).  Surface the
    # paused state per room so the frontend can show an explanatory badge.
    outdoor_available = coordinator is not None and coordinator.outdoor_temp_effective is not None
    settings = store.get_settings()
    learning_disabled = set(settings.get("learning_disabled_rooms", []))

    # Build response: config + live state per room
    # Override fields are computed from the store (always up-to-date) rather
    # than from the coordinator (which refreshes on a 10s cycle).
    result = {}
    for area_id, room_config in rooms.items():
        room_data = dict(room_config)
        # Managed Mode rooms train too (on the device reading), only outdoor areas never do.
        if not outdoor_available and area_id not in learning_disabled and not room_config.get("is_outdoor", False):
            learning_paused_reason: str | None = "outdoor_unavailable"
        else:
            learning_paused_reason = None
        room_data["live"] = _build_room_live(coordinator, area_id, room_config, settings, learning_paused_reason)
        result[area_id] = room_data

    # The plant has not run yet (no coordinator or first cycle still pending):
    # answer with the configured settings so the panel can render.
    central_plant = _central_plant_live(coordinator) or get_central_plant_settings(settings)

    # Vacation state from settings
    vacation_until = settings.get("vacation_until")
    vacation_active = bool(vacation_until and time.time() < vacation_until)

    connection.send_result(
        msg["id"],
        {
            "rooms": result,
            "rooms_without_area": [
                {"area_id": area_id, "display_name": room.get("display_name", "")}
                for area_id, room in rooms_without_area(hass, stored).items()
            ],
            "central_plant": central_plant,
            "house": _house_live(coordinator, store),
            # Countdowns in the browser compare against this, not against the device's own clock
            "server_time": time.time(),
            "outdoor_temp": coordinator.outdoor_temp_effective if coordinator else None,
            "outdoor_humidity": coordinator.outdoor_humidity if coordinator else None,
            "vacation_active": vacation_active,
            "vacation_temp": settings.get("vacation_temp") if vacation_active else None,
            "vacation_until": vacation_until if vacation_active else None,
            "hidden_rooms": settings.get("hidden_rooms", []),
            "room_order": settings.get("room_order", []),
            "group_by_floor": settings.get("group_by_floor", False),
            "control_mode": settings.get("control_mode", DEFAULT_CONTROL_MODE),
            "climate_control_active": settings.get("climate_control_active", True),
            "presence_enabled": settings.get("presence_enabled", False),
            "price_entity": settings.get("price_entity", ""),
            "price_status": coordinator.price_status() if coordinator else None,
            "presence_persons": settings.get("presence_persons", []),
            "presence_away_action": settings.get("presence_away_action", "eco"),
            "presence_clears_override": settings.get("presence_clears_override", False),
            "schedule_off_action": settings.get("schedule_off_action", "eco"),
            "anyone_home": _compute_anyone_home(hass, settings),
        },
    )


# ---------------------------------------------------------------------------
# Save room (upsert: create or update)
# ---------------------------------------------------------------------------


@websocket_api.websocket_command(
    {
        vol.Required("type"): "roommind/rooms/save",
        vol.Required("area_id"): str,
        vol.Optional("temperature_sensor"): str,
        vol.Optional("humidity_sensor"): str,
        vol.Optional("occupancy_sensors"): [str],
        vol.Optional("occupancy_mode"): vol.In(OCCUPANCY_MODES),
        vol.Optional("occupancy_dwell_minutes"): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=15.0)),
        # Legacy bool alias; the store keeps it in sync with occupancy_mode
        vol.Optional("occupancy_eco_enabled"): bool,
        vol.Optional("occupancy_eco_delay"): vol.All(
            vol.Coerce(int), vol.Range(min=MIN_OCCUPANCY_ECO_DELAY_MINUTES, max=240)
        ),
        vol.Optional("preheat_enabled"): bool,
        vol.Optional("price_shift"): vol.All(vol.Coerce(float), vol.Range(min=0, max=MAX_PRICE_SHIFT_K)),
        vol.Optional("climate_mode"): vol.In(CLIMATE_MODES),
        vol.Optional("schedules"): [{vol.Required("entity_id"): str}],
        vol.Optional("schedule_selector_entity"): str,
        # An Exterior Opening pauses the whole plant; window_sensors is the legacy alias
        vol.Optional("exterior_window_sensors"): [str],
        vol.Optional("window_sensors"): [str],
        vol.Optional("window_open_delay"): vol.All(vol.Coerce(int), vol.Range(min=0, max=3600)),
        vol.Optional("window_close_delay"): vol.All(vol.Coerce(int), vol.Range(min=0, max=3600)),
        # An Interior Door cuts the room off from the House Reference Temperature
        vol.Optional("interior_door_sensors"): [str],
        vol.Optional("interior_door_delay"): vol.All(vol.Coerce(int), vol.Range(min=0, max=600)),
        vol.Optional("door_closed_mix_factor"): vol.All(vol.Coerce(float), vol.Range(min=0.1, max=1.0)),
        vol.Optional("comfort_temp"): _TARGET_TEMP,
        vol.Optional("eco_temp"): _TARGET_TEMP,
        vol.Optional("comfort_heat"): _TARGET_TEMP,
        vol.Optional("comfort_cool"): _TARGET_TEMP,
        vol.Optional("eco_heat"): _TARGET_TEMP,
        vol.Optional("eco_cool"): _TARGET_TEMP,
        vol.Optional("presence_persons"): [str],
        vol.Optional("display_name"): str,
        vol.Optional("covers"): [str],
        vol.Optional("read_only_covers"): [vol.All(str, _read_only_cover_entity)],
        vol.Optional("covers_auto_enabled"): bool,
        vol.Optional("covers_deploy_threshold"): vol.All(vol.Coerce(float), vol.Range(min=0)),
        vol.Optional("covers_min_position"): vol.All(vol.Coerce(int), vol.Range(min=0, max=99)),
        vol.Optional("covers_outdoor_min_temp"): vol.Any(None, vol.All(vol.Coerce(float), vol.Range(min=0, max=35))),
        vol.Optional("covers_override_minutes"): vol.All(vol.Coerce(int), vol.Range(min=0, max=480)),
        vol.Optional("cover_schedules"): [
            {
                vol.Required("entity_id"): str,
                vol.Optional("mode", default="force"): vol.In(["force", "gate"]),
            }
        ],
        vol.Optional("cover_schedule_selector_entity"): str,
        vol.Optional("cover_orientations"): {str: vol.All(vol.Coerce(int), vol.Range(min=0, max=359))},
        vol.Optional("covers_night_close"): bool,
        vol.Optional("covers_night_close_elevation"): vol.All(vol.Coerce(float), vol.Range(min=-18, max=10)),
        vol.Optional("covers_night_close_offset_minutes"): vol.All(vol.Coerce(int), vol.Range(min=-120, max=120)),
        vol.Optional("covers_night_position"): vol.All(vol.Coerce(int), vol.Range(min=0, max=100)),
        vol.Optional("covers_snap_deploy"): bool,
        vol.Optional("covers_window_hold"): bool,
        vol.Optional("cover_min_positions"): {str: vol.All(vol.Coerce(int), vol.Range(min=0, max=99))},
        vol.Optional("ignore_presence"): bool,
        vol.Optional("is_outdoor"): bool,
        vol.Optional("climate_control_enabled"): bool,
        vol.Optional("co2_sensor"): str,
        vol.Optional("co2_threshold"): vol.All(
            vol.Coerce(int), vol.Range(min=MIN_CO2_THRESHOLD, max=MAX_CO2_THRESHOLD)
        ),
        vol.Optional("voc_sensor"): str,
        # In the sensor's own unit (index, ppb, µg/m³); None = the reading is shown, not judged
        vol.Optional("voc_threshold"): vol.Any(None, vol.All(vol.Coerce(float), vol.Range(min=0.001, max=1e6))),
        vol.Optional("ventilation_entity"): vol.Any(
            "", vol.All(str, vol.Match(rf"^({'|'.join(VENTILATION_DOMAINS)})\.[a-z0-9_]+$"))
        ),
        vol.Optional("follow_house"): bool,
    }
)
@websocket_api.async_response
async def websocket_save_room(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: dict,
) -> None:
    """Create or update a room configuration."""
    store = hass.data[DOMAIN]["store"]
    area_id = msg["area_id"]

    # Build config dict from optional fields present in the message
    config: dict = {}
    for key in _ROOM_SAVE_FIELDS:
        if key in msg:
            config[key] = msg[key]

    # Non-admins (dashboard card) may change targets, mode and control of an existing room only
    user = connection.user
    if (user is None or not user.is_admin) and (store.get_room(area_id) is None or set(config) - _USER_ROOM_FIELDS):
        connection.send_error(msg["id"], websocket_api.ERR_UNAUTHORIZED, "Only administrators can change this")
        return

    # Reject RoomMind's own entities to prevent self-assignment (#86)
    own_prefix = f"{DOMAIN}_"
    err = _validate_no_own_entities(config, own_prefix)
    if err:
        connection.send_error(msg["id"], "invalid_entity", err)
        return

    room = await store.async_save_room(area_id, config)

    # Notify coordinator to create/update sensor entities for the room
    coordinator = _get_coordinator(hass)
    if coordinator:
        await coordinator.async_room_added(room)

    connection.send_result(msg["id"], {"room": room})


# ---------------------------------------------------------------------------
# Delete room
# ---------------------------------------------------------------------------


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "roommind/rooms/delete",
        vol.Required("area_id"): str,
    }
)
@websocket_api.async_response
async def websocket_delete_room(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: dict,
) -> None:
    """Delete a room."""
    area_id = msg["area_id"]
    if not await async_remove_room(hass, area_id):
        connection.send_error(msg["id"], "not_found", f"Room '{area_id}' not found")
        return
    connection.send_result(msg["id"], {"success": True})


async def _async_drop_boost_anchors(store: RoomMindStore, area_ids: list[str] | None = None) -> None:
    """Forget the boost anchor with the model it counted on (None = every room)."""
    anchors = store.get_settings().get("boost_applied_at", {})
    if not anchors:
        return
    kept = {} if area_ids is None else {a: n for a, n in anchors.items() if a not in area_ids}
    if kept != anchors:
        await store.async_save_settings({"boost_applied_at": kept})


async def _async_forget_in_room_lists(store: RoomMindStore, area_id: str) -> None:
    """Drop a deleted room from the settings lists, else a new room in the same area inherits them."""
    settings = store.get_settings()
    changes = {
        key: [a for a in settings[key] if a != area_id]
        for key in ("hidden_rooms", "room_order", "learning_disabled_rooms")
        if area_id in (settings.get(key) or [])
    }
    if changes:
        await store.async_save_settings(changes)


async def async_remove_room(hass: HomeAssistant, area_id: str) -> bool:
    """Delete a room with its entities, model and history; False if it is not stored."""
    store = hass.data[DOMAIN]["store"]
    room = store.get_room(area_id)
    if room is None:
        return False
    coordinator = _get_coordinator(hass)
    if coordinator:
        coordinator.note_room_deleted(area_id, room)
    await store.async_delete_room(area_id)

    await _async_forget_in_room_lists(store, area_id)

    if coordinator:
        await coordinator.async_room_removed(area_id)
    # Stored right away, after the in-memory model is gone: else it only went with the next thermal save
    await store.async_clear_thermal_data_room(area_id)
    await _async_drop_boost_anchors(store, [area_id])
    return True


async def async_pause_room(hass: HomeAssistant, area_id: str) -> None:
    """Stop running a room whose area was deleted, keeping what is stored for it (#494)."""
    coordinator = _get_coordinator(hass)
    if hass.data[DOMAIN]["store"].get_room(area_id) is not None and coordinator:
        await coordinator.async_room_paused(area_id)


async def async_resume_room(hass: HomeAssistant, area_id: str) -> None:
    """Run a resting room again once an area with its id exists (#494)."""
    store = hass.data[DOMAIN]["store"]
    if store.get_room(area_id) is None:
        return
    await store.async_update_room(area_id, {REST_SWITCHED_OFF: False})
    coordinator = _get_coordinator(hass)
    if coordinator:
        await coordinator.async_room_added(store.get_room(area_id))


# ---------------------------------------------------------------------------
# Set override
# ---------------------------------------------------------------------------


@websocket_api.websocket_command(
    {
        vol.Required("type"): "roommind/override/set",
        vol.Required("area_id"): str,
        vol.Required("override_type"): vol.In(OVERRIDE_TYPES),
        vol.Optional("heat"): _TARGET_TEMP,
        vol.Optional("cool"): _TARGET_TEMP,
        vol.Optional("duration"): vol.Coerce(float),  # hours (omit or 0 for permanent)
        # Until the active schedule's next block boundary; `duration` is the fallback without one
        vol.Optional("until_schedule_change", default=False): bool,
    }
)
@websocket_api.async_response
async def websocket_override_set(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: dict,
) -> None:
    """Set a temporary override for a room."""
    store = hass.data[DOMAIN]["store"]
    area_id = msg["area_id"]
    override_type = msg["override_type"]
    duration_hours = msg.get("duration")

    room = store.get_room(area_id)
    if room is None:
        connection.send_error(msg["id"], "not_found", f"Room '{area_id}' not found")
        return

    if override_type == OVERRIDE_CUSTOM:
        heat = msg.get("heat")
        cool = msg.get("cool")
        if heat is None and cool is None:
            connection.send_error(msg["id"], "invalid", "Custom override requires heat and/or cool")
            return
    else:
        heat, cool = override_preset_band(room, override_type)

    heat, cool = mask_override_band(heat, cool, room.get("climate_mode", "auto"))

    if heat is not None and cool is not None and cool < heat:
        connection.send_error(msg["id"], "invalid", "Cooling target must be >= heating target")
        return

    override_until = (time.time() + duration_hours * 3600) if duration_hours else None
    if msg.get("until_schedule_change"):
        next_change = next_schedule_change(hass, room)
        if next_change is not None and next_change > time.time():
            override_until = next_change

    await store.async_update_room(
        area_id,
        {
            "override_heat": heat,
            "override_cool": cool,
            "override_until": override_until,
            "override_type": override_type,
        },
    )

    coordinator = _get_coordinator(hass)
    if coordinator:
        coordinator.clear_override_seed(area_id)
        await coordinator.async_request_refresh()

    connection.send_result(msg["id"], {"success": True})


# ---------------------------------------------------------------------------
# Clear override
# ---------------------------------------------------------------------------


@websocket_api.websocket_command(
    {
        vol.Required("type"): "roommind/override/clear",
        vol.Required("area_id"): str,
    }
)
@websocket_api.async_response
async def websocket_override_clear(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: dict,
) -> None:
    """Clear an active override for a room."""
    store = hass.data[DOMAIN]["store"]
    area_id = msg["area_id"]

    room = store.get_room(area_id)
    if room is None:
        connection.send_error(msg["id"], "not_found", f"Room '{area_id}' not found")
        return

    await store.async_update_room(
        area_id,
        {
            "override_heat": None,
            "override_cool": None,
            "override_until": None,
            "override_type": None,
        },
    )

    coordinator = _get_coordinator(hass)
    if coordinator:
        coordinator.clear_override_seed(area_id)
        await coordinator.async_request_refresh()

    connection.send_result(msg["id"], {"success": True})


# ---------------------------------------------------------------------------
# Get settings
# ---------------------------------------------------------------------------


@websocket_api.websocket_command({vol.Required("type"): "roommind/settings/get"})
@websocket_api.async_response
async def websocket_get_settings(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: dict,
) -> None:
    """Return global settings, with the Central Plant block fully resolved."""
    store = hass.data[DOMAIN]["store"]
    settings = store.get_settings()
    settings["central_plant"] = get_central_plant_settings(settings)
    settings["house"] = get_house_settings(settings)
    connection.send_result(msg["id"], {"settings": settings})


# ---------------------------------------------------------------------------
# Save settings
# ---------------------------------------------------------------------------


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "roommind/settings/save",
        vol.Optional("outdoor_temp_sensor"): str,
        vol.Optional("price_entity"): str,
        vol.Optional("outdoor_humidity_sensor"): str,
        vol.Optional("outdoor_cooling_min"): vol.Coerce(float),
        vol.Optional("outdoor_heating_max"): vol.Coerce(float),
        vol.Optional("control_mode"): vol.In([CONTROL_MODE_MPC, CONTROL_MODE_BANGBANG]),
        vol.Optional("comfort_weight"): vol.Coerce(float),
        vol.Optional("weather_entity"): str,
        vol.Optional("outdoor_unavailable_notify"): bool,
        vol.Optional("climate_control_active"): bool,
        vol.Optional("learning_disabled_rooms"): [str],
        vol.Optional("hidden_rooms"): [str],
        vol.Optional("prediction_enabled"): bool,
        vol.Optional("vacation_temp"): vol.Coerce(float),
        vol.Optional("vacation_until"): vol.Any(vol.Coerce(float), None),
        vol.Optional("presence_enabled"): bool,
        vol.Optional("presence_persons"): [str],
        vol.Optional("presence_away_action"): vol.In(["eco", "off"]),
        vol.Optional("presence_clears_override"): bool,
        vol.Optional("schedule_off_action"): vol.In(["eco", "off"]),
        vol.Optional("mold_detection_enabled"): bool,
        vol.Optional("mold_humidity_threshold"): vol.All(vol.Coerce(float), vol.Range(min=50, max=90)),
        vol.Optional("mold_f_rsi"): vol.All(vol.Coerce(float), vol.Range(min=0.6, max=0.95)),
        vol.Optional("mold_sustained_minutes"): vol.All(vol.Coerce(int), vol.Range(min=5, max=120)),
        vol.Optional("mold_notification_cooldown"): vol.All(vol.Coerce(int), vol.Range(min=10, max=1440)),
        vol.Optional("mold_notifications_enabled"): bool,
        vol.Optional("mold_notification_targets"): [
            {
                vol.Required("entity_id"): str,
                vol.Optional("person_entity", default=""): str,
                vol.Optional("notify_when", default="always"): vol.In(["always", "home_only"]),
            }
        ],
        vol.Optional("mold_prevention_enabled"): bool,
        vol.Optional("mold_prevention_intensity"): vol.In(["light", "medium", "strong"]),
        vol.Optional("mold_prevention_notify_enabled"): bool,
        vol.Optional("mold_prevention_notify_targets"): [
            {
                vol.Required("entity_id"): str,
                vol.Optional("person_entity", default=""): str,
                vol.Optional("notify_when", default="always"): vol.In(["always", "home_only"]),
            }
        ],
        vol.Optional("room_order"): [str],
        vol.Optional("group_by_floor"): bool,
        vol.Optional("central_plant"): _CENTRAL_PLANT_SCHEMA,
        vol.Optional("house"): _HOUSE_SCHEMA,
    }
)
@websocket_api.async_response
async def websocket_save_settings(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: dict,
) -> None:
    """Save global settings (partial merge)."""
    store = hass.data[DOMAIN]["store"]
    changes: dict = {}
    for key in _SETTINGS_SAVE_FIELDS:
        if key in msg:
            changes[key] = msg[key]

    # The plant's fallback band has to stay ordered: the thermostat parks on it
    # whenever RoomMind is not driving, so an inverted band would fight itself.
    plant_changes = changes.get("central_plant")
    if isinstance(plant_changes, dict) and not _validate_central_plant_band(store.get_settings(), plant_changes):
        connection.send_error(
            msg["id"],
            "invalid_central_plant",
            "fallback_cool_c must be greater than fallback_heat_c",
        )
        return

    house_changes = changes.get("house")
    if isinstance(house_changes, dict) and not _validate_house_band(store.get_settings(), house_changes):
        connection.send_error(
            msg["id"],
            "invalid_house_band",
            "cool_c must be greater than heat_c and eco_cool_c must be >= eco_heat_c",
        )
        return

    settings = await store.async_save_settings(changes)
    connection.send_result(msg["id"], {"settings": settings})
    coordinator = _get_coordinator(hass)
    if coordinator and ("climate_control_active" in changes or "central_plant" in changes or "house" in changes):
        # Stop/resume control now (#291) and pick up a new Central Thermostat at once.
        await coordinator.async_request_refresh()


def _validate_central_plant_band(settings: dict, changes: dict) -> bool:
    """True when the merged fallback band keeps cooling above heating."""
    resolved: dict = {**get_central_plant_settings(settings), **changes}
    heat, cool = resolved.get("fallback_heat_c"), resolved.get("fallback_cool_c")
    if not isinstance(heat, (int, float)) or not isinstance(cool, (int, float)):
        return True
    return float(cool) > float(heat)


# ---------------------------------------------------------------------------
# Clear a Central Thermostat override
# ---------------------------------------------------------------------------


@websocket_api.websocket_command({vol.Required("type"): "roommind/plant/clear_override"})
@websocket_api.async_response
async def websocket_plant_clear_override(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: dict,
) -> None:
    """Drop a manual override on the Central Thermostat and hand the plant back to RoomMind."""
    await _async_clear_plant_override(hass)
    connection.send_result(msg["id"], {"cleared": True})


async def _async_clear_plant_override(hass: HomeAssistant) -> None:
    """Clear the stored plant override and tell the controller, if it exposes the hook."""
    store: RoomMindStore = hass.data[DOMAIN]["store"]
    await store.async_save_settings(
        {
            "plant_override_heat": None,
            "plant_override_cool": None,
            "plant_override_until": None,
        }
    )
    coordinator = _get_coordinator(hass)
    if coordinator is None:
        return
    clear = getattr(coordinator, "async_clear_plant_override", None)
    result = clear() if callable(clear) else None
    if inspect.isawaitable(result):
        await result
    else:
        # Older coordinator: the settings are already saved, just pick them up now.
        await coordinator.async_request_refresh()


# ---------------------------------------------------------------------------
# Whole-House settings commands
# ---------------------------------------------------------------------------


@websocket_api.websocket_command(
    {
        vol.Required("type"): "roommind/house/save",
        **_HOUSE_SCHEMA.schema,
    }
)
@websocket_api.async_response
async def websocket_house_save(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: dict,
) -> None:
    """Save a partial change to the whole-House settings block."""
    store: RoomMindStore = hass.data[DOMAIN]["store"]
    # The schema's keys are voluptuous ``Marker`` objects whose string form is the
    # field name; convert, or the persisted dict would carry Marker keys that the
    # HA store cannot serialise.
    changes: dict = {str(key): msg[key] for key in _HOUSE_SCHEMA.schema if key in msg}

    # Non-admins may write targets/overrides only; structure stays admin-only.
    user = connection.user
    if user is None or not user.is_admin:
        disallowed = set(changes) - _USER_HOUSE_FIELDS
        if disallowed:
            connection.send_error(msg["id"], websocket_api.ERR_UNAUTHORIZED, "Only administrators can change this")
            return

    if not _validate_house_band(store.get_settings(), changes):
        connection.send_error(
            msg["id"],
            "invalid_house_band",
            "cool_c must be greater than heat_c and eco_cool_c must be >= eco_heat_c",
        )
        return

    # A first target write activates the layer unless enabled was sent explicitly.
    if ("heat_c" in changes or "cool_c" in changes) and "enabled" not in changes:
        changes["enabled"] = True

    await store.async_save_settings({"house": changes})
    house = get_house_settings(store.get_settings())
    coordinator = _get_coordinator(hass)
    if coordinator:
        await coordinator.async_request_refresh()
    connection.send_result(msg["id"], {"house": house})


@websocket_api.websocket_command({vol.Required("type"): "roommind/house/clear_override"})
@websocket_api.async_response
async def websocket_house_clear_override(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: dict,
) -> None:
    """Clear the whole-House manual override; any authenticated user may do this."""
    store: RoomMindStore = hass.data[DOMAIN]["store"]
    await store.async_save_settings(
        {
            "house": {
                "override_heat": None,
                "override_cool": None,
                "override_until": None,
                "override_type": None,
            }
        }
    )
    coordinator = _get_coordinator(hass)
    if coordinator:
        await coordinator.async_request_refresh()
    connection.send_result(msg["id"], {"cleared": True})


# ---------------------------------------------------------------------------
# Target temperature forecast (for analytics chart)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Get analytics data
# ---------------------------------------------------------------------------


@websocket_api.websocket_command(
    {
        vol.Required("type"): "roommind/analytics/get",
        vol.Required("area_id"): str,
        vol.Optional("range"): vol.In(["12h", "24h", "2d", "7d", "14d", "30d", "90d"]),
        vol.Optional("start_ts"): vol.Coerce(float),
        vol.Optional("end_ts"): vol.Coerce(float),
    }
)
@websocket_api.async_response
async def websocket_get_analytics(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: dict,
) -> None:
    """Return analytics data for a room."""
    store = hass.data[DOMAIN]["store"]
    if store.get_room(msg["area_id"]) is None:
        connection.send_error(msg["id"], "not_found", f"Room '{msg['area_id']}' not found")
        return
    coordinator = _get_coordinator(hass)
    result = await build_analytics_data(
        hass,
        msg["area_id"],
        msg.get("range", "12h"),
        store,
        coordinator,
        custom_start=msg.get("start_ts"),
        custom_end=msg.get("end_ts"),
    )
    connection.send_result(msg["id"], result)


# ---------------------------------------------------------------------------
# Reset thermal model (per room)
# ---------------------------------------------------------------------------


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "roommind/thermal/reset",
        vol.Required("area_id"): str,
    }
)
@websocket_api.async_response
async def websocket_thermal_reset(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: dict,
) -> None:
    """Reset thermal model and history for a single room."""
    store = hass.data[DOMAIN]["store"]
    area_id = msg["area_id"]
    if store.get_room(area_id) is None:
        connection.send_error(msg["id"], "not_found", f"Room '{area_id}' not found")
        return
    coordinator = _get_coordinator(hass)

    # Clear learned model and residual heat tracking
    if coordinator:
        coordinator.reset_thermal_room(area_id)

    # Clear persisted thermal data
    await store.async_clear_thermal_data_room(area_id)
    await _async_drop_boost_anchors(store, [area_id])

    # Clear history CSV files
    if coordinator and coordinator.history_store:
        await hass.async_add_executor_job(coordinator.history_store.remove_room, area_id)

    connection.send_result(msg["id"], {"success": True})


# ---------------------------------------------------------------------------
# Reset thermal model (all rooms)
# ---------------------------------------------------------------------------


@websocket_api.require_admin
@websocket_api.websocket_command({vol.Required("type"): "roommind/thermal/reset_all"})
@websocket_api.async_response
async def websocket_thermal_reset_all(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: dict,
) -> None:
    """Reset thermal model and history for all rooms."""
    store = hass.data[DOMAIN]["store"]
    coordinator = _get_coordinator(hass)

    # Clear all learned models — replace entire manager for clean state
    room_ids: list[str] = []
    if coordinator:
        room_ids = coordinator.reset_thermal_all()

    # Clear persisted thermal data
    await store.async_clear_all_thermal_data()
    await _async_drop_boost_anchors(store)

    # Clear history CSV files for all rooms, also those without a model (outdoor areas)
    if coordinator and coordinator.history_store:
        for area_id in sorted(set(room_ids) | set(store.get_rooms())):
            await hass.async_add_executor_job(coordinator.history_store.remove_room, area_id)

    connection.send_result(msg["id"], {"success": True})


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "roommind/model/boost_learning",
        vol.Required("area_id"): str,
    }
)
@websocket_api.async_response
async def websocket_boost_learning(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: dict,
) -> None:
    """Boost EKF covariance for a room to accelerate re-learning."""
    store = hass.data[DOMAIN]["store"]
    coordinator = _get_coordinator(hass)
    area_id = msg["area_id"]

    if not coordinator:
        connection.send_error(msg["id"], "no_coordinator", "Coordinator not ready")
        return

    n_obs = coordinator.boost_learning(area_id)
    if n_obs is None:
        connection.send_error(msg["id"], "no_model", "Room has no learned model yet")
        return

    # The frontend counts the fade from this anchor
    settings = store.get_settings()
    boost_applied = dict(settings.get("boost_applied_at", {}))
    boost_applied[area_id] = n_obs
    await store.async_save_settings({"boost_applied_at": boost_applied})

    connection.send_result(msg["id"], {"success": True, "n_observations": n_obs})


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@websocket_api.require_admin
@websocket_api.websocket_command({vol.Required("type"): "roommind/diagnostics/get"})
@websocket_api.async_response
async def websocket_get_diagnostics(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict,
) -> None:
    """Return full integration diagnostics via WebSocket."""
    from .diagnostics import async_get_config_entry_diagnostics

    entries = hass.config_entries.async_entries(DOMAIN)
    if not entries:
        connection.send_error(msg["id"], "not_found", "No config entry found")
        return
    result = await async_get_config_entry_diagnostics(hass, entries[0])
    connection.send_result(msg["id"], result)


# ---------------------------------------------------------------------------
# Clear cover user override
# ---------------------------------------------------------------------------


@websocket_api.websocket_command(
    {
        vol.Required("type"): "roommind/covers/clear_override",
        vol.Required("area_id"): str,
    }
)
@websocket_api.async_response
async def websocket_covers_clear_override(
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: dict,
) -> None:
    """Clear a user cover override so automatic cover control resumes."""
    store = hass.data[DOMAIN]["store"]
    area_id = msg["area_id"]
    if store.get_room(area_id) is None:
        connection.send_error(msg["id"], "not_found", f"Room '{area_id}' not found")
        return

    coordinator = _get_coordinator(hass)
    if coordinator:
        coordinator.clear_cover_override(area_id)
        await coordinator.async_request_refresh()

    connection.send_result(msg["id"], {"success": True})


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@callback
def async_register_websocket_commands(hass: HomeAssistant) -> None:
    """Register all RoomMind WebSocket commands."""
    websocket_api.async_register_command(hass, websocket_list_rooms)
    websocket_api.async_register_command(hass, websocket_save_room)
    websocket_api.async_register_command(hass, websocket_delete_room)
    websocket_api.async_register_command(hass, websocket_override_set)
    websocket_api.async_register_command(hass, websocket_override_clear)
    websocket_api.async_register_command(hass, websocket_get_settings)
    websocket_api.async_register_command(hass, websocket_save_settings)
    websocket_api.async_register_command(hass, websocket_house_save)
    websocket_api.async_register_command(hass, websocket_house_clear_override)
    websocket_api.async_register_command(hass, websocket_get_analytics)
    websocket_api.async_register_command(hass, websocket_thermal_reset)
    websocket_api.async_register_command(hass, websocket_thermal_reset_all)
    websocket_api.async_register_command(hass, websocket_boost_learning)
    websocket_api.async_register_command(hass, websocket_get_diagnostics)
    websocket_api.async_register_command(hass, websocket_covers_clear_override)
    websocket_api.async_register_command(hass, websocket_plant_clear_override)
