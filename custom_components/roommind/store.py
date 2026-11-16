"""Room persistence layer for RoomMind."""

from __future__ import annotations

import copy
import logging

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import (
    DEFAULT_CO2_THRESHOLD,
    DEFAULT_COMFORT_COOL,
    DEFAULT_COMFORT_HEAT,
    DEFAULT_COVERS_OUTDOOR_MIN_TEMP,
    DEFAULT_DOOR_CLOSED_MIX_FACTOR,
    DEFAULT_ECO_COOL,
    DEFAULT_ECO_HEAT,
    DEFAULT_INTERIOR_DOOR_DELAY_S,
    DEFAULT_OCCUPANCY_DWELL_MINUTES,
    DEFAULT_OCCUPANCY_ECO_DELAY_MINUTES,
    DOMAIN,
    OCCUPANCY_MODE_FOLLOW_SCHEDULE,
    OCCUPANCY_MODE_REACTIVE_TREND,
    OCCUPANCY_MODE_SCHEDULE_ASSUMED,
    get_central_plant_settings,
)

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 2
STORAGE_KEY = DOMAIN

# v1 global settings the Central HVAC schema retired: they moved into the
# ``central_plant`` block and are dropped from the persisted payload.
_RETIRED_SETTINGS_KEYS = (
    "heating_threshold",
    "cooling_threshold",
    "compressor_groups",
    "valve_protection_enabled",
    "valve_protection_interval_days",
    "valve_last_actuation",
    "coil_dry_enabled",
    "coil_dry_minutes",
    "coil_dry_mode",
    "coil_dry_fan_mode",
    "coil_dry_min_cooling_minutes",
    "coil_dry_drain_minutes",
    "coil_dry_state",
    "ac_head_offsets",
)

# Per-room v1 device configuration the Central Plant owns now: the plant is one
# shared forced-air system, so a room neither lists nor controls its own devices.
# ``devices``/``acs``/``thermostats`` are read once by ``_promote_shared_thermostat``
# (to adopt a single shared ``climate.*`` entity as the Central Thermostat) and
# then dropped here together with their hydronic and heat-source companions.
RETIRED_ROOM_KEYS = (
    "devices",
    "thermostats",
    "acs",
    "heating_system_type",
    "follow_device_setpoint",
    "valve_protection_exclude",
    "heat_source_orchestration",
    "heat_source_primary_delta",
    "heat_source_outdoor_threshold",
    "heat_source_ac_min_outdoor",
    "heat_source_price_aware",
)


def _migrate_room_temps(room: dict) -> dict:
    """Migrate legacy single comfort/eco temps to split heat/cool fields.

    In a cool_only room the single value is the cooling target.
    """
    cool_only = room.get("climate_mode") == "cool_only"
    if "comfort_heat" not in room:
        room["comfort_heat"] = room.get("comfort_temp", DEFAULT_COMFORT_HEAT)
    if "comfort_cool" not in room:
        room["comfort_cool"] = room.get("comfort_temp", DEFAULT_COMFORT_COOL) if cool_only else DEFAULT_COMFORT_COOL
    if "eco_heat" not in room:
        room["eco_heat"] = room.get("eco_temp", DEFAULT_ECO_HEAT)
    if "eco_cool" not in room:
        room["eco_cool"] = room.get("eco_temp", DEFAULT_ECO_COOL) if cool_only else DEFAULT_ECO_COOL
    return room


def _single_temps_to_cool_side(target: dict, config: dict, climate_mode: str) -> None:
    """A legacy/API caller that sends only comfort_temp/eco_temp to a cool_only room means the cooling target."""
    if climate_mode != "cool_only":
        return
    if "comfort_temp" in config and "comfort_cool" not in config:
        target["comfort_cool"] = config["comfort_temp"]
    if "eco_temp" in config and "eco_cool" not in config:
        target["eco_cool"] = config["eco_temp"]


def _migrate_override_fields(room: dict) -> dict:
    """Migrate legacy single override_temp to split override_heat/override_cool."""
    if "override_temp" not in room:
        return room
    legacy = room.pop("override_temp")
    if legacy is None or room.get("override_heat") is not None or room.get("override_cool") is not None:
        return room
    climate_mode = room.get("climate_mode", "auto")
    legacy = float(legacy)
    if climate_mode == "cool_only":
        room["override_heat"] = None
        room["override_cool"] = legacy
    elif climate_mode == "heat_only":
        room["override_heat"] = legacy
        room["override_cool"] = None
    else:
        room["override_heat"] = legacy
        room["override_cool"] = max(legacy, room.get("comfort_cool", DEFAULT_COMFORT_COOL))
    return room


def _sync_window_sensors(room: dict, changes: dict) -> None:
    """Mirror the canonical ``exterior_window_sensors`` field onto legacy ``window_sensors``.

    ``exterior_window_sensors`` wins when a caller sends both; a caller still
    using the legacy name gets the value promoted to the canonical field.
    """
    if "exterior_window_sensors" in changes:
        sensors = list(changes.get("exterior_window_sensors") or [])
    elif "window_sensors" in changes:
        sensors = list(changes.get("window_sensors") or [])
    else:
        return
    room["exterior_window_sensors"] = list(sensors)
    room["window_sensors"] = list(sensors)


def _migrate_window_sensors(room: dict) -> None:
    """Fill ``exterior_window_sensors`` from the legacy field and mirror it back for readers."""
    sensors = room.get("exterior_window_sensors")
    if sensors is None:
        sensors = room.get("window_sensors")
    sensors = list(sensors or [])
    room["exterior_window_sensors"] = list(sensors)
    room["window_sensors"] = list(sensors)


def _derive_occupancy_mode(room: dict) -> str:
    """Derive the occupancy mode from the legacy occupancy fields."""
    if not room.get("occupancy_sensors"):
        return OCCUPANCY_MODE_FOLLOW_SCHEDULE
    if room.get("occupancy_eco_enabled", False):
        return OCCUPANCY_MODE_REACTIVE_TREND
    return OCCUPANCY_MODE_SCHEDULE_ASSUMED


def _apply_occupancy_mode(room: dict) -> None:
    """Add ``occupancy_mode`` and keep the legacy ``occupancy_eco_enabled`` flag in sync."""
    if "occupancy_mode" not in room:
        room["occupancy_mode"] = _derive_occupancy_mode(room)
        # Legacy readers still branch on occupancy_eco_enabled until Wave 3.
        room["occupancy_eco_enabled"] = room["occupancy_mode"] == OCCUPANCY_MODE_REACTIVE_TREND
    else:
        room.setdefault("occupancy_eco_enabled", room["occupancy_mode"] == OCCUPANCY_MODE_REACTIVE_TREND)


def _apply_room_defaults(room: dict) -> None:
    """Add the Central HVAC room fields with their defaults when missing."""
    room.setdefault("interior_door_sensors", [])
    room.setdefault("interior_door_delay", DEFAULT_INTERIOR_DOOR_DELAY_S)
    room.setdefault("door_closed_mix_factor", DEFAULT_DOOR_CLOSED_MIX_FACTOR)
    room.setdefault("occupancy_dwell_minutes", DEFAULT_OCCUPANCY_DWELL_MINUTES)
    room.setdefault("read_only_covers", [])


def strip_retired_room_keys(room: dict) -> dict:
    """Drop the per-room device configuration the Central Plant owns now."""
    for key in RETIRED_ROOM_KEYS:
        room.pop(key, None)
    return room


def _migrate_room(room: dict) -> dict:
    """Apply all per-room migrations and defaults (idempotent).

    The v1 per-room device fields are inspected once by
    ``_promote_shared_thermostat`` and then dropped: the Central HVAC schema has
    the plant own the hardware, so nothing reads or writes them any more.
    """
    _migrate_room_temps(room)
    _migrate_override_fields(room)
    _migrate_window_sensors(room)
    _apply_room_defaults(room)
    _apply_occupancy_mode(room)
    strip_retired_room_keys(room)
    return room


def _room_climate_entities(room: dict) -> list[str]:
    """Every ``climate.*`` entity id a v1 room owns, from devices[] and the legacy lists."""
    entities: list[str] = []
    for key in ("devices", "acs", "thermostats"):
        entries = room.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            entity_id = entry.get("entity_id") if isinstance(entry, dict) else entry
            if isinstance(entity_id, str) and entity_id.startswith("climate."):
                entities.append(entity_id)
    return entities


def _promote_shared_thermostat(rooms: dict, settings: dict) -> None:
    """Promote a single shared climate entity into the central plant settings.

    A v1 install whose rooms together reference exactly one distinct
    ``climate.*`` entity is what the Central HVAC schema calls a Central
    Thermostat; with two or more the plant entity stays unset so the user picks.
    Must run before the retired per-room keys are stripped.
    """
    central = settings.get("central_plant")
    if not isinstance(central, dict):
        central = {}
    if central.get("thermostat_entity", ""):
        return
    owners: dict[str, set[str]] = {}
    for area_id, room in rooms.items():
        if not isinstance(room, dict):
            continue
        for entity_id in _room_climate_entities(room):
            owners.setdefault(entity_id, set()).add(area_id)
    if len(owners) != 1:
        return
    entity_id, area_ids = next(iter(owners.items()))
    central["thermostat_entity"] = entity_id
    if len(area_ids) == 1:
        area_id = next(iter(area_ids))
        room = rooms.get(area_id)
        central["thermostat_area_id"] = room.get("area_id", area_id) if isinstance(room, dict) else area_id
    settings["central_plant"] = central


def migrate_store_data(raw: dict) -> dict:
    """Migrate stored data to the current schema (v1 -> v2). Idempotent.

    Returns a fresh dict and never mutates ``raw``, so the caller can compare
    both to decide whether a migration save is needed.
    """
    data = copy.deepcopy(raw) if isinstance(raw, dict) else {}
    raw_rooms = data.get("rooms")
    rooms: dict[str, dict] = raw_rooms if isinstance(raw_rooms, dict) else {}
    raw_settings = data.get("settings")
    settings: dict = raw_settings if isinstance(raw_settings, dict) else {}

    _promote_shared_thermostat(rooms, settings)
    for key in _RETIRED_SETTINGS_KEYS:
        settings.pop(key, None)
    for room in rooms.values():
        if isinstance(room, dict):
            _migrate_room(room)

    settings["central_plant"] = get_central_plant_settings(settings)
    data["rooms"] = rooms
    data["settings"] = settings
    if not isinstance(data.get("thermal_data"), dict):
        data["thermal_data"] = {}
    if not isinstance(data.get("occupancy_trends"), dict):
        data["occupancy_trends"] = {}
    return data


class _RoomMindHaStore(Store[dict]):
    """Home Assistant store that migrates older RoomMind storage versions."""

    async def _async_migrate_func(self, old_major_version: int, old_minor_version: int, old_data: dict) -> dict:
        """Migrate stored data from an older schema version."""
        _LOGGER.info(
            "Migrating RoomMind storage from version %s.%s to %s",
            old_major_version,
            old_minor_version,
            STORAGE_VERSION,
        )
        return migrate_store_data(old_data)


class RoomMindStore:
    """Manage room configuration storage for RoomMind."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialise the store."""
        self._store: Store[dict] = _RoomMindHaStore(hass, STORAGE_VERSION, STORAGE_KEY)
        self._data: dict[str, dict] = {}
        self._settings: dict = {}
        self._thermal_data: dict = {}
        self._occupancy_trends: dict = {}

    async def async_load(self) -> None:
        """Load room data from the HA store, migrating older payloads.

        HA's ``Store`` already runs ``_async_migrate_func`` on a version bump;
        running ``migrate_store_data`` here as well makes a mocked
        ``_store.async_load`` (tests) and a real store behave identically. The
        migration is idempotent, so this is a no-op for current data.
        """
        stored = await self._store.async_load()
        if not stored:
            self._data = {}
            self._settings = {}
            self._thermal_data = {}
            self._occupancy_trends = {}
            return

        migrated = migrate_store_data(stored)
        self._data = migrated["rooms"]
        self._settings = migrated["settings"]
        self._thermal_data = migrated["thermal_data"]
        self._occupancy_trends = migrated["occupancy_trends"]
        if migrated != stored:
            _LOGGER.info("Migrated stored RoomMind data to the current schema")
            await self._async_save()

    async def _async_save(self) -> None:
        """Persist current room data to the HA store."""
        await self._store.async_save(
            {
                "rooms": self._data,
                "settings": self._settings,
                "thermal_data": self._thermal_data,
                # getattr: instances built via __new__ in older tests may predate this field
                "occupancy_trends": getattr(self, "_occupancy_trends", {}),
            }
        )

    def get_rooms(self) -> dict[str, dict]:
        """Return a deep copy of all rooms (with read-time migrations applied)."""
        rooms = copy.deepcopy(dict(self._data))
        for room in rooms.values():
            _migrate_room(room)
        return rooms

    def get_room(self, area_id: str) -> dict | None:
        """Return a deep copy of a single room by area ID, or None if not found."""
        room = self._data.get(area_id)
        if room is None:
            return None
        result = copy.deepcopy(room)
        _migrate_room(result)
        return result

    def get_settings(self) -> dict:
        """Return a deep copy of global settings."""
        return copy.deepcopy(dict(self._settings))

    async def async_save_settings(self, changes: dict) -> dict:
        """Merge changes into global settings and persist.

        A ``central_plant`` change is merged onto the resolved central plant
        settings so a partial update cannot wipe the keys it did not mention.
        """
        updates = dict(changes)
        central = updates.get("central_plant")
        if isinstance(central, dict):
            updates["central_plant"] = {**get_central_plant_settings(self._settings), **central}
        self._settings.update(updates)
        await self._async_save()
        return dict(self._settings)

    def get_thermal_data(self) -> dict:
        """Return a deep copy of thermal learning data."""
        return copy.deepcopy(dict(self._thermal_data))

    async def async_save_thermal_data(self, data: dict) -> None:
        """Replace thermal learning data and persist."""
        self._thermal_data = data
        await self._async_save()

    async def async_clear_thermal_data_room(self, area_id: str) -> None:
        """Clear thermal learning data for a single room."""
        self._thermal_data.pop(area_id, None)
        await self._async_save()

    async def async_clear_all_thermal_data(self) -> None:
        """Clear all thermal learning data."""
        self._thermal_data = {}
        await self._async_save()

    def get_occupancy_trends(self) -> dict:
        """Return a deep copy of the learned occupancy trends."""
        return copy.deepcopy(dict(self._occupancy_trends))

    async def async_save_occupancy_trends(self, data: dict) -> None:
        """Replace the occupancy trend data and persist."""
        self._occupancy_trends = data
        await self._async_save()

    async def async_clear_occupancy_trends_room(self, area_id: str) -> None:
        """Clear the occupancy trend of a single room."""
        self._occupancy_trends.pop(area_id, None)
        await self._async_save()

    async def async_clear_all_occupancy_trends(self) -> None:
        """Clear every room's occupancy trend."""
        self._occupancy_trends = {}
        await self._async_save()

    def _merge_room(self, area_id: str, config: dict) -> dict:
        """Merge config changes into an existing room."""
        existing = self._data[area_id]
        for key, value in config.items():
            if key == "area_id" or key in RETIRED_ROOM_KEYS:
                continue
            existing[key] = value
        strip_retired_room_keys(existing)
        # Sync legacy fields from split fields for backward compat
        if "comfort_heat" in config:
            existing["comfort_temp"] = config["comfort_heat"]
        if "eco_heat" in config:
            existing["eco_temp"] = config["eco_heat"]
        # Reverse-sync: legacy callers sending only comfort_temp/eco_temp
        if "comfort_temp" in config and "comfort_heat" not in config:
            existing["comfort_heat"] = config["comfort_temp"]
        if "eco_temp" in config and "eco_heat" not in config:
            existing["eco_heat"] = config["eco_temp"]
        _single_temps_to_cool_side(existing, config, existing.get("climate_mode", "auto"))
        # Window sensors: canonical field + legacy mirror stay identical
        _sync_window_sensors(existing, config)
        if "occupancy_mode" in config:
            existing["occupancy_eco_enabled"] = existing["occupancy_mode"] == OCCUPANCY_MODE_REACTIVE_TREND
        return existing

    def _create_room(self, area_id: str, config: dict) -> dict:
        """Create a new room with defaults."""
        # Derive split fields from legacy if needed
        comfort_heat = config.get("comfort_heat", config.get("comfort_temp", DEFAULT_COMFORT_HEAT))
        eco_heat = config.get("eco_heat", config.get("eco_temp", DEFAULT_ECO_HEAT))
        occupancy_mode = config.get("occupancy_mode") or _derive_occupancy_mode(config)
        room = {
            "area_id": area_id,
            "temperature_sensor": config.get("temperature_sensor", ""),
            "humidity_sensor": config.get("humidity_sensor", ""),
            "occupancy_sensors": config.get("occupancy_sensors", []),
            "occupancy_mode": occupancy_mode,
            "occupancy_eco_enabled": config.get(
                "occupancy_eco_enabled", occupancy_mode == OCCUPANCY_MODE_REACTIVE_TREND
            ),
            "occupancy_eco_delay": config.get("occupancy_eco_delay", DEFAULT_OCCUPANCY_ECO_DELAY_MINUTES),
            "occupancy_dwell_minutes": config.get("occupancy_dwell_minutes", DEFAULT_OCCUPANCY_DWELL_MINUTES),
            "preheat_enabled": config.get("preheat_enabled", True),
            "price_shift": config.get("price_shift", 0.0),
            "climate_mode": config.get("climate_mode", "auto"),
            "schedules": config.get("schedules", []),
            "schedule_selector_entity": config.get("schedule_selector_entity", ""),
            "window_sensors": config.get("window_sensors", []),
            "exterior_window_sensors": config.get("exterior_window_sensors", config.get("window_sensors", [])),
            "window_open_delay": config.get("window_open_delay", 0),
            "window_close_delay": config.get("window_close_delay", 0),
            "interior_door_sensors": config.get("interior_door_sensors", []),
            "interior_door_delay": config.get("interior_door_delay", DEFAULT_INTERIOR_DOOR_DELAY_S),
            "door_closed_mix_factor": config.get("door_closed_mix_factor", DEFAULT_DOOR_CLOSED_MIX_FACTOR),
            "comfort_temp": comfort_heat,
            "eco_temp": eco_heat,
            "comfort_heat": comfort_heat,
            "comfort_cool": config.get("comfort_cool", DEFAULT_COMFORT_COOL),
            "eco_heat": eco_heat,
            "eco_cool": config.get("eco_cool", DEFAULT_ECO_COOL),
            "presence_persons": config.get("presence_persons", []),
            "display_name": config.get("display_name", ""),
            "override_heat": config.get("override_heat"),
            "override_cool": config.get("override_cool"),
            "override_until": config.get("override_until"),
            "override_type": config.get("override_type"),
            "covers": config.get("covers", []),
            "covers_auto_enabled": config.get("covers_auto_enabled", False),
            "covers_deploy_threshold": config.get("covers_deploy_threshold", 1.5),
            "covers_min_position": config.get("covers_min_position", 0),
            "covers_outdoor_min_temp": config.get("covers_outdoor_min_temp", DEFAULT_COVERS_OUTDOOR_MIN_TEMP),
            "covers_override_minutes": config.get("covers_override_minutes", 60),
            "cover_schedules": config.get("cover_schedules", []),
            "cover_schedule_selector_entity": config.get("cover_schedule_selector_entity", ""),
            "cover_orientations": config.get("cover_orientations", {}),
            "covers_night_close": config.get("covers_night_close", False),
            "covers_night_close_elevation": config.get("covers_night_close_elevation", 0),
            "covers_night_close_offset_minutes": config.get("covers_night_close_offset_minutes", 0),
            "covers_night_position": config.get("covers_night_position", 0),
            "covers_snap_deploy": config.get("covers_snap_deploy", False),
            "covers_window_hold": config.get("covers_window_hold", False),
            "cover_min_positions": config.get("cover_min_positions", {}),
            "read_only_covers": config.get("read_only_covers", []),
            "ignore_presence": config.get("ignore_presence", False),
            "is_outdoor": config.get("is_outdoor", False),
            "climate_control_enabled": config.get("climate_control_enabled", True),
            "co2_sensor": config.get("co2_sensor", ""),
            "co2_threshold": config.get("co2_threshold", DEFAULT_CO2_THRESHOLD),
            "voc_sensor": config.get("voc_sensor", ""),
            "voc_threshold": config.get("voc_threshold"),
            "ventilation_entity": config.get("ventilation_entity", ""),
        }
        _single_temps_to_cool_side(room, config, room["climate_mode"])
        _sync_window_sensors(room, config)
        return room

    async def async_save_room(self, area_id: str, config: dict) -> dict:
        """Create or update room configuration for an area."""
        if area_id in self._data:
            room = self._merge_room(area_id, config)
        else:
            room = self._create_room(area_id, config)
            self._data[area_id] = room
        await self._async_save()
        return room

    async def async_update_room(self, area_id: str, changes: dict) -> dict:
        """Merge changes into an existing room. Raises KeyError if not found.

        Retired per-room device fields in *changes* are ignored: the Central
        Plant owns the hardware, so a legacy caller cannot put them back.
        """
        if area_id not in self._data:
            raise KeyError(f"Room '{area_id}' not found")

        for retired in RETIRED_ROOM_KEYS:
            changes.pop(retired, None)
        # Prevent overriding the area_id
        changes.pop("area_id", None)

        room = self._data[area_id]
        room.update(changes)
        _sync_window_sensors(room, changes)
        if "occupancy_mode" in changes:
            room["occupancy_eco_enabled"] = room["occupancy_mode"] == OCCUPANCY_MODE_REACTIVE_TREND
        await self._async_save()
        return room

    async def async_delete_room(self, area_id: str) -> None:
        """Delete a room and its learned data. Raises KeyError if not found."""
        if area_id not in self._data:
            raise KeyError(f"Room '{area_id}' not found")

        del self._data[area_id]
        self._occupancy_trends.pop(area_id, None)
        self._thermal_data.pop(area_id, None)
        await self._async_save()
