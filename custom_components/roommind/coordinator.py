"""DataUpdateCoordinator for RoomMind's Central Plant.

RoomMind drives one shared forced-air system through a single Central Thermostat.
Every control cycle reads every Thermal Zone's sensors, targets, openings,
occupancy and learned behaviour, hands the whole set to the joint whole-home MPC
(:mod:`~.control.central_mpc`), and applies the one resulting plant action to the
plant while the per-room loops keep covers, ventilation and learning running.

The loop is deliberately a pipeline:

1. read the cycle inputs (outdoor, weather, price, the Central Thermostat),
2. collect every room's state (sensors, openings, occupancy, targets, covers),
3. compute the House Reference Temperature,
4. evaluate the plant guards and the wet-coil lockout,
5. run the joint MPC for the whole home,
6. actuate the Central Thermostat, the covers and the ventilation,
7. train the room models, write the history and persist the learned state.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial
from typing import Any

from homeassistant.components.persistent_notification import async_create as async_create_notification
from homeassistant.components.persistent_notification import async_dismiss as async_dismiss_notification
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CoreState, HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .const import (
    CLIMATE_MODE_COOL_ONLY,
    CLIMATE_MODE_HEAT_ONLY,
    DEFAULT_CO2_THRESHOLD,
    DEFAULT_COMFORT_COOL,
    DEFAULT_COMFORT_HEAT,
    DEFAULT_ECO_COOL,
    DEFAULT_ECO_HEAT,
    DOMAIN,
    GLOBAL_ENTITY_UNIQUE_IDS,
    HISTORY_ROTATE_CYCLES,
    HISTORY_WRITE_CYCLES,
    MAX_SENSOR_STALENESS,
    MODE_CIRCULATING,
    MODE_COOLING,
    MODE_HEATING,
    MODE_IDLE,
    OUTDOOR_UNAVAILABLE_NOTIFICATION_ID,
    OUTDOOR_UNAVAILABLE_NOTIFY_CYCLES,
    OVERRIDE_TURN_ON_REFRESH_DELAY_S,
    TARGET_REASON_AWAY,
    TARGET_REASON_HOUSE,
    TARGET_REASON_OVERRIDE,
    TARGET_REASON_PLANT_OVERRIDE,
    TARGET_REASON_VACANT,
    TARGET_REASON_VACATION,
    THERMAL_SAVE_CYCLES,
    UPDATE_INTERVAL,
    TargetTemps,
    build_override_live,
    get_central_plant_settings,
    get_house_settings,
    is_house_active,
    is_override_active,
    is_override_suppressed,
    is_plant_override_active,
)
from .control.central_mpc import (
    DEFAULT_GUARDRAIL_COOL,
    DEFAULT_GUARDRAIL_HEAT,
    MIN_HORIZON_BLOCKS,
    PLAN_DT_MINUTES,
    CentralMPCController,
    CentralMPCPlan,
    RoomHorizonInput,
    is_room_mpc_ready,
)
from .control.rebound import ForcedAirReboundTracker
from .control.solar import (
    build_oriented_solar_series,
    build_solar_series,
    compute_q_solar_norm,
    plane_of_array_irradiance,
)
from .control.thermal_model import RoomModelManager, regressor_signature
from .control.thermostat_actuator import ActuationResult, ThermostatActuator
from .control.topology import RoomGraph, build_room_graph, reference_temperature, reference_weights
from .managers.air_quality_manager import AirQualityTracker, VentilationController
from .managers.central_plant_manager import CentralPlantManager, ExteriorGateResult, PlantGuardState
from .managers.cover_manager import CoverManager
from .managers.cover_orchestrator import CoverOrchestrator, CoverResult, HouseCoverContext
from .managers.ekf_training_manager import EkfTrainingManager
from .managers.interior_door_manager import InteriorDoorManager, InteriorDoorState
from .managers.mold_manager import MoldManager, MoldResult
from .managers.occupancy_manager import OccupancyManager, RoomOccupancyState
from .managers.price_manager import PriceManager
from .managers.recent_temps import RECENT_WINDOW_S, RecentTemps
from .managers.thermostat_override_tracker import ThermostatOverrideTracker
from .managers.weather_manager import WeatherManager
from .managers.wet_coil_manager import WetCoilTracker
from .managers.window_manager import WindowManager
from .utils.area_utils import rooms_with_area
from .utils.history_store import HistoryStore
from .utils.humidity_utils import effective_temperature_c
from .utils.presence_utils import is_presence_away
from .utils.price_utils import PriceSignal
from .utils.room_utils import room_solar_azimuths
from .utils.schedule_utils import (
    HouseContext,
    TargetSource,
    cap_cool_only_heat,
    find_active_block,
    find_rejected_block_temps,
    get_active_schedule_entity,
    house_override_active_at,
    house_target_source,
    make_house_target_resolver,
    make_target_resolver,
    next_schedule_change,
    price_shifted,
    read_schedule_blocks,
    resolve_schedule_index,
    resolve_targets_with_source,
    schedule_block_sets_target,
)
from .utils.sensor_utils import read_sensor_value
from .utils.temp_utils import celsius_delta_to_ha, ha_temp_to_celsius, ha_temp_unit_str

_LOGGER = logging.getLogger(__name__)


def _get_area_name(hass: HomeAssistant, area_id: str) -> str:
    """Get human-readable area name from area registry."""
    try:
        area_reg = ar.async_get(hass)
        area = area_reg.async_get_area(area_id)
        return area.name if area else area_id
    except Exception:  # noqa: BLE001
        return area_id


# Per-room entity unique_id suffixes (uid = ``{DOMAIN}_{area_id}{suffix}``).
# All suffixes end differently, so a given uid maps to exactly one
# (area_id, suffix) pair — enabling unambiguous exact matching.
ROOM_ENTITY_SUFFIXES = (
    "_climate",
    "_target_temp",
    "_mode",
    "_override",
    "_climate_control",
    "_cover_auto",
    "_cover_paused",
    "_air_quality",
)
# Suffixes only valid when the room has covers configured.
COVER_ENTITY_SUFFIXES = ("_cover_auto", "_cover_paused")
# Suffixes only valid when the room has a CO2 or VOC sensor (#267).
AIR_QUALITY_ENTITY_SUFFIXES = ("_air_quality",)

_UNAVAILABLE_STATES = ("unavailable", "unknown")


def has_air_quality_sensor(room: dict) -> bool:
    """Whether a room judges its air (CO2 or VOC sensor, not outdoor) and gets the air quality entity."""
    return bool(room.get("co2_sensor") or room.get("voc_sensor")) and not room.get("is_outdoor", False)


def _match_room_entity(parts: str, rooms: dict) -> tuple[str, str] | None:
    """Match ``parts`` (a uid without the ``{DOMAIN}_`` prefix) to an owning room.

    Returns ``(area_id, suffix)`` when ``parts`` equals ``{area_id}{suffix}`` for
    an existing room and a known per-room suffix, else ``None``.

    Uses exact matching rather than prefix ``startswith`` so a shorter area_id
    (e.g. ``bedroom``) never falsely claims a longer room's entity
    (``bedroom_2_l_override``). (#340)
    """
    for area_id in rooms:
        for suffix in ROOM_ENTITY_SUFFIXES:
            if parts == f"{area_id}{suffix}":
                return area_id, suffix
    return None


_NO_AIR_QUALITY: dict[str, Any] = {
    "co2": None,
    "voc": None,
    "air_quality_bad": False,
    "air_quality_reason": None,
    "ventilation_active": False,
}

_EMPTY_MOLD = MoldResult()


def _numeric(value: Any) -> float | None:
    """Return *value* as a float, or None when it is not a usable number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _as_int(value: Any, default: int = 0) -> int:
    """Return *value* as an int, falling back for missing/corrupt settings."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class _RoomCycle:
    """Everything one room contributes to one control cycle."""

    area_id: str
    room: dict
    is_outdoor: bool = False
    current_temp: float | None = None
    current_temp_raw: float | None = None
    current_humidity: float | None = None
    has_external_sensor: bool = False
    air: dict[str, Any] = field(default_factory=lambda: dict(_NO_AIR_QUALITY))
    mold: MoldResult = field(default_factory=lambda: _EMPTY_MOLD)
    window_open: bool = False
    raw_open: bool = False
    window_pending: bool = False
    door: InteriorDoorState | None = None
    schedule_blocks: dict | None = None
    schedule_entity_id: str | None = None
    schedule_state: str | None = None
    schedule_warnings: list[dict[str, Any]] = field(default_factory=list)
    active_schedule_index: int = -1
    schedule_active_now: bool = False
    occupancy: RoomOccupancyState = field(default_factory=RoomOccupancyState)
    targets: TargetTemps = field(default_factory=TargetTemps)
    target_reason: str = ""
    price_shift: tuple[float, float] = (0.0, 0.0)
    mold_boost: float = 0.0
    effective_temp: float | None = None
    shading_factor: float = 1.0
    cover_result: CoverResult | None = None
    force_off: bool = False
    presence_away: bool = False
    vacation_active: bool = False
    house_following: bool = False

    @property
    def controllable(self) -> bool:
        """Whether the plant may be driven for this room."""
        return not self.is_outdoor and bool(self.room.get("climate_control_enabled", True))


class RoomMindCoordinator(DataUpdateCoordinator):
    """Central coordinator for the shared Central Plant and every Thermal Zone."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=UPDATE_INTERVAL),
        )
        self.entry = entry
        self.rooms: dict[str, dict] = {}
        self.central_plant: dict[str, Any] = {}
        self.outdoor_temp: float | None = None
        self.outdoor_temp_effective: float | None = None
        self.outdoor_temp_source: str = "none"
        self.outdoor_humidity: float | None = None
        self._outdoor_unavailable_cycles: int = 0
        self._outdoor_warning_sent: bool = False

        # Sub-managers, one per responsibility of the Central HVAC schema.
        self._model_manager = RoomModelManager()
        self._ekf_training = EkfTrainingManager(self._model_manager)
        self._rebound_tracker = ForcedAirReboundTracker()
        self._occupancy_manager = OccupancyManager()
        self._window_manager = WindowManager()
        self._door_manager = InteriorDoorManager()
        self._wet_coil = WetCoilTracker()
        self._plant_manager = CentralPlantManager()
        self._actuator = ThermostatActuator()
        self._thermostat_overrides = ThermostatOverrideTracker()
        self._mpc_controller = CentralMPCController()
        self._mold_manager = MoldManager(hass)
        self._cover_orchestrator = CoverOrchestrator(hass, CoverManager(), self._model_manager)
        self._weather_manager = WeatherManager(hass)
        self._price_manager = PriceManager(hass)
        self._air_quality = AirQualityTracker()
        self._ventilation = VentilationController(hass)
        self._recent_temps = RecentTemps()

        # Cycle bookkeeping and persistence throttles.
        self._model_loaded = False
        self._thermal_save_count: int = 0
        self._history_store: HistoryStore | None = None
        self._history_write_count: int = 0
        self._history_rotate_count: int = 0
        self._current_q_solar: float = 0.0
        # Per-room oriented solar gain (normalised W/m2 / 1000), keyed by area_id.
        # Rooms without exterior-glazing azimuths keep ``self._current_q_solar``.
        self._current_q_solar_by_room: dict[str, float] = {}
        # Inputs the model learned from this cycle, recorded for the analytics backtest.
        self._model_inputs: dict[str, dict[str, float | None]] = {}
        # Live per-room display/mode history for diagnostics and the old REST surface.
        self._previous_modes: dict[str, str] = {}
        self._mode_on_since: dict[str, float] = {}
        # Sensor dropout fallback: last valid temperature per room.
        self._last_valid_temps: dict[str, tuple[float, float]] = {}
        # Startup guard: rooms with at least one valid reading since coordinator start.
        self._had_valid_temp: set[str] = set()
        self._startup_ts: float = time.monotonic()
        self._startup_guard_warned: set[str] = set()
        # Out-of-range schedule block temps already warned about (#395).
        self._block_temp_warned: set[tuple[str, str, str]] = set()
        # monotonic time at which the override entity seeded an override per room (#447)
        self._override_seeded_at: dict[str, float] = {}
        # Per-entity cache of schedule blocks; fallback when schedule.get_schedule fails (#308)
        self._schedule_blocks_cache: dict[str, dict] = {}
        # House-layer schedule blocks, cached per entity like the room schedules.
        self._house_blocks_cache: dict[str, dict] = {}
        # Sides of the last run, so an auto room's cards keep showing the side it ran.
        self._recent_temps_side: dict[str, str] = {}
        # Track which rooms already have entity platform entities registered.
        self._entity_areas: set[str] = set()
        self._switch_entity_areas: set[str] = set()
        self._climate_control_switch_areas: set[str] = set()
        self._binary_sensor_entity_areas: set[str] = set()
        self._air_quality_entity_areas: set[str] = set()
        self._climate_entity_areas: set[str] = set()
        # Entity platform callbacks, set by the platform async_setup_entry functions.
        self.async_add_entities: Any = None
        self.async_add_switch_entities: Any = None
        self.async_add_climate_entities: Any = None
        self.async_add_binary_sensor_entities: Any = None

    # ------------------------------------------------------------------ #
    # The control loop
    # ------------------------------------------------------------------ #

    async def _async_update_data(self) -> dict:
        """Read every room, run the joint plant plan, actuate and learn."""
        store = self.hass.data[DOMAIN]["store"]
        stored_rooms = store.get_rooms()
        rooms = rooms_with_area(self.hass, stored_rooms)
        settings = store.get_settings()
        plant_settings = get_central_plant_settings(settings)
        now = time.time()
        dt_minutes = UPDATE_INTERVAL / 60.0

        # -- 2. Outdoor conditions, forecast, solar and price -----------------
        self.outdoor_temp = self._read_outdoor_temp(settings)
        self.outdoor_humidity = read_sensor_value(
            self.hass, settings.get("outdoor_humidity_sensor"), "global", "outdoor humidity"
        )
        self.outdoor_temp_effective, self.outdoor_temp_source = self._resolve_outdoor_temp(settings)
        self._update_outdoor_unavailable_notification(settings)
        outdoor_forecast = await self._weather_manager.async_read_forecast(settings)
        cloud_series = WeatherManager.extract_cloud_series(outdoor_forecast)
        self._cover_orchestrator.set_cloud_series(cloud_series)
        # One cloud-coverage read per cycle, shared by the house-wide and the
        # per-room solar estimates so every room uses the same weather snapshot.
        cloud_coverage = self._read_cloud_coverage(settings)
        self._current_q_solar = self._read_current_q_solar(settings, cloud_coverage, cloud_coverage_known=True)
        await self._price_manager.async_update(settings, self._async_load_price_history)
        self._load_learned_state(store, settings)

        # -- 3. The Central Thermostat ----------------------------------------
        thermostat_entity = str(plant_settings.get("thermostat_entity", ""))
        thermostat_state = self.hass.states.get(thermostat_entity) if thermostat_entity else None
        thermostat_temp_c = self._thermostat_temperature(thermostat_state, thermostat_entity)

        # -- 4. A manual change at the wall thermostat is a plant override -----
        settings, plant_settings = await self._async_refresh_plant_override(
            store,
            settings,
            plant_settings,
            thermostat_entity,
            thermostat_state,
            now,
        )

        # -- 5. House layer, then per-room input collection -------------------
        house, house_ctx, house_blocks = await self._async_house_state(settings)
        house_active = house_ctx is not None
        cycles: list[_RoomCycle] = []
        for area_id, room in rooms.items():
            # Live, not this cycle's snapshot: an area or room deleted while the cycle awaited.
            if not rooms_with_area(self.hass, {area_id: room}) or store.get_room(area_id) is None:
                continue
            cycle = await self._async_collect_room(
                area_id,
                room,
                settings,
                plant_settings,
                thermostat_entity,
                thermostat_temp_c,
                now,
                house,
                house_ctx,
            )
            cycles.append(cycle)

        # -- 6. House Reference Temperature -----------------------------------
        house_ref_temp = self._house_ref_temp(cycles, thermostat_temp_c, plant_settings)
        # Per-room solar and topology inputs are built once per cycle from the
        # same weather snapshot and room set, then shared by the MPC and training.
        self._refresh_room_q_solar(cycles, cloud_coverage)
        graph = self._build_room_graph(cycles)
        included = self._house_ref_included_area_ids(cycles)
        temperatures = {cycle.area_id: cycle.current_temp for cycle in cycles if cycle.current_temp is not None}

        # -- 7. Plant guards, wet-coil lockout and the exterior-opening gate ---
        observed_action, fan_circulating = self._plant_manager.observe_plant_action(
            thermostat_state,
            plant_settings,
            self._plant_manager.active_action,
        )
        wet_coil_locked = self._step_wet_coil(cycles, plant_settings, observed_action, now)
        any_override = self._any_override(cycles, settings)
        all_force_off = self._all_force_off(cycles, settings)
        climate_mode = self._house_climate_mode(cycles, house, house_active)
        startup_waiting = self._startup_waiting(cycles, thermostat_entity)
        exterior_gate = self._plant_manager.evaluate_exterior_gate(
            rooms,
            {cycle.area_id: cycle.window_open for cycle in cycles if not cycle.is_outdoor},
            {cycle.area_id: bool(cycle.door and cycle.door.is_open) for cycle in cycles if not cycle.is_outdoor},
            {cycle.area_id for cycle in cycles if self._door_manager.has_sensors(cycle.room)},
            plant_settings,
            any_override_active=any_override,
        )
        guard_state = self._plant_manager.allowed_actions(
            plant_settings,
            climate_mode,
            self.outdoor_temp_effective,
            settings,
            exterior_paused=exterior_gate.paused,
            wet_coil_locked=wet_coil_locked,
            force_off=all_force_off,
            any_override=any_override,
            now_ts=now,
        )

        # -- 8. Joint whole-home MPC ------------------------------------------
        plan = self._evaluate_central_mpc(
            cycles,
            settings,
            plant_settings,
            guard_state,
            climate_mode,
            outdoor_forecast,
            cloud_series,
            now,
            wet_coil_locked,
            house,
            house_ctx,
            house_active,
            graph=graph,
            included=included,
        )
        commanded_action = plan.current_action()
        if not thermostat_entity or not settings.get("climate_control_active", True) or startup_waiting:
            commanded_action = MODE_IDLE

        # -- 9. Actuate the plant, the covers and the ventilation -------------
        display_action, fan_circulating, actuation = await self._async_actuate_plant(
            plant_settings,
            settings,
            cycles,
            thermostat_entity,
            thermostat_state,
            house_ref_temp,
            commanded_action,
            climate_mode,
            all_force_off,
            startup_waiting,
        )
        if not thermostat_entity or startup_waiting:
            display_action = MODE_IDLE
            fan_circulating = False

        cool_sensitive = self._any_cool_sensitive(cycles)
        house_context = HouseCoverContext(
            house_ref_temp=house_ref_temp,
            plant_demand=display_action,
            any_occupied_cool_sensitive=cool_sensitive,
        )
        for cycle in cycles:
            if cycle.is_outdoor:
                continue
            cycle.cover_result = await self._async_process_covers(cycle, house_context)

        # -- 10. Train the room models and persist what was learned -----------
        ekf_mode = observed_action if thermostat_entity else MODE_IDLE
        room_states = self._build_room_states(cycles, settings, plan, display_action, startup_waiting, fan_circulating)
        self._train_and_record(
            cycles,
            room_states,
            settings,
            house_ref_temp,
            ekf_mode,
            dt_minutes,
            graph=graph,
            temperatures=temperatures,
            included=included,
        )
        await self._async_reconcile_ventilation(settings, room_states)
        await self._async_record_recent_temps(room_states)
        await self._async_write_history(cycles, plan, room_states, settings, now)

        # -- 11. Publish the cycle --------------------------------------------
        self.central_plant = self._plant_live_dict(
            plant=plant_settings,
            action=commanded_action,
            display_action=display_action,
            climate_mode=climate_mode,
            guard_state=guard_state,
            exterior_gate=exterior_gate,
            wet_coil_locked=wet_coil_locked,
            house_ref_temp=house_ref_temp,
            thermostat_entity=thermostat_entity,
            thermostat_temp_c=thermostat_temp_c,
            plan=plan,
            startup_waiting=startup_waiting,
            all_force_off=all_force_off,
            any_override=any_override,
            fan_circulating=fan_circulating,
            actuation=actuation,
            observed_hvac_action=self._observed_hvac_action(thermostat_state),
            plant_override=settings,
        )
        await self._async_persist(store, room_states)
        self.rooms = room_states
        house_live = self._house_live_dict(house, house_ctx, house_blocks, cycles, now)
        return {"rooms": room_states, "central_plant": self.central_plant, "house": house_live}

    # ------------------------------------------------------------------ #
    # House layer
    # ------------------------------------------------------------------ #

    async def _async_house_state(self, settings: dict) -> tuple[dict, HouseContext | None, dict | None]:
        """Resolve the house settings, its live context and its schedule blocks.

        Called once per cycle after the settings are final. ``house_ctx`` is
        None unless the house layer is active, and ``house_blocks`` is the same
        schedule data the house target resolver uses (cached per entity).
        """
        house = get_house_settings(settings)
        if not is_house_active({"house": house}):
            return house, None, None
        house_entity = get_active_schedule_entity(self.hass, house)
        blocks = (
            await read_schedule_blocks(self.hass, house_entity, cache=self._house_blocks_cache)
            if house_entity
            else None
        )
        hold_active_at = lambda ts: house_override_active_at(ts, house)  # noqa: E731
        ctx = HouseContext(
            active=True,
            heat_c=house.get("heat_c"),
            cool_c=house.get("cool_c"),
            eco_heat_c=house.get("eco_heat_c"),
            eco_cool_c=house.get("eco_cool_c"),
            target_at=make_house_target_resolver(blocks, house, self.hass),
            price_shift=float(house.get("price_shift") or 0.0),
            hold_active_at=hold_active_at,
        )
        return house, ctx, blocks

    @staticmethod
    def _room_follows_house(room: dict, house: dict) -> bool:
        """Whether a room follows the active house layer."""
        del house  # the decision currently lives on the room; the house is passed for symmetry.
        return bool(room.get("follow_house", True))

    def _effective_climate_mode(self, room: dict, house: dict, house_active: bool) -> str:
        """The climate mode that governs a room's targets and MPC massking.

        A room following an active house uses the house mode, so a heat_only /
        cool_only house masks the series the same way for every follower.
        """
        if house_active and self._room_follows_house(room, house):
            return str(house.get("climate_mode", "auto"))
        return str(room.get("climate_mode", "auto"))

    @staticmethod
    def _active_block_window(blocks: dict | None, ts: float) -> dict[str, Any] | None:
        """The active schedule block's ``from``/``to`` window, for the live payload."""
        if not blocks:
            return None
        from datetime import UTC, datetime

        local = dt_util.as_local(datetime.fromtimestamp(ts, tz=UTC))
        day_blocks = blocks.get(local.strftime("%A").lower(), []) or []
        now_t = local.time()
        for block in day_blocks:
            from_raw = block.get("from", "00:00:00")
            to_raw = block.get("to", "00:00:00")
            from_t = from_raw if hasattr(from_raw, "hour") else datetime.strptime(str(from_raw), "%H:%M:%S").time()
            to_t = to_raw if hasattr(to_raw, "hour") else datetime.strptime(str(to_raw), "%H:%M:%S").time()
            if from_t <= now_t < to_t:
                return {"from": block.get("from"), "to": block.get("to")}
        return None

    def _house_live_dict(
        self,
        house: dict,
        house_ctx: HouseContext | None,
        house_blocks: dict | None,
        cycles: list[_RoomCycle],
        now: float,
    ) -> dict[str, Any]:
        """The live house-layer payload read by the card, panel and entities.

        Always present (also before the first plan). It carries every resolved
        house setting plus the derived live fields the frontend consumes.
        """
        active = house_ctx is not None
        converter = partial(ha_temp_to_celsius, self.hass)
        band_targets = (
            house_ctx.target_at(now)
            if house_ctx is not None
            else TargetTemps(heat=house.get("heat_c"), cool=house.get("cool_c"))
        )
        schedule_entity = get_active_schedule_entity(self.hass, house) if active else None
        state = self.hass.states.get(schedule_entity) if schedule_entity else None
        schedule_name = str(state.attributes.get("friendly_name", "")) if state is not None else ""
        schedule_block = self._active_block_window(house_blocks, now) if active else None
        live: dict[str, Any] = dict(house)
        live.update(
            {
                "active": active,
                "band": {"heat_c": band_targets.heat, "cool_c": band_targets.cool},
                "source": house_target_source(now, house_blocks, house, converter) if active else None,
                "override_active": house_override_active_at(now, house) if active else False,
                "schedule_entity": schedule_entity or "",
                "schedule_name": schedule_name,
                "schedule_block": schedule_block,
                "next_change": next_schedule_change(self.hass, house) if active else None,
                "following_rooms": sum(1 for cycle in cycles if cycle.controllable and cycle.house_following),
            }
        )
        return live

    # ------------------------------------------------------------------ #
    # Cycle inputs
    # ------------------------------------------------------------------ #

    def _load_learned_state(self, store: Any, settings: dict) -> None:
        """Restore every learned sub-manager state once (first cycle after a start)."""
        if self._model_loaded:
            return
        thermal_data = store.get_thermal_data()
        if thermal_data:
            self._model_manager = RoomModelManager.from_dict(thermal_data)
            self._ekf_training.set_model_manager(self._model_manager)
            self._cover_orchestrator.set_model_manager(self._model_manager)
        self._occupancy_manager.load_data(store.get_occupancy_trends())
        self._plant_manager.load_state(settings.get("central_plant_state", {}))
        self._wet_coil.load_state(settings.get("wet_coil_state", {}))
        self._ventilation.load_state(settings.get("ventilation_state", {}))
        if self._history_store is None:
            self._history_store = HistoryStore(self.hass.config.path(".storage/roommind_history"))
        self._model_loaded = True

    def _read_outdoor_temp(self, settings: dict) -> float | None:
        """Read the configured outdoor temperature sensor in Celsius."""
        entity_id = settings.get("outdoor_temp_sensor")
        raw = read_sensor_value(self.hass, entity_id, "global", "outdoor temperature")
        if raw is None:
            return None
        return ha_temp_to_celsius(self.hass, raw, entity_id=entity_id)

    def _resolve_outdoor_temp(self, settings: dict) -> tuple[float | None, str]:
        """Return (temp, source): outdoor sensor, then weather entity, else none.

        The EKF must not train with a degenerate fallback (e.g. room temp); see
        ``EkfTrainingManager.process`` where this gates updates (#301).
        """
        if self.outdoor_temp is not None:
            return self.outdoor_temp, "sensor"

        weather_eid = settings.get("weather_entity") or ""
        if weather_eid:
            state = self.hass.states.get(weather_eid)
            if state is not None and state.state not in _UNAVAILABLE_STATES:
                temp_attr = state.attributes.get("temperature")
                converted = _numeric(temp_attr)
                if converted is not None:
                    return ha_temp_to_celsius(self.hass, converted, entity_id=weather_eid), "weather"

        return None, "none"

    def _read_cloud_coverage(self, settings: dict) -> float | None:
        """Read the weather entity's cloud coverage once, in percent (None when unavailable)."""
        cloud_coverage = None
        weather_entity = settings.get("weather_entity")
        if weather_entity:
            state = self.hass.states.get(weather_entity)
            if state is not None:
                coverage = state.attributes.get("cloud_coverage")
                if isinstance(coverage, int | float) and not isinstance(coverage, bool):
                    cloud_coverage = float(coverage)
        return cloud_coverage

    def _read_current_q_solar(
        self,
        settings: dict,
        cloud_coverage: float | None = None,
        *,
        cloud_coverage_known: bool = False,
    ) -> float:
        """Normalized clear-sky-adjusted solar irradiance of the current cycle.

        The control loop reads the weather entity once and passes the value in
        (``cloud_coverage_known=True``) so the per-room pass can reuse it. Callers
        that invoke this helper directly keep the original behaviour.
        """
        if not cloud_coverage_known:
            cloud_coverage = self._read_cloud_coverage(settings)
        return compute_q_solar_norm(
            self.hass.config.latitude,
            self.hass.config.longitude,
            time.time(),
            cloud_coverage,
        )

    def _refresh_room_q_solar(self, cycles: list[_RoomCycle], cloud_coverage: float | None) -> None:
        """Compute every room's oriented solar gain from one timestamp and one weather read.

        A room with no exterior glazing azimuths keeps ``self._current_q_solar``,
        the unoriented horizontal-plane value every room received before the
        feature existed. Oriented rooms receive the mean plane-of-array
        irradiance over their azimuths, normalised the same way (W/m2 / 1000).
        """
        ts = time.time()
        latitude = self.hass.config.latitude
        longitude = self.hass.config.longitude
        result: dict[str, float] = {}
        for cycle in cycles:
            if cycle.is_outdoor:
                continue
            azimuths = room_solar_azimuths(cycle.room)
            if not azimuths:
                result[cycle.area_id] = self._current_q_solar
                continue
            poa_total = sum(
                plane_of_array_irradiance(
                    latitude,
                    longitude,
                    ts,
                    azimuth,
                    cloud_coverage=cloud_coverage,
                )
                for azimuth in azimuths
            )
            result[cycle.area_id] = poa_total / len(azimuths) / 1000.0
        self._current_q_solar_by_room = result

    def _room_q_solar(self, cycle: _RoomCycle) -> float:
        """This room's solar gain regressor, falling back to the house-wide value."""
        return self._current_q_solar_by_room.get(cycle.area_id, self._current_q_solar)

    def _update_outdoor_unavailable_notification(self, settings: dict) -> None:
        """Raise/clear the single notification when no outdoor temperature arrives."""
        if self.outdoor_temp_effective is not None:
            self._outdoor_unavailable_cycles = 0
            if self._outdoor_warning_sent:
                self._outdoor_warning_sent = False
                async_dismiss_notification(self.hass, OUTDOOR_UNAVAILABLE_NOTIFICATION_ID)
            return

        self._outdoor_unavailable_cycles += 1
        if self._outdoor_warning_sent:
            return
        if self._outdoor_unavailable_cycles < OUTDOOR_UNAVAILABLE_NOTIFY_CYCLES:
            return
        if not settings.get("outdoor_unavailable_notify", True):
            return

        sensor_id = settings.get("outdoor_temp_sensor") or "(not configured)"
        weather_eid = settings.get("weather_entity") or "(not configured)"
        message = (
            "RoomMind gets no outdoor temperature, so learning is paused in every "
            "room until one arrives again.\n\n"
            f"• Outdoor sensor: `{sensor_id}`\n"
            f"• Weather entity: `{weather_eid}`\n\n"
            "Check that the sensor reports values, or set a weather forecast in "
            "RoomMind under Settings › Sensors & data sources."
        )
        _LOGGER.warning(
            "Outdoor temperature unavailable for %d cycles — EKF learning paused",
            self._outdoor_unavailable_cycles,
        )
        async_create_notification(
            self.hass,
            message,
            title="RoomMind: no outdoor temperature",
            notification_id=OUTDOOR_UNAVAILABLE_NOTIFICATION_ID,
        )
        self._outdoor_warning_sent = True

    def _thermostat_temperature(self, thermostat_state: Any, thermostat_entity: str) -> float | None:
        """The Central Thermostat's own temperature reading in Celsius, if it reports one."""
        if thermostat_state is None or not thermostat_entity:
            return None
        raw = thermostat_state.attributes.get("current_temperature")
        value = _numeric(raw)
        if value is None:
            return None
        return ha_temp_to_celsius(self.hass, value, entity_id=thermostat_entity)

    async def _async_refresh_plant_override(
        self,
        store: Any,
        settings: dict,
        plant_settings: dict,
        thermostat_entity: str,
        thermostat_state: Any,
        now: float,
    ) -> tuple[dict, dict]:
        """Persist a manual wall-thermostat change and clear an expired plant override."""
        external = self._thermostat_overrides.check_external_change(
            self.hass,
            thermostat_entity,
            thermostat_state,
            plant_settings,
            None,
            now_ts=now,
        )
        if external:
            await store.async_save_settings(external)
            settings = store.get_settings()
            plant_settings = get_central_plant_settings(settings)
        elif not is_plant_override_active(settings) and settings.get("plant_override_until") is not None:
            await store.async_save_settings(
                {
                    "plant_override_heat": None,
                    "plant_override_cool": None,
                    "plant_override_until": None,
                }
            )
            settings = store.get_settings()
            plant_settings = get_central_plant_settings(settings)
        return settings, plant_settings

    # ------------------------------------------------------------------ #
    # Per-room collection
    # ------------------------------------------------------------------ #

    async def _async_collect_room(
        self,
        area_id: str,
        room: dict,
        settings: dict,
        plant_settings: dict,
        thermostat_entity: str,
        thermostat_temp_c: float | None,
        now: float,
        house: dict,
        house_ctx: HouseContext | None,
    ) -> _RoomCycle:
        """Read one room's sensors, openings, occupancy, targets and covers."""
        room = self._without_pending_override_seed(room)
        if room.get("area_id") != area_id:
            room = {**room, "area_id": area_id}
        is_outdoor = bool(room.get("is_outdoor", False))
        cycle = _RoomCycle(area_id=area_id, room=room, is_outdoor=is_outdoor)

        current_temp, current_temp_raw, current_humidity, has_external_sensor = self._read_room_sensors(
            room,
            area_id,
            plant_settings,
            thermostat_entity,
            thermostat_temp_c,
        )
        cycle.current_temp = current_temp
        cycle.current_temp_raw = current_temp_raw
        cycle.current_humidity = current_humidity
        cycle.has_external_sensor = has_external_sensor

        if is_outdoor:
            cycle.targets = TargetTemps()
            cycle.force_off = True
            cycle.door = self._door_manager.state_for(area_id)
            return cycle

        cycle.presence_away = not room.get("ignore_presence", False) and is_presence_away(self.hass, room, settings)
        vacation_until = settings.get("vacation_until")
        cycle.vacation_active = vacation_until is not None and now < float(vacation_until)
        cycle.air = self._evaluate_air_quality(area_id, room)
        cycle.mold = await self._evaluate_mold_risk(area_id, room, current_temp, current_humidity, settings)
        await self._async_read_schedule(area_id, room, cycle, now)
        cycle.raw_open, cycle.window_open, cycle.window_pending = self._read_window_state(area_id, room)
        cycle.door = self._door_manager.resolve_and_update(self.hass, area_id, room, now)
        cycle.house_following = self._room_follows_house(room, house) and house_ctx is not None
        converter = partial(ha_temp_to_celsius, self.hass)
        cycle.occupancy = self._occupancy_manager.step_room(
            self.hass,
            area_id,
            room,
            now,
            schedule_active_now=cycle.schedule_active_now,
            presence_away=cycle.presence_away,
            vacation_active=cycle.vacation_active,
            override_active=is_override_active(room),
            room_setting_active=schedule_block_sets_target(now, cycle.schedule_blocks, converter),
            window_open=cycle.window_open,
            current_temp_c=current_temp,
        )
        cycle.targets, cycle.target_reason, cycle.mold_boost, cycle.price_shift = self._resolve_targets(
            room,
            settings,
            cycle,
            now,
            house,
            house_ctx,
        )
        cycle.force_off = self._room_force_off(room, cycle.targets)
        cycle.effective_temp = (
            effective_temperature_c(
                current_temp,
                current_humidity,
                enabled=bool(get_central_plant_settings(settings).get("humidity_comfort_enabled", True)),
            )
            if current_temp is not None
            else None
        )
        cover_positions = self._cover_orchestrator.read_positions(area_id, room)
        cycle.shading_factor = cover_positions.shading_factor
        return cycle

    def _read_room_sensors(
        self,
        room: dict,
        area_id: str,
        plant_settings: dict,
        thermostat_entity: str,
        thermostat_temp_c: float | None,
    ) -> tuple[float | None, float | None, float | None, bool]:
        """Read temperature and humidity, with the Central Thermostat as a room-sensor stand-in.

        A Central Thermostat in (or near) a Thermal Zone is that zone's only
        temperature reading in many homes. When the room has no own sensor and
        the thermostat is assigned to it, the thermostat's ``current_temperature``
        is used instead of leaving the room blind.
        """
        temp_sensor_id = room.get("temperature_sensor")
        has_external_sensor = bool(temp_sensor_id)
        raw_temp = read_sensor_value(self.hass, temp_sensor_id, area_id, "temperature")
        current_temp: float | None = None
        if raw_temp is not None:
            current_temp = ha_temp_to_celsius(self.hass, raw_temp, entity_id=temp_sensor_id)
        elif not has_external_sensor and thermostat_temp_c is not None:
            if plant_settings.get("thermostat_area_id") == area_id and thermostat_entity:
                current_temp = thermostat_temp_c

        current_temp_raw = current_temp
        if current_temp is not None:
            self._last_valid_temps[area_id] = (current_temp, time.monotonic())
            self._had_valid_temp.add(area_id)
        elif area_id in self._last_valid_temps:
            cached_temp, cached_ts = self._last_valid_temps[area_id]
            if time.monotonic() - cached_ts < MAX_SENSOR_STALENESS:
                current_temp = cached_temp
                _LOGGER.debug(
                    "Room '%s': sensor unavailable, using cached temp %.1f°C (age %.0fs)",
                    area_id,
                    cached_temp,
                    time.monotonic() - cached_ts,
                )
            else:
                del self._last_valid_temps[area_id]

        current_humidity = read_sensor_value(self.hass, room.get("humidity_sensor"), area_id, "humidity")
        return current_temp, current_temp_raw, current_humidity, has_external_sensor

    async def _async_read_schedule(self, area_id: str, room: dict, cycle: _RoomCycle, now: float) -> None:
        """Fetch and cache the room's active schedule blocks, index and warnings."""
        del area_id
        cycle.schedule_entity_id = get_active_schedule_entity(self.hass, room)
        cycle.active_schedule_index = resolve_schedule_index(self.hass, room)
        if cycle.schedule_entity_id:
            cycle.schedule_blocks = await read_schedule_blocks(
                self.hass,
                cycle.schedule_entity_id,
                cache=self._schedule_blocks_cache,
            )
        state = self.hass.states.get(cycle.schedule_entity_id) if cycle.schedule_entity_id else None
        cycle.schedule_state = None if state is None else state.state
        cycle.schedule_warnings = find_rejected_block_temps(
            cycle.schedule_blocks,
            partial(ha_temp_to_celsius, self.hass),
        )
        cycle.schedule_active_now = self._schedule_active_at(
            cycle.schedule_blocks,
            cycle.schedule_state,
            now,
        )

    def _schedule_active_at(self, blocks: dict | None, state: str | None, ts: float) -> bool:
        """Whether a schedule block is in force at *ts* for the known block list.

        A room without a schedule follows its comfort band all day, which the
        Occupancy Trend's schedule assumption reads as "on".
        """
        if blocks:
            return find_active_block(blocks, ts) is not None
        if state is None:
            return True
        return state == "on"

    def _read_window_state(self, area_id: str, room: dict) -> tuple[bool, bool, bool]:
        """Return ``(raw_open, window_open, pending)`` for the room's exterior openings."""
        sensors = list(room.get("exterior_window_sensors") or room.get("window_sensors") or [])
        states: dict[str, str | None] = {}
        for entity_id in sensors:
            state = self.hass.states.get(entity_id)
            states[entity_id] = None if state is None else state.state
        raw = self._window_manager.resolve_raw(area_id, states, self._missing_window_sensors(states))
        if raw is None:
            return True, self._window_manager.is_paused(area_id), True
        window_open = self._window_manager.update(
            area_id,
            raw,
            _as_int(room.get("window_open_delay", 0)),
            _as_int(room.get("window_close_delay", 0)),
        )
        return raw, window_open, False

    def _missing_window_sensors(self, states: dict[str, str | None]) -> frozenset[str]:
        """Sensors without a state that HA also does not know in the entity registry.

        Only once HA is fully running: while it is still starting, entities may
        not be loaded yet, so a missing state means "not yet", not "gone".
        """
        stateless = [entity_id for entity_id, state in states.items() if state is None]
        if not stateless or self.hass.state is not CoreState.running:
            return frozenset()
        from homeassistant.helpers import entity_registry as er

        registry = er.async_get(self.hass)
        return frozenset(entity_id for entity_id in stateless if registry.async_get(entity_id) is None)

    def _room_force_off(self, room: dict, targets: TargetTemps) -> bool:
        """Whether the room's own configuration leaves it out of control."""
        if not room.get("climate_control_enabled", True):
            return True
        return targets.heat is None and targets.cool is None

    async def _evaluate_mold_risk(
        self,
        area_id: str,
        room: dict,
        current_temp: float | None,
        current_humidity: float | None,
        settings: dict,
    ) -> MoldResult:
        """Evaluate the room's mold risk and prevention boost."""
        can_heat = room.get("climate_mode", "auto") != CLIMATE_MODE_COOL_ONLY
        return await self._mold_manager.evaluate(
            area_id,
            _get_area_name(self.hass, area_id),
            current_temp,
            current_humidity,
            self.outdoor_temp_effective,
            settings,
            celsius_delta_to_ha_fn=lambda delta: celsius_delta_to_ha(self.hass, delta),  # type: ignore[misc]
            ha_temp_unit_str_fn=lambda: ha_temp_unit_str(self.hass),  # type: ignore[misc]
            can_heat=can_heat,
        )

    def _evaluate_air_quality(self, area_id: str, room: dict) -> dict[str, Any]:
        """CO2/VOC readings and whether the air is bad, with hysteresis (#267)."""
        result = dict(_NO_AIR_QUALITY)
        bad: list[str] = []
        venting = self._ventilation.is_running(room.get("ventilation_entity") or "")
        now = time.time()
        for kind, threshold in (
            ("co2", room.get("co2_threshold") or DEFAULT_CO2_THRESHOLD),
            ("voc", room.get("voc_threshold")),
        ):
            sensor = room.get(f"{kind}_sensor")
            if not sensor:
                continue
            value = read_sensor_value(self.hass, sensor, area_id, kind.upper())
            result[kind] = value
            if self._air_quality.evaluate(area_id, kind, value, threshold, now, assume_bad=venting):
                bad.append(kind)
        result["air_quality_bad"] = bool(bad)
        result["air_quality_reason"] = bad[0] if bad else None
        return result

    @staticmethod
    def _reason_for_source(source: TargetSource, room_vacant: bool) -> str:
        """Live ``target_reason`` string for a resolved target source.

        The public strings stay exactly as before: only the house rung is new.
        Room schedule / room base deliberately report an empty reason.
        """
        if source == TargetSource.OVERRIDE:
            return TARGET_REASON_OVERRIDE
        if source == TargetSource.PLANT_OVERRIDE:
            return TARGET_REASON_PLANT_OVERRIDE
        if source == TargetSource.VACATION:
            return TARGET_REASON_VACATION
        if source == TargetSource.AWAY:
            return TARGET_REASON_AWAY
        if room_vacant:
            return TARGET_REASON_VACANT
        if source == TargetSource.HOUSE:
            return TARGET_REASON_HOUSE
        return ""

    def _resolve_targets(
        self,
        room: dict,
        settings: dict,
        cycle: _RoomCycle,
        now: float,
        house: dict,
        house_ctx: HouseContext | None,
    ) -> tuple[TargetTemps, str, float, tuple[float, float]]:
        """Resolve the room's targets, what set them, its mold boost and price shift."""
        room_vacant = cycle.occupancy.is_vacant
        climate_mode = self._effective_climate_mode(room, house, house_ctx is not None)
        targets, source = resolve_targets_with_source(
            now,
            cycle.schedule_blocks,
            room.get("override_until"),
            room.get("override_heat"),
            room.get("override_cool"),
            settings.get("vacation_until"),
            settings.get("vacation_temp"),
            room.get("comfort_heat", room.get("comfort_temp", DEFAULT_COMFORT_HEAT)),
            room.get("comfort_cool", DEFAULT_COMFORT_COOL),
            room.get("eco_heat", room.get("eco_temp", DEFAULT_ECO_HEAT)),
            room.get("eco_cool", DEFAULT_ECO_COOL),
            presence_away=cycle.presence_away,
            block_temp_converter=partial(ha_temp_to_celsius, self.hass),
            presence_away_action=settings.get("presence_away_action", "eco"),
            schedule_off_action=settings.get("schedule_off_action", "eco"),
            presence_clears_override=bool(settings.get("presence_clears_override", False)),
            room_vacant=room_vacant,
            plant_override_until=settings.get("plant_override_until"),
            plant_override_heat=settings.get("plant_override_heat"),
            plant_override_cool=settings.get("plant_override_cool"),
            house=house_ctx,
            follow_house=cycle.house_following,
        )
        reason = self._reason_for_source(source, room_vacant)
        # ``make_target_resolver`` (MPC) applies the house price shift and the
        # per-timestamp hold suppression itself; mirror that here for the live
        # band so a following room's displayed price shift matches the plan.
        hold_active = (
            bool(house_ctx.hold_active_at(now))
            if house_ctx is not None and house_ctx.hold_active_at is not None
            else False
        )
        shift_override = house_ctx.price_shift if house_ctx is not None and cycle.house_following else None
        targets, shift = price_shifted(
            targets,
            now,
            room,
            settings,
            self._price_manager.signal,
            presence_away=cycle.presence_away,
            room_vacant=room_vacant,
            shift_override=shift_override,
            house_override_active=hold_active,
        )

        mold_boost = 0.0
        if cycle.mold.prevention_active and cycle.mold.prevention_delta > 0:
            mold_boost = float(cycle.mold.prevention_delta)
            if targets.heat is None and targets.cool is None:
                # A moldy wall is structural damage: prevention lifts "off" to eco + boost.
                targets = TargetTemps(
                    heat=room.get("eco_heat", room.get("eco_temp", DEFAULT_ECO_HEAT)) + mold_boost,
                    cool=room.get("eco_cool", DEFAULT_ECO_COOL),
                )
            elif targets.heat is not None:
                targets = TargetTemps(heat=targets.heat + mold_boost, cool=targets.cool)
        targets = TargetTemps(*cap_cool_only_heat(targets.heat, targets.cool, climate_mode))
        if cycle.mold.prevention_active and targets.heat is None and targets.cool is not None:
            # The boost cannot heat a room without a heating side; nothing to lift.
            mold_boost = 0.0
        return targets, reason, mold_boost, shift

    # ------------------------------------------------------------------ #
    # House reference, guards and plan
    # ------------------------------------------------------------------ #

    def _house_ref_temp(
        self,
        cycles: list[_RoomCycle],
        thermostat_temp_c: float | None,
        plant_settings: dict,
    ) -> float | None:
        """Mean room temperature of the conditioned envelope, plus the thermostat's own reading."""
        included = [
            cycle.current_temp
            for cycle in cycles
            if not cycle.is_outdoor and not cycle.window_open and cycle.current_temp is not None
        ]
        own_temp = [float(value) for value in included if value is not None]
        if not own_temp:
            return thermostat_temp_c
        if thermostat_temp_c is None:
            return sum(own_temp) / len(own_temp)
        thermostat_area_id = str(plant_settings.get("thermostat_area_id", ""))
        covered = {cycle.area_id for cycle in cycles if not cycle.is_outdoor and not cycle.window_open}
        if thermostat_area_id and thermostat_area_id in covered:
            return sum(own_temp) / len(own_temp)
        own_temp.append(thermostat_temp_c)
        return sum(own_temp) / len(own_temp)

    @staticmethod
    def _build_room_graph(cycles: list[_RoomCycle]) -> RoomGraph:
        """Adjacency graph of the conditioned rooms; outdoor areas are never neighbours."""
        rooms = {cycle.area_id: cycle.room for cycle in cycles if not cycle.is_outdoor}
        return build_room_graph(rooms)

    @staticmethod
    def _house_ref_included_area_ids(cycles: list[_RoomCycle]) -> set[str]:
        """Rooms contributing to the house mean: indoor, closed window, with a reading.

        This is the area-id form of the filter in :meth:`_house_ref_temp`, kept in
        one place so training and the MPC use exactly the same include set.
        """
        return {
            cycle.area_id
            for cycle in cycles
            if not cycle.is_outdoor and not cycle.window_open and cycle.current_temp is not None
        }

    @staticmethod
    def _room_ref_temp(
        cycle: _RoomCycle,
        graph: RoomGraph,
        temperatures: Mapping[str, float],
        included: Collection[str],
        fallback: float | None,
    ) -> float | None:
        """Neighbour-weighted local reference for a room; *fallback* when it has none."""
        weights = reference_weights(graph, cycle.area_id, included)
        if not weights or fallback is None:
            # No declared neighbours, or no usable house reading to fall back on:
            # the caller records the reference as unknown rather than inventing one.
            return fallback
        return reference_temperature(weights, temperatures, fallback)

    def _step_wet_coil(
        self,
        cycles: list[_RoomCycle],
        plant_settings: dict,
        observed_action: str | None,
        now: float,
    ) -> bool:
        """Advance the wet-coil charge and return whether Circulation is locked out."""
        self._wet_coil.step(observed_action, UPDATE_INTERVAL / 60.0, now)
        weighted_rh = self._weighted_indoor_rh(cycles)
        return self._wet_coil.is_circulation_locked(weighted_rh, plant_settings, now)

    @staticmethod
    def _weighted_indoor_rh(cycles: list[_RoomCycle]) -> float | None:
        """Room Priority weighted relative humidity across the controlled rooms."""
        weighted = 0.0
        total = 0.0
        for cycle in cycles:
            if cycle.is_outdoor or cycle.current_humidity is None:
                continue
            weight = max(0.0, cycle.occupancy.priority_weight)
            if weight <= 0.0:
                continue
            weighted += weight * cycle.current_humidity
            total += weight
        return weighted / total if total > 0.0 else None

    def _any_override(self, cycles: list[_RoomCycle], settings: dict) -> bool:
        """Whether any room override or the whole-home plant override holds priority."""
        if is_plant_override_active(settings):
            return True
        return any(is_override_active(cycle.room) for cycle in cycles if not cycle.is_outdoor)

    @staticmethod
    def _all_force_off(cycles: list[_RoomCycle], settings: dict) -> bool:
        """Whether the whole home is off, by setting or by every room being forced off."""
        if not settings.get("climate_control_active", True):
            return True
        controlled = [cycle for cycle in cycles if cycle.controllable]
        return not controlled or all(cycle.force_off for cycle in controlled)

    def _house_climate_mode(self, cycles: list[_RoomCycle], house: dict, house_active: bool) -> str:
        """The plant's climate mode: heat_only, cool_only or auto from the enabled rooms.

        Only a house whose every enabled room allows one side gets that
        one-sided plant mode; a single auto room keeps both sides available.
        A room following an active house contributes the house's mode.
        """
        modes = [
            self._effective_climate_mode(cycle.room, house, house_active) for cycle in cycles if cycle.controllable
        ]
        if modes and all(mode == CLIMATE_MODE_HEAT_ONLY for mode in modes):
            return CLIMATE_MODE_HEAT_ONLY
        if modes and all(mode == CLIMATE_MODE_COOL_ONLY for mode in modes):
            return CLIMATE_MODE_COOL_ONLY
        return "auto"

    def _startup_waiting(self, cycles: list[_RoomCycle], thermostat_entity: str) -> bool:
        """Hold the plant while a controlled room has no first reading or unresolved window yet."""
        if not thermostat_entity:
            return False
        for cycle in cycles:
            if not cycle.controllable:
                continue
            has_sensor = bool(cycle.room.get("temperature_sensor"))
            waiting_for_temp = has_sensor and self._waiting_for_first_reading(cycle.area_id)
            if waiting_for_temp or cycle.window_pending:
                return True
        return False

    def _waiting_for_first_reading(self, area_id: str) -> bool:
        """True while a room has had no valid reading since startup, within the grace period."""
        if area_id in self._had_valid_temp:
            return False
        elapsed = time.monotonic() - self._startup_ts
        if elapsed >= MAX_SENSOR_STALENESS:
            if area_id not in self._startup_guard_warned:
                self._startup_guard_warned.add(area_id)
                _LOGGER.warning(
                    "Room '%s': no temperature reading within %ds after startup, resuming control",
                    area_id,
                    MAX_SENSOR_STALENESS,
                )
            return False
        return True

    def _evaluate_central_mpc(
        self,
        cycles: list[_RoomCycle],
        settings: dict,
        plant_settings: dict,
        guard_state: PlantGuardState,
        climate_mode: str,
        outdoor_forecast: list[dict],
        cloud_series: list[float | None] | None,
        now: float,
        wet_coil_locked: bool,
        house: dict,
        house_ctx: HouseContext | None,
        house_active: bool,
        *,
        graph: RoomGraph | None = None,
        included: Collection[str] | None = None,
    ) -> CentralMPCPlan:
        """Build the room horizon inputs and hand the whole home to the joint MPC.

        ``graph``/``included`` are the per-cycle topology inputs, threaded in from
        the same pass that trains the EKFs. When omitted (direct callers) an empty
        graph keeps every room's reference at the house mean, as before.
        """
        empty_graph = build_room_graph({})
        topology = graph if graph is not None else empty_graph
        include_set: Collection[str] = included if included is not None else ()
        n_blocks = MIN_HORIZON_BLOCKS
        can_heat = climate_mode != CLIMATE_MODE_COOL_ONLY
        can_cool = climate_mode != CLIMATE_MODE_HEAT_ONLY
        step_seconds = PLAN_DT_MINUTES * 60.0
        inputs: list[RoomHorizonInput] = []
        estimators: dict[str, Any] = {}
        solar_by_azimuths: dict[tuple[int, ...], list[float]] = {}
        house_solar: list[float] | None = None
        for cycle in cycles:
            if not cycle.controllable or cycle.current_temp is None:
                continue
            room = cycle.room
            effective_mode = self._effective_climate_mode(room, house, house_active)
            room_can_heat = effective_mode != CLIMATE_MODE_COOL_ONLY
            room_can_cool = effective_mode != CLIMATE_MODE_HEAT_ONLY
            estimator = self._model_manager.get_estimator(cycle.area_id)
            estimators[cycle.area_id] = estimator
            priority, occupied = self._occupancy_series(cycle, settings, now, n_blocks)
            resolver = make_target_resolver(
                cycle.schedule_blocks,
                room,
                settings,
                hass=self.hass,
                presence_away=cycle.presence_away,
                mold_prevention_delta=cycle.mold.prevention_delta,
                room_vacant=cycle.occupancy.is_vacant,
                price_signal=self._price_manager.signal,
                house=house_ctx,
                follow_house=cycle.house_following,
            )
            targets = [resolver(now + k * step_seconds) for k in range(n_blocks)]
            # A room that may only cool never asks the whole-home plan for heat (and
            # the other way round); its Guardrail Band still protects it.
            comfort_heat: list[float | None] = [item.heat for item in targets]
            comfort_cool: list[float | None] = [item.cool for item in targets]
            if effective_mode == CLIMATE_MODE_COOL_ONLY:
                comfort_heat = [None] * n_blocks
            elif effective_mode == CLIMATE_MODE_HEAT_ONLY:
                comfort_cool = [None] * n_blocks
            # Memoise by azimuth tuple: ten south-facing rooms share one series.
            # The unoriented GHI series is also computed once when any room needs it.
            azimuths = room_solar_azimuths(room)
            if azimuths:
                solar = solar_by_azimuths.get(azimuths)
                if solar is None:
                    solar = build_oriented_solar_series(
                        self.hass.config.latitude,
                        self.hass.config.longitude,
                        n_blocks,
                        list(azimuths),
                        PLAN_DT_MINUTES,
                        start_ts=now,
                        cloud_series=cloud_series,
                    )
                    solar_by_azimuths[azimuths] = solar
            else:
                if house_solar is None:
                    house_solar = build_solar_series(
                        self.hass.config.latitude,
                        self.hass.config.longitude,
                        n_blocks,
                        PLAN_DT_MINUTES,
                        start_ts=now,
                        cloud_series=cloud_series,
                    )
                solar = house_solar
            inputs.append(
                RoomHorizonInput(
                    area_id=cycle.area_id,
                    model=self._model_manager.get_model(cycle.area_id),
                    T_init=cycle.current_temp,
                    rh_init=cycle.current_humidity,
                    rebound_init=self._rebound_tracker.charge(cycle.area_id),
                    door_mix_factor=cycle.door.mix_factor if cycle.door else 1.0,
                    door_supply_factor=cycle.door.supply_factor if cycle.door else 1.0,
                    priority_series=priority,
                    occupied_prob_series=occupied,
                    comfort_heat_series=comfort_heat,
                    comfort_cool_series=comfort_cool,
                    guardrail_heat=float(settings.get("guardrail_heat_c", DEFAULT_GUARDRAIL_HEAT)),
                    guardrail_cool=float(settings.get("guardrail_cool_c", DEFAULT_GUARDRAIL_COOL)),
                    solar_series=[value * cycle.shading_factor for value in solar],
                    occupancy_gain_series=[cycle.occupancy.effective_occupancy] * n_blocks,
                    preheat_enabled=bool(room.get("preheat_enabled", True)),
                    include_in_house_mean=not cycle.window_open,
                    neighbours=reference_weights(topology, cycle.area_id, include_set),
                    mixing_group=topology.mixing_group.get(cycle.area_id),
                    mpc_ready=is_room_mpc_ready(estimator, room_can_heat, room_can_cool),
                )
            )
        outdoor_series = self._outdoor_series(outdoor_forecast, now, n_blocks)
        if not inputs:
            return CentralMPCPlan(actions=[], room_temperatures={}, house_temperatures=[])
        plan = self._mpc_controller.evaluate(
            rooms=inputs,
            T_outdoor_series=outdoor_series,
            settings=settings,
            estimators=estimators,
            legal_actions=guard_state.legal_actions,
            forced_action=guard_state.forced_action,
            initial_action=self._plant_manager.active_action,
            initial_run_blocks=self._initial_run_blocks(now),
            initial_off_blocks=0,
            can_heat=can_heat,
            can_cool=can_cool,
            dt_minutes=PLAN_DT_MINUTES,
            horizon_blocks=n_blocks,
            wet_coil_locked_blocks=self._wet_coil_blocks_left(plant_settings, now) if wet_coil_locked else 0,
        )
        return plan

    def _occupancy_series(
        self,
        cycle: _RoomCycle,
        settings: dict,
        now: float,
        n_blocks: int,
    ) -> tuple[list[float], list[float]]:
        """Room Priority weights and occupancy probabilities for the plan horizon."""
        override_until = cycle.room.get("override_until")
        override_until = None if override_until is None else float(override_until)

        def schedule_active_at(ts: float) -> bool:
            return self._schedule_active_at(cycle.schedule_blocks, cycle.schedule_state, ts)

        def override_active_at(ts: float) -> bool:
            return is_override_active(cycle.room) and (override_until is None or ts < override_until)

        converter = partial(ha_temp_to_celsius, self.hass)

        def room_setting_at(ts: float) -> bool:
            return schedule_block_sets_target(ts, cycle.schedule_blocks, converter)

        return self._occupancy_manager.build_priority_series(
            cycle.area_id,
            cycle.room,
            settings,
            now,
            n_blocks,
            PLAN_DT_MINUTES,
            schedule_active_at=schedule_active_at,
            override_active_at=override_active_at,
            room_setting_at=room_setting_at,
            presence_away=cycle.presence_away,
            vacation_active=cycle.vacation_active,
            window_open=cycle.window_open,
            current_temp_c=cycle.current_temp,
        )

    def _outdoor_series(self, forecast: list[dict], now: float, n_blocks: int) -> list[float]:
        """Outdoor temperature per plan block, from the hourly forecast when there is one."""
        fallback = self.outdoor_temp_effective if self.outdoor_temp_effective is not None else 21.0
        series: list[float] = []
        for k in range(n_blocks):
            block_ts = now + k * PLAN_DT_MINUTES * 60.0
            value = fallback
            if forecast:
                index = min(len(forecast) - 1, max(0, int((block_ts - now) // 3600)))
                temp = _numeric(forecast[index].get("temperature"))
                if temp is not None:
                    value = temp
            series.append(float(value))
        return series

    def _initial_run_blocks(self, now: float) -> int:
        """Whole plan blocks the running action already ran."""
        started = self._plant_manager.action_started_at
        if started is None or self._plant_manager.active_action == MODE_IDLE:
            return 0
        return max(0, int((now - started) / (PLAN_DT_MINUTES * 60.0)))

    def _wet_coil_blocks_left(self, plant_settings: dict, now: float) -> int:
        """Blocks of Circulation lockout left while the coil is still wet."""
        lockout_minutes = _as_int(plant_settings.get("wet_coil_lockout_minutes", 25), 25)
        last_cooling = self._wet_coil.last_cooling_ts
        elapsed = 0.0 if last_cooling is None else max(0.0, now - last_cooling) / 60.0
        remaining = max(0.0, lockout_minutes - elapsed)
        return max(0, round(remaining / PLAN_DT_MINUTES))

    # ------------------------------------------------------------------ #
    # Actuation
    # ------------------------------------------------------------------ #

    async def _async_actuate_plant(
        self,
        plant_settings: dict,
        settings: dict,
        cycles: list[_RoomCycle],
        thermostat_entity: str,
        thermostat_state: Any,
        house_ref_temp: float | None,
        commanded_action: str,
        climate_mode: str,
        all_force_off: bool,
        startup_waiting: bool,
    ) -> tuple[str, bool, ActuationResult | None]:
        """Apply the commanded action to the Central Thermostat and re-observe the equipment.

        Returns ``(display_action, fan_circulating, actuation)`` where *actuation*
        is the :class:`~.control.thermostat_actuator.ActuationResult` (or None when
        no command was sent): the panel shows the bounded setpoints it holds.
        """
        if not thermostat_entity or not settings.get("climate_control_active", True) or startup_waiting:
            return commanded_action, False, None

        active_heat = [
            cycle.targets.heat
            for cycle in cycles
            if cycle.controllable and not cycle.window_open and cycle.targets.heat is not None
        ]
        active_cool = [
            cycle.targets.cool
            for cycle in cycles
            if cycle.controllable and not cycle.window_open and cycle.targets.cool is not None
        ]
        actuation = await self._actuator.async_apply(
            self.hass,
            plant_settings,
            commanded_action,
            climate_mode=climate_mode,
            house_ref_temp_c=house_ref_temp,
            active_heat_ceiling_c=max(active_heat) if active_heat else None,
            active_cool_floor_c=min(active_cool) if active_cool else None,
            force_off=all_force_off,
            override_tracker=self._thermostat_overrides,
        )
        self._plant_manager.record_commanded_action(commanded_action)
        observed_action, fan_circulating = self._plant_manager.observe_plant_action(
            thermostat_state,
            plant_settings,
            commanded_action,
        )
        if self._thermostat_lacks_hvac_action(thermostat_state):
            return commanded_action, fan_circulating, actuation
        return (observed_action or MODE_IDLE), fan_circulating, actuation

    @staticmethod
    def _thermostat_lacks_hvac_action(thermostat_state: Any) -> bool:
        """Whether the thermostat integration reports no ``hvac_action`` at all."""
        attributes = getattr(thermostat_state, "attributes", None)
        if not isinstance(attributes, dict):
            return True
        return attributes.get("hvac_action") is None

    async def _async_process_covers(
        self,
        cycle: _RoomCycle,
        house_context: HouseCoverContext,
    ) -> CoverResult | None:
        """Run the cover loop for one room with the plant's whole-home context."""
        if not cycle.room.get("covers"):
            return None
        return await self._cover_orchestrator.async_process(
            cycle.area_id,
            cycle.room,
            cycle.targets,
            self._room_display_mode(cycle, house_context.plant_demand, self._room_offline(cycle)),
            cycle.current_temp,
            self.outdoor_temp_effective,
            self._current_q_solar,
            None,
            is_override_active(cycle.room),
            cycle.raw_open or cycle.window_open,
            house_context=house_context,
        )

    @staticmethod
    def _room_offline(cycle: _RoomCycle) -> bool:
        """Whether the room cannot take a plant action this cycle."""
        return cycle.force_off or cycle.window_open

    def _any_cool_sensitive(self, cycles: list[_RoomCycle]) -> bool:
        """Whether an occupied room sits close enough to its heating floor to veto cooling."""
        for cycle in cycles:
            if cycle.is_outdoor or cycle.occupancy.effective_occupancy < 0.5:
                continue
            if cycle.current_temp is None or cycle.targets.heat is None:
                continue
            if cycle.current_temp - cycle.targets.heat <= 0.8:
                return True
        return False

    def _room_display_mode(self, cycle: _RoomCycle, plant_action: str, forced_off: bool) -> str:
        """What the room shows: the plant action when it serves the room, else idle."""
        if cycle.is_outdoor or forced_off or cycle.window_open:
            return MODE_IDLE
        if plant_action in (MODE_HEATING, MODE_COOLING, MODE_CIRCULATING):
            return plant_action
        return MODE_IDLE

    # ------------------------------------------------------------------ #
    # Learned state, history and persistence
    # ------------------------------------------------------------------ #

    def _train_and_record(
        self,
        cycles: list[_RoomCycle],
        room_states: dict[str, dict],
        settings: dict,
        house_ref_temp: float | None,
        ekf_mode: str | None,
        dt_minutes: float,
        *,
        graph: RoomGraph | None = None,
        temperatures: Mapping[str, float] | None = None,
        included: Collection[str] | None = None,
    ) -> None:
        """Step the forced-air charge, train each room's EKF and track the display modes."""
        topology = graph if graph is not None else build_room_graph({})
        temp_map: Mapping[str, float] = temperatures if temperatures is not None else {}
        include_set: Collection[str] = included if included is not None else ()
        learning_disabled = set(settings.get("learning_disabled_rooms", []))
        for cycle in cycles:
            if cycle.is_outdoor:
                continue
            # Configuration changes take effect on the first cycle after save,
            # even while the room is not trainable. get_estimator creates it.
            azimuths = room_solar_azimuths(cycle.room)
            local_reference = bool(reference_weights(topology, cycle.area_id, include_set))
            self._model_manager.get_estimator(cycle.area_id).apply_regressor_signature(
                regressor_signature(azimuths, local_reference)
            )
            mean_charge, _ = self._rebound_tracker.step(cycle.area_id, ekf_mode, dt_minutes)
            display_mode = room_states.get(cycle.area_id, {}).get("mode", MODE_IDLE)
            self._note_room_mode(cycle.area_id, display_mode)
            current_temp = cycle.current_temp_raw
            outdoor_temp = self.outdoor_temp_effective
            if cycle.area_id in learning_disabled or current_temp is None or outdoor_temp is None or ekf_mode is None:
                self._model_inputs.pop(cycle.area_id, None)
                self._ekf_training.clear(cycle.area_id)
                continue
            room_ref_temp = self._room_ref_temp(cycle, topology, temp_map, include_set, house_ref_temp)
            self._ekf_training.process(
                area_id=cycle.area_id,
                current_temp=current_temp,
                T_outdoor=outdoor_temp,
                ekf_mode=ekf_mode,
                ekf_pf=self._training_power(ekf_mode),
                window_open=cycle.window_open,
                raw_open=cycle.raw_open or cycle.window_pending,
                q_residual=0.0,
                shading_factor=cycle.shading_factor,
                q_solar=self._room_q_solar(cycle),
                can_heat=cycle.room.get("climate_mode", "auto") != CLIMATE_MODE_COOL_ONLY,
                can_cool=cycle.room.get("climate_mode", "auto") != CLIMATE_MODE_HEAT_ONLY,
                dt_minutes=dt_minutes,
                q_occupancy=cycle.occupancy.effective_occupancy,
                q_rebound=mean_charge,
                T_house=room_ref_temp,
                door_mix_factor=cycle.door.mix_factor if cycle.door else 1.0,
                door_supply_factor=cycle.door.supply_factor if cycle.door else 1.0,
            )
            self._model_inputs[cycle.area_id] = self._model_input_row(cycle, ekf_mode, room_ref_temp)
            # Attach the training inputs to the live state for analytics/diagnostics.
            state = room_states.get(cycle.area_id)
            if state is not None:
                state.update(self._model_inputs[cycle.area_id])

    @staticmethod
    def _training_power(ekf_mode: str | None) -> float:
        """Power label the EKF trains with for a plant action."""
        if ekf_mode == MODE_HEATING or ekf_mode == MODE_COOLING or ekf_mode == MODE_CIRCULATING:
            return 1.0
        return 0.0

    def _model_input_row(
        self,
        cycle: _RoomCycle,
        ekf_mode: str | None,
        house_ref_temp: float | None,
    ) -> dict[str, float | None]:
        """The inputs this cycle's learning used, in ``history_store.MODEL_INPUT_FIELDS`` shape."""
        if ekf_mode == MODE_HEATING:
            power = 1.0
        elif ekf_mode == MODE_COOLING:
            power = -1.0
        else:
            power = 0.0
        return {
            "model_power": round(power, 3),
            # The backtest replays the oriented per-room regressor the EKF trained
            # on, including the shading factor (applied here, as in training).
            "model_solar": round(self._room_q_solar(cycle) * cycle.shading_factor, 3),
            "model_occupancy": round(cycle.occupancy.effective_occupancy, 3),
            # No house reference means "unknown", never a 0 °C house sink: the history
            # store writes "" and the backtest then falls back to the 1R1C model.
            "model_house_temp": round(house_ref_temp, 3) if house_ref_temp is not None else None,
            "model_fan": 1.0 if ekf_mode == MODE_CIRCULATING else 0.0,
            "model_door_mix": round(cycle.door.mix_factor if cycle.door else 1.0, 3),
            "model_door_supply": round(cycle.door.supply_factor if cycle.door else 1.0, 3),
        }

    def _note_room_mode(self, area_id: str, mode: str) -> None:
        """Track when a room's displayed mode changed (diagnostics/REST surface)."""
        previous = self._previous_modes.get(area_id, MODE_IDLE)
        if mode != MODE_IDLE and previous != mode:
            self._mode_on_since[area_id] = time.time()
        elif mode == MODE_IDLE:
            self._mode_on_since.pop(area_id, None)
        self._previous_modes[area_id] = mode

    def _build_room_states(
        self,
        cycles: list[_RoomCycle],
        settings: dict,
        plan: CentralMPCPlan,
        plant_action: str,
        startup_waiting: bool,
        fan_circulating: bool = False,
    ) -> dict[str, dict]:
        """Build the live state dictionary the entities, REST and diagnostics read."""
        states: dict[str, dict] = {}
        modeled = set(self._model_manager.get_room_ids())
        for cycle in cycles:
            if cycle.is_outdoor:
                states[cycle.area_id] = self._outdoor_room_state(cycle, plant_action)
                continue
            offline = self._room_offline(cycle) or startup_waiting
            display_mode = self._room_display_mode(cycle, plant_action, offline)
            states[cycle.area_id] = self._room_state(
                cycle, settings, plan, display_mode, plant_action, fan_circulating, modeled
            )
        return states

    def _room_state(
        self,
        cycle: _RoomCycle,
        settings: dict,
        plan: CentralMPCPlan,
        display_mode: str,
        plant_action: str,
        fan_circulating: bool,
        modeled: set[str],
    ) -> dict[str, Any]:
        """One room's live state."""
        area_id = cycle.area_id
        room = cycle.room
        predicted = plan.room_temperatures.get(area_id, [])
        cover_result = cycle.cover_result
        has_covers = bool(room.get("covers"))
        price_signal = self._price_manager.signal
        return {
            "area_id": area_id,
            "current_temp": cycle.current_temp,
            "current_temp_raw": cycle.current_temp_raw,
            "current_humidity": cycle.current_humidity,
            "target_temp": self._display_target(room, cycle.targets, display_mode),
            "heat_target": cycle.targets.heat,
            "cool_target": cycle.targets.cool,
            "schedule_temp_warnings": cycle.schedule_warnings,
            "mode": display_mode,
            "commanded_mode": plant_action,
            "heating_power": 100 if display_mode in (MODE_HEATING, MODE_COOLING) else 0,
            "device_setpoint": None,
            "window_open": cycle.window_open,
            **build_override_live(
                room,
                suppressed=is_override_suppressed(room, settings, cycle.presence_away),
            ),
            "active_schedule_index": cycle.active_schedule_index,
            "confidence": self._model_manager.get_confidence(area_id),
            "mpc_active": plan.mpc_active,
            "preheat": plan.preheat_info.get(area_id),
            "presence_away": cycle.presence_away,
            "occupancy_vacant": cycle.occupancy.is_vacant,
            "force_off": cycle.force_off,
            "mold_risk_level": cycle.mold.risk_level,
            "mold_surface_rh": round(cycle.mold.surface_rh, 1) if cycle.mold.surface_rh is not None else None,
            "mold_surface_temp": round(cycle.mold.surface_temp, 1) if cycle.mold.surface_temp is not None else None,
            "mold_prevention_active": cycle.mold.prevention_active,
            "mold_prevention_delta": cycle.mold.prevention_delta,
            "target_reason": cycle.target_reason,
            "house_following": cycle.house_following,
            "mold_boost": cycle.mold_boost,
            "price_shift_heat": cycle.price_shift[0],
            "price_shift_cool": cycle.price_shift[1],
            "price_level": price_signal.current_level if price_signal else None,
            "shading_factor": cycle.shading_factor,
            "q_occupancy": cycle.occupancy.effective_occupancy,
            "q_residual": 0.0,
            "n_observations": self._model_manager.get_n_observations(area_id),
            "blind_position": self._cover_orchestrator.get_current_position(area_id) if has_covers else None,
            "cover_auto_paused": self._cover_orchestrator.is_user_override_active(area_id) if has_covers else False,
            "cover_override_until": self._cover_orchestrator.get_user_override_until(area_id) if has_covers else None,
            "cover_reason": cover_result.decision.reason if cover_result and has_covers else "",
            "cover_forced_reason": cover_result.forced_reason if cover_result and has_covers else "",
            "active_cover_schedule_index": (
                cover_result.active_cover_schedule_index if cover_result and has_covers else -1
            ),
            **cycle.air,
            "effective_temp": cycle.effective_temp,
            "priority_weight": cycle.occupancy.priority_weight,
            "occupancy_probability": cycle.occupancy.effective_occupancy,
            "occupancy_dwell": round(cycle.occupancy.dwell_score, 3),
            "fan_circulating": bool(fan_circulating),
            "beta_fan_rate": (round(self._model_manager.get_model(area_id).Q_fan, 3) if area_id in modeled else None),
            "interior_door_open": cycle.door.is_open if cycle.door else True,
            "exterior_window_open": cycle.raw_open,
            "predicted_temp": predicted[1] if len(predicted) > 1 else None,
        }

    @staticmethod
    def _display_target(room: dict, targets: TargetTemps, display_mode: str) -> float | None:
        """The one target the room card shows: the side the plant serves, else the schedule's."""
        climate_mode = room.get("climate_mode", "auto")
        if climate_mode == CLIMATE_MODE_COOL_ONLY:
            return targets.cool
        if climate_mode == CLIMATE_MODE_HEAT_ONLY:
            return targets.heat
        if display_mode == MODE_COOLING and targets.cool is not None:
            return targets.cool
        if targets.heat is not None:
            return targets.heat
        return targets.cool

    def _outdoor_room_state(self, cycle: _RoomCycle, plant_action: str) -> dict[str, Any]:
        """A passive state for a room configured as an outdoor area."""
        return {
            "area_id": cycle.area_id,
            "current_temp": cycle.current_temp,
            "current_temp_raw": cycle.current_temp_raw,
            "current_humidity": cycle.current_humidity,
            "target_temp": None,
            "heat_target": None,
            "cool_target": None,
            "schedule_temp_warnings": [],
            "mode": MODE_IDLE,
            "commanded_mode": plant_action,
            "heating_power": 0,
            "device_setpoint": None,
            "window_open": False,
            **build_override_live(cycle.room),
            "active_schedule_index": -1,
            "confidence": None,
            "mpc_active": False,
            "preheat": None,
            "presence_away": False,
            "occupancy_vacant": False,
            "force_off": True,
            "mold_risk_level": "ok",
            "mold_surface_rh": None,
            "mold_surface_temp": None,
            "mold_prevention_active": False,
            "mold_prevention_delta": 0,
            "target_reason": "",
            "house_following": False,
            "mold_boost": 0.0,
            "price_shift_heat": 0.0,
            "price_shift_cool": 0.0,
            "price_level": self._price_manager.signal.current_level if self._price_manager.signal else None,
            "shading_factor": 1.0,
            "q_occupancy": 0.0,
            "q_residual": 0.0,
            "n_observations": 0,
            "blind_position": None,
            "cover_auto_paused": False,
            "cover_override_until": None,
            "cover_reason": "",
            "cover_forced_reason": "",
            "active_cover_schedule_index": -1,
            **_NO_AIR_QUALITY,
            "effective_temp": None,
            "priority_weight": 0.0,
            "occupancy_probability": 0.0,
            "occupancy_dwell": 0.0,
            "fan_circulating": False,
            "beta_fan_rate": None,
            "interior_door_open": True,
            "exterior_window_open": False,
        }

    async def _async_write_history(
        self,
        cycles: list[_RoomCycle],
        plan: CentralMPCPlan,
        room_states: dict[str, dict],
        settings: dict,
        now: float,
    ) -> None:
        """Append the throttled 3-minute history rows and rotate the CSVs."""
        self._history_write_count += 1
        learning_disabled = set(settings.get("learning_disabled_rooms", []))
        if self._history_write_count >= HISTORY_WRITE_CYCLES and self._history_store:
            self._history_write_count = 0
            price = self._price_manager.signal
            price_now = price.price_at(now) if price else None
            by_id = {cycle.area_id: cycle for cycle in cycles}
            for area_id, cycle in by_id.items():
                if cycle.is_outdoor or area_id in learning_disabled:
                    continue
                state = room_states.get(area_id, {})
                await self._async_record_history_row(area_id, cycle, state, price_now, now)

        self._history_rotate_count += 1
        if self._history_rotate_count >= HISTORY_ROTATE_CYCLES and self._history_store:
            self._history_rotate_count = 0
            for area_id in {cycle.area_id for cycle in cycles}:
                try:
                    await self.hass.async_add_executor_job(self._history_store.rotate, area_id)
                except Exception:  # noqa: BLE001
                    _LOGGER.warning("History rotation failed for '%s'", area_id)

    async def _async_record_history_row(
        self,
        area_id: str,
        cycle: _RoomCycle,
        state: dict,
        price_now: float | None,
        now: float,
    ) -> None:
        """Record one room's history row for the analytics backtest."""
        if self._history_store is None:
            return
        shift = (state.get("price_shift_heat") or 0.0) or -(state.get("price_shift_cool") or 0.0)
        row = {
            "room_temp": cycle.current_temp_raw if cycle.current_temp_raw is not None else cycle.current_temp,
            "outdoor_temp": self.outdoor_temp_effective,
            "target_temp": state.get("target_temp"),
            "mode": state.get("mode", MODE_IDLE),
            "predicted_temp": state.get("predicted_temp"),
            "window_open": cycle.window_open,
            "heating_power": state.get("heating_power", 0),
            "solar_irradiance": round(self._current_q_solar, 3),
            "blind_position": state.get("blind_position"),
            "cover_reason": state.get("cover_reason", ""),
            "device_setpoint": None,
            "occupancy": cycle.occupancy.effective_occupancy > 0.5,
            "humidity": cycle.current_humidity,
            "heat_target": cycle.targets.heat,
            "cool_target": cycle.targets.cool,
            "target_reason": cycle.target_reason,
            "mold_boost": cycle.mold_boost or "",
            "price": price_now if price_now is not None else "",
            "price_level": state.get("price_level") or "",
            "price_shift": shift or "",
            "co2": cycle.air.get("co2"),
            "effective_temp": cycle.effective_temp,
            "priority_weight": cycle.occupancy.priority_weight,
            "door_open": bool(cycle.door and cycle.door.is_open),
            "fan_circulating": state.get("mode") == MODE_CIRCULATING,
            **self._model_inputs.get(area_id, {}),
        }
        try:
            await self.hass.async_add_executor_job(self._history_store.record, area_id, row, now)
        except Exception:  # noqa: BLE001
            _LOGGER.warning("History record failed for '%s'", area_id)

    async def _async_reconcile_ventilation(self, settings: dict, room_states: dict[str, dict]) -> None:
        """Run each room's ventilation entity while its air is bad (#267)."""
        rooms = rooms_with_area(self.hass, self.hass.data[DOMAIN]["store"].get_rooms())
        wanted: dict[str, bool] = {}
        if settings.get("climate_control_active", True):
            for area_id, room in rooms.items():
                entity_id = room.get("ventilation_entity")
                if not entity_id or not has_air_quality_sensor(room) or not room.get("climate_control_enabled", True):
                    continue
                bad = bool((room_states.get(area_id) or {}).get("air_quality_bad"))
                wanted[entity_id] = wanted.get(entity_id, False) or bad
        try:
            await self._ventilation.async_reconcile(wanted, time.time())
        except Exception:  # noqa: BLE001
            _LOGGER.warning("Ventilation control failed", exc_info=True)
        for area_id, state in room_states.items():
            entity_id = (rooms.get(area_id) or {}).get("ventilation_entity")
            state["ventilation_active"] = bool(entity_id and self._ventilation.is_running(entity_id))

    async def async_release_ventilation(self) -> None:
        """Switch off the ventilation units RoomMind switched on (unload)."""
        try:
            await self._ventilation.async_reconcile({}, time.time())
        except Exception:  # noqa: BLE001
            _LOGGER.warning("Releasing ventilation failed", exc_info=True)
        if self._ventilation.dirty:
            await self.hass.data[DOMAIN]["store"].async_save_settings(
                {"ventilation_state": self._ventilation.get_state()}
            )
            self._ventilation.dirty = False

    async def _async_record_recent_temps(self, room_states: dict[str, dict]) -> None:
        """Feed the short series the dashboard cards draw; seeded from the history once per room."""
        now = time.time()
        for area_id, state in room_states.items():
            if self._recent_temps.needs_seed(area_id) and self._history_store:
                try:
                    rows = await self.hass.async_add_executor_job(
                        self._history_store.read_detail, area_id, RECENT_WINDOW_S
                    )
                except Exception:  # noqa: BLE001
                    _LOGGER.debug("Room '%s': reading recent history failed", area_id, exc_info=True)
                    rows = []
                self._recent_temps.seed(area_id, rows, now)
            target = self._recent_temps.target_for(
                area_id,
                state.get("mode"),
                state.get("heat_target"),
                state.get("target_temp"),
                state.get("cool_target"),
            )
            self._recent_temps.add(area_id, now, state.get("current_temp"), target)

    async def _async_persist(self, store: Any, room_states: dict[str, dict]) -> None:
        """Persist the plant, coil, ventilation and throttled learned state."""
        self._thermal_save_count += 1
        if self._thermal_save_count >= THERMAL_SAVE_CYCLES:
            self._thermal_save_count = 0
            self._note_confidence_days(room_states)
            await store.async_save_thermal_data(self._model_manager.to_dict())
            await store.async_save_occupancy_trends(self._occupancy_manager.to_dict())
        if self._plant_manager.state_dirty:
            self._plant_manager.state_dirty = False
            await store.async_save_settings({"central_plant_state": self._plant_manager.to_dict()})
        if self._wet_coil.state_dirty:
            self._wet_coil.state_dirty = False
            await store.async_save_settings({"wet_coil_state": self._wet_coil.to_dict()})
        if self._ventilation.dirty:
            self._ventilation.dirty = False
            await store.async_save_settings({"ventilation_state": self._ventilation.get_state()})

    def _note_confidence_days(self, room_states: dict[str, dict]) -> None:
        """Daily confidence snapshot per learned room for the analytics model chart."""
        day = dt_util.now().date().isoformat()
        known = set(self._model_manager.get_room_ids())
        for area_id, state in room_states.items():
            if area_id in known:
                self._model_manager.get_estimator(area_id).note_confidence_day(day, bool(state.get("mpc_active")))

    async def _async_load_price_history(self) -> list[tuple[float, float]]:
        """Prices recorded with the history, so ranking without a forecast survives a restart."""
        if self._history_store is None:
            return []
        store = self.hass.data.get(DOMAIN, {}).get("store")
        area_ids = list(store.get_rooms()) if store else []
        for area_id in area_ids:
            rows = await self.hass.async_add_executor_job(self._history_store.read_detail, area_id)
            rows += await self.hass.async_add_executor_job(self._history_store.read_history, area_id)
            prices: list[tuple[float, float]] = []
            for row in rows:
                try:
                    prices.append((float(row["timestamp"]), float(row["price"])))
                except (KeyError, TypeError, ValueError):
                    continue
            if prices:
                return prices
        return []

    # ------------------------------------------------------------------ #
    # Public surface used by the entities, the websocket API and diagnostics
    # ------------------------------------------------------------------ #

    @property
    def price_signal(self) -> PriceSignal | None:
        """The electricity price signal of the current cycle, None without a price source."""
        return self._price_manager.signal

    def price_status(self) -> dict | None:
        """What the settings page shows about the price source."""
        return self._price_manager.status()

    @property
    def history_store(self) -> HistoryStore | None:
        """Access to the history store for cleanup operations."""
        return self._history_store

    @property
    def recent_temps(self) -> RecentTemps:
        """The short temperature series the dashboard cards draw (read by the websocket API)."""
        return self._recent_temps

    def note_override_seed(self, area_id: str) -> None:
        """Mark an override just seeded by switching the override entity on.

        Until the caller's real targets arrive (or the grace period ends) the seed
        is not evaluated: acting on a placeholder band can start the plant for the
        wrong side (#447). In memory only.
        """
        self._override_seeded_at[area_id] = time.monotonic()

    def clear_override_seed(self, area_id: str) -> None:
        """Forget the seed marker: the override now holds values the user chose."""
        self._override_seeded_at.pop(area_id, None)

    def is_override_seed_pending(self, area_id: str) -> bool:
        """Return True while a fresh seed is still masked (read-only, unlike the masking itself)."""
        seeded_at = self._override_seeded_at.get(area_id)
        return seeded_at is not None and time.monotonic() - seeded_at < OVERRIDE_TURN_ON_REFRESH_DELAY_S

    def _without_pending_override_seed(self, room: dict) -> dict:
        """Return *room* as if no override existed while a fresh seed is pending."""
        area_id = room.get("area_id", "unknown")
        seeded_at = self._override_seeded_at.get(area_id)
        if seeded_at is None:
            return room
        if time.monotonic() - seeded_at >= OVERRIDE_TURN_ON_REFRESH_DELAY_S:
            del self._override_seeded_at[area_id]
            return room
        return {**room, "override_heat": None, "override_cool": None, "override_until": None, "override_type": None}

    async def async_clear_plant_override(self) -> None:
        """Release a manual Central Thermostat override and return to the schedule."""
        store = self.hass.data[DOMAIN]["store"]
        await store.async_save_settings(
            {
                "plant_override_heat": None,
                "plant_override_cool": None,
                "plant_override_until": None,
            }
        )
        await self.async_request_refresh()

    async def async_turn_room_off(self, area_id: str) -> None:
        """Stop controlling a room (its climate entity was switched off)."""
        store = self.hass.data[DOMAIN]["store"]
        await store.async_update_room(area_id, {"climate_control_enabled": False})
        await self.async_request_refresh()

    async def async_room_added(self, room: dict) -> None:
        """Create entity platform entities for a newly added/updated room and refresh data."""
        area_id = room["area_id"]
        has_covers = bool(room.get("covers"))

        if area_id not in self._entity_areas and hasattr(self, "async_add_entities") and self.async_add_entities:
            from .sensor import _create_room_entities

            entities = _create_room_entities(self, area_id)
            self.async_add_entities(entities)
            self._entity_areas.add(area_id)

        if (
            area_id not in self._climate_entity_areas
            and hasattr(self, "async_add_climate_entities")
            and self.async_add_climate_entities
        ):
            from .climate import _create_room_climates

            self.async_add_climate_entities(_create_room_climates(self, area_id))
            self._climate_entity_areas.add(area_id)

        if (
            area_id not in self._climate_control_switch_areas
            and hasattr(self, "async_add_switch_entities")
            and self.async_add_switch_entities
        ):
            from .switch import RoomMindClimateControlSwitch

            self.async_add_switch_entities([RoomMindClimateControlSwitch(self, area_id)])
            self._climate_control_switch_areas.add(area_id)

        if has_covers:
            if (
                area_id not in self._switch_entity_areas
                and hasattr(self, "async_add_switch_entities")
                and self.async_add_switch_entities
            ):
                from .switch import _create_room_switches

                self.async_add_switch_entities(_create_room_switches(self, area_id))
                self._switch_entity_areas.add(area_id)
            if (
                area_id not in self._binary_sensor_entity_areas
                and hasattr(self, "async_add_binary_sensor_entities")
                and self.async_add_binary_sensor_entities
            ):
                from .binary_sensor import _create_room_binary_sensors

                self.async_add_binary_sensor_entities(_create_room_binary_sensors(self, area_id))
                self._binary_sensor_entity_areas.add(area_id)

        if (
            has_air_quality_sensor(room)
            and area_id not in self._air_quality_entity_areas
            and self.async_add_binary_sensor_entities
        ):
            from .binary_sensor import RoomMindAirQualitySensor

            self.async_add_binary_sensor_entities([RoomMindAirQualitySensor(self, area_id)])
            self._air_quality_entity_areas.add(area_id)

        await self.async_request_refresh()

    async def async_room_removed(self, area_id: str) -> None:
        """Remove a deleted room's entities, run state, model and history, then refresh."""
        self._forget_room(area_id)
        self._model_manager.remove_room(area_id)
        self._occupancy_manager.remove_room(area_id)
        if self._history_store:
            await self.hass.async_add_executor_job(self._history_store.remove_room, area_id)
        await self.async_request_refresh()

    async def async_room_paused(self, area_id: str) -> None:
        """Let a room whose area was deleted rest; entities and run state go, model and history stay."""
        self._forget_room(area_id)

    def note_room_deleted(self, area_id: str, room: dict) -> bool:
        """Remember a room deleted in the panel so the plant stops serving it.

        The Central Plant owns the hardware, so there is no room device to switch
        off; the room's contribution simply disappears with the next cycle. Returns
        whether the room was one the plant served.
        """
        if room.get("is_outdoor"):
            return False
        served = bool(room.get("climate_control_enabled", True))
        self._forget_room(area_id)
        return served

    def cleanup_orphaned_entities(self) -> None:
        """Remove entities that no longer match any registered entity type."""
        from homeassistant.helpers import entity_registry as er

        store = self.hass.data[DOMAIN]["store"]
        rooms = rooms_with_area(self.hass, store.get_rooms())
        registry = er.async_get(self.hass)

        global_uids = GLOBAL_ENTITY_UNIQUE_IDS

        to_remove: list[str] = []
        for entity_entry in registry.entities.values():
            if entity_entry.platform != DOMAIN:
                continue
            uid = entity_entry.unique_id
            if not isinstance(uid, str) or not uid.startswith(f"{DOMAIN}_"):
                continue
            if uid in global_uids:
                continue
            parts = uid.removeprefix(f"{DOMAIN}_")
            match = _match_room_entity(parts, rooms)
            if match is None:
                to_remove.append(entity_entry.entity_id)
                continue
            area_id, suffix = match
            if suffix in COVER_ENTITY_SUFFIXES and not rooms[area_id].get("covers"):
                to_remove.append(entity_entry.entity_id)
            elif suffix in AIR_QUALITY_ENTITY_SUFFIXES and not has_air_quality_sensor(rooms[area_id]):
                to_remove.append(entity_entry.entity_id)

        for entity_id in to_remove:
            _LOGGER.info("Removing orphaned entity: %s", entity_id)
            registry.async_remove(entity_id)

    def reset_thermal_room(self, area_id: str) -> None:
        """Reset the thermal model and the forced-air charge of one room."""
        self._model_manager.remove_room(area_id)
        self._ekf_training.last_temps.pop(area_id, None)
        self._rebound_tracker.remove_room(area_id)

    def reset_occupancy_room(self, area_id: str) -> None:
        """Reset one room's learned occupancy trend, keeping every other room."""
        self._occupancy_manager.reset_room(area_id)
        room = self.hass.data[DOMAIN]["store"].get_room(area_id)
        if room is not None:
            self.hass.async_create_task(
                self.hass.data[DOMAIN]["store"].async_save_occupancy_trends(self._occupancy_manager.to_dict())
            )

    def reset_thermal_all(self) -> list[str]:
        """Reset every thermal model. Returns the affected room ids."""
        room_ids = self._model_manager.get_room_ids()
        self._model_manager = RoomModelManager()
        self._ekf_training.set_model_manager(self._model_manager)
        self._cover_orchestrator.set_model_manager(self._model_manager)
        self._ekf_training.last_temps.clear()
        self._rebound_tracker.reset()
        return room_ids

    def boost_learning(self, area_id: str) -> int | None:
        """Boost EKF covariance for a room. Returns n_observations, None without a model."""
        return self._model_manager.boost_learning(area_id)

    def clear_cover_override(self, area_id: str) -> None:
        """Clear a user cover override so automatic cover control resumes."""
        self._cover_orchestrator.clear_user_override(area_id)

    # ------------------------------------------------------------------ #
    # In-memory cleanup
    # ------------------------------------------------------------------ #

    def _forget_room(self, area_id: str) -> None:
        """Drop a room's entities and in-memory run state."""
        from homeassistant.helpers import entity_registry as er

        registry = er.async_get(self.hass)
        owned_uids = {f"{DOMAIN}_{area_id}{suffix}" for suffix in ROOM_ENTITY_SUFFIXES}
        entries_to_remove = [
            entity_entry.entity_id
            for entity_entry in registry.entities.values()
            if entity_entry.platform == DOMAIN
            and isinstance(entity_entry.unique_id, str)
            and entity_entry.unique_id in owned_uids
        ]
        for entity_id in entries_to_remove:
            registry.async_remove(entity_id)

        self._window_manager.remove_room(area_id)
        self._door_manager.remove_room(area_id)
        self._occupancy_manager.remove_room(area_id)
        self._thermostat_overrides.reset_baseline(area_id)
        self._previous_modes.pop(area_id, None)
        self._override_seeded_at.pop(area_id, None)
        self._last_valid_temps.pop(area_id, None)
        self._recent_temps.forget(area_id)
        self._had_valid_temp.discard(area_id)
        self._startup_guard_warned.discard(area_id)
        self._block_temp_warned = {key for key in self._block_temp_warned if key[0] != area_id}
        self._ekf_training.remove_room(area_id)
        self._model_inputs.pop(area_id, None)
        self._rebound_tracker.remove_room(area_id)
        self._cover_orchestrator.remove_room(area_id)
        self._entity_areas.discard(area_id)
        self._mode_on_since.pop(area_id, None)
        self._switch_entity_areas.discard(area_id)
        self._climate_control_switch_areas.discard(area_id)
        self._binary_sensor_entity_areas.discard(area_id)
        self._air_quality_entity_areas.discard(area_id)
        self._air_quality.remove_room(area_id)
        self._climate_entity_areas.discard(area_id)

    # ------------------------------------------------------------------ #
    # Plant live state
    # ------------------------------------------------------------------ #

    def _plant_live_dict(
        self,
        *,
        plant: dict,
        action: str,
        display_action: str,
        climate_mode: str,
        guard_state: PlantGuardState,
        exterior_gate: ExteriorGateResult,
        wet_coil_locked: bool,
        house_ref_temp: float | None,
        thermostat_entity: str,
        thermostat_temp_c: float | None,
        plan: CentralMPCPlan,
        startup_waiting: bool,
        all_force_off: bool,
        any_override: bool,
        fan_circulating: bool,
        actuation: ActuationResult | None,
        observed_hvac_action: str | None,
        plant_override: dict,
    ) -> dict[str, Any]:
        """The live whole-plant state the entities, the panel and the websocket API read."""
        override_active = is_plant_override_active(plant_override)
        return {
            # ``action`` is the plant action the whole-home optimizer chose and
            # ``display_action`` what the equipment reports (or the command while
            # the integration has no hvac_action feedback).
            "configured": bool(thermostat_entity),
            "action": action,
            "commanded_action": action,
            "display_action": display_action,
            "observed_hvac_action": observed_hvac_action,
            "fan_circulating": fan_circulating,
            "climate_mode": climate_mode,
            "thermostat_entity": thermostat_entity,
            "thermostat_area_id": str(plant.get("thermostat_area_id", "")),
            "thermostat_temp": thermostat_temp_c,
            "house_ref_temp": house_ref_temp,
            # The bounded setpoints the actuator holds on the wall thermostat.
            "commanded_low": actuation.target_temp_low_c if actuation is not None else None,
            "commanded_high": actuation.target_temp_high_c if actuation is not None else None,
            "commanded_single": actuation.target_temp_single_c if actuation is not None else None,
            "legal_actions": sorted(guard_state.legal_actions),
            "forced_action": guard_state.forced_action,
            "guard_active": guard_state.guard_active,
            "guard_reason": guard_state.guard_reason,
            "guard_until": guard_state.guard_until,
            "paused": exterior_gate.paused,
            "blocking_rooms": exterior_gate.blocking_rooms,
            "exterior_paused_rooms": exterior_gate.blocking_rooms,
            "wet_coil_locked": wet_coil_locked,
            "wet_coil_charge": round(self._wet_coil.charge, 3),
            "plant_override_active": override_active,
            "plant_override_heat": plant_override.get("plant_override_heat") if override_active else None,
            "plant_override_cool": plant_override.get("plant_override_cool") if override_active else None,
            "plant_override_until": plant_override.get("plant_override_until") if override_active else None,
            "decision_reason": plan.decision_reason,
            "mpc_active": plan.mpc_active,
            "driving_rooms": plan.driving_rooms,
            "preheat": plan.preheat_info,
            "actions": plan.actions,
            "outdoor_temp": self.outdoor_temp_effective,
            "outdoor_source": self.outdoor_temp_source,
            "startup_waiting": startup_waiting,
            "all_force_off": all_force_off,
            "any_override": any_override,
            "price_level": self._price_manager.signal.current_level if self._price_manager.signal else None,
            "plant": dict(plant),
        }

    @staticmethod
    def _observed_hvac_action(thermostat_state: Any) -> str | None:
        """The thermostat's raw ``hvac_action`` string, for the panel and diagnostics."""
        attributes = getattr(thermostat_state, "attributes", None)
        if not isinstance(attributes, dict):
            return None
        action = attributes.get("hvac_action")
        return None if action is None else str(action)
