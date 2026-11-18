"""Central Plant equipment guards and the Whole-Home Exterior Opening Gate.

The Central Plant is one shared forced-air system operated through a single
thermostat.  The manager keeps the equipment-protection state that outlives a
single control cycle (minimum run, minimum off and changeover timers), reports
which plant actions the controller is currently allowed to choose, and decides
whether open exterior openings should pause the whole home.

All timestamps are wall-clock seconds because the state is persisted across
restarts.  ``action_started_at`` therefore also survives a Home Assistant
restart; a run that was interrupted by the restart keeps honouring its
``min_run`` window instead of short-cycling the compressor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..const import (
    CLIMATE_MODE_COOL_ONLY,
    CLIMATE_MODE_HEAT_ONLY,
    DEFAULT_OUTDOOR_COOLING_MIN,
    DEFAULT_OUTDOOR_HEATING_MAX,
    MODE_COOLING,
    MODE_HEATING,
    MODE_IDLE,
)

# Circulating runs the central blower without firing the furnace or heat pump.
MODE_CIRCULATING = "circulating"

SENSOR_UNAVAILABLE_STATES = ("unavailable", "unknown")
TRANSIENT_HVAC_ACTIONS = ("preheating", "defrosting", "drying")
EXTERIOR_OPEN_POLICY = "pause_unless_door_closed"
EXTERIOR_OPEN_POLICY_EXCLUDE_ROOM = "exclude_room_only"

GUARD_MIN_RUN = "min_run"
GUARD_MIN_OFF = "min_off"
GUARD_MIN_FAN = "min_fan"
GUARD_CHANGEOVER = "changeover"
GUARD_EXTERIOR_OPEN = "exterior_open"
GUARD_WET_COIL = "wet_coil"


def _as_float(value: Any, default: float) -> float:
    """Return *value* as a float, falling back to *default* for missing/corrupt settings."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_timestamp(value: Any) -> float | None:
    """Return a persisted optional timestamp, or None when it is missing/corrupt."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _outdoor_limit(settings: dict[str, Any], plant_settings: dict[str, Any], key: str, default: float) -> float:
    """Resolve one outdoor lockout: global settings, then the plant block, then the default.

    The panel saves ``outdoor_cooling_min``/``outdoor_heating_max`` in the global
    settings, so reading only the Central Plant block pinned both limits to their
    defaults and silently ignored whatever the Caller configured. The plant block
    stays a fallback for data stored by builds that kept them there.
    """
    for source in (settings, plant_settings):
        value = source.get(key)
        if value is not None:
            return _as_float(value, default)
    return default


@dataclass
class ExteriorGateResult:
    """Outcome of the Whole-Home Exterior Opening Gate."""

    paused: bool
    blocking_rooms: list[str]


@dataclass
class PlantGuardState:
    """Plant actions the controller may choose right now, plus why some are blocked."""

    legal_actions: set[str]
    forced_action: str | None
    guard_active: bool
    guard_reason: str | None
    guard_until: float | None
    #: Conditioning directions the outdoor-temperature lockout withheld this cycle.
    #: Reported separately from ``guard_reason``: the lockout is a standing policy
    #: limit, not an equipment protection the plant is waiting out, so it must not
    #: raise the Guard chip while the plant is busy doing something else.
    outdoor_blocked: tuple[str, ...] = ()


class CentralPlantManager:
    """Equipment-protection guards for the single shared Central Plant."""

    def __init__(self) -> None:
        self.active_action: str = MODE_IDLE
        self.action_started_at: float | None = None
        self.last_heat_ended_at: float | None = None
        self.last_cool_ended_at: float | None = None
        self.state_dirty: bool = False

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Return the subset of state that survives a restart."""
        return {
            "active_action": self.active_action,
            "action_started_at": self.action_started_at,
            "last_heat_ended_at": self.last_heat_ended_at,
            "last_cool_ended_at": self.last_cool_ended_at,
            "state_dirty": self.state_dirty,
        }

    def load_state(self, data: dict[str, Any] | None = None) -> None:
        """Restore persisted state, tolerating missing and malformed entries."""
        self.active_action = MODE_IDLE
        self.action_started_at = None
        self.last_heat_ended_at = None
        self.last_cool_ended_at = None
        self.state_dirty = False
        if not isinstance(data, dict):
            return

        action = data.get("active_action")
        if isinstance(action, str) and action in (
            MODE_IDLE,
            MODE_HEATING,
            MODE_COOLING,
            MODE_CIRCULATING,
        ):
            self.active_action = action
        self.action_started_at = _as_timestamp(data.get("action_started_at"))
        self.last_heat_ended_at = _as_timestamp(data.get("last_heat_ended_at"))
        self.last_cool_ended_at = _as_timestamp(data.get("last_cool_ended_at"))
        self.state_dirty = bool(data.get("state_dirty", False))

    # ------------------------------------------------------------------ #
    # Hardware observation
    # ------------------------------------------------------------------ #

    def observe_plant_action(
        self,
        thermostat_state: Any,
        plant_settings: dict[str, Any],
        commanded_action: str,
    ) -> tuple[str | None, bool]:
        """Map the Central Thermostat's reported state onto one plant action.

        Returns ``(observed_action, fan_circulating)``.  ``observed_action`` is
        None while the equipment reports a transient state (preheating, defrost,
        drying) or is unavailable: those readings must never train the thermal
        model.  ``fan_circulating`` is True when the blower is moving air even
        though the plant is not heating or cooling.
        """
        if thermostat_state is None:
            return None, False

        state_str = str(getattr(thermostat_state, "state", ""))
        if state_str in SENSOR_UNAVAILABLE_STATES:
            return None, False

        raw_attrs = getattr(thermostat_state, "attributes", None)
        attrs: dict[str, Any] = raw_attrs if isinstance(raw_attrs, dict) else {}
        hvac_action = attrs.get("hvac_action")
        fan_mode = attrs.get("fan_mode")
        circ_fan_name = str(plant_settings.get("circulate_fan_mode", "on"))

        if hvac_action == "heating":
            return MODE_HEATING, True
        if hvac_action == "cooling":
            return MODE_COOLING, True
        if hvac_action == "fan" or state_str == "fan_only":
            return MODE_CIRCULATING, True
        if hvac_action in ("idle", "off"):
            if fan_mode == circ_fan_name and state_str != "off":
                return MODE_CIRCULATING, True
            return MODE_IDLE, False
        if hvac_action in TRANSIENT_HVAC_ACTIONS:
            # A defrost/preheat/dry cycle says nothing about the room's heat
            # balance; skip EKF training for it.
            return None, False
        if hvac_action is None:
            # Integration without hvac_action feedback: trust what was commanded.
            return commanded_action, commanded_action != MODE_IDLE
        # An unknown hvac_action is treated like missing feedback rather than
        # silently training the model on a value we do not understand.
        return commanded_action, commanded_action != MODE_IDLE

    # ------------------------------------------------------------------ #
    # Whole-Home Exterior Opening Gate
    # ------------------------------------------------------------------ #

    def evaluate_exterior_gate(
        self,
        rooms_config: dict[str, dict[str, Any]],
        window_states: dict[str, bool],
        door_open_states: dict[str, bool],
        rooms_with_door_sensors: set[str],
        plant_settings: dict[str, Any],
        *,
        any_override_active: bool = False,
    ) -> ExteriorGateResult:
        """Decide whether open exterior openings pause the whole home.

        ``pause_always`` pauses while any controlled, non-outdoor room reports an
        open exterior opening.  The default ``pause_unless_door_closed`` is the
        same, except a room with its own interior door sensor whose door is
        confirmed closed is left out: the open window is then a local problem,
        not a whole-home envelope loss.  ``exclude_room_only`` never pauses the
        whole home, and an active override always keeps the plant running.
        """
        policy = str(plant_settings.get("exterior_open_policy", EXTERIOR_OPEN_POLICY))
        if policy == EXTERIOR_OPEN_POLICY_EXCLUDE_ROOM or any_override_active:
            return ExteriorGateResult(False, [])

        blocking: list[str] = []
        for area_id, room in rooms_config.items():
            if not isinstance(room, dict):
                continue
            if not room.get("climate_control_enabled", True):
                continue
            if room.get("is_outdoor", False):
                continue
            if not window_states.get(area_id, False):
                continue
            if (
                policy == EXTERIOR_OPEN_POLICY
                and area_id in rooms_with_door_sensors
                and not door_open_states.get(area_id, True)
            ):
                # Interior door confirmed closed: the open window cannot drive
                # whole-home loss, so it does not pause the plant.
                continue
            blocking.append(area_id)

        return ExteriorGateResult(bool(blocking), sorted(blocking))

    # ------------------------------------------------------------------ #
    # Allowed actions
    # ------------------------------------------------------------------ #

    def allowed_actions(
        self,
        plant_settings: dict[str, Any],
        climate_mode: str,
        outdoor_temp: float | None,
        settings: dict[str, Any],
        *,
        exterior_paused: bool = False,
        wet_coil_locked: bool = False,
        force_off: bool = False,
        any_override: bool = False,
        now_ts: float | None = None,
    ) -> PlantGuardState:
        """Return the plant actions the controller may choose right now."""
        now = time.time() if now_ts is None else now_ts

        if force_off or not settings.get("climate_control_active", True):
            return PlantGuardState({MODE_IDLE}, MODE_IDLE, False, None, None)

        if exterior_paused:
            return PlantGuardState({MODE_IDLE}, MODE_IDLE, True, GUARD_EXTERIOR_OPEN, None)

        can_heat = climate_mode != CLIMATE_MODE_COOL_ONLY
        can_cool = climate_mode != CLIMATE_MODE_HEAT_ONLY

        outdoor_blocked: list[str] = []
        if outdoor_temp is not None and not any_override:
            outdoor_cooling_min = _outdoor_limit(
                settings, plant_settings, "outdoor_cooling_min", DEFAULT_OUTDOOR_COOLING_MIN
            )
            outdoor_heating_max = _outdoor_limit(
                settings, plant_settings, "outdoor_heating_max", DEFAULT_OUTDOOR_HEATING_MAX
            )
            if outdoor_temp < outdoor_cooling_min:
                if can_cool:
                    outdoor_blocked.append(MODE_COOLING)
                can_cool = False
            if outdoor_temp > outdoor_heating_max:
                if can_heat:
                    outdoor_blocked.append(MODE_HEATING)
                can_heat = False

        legal: set[str] = {MODE_IDLE, MODE_CIRCULATING}
        if can_heat:
            legal.add(MODE_HEATING)
        if can_cool:
            legal.add(MODE_COOLING)

        guard_active = False
        guard_reason: str | None = None
        guard_until: float | None = None

        if wet_coil_locked:
            # Evaporating moisture off a wet indoor coil raises indoor humidity;
            # hold the blower off until the coil has drained.
            legal.discard(MODE_CIRCULATING)
            guard_active = True
            guard_reason = GUARD_WET_COIL

        # ----------------------------------------------------------------
        # Active-run holds: once the plant is running, it finishes its
        # minimum run.  A circulating blower may upgrade to real heating or
        # cooling at any time, but may not short-cycle back to idle.
        # ----------------------------------------------------------------
        if (
            self.active_action in (MODE_HEATING, MODE_COOLING)
            and self.action_started_at is not None
            and self.active_action in legal
        ):
            until = self.action_started_at + _as_float(plant_settings.get("min_run_minutes", 10), 10.0) * 60.0
            if now < until:
                return PlantGuardState({self.active_action}, self.active_action, True, GUARD_MIN_RUN, until)

        if self.active_action == MODE_CIRCULATING and self.action_started_at is not None and MODE_CIRCULATING in legal:
            until = self.action_started_at + _as_float(plant_settings.get("min_fan_minutes", 10), 10.0) * 60.0
            if now < until:
                upgrades = legal & {MODE_HEATING, MODE_COOLING}
                legal.discard(MODE_IDLE)
                if upgrades:
                    # A blower already running must never delay real heating or
                    # cooling, but it must not stop before min_fan either.
                    return PlantGuardState(
                        {MODE_CIRCULATING, *upgrades},
                        None,
                        True,
                        GUARD_MIN_FAN,
                        until,
                    )
                return PlantGuardState({MODE_CIRCULATING}, MODE_CIRCULATING, True, GUARD_MIN_FAN, until)

        # ----------------------------------------------------------------
        # Minimum-off cooldown after the plant stopped heating/cooling.
        # ----------------------------------------------------------------
        if self.active_action not in (MODE_HEATING, MODE_COOLING):
            ended = [t for t in (self.last_heat_ended_at, self.last_cool_ended_at) if t is not None]
            last_end = max(ended) if ended else None
            if last_end is not None:
                off_until = last_end + _as_float(plant_settings.get("min_off_minutes", 5), 5.0) * 60.0
                if now < off_until:
                    legal.discard(MODE_HEATING)
                    legal.discard(MODE_COOLING)
                    guard_active = True
                    guard_reason = GUARD_MIN_OFF
                    guard_until = off_until

        # ----------------------------------------------------------------
        # Changeover lockout: a heat pump reversing between heating and
        # cooling must pass through idle and wait out the changeover time.
        # ----------------------------------------------------------------
        changeover_s = _as_float(plant_settings.get("changeover_minutes", 30), 30.0) * 60.0
        if self.active_action == MODE_HEATING:
            legal.discard(MODE_COOLING)
        elif self.active_action == MODE_COOLING:
            legal.discard(MODE_HEATING)

        if self.last_heat_ended_at is not None:
            heat_until = self.last_heat_ended_at + changeover_s
            if now < heat_until:
                legal.discard(MODE_COOLING)
                if guard_reason is None:
                    guard_active = True
                    guard_reason = GUARD_CHANGEOVER
                    guard_until = heat_until
        if self.last_cool_ended_at is not None:
            cool_until = self.last_cool_ended_at + changeover_s
            if now < cool_until:
                legal.discard(MODE_HEATING)
                if guard_reason is None:
                    guard_active = True
                    guard_reason = GUARD_CHANGEOVER
                    guard_until = cool_until

        return PlantGuardState(legal, None, guard_active, guard_reason, guard_until, tuple(outdoor_blocked))

    # ------------------------------------------------------------------ #
    # Command bookkeeping
    # ------------------------------------------------------------------ #

    def record_commanded_action(self, action: str, now_ts: float | None = None) -> None:
        """Record the action the controller commanded this cycle.

        A repeated command keeps the original start time so ``min_run`` and
        ``min_fan`` measure the real run, not the time since the last cycle.
        Leaving heating or cooling stamps the matching end time for the
        minimum-off and changeover guards.
        """
        now = time.time() if now_ts is None else now_ts

        if action == self.active_action:
            if self.action_started_at is None:
                self.action_started_at = now
                self.state_dirty = True
            return

        previous = self.active_action
        if previous == MODE_HEATING:
            self.last_heat_ended_at = now
        elif previous == MODE_COOLING:
            self.last_cool_ended_at = now

        self.active_action = action
        self.action_started_at = now
        self.state_dirty = True
