"""Bounded relative setpoint nudging for the single Central Thermostat.

The Central Thermostat senses temperature in one hallway or room, so sending a
remote room's target temperature fails whenever the hallway is already
satisfied.  Instead the actuator shifts the thermostat's own setpoint just past
its current reading by a bounded nudge, and parks it in a neutral Fallback Band
during idle and Circulation.  ``hvac_mode: off`` is never used for normal idle:
the thermostat must keep protecting the home on its own if Home Assistant goes
away.

Everything is calculated in Celsius internally and converted to the
thermostat's native unit only for rounding and service calls.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant

from ..const import (
    CLIMATE_MODE_COOL_ONLY,
    MODE_COOLING,
    MODE_HEATING,
    make_roommind_context,
)
from ..managers.thermostat_override_tracker import ThermostatOverrideTracker
from ..utils.temp_utils import celsius_to_ha_temp, ha_temp_to_celsius

_LOGGER = logging.getLogger(__name__)

CLIMATE_MODE_AUTO = "auto"
MODE_CIRCULATING = "circulating"

SETPOINT_DWELL_SECONDS = 300.0
SUPPORT_TARGET_TEMPERATURE_RANGE = 2
UNAVAILABLE_STATES = ("unavailable", "unknown")

DEFAULT_NUDGE_DELTA_C = 1.2
DEFAULT_HEADROOM_C = 2.5
DEFAULT_FALLBACK_HEAT_C = 16.5
DEFAULT_FALLBACK_COOL_C = 26.5
DEFAULT_MIN_DEADBAND_C = 1.7
DEFAULT_DEVICE_MIN_C = 5.0
DEFAULT_DEVICE_MAX_C = 35.0


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _setting(plant_settings: dict[str, Any], key: str, default: float) -> float:
    value = _as_float(plant_settings.get(key))
    return default if value is None else value


def _attributes(state: Any) -> dict[str, Any]:
    raw = getattr(state, "attributes", None)
    return raw if isinstance(raw, dict) else {}


def _modes(value: Any) -> tuple[str, ...]:
    if isinstance(value, list | tuple | set | frozenset):
        return tuple(str(item) for item in value if item is not None)
    return ()


def _device_is_fahrenheit(hass: HomeAssistant, attributes: dict[str, Any]) -> bool:
    unit = attributes.get("unit_of_measurement")
    if unit is not None:
        return str(unit) == "°F"
    return bool(hass.config.units.temperature_unit == UnitOfTemperature.FAHRENHEIT)


def _to_celsius(hass: HomeAssistant, value: Any, entity_id: str) -> float | None:
    number = _as_float(value)
    if number is None:
        return None
    return ha_temp_to_celsius(hass, number, entity_id=entity_id)


def _from_native_celsius(hass: HomeAssistant, value: float, entity_id: str) -> float:
    return ha_temp_to_celsius(hass, value, entity_id=entity_id)


def _to_native(hass: HomeAssistant, value_c: float, fahrenheit: bool) -> float:
    ha_fahrenheit = hass.config.units.temperature_unit == UnitOfTemperature.FAHRENHEIT
    if fahrenheit == ha_fahrenheit:
        return celsius_to_ha_temp(hass, value_c)
    if fahrenheit:
        return value_c * 9.0 / 5.0 + 32.0
    return value_c


def _snap_up(value: float, step: float) -> float:
    return round(math.ceil(value / step - 1e-9) * step, 6)


def _snap_down(value: float, step: float) -> float:
    return round(math.floor(value / step + 1e-9) * step, 6)


def _same(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(left - right) < 1e-6


@dataclass
class ActuationResult:
    """What the actuator wanted and whether it had to send anything."""

    applied: bool
    hvac_mode: str | None
    fan_mode: str | None
    target_temp_low_c: float | None
    target_temp_high_c: float | None
    target_temp_single_c: float | None
    thermostat_temp_c: float | None


@dataclass(frozen=True)
class TargetPlan:
    """The thermostat state the controller wants, in Celsius."""

    hvac_mode: str
    fan_mode: str | None
    low_c: float | None
    high_c: float | None
    single_c: float | None
    thermostat_temp_c: float
    nudge_c: float
    deadband_c: float
    max_heat_c: float
    min_cool_c: float
    heat_capped: bool
    cool_floored: bool


@dataclass
class _EntityCommandCache:
    """Last command sent to one Central Thermostat."""

    last_sent_ts: float = 0.0
    last_action: str | None = None
    last_low_native: float | None = None
    last_high_native: float | None = None
    last_single_native: float | None = None
    last_hvac_mode: str | None = None
    last_fan_mode: str | None = None


class ThermostatActuator:
    """Calculates and applies the bounded relative setpoint for one thermostat."""

    SETPOINT_DWELL_SECONDS = SETPOINT_DWELL_SECONDS

    def __init__(self) -> None:
        self._cache: dict[str, _EntityCommandCache] = {}

    def clear(self) -> None:
        """Forget every per-entity command (tests, integration unload)."""
        self._cache.clear()

    # ------------------------------------------------------------------ #
    # Target calculation
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_targets_c(
        plant_settings: dict[str, Any],
        action: str,
        *,
        climate_mode: str = CLIMATE_MODE_AUTO,
        thermostat_temp_c: float,
        device_min_c: float,
        device_max_c: float,
        step_c: float,
        supports_range: bool,
        hvac_modes: tuple[str, ...] = (),
        has_fan_modes: bool = True,
        current_hvac_state: str = "",
        active_heat_ceiling_c: float | None = None,
        active_cool_floor_c: float | None = None,
        renudge_from_c: float | None = None,
    ) -> TargetPlan:
        """Return the target mode and setpoints in Celsius for one plant action."""
        step = max(step_c, 0.01)
        device_max_c = max(device_max_c, device_min_c)

        nudge_c = max(_setting(plant_settings, "nudge_delta_c", DEFAULT_NUDGE_DELTA_C), step * 1.6)
        headroom_c = _setting(plant_settings, "max_nudge_headroom_c", DEFAULT_HEADROOM_C)
        fallback_heat_c = _setting(plant_settings, "fallback_heat_c", DEFAULT_FALLBACK_HEAT_C)
        fallback_cool_c = _setting(plant_settings, "fallback_cool_c", DEFAULT_FALLBACK_COOL_C)

        if active_heat_ceiling_c is not None:
            max_heat_c = min(device_max_c, active_heat_ceiling_c + headroom_c)
        else:
            max_heat_c = min(device_max_c, fallback_cool_c)
        if active_cool_floor_c is not None:
            min_cool_c = max(device_min_c, active_cool_floor_c - headroom_c)
        else:
            min_cool_c = max(device_min_c, fallback_heat_c)

        idle_low_c = max(device_min_c, min(fallback_heat_c, thermostat_temp_c - nudge_c))
        idle_high_c = min(device_max_c, max(fallback_cool_c, thermostat_temp_c + nudge_c))

        heat_target_c = max(thermostat_temp_c + nudge_c, fallback_heat_c)
        if renudge_from_c is not None:
            # Hallway catch-up: step the nudge ahead of the setpoint already
            # sent, so the thermostat does not cut the run short on its own
            # satisfied sensor.
            heat_target_c = max(heat_target_c, renudge_from_c + nudge_c)
        heat_sp_c = max(device_min_c, min(max_heat_c, heat_target_c))
        heat_capped = max_heat_c < heat_target_c - 1e-9

        cool_target_c = min(thermostat_temp_c - nudge_c, fallback_cool_c)
        if renudge_from_c is not None:
            cool_target_c = min(cool_target_c, renudge_from_c - nudge_c)
        cool_sp_c = min(device_max_c, max(min_cool_c, cool_target_c))
        cool_floored = min_cool_c > cool_target_c + 1e-9

        deadband_c = max(_setting(plant_settings, "min_deadband_c", DEFAULT_MIN_DEADBAND_C), step * 2.0)

        fan_mode: str | None
        if action == MODE_CIRCULATING:
            fan_mode = str(plant_settings.get("circulate_fan_mode", "on"))
        else:
            fan_mode = str(plant_settings.get("idle_fan_mode", "auto"))

        range_path = climate_mode == CLIMATE_MODE_AUTO and supports_range
        fan_only_fallback = action == MODE_CIRCULATING and not has_fan_modes and "fan_only" in hvac_modes
        if fan_only_fallback:
            return TargetPlan(
                "fan_only",
                None,
                None,
                None,
                None,
                thermostat_temp_c,
                nudge_c,
                deadband_c,
                max_heat_c,
                min_cool_c,
                heat_capped,
                cool_floored,
            )

        if range_path:
            if action == MODE_HEATING:
                low_c = heat_sp_c
                high_c = min(device_max_c, max(idle_high_c, low_c + deadband_c))
                if high_c - low_c < deadband_c:
                    low_c = max(device_min_c, high_c - deadband_c)
            elif action == MODE_COOLING:
                high_c = cool_sp_c
                low_c = max(device_min_c, min(idle_low_c, high_c - deadband_c))
                if high_c - low_c < deadband_c:
                    high_c = min(device_max_c, low_c + deadband_c)
            else:
                low_c = idle_low_c
                high_c = min(device_max_c, max(idle_high_c, low_c + deadband_c))
            return TargetPlan(
                "heat_cool",
                fan_mode,
                low_c,
                high_c,
                None,
                thermostat_temp_c,
                nudge_c,
                deadband_c,
                max_heat_c,
                min_cool_c,
                heat_capped,
                cool_floored,
            )

        if action == MODE_HEATING:
            return TargetPlan(
                "heat",
                fan_mode,
                None,
                None,
                heat_sp_c,
                thermostat_temp_c,
                nudge_c,
                deadband_c,
                max_heat_c,
                min_cool_c,
                heat_capped,
                cool_floored,
            )
        if action == MODE_COOLING:
            return TargetPlan(
                "cool",
                fan_mode,
                None,
                None,
                cool_sp_c,
                thermostat_temp_c,
                nudge_c,
                deadband_c,
                max_heat_c,
                min_cool_c,
                heat_capped,
                cool_floored,
            )

        # Idle and Circulation stay in a safe Fallback Band.  Never "off".
        choose_cool = climate_mode == CLIMATE_MODE_COOL_ONLY or (
            current_hvac_state == "cool" and climate_mode == CLIMATE_MODE_AUTO
        )
        if choose_cool:
            target_mode = "cool"
            single_c = idle_high_c
        elif "heat" in hvac_modes:
            target_mode = "heat"
            single_c = idle_low_c
        else:
            target_mode = "cool"
            single_c = idle_high_c
        return TargetPlan(
            target_mode,
            fan_mode,
            None,
            None,
            single_c,
            thermostat_temp_c,
            nudge_c,
            deadband_c,
            max_heat_c,
            min_cool_c,
            heat_capped,
            cool_floored,
        )

    # ------------------------------------------------------------------ #
    # Actuation
    # ------------------------------------------------------------------ #

    async def async_apply(
        self,
        hass: HomeAssistant,
        plant_settings: dict[str, Any],
        action: str,
        *,
        climate_mode: str = "auto",
        house_ref_temp_c: float | None = None,
        active_heat_ceiling_c: float | None = None,
        active_cool_floor_c: float | None = None,
        force_off: bool = False,
        override_tracker: ThermostatOverrideTracker | None = None,
        now_ts: float | None = None,
    ) -> ActuationResult:
        """Apply *action* to the Central Thermostat and report what was sent."""
        entity_id = str(plant_settings.get("thermostat_entity", ""))
        if not entity_id:
            return ActuationResult(False, None, None, None, None, None, None)

        state = hass.states.get(entity_id)
        if state is None or str(getattr(state, "state", "")) in UNAVAILABLE_STATES:
            return ActuationResult(False, None, None, None, None, None, None)

        attributes = _attributes(state)
        fahrenheit = _device_is_fahrenheit(hass, attributes)
        now = time.time() if now_ts is None else now_ts

        thermostat_temp_c = _to_celsius(hass, attributes.get("current_temperature"), entity_id)
        if thermostat_temp_c is None:
            thermostat_temp_c = house_ref_temp_c if house_ref_temp_c is not None else 21.0

        device_min_c = _to_celsius(hass, attributes.get("min_temp"), entity_id)
        if device_min_c is None:
            device_min_c = DEFAULT_DEVICE_MIN_C
        device_max_c = _to_celsius(hass, attributes.get("max_temp"), entity_id)
        if device_max_c is None:
            device_max_c = DEFAULT_DEVICE_MAX_C
        device_max_c = max(device_max_c, device_min_c)

        step_native = _as_float(attributes.get("target_temp_step"))
        if step_native is None or step_native <= 0:
            step_native = 1.0 if fahrenheit else 0.5
        step_c = step_native * 5.0 / 9.0 if fahrenheit else step_native

        hvac_modes = _modes(attributes.get("hvac_modes"))
        fan_modes = _modes(attributes.get("fan_modes"))
        supports_range = (
            "heat_cool" in hvac_modes
            and (_as_int(attributes.get("supported_features")) & SUPPORT_TARGET_TEMPERATURE_RANGE) != 0
        )

        cache = self._cache.setdefault(entity_id, _EntityCommandCache())
        idle_fan_mode = str(plant_settings.get("idle_fan_mode", "auto"))

        if force_off:
            return await self._async_apply_off(
                hass,
                entity_id,
                state,
                attributes,
                hvac_modes,
                fan_modes,
                idle_fan_mode,
                thermostat_temp_c,
                cache,
            )

        ongoing = action in (MODE_HEATING, MODE_COOLING) and cache.last_action == action
        within_dwell = now - cache.last_sent_ts < SETPOINT_DWELL_SECONDS
        suppress_setpoint = False
        renudge_from_c: float | None = None
        if ongoing and within_dwell:
            anchor_native = cache.last_low_native if action == MODE_HEATING else cache.last_high_native
            if anchor_native is None:
                anchor_native = cache.last_single_native
            if anchor_native is None:
                suppress_setpoint = True
            else:
                anchor_c = _from_native_celsius(hass, anchor_native, entity_id)
                if abs(thermostat_temp_c - anchor_c) <= step_c * 1.1:
                    # The hallway sensor caught up with the setpoint we sent;
                    # move the nudge ahead immediately.
                    renudge_from_c = anchor_c
                else:
                    suppress_setpoint = True

        plan = self.compute_targets_c(
            plant_settings,
            action,
            climate_mode=climate_mode,
            thermostat_temp_c=thermostat_temp_c,
            device_min_c=device_min_c,
            device_max_c=device_max_c,
            step_c=step_c,
            supports_range=supports_range,
            hvac_modes=hvac_modes,
            has_fan_modes=bool(fan_modes),
            current_hvac_state=str(getattr(state, "state", "")),
            active_heat_ceiling_c=active_heat_ceiling_c,
            active_cool_floor_c=active_cool_floor_c,
            renudge_from_c=renudge_from_c,
        )

        min_native = _to_native(hass, device_min_c, fahrenheit)
        max_native = _to_native(hass, device_max_c, fahrenheit)
        therm_native = _to_native(hass, thermostat_temp_c, fahrenheit)
        deadband_native = plan.deadband_c * 9.0 / 5.0 if fahrenheit else plan.deadband_c
        deadband_native = max(step_native * 2.0, round(deadband_native) if fahrenheit else deadband_native)

        low_native: float | None = None
        high_native: float | None = None
        single_native: float | None = None
        if plan.low_c is not None and plan.high_c is not None:
            low_native = self._round_native(hass, plan.low_c, fahrenheit, step_native, min_native, max_native)
            high_native = self._round_native(hass, plan.high_c, fahrenheit, step_native, min_native, max_native)
            low_native, high_native = self._enforce_deadband(
                low_native,
                high_native,
                action,
                deadband_native,
                step_native,
                min_native,
                max_native,
            )
        elif plan.single_c is not None:
            single_native = self._round_native(hass, plan.single_c, fahrenheit, step_native, min_native, max_native)

        # Verify in the thermostat's native unit that an active call is at least
        # one whole step past the thermostat's own rounded reading.
        if action == MODE_HEATING and not plan.heat_capped:
            minimum_native = round(therm_native) + step_native
            if single_native is not None and single_native < minimum_native - 1e-9:
                single_native = self._at_least(minimum_native, step_native, min_native, max_native)
            if low_native is not None and low_native < minimum_native - 1e-9:
                low_native = self._at_least(minimum_native, step_native, min_native, max_native)
        elif action == MODE_COOLING and not plan.cool_floored:
            maximum_native = round(therm_native) - step_native
            if single_native is not None and single_native > maximum_native + 1e-9:
                single_native = self._at_most(maximum_native, step_native, min_native, max_native)
            if high_native is not None and high_native > maximum_native + 1e-9:
                high_native = self._at_most(maximum_native, step_native, min_native, max_native)

        if low_native is not None and high_native is not None:
            low_native, high_native = self._enforce_deadband(
                low_native,
                high_native,
                action,
                deadband_native,
                step_native,
                min_native,
                max_native,
            )

        if low_native is not None and high_native is not None:
            current_low = _as_float(attributes.get("target_temp_low"))
            current_high = _as_float(attributes.get("target_temp_high"))
            targets_differ = not _same(current_low, low_native) or not _same(current_high, high_native)
            range_target = True
        elif single_native is not None:
            current_single = _as_float(attributes.get("temperature"))
            targets_differ = not _same(current_single, single_native)
            range_target = False
        else:
            targets_differ = False
            range_target = False

        if suppress_setpoint:
            targets_differ = False

        applied = False
        if str(getattr(state, "state", "")) != plan.hvac_mode:
            if await self._async_call(
                hass,
                "set_hvac_mode",
                {"entity_id": entity_id, "hvac_mode": plan.hvac_mode},
            ):
                applied = True

        if targets_differ:
            # Record before awaiting: the echo state event can arrive while the
            # service call is still in flight.
            if override_tracker is not None:
                if range_target:
                    override_tracker.record_sent(
                        entity_id,
                        low_c=plan.low_c,
                        high_c=plan.high_c,
                        hvac_mode=plan.hvac_mode,
                        now_ts=now,
                    )
                else:
                    override_tracker.record_sent(
                        entity_id,
                        single_c=plan.single_c,
                        hvac_mode=plan.hvac_mode,
                        now_ts=now,
                    )
            data: dict[str, Any]
            if range_target:
                data = {
                    "entity_id": entity_id,
                    "target_temp_low": low_native,
                    "target_temp_high": high_native,
                }
            else:
                data = {"entity_id": entity_id, "temperature": single_native}
            if await self._async_call(hass, "set_temperature", data):
                applied = True
                cache.last_sent_ts = now
                cache.last_low_native = low_native
                cache.last_high_native = high_native
                cache.last_single_native = single_native
        elif not suppress_setpoint and (range_target or single_native is not None):
            # The device already shows the target; keep the catch-up anchor in
            # sync so a later cycle measures against the real setpoint.
            cache.last_low_native = low_native
            cache.last_high_native = high_native
            cache.last_single_native = single_native

        target_fan = plan.fan_mode
        if target_fan is not None and target_fan in fan_modes and attributes.get("fan_mode") != target_fan:
            if await self._async_call(
                hass,
                "set_fan_mode",
                {"entity_id": entity_id, "fan_mode": target_fan},
            ):
                applied = True

        cache.last_action = action
        cache.last_hvac_mode = plan.hvac_mode
        cache.last_fan_mode = target_fan if target_fan in fan_modes else None

        return ActuationResult(
            applied,
            plan.hvac_mode,
            target_fan,
            plan.low_c,
            plan.high_c,
            plan.single_c,
            plan.thermostat_temp_c,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    async def _async_apply_off(
        self,
        hass: HomeAssistant,
        entity_id: str,
        state: Any,
        attributes: dict[str, Any],
        hvac_modes: tuple[str, ...],
        fan_modes: tuple[str, ...],
        idle_fan_mode: str,
        thermostat_temp_c: float,
        cache: _EntityCommandCache,
    ) -> ActuationResult:
        """Force the thermostat off, or leave it alone when it has no off mode."""
        if "off" not in hvac_modes:
            return ActuationResult(False, None, None, None, None, None, thermostat_temp_c)

        applied = False
        if str(getattr(state, "state", "")) != "off":
            if await self._async_call(hass, "set_hvac_mode", {"entity_id": entity_id, "hvac_mode": "off"}):
                applied = True
        target_fan = idle_fan_mode if idle_fan_mode in fan_modes else None
        if target_fan is not None and attributes.get("fan_mode") != target_fan:
            if await self._async_call(
                hass,
                "set_fan_mode",
                {"entity_id": entity_id, "fan_mode": target_fan},
            ):
                applied = True

        cache.last_sent_ts = 0.0
        cache.last_action = "off"
        cache.last_low_native = None
        cache.last_high_native = None
        cache.last_single_native = None
        cache.last_hvac_mode = "off"
        cache.last_fan_mode = target_fan
        return ActuationResult(applied, "off", target_fan, None, None, None, thermostat_temp_c)

    @staticmethod
    def _round_native(
        hass: HomeAssistant,
        value_c: float,
        fahrenheit: bool,
        step_native: float,
        min_native: float,
        max_native: float,
    ) -> float:
        raw = _to_native(hass, value_c, fahrenheit)
        rounded = round(raw / step_native) * step_native
        return round(min(max_native, max(min_native, rounded)), 6)

    @staticmethod
    def _at_least(value: float, step: float, min_native: float, max_native: float) -> float:
        return round(min(max_native, max(min_native, _snap_up(value, step))), 6)

    @staticmethod
    def _at_most(value: float, step: float, min_native: float, max_native: float) -> float:
        return round(min(max_native, max(min_native, _snap_down(value, step))), 6)

    @staticmethod
    def _enforce_deadband(
        low_native: float,
        high_native: float,
        action: str,
        deadband_native: float,
        step_native: float,
        min_native: float,
        max_native: float,
    ) -> tuple[float, float]:
        """Make room for the minimum deadband without leaving the device range."""
        need = deadband_native - (high_native - low_native)
        if need <= 1e-9:
            return low_native, high_native

        if action == MODE_COOLING:
            room_down = low_native - min_native
            move_down = min(need, math.floor(room_down / step_native + 1e-9) * step_native)
            low_native = round(low_native - move_down, 6)
            need = deadband_native - (high_native - low_native)
            if need > 1e-9:
                room_up = max_native - high_native
                move_up = min(need, math.floor(room_up / step_native + 1e-9) * step_native)
                high_native = round(high_native + move_up, 6)
            return low_native, high_native

        room_up = max_native - high_native
        move_up = min(need, math.floor(room_up / step_native + 1e-9) * step_native)
        high_native = round(high_native + move_up, 6)
        need = deadband_native - (high_native - low_native)
        if need > 1e-9:
            room_down = low_native - min_native
            move_down = min(need, math.floor(room_down / step_native + 1e-9) * step_native)
            low_native = round(low_native - move_down, 6)
        return low_native, high_native

    @staticmethod
    async def _async_call(hass: HomeAssistant, service: str, data: dict[str, Any]) -> bool:
        try:
            await hass.services.async_call(
                "climate",
                service,
                data,
                blocking=True,
                context=make_roommind_context(),
            )
            return True
        except Exception:  # noqa: BLE001
            _LOGGER.warning("Central Thermostat call climate.%s(%s) failed", service, data, exc_info=True)
            return False
