"""Central Thermostat HVAC state helpers for RoomMind."""

from __future__ import annotations

from typing import Any

from .temp_utils import ha_temp_to_celsius

# HVAC modes that indicate a device can actually heat, cool or swing.
_ACTIVE_HVAC_MODES = {"heat", "cool", "heat_cool", "auto"}

# Bit flags mirrored from homeassistant.components.climate.ClimateEntityFeature.
TARGET_TEMPERATURE_RANGE_FLAG = 2
FAN_MODE_FLAG = 8


def has_reliable_hvac_modes(state: Any) -> bool:
    """Check if a device's hvac_modes attribute is trustworthy.

    Some integrations report only limited modes like ``["off", "fan_only"]``
    regardless of device state, hiding actual capabilities.
    """
    if state is None:
        return False
    modes = set(state.attributes.get("hvac_modes") or [])
    return bool(modes & _ACTIVE_HVAC_MODES)


def supports_target_range(state: Any) -> bool:
    """Return True when the state supports a heating/cooling target band."""
    if state is None:
        return False
    if "heat_cool" not in (state.attributes.get("hvac_modes") or []):
        return False
    supported = int(state.attributes.get("supported_features") or 0)
    return (supported & TARGET_TEMPERATURE_RANGE_FLAG) != 0


def supported_fan_modes(state: Any) -> list[str]:
    """Return the fan modes supported by the state."""
    if state is None:
        return []
    return [str(m) for m in (state.attributes.get("fan_modes") or [])]


def supports_fan_modes(state: Any) -> bool:
    """Return True when the state advertises at least one fan mode."""
    return bool(supported_fan_modes(state))


def supports_fan_only_hvac_mode(state: Any) -> bool:
    """Return True when the state can run in fan_only mode."""
    if state is None:
        return False
    return "fan_only" in (state.attributes.get("hvac_modes") or [])


def is_fahrenheit_state(hass: Any, state: Any | None = None) -> bool:
    """Return True when the temperature unit is Fahrenheit."""
    if state is not None:
        unit = state.attributes.get("unit_of_measurement") or state.attributes.get("temperature_unit")
        if unit is not None:
            return str(unit) in ("°F", "F")
    units = getattr(getattr(hass, "config", None), "units", None)
    unit = getattr(units, "temperature_unit", None)
    if unit is None:
        return False
    return str(unit) in ("°F", "F")


def thermostat_step_native(hass: Any, state: Any) -> float:
    """Return the thermostat's native target step in Celsius.

    Falls back to 1.0 for Fahrenheit-configured units and 0.5 for Celsius.
    """
    step: float | None = None
    if state is not None:
        raw_step = state.attributes.get("target_temp_step")
        if raw_step is not None:
            try:
                candidate = float(raw_step)
            except (TypeError, ValueError):
                candidate = None
            if candidate is not None and candidate > 0:
                step = candidate
    if step is not None:
        return float(step)
    return 1.0 if is_fahrenheit_state(hass, state) else 0.5


def round_to_step(value: float, step: float) -> float:
    """Round value onto a step grid, cleanly and without float drift."""
    if step <= 0:
        return round(value, 1)
    res = round(round(value / step) * step, 4)
    if step >= 1.0 and step == int(step):
        return float(round(res))
    return round(res, 2)


def thermostat_bounds_c(
    hass: Any,
    state: Any,
    default_min_c: float = 7.0,
    default_max_c: float = 35.0,
) -> tuple[float, float]:
    """Return the (min, max) target temperature band in Celsius."""
    lo = default_min_c
    hi = default_max_c
    if state is not None:
        raw_min = state.attributes.get("min_temp")
        raw_max = state.attributes.get("max_temp")
        if raw_min is not None:
            lo = ha_temp_to_celsius(hass, float(raw_min))
        if raw_max is not None:
            hi = ha_temp_to_celsius(hass, float(raw_max))
    if lo < hi:
        return (float(lo), float(hi))
    return (float(default_min_c), float(default_max_c))
