"""Detect setpoint changes made on the physical Central Thermostat.

The wall thermostat is also a normal climate entity that a person can walk up
to and turn.  RoomMind records every setpoint it sends, so the echo of its own
command is not mistaken for a manual change, and keeps the last setpoint it
observed, so the first reading after startup or after the thermostat was
unavailable never looks like a change either.

Home Assistant state events carry no trace of who changed a setpoint, so the
judgement is by value and time:

* a reading that differs from the previous one by more than the device's step
  is a candidate;
* a candidate that matches a setpoint RoomMind sent within
  ``ECHO_GRACE_SECONDS`` is RoomMind's own echo;
* everything else was changed by a person, a device program, or an automation
  and becomes a Central Plant override until its configured expiry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant

from ..utils.temp_utils import ha_temp_to_celsius

ECHO_GRACE_SECONDS = 75.0

_UNAVAILABLE_STATES = ("unavailable", "unknown")


def _as_float(value: Any) -> float | None:
    """Return *value* as a float, or None when it is missing/not numeric."""
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _unit_is_fahrenheit(hass: HomeAssistant, attributes: dict[str, Any]) -> bool:
    """True when the thermostat reports its temperatures in °F."""
    unit = attributes.get("unit_of_measurement")
    if unit is not None:
        return str(unit) == "°F"
    return bool(hass.config.units.temperature_unit == UnitOfTemperature.FAHRENHEIT)


def _to_celsius(hass: HomeAssistant, value: Any, entity_id: str) -> float | None:
    number = _as_float(value)
    if number is None:
        return None
    return ha_temp_to_celsius(hass, number, entity_id=entity_id)


@dataclass
class _ObservedSetpoint:
    """The thermostat's setpoint as last observed, in Celsius."""

    single_c: float | None
    low_c: float | None
    high_c: float | None


@dataclass
class _SentSetpoint:
    """A setpoint RoomMind sent, in Celsius, with the time it was sent."""

    at: float
    single_c: float | None
    low_c: float | None
    high_c: float | None
    hvac_mode: str | None


class ThermostatOverrideTracker:
    """Tells manual wall-thermostat setpoint changes apart from RoomMind echoes."""

    def __init__(self) -> None:
        self._sent: dict[str, _SentSetpoint] = {}
        self._baseline: dict[str, _ObservedSetpoint] = {}

    def record_sent(
        self,
        entity_id: str,
        *,
        low_c: float | None = None,
        high_c: float | None = None,
        single_c: float | None = None,
        hvac_mode: str | None = None,
        now_ts: float | None = None,
    ) -> None:
        """Remember a setpoint RoomMind just sent, so its echo is recognised."""
        if not entity_id:
            return
        now = time.time() if now_ts is None else now_ts
        self._sent[entity_id] = _SentSetpoint(
            at=now,
            single_c=_as_float(single_c),
            low_c=_as_float(low_c),
            high_c=_as_float(high_c),
            hvac_mode=hvac_mode,
        )

    def reset_baseline(self, entity_id: str) -> None:
        """Forget the last observation: the thermostat was unavailable and may come back changed."""
        self._baseline.pop(entity_id, None)

    def check_external_change(
        self,
        hass: HomeAssistant,
        thermostat_entity: str,
        thermostat_state: Any,
        plant_settings: dict[str, Any],
        house_ref_temp_c: float | None,
        now_ts: float | None = None,
    ) -> dict[str, Any] | None:
        """Return plant-override fields for a manual setpoint change, else None.

        ``house_ref_temp_c`` is accepted for symmetry with the controller's
        cycle data; wall-thermostat setpoints are absolute temperatures and do
        not depend on the House Reference Temperature.
        """
        del house_ref_temp_c

        if not plant_settings.get("follow_thermostat_setpoint", True):
            return None
        if not thermostat_entity or thermostat_state is None:
            return None

        entity_id = str(thermostat_entity)
        state_str = str(getattr(thermostat_state, "state", ""))
        if state_str in _UNAVAILABLE_STATES:
            self.reset_baseline(entity_id)
            return None
        if state_str == "off":
            return None

        raw_attrs = getattr(thermostat_state, "attributes", None)
        attributes: dict[str, Any] = raw_attrs if isinstance(raw_attrs, dict) else {}
        current = _ObservedSetpoint(
            single_c=_to_celsius(hass, attributes.get("temperature"), entity_id),
            low_c=_to_celsius(hass, attributes.get("target_temp_low"), entity_id),
            high_c=_to_celsius(hass, attributes.get("target_temp_high"), entity_id),
        )
        if current.single_c is None and current.low_c is None and current.high_c is None:
            # No setpoint feedback yet (attribute-less integration or the
            # moment after a restart); this is not a valid observation.
            return None

        previous = self._baseline.get(entity_id)
        if previous is None:
            # Startup or recovery from unavailability: this is what the wall
            # thermostat looks like, not a change somebody just made.
            self._baseline[entity_id] = current
            return None

        tolerance_c = self._tolerance(hass, attributes)
        range_changed = (
            current.low_c is not None
            and current.high_c is not None
            and (
                previous.low_c is None
                or previous.high_c is None
                or abs(current.low_c - previous.low_c) > tolerance_c
                or abs(current.high_c - previous.high_c) > tolerance_c
            )
        )
        single_changed = current.single_c is not None and (
            previous.single_c is None or abs(current.single_c - previous.single_c) > tolerance_c
        )

        now = time.time() if now_ts is None else now_ts
        sent = self._sent.get(entity_id)
        if sent is not None and now - sent.at <= ECHO_GRACE_SECONDS and self._matches_sent(sent, current, tolerance_c):
            # Our own command coming back through the state machine.
            self._baseline[entity_id] = current
            return None

        self._baseline[entity_id] = current
        configured_hours = _as_float(plant_settings.get("thermostat_override_hours", 2.0))
        hours = 2.0 if configured_hours is None else configured_hours
        until = (now + hours * 3600.0) if hours > 0 else None

        if range_changed:
            assert current.low_c is not None and current.high_c is not None  # narrowed above
            return {
                "plant_override_heat": round(current.low_c, 2),
                "plant_override_cool": round(current.high_c, 2),
                "plant_override_until": until,
            }
        if single_changed:
            assert current.single_c is not None  # narrowed above
            if state_str == "cool":
                return {
                    "plant_override_heat": None,
                    "plant_override_cool": round(current.single_c, 2),
                    "plant_override_until": until,
                }
            return {
                "plant_override_heat": round(current.single_c, 2),
                "plant_override_cool": None,
                "plant_override_until": until,
            }
        return None

    @staticmethod
    def _tolerance(hass: HomeAssistant, attributes: dict[str, Any]) -> float:
        """Smallest setpoint change that counts, in Celsius.

        Half a device step would miss a whole-degree change on a coarse °F
        thermostat; ``60 %`` catches it while ``max(0.35, ...)`` absorbs cloud
        integrations that round a little.
        """
        fahrenheit = _unit_is_fahrenheit(hass, attributes)
        step_native = _as_float(attributes.get("target_temp_step"))
        if step_native is None or step_native <= 0:
            step_native = 1.0 if fahrenheit else 0.5
        step_c = step_native * 5.0 / 9.0 if fahrenheit else step_native
        return max(0.35, step_c * 0.6)

    @staticmethod
    def _matches_sent(sent: _SentSetpoint, current: _ObservedSetpoint, tolerance_c: float) -> bool:
        """True when the current reading is the echo of the setpoint RoomMind sent."""
        if sent.low_c is not None and sent.high_c is not None:
            return (
                current.low_c is not None
                and current.high_c is not None
                and abs(current.low_c - sent.low_c) <= tolerance_c
                and abs(current.high_c - sent.high_c) <= tolerance_c
            )
        if sent.single_c is not None:
            return current.single_c is not None and abs(current.single_c - sent.single_c) <= tolerance_c
        return False
