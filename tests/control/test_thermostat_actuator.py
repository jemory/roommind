"""Tests for the bounded relative setpoint actuator on the Central Thermostat."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.const import UnitOfTemperature

from custom_components.roommind.const import (
    CLIMATE_MODE_COOL_ONLY,
    CLIMATE_MODE_HEAT_ONLY,
    MODE_COOLING,
    MODE_HEATING,
    MODE_IDLE,
)
from custom_components.roommind.control.thermostat_actuator import (
    MODE_CIRCULATING,
    SETPOINT_DWELL_SECONDS,
    ActuationResult,
    TargetPlan,
    ThermostatActuator,
    _EntityCommandCache,
    _snap_down,
    _snap_up,
)
from custom_components.roommind.managers.thermostat_override_tracker import ThermostatOverrideTracker

EID = "climate.central"
RANGE_MODES = ["heat_cool", "heat", "cool", "off", "fan_only"]
SINGLE_HEAT_MODES = ["heat", "off"]
SINGLE_COOL_MODES = ["cool", "off"]


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


def _hass(unit: str = "°C") -> MagicMock:
    hass = MagicMock()
    hass.config.units.temperature_unit = UnitOfTemperature.FAHRENHEIT if unit == "°F" else UnitOfTemperature.CELSIUS
    hass.services.async_call = AsyncMock()
    hass.states.get = MagicMock(return_value=None)
    return hass


def _state(
    *,
    state: str = "heat_cool",
    unit: str = "°C",
    current: float | None = 20.0,
    min_temp: float | None = 5.0,
    max_temp: float | None = 35.0,
    step: Any = 0.5,
    hvac_modes: list[str] | None = None,
    fan_modes: list[str] | None = None,
    fan_mode: str = "auto",
    supported_features: int = 3,
    **extra: Any,
) -> SimpleNamespace:
    attributes: dict[str, Any] = {"unit_of_measurement": unit}
    if current is not None:
        attributes["current_temperature"] = current
    if min_temp is not None:
        attributes["min_temp"] = min_temp
    if max_temp is not None:
        attributes["max_temp"] = max_temp
    if step is not None:
        attributes["target_temp_step"] = step
    if hvac_modes is not None:
        attributes["hvac_modes"] = hvac_modes
    if fan_modes is not None:
        attributes["fan_modes"] = fan_modes
    attributes["fan_mode"] = fan_mode
    attributes["supported_features"] = supported_features
    attributes.update(extra)
    return SimpleNamespace(state=state, attributes=attributes)


def _install(hass: MagicMock, state: SimpleNamespace) -> SimpleNamespace:
    hass.states.get = MagicMock(return_value=state)
    return state


def _settings(**overrides: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {"thermostat_entity": EID}
    settings.update(overrides)
    return settings


async def _apply(
    actuator: ThermostatActuator,
    hass: MagicMock,
    state: SimpleNamespace,
    action: str,
    *,
    settings: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ActuationResult:
    _install(hass, state)
    return await actuator.async_apply(hass, settings or _settings(), action, **kwargs)


def _service_data(hass: MagicMock, service: str) -> list[dict[str, Any]]:
    return [call.args[2] for call in hass.services.async_call.call_args_list if call.args[1] == service]


def _services(hass: MagicMock) -> list[str]:
    return [call.args[1] for call in hass.services.async_call.call_args_list]


def _ecobee_state(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "state": "heat_cool",
        "unit": "°F",
        "current": 68.0,
        "min_temp": 45.0,
        "max_temp": 95.0,
        "step": 1.0,
        "hvac_modes": RANGE_MODES,
        "fan_modes": ["auto", "on"],
        "fan_mode": "auto",
        "supported_features": 3,
    }
    values.update(overrides)
    return _state(**values)


def _hass_f() -> MagicMock:
    return _hass("°F")


# --------------------------------------------------------------------------- #
# compute_targets_c
# --------------------------------------------------------------------------- #


class TestComputeTargets:
    @staticmethod
    def _plan(actuator: ThermostatActuator, action: str, **overrides: Any) -> TargetPlan:
        kwargs: dict[str, Any] = {
            "climate_mode": "auto",
            "thermostat_temp_c": 20.0,
            "device_min_c": 5.0,
            "device_max_c": 35.0,
            "step_c": 0.5,
            "supports_range": True,
            "hvac_modes": tuple(RANGE_MODES),
            "has_fan_modes": True,
            "current_hvac_state": "heat_cool",
        }
        kwargs.update(overrides)
        return actuator.compute_targets_c({}, action, **kwargs)

    def test_range_heating_sets_low_one_nudge_above_the_hallway(self) -> None:
        plan = self._plan(ThermostatActuator(), MODE_HEATING)
        assert plan.hvac_mode == "heat_cool"
        assert plan.low_c == pytest.approx(21.2)
        assert plan.high_c == pytest.approx(26.5)
        assert plan.single_c is None
        assert plan.heat_capped is False
        assert plan.deadband_c == pytest.approx(1.7)

    def test_range_cooling_sets_high_one_nudge_below_the_hallway(self) -> None:
        plan = self._plan(ThermostatActuator(), MODE_COOLING, thermostat_temp_c=24.0)
        assert plan.hvac_mode == "heat_cool"
        assert plan.high_c == pytest.approx(22.8)
        assert plan.low_c == pytest.approx(16.5)
        assert plan.cool_floored is False

    def test_range_idle_parks_in_the_fallback_band(self) -> None:
        plan = self._plan(ThermostatActuator(), MODE_IDLE)
        assert plan.hvac_mode == "heat_cool"
        assert plan.low_c == pytest.approx(16.5)
        assert plan.high_c == pytest.approx(26.5)

    def test_range_circulation_is_an_idle_band_with_the_circulation_fan(self) -> None:
        plan = self._plan(ThermostatActuator(), MODE_CIRCULATING)
        assert plan.hvac_mode == "heat_cool"
        assert plan.low_c == pytest.approx(16.5)
        assert plan.high_c == pytest.approx(26.5)
        assert plan.fan_mode == "on"

    def test_heat_is_capped_by_the_room_target_plus_headroom(self) -> None:
        plan = self._plan(ThermostatActuator(), MODE_HEATING, thermostat_temp_c=24.0, active_heat_ceiling_c=18.0)
        assert plan.max_heat_c == pytest.approx(20.5)
        assert plan.low_c == pytest.approx(20.5)
        assert plan.heat_capped is True

    def test_cooling_is_floored_by_the_room_target_minus_headroom(self) -> None:
        plan = self._plan(ThermostatActuator(), MODE_COOLING, thermostat_temp_c=20.0, active_cool_floor_c=24.0)
        assert plan.min_cool_c == pytest.approx(21.5)
        assert plan.high_c == pytest.approx(21.5)
        assert plan.cool_floored is True

    def test_nudge_never_shrinks_below_two_whole_fahrenheit_steps(self) -> None:
        plan = self._plan(ThermostatActuator(), MODE_HEATING, step_c=1.0)
        # max(1.2 °C, 1.6 °C)
        assert plan.nudge_c == pytest.approx(1.6)
        assert plan.deadband_c == pytest.approx(2.0)

    def test_custom_settings_are_honoured(self) -> None:
        actuator = ThermostatActuator()
        settings = {
            "nudge_delta_c": 0.8,
            "max_nudge_headroom_c": 3.0,
            "fallback_heat_c": 18.0,
            "fallback_cool_c": 25.0,
            "min_deadband_c": 2.2,
        }
        plan = actuator.compute_targets_c(
            settings,
            MODE_IDLE,
            thermostat_temp_c=20.0,
            device_min_c=5.0,
            device_max_c=35.0,
            step_c=0.5,
            supports_range=True,
            hvac_modes=tuple(RANGE_MODES),
        )
        assert plan.low_c == pytest.approx(18.0)
        assert plan.high_c == pytest.approx(25.0)

    def test_single_heat_and_cool_targets(self) -> None:
        actuator = ThermostatActuator()
        heat = actuator.compute_targets_c(
            {},
            MODE_HEATING,
            thermostat_temp_c=19.0,
            device_min_c=5.0,
            device_max_c=35.0,
            step_c=0.5,
            supports_range=False,
            hvac_modes=tuple(SINGLE_HEAT_MODES),
            has_fan_modes=False,
        )
        assert heat.hvac_mode == "heat"
        assert heat.single_c == pytest.approx(20.2)
        cool = actuator.compute_targets_c(
            {},
            MODE_COOLING,
            thermostat_temp_c=25.0,
            device_min_c=5.0,
            device_max_c=35.0,
            step_c=0.5,
            supports_range=False,
            hvac_modes=tuple(SINGLE_COOL_MODES),
            has_fan_modes=False,
        )
        assert cool.hvac_mode == "cool"
        assert cool.single_c == pytest.approx(23.8)

    def test_single_idle_prefers_heat_when_the_device_offers_it(self) -> None:
        plan = self._plan(
            ThermostatActuator(),
            MODE_IDLE,
            supports_range=False,
            hvac_modes=tuple(SINGLE_HEAT_MODES),
        )
        assert plan.hvac_mode == "heat"
        assert plan.single_c == pytest.approx(16.5)

    def test_single_idle_keeps_a_cooling_device_cooling(self) -> None:
        plan = self._plan(
            ThermostatActuator(),
            MODE_IDLE,
            supports_range=False,
            hvac_modes=tuple(SINGLE_COOL_MODES),
            current_hvac_state="cool",
        )
        assert plan.hvac_mode == "cool"
        assert plan.single_c == pytest.approx(26.5)

    def test_cool_only_idle_always_uses_the_cooling_side(self) -> None:
        plan = self._plan(
            ThermostatActuator(),
            MODE_IDLE,
            climate_mode=CLIMATE_MODE_COOL_ONLY,
            supports_range=False,
            hvac_modes=("cool",),
            current_hvac_state="heat",
        )
        assert plan.hvac_mode == "cool"
        assert plan.single_c == pytest.approx(26.5)

    def test_heat_only_idle_without_heat_falls_back_to_cooling(self) -> None:
        plan = self._plan(
            ThermostatActuator(),
            MODE_IDLE,
            climate_mode=CLIMATE_MODE_HEAT_ONLY,
            supports_range=False,
            hvac_modes=("cool",),
        )
        assert plan.hvac_mode == "cool"

    def test_circulation_without_fan_modes_uses_fan_only(self) -> None:
        plan = self._plan(
            ThermostatActuator(),
            MODE_CIRCULATING,
            hvac_modes=("heat_cool", "fan_only"),
            has_fan_modes=False,
        )
        assert plan.hvac_mode == "fan_only"
        assert plan.fan_mode is None
        assert plan.low_c is None and plan.high_c is None and plan.single_c is None

    def test_range_heating_reduces_the_low_side_when_the_device_max_clamps(self) -> None:
        plan = self._plan(
            ThermostatActuator(),
            MODE_HEATING,
            device_min_c=21.0,
            device_max_c=21.5,
            active_heat_ceiling_c=None,
        )
        assert plan.low_c == pytest.approx(21.0)
        assert plan.high_c == pytest.approx(21.5)

    def test_range_cooling_raises_the_high_side_when_the_device_min_clamps(self) -> None:
        plan = self._plan(
            ThermostatActuator(),
            MODE_COOLING,
            thermostat_temp_c=28.0,
            device_min_c=27.0,
            device_max_c=27.5,
            active_cool_floor_c=None,
        )
        assert plan.low_c == pytest.approx(27.0)
        assert plan.high_c == pytest.approx(27.5)


# --------------------------------------------------------------------------- #
# Ecobee-like whole-degree °F heat_cool thermostat
# --------------------------------------------------------------------------- #


class TestEcobeeHeatCool:
    @pytest.mark.asyncio
    async def test_heating_nudges_two_degrees_fahrenheit(self) -> None:
        hass = _hass_f()
        state = _ecobee_state()
        result = await _apply(ThermostatActuator(), hass, state, MODE_HEATING, now_ts=1000.0)

        assert result.applied is True
        assert result.hvac_mode == "heat_cool"
        assert result.target_temp_low_c == pytest.approx(21.2)
        assert result.target_temp_high_c == pytest.approx(26.5)
        # The hallway reads 68 °F; the setpoint must be 70 °F, a 2 °F nudge.
        assert _service_data(hass, "set_temperature") == [
            {"entity_id": EID, "target_temp_low": 70.0, "target_temp_high": 80.0}
        ]
        assert _services(hass) == ["set_temperature"]

    @pytest.mark.asyncio
    async def test_three_degree_fahrenheit_deadband(self) -> None:
        hass = _hass_f()
        state = _ecobee_state()
        settings = _settings(fallback_cool_c=22.9)
        await _apply(ThermostatActuator(), hass, state, MODE_HEATING, settings=settings, now_ts=1000.0)
        sent = _service_data(hass, "set_temperature")[0]
        assert sent["target_temp_high"] - sent["target_temp_low"] == 3.0

    @pytest.mark.asyncio
    async def test_idle_parks_in_the_fallback_band_and_never_turns_off(self) -> None:
        hass = _hass_f()
        state = _ecobee_state()
        result = await _apply(ThermostatActuator(), hass, state, MODE_IDLE, now_ts=1000.0)

        assert result.applied is True
        assert _service_data(hass, "set_temperature") == [
            {"entity_id": EID, "target_temp_low": 62.0, "target_temp_high": 80.0}
        ]
        assert "off" not in [data.get("hvac_mode") for data in _service_data(hass, "set_hvac_mode")]

    @pytest.mark.asyncio
    async def test_circulation_uses_the_circulation_fan(self) -> None:
        hass = _hass_f()
        state = _ecobee_state()
        await _apply(ThermostatActuator(), hass, state, MODE_CIRCULATING, now_ts=1000.0)
        assert _service_data(hass, "set_fan_mode") == [{"entity_id": EID, "fan_mode": "on"}]
        assert _service_data(hass, "set_temperature") == [
            {"entity_id": EID, "target_temp_low": 62.0, "target_temp_high": 80.0}
        ]

    @pytest.mark.asyncio
    async def test_idle_fan_mode_is_applied(self) -> None:
        hass = _hass_f()
        state = _ecobee_state()
        settings = _settings(idle_fan_mode="on")
        await _apply(ThermostatActuator(), hass, state, MODE_IDLE, settings=settings, now_ts=1000.0)
        assert _service_data(hass, "set_fan_mode") == [{"entity_id": EID, "fan_mode": "on"}]

    @pytest.mark.asyncio
    async def test_no_fan_modes_uses_fan_only(self) -> None:
        hass = _hass_f()
        state = _ecobee_state(fan_modes=[])
        result = await _apply(ThermostatActuator(), hass, state, MODE_CIRCULATING, now_ts=1000.0)
        assert result.hvac_mode == "fan_only"
        assert _service_data(hass, "set_hvac_mode") == [{"entity_id": EID, "hvac_mode": "fan_only"}]
        assert _service_data(hass, "set_temperature") == []
        assert _service_data(hass, "set_fan_mode") == []

    @pytest.mark.asyncio
    async def test_offline_safety_clamp_caps_the_nudge_below_the_room_target(self) -> None:
        hass = _hass_f()
        state = _ecobee_state()
        result = await _apply(
            ThermostatActuator(),
            hass,
            state,
            MODE_HEATING,
            active_heat_ceiling_c=18.0,
            now_ts=1000.0,
        )
        # 18 °C room target + 2.5 °C headroom = 20.5 °C = 69 °F, not 70 °F.
        assert _service_data(hass, "set_temperature")[0]["target_temp_low"] == 69.0
        assert result.target_temp_low_c == pytest.approx(20.5)

    @pytest.mark.asyncio
    async def test_cooling_floor_prevents_overcooling(self) -> None:
        hass = _hass_f()
        state = _ecobee_state(current=70.0)
        await _apply(
            ThermostatActuator(),
            hass,
            state,
            MODE_COOLING,
            active_cool_floor_c=24.0,
            now_ts=1000.0,
        )
        assert _service_data(hass, "set_temperature")[0]["target_temp_high"] == 71.0

    @pytest.mark.asyncio
    async def test_cooling_nudges_below_the_hallway(self) -> None:
        hass = _hass_f()
        state = _ecobee_state(current=76.0)
        await _apply(ThermostatActuator(), hass, state, MODE_COOLING, now_ts=1000.0)
        sent = _service_data(hass, "set_temperature")[0]
        assert sent["target_temp_high"] == 74.0
        assert sent["target_temp_low"] == 62.0

    @pytest.mark.asyncio
    async def test_heating_from_off_switches_to_heat_cool(self) -> None:
        hass = _hass_f()
        state = _ecobee_state(state="off")
        await _apply(ThermostatActuator(), hass, state, MODE_HEATING, now_ts=1000.0)
        assert _service_data(hass, "set_hvac_mode") == [{"entity_id": EID, "hvac_mode": "heat_cool"}]

    @pytest.mark.asyncio
    async def test_hallway_catch_up_re_nudges_immediately(self) -> None:
        hass = _hass_f()
        state = _ecobee_state()
        actuator = ThermostatActuator()
        await _apply(actuator, hass, state, MODE_HEATING, now_ts=1000.0)
        assert _service_data(hass, "set_temperature")[0]["target_temp_low"] == 70.0

        # The hallway sensor reached the setpoint we sent; move the nudge ahead.
        state.attributes["current_temperature"] = 69.0
        result = await _apply(actuator, hass, state, MODE_HEATING, now_ts=1100.0)
        assert result.applied is True
        assert _service_data(hass, "set_temperature")[-1]["target_temp_low"] == 72.0

    @pytest.mark.asyncio
    async def test_300_second_dwell_suppresses_minor_resends(self) -> None:
        hass = _hass_f()
        state = _ecobee_state()
        actuator = ThermostatActuator()
        await _apply(actuator, hass, state, MODE_HEATING, now_ts=1000.0)
        calls_after_first = len(hass.services.async_call.call_args_list)

        # The hallway did not catch up, so the same run must not re-send early.
        state.attributes["current_temperature"] = 67.0
        result = await _apply(actuator, hass, state, MODE_HEATING, now_ts=1100.0)
        assert result.applied is False
        assert len(hass.services.async_call.call_args_list) == calls_after_first

    @pytest.mark.asyncio
    async def test_dwell_expires_after_300_seconds(self) -> None:
        hass = _hass_f()
        state = _ecobee_state()
        actuator = ThermostatActuator()
        await _apply(actuator, hass, state, MODE_HEATING, now_ts=1000.0)
        state.attributes["current_temperature"] = 67.0
        await _apply(actuator, hass, state, MODE_HEATING, now_ts=1000.0 + SETPOINT_DWELL_SECONDS)
        assert len(_service_data(hass, "set_temperature")) == 2

    @pytest.mark.asyncio
    async def test_dwell_suppression_without_a_known_anchor(self) -> None:
        hass = _hass_f()
        state = _ecobee_state()
        actuator = ThermostatActuator()
        # Directly seed a run that never sent a setpoint (mode-only command).
        cache = actuator._cache.setdefault(EID, _EntityCommandCache())
        cache.last_action = MODE_HEATING
        cache.last_sent_ts = 1000.0
        result = await _apply(actuator, hass, state, MODE_HEATING, now_ts=1100.0)
        assert result.applied is False
        assert _service_data(hass, "set_temperature") == []

    @pytest.mark.asyncio
    async def test_already_matching_state_skips_service_calls(self) -> None:
        hass = _hass_f()
        state = _ecobee_state(
            state="heat_cool",
            target_temp_low=62.0,
            target_temp_high=80.0,
        )
        result = await _apply(ThermostatActuator(), hass, state, MODE_IDLE, now_ts=1000.0)
        assert result.applied is False
        assert hass.services.async_call.call_args_list == []

    @pytest.mark.asyncio
    async def test_fahrenheit_single_setpoint_device(self) -> None:
        hass = _hass_f()
        state = _ecobee_state(
            state="heat",
            hvac_modes=SINGLE_HEAT_MODES,
            fan_modes=[],
            supported_features=1,
        )
        result = await _apply(ThermostatActuator(), hass, state, MODE_HEATING, now_ts=1000.0)
        assert result.hvac_mode == "heat"
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 70.0}]
        assert _service_data(hass, "set_hvac_mode") == []


# --------------------------------------------------------------------------- #
# Single-setpoint °C thermostats
# --------------------------------------------------------------------------- #


class TestSingleSetpointCelsius:
    @pytest.mark.asyncio
    async def test_heating_single_setpoint(self) -> None:
        hass = _hass()
        state = _state(
            state="heat", current=19.0, hvac_modes=SINGLE_HEAT_MODES, fan_modes=["auto"], supported_features=1
        )
        result = await _apply(ThermostatActuator(), hass, state, MODE_HEATING, now_ts=1000.0)
        assert result.target_temp_single_c == pytest.approx(20.2)
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 20.0}]

    @pytest.mark.asyncio
    async def test_cooling_single_setpoint(self) -> None:
        hass = _hass()
        state = _state(
            state="cool", current=25.0, hvac_modes=SINGLE_COOL_MODES, fan_modes=["auto"], supported_features=1
        )
        result = await _apply(
            ThermostatActuator(),
            hass,
            state,
            MODE_COOLING,
            climate_mode=CLIMATE_MODE_COOL_ONLY,
            now_ts=1000.0,
        )
        assert result.target_temp_single_c == pytest.approx(23.8)
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 24.0}]

    @pytest.mark.asyncio
    async def test_idle_on_a_heat_only_device_stays_heat(self) -> None:
        hass = _hass()
        state = _state(state="heat", hvac_modes=SINGLE_HEAT_MODES, fan_modes=["auto"], supported_features=1)
        result = await _apply(
            ThermostatActuator(),
            hass,
            state,
            MODE_IDLE,
            climate_mode=CLIMATE_MODE_HEAT_ONLY,
            now_ts=1000.0,
        )
        assert result.hvac_mode == "heat"
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 16.5}]
        assert _service_data(hass, "set_hvac_mode") == []

    @pytest.mark.asyncio
    async def test_idle_on_a_cool_only_device_stays_cool(self) -> None:
        hass = _hass()
        state = _state(state="cool", hvac_modes=SINGLE_COOL_MODES, fan_modes=["auto"], supported_features=1)
        await _apply(
            ThermostatActuator(),
            hass,
            state,
            MODE_IDLE,
            climate_mode=CLIMATE_MODE_COOL_ONLY,
            now_ts=1000.0,
        )
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 26.5}]
        assert _service_data(hass, "set_hvac_mode") == []

    @pytest.mark.asyncio
    async def test_idle_without_heat_mode_switches_to_cool(self) -> None:
        hass = _hass()
        state = _state(state="heat", hvac_modes=SINGLE_COOL_MODES, fan_modes=["auto"], supported_features=1)
        await _apply(
            ThermostatActuator(),
            hass,
            state,
            MODE_IDLE,
            climate_mode=CLIMATE_MODE_HEAT_ONLY,
            now_ts=1000.0,
        )
        assert _service_data(hass, "set_hvac_mode") == [{"entity_id": EID, "hvac_mode": "cool"}]
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 26.5}]

    @pytest.mark.asyncio
    async def test_auto_idle_on_a_cooling_device_keeps_cooling(self) -> None:
        hass = _hass()
        state = _state(
            state="cool",
            hvac_modes=["heat", "cool", "off"],
            fan_modes=["auto"],
            supported_features=1,
        )
        result = await _apply(ThermostatActuator(), hass, state, MODE_IDLE, now_ts=1000.0)
        assert result.hvac_mode == "cool"
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 26.5}]

    @pytest.mark.asyncio
    async def test_fahrenheit_single_auto_idle_keeps_heat(self) -> None:
        hass = _hass_f()
        state = _ecobee_state(state="heat", hvac_modes=SINGLE_HEAT_MODES, fan_modes=[], supported_features=1)
        result = await _apply(ThermostatActuator(), hass, state, MODE_IDLE, now_ts=1000.0)
        assert result.hvac_mode == "heat"
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 62.0}]
        assert _service_data(hass, "set_hvac_mode") == []

    @pytest.mark.asyncio
    async def test_fahrenheit_single_cool_only_cools(self) -> None:
        hass = _hass_f()
        state = _ecobee_state(
            state="cool",
            current=76.0,
            hvac_modes=SINGLE_COOL_MODES,
            fan_modes=[],
            supported_features=1,
        )
        result = await _apply(
            ThermostatActuator(),
            hass,
            state,
            MODE_COOLING,
            climate_mode=CLIMATE_MODE_COOL_ONLY,
            now_ts=1000.0,
        )
        assert result.hvac_mode == "cool"
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 74.0}]

    @pytest.mark.asyncio
    async def test_fahrenheit_single_force_off(self) -> None:
        hass = _hass_f()
        state = _ecobee_state(
            state="heat",
            hvac_modes=["heat", "off"],
            fan_modes=["auto"],
            fan_mode="auto",
            supported_features=1,
        )
        result = await _apply(ThermostatActuator(), hass, state, MODE_IDLE, force_off=True, now_ts=1000.0)
        assert result.hvac_mode == "off"
        assert _service_data(hass, "set_hvac_mode") == [{"entity_id": EID, "hvac_mode": "off"}]

    @pytest.mark.asyncio
    async def test_heating_switches_mode_when_needed(self) -> None:
        hass = _hass()
        state = _state(state="cool", hvac_modes=["heat", "cool", "off"], fan_modes=["auto"], supported_features=1)
        await _apply(ThermostatActuator(), hass, state, MODE_HEATING, now_ts=1000.0)
        assert _service_data(hass, "set_hvac_mode") == [{"entity_id": EID, "hvac_mode": "heat"}]
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 21.0}]

    @pytest.mark.asyncio
    async def test_heat_cool_in_modes_without_range_feature_uses_single_setpoint(self) -> None:
        hass = _hass()
        state = _state(
            state="heat_cool",
            hvac_modes=RANGE_MODES,
            fan_modes=["auto"],
            supported_features=1,
        )
        result = await _apply(ThermostatActuator(), hass, state, MODE_HEATING, now_ts=1000.0)
        assert result.hvac_mode == "heat"
        assert _service_data(hass, "set_hvac_mode") == [{"entity_id": EID, "hvac_mode": "heat"}]
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 21.0}]


# --------------------------------------------------------------------------- #
# Device data fallbacks
# --------------------------------------------------------------------------- #


class TestDeviceFallbacks:
    @pytest.mark.asyncio
    async def test_missing_current_temperature_uses_house_reference(self) -> None:
        hass = _hass()
        state = _state(state="heat", current=None, hvac_modes=SINGLE_HEAT_MODES, fan_modes=["auto"])
        result = await _apply(
            ThermostatActuator(),
            hass,
            state,
            MODE_HEATING,
            house_ref_temp_c=19.0,
            now_ts=1000.0,
        )
        assert result.thermostat_temp_c == pytest.approx(19.0)
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 20.0}]

    @pytest.mark.asyncio
    async def test_missing_current_temperature_and_house_reference_uses_21(self) -> None:
        hass = _hass()
        state = _state(state="heat", current=None, hvac_modes=SINGLE_HEAT_MODES, fan_modes=["auto"])
        result = await _apply(ThermostatActuator(), hass, state, MODE_HEATING, now_ts=1000.0)
        assert result.thermostat_temp_c == pytest.approx(21.0)
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 22.0}]

    @pytest.mark.asyncio
    async def test_missing_min_max_and_step_use_defaults(self) -> None:
        hass = _hass()
        state = _state(
            state="heat",
            current=20.0,
            min_temp=None,
            max_temp=None,
            step=None,
            hvac_modes=SINGLE_HEAT_MODES,
            fan_modes=[],
        )
        await _apply(ThermostatActuator(), hass, state, MODE_HEATING, now_ts=1000.0)
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 21.0}]

    @pytest.mark.asyncio
    async def test_invalid_step_uses_the_default(self) -> None:
        hass = _hass()
        state = _state(
            state="heat",
            current=20.0,
            step=0,
            hvac_modes=SINGLE_HEAT_MODES,
            fan_modes=[],
        )
        await _apply(ThermostatActuator(), hass, state, MODE_HEATING, now_ts=1000.0)
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 21.0}]

    @pytest.mark.asyncio
    async def test_missing_attributes_are_tolerated(self) -> None:
        hass = _hass()
        state = SimpleNamespace(state="heat", attributes=None)
        result = await _apply(ThermostatActuator(), hass, state, MODE_HEATING, now_ts=1000.0)
        assert result.applied is True
        # No HVAC modes: idle/heat chooses the fallback single-setpoint path.
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 22.0}]

    @pytest.mark.asyncio
    async def test_missing_entity_or_unavailable_device_does_nothing(self) -> None:
        hass = _hass()
        state = _state(state="heat")
        actuator = ThermostatActuator()
        _install(hass, state)
        no_entity = await actuator.async_apply(hass, {}, MODE_HEATING, now_ts=1000.0)
        assert no_entity == ActuationResult(False, None, None, None, None, None, None)

        state.state = "unavailable"
        result = await _apply(actuator, hass, state, MODE_HEATING, now_ts=1000.0)
        assert result == ActuationResult(False, None, None, None, None, None, None)
        assert hass.services.async_call.call_args_list == []

        hass.states.get = MagicMock(return_value=None)
        missing = await actuator.async_apply(hass, _settings(), MODE_HEATING, now_ts=1000.0)
        assert missing == ActuationResult(False, None, None, None, None, None, None)


# --------------------------------------------------------------------------- #
# Force off and failure handling
# --------------------------------------------------------------------------- #


class TestForceOff:
    @pytest.mark.asyncio
    async def test_force_off_switches_mode_and_idle_fan(self) -> None:
        hass = _hass()
        state = _state(
            state="heat",
            hvac_modes=["heat", "off"],
            fan_modes=["auto", "on"],
            fan_mode="on",
            supported_features=1,
        )
        result = await _apply(ThermostatActuator(), hass, state, MODE_IDLE, force_off=True, now_ts=1000.0)
        assert result == ActuationResult(True, "off", "auto", None, None, None, 20.0)
        assert _service_data(hass, "set_hvac_mode") == [{"entity_id": EID, "hvac_mode": "off"}]
        assert _service_data(hass, "set_fan_mode") == [{"entity_id": EID, "fan_mode": "auto"}]

    @pytest.mark.asyncio
    async def test_force_off_is_a_no_op_when_already_off(self) -> None:
        hass = _hass()
        state = _state(
            state="off",
            hvac_modes=["heat", "off"],
            fan_modes=["auto"],
            fan_mode="auto",
            supported_features=1,
        )
        result = await _apply(ThermostatActuator(), hass, state, MODE_IDLE, force_off=True, now_ts=1000.0)
        assert result.applied is False
        assert hass.services.async_call.call_args_list == []

    @pytest.mark.asyncio
    async def test_force_off_without_an_off_mode_is_a_no_op(self) -> None:
        hass = _hass()
        state = _state(state="heat", hvac_modes=["heat"], fan_modes=[], supported_features=1)
        result = await _apply(ThermostatActuator(), hass, state, MODE_IDLE, force_off=True, now_ts=1000.0)
        assert result.applied is False
        assert hass.services.async_call.call_args_list == []


class TestFailures:
    @pytest.mark.asyncio
    async def test_failing_setpoint_call_reports_not_applied(self) -> None:
        hass = _hass()
        state = _state(state="heat", hvac_modes=SINGLE_HEAT_MODES, fan_modes=["auto"], supported_features=1)
        hass.services.async_call = AsyncMock(side_effect=RuntimeError("boom"))
        result = await _apply(ThermostatActuator(), hass, state, MODE_HEATING, now_ts=1000.0)
        assert result.applied is False
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 21.0}]

    @pytest.mark.asyncio
    async def test_failing_mode_call_still_applies_the_setpoint(self) -> None:
        hass = _hass()
        state = _state(state="cool", hvac_modes=["heat", "cool", "off"], fan_modes=["auto"], supported_features=1)

        async def _call(domain: str, service: str, data: dict[str, Any], **kwargs: Any) -> None:
            if service == "set_hvac_mode":
                raise RuntimeError("mode failed")

        hass.services.async_call = AsyncMock(side_effect=_call)
        result = await _apply(ThermostatActuator(), hass, state, MODE_HEATING, now_ts=1000.0)
        assert result.applied is True
        assert result.hvac_mode == "heat"


# --------------------------------------------------------------------------- #
# Cache and override tracker integration
# --------------------------------------------------------------------------- #


class TestCacheAndTracker:
    def test_clear_forgets_cached_commands(self) -> None:
        actuator = ThermostatActuator()
        actuator._cache[EID] = _EntityCommandCache()
        actuator.clear()
        assert actuator._cache == {}

    def test_snap_helpers_respect_the_step(self) -> None:
        assert _snap_up(70.1, 1.0) == 71.0
        assert _snap_up(70.0, 1.0) == 70.0
        assert _snap_down(70.9, 1.0) == 70.0
        assert _snap_down(70.0, 1.0) == 70.0

    @pytest.mark.asyncio
    async def test_tracker_learns_the_sent_range_and_ignores_the_echo(self) -> None:
        hass = _hass_f()
        state = _ecobee_state()
        tracker = ThermostatOverrideTracker()
        await _apply(
            ThermostatActuator(),
            hass,
            state,
            MODE_HEATING,
            override_tracker=tracker,
            now_ts=1000.0,
        )
        sent = _service_data(hass, "set_temperature")[0]
        state.attributes["target_temp_low"] = sent["target_temp_low"]
        state.attributes["target_temp_high"] = sent["target_temp_high"]
        assert tracker.check_external_change(hass, EID, state, {}, 21.0, now_ts=1010.0) is None

    @pytest.mark.asyncio
    async def test_tracker_records_before_the_service_call_returns(self) -> None:
        hass = _hass()
        state = _state(
            state="heat", current=19.0, hvac_modes=SINGLE_HEAT_MODES, fan_modes=["auto"], supported_features=1
        )
        tracker = ThermostatOverrideTracker()
        recorded_while_in_flight: list[bool] = []

        async def _call(domain: str, service: str, data: dict[str, Any], **kwargs: Any) -> None:
            if service == "set_temperature":
                recorded_while_in_flight.append(bool(tracker._sent))
                assert kwargs["blocking"] is True
                assert kwargs["context"].parent_id == "roommind"

        hass.services.async_call = AsyncMock(side_effect=_call)
        await _apply(
            ThermostatActuator(),
            hass,
            state,
            MODE_HEATING,
            override_tracker=tracker,
            now_ts=1000.0,
        )
        assert recorded_while_in_flight == [True]

    @pytest.mark.asyncio
    async def test_tracker_learns_a_sent_single_setpoint(self) -> None:
        hass = _hass()
        state = _state(
            state="heat", current=19.0, hvac_modes=SINGLE_HEAT_MODES, fan_modes=["auto"], supported_features=1
        )
        tracker = ThermostatOverrideTracker()
        await _apply(
            ThermostatActuator(),
            hass,
            state,
            MODE_HEATING,
            override_tracker=tracker,
            now_ts=1000.0,
        )
        state.attributes["temperature"] = 20.0
        assert tracker.check_external_change(hass, EID, state, {}, 21.0, now_ts=1010.0) is None

    @pytest.mark.asyncio
    async def test_first_run_records_the_anchor_and_second_run_reuses_it(self) -> None:
        hass = _hass()
        state = _state(
            state="heat", current=19.0, hvac_modes=SINGLE_HEAT_MODES, fan_modes=["auto"], supported_features=1
        )
        actuator = ThermostatActuator()
        await _apply(actuator, hass, state, MODE_HEATING, now_ts=1000.0)
        cache = actuator._cache[EID]
        assert cache.last_single_native == 20.0
        assert cache.last_action == MODE_HEATING
        assert cache.last_hvac_mode == "heat"
        assert cache.last_sent_ts == 1000.0

        # Caught up: re-nudge without waiting out the dwell.
        state.attributes["current_temperature"] = 20.0
        await _apply(actuator, hass, state, MODE_HEATING, now_ts=1100.0)
        assert _service_data(hass, "set_temperature")[-1]["temperature"] == 21.0

    @pytest.mark.asyncio
    async def test_force_off_resets_the_catch_up_anchor(self) -> None:
        hass = _hass()
        state = _state(
            state="heat",
            current=19.0,
            hvac_modes=["heat", "off"],
            fan_modes=["auto"],
            supported_features=1,
        )
        actuator = ThermostatActuator()
        await _apply(actuator, hass, state, MODE_HEATING, now_ts=1000.0)
        await _apply(actuator, hass, state, MODE_IDLE, force_off=True, now_ts=1100.0)
        cache = actuator._cache[EID]
        assert cache.last_sent_ts == 0.0
        assert cache.last_low_native is None
        assert cache.last_high_native is None
        assert cache.last_single_native is None
        assert cache.last_action == "off"


def test_setpoint_dwell_constant_is_300_seconds() -> None:
    assert SETPOINT_DWELL_SECONDS == 300.0
    assert ThermostatActuator.SETPOINT_DWELL_SECONDS == 300.0
    assert time.time() > 0


# --------------------------------------------------------------------------- #
# Numeric helpers, unit mismatch and deadband edge cases
# --------------------------------------------------------------------------- #


class TestNumericEdgeCases:
    @pytest.mark.asyncio
    async def test_non_numeric_current_temperature_falls_back_to_the_house(self) -> None:
        hass = _hass()
        state = _state(
            state="heat",
            current=None,
            hvac_modes=SINGLE_HEAT_MODES,
            fan_modes=[],
            current_temperature="warm",
        )
        result = await _apply(
            ThermostatActuator(),
            hass,
            state,
            MODE_HEATING,
            house_ref_temp_c=19.0,
            now_ts=1000.0,
        )
        assert result.thermostat_temp_c == pytest.approx(19.0)

    @pytest.mark.asyncio
    async def test_non_numeric_supported_features_is_zero(self) -> None:
        hass = _hass()
        state = _state(
            state="heat",
            hvac_modes=RANGE_MODES,
            fan_modes=["auto"],
            supported_features="lots",  # type: ignore[arg-type]
        )
        result = await _apply(ThermostatActuator(), hass, state, MODE_HEATING, now_ts=1000.0)
        assert result.hvac_mode == "heat"

    @pytest.mark.asyncio
    async def test_corrupt_nudge_setting_falls_back_to_the_default(self) -> None:
        hass = _hass()
        state = _state(state="heat", current=19.0, hvac_modes=SINGLE_HEAT_MODES, fan_modes=[])
        settings = _settings(nudge_delta_c="a lot")
        result = await _apply(ThermostatActuator(), hass, state, MODE_HEATING, settings=settings, now_ts=1000.0)
        assert result.target_temp_single_c == pytest.approx(20.2)

    @pytest.mark.asyncio
    async def test_fahrenheit_default_step_without_an_attribute(self) -> None:
        hass = _hass_f()
        state = _ecobee_state(state="heat", hvac_modes=SINGLE_HEAT_MODES, fan_modes=[], step=None)
        await _apply(ThermostatActuator(), hass, state, MODE_HEATING, now_ts=1000.0)
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 70.0}]

    @pytest.mark.asyncio
    async def test_device_fahrenheit_while_ha_is_celsius(self) -> None:
        hass = _hass("°C")
        state = _ecobee_state()
        result = await _apply(ThermostatActuator(), hass, state, MODE_HEATING, now_ts=1000.0)
        assert result.thermostat_temp_c == pytest.approx(20.0)
        assert _service_data(hass, "set_temperature")[0]["target_temp_low"] == 70.0

    @pytest.mark.asyncio
    async def test_device_celsius_while_ha_is_fahrenheit(self) -> None:
        hass = _hass_f()
        state = _state(state="heat", unit="°C", current=19.0, hvac_modes=SINGLE_HEAT_MODES, fan_modes=[])
        await _apply(ThermostatActuator(), hass, state, MODE_HEATING, now_ts=1000.0)
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 20.0}]

    @pytest.mark.asyncio
    async def test_heating_floor_is_enforced_when_a_plan_rounds_short(self) -> None:
        hass = _hass()
        state = _state(state="heat_cool", current=20.0, hvac_modes=RANGE_MODES, fan_modes=["auto"])
        actuator = ThermostatActuator()
        plan = TargetPlan("heat_cool", "auto", 20.0, 21.7, None, 20.0, 1.2, 1.7, 26.5, 16.5, False, False)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(actuator, "compute_targets_c", lambda *args, **kwargs: plan)
            await _apply(actuator, hass, state, MODE_HEATING, now_ts=1000.0)
        # round(20 °C) + one 0.5 °C step is 20.5 °C, not the rounded 20.0.
        assert _service_data(hass, "set_temperature")[0]["target_temp_low"] == 20.5

    @pytest.mark.asyncio
    async def test_heating_floor_is_enforced_for_a_single_setpoint(self) -> None:
        hass = _hass()
        state = _state(state="heat", current=20.0, hvac_modes=SINGLE_HEAT_MODES, fan_modes=[])
        actuator = ThermostatActuator()
        plan = TargetPlan("heat", None, None, None, 20.0, 20.0, 1.2, 1.7, 26.5, 16.5, False, False)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(actuator, "compute_targets_c", lambda *args, **kwargs: plan)
            await _apply(actuator, hass, state, MODE_HEATING, now_ts=1000.0)
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 20.5}]

    @pytest.mark.asyncio
    async def test_cooling_ceiling_is_enforced_for_a_single_setpoint(self) -> None:
        hass = _hass()
        state = _state(state="cool", current=20.0, hvac_modes=SINGLE_COOL_MODES, fan_modes=[])
        actuator = ThermostatActuator()
        plan = TargetPlan("cool", None, None, None, 20.5, 20.0, 1.2, 1.7, 26.5, 16.5, False, False)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(actuator, "compute_targets_c", lambda *args, **kwargs: plan)
            await _apply(actuator, hass, state, MODE_COOLING, now_ts=1000.0)
        assert _service_data(hass, "set_temperature") == [{"entity_id": EID, "temperature": 19.5}]

    @pytest.mark.asyncio
    async def test_cooling_ceiling_is_enforced_when_a_plan_rounds_high(self) -> None:
        hass = _hass()
        state = _state(state="heat_cool", current=20.0, hvac_modes=RANGE_MODES, fan_modes=["auto"])
        actuator = ThermostatActuator()
        plan = TargetPlan("heat_cool", "auto", 16.5, 20.5, None, 20.0, 1.2, 1.7, 26.5, 16.5, False, False)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(actuator, "compute_targets_c", lambda *args, **kwargs: plan)
            await _apply(actuator, hass, state, MODE_COOLING, now_ts=1000.0)
        assert _service_data(hass, "set_temperature")[0]["target_temp_high"] == 19.5

    def test_snap_helpers_are_safe(self) -> None:
        assert ThermostatActuator._at_least(20.1, 0.5, 5.0, 35.0) == 20.5
        assert ThermostatActuator._at_most(20.9, 0.5, 5.0, 35.0) == 20.5
        assert ThermostatActuator._at_least(20.1, 0.5, 21.0, 35.0) == 21.0
        assert ThermostatActuator._at_most(20.9, 0.5, 5.0, 20.0) == 20.0

    def test_enforce_deadband_no_op(self) -> None:
        assert ThermostatActuator._enforce_deadband(10.0, 14.0, MODE_HEATING, 3.0, 1.0, 5.0, 35.0) == (10.0, 14.0)

    def test_enforce_deadband_lowers_the_cooling_side(self) -> None:
        assert ThermostatActuator._enforce_deadband(10.0, 11.0, MODE_COOLING, 3.0, 1.0, 5.0, 35.0) == (8.0, 11.0)

    def test_enforce_deadband_raises_the_cooling_side_when_the_floor_blocks(self) -> None:
        assert ThermostatActuator._enforce_deadband(5.0, 6.0, MODE_COOLING, 3.0, 1.0, 5.0, 35.0) == (5.0, 8.0)

    def test_enforce_deadband_raises_the_heating_side(self) -> None:
        assert ThermostatActuator._enforce_deadband(10.0, 11.0, MODE_HEATING, 3.0, 1.0, 5.0, 12.0) == (9.0, 12.0)

    def test_enforce_deadband_gives_up_at_the_device_limits(self) -> None:
        assert ThermostatActuator._enforce_deadband(5.0, 12.0, MODE_HEATING, 15.0, 1.0, 5.0, 12.0) == (5.0, 12.0)

    def test_compute_flips_degenerate_device_limits(self) -> None:
        actuator = ThermostatActuator()
        plan = actuator.compute_targets_c(
            {},
            MODE_IDLE,
            thermostat_temp_c=20.0,
            device_min_c=25.0,
            device_max_c=20.0,
            step_c=0.5,
            supports_range=True,
            hvac_modes=tuple(RANGE_MODES),
        )
        assert plan.high_c is not None
        assert plan.high_c >= 25.0
