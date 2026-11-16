"""Tests for thermostat_utils.py — Central Thermostat HVAC state helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from custom_components.roommind.utils.thermostat_utils import (
    FAN_MODE_FLAG,
    TARGET_TEMPERATURE_RANGE_FLAG,
    has_reliable_hvac_modes,
    is_fahrenheit_state,
    round_to_step,
    supported_fan_modes,
    supports_fan_modes,
    supports_fan_only_hvac_mode,
    supports_target_range,
    thermostat_bounds_c,
    thermostat_step_native,
)


def _make_state(
    hvac_modes: list[str] | None = None,
    supported_features: int | None = None,
    fan_modes: list[str] | None = None,
    unit_of_measurement: str | None = None,
    temperature_unit: str | None = None,
    target_temp_step: float | int | str | None = None,
    min_temp: float | int | None = None,
    max_temp: float | int | None = None,
) -> MagicMock:
    """Return a mock climate state with the given attributes."""
    state = MagicMock()
    attrs: dict[str, Any] = {}
    if hvac_modes is not None:
        attrs["hvac_modes"] = hvac_modes
    if supported_features is not None:
        attrs["supported_features"] = supported_features
    if fan_modes is not None:
        attrs["fan_modes"] = fan_modes
    if unit_of_measurement is not None:
        attrs["unit_of_measurement"] = unit_of_measurement
    if temperature_unit is not None:
        attrs["temperature_unit"] = temperature_unit
    if target_temp_step is not None:
        attrs["target_temp_step"] = target_temp_step
    if min_temp is not None:
        attrs["min_temp"] = min_temp
    if max_temp is not None:
        attrs["max_temp"] = max_temp
    state.attributes = attrs
    return state


def _make_hass(temperature_unit: str | None = None) -> MagicMock:
    """Return a mock hass with an optional temperature unit."""
    hass = MagicMock()
    if temperature_unit is None:
        hass.config.units = None
    else:
        hass.config.units.temperature_unit = temperature_unit
    return hass


class TestHasReliableHvacModes:
    """Unit tests for has_reliable_hvac_modes."""

    def test_none_state(self):
        """A missing state is never reliable."""
        assert has_reliable_hvac_modes(None) is False

    def test_empty_modes_unreliable(self):
        """An empty hvac_modes list is not reliable."""
        assert has_reliable_hvac_modes(_make_state(hvac_modes=[])) is False

    def test_missing_modes_unreliable(self):
        """A state without an hvac_modes attribute is not reliable."""
        assert has_reliable_hvac_modes(_make_state()) is False

    def test_fan_only_only_unreliable(self):
        """Only fan_only plus off hides real capabilities."""
        assert has_reliable_hvac_modes(_make_state(hvac_modes=["off", "fan_only"])) is False

    def test_off_only_unreliable(self):
        """Only off is not an active mode."""
        assert has_reliable_hvac_modes(_make_state(hvac_modes=["off"])) is False

    def test_heat_reliable(self):
        """Heating is an active mode."""
        assert has_reliable_hvac_modes(_make_state(hvac_modes=["off", "heat"])) is True

    def test_cool_reliable(self):
        """Cooling is an active mode."""
        assert has_reliable_hvac_modes(_make_state(hvac_modes=["off", "cool"])) is True

    def test_auto_reliable(self):
        """Auto is an active mode."""
        assert has_reliable_hvac_modes(_make_state(hvac_modes=["off", "auto"])) is True

    def test_heat_cool_reliable(self):
        """Heat/cool swing is an active mode."""
        assert has_reliable_hvac_modes(_make_state(hvac_modes=["off", "heat_cool"])) is True


class TestSupportsTargetRange:
    """Unit tests for supports_target_range."""

    def test_none_state(self):
        """A missing state cannot support a range."""
        assert supports_target_range(None) is False

    def test_heat_only_not_a_range(self):
        """A heat-only device has no heating/cooling band."""
        state = _make_state(hvac_modes=["off", "heat"], supported_features=TARGET_TEMPERATURE_RANGE_FLAG)
        assert supports_target_range(state) is False

    def test_heat_cool_without_range_flag(self):
        """heat_cool without the range feature flag is not a range."""
        state = _make_state(hvac_modes=["off", "heat_cool"], supported_features=0)
        assert supports_target_range(state) is False

    def test_heat_cool_with_range_flag(self):
        """heat_cool with the range feature flag supports a band."""
        state = _make_state(hvac_modes=["off", "heat_cool"], supported_features=TARGET_TEMPERATURE_RANGE_FLAG)
        assert supports_target_range(state) is True

    def test_single_temp_feature_not_range(self):
        """The single-target-temperature feature is not a range."""
        state = _make_state(hvac_modes=["off", "heat_cool"], supported_features=1)
        assert supports_target_range(state) is False


class TestHvacFeatureFlags:
    """Unit tests for the ClimateEntityFeature flag constants."""

    def test_target_temperature_range_flag_value(self):
        """The target temperature range flag is bit 2 of ClimateEntityFeature."""
        assert TARGET_TEMPERATURE_RANGE_FLAG == 2

    def test_fan_mode_flag_value(self):
        """The fan mode flag is bit 8 of ClimateEntityFeature."""
        assert FAN_MODE_FLAG == 8


class TestSupportedFanModes:
    """Unit tests for supported_fan_modes / supports_fan_modes."""

    def test_none_state_has_no_fan_modes(self):
        """A missing state has no fan modes."""
        assert supported_fan_modes(None) == []
        assert supports_fan_modes(None) is False

    def test_fan_modes_list(self):
        """The advertised fan modes are returned as strings."""
        state = _make_state(fan_modes=["off", "low", "high"])
        assert supported_fan_modes(state) == ["off", "low", "high"]
        assert supports_fan_modes(state) is True

    def test_empty_fan_modes(self):
        """An empty fan mode list means no fan support."""
        assert supported_fan_modes(_make_state(fan_modes=[])) == []
        assert supports_fan_modes(_make_state(fan_modes=[])) is False


class TestSupportsFanOnlyHvacMode:
    """Unit tests for supports_fan_only_hvac_mode."""

    def test_none_state(self):
        """A missing state cannot run fan only."""
        assert supports_fan_only_hvac_mode(None) is False

    def test_fan_only_supported(self):
        """fan_only in the mode list is supported."""
        assert supports_fan_only_hvac_mode(_make_state(hvac_modes=["off", "fan_only"])) is True

    def test_heat_only_unsupported(self):
        """fan_only is not advertised for a heat-only device."""
        assert supports_fan_only_hvac_mode(_make_state(hvac_modes=["off", "heat"])) is False


class TestIsFahrenheitState:
    """Unit tests for is_fahrenheit_state."""

    def test_state_unit_fahrenheit(self):
        """A state reporting °F is a Fahrenheit state."""
        assert is_fahrenheit_state(_make_hass("°C"), _make_state(unit_of_measurement="°F")) is True

    def test_state_unit_short_f(self):
        """A state reporting 'F' is a Fahrenheit state."""
        assert is_fahrenheit_state(_make_hass("°C"), _make_state(unit_of_measurement="F")) is True

    def test_state_unit_celsius(self):
        """A state reporting °C is not a Fahrenheit state."""
        assert is_fahrenheit_state(_make_hass("°C"), _make_state(unit_of_measurement="°C")) is False

    def test_state_temperature_unit_attribute(self):
        """The temperature_unit attribute is honoured when unit_of_measurement is absent."""
        assert is_fahrenheit_state(_make_hass("°C"), _make_state(temperature_unit="°F")) is True

    def test_state_falls_back_to_hass(self):
        """A state without a unit falls back to the hass configuration."""
        assert is_fahrenheit_state(_make_hass("°F"), _make_state()) is True
        assert is_fahrenheit_state(_make_hass("°C"), _make_state()) is False

    def test_state_none_uses_hass(self):
        """A missing state inspects the hass configuration."""
        assert is_fahrenheit_state(_make_hass("°F"), None) is True
        assert is_fahrenheit_state(_make_hass("°C"), None) is False

    def test_hass_without_units(self):
        """A hass without a units object is treated as not Fahrenheit."""
        assert is_fahrenheit_state(_make_hass(None), None) is False


class TestThermostatStepNative:
    """Unit tests for thermostat_step_native."""

    def test_explicit_step_used(self):
        """A positive target_temp_step wins over the unit default."""
        state = _make_state(target_temp_step=0.5)
        assert thermostat_step_native(_make_hass("°C"), state) == 0.5

    def test_explicit_int_step_used(self):
        """An integer target_temp_step is converted to float."""
        state = _make_state(target_temp_step=2)
        assert thermostat_step_native(_make_hass("°F"), state) == 2.0

    def test_state_none_uses_unit_default(self):
        """Without a state the unit default applies."""
        assert thermostat_step_native(_make_hass("°C"), None) == 0.5
        assert thermostat_step_native(_make_hass("°F"), None) == 1.0

    def test_missing_step_uses_unit_default(self):
        """Without a target_temp_step the unit default applies."""
        state = _make_state()
        assert thermostat_step_native(_make_hass("°C"), state) == 0.5
        assert thermostat_step_native(_make_hass("°F"), state) == 1.0

    def test_zero_step_uses_unit_default(self):
        """A non-positive target_temp_step falls back."""
        state = _make_state(target_temp_step=0)
        assert thermostat_step_native(_make_hass("°C"), state) == 0.5
        assert thermostat_step_native(_make_hass("°F"), state) == 1.0

    def test_negative_step_uses_unit_default(self):
        """A negative target_temp_step falls back."""
        state = _make_state(target_temp_step=-0.5)
        assert thermostat_step_native(_make_hass("°C"), state) == 0.5

    def test_invalid_step_uses_unit_default(self):
        """A non-numeric target_temp_step falls back."""
        state = _make_state(target_temp_step="not-a-number")
        assert thermostat_step_native(_make_hass("°C"), state) == 0.5


class TestRoundToStep:
    """Unit tests for round_to_step."""

    def test_non_positive_step_rounds_to_tenth(self):
        """A non-positive step rounds to a single decimal place."""
        assert round_to_step(7.34, 0) == 7.3
        assert round_to_step(7.34, -2.0) == 7.3

    def test_integer_step_rounds_cleanly(self):
        """An integer step rounds to the nearest multiple without drift."""
        assert round_to_step(6.3, 2.0) == 6.0
        assert round_to_step(7.3, 2.0) == 8.0

    def test_fractional_step_rounds_to_two_decimals(self):
        """A fractional step rounds to two decimals."""
        assert round_to_step(7.3, 0.5) == 7.5
        assert round_to_step(7.34, 0.5) == 7.5


class TestThermostatBounds:
    """Unit tests for thermostat_bounds_c."""

    def test_none_state_returns_defaults(self):
        """Without a state the default bounds apply."""
        assert thermostat_bounds_c(_make_hass("°C"), None) == (7.0, 35.0)
        assert thermostat_bounds_c(_make_hass("°C"), None, default_min_c=10.0, default_max_c=30.0) == (10.0, 30.0)

    def test_celsius_bounds(self):
        """Celsius bounds are returned without conversion."""
        state = _make_state(min_temp=10.0, max_temp=30.0)
        assert thermostat_bounds_c(_make_hass("°C"), state) == (10.0, 30.0)

    def test_fahrenheit_bounds_converted(self):
        """Fahrenheit bounds are converted to Celsius."""
        state = _make_state(min_temp=50.0, max_temp=86.0)
        assert thermostat_bounds_c(_make_hass("°F"), state) == (10.0, 30.0)

    def test_min_only_uses_default_max(self):
        """A missing max falls back to the default maximum."""
        state = _make_state(min_temp=10.0)
        assert thermostat_bounds_c(_make_hass("°C"), state) == (10.0, 35.0)

    def test_max_only_uses_default_min(self):
        """A missing min falls back to the default minimum."""
        state = _make_state(max_temp=30.0)
        assert thermostat_bounds_c(_make_hass("°C"), state) == (7.0, 30.0)

    def test_inverted_bounds_fall_back_to_defaults(self):
        """An inverted band is rejected in favour of the defaults."""
        state = _make_state(min_temp=30.0, max_temp=10.0)
        assert thermostat_bounds_c(_make_hass("°C"), state) == (7.0, 35.0)

    def test_equal_bounds_fall_back_to_defaults(self):
        """Equal bounds are rejected in favour of the defaults."""
        state = _make_state(min_temp=10.0, max_temp=10.0)
        assert thermostat_bounds_c(_make_hass("°C"), state) == (7.0, 35.0)
