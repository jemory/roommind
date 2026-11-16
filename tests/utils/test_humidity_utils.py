"""Tests for humidity_utils.py — psychrometric and comfort humidity helpers."""

from __future__ import annotations

import math

import pytest

from custom_components.roommind.utils.humidity_utils import (
    dehumidify_cool_offset_c,
    effective_temperature_c,
    predict_rh_at_temp,
    saturation_vapor_pressure_hpa,
    vapor_pressure_hpa,
)


class TestSaturationVaporPressure:
    """Unit tests for saturation_vapor_pressure_hpa."""

    def test_at_20_c(self):
        """Saturation vapor pressure at 20 °C is about 23.3 hPa."""
        assert saturation_vapor_pressure_hpa(20.0) == pytest.approx(23.28, rel=0.05)

    def test_at_zero_c(self):
        """Saturation vapor pressure at 0 °C is about 6.1 hPa."""
        assert saturation_vapor_pressure_hpa(0.0) == pytest.approx(6.105, rel=0.01)

    def test_at_negative_c(self):
        """Saturation vapor pressure stays positive in the cold range."""
        assert saturation_vapor_pressure_hpa(-10.0) > 0.0


class TestVaporPressure:
    """Unit tests for vapor_pressure_hpa."""

    def test_half_rh(self):
        """Half the relative humidity halves the vapor pressure."""
        assert vapor_pressure_hpa(20.0, 50.0) == pytest.approx(0.5 * saturation_vapor_pressure_hpa(20.0), rel=0.05)

    def test_rh_clamped_to_100(self):
        """Relative humidity above 100 percent is clamped to saturation."""
        assert vapor_pressure_hpa(20.0, 150.0) == pytest.approx(saturation_vapor_pressure_hpa(20.0), rel=0.05)

    def test_rh_clamped_to_zero(self):
        """Relative humidity below zero percent clamps to zero vapor pressure."""
        assert vapor_pressure_hpa(20.0, -10.0) == 0.0


class TestEffectiveTemperature:
    """Unit tests for effective_temperature_c."""

    def test_disabled_returns_input_temp(self):
        """The moisture adjustment is skipped when disabled."""
        assert effective_temperature_c(21.5, 60.0, enabled=False) == 21.5

    def test_none_rh_returns_input_temp(self):
        """A missing relative humidity returns the input temperature."""
        assert effective_temperature_c(21.5, None) == 21.5

    def test_nan_rh_returns_input_temp(self):
        """A NaN relative humidity returns the input temperature."""
        assert effective_temperature_c(21.5, math.nan) == 21.5

    def test_reference_humidity_is_identity(self):
        """At the reference humidity the adjusted temperature equals the input."""
        assert effective_temperature_c(20.0, 45.0) == pytest.approx(20.0)

    def test_cold_temp_ramp_is_zero(self):
        """Below the ramp start the moisture offset is zero."""
        assert effective_temperature_c(10.0, 80.0) == pytest.approx(10.0)

    def test_wet_room_gets_adjusted_temp(self):
        """A very wet room gets a warmer effective temperature."""
        adjusted = effective_temperature_c(20.0, 80.0)
        assert adjusted > 20.0
        assert adjusted == pytest.approx(20.0 + 1.47, abs=0.1)


class TestPredictRhAtTemp:
    """Unit tests for predict_rh_at_temp."""

    def test_no_cooling_prediction(self):
        """Without cooling the predicted RH is derived from the current vapor pressure."""
        rh = predict_rh_at_temp(20.0, 50.0, 25.0)
        assert 5.0 < rh < 100.0

    def test_cooling_dehumidifies(self):
        """Active cooling removes vapor pressure and lowers the future RH."""
        rh_cooling = predict_rh_at_temp(20.0, 50.0, 25.0, cooling_active=True, dt_minutes=5.0)
        rh_idle = predict_rh_at_temp(20.0, 50.0, 25.0, cooling_active=False, dt_minutes=5.0)
        assert rh_cooling < rh_idle

    def test_zero_dt_minutes_keeps_rh(self):
        """A zero decay window leaves the vapor pressure untouched."""
        rh_zero = predict_rh_at_temp(20.0, 50.0, 25.0, cooling_active=True, dt_minutes=0.0)
        rh_idle = predict_rh_at_temp(20.0, 50.0, 25.0, cooling_active=False, dt_minutes=5.0)
        assert rh_zero == pytest.approx(rh_idle)

    def test_rh_clamped_to_bounds(self):
        """Predicted RH stays within the 5-100 percent comfort band."""
        rh = predict_rh_at_temp(20.0, 100.0, 30.0)
        assert 5.0 <= rh <= 100.0


class TestDehumidifyCoolOffset:
    """Unit tests for dehumidify_cool_offset_c."""

    def test_none_rh_is_zero(self):
        """A missing weighted humidity needs no overcooling."""
        assert dehumidify_cool_offset_c(None) == 0.0

    def test_zero_max_overcool_is_zero(self):
        """No overcooling budget means no offset."""
        assert dehumidify_cool_offset_c(90.0, max_overcool_c=0.0) == 0.0

    def test_below_threshold_is_zero(self):
        """A weighted humidity at or below the threshold needs no offset."""
        assert dehumidify_cool_offset_c(60.0) == 0.0
        assert dehumidify_cool_offset_c(59.0) == 0.0

    def test_above_threshold_full_offset(self):
        """A very wet room is fully overcooled."""
        assert dehumidify_cool_offset_c(80.0) == pytest.approx(0.8, abs=0.001)

    def test_above_threshold_partial_offset(self):
        """The offset ramps linearly over the 15 percent band."""
        assert dehumidify_cool_offset_c(70.0) == pytest.approx(0.533, abs=0.001)
