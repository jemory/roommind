"""Tests for the wall-thermostat override tracker."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from homeassistant.const import UnitOfTemperature

from custom_components.roommind.managers.thermostat_override_tracker import ThermostatOverrideTracker

CENTRAL = "climate.central"


def _make_hass(unit: str = "°C") -> MagicMock:
    hass = MagicMock()
    hass.config.units.temperature_unit = UnitOfTemperature.FAHRENHEIT if unit == "°F" else UnitOfTemperature.CELSIUS
    hass.states.get = MagicMock(return_value=None)
    return hass


def _state(
    hass: MagicMock,
    state: str = "heat_cool",
    unit: str = "°C",
    **attributes: Any,
) -> SimpleNamespace:
    attrs: dict[str, Any] = {"unit_of_measurement": unit}
    attrs.update(attributes)
    obj = SimpleNamespace(state=state, attributes=attrs)
    hass.states.get = MagicMock(return_value=obj)
    return obj


def _check(
    tracker: ThermostatOverrideTracker,
    hass: MagicMock,
    state_obj: Any,
    *,
    settings: dict[str, Any] | None = None,
    now: float = 1000.0,
) -> dict[str, Any] | None:
    return tracker.check_external_change(hass, CENTRAL, state_obj, settings or {}, 21.0, now_ts=now)


def _setup(unit: str = "°C", **attributes: Any) -> tuple[ThermostatOverrideTracker, MagicMock, SimpleNamespace]:
    hass = _make_hass(unit)
    tracker = ThermostatOverrideTracker()
    state = _state(hass, "heat", unit=unit, temperature=21.0 if unit == "°C" else 68.0, **attributes)
    assert _check(tracker, hass, state) is None
    return tracker, hass, state


# --------------------------------------------------------------------------- #
# Guards and baselines
# --------------------------------------------------------------------------- #


def test_follow_setting_off_ignores_everything() -> None:
    hass = _make_hass()
    tracker = ThermostatOverrideTracker()
    state = _state(hass, "heat", temperature=21.0)
    assert _check(tracker, hass, state, settings={"follow_thermostat_setpoint": False}) is None
    # Turning the option on later still starts with a fresh baseline.
    assert _check(tracker, hass, state) is None
    state = _state(hass, "heat", temperature=25.0)
    assert _check(tracker, hass, state, now=1100.0) == {
        "plant_override_heat": 25.0,
        "plant_override_cool": None,
        "plant_override_until": 8300.0,
    }


def test_missing_entity_or_state_is_ignored() -> None:
    hass = _make_hass()
    tracker = ThermostatOverrideTracker()
    state = _state(hass, "heat", temperature=21.0)
    assert tracker.check_external_change(hass, "", state, {}, None, now_ts=1000.0) is None
    assert tracker.check_external_change(hass, CENTRAL, None, {}, None, now_ts=1000.0) is None


def test_first_valid_observation_only_stores_the_baseline() -> None:
    hass = _make_hass()
    tracker = ThermostatOverrideTracker()
    state = _state(hass, "heat", temperature=21.0)
    assert _check(tracker, hass, state) is None
    # The same reading again can never be a manual change.
    assert _check(tracker, hass, state, now=1030.0) is None


def test_unavailable_resets_the_baseline() -> None:
    hass = _make_hass()
    tracker = ThermostatOverrideTracker()
    _check(tracker, hass, _state(hass, "heat", temperature=21.0))
    assert _check(tracker, hass, _state(hass, "unavailable", temperature=25.0), now=1010.0) is None
    # The first reading after recovery is the new baseline, however different.
    assert _check(tracker, hass, _state(hass, "heat", temperature=25.0), now=1020.0) is None
    assert _check(tracker, hass, _state(hass, "heat", temperature=26.0), now=1030.0) == {
        "plant_override_heat": 26.0,
        "plant_override_cool": None,
        "plant_override_until": 8230.0,
    }


def test_unknown_resets_the_baseline() -> None:
    hass = _make_hass()
    tracker = ThermostatOverrideTracker()
    _check(tracker, hass, _state(hass, "heat", temperature=21.0))
    assert _check(tracker, hass, _state(hass, "unknown"), now=1010.0) is None
    assert _check(tracker, hass, _state(hass, "heat", temperature=24.0), now=1020.0) is None
    assert _check(tracker, hass, _state(hass, "heat", temperature=25.0), now=1030.0) is not None


def test_reset_baseline_can_be_called_explicitly() -> None:
    hass = _make_hass()
    tracker = ThermostatOverrideTracker()
    _check(tracker, hass, _state(hass, "heat", temperature=21.0))
    tracker.reset_baseline(CENTRAL)
    assert _check(tracker, hass, _state(hass, "heat", temperature=25.0), now=1010.0) is None
    tracker.reset_baseline("climate.unknown")  # unknown entity is harmless


def test_off_state_is_ignored_but_keeps_the_baseline() -> None:
    hass = _make_hass()
    tracker = ThermostatOverrideTracker()
    _check(tracker, hass, _state(hass, "heat", temperature=21.0))
    assert _check(tracker, hass, _state(hass, "off", temperature=25.0), now=1010.0) is None
    # Back on, the changed setpoint is judged against the old baseline.
    assert _check(tracker, hass, _state(hass, "heat", temperature=25.0), now=1020.0) is not None


def test_missing_attributes_are_not_a_valid_observation() -> None:
    hass = _make_hass()
    tracker = ThermostatOverrideTracker()
    state = SimpleNamespace(state="heat", attributes=None)
    assert _check(tracker, hass, state) is None
    assert _check(tracker, hass, state, now=1030.0) is None
    # The first real setpoint reading then becomes the baseline.
    assert _check(tracker, hass, _state(hass, "heat", temperature=25.0), now=1040.0) is None


def test_state_without_any_setpoint_is_not_a_valid_observation() -> None:
    hass = _make_hass()
    tracker = ThermostatOverrideTracker()
    assert _check(tracker, hass, _state(hass, "heat"), now=1000.0) is None
    assert _check(tracker, hass, _state(hass, "heat", temperature=21.0), now=1010.0) is None
    assert _check(tracker, hass, _state(hass, "heat", temperature=22.0), now=1020.0) is not None


# --------------------------------------------------------------------------- #
# Tolerance
# --------------------------------------------------------------------------- #


def test_default_celsius_step_tolerance_is_035_c() -> None:
    tracker, hass, _ = _setup()
    assert _check(tracker, hass, _state(hass, "heat", temperature=21.3), now=1010.0) is None
    assert _check(tracker, hass, _state(hass, "heat", temperature=21.7), now=1020.0) == {
        "plant_override_heat": 21.7,
        "plant_override_cool": None,
        "plant_override_until": 8220.0,
    }


def test_celsius_step_widens_the_tolerance() -> None:
    tracker, hass, _ = _setup(target_temp_step=1.0)
    # 0.6 °C is now the smallest change that counts.
    assert _check(tracker, hass, _state(hass, "heat", target_temp_step=1.0, temperature=21.5), now=1010.0) is None
    assert _check(tracker, hass, _state(hass, "heat", target_temp_step=1.0, temperature=22.0), now=1020.0) is None
    assert _check(tracker, hass, _state(hass, "heat", target_temp_step=1.0, temperature=22.7), now=1030.0) is not None


def test_fahrenheit_whole_degree_step_is_detected() -> None:
    tracker, hass, _ = _setup("°F", target_temp_step=1.0)
    result = _check(tracker, hass, _state(hass, "heat", unit="°F", target_temp_step=1.0, temperature=69.0))
    assert result is not None
    assert result["plant_override_heat"] == pytest.approx(20.56)


def test_half_a_fahrenheit_degree_is_inside_the_tolerance() -> None:
    tracker, hass, _ = _setup("°F", target_temp_step=1.0)
    assert _check(tracker, hass, _state(hass, "heat", unit="°F", target_temp_step=1.0, temperature=68.5)) is None


def test_fahrenheit_default_step_is_one_degree() -> None:
    tracker, hass, _ = _setup("°F")
    assert _check(tracker, hass, _state(hass, "heat", unit="°F", temperature=69.0)) is not None


def test_invalid_step_falls_back_to_the_default() -> None:
    tracker, hass, _ = _setup(target_temp_step=0)
    assert _check(tracker, hass, _state(hass, "heat", target_temp_step=0, temperature=21.3)) is None
    assert _check(tracker, hass, _state(hass, "heat", target_temp_step="wide", temperature=21.8)) is not None


def test_missing_unit_attribute_falls_back_to_the_ha_unit() -> None:
    hass = _make_hass("°F")
    tracker = ThermostatOverrideTracker()
    state = SimpleNamespace(state="heat", attributes={"temperature": 68.0, "target_temp_step": 1.0})
    hass.states.get = MagicMock(return_value=state)
    assert _check(tracker, hass, state) is None
    state = SimpleNamespace(state="heat", attributes={"temperature": 69.0, "target_temp_step": 1.0})
    hass.states.get = MagicMock(return_value=state)
    result = _check(tracker, hass, state, now=1010.0)
    assert result is not None
    assert result["plant_override_heat"] == pytest.approx(20.56)


# --------------------------------------------------------------------------- #
# Single setpoint
# --------------------------------------------------------------------------- #


def test_heating_change_creates_a_heat_override() -> None:
    tracker, hass, _ = _setup()
    state = _state(hass, "heat", temperature=23.0)
    assert _check(tracker, hass, state, now=1000.0) == {
        "plant_override_heat": 23.0,
        "plant_override_cool": None,
        "plant_override_until": 8200.0,
    }


def test_cooling_change_creates_a_cool_override() -> None:
    tracker, hass, _ = _setup()
    state = _state(hass, "cool", temperature=24.0)
    assert _check(tracker, hass, state, now=1000.0) == {
        "plant_override_heat": None,
        "plant_override_cool": 24.0,
        "plant_override_until": 8200.0,
    }


def test_auto_mode_single_change_defaults_to_heat() -> None:
    tracker, hass, _ = _setup()
    state = _state(hass, "heat_cool", temperature=23.0)
    result = _check(tracker, hass, state, now=1000.0)
    assert result is not None
    assert result["plant_override_heat"] == 23.0


@pytest.mark.parametrize(
    ("hours", "until"),
    [("0", None), (-2, None), ("3", 1000.0 + 3 * 3600.0), ("oops", 1000.0 + 2 * 3600.0)],
)
def test_override_hours_control_the_expiry(hours: Any, until: float | None) -> None:
    tracker, hass, _ = _setup()
    state = _state(hass, "heat", temperature=23.0)
    result = _check(tracker, hass, state, settings={"thermostat_override_hours": hours}, now=1000.0)
    assert result is not None
    assert result["plant_override_until"] == until


def test_missing_hours_use_the_two_hour_default() -> None:
    tracker, hass, _ = _setup()
    result = _check(tracker, hass, _state(hass, "heat", temperature=23.0), now=1000.0)
    assert result is not None
    assert result["plant_override_until"] == 8200.0


# --------------------------------------------------------------------------- #
# Range setpoints
# --------------------------------------------------------------------------- #


def test_range_change_reports_both_sides() -> None:
    hass = _make_hass()
    tracker = ThermostatOverrideTracker()
    _check(tracker, hass, _state(hass, "heat_cool", target_temp_low=19.0, target_temp_high=24.0))
    result = _check(
        tracker,
        hass,
        _state(hass, "heat_cool", target_temp_low=20.0, target_temp_high=25.0),
        now=1000.0,
    )
    assert result == {
        "plant_override_heat": 20.0,
        "plant_override_cool": 25.0,
        "plant_override_until": 8200.0,
    }


def test_one_changed_range_side_reports_the_whole_range() -> None:
    hass = _make_hass()
    tracker = ThermostatOverrideTracker()
    _check(tracker, hass, _state(hass, "heat_cool", target_temp_low=19.0, target_temp_high=24.0))
    result = _check(
        tracker,
        hass,
        _state(hass, "heat_cool", target_temp_low=19.0, target_temp_high=25.0),
        now=1030.0,
    )
    assert result == {
        "plant_override_heat": 19.0,
        "plant_override_cool": 25.0,
        "plant_override_until": 8230.0,
    }


def test_range_change_takes_precedence_over_a_single_change() -> None:
    hass = _make_hass()
    tracker = ThermostatOverrideTracker()
    _check(tracker, hass, _state(hass, "heat_cool", target_temp_low=19.0, target_temp_high=24.0))
    result = _check(
        tracker,
        hass,
        _state(hass, "heat_cool", target_temp_low=20.0, target_temp_high=25.0, temperature=30.0),
        now=1030.0,
    )
    assert result is not None
    assert result["plant_override_heat"] == 20.0
    assert result["plant_override_cool"] == 25.0


def test_a_new_range_after_a_single_baseline_is_a_change() -> None:
    tracker, hass, _ = _setup()
    result = _check(
        tracker,
        hass,
        _state(hass, "heat_cool", target_temp_low=20.0, target_temp_high=25.0),
        now=1010.0,
    )
    assert result == {
        "plant_override_heat": 20.0,
        "plant_override_cool": 25.0,
        "plant_override_until": 8210.0,
    }


def test_a_new_single_after_a_range_baseline_is_a_change() -> None:
    hass = _make_hass()
    tracker = ThermostatOverrideTracker()
    _check(tracker, hass, _state(hass, "heat_cool", target_temp_low=19.0, target_temp_high=24.0))
    result = _check(tracker, hass, _state(hass, "heat", temperature=22.0), now=1030.0)
    assert result == {
        "plant_override_heat": 22.0,
        "plant_override_cool": None,
        "plant_override_until": 8230.0,
    }


def test_partial_range_reading_without_single_setpoint_is_not_a_change() -> None:
    hass = _make_hass()
    tracker = ThermostatOverrideTracker()
    _check(tracker, hass, _state(hass, "heat_cool", target_temp_low=19.0, target_temp_high=24.0))
    # Only one range side reported: nothing to compare, so no override.
    assert _check(tracker, hass, _state(hass, "heat_cool", target_temp_low=20.0), now=1030.0) is None


def test_fahrenheit_range_change_is_reported_in_celsius() -> None:
    hass = _make_hass("°F")
    tracker = ThermostatOverrideTracker()
    _check(tracker, hass, _state(hass, "heat_cool", unit="°F", target_temp_low=66.0, target_temp_high=75.0))
    result = _check(
        tracker,
        hass,
        _state(hass, "heat_cool", unit="°F", target_temp_low=68.0, target_temp_high=77.0),
        now=1030.0,
    )
    assert result is not None
    assert result["plant_override_heat"] == pytest.approx(20.0)
    assert result["plant_override_cool"] == pytest.approx(25.0)


# --------------------------------------------------------------------------- #
# RoomMind echoes
# --------------------------------------------------------------------------- #


def test_single_setpoint_echo_within_grace_is_ignored() -> None:
    tracker, hass, _ = _setup()
    tracker.record_sent(CENTRAL, single_c=23.0, hvac_mode="heat", now_ts=1000.0)
    assert _check(tracker, hass, _state(hass, "heat", temperature=23.0), now=1030.0) is None
    # The echo became the new baseline.
    assert _check(tracker, hass, _state(hass, "heat", temperature=23.0), now=1040.0) is None


def test_an_echo_is_ours_however_late_the_device_reports_it() -> None:
    """A thermostat can take a long time to report what it was told.

    The match used to expire 75 s after the send, so any control cycle longer
    than that turned RoomMind's own nudge into a manual override.
    """
    tracker, hass, _ = _setup()
    tracker.record_sent(CENTRAL, single_c=23.0, now_ts=1000.0)
    assert _check(tracker, hass, _state(hass, "heat", temperature=23.0), now=1000.0 + 600) is None
    # The echo became the new baseline, so a second look is still quiet.
    assert _check(tracker, hass, _state(hass, "heat", temperature=23.0), now=1000.0 + 630) is None


def test_an_echo_of_an_earlier_command_is_still_ours() -> None:
    """A cloud device can report the commands out of order; either is our echo."""
    tracker, hass, _ = _setup()
    tracker.record_sent(CENTRAL, single_c=23.0, now_ts=1000.0)
    tracker.record_sent(CENTRAL, single_c=24.0, now_ts=1030.0)
    assert _check(tracker, hass, _state(hass, "heat", temperature=23.0), now=1060.0) is None


def test_a_different_value_is_external_however_late_it_arrives() -> None:
    tracker, hass, _ = _setup()
    tracker.record_sent(CENTRAL, single_c=23.0, now_ts=1000.0)
    result = _check(tracker, hass, _state(hass, "heat", temperature=25.0), now=1000.0 + 600)
    assert result is not None
    assert result["plant_override_heat"] == 25.0


def test_a_different_value_is_not_an_echo() -> None:
    tracker, hass, _ = _setup()
    tracker.record_sent(CENTRAL, single_c=23.0, now_ts=1000.0)
    result = _check(tracker, hass, _state(hass, "heat", temperature=24.0), now=1030.0)
    assert result is not None
    assert result["plant_override_heat"] == 24.0


def test_range_echo_is_ignored() -> None:
    hass = _make_hass()
    tracker = ThermostatOverrideTracker()
    _check(tracker, hass, _state(hass, "heat_cool", target_temp_low=19.0, target_temp_high=24.0))
    tracker.record_sent(CENTRAL, low_c=20.0, high_c=25.0, hvac_mode="heat_cool", now_ts=1000.0)
    assert (
        _check(
            tracker,
            hass,
            _state(hass, "heat_cool", target_temp_low=20.1, target_temp_high=24.9),
            now=1030.0,
        )
        is None
    )


def test_range_echo_with_one_mismatched_side_is_external() -> None:
    hass = _make_hass()
    tracker = ThermostatOverrideTracker()
    _check(tracker, hass, _state(hass, "heat_cool", target_temp_low=19.0, target_temp_high=24.0))
    tracker.record_sent(CENTRAL, low_c=20.0, high_c=25.0, now_ts=1000.0)
    result = _check(
        tracker,
        hass,
        _state(hass, "heat_cool", target_temp_low=20.1, target_temp_high=26.0),
        now=1030.0,
    )
    assert result is not None
    assert result["plant_override_cool"] == 26.0


def test_sent_values_without_a_comparable_setpoint_do_not_hide_a_change() -> None:
    tracker, hass, _ = _setup()
    tracker.record_sent(CENTRAL, hvac_mode="heat", now_ts=1000.0)
    result = _check(tracker, hass, _state(hass, "heat", temperature=23.0), now=1030.0)
    assert result is not None


def test_partial_sent_range_does_not_match_a_range_echo() -> None:
    hass = _make_hass()
    tracker = ThermostatOverrideTracker()
    _check(tracker, hass, _state(hass, "heat_cool", target_temp_low=19.0, target_temp_high=24.0))
    tracker.record_sent(CENTRAL, low_c=20.0, now_ts=1000.0)
    result = _check(
        tracker,
        hass,
        _state(hass, "heat_cool", target_temp_low=20.0, target_temp_high=25.0),
        now=1030.0,
    )
    assert result is not None


def test_record_sent_for_an_empty_entity_is_a_no_op() -> None:
    tracker, hass, _ = _setup()
    tracker.record_sent("", single_c=23.0, now_ts=1000.0)
    result = _check(tracker, hass, _state(hass, "heat", temperature=23.0), now=1030.0)
    assert result is not None


def test_record_sent_tolerates_non_numeric_values() -> None:
    tracker, hass, _ = _setup()
    tracker.record_sent(CENTRAL, low_c="warm", high_c=["not", "a", "number"], now_ts=1000.0)  # type: ignore[arg-type]
    result = _check(tracker, hass, _state(hass, "heat", temperature=23.0), now=1030.0)
    assert result is not None


def test_echo_baseline_update_prevents_a_second_override() -> None:
    tracker, hass, _ = _setup()
    tracker.record_sent(CENTRAL, single_c=23.0, now_ts=1000.0)
    assert _check(tracker, hass, _state(hass, "heat", temperature=23.0), now=1010.0) is None
    assert _check(tracker, hass, _state(hass, "heat", temperature=23.0), now=1020.0) is None
