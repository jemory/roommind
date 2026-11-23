"""Tests for schedule_utils.py."""

from __future__ import annotations

import contextlib
import logging
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest
from homeassistant.util import dt as dt_util

from custom_components.roommind.const import SCHEDULE_UNREADABLE_HOLD_HOURS, TargetTemps
from custom_components.roommind.utils.price_utils import LEVEL_CHEAP, PriceSignal
from custom_components.roommind.utils.schedule_utils import (
    HouseContext,
    TargetSource,
    band_in_force,
    find_active_block,
    get_active_schedule_entity,
    house_override_active_at,
    house_override_hold_until,
    house_target_source,
    is_plant_override_at,
    make_house_target_resolver,
    make_target_resolver,
    mask_override_band,
    override_hold_until,
    override_preset_band,
    price_shifted,
    read_schedule_blocks,
    resolve_house_targets_at_time,
    resolve_schedule_index,
    resolve_target_at_time,
    resolve_targets_at_time,
    resolve_targets_with_source,
    schedule_block_sets_target,
    schedule_sets_target,
    set_band_sides,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hass(entity_id: str | None = None, state_value: str | None = None):
    """Return a mock hass with optional entity state."""
    hass = MagicMock()
    if entity_id is None:
        hass.states.get.return_value = None
    else:
        mock_state = MagicMock()
        mock_state.state = state_value
        hass.states.get.return_value = mock_state
    return hass


def _make_room(**overrides):
    room = {
        "area_id": "bedroom",
        "schedules": [
            {"entity_id": "schedule.bedroom_weekday"},
            {"entity_id": "schedule.bedroom_weekend"},
        ],
        "schedule_selector_entity": "",
        "comfort_temp": 21.0,
        "eco_temp": 18.0,
    }
    room.update(overrides)
    return room


# ---------------------------------------------------------------------------
# resolve_schedule_index
# ---------------------------------------------------------------------------


class TestResolveScheduleIndex:
    """Tests for resolve_schedule_index."""

    def test_no_schedules_returns_minus_one(self):
        """Empty schedules list returns -1."""
        hass = _make_hass()
        room = _make_room(schedules=[])
        assert resolve_schedule_index(hass, room) == -1

    def test_no_selector_entity_returns_zero(self):
        """Without selector entity, first schedule is selected."""
        hass = _make_hass()
        room = _make_room(schedule_selector_entity="")
        assert resolve_schedule_index(hass, room) == 0

    def test_input_boolean_on_returns_one(self):
        """input_boolean on → index 1."""
        hass = _make_hass("input_boolean.schedule_mode", "on")
        room = _make_room(schedule_selector_entity="input_boolean.schedule_mode")
        assert resolve_schedule_index(hass, room) == 1

    def test_input_boolean_off_returns_zero(self):
        """input_boolean off → index 0."""
        hass = _make_hass("input_boolean.schedule_mode", "off")
        room = _make_room(schedule_selector_entity="input_boolean.schedule_mode")
        assert resolve_schedule_index(hass, room) == 0

    def test_input_number_one_based_to_zero_based(self):
        """input_number value 2 → index 1 (1-based to 0-based)."""
        hass = _make_hass("input_number.schedule_select", "2")
        room = _make_room(schedule_selector_entity="input_number.schedule_select")
        assert resolve_schedule_index(hass, room) == 1

    def test_input_number_value_one_returns_zero(self):
        """input_number value 1 → index 0."""
        hass = _make_hass("input_number.schedule_select", "1")
        room = _make_room(schedule_selector_entity="input_number.schedule_select")
        assert resolve_schedule_index(hass, room) == 0

    def test_input_number_out_of_range_returns_minus_one(self):
        """input_number value beyond schedule count → -1."""
        hass = _make_hass("input_number.schedule_select", "10")
        room = _make_room(schedule_selector_entity="input_number.schedule_select")
        assert resolve_schedule_index(hass, room) == -1

    def test_input_number_zero_returns_minus_one(self):
        """input_number value 0 (1-based) → index -1 (out of range)."""
        hass = _make_hass("input_number.schedule_select", "0")
        room = _make_room(schedule_selector_entity="input_number.schedule_select")
        assert resolve_schedule_index(hass, room) == -1

    def test_input_number_invalid_value_returns_zero(self):
        """input_number with non-numeric value → fallback 0."""
        hass = _make_hass("input_number.schedule_select", "abc")
        room = _make_room(schedule_selector_entity="input_number.schedule_select")
        assert resolve_schedule_index(hass, room) == 0

    def test_unavailable_entity_returns_zero(self):
        """Unavailable entity → fallback to first schedule (0)."""
        hass = _make_hass("input_boolean.schedule_mode", "unavailable")
        room = _make_room(schedule_selector_entity="input_boolean.schedule_mode")
        assert resolve_schedule_index(hass, room) == 0

    def test_unknown_entity_returns_zero(self):
        """Unknown entity state → fallback to first schedule (0)."""
        hass = _make_hass("input_boolean.schedule_mode", "unknown")
        room = _make_room(schedule_selector_entity="input_boolean.schedule_mode")
        assert resolve_schedule_index(hass, room) == 0

    def test_missing_entity_returns_zero(self):
        """Entity not found in hass.states → fallback 0."""
        hass = MagicMock()
        hass.states.get.return_value = None
        room = _make_room(schedule_selector_entity="input_boolean.nonexistent")
        assert resolve_schedule_index(hass, room) == 0

    def test_unknown_domain_returns_zero(self):
        """Unrecognized entity domain → fallback 0."""
        hass = _make_hass("sensor.something", "42")
        room = _make_room(schedule_selector_entity="sensor.something")
        assert resolve_schedule_index(hass, room) == 0

    def test_input_number_float_value(self):
        """input_number with float string like '2.0' → index 1."""
        hass = _make_hass("input_number.schedule_select", "2.0")
        room = _make_room(schedule_selector_entity="input_number.schedule_select")
        assert resolve_schedule_index(hass, room) == 1


# ---------------------------------------------------------------------------
# resolve_target_at_time
# ---------------------------------------------------------------------------


class TestResolveTargetAtTime:
    """Tests for resolve_target_at_time."""

    def test_override_active(self):
        """Active override returns override_temp."""
        now = time.time()
        result = resolve_target_at_time(
            ts=now,
            schedule_blocks=None,
            override_until=now + 3600,
            override_temp=25.0,
            vacation_until=None,
            vacation_temp=None,
            comfort_temp=21.0,
            eco_temp=18.0,
        )
        assert result == 25.0

    def test_override_expired(self):
        """Expired override falls through to next priority."""
        now = time.time()
        result = resolve_target_at_time(
            ts=now,
            schedule_blocks=None,
            override_until=now - 100,
            override_temp=25.0,
            vacation_until=None,
            vacation_temp=None,
            comfort_temp=21.0,
            eco_temp=18.0,
        )
        # No vacation, no blocks → comfort_temp
        assert result == 21.0

    def test_vacation_active(self):
        """Active vacation returns vacation_temp."""
        now = time.time()
        result = resolve_target_at_time(
            ts=now,
            schedule_blocks=None,
            override_until=None,
            override_temp=None,
            vacation_until=now + 86400,
            vacation_temp=16.0,
            comfort_temp=21.0,
            eco_temp=18.0,
        )
        assert result == 16.0

    def test_override_beats_vacation(self):
        """Override has higher priority than vacation."""
        now = time.time()
        result = resolve_target_at_time(
            ts=now,
            schedule_blocks=None,
            override_until=now + 3600,
            override_temp=25.0,
            vacation_until=now + 86400,
            vacation_temp=16.0,
            comfort_temp=21.0,
            eco_temp=18.0,
        )
        assert result == 25.0

    def test_presence_away_returns_eco(self):
        """presence_away=True returns eco_temp."""
        now = time.time()
        result = resolve_target_at_time(
            ts=now,
            schedule_blocks={"monday": []},
            override_until=None,
            override_temp=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_temp=21.0,
            eco_temp=18.0,
            presence_away=True,
        )
        assert result == 18.0

    def test_no_schedule_blocks_returns_comfort(self):
        """schedule_blocks=None returns comfort_temp."""
        now = time.time()
        result = resolve_target_at_time(
            ts=now,
            schedule_blocks=None,
            override_until=None,
            override_temp=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_temp=21.0,
            eco_temp=18.0,
        )
        assert result == 21.0

    def test_inside_block_with_temperature(self):
        """Inside a schedule block that has a temperature → returns that temperature."""
        # Create a timestamp that falls on a known day and time
        # Use a Monday at 10:00
        dt = datetime(2025, 1, 6, 10, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE)  # Monday
        ts = dt.timestamp()
        schedule_blocks = {
            "monday": [
                {
                    "from": "08:00:00",
                    "to": "12:00:00",
                    "data": {"temperature": 22.5},
                },
            ],
        }
        result = resolve_target_at_time(
            ts=ts,
            schedule_blocks=schedule_blocks,
            override_until=None,
            override_temp=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_temp=21.0,
            eco_temp=18.0,
        )
        assert result == 22.5

    def test_inside_block_without_temperature(self):
        """Inside a block without temperature data → returns comfort_temp."""
        dt = datetime(2025, 1, 6, 10, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE)  # Monday
        ts = dt.timestamp()
        schedule_blocks = {
            "monday": [
                {
                    "from": "08:00:00",
                    "to": "12:00:00",
                    "data": {},
                },
            ],
        }
        result = resolve_target_at_time(
            ts=ts,
            schedule_blocks=schedule_blocks,
            override_until=None,
            override_temp=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_temp=21.0,
            eco_temp=18.0,
        )
        assert result == 21.0

    def test_outside_all_blocks_returns_eco(self):
        """Outside all schedule blocks → returns eco_temp."""
        dt = datetime(2025, 1, 6, 6, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE)  # Monday 06:00 — before any block
        ts = dt.timestamp()
        schedule_blocks = {
            "monday": [
                {
                    "from": "08:00:00",
                    "to": "12:00:00",
                    "data": {"temperature": 22.0},
                },
            ],
        }
        result = resolve_target_at_time(
            ts=ts,
            schedule_blocks=schedule_blocks,
            override_until=None,
            override_temp=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_temp=21.0,
            eco_temp=18.0,
        )
        assert result == 18.0

    def test_day_with_no_blocks_returns_eco(self):
        """Day without any blocks → returns eco_temp."""
        dt = datetime(2025, 1, 7, 10, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE)  # Tuesday
        ts = dt.timestamp()
        schedule_blocks = {
            "monday": [
                {
                    "from": "08:00:00",
                    "to": "12:00:00",
                    "data": {"temperature": 22.0},
                },
            ],
        }
        result = resolve_target_at_time(
            ts=ts,
            schedule_blocks=schedule_blocks,
            override_until=None,
            override_temp=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_temp=21.0,
            eco_temp=18.0,
        )
        assert result == 18.0

    def test_block_with_invalid_temperature(self):
        """Block with non-numeric temperature → falls back to comfort_temp."""
        dt = datetime(2025, 1, 6, 10, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE)  # Monday
        ts = dt.timestamp()
        schedule_blocks = {
            "monday": [
                {
                    "from": "08:00:00",
                    "to": "12:00:00",
                    "data": {"temperature": "not_a_number"},
                },
            ],
        }
        result = resolve_target_at_time(
            ts=ts,
            schedule_blocks=schedule_blocks,
            override_until=None,
            override_temp=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_temp=21.0,
            eco_temp=18.0,
        )
        assert result == 21.0

    def test_vacation_expired_falls_through(self):
        """Expired vacation falls through to schedule."""
        dt = datetime(2025, 1, 6, 10, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE)  # Monday
        ts = dt.timestamp()
        schedule_blocks = {
            "monday": [
                {
                    "from": "08:00:00",
                    "to": "12:00:00",
                    "data": {"temperature": 22.0},
                },
            ],
        }
        result = resolve_target_at_time(
            ts=ts,
            schedule_blocks=schedule_blocks,
            override_until=None,
            override_temp=None,
            vacation_until=ts - 100,  # expired
            vacation_temp=16.0,
            comfort_temp=21.0,
            eco_temp=18.0,
        )
        assert result == 22.0

    def test_presence_away_action_off_returns_none(self):
        """When presence_away_action is 'off' and presence away, returns None."""
        now = time.time()
        result = resolve_target_at_time(
            ts=now,
            schedule_blocks=None,
            override_until=None,
            override_temp=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_temp=21.0,
            eco_temp=18.0,
            presence_away=True,
            presence_away_action="off",
        )
        assert result is None

    def test_presence_away_action_eco_returns_eco(self):
        """When presence_away_action is 'eco' and presence away, returns eco_temp."""
        now = time.time()
        result = resolve_target_at_time(
            ts=now,
            schedule_blocks=None,
            override_until=None,
            override_temp=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_temp=21.0,
            eco_temp=18.0,
            presence_away=True,
            presence_away_action="eco",
        )
        assert result == 18.0

    def test_schedule_off_action_off_returns_none(self):
        """When schedule_off_action is 'off' and outside schedule blocks, returns None."""
        dt = datetime(2025, 1, 6, 23, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE)  # Monday 23:00 - outside blocks
        ts = dt.timestamp()
        schedule_blocks = {
            "monday": [
                {"from": "08:00:00", "to": "12:00:00", "data": {"temperature": 22.0}},
            ],
        }
        result = resolve_target_at_time(
            ts=ts,
            schedule_blocks=schedule_blocks,
            override_until=None,
            override_temp=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_temp=21.0,
            eco_temp=18.0,
            schedule_off_action="off",
        )
        assert result is None

    def test_schedule_off_action_eco_returns_eco(self):
        """When schedule_off_action is 'eco' (default), returns eco_temp."""
        dt = datetime(2025, 1, 6, 23, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE)  # Monday 23:00 - outside blocks
        ts = dt.timestamp()
        schedule_blocks = {
            "monday": [
                {"from": "08:00:00", "to": "12:00:00", "data": {"temperature": 22.0}},
            ],
        }
        result = resolve_target_at_time(
            ts=ts,
            schedule_blocks=schedule_blocks,
            override_until=None,
            override_temp=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_temp=21.0,
            eco_temp=18.0,
            schedule_off_action="eco",
        )
        assert result == 18.0

    def test_override_beats_presence_away_off(self):
        """Active override takes priority even when presence_away_action is 'off'."""
        now = time.time()
        result = resolve_target_at_time(
            ts=now,
            schedule_blocks=None,
            override_until=now + 3600,
            override_temp=25.0,
            vacation_until=None,
            vacation_temp=None,
            comfort_temp=21.0,
            eco_temp=18.0,
            presence_away=True,
            presence_away_action="off",
        )
        assert result == 25.0


class TestMakeTargetResolverOffActions:
    """Tests for off actions in make_target_resolver."""

    def test_resolver_returns_none_for_presence_off(self):
        """Resolver returns None when presence_away_action is 'off' and away."""
        room = {"comfort_temp": 21.0, "eco_temp": 17.0}
        settings = {"presence_away_action": "off"}
        resolver = make_target_resolver(
            None,
            room,
            settings,
            presence_away=True,
        )
        assert resolver(time.time()) == TargetTemps(heat=None, cool=None)

    def test_resolver_returns_none_for_schedule_off(self):
        """Resolver returns None when schedule_off_action is 'off' and outside blocks."""
        room = {"comfort_temp": 21.0, "eco_temp": 17.0}
        settings = {"schedule_off_action": "off"}
        make_target_resolver(
            None,
            room,
            settings,
        )
        # No schedule blocks → falls through to eco/off logic
        # Actually with no schedule blocks, it returns comfort_temp (no blocks = comfort)
        # So we need schedule_blocks with no matching block
        dt = datetime(2025, 1, 6, 23, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE)
        ts = dt.timestamp()
        blocks = {"monday": [{"from": "08:00:00", "to": "12:00:00", "data": {"temperature": 22.0}}]}
        resolver2 = make_target_resolver(blocks, room, settings)
        assert resolver2(ts) == TargetTemps(heat=None, cool=None)

    def test_resolver_skips_mold_delta_when_none(self):
        """Mold delta is NOT added when base target is None."""
        room = {"comfort_temp": 21.0, "eco_temp": 17.0}
        settings = {"presence_away_action": "off"}
        resolver = make_target_resolver(
            None,
            room,
            settings,
            presence_away=True,
            mold_prevention_delta=2.0,
        )
        assert resolver(time.time()) == TargetTemps(heat=None, cool=None)


# ---------------------------------------------------------------------------
# get_active_schedule_entity
# ---------------------------------------------------------------------------


class TestGetActiveScheduleEntity:
    """Tests for get_active_schedule_entity."""

    def test_returns_entity_for_first_schedule(self):
        """No selector → returns first schedule entity."""
        hass = _make_hass()
        room = _make_room()
        result = get_active_schedule_entity(hass, room)
        assert result == "schedule.bedroom_weekday"

    def test_returns_entity_for_second_schedule(self):
        """input_boolean on → returns second schedule entity."""
        hass = _make_hass("input_boolean.schedule_mode", "on")
        room = _make_room(schedule_selector_entity="input_boolean.schedule_mode")
        result = get_active_schedule_entity(hass, room)
        assert result == "schedule.bedroom_weekend"

    def test_no_schedules_returns_none(self):
        """No schedules → None."""
        hass = _make_hass()
        room = _make_room(schedules=[])
        result = get_active_schedule_entity(hass, room)
        assert result is None

    def test_index_out_of_range_returns_none(self):
        """Out-of-range index → None."""
        hass = _make_hass("input_number.schedule_select", "10")
        room = _make_room(schedule_selector_entity="input_number.schedule_select")
        result = get_active_schedule_entity(hass, room)
        assert result is None

    def test_empty_entity_id_returns_none(self):
        """Schedule with empty entity_id → None."""
        hass = _make_hass()
        room = _make_room(schedules=[{"entity_id": ""}])
        result = get_active_schedule_entity(hass, room)
        assert result is None

    def test_schedule_without_entity_id_returns_none(self):
        """Schedule dict missing entity_id key → None."""
        hass = _make_hass()
        room = _make_room(schedules=[{}])
        result = get_active_schedule_entity(hass, room)
        assert result is None


# ---------------------------------------------------------------------------
# make_target_resolver with mold_prevention_delta
# ---------------------------------------------------------------------------


class TestMakeTargetResolverMoldDelta:
    """Tests for mold_prevention_delta parameter in make_target_resolver."""

    def test_resolver_adds_mold_delta(self):
        """Resolver should add mold_prevention_delta to every resolved target."""
        room = {"comfort_temp": 21.0, "eco_temp": 17.0}
        settings: dict = {}
        resolver = make_target_resolver(
            None,
            room,
            settings,
            mold_prevention_delta=2.0,
        )
        # No schedule → comfort_temp 21 + delta 2 = 23 (mold delta only on .heat)
        assert resolver(time.time()).heat == 23.0

    def test_resolver_zero_delta_no_change(self):
        """With zero delta, resolver returns base target unchanged."""
        room = {"comfort_temp": 21.0, "eco_temp": 17.0}
        settings: dict = {}
        resolver = make_target_resolver(
            None,
            room,
            settings,
            mold_prevention_delta=0.0,
        )
        assert resolver(time.time()).heat == 21.0


class TestFahrenheitBlockConversion:
    """Tests for Fahrenheit temperature handling in schedule resolution."""

    def test_block_temp_converter_converts_fahrenheit_to_celsius(self):
        """resolve_target_at_time applies block_temp_converter to schedule block temps."""
        from datetime import datetime

        dt = datetime(2025, 1, 6, 10, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE)  # Monday 10:00
        ts = dt.timestamp()
        schedule_blocks = {
            "monday": [
                {
                    "from": "08:00:00",
                    "to": "12:00:00",
                    # Block temp stored in Fahrenheit: 71.6°F = 22°C
                    "data": {"temperature": 71.6},
                },
            ],
        }
        # Converter simulates Fahrenheit → Celsius
        converter = lambda v: (v - 32) * 5 / 9  # noqa: E731

        result = resolve_target_at_time(
            ts=ts,
            schedule_blocks=schedule_blocks,
            override_until=None,
            override_temp=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_temp=21.0,
            eco_temp=18.0,
            block_temp_converter=converter,
        )
        import pytest as _pytest

        assert result == _pytest.approx(22.0, abs=0.1)

    def test_block_temp_converter_not_applied_without_block_temp(self):
        """Converter is not called when block has no temperature data."""
        from datetime import datetime

        dt = datetime(2025, 1, 6, 10, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE)  # Monday 10:00
        ts = dt.timestamp()
        schedule_blocks = {
            "monday": [
                {
                    "from": "08:00:00",
                    "to": "12:00:00",
                    "data": {},  # no temperature
                },
            ],
        }
        converter_called = False

        def converter(v):
            nonlocal converter_called
            converter_called = True
            return v

        result = resolve_target_at_time(
            ts=ts,
            schedule_blocks=schedule_blocks,
            override_until=None,
            override_temp=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_temp=21.0,
            eco_temp=18.0,
            block_temp_converter=converter,
        )
        assert result == 21.0  # falls back to comfort_temp
        assert not converter_called

    def test_make_target_resolver_fahrenheit_block_conversion(self):
        """make_target_resolver with Fahrenheit hass converts block temps to Celsius."""
        from datetime import datetime

        from homeassistant.const import UnitOfTemperature

        hass = _make_hass()
        hass.config.units.temperature_unit = UnitOfTemperature.FAHRENHEIT

        room = _make_room(comfort_temp=21.0, eco_temp=18.0)
        settings: dict = {}

        # Monday 10:00 schedule block with 71.6°F
        dt = datetime(2025, 1, 6, 10, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE)
        ts = dt.timestamp()
        schedule_blocks = {
            "monday": [
                {
                    "from": "08:00:00",
                    "to": "12:00:00",
                    "data": {"temperature": 71.6},  # 71.6°F = 22°C
                },
            ],
        }

        resolver = make_target_resolver(
            schedule_blocks,
            room,
            settings,
            hass=hass,
        )
        import pytest as _pytest

        assert resolver(ts).heat == _pytest.approx(22.0, abs=0.1)

    def test_make_target_resolver_celsius_no_conversion(self):
        """make_target_resolver with Celsius hass does not alter block temps."""
        from datetime import datetime

        from homeassistant.const import UnitOfTemperature

        hass = _make_hass()
        hass.config.units.temperature_unit = UnitOfTemperature.CELSIUS

        room = _make_room(comfort_temp=21.0, eco_temp=18.0)
        settings: dict = {}

        dt = datetime(2025, 1, 6, 10, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE)
        ts = dt.timestamp()
        schedule_blocks = {
            "monday": [
                {
                    "from": "08:00:00",
                    "to": "12:00:00",
                    "data": {"temperature": 22.5},
                },
            ],
        }

        resolver = make_target_resolver(
            schedule_blocks,
            room,
            settings,
            hass=hass,
        )
        assert resolver(ts).heat == 22.5


# ---------------------------------------------------------------------------
# resolve_targets_at_time (dual heat/cool)
# ---------------------------------------------------------------------------


class TestResolveTargetsAtTime:
    """Tests for the new resolve_targets_at_time with split heat/cool fields."""

    def test_comfort_fields_when_schedule_on(self):
        """Inside a schedule block without custom temp returns comfort_heat/comfort_cool."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        dt = datetime(2025, 1, 6, 10, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE)  # Monday 10:00
        ts = dt.timestamp()
        schedule_blocks = {
            "monday": [
                {"from": "08:00:00", "to": "12:00:00", "data": {}},
            ],
        }
        result = resolve_targets_at_time(
            ts=ts,
            schedule_blocks=schedule_blocks,
            override_until=None,
            override_heat=None,
            override_cool=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=21.0,
            comfort_cool=24.0,
            eco_heat=17.0,
            eco_cool=27.0,
        )
        assert result == TargetTemps(heat=21.0, cool=24.0)

    def test_eco_fields_when_schedule_off(self):
        """Outside schedule blocks returns eco_heat/eco_cool."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        dt = datetime(2025, 1, 6, 6, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE)  # Monday 06:00 - before blocks
        ts = dt.timestamp()
        schedule_blocks = {
            "monday": [
                {"from": "08:00:00", "to": "12:00:00", "data": {}},
            ],
        }
        result = resolve_targets_at_time(
            ts=ts,
            schedule_blocks=schedule_blocks,
            override_until=None,
            override_heat=None,
            override_cool=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=21.0,
            comfort_cool=24.0,
            eco_heat=17.0,
            eco_cool=27.0,
        )
        assert result == TargetTemps(heat=17.0, cool=27.0)

    def test_presence_away_action_off_returns_none_none(self):
        """presence_away_action='off' returns TargetTemps(None, None)."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        now = time.time()
        result = resolve_targets_at_time(
            ts=now,
            schedule_blocks=None,
            override_until=None,
            override_heat=None,
            override_cool=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=21.0,
            comfort_cool=24.0,
            eco_heat=17.0,
            eco_cool=27.0,
            presence_away=True,
            presence_away_action="off",
        )
        assert result == TargetTemps(heat=None, cool=None)

    def test_presence_away_eco_returns_eco_temps(self):
        """presence_away_action='eco' returns eco heat/cool."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        now = time.time()
        result = resolve_targets_at_time(
            ts=now,
            schedule_blocks=None,
            override_until=None,
            override_heat=None,
            override_cool=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=21.0,
            comfort_cool=24.0,
            eco_heat=17.0,
            eco_cool=27.0,
            presence_away=True,
            presence_away_action="eco",
        )
        assert result == TargetTemps(heat=17.0, cool=27.0)

    def test_override_creates_single_point(self):
        """Active override creates TargetTemps(heat=override, cool=override)."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        now = time.time()
        result = resolve_targets_at_time(
            ts=now,
            schedule_blocks=None,
            override_until=now + 3600,
            override_heat=25.0,
            override_cool=25.0,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=21.0,
            comfort_cool=24.0,
            eco_heat=17.0,
            eco_cool=27.0,
        )
        assert result == TargetTemps(heat=25.0, cool=25.0)

    def test_presence_clears_override_suppresses_override(self):
        """presence_clears_override=True + presence_away skips override branch (#306)."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        now = time.time()
        result = resolve_targets_at_time(
            ts=now,
            schedule_blocks=None,
            override_until=now + 3600,
            override_heat=25.0,
            override_cool=25.0,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=21.0,
            comfort_cool=24.0,
            eco_heat=17.0,
            eco_cool=27.0,
            presence_away=True,
            presence_clears_override=True,
        )
        assert result == TargetTemps(heat=17.0, cool=27.0)

    def test_presence_clears_override_disabled_keeps_override(self):
        """presence_clears_override=False keeps override active even when away."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        now = time.time()
        result = resolve_targets_at_time(
            ts=now,
            schedule_blocks=None,
            override_until=now + 3600,
            override_heat=25.0,
            override_cool=25.0,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=21.0,
            comfort_cool=24.0,
            eco_heat=17.0,
            eco_cool=27.0,
            presence_away=True,
            presence_clears_override=False,
        )
        assert result == TargetTemps(heat=25.0, cool=25.0)

    def test_schedule_block_with_temperature_creates_single_point(self):
        """Schedule block with custom temperature creates TargetTemps(heat=t, cool=t)."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        dt = datetime(2025, 1, 6, 10, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE)  # Monday 10:00
        ts = dt.timestamp()
        schedule_blocks = {
            "monday": [
                {"from": "08:00:00", "to": "12:00:00", "data": {"temperature": 22.5}},
            ],
        }
        result = resolve_targets_at_time(
            ts=ts,
            schedule_blocks=schedule_blocks,
            override_until=None,
            override_heat=None,
            override_cool=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=21.0,
            comfort_cool=24.0,
            eco_heat=17.0,
            eco_cool=27.0,
        )
        assert result == TargetTemps(heat=22.5, cool=22.5)

    def test_override_returns_split_dead_band(self):
        """Active override with both targets returns split heat/cool dead-band."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        result = resolve_targets_at_time(
            ts=1000.0,
            schedule_blocks=None,
            override_until=None,
            override_heat=21.0,
            override_cool=24.0,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=20.0,
            comfort_cool=25.0,
            eco_heat=17.0,
            eco_cool=27.0,
        )
        assert result.heat == 21.0
        assert result.cool == 24.0

    def test_override_heat_only_leaves_cool_none(self):
        """Heat-only override keeps cool at None (direction off)."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        result = resolve_targets_at_time(
            ts=1000.0,
            schedule_blocks=None,
            override_until=None,
            override_heat=21.0,
            override_cool=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=20.0,
            comfort_cool=25.0,
            eco_heat=17.0,
            eco_cool=27.0,
        )
        assert result.heat == 21.0
        assert result.cool is None

    def test_override_expired_falls_through_to_comfort(self):
        """Expired override falls through to comfort targets."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        result = resolve_targets_at_time(
            ts=1000.0,
            schedule_blocks=None,
            override_until=500.0,
            override_heat=21.0,
            override_cool=24.0,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=20.0,
            comfort_cool=25.0,
            eco_heat=17.0,
            eco_cool=27.0,
        )
        assert result.heat == 20.0
        assert result.cool == 25.0


class TestMakeTargetResolverCoolValues:
    """Tests that make_target_resolver returns correct .cool values."""

    def test_resolver_returns_correct_cool(self):
        """Resolver returns correct .cool value from room config."""
        room = {
            "comfort_temp": 21.0,
            "eco_temp": 17.0,
            "comfort_heat": 21.0,
            "comfort_cool": 24.0,
            "eco_heat": 17.0,
            "eco_cool": 27.0,
        }
        settings: dict = {}
        resolver = make_target_resolver(None, room, settings)
        result = resolver(time.time())
        assert result.heat == 21.0
        assert result.cool == 24.0

    def test_mold_delta_applies_only_to_heat(self):
        """Mold prevention delta only raises .heat, not .cool."""
        room = {
            "comfort_temp": 21.0,
            "eco_temp": 17.0,
            "comfort_heat": 21.0,
            "comfort_cool": 24.0,
            "eco_heat": 17.0,
            "eco_cool": 27.0,
        }
        settings: dict = {}
        resolver = make_target_resolver(None, room, settings, mold_prevention_delta=2.0)
        result = resolver(time.time())
        assert result.heat == 23.0  # 21 + 2
        assert result.cool == 24.0  # unchanged


# ---------------------------------------------------------------------------
# Split block temperature support (resolve_targets_at_time)
# ---------------------------------------------------------------------------


class TestResolveTargetsAtTimeSplitBlockTemps:
    """Tests for schedule blocks with heat_temperature / cool_temperature."""

    def test_resolve_targets_with_split_block_temps(self):
        """Block with heat_temperature and cool_temperature returns split TargetTemps."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        dt = datetime(2025, 1, 6, 10, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE)  # Monday 10:00
        ts = dt.timestamp()
        schedule_blocks = {
            "monday": [
                {
                    "from": "08:00:00",
                    "to": "12:00:00",
                    "data": {"heat_temperature": 21, "cool_temperature": 24},
                },
            ],
        }
        result = resolve_targets_at_time(
            ts=ts,
            schedule_blocks=schedule_blocks,
            override_until=None,
            override_heat=None,
            override_cool=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=20.0,
            comfort_cool=26.0,
            eco_heat=17.0,
            eco_cool=27.0,
        )
        assert result == TargetTemps(heat=21.0, cool=24.0)

    def test_resolve_targets_with_only_heat_block_temp(self):
        """Block with only heat_temperature falls back to comfort_cool."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        dt = datetime(2025, 1, 6, 10, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE)  # Monday 10:00
        ts = dt.timestamp()
        schedule_blocks = {
            "monday": [
                {
                    "from": "08:00:00",
                    "to": "12:00:00",
                    "data": {"heat_temperature": 21},
                },
            ],
        }
        result = resolve_targets_at_time(
            ts=ts,
            schedule_blocks=schedule_blocks,
            override_until=None,
            override_heat=None,
            override_cool=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=20.0,
            comfort_cool=26.0,
            eco_heat=17.0,
            eco_cool=27.0,
        )
        assert result == TargetTemps(heat=21.0, cool=26.0)

    def test_resolve_targets_with_only_cool_block_temp(self):
        """Block with only cool_temperature falls back to comfort_heat."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        dt = datetime(2025, 1, 6, 10, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE)  # Monday 10:00
        ts = dt.timestamp()
        schedule_blocks = {
            "monday": [
                {
                    "from": "08:00:00",
                    "to": "12:00:00",
                    "data": {"cool_temperature": 24},
                },
            ],
        }
        result = resolve_targets_at_time(
            ts=ts,
            schedule_blocks=schedule_blocks,
            override_until=None,
            override_heat=None,
            override_cool=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=20.0,
            comfort_cool=26.0,
            eco_heat=17.0,
            eco_cool=27.0,
        )
        assert result == TargetTemps(heat=20.0, cool=24.0)

    def test_resolve_targets_single_temp_still_works(self):
        """Block with only temperature creates single-point (backward compat)."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        dt = datetime(2025, 1, 6, 10, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE)  # Monday 10:00
        ts = dt.timestamp()
        schedule_blocks = {
            "monday": [
                {
                    "from": "08:00:00",
                    "to": "12:00:00",
                    "data": {"temperature": 22},
                },
            ],
        }
        result = resolve_targets_at_time(
            ts=ts,
            schedule_blocks=schedule_blocks,
            override_until=None,
            override_heat=None,
            override_cool=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=20.0,
            comfort_cool=26.0,
            eco_heat=17.0,
            eco_cool=27.0,
        )
        assert result == TargetTemps(heat=22.0, cool=22.0)

    def test_vacation_cool_target_uses_eco_cool(self):
        """Vacation should keep cool at eco_cool, not collapse to vacation_temp."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        ts = time.time()
        result = resolve_targets_at_time(
            ts=ts,
            schedule_blocks=None,
            override_until=None,
            override_heat=None,
            override_cool=None,
            vacation_until=ts + 3600,
            vacation_temp=17.0,
            comfort_heat=21.0,
            comfort_cool=24.0,
            eco_heat=17.0,
            eco_cool=27.0,
        )
        assert result.heat == 17.0
        assert result.cool == 27.0  # eco_cool, not 17

    def test_vacation_cool_target_at_least_vacation_temp(self):
        """If vacation_temp > eco_cool, cool should be vacation_temp (max)."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        ts = time.time()
        result = resolve_targets_at_time(
            ts=ts,
            schedule_blocks=None,
            override_until=None,
            override_heat=None,
            override_cool=None,
            vacation_until=ts + 3600,
            vacation_temp=30.0,
            comfort_heat=21.0,
            comfort_cool=24.0,
            eco_heat=17.0,
            eco_cool=27.0,
        )
        assert result.heat == 30.0
        assert result.cool == 30.0  # max(30, 27) = 30


# ---------------------------------------------------------------------------
# read_schedule_blocks — current behavior (regression safety net)
# ---------------------------------------------------------------------------


def _make_async_hass(service_result=None, service_raises: Exception | None = None):
    """Create a hass mock with a configurable schedule.get_schedule response."""
    hass = MagicMock()
    if service_raises is not None:
        hass.services.async_call = AsyncMock(side_effect=service_raises)
    else:
        hass.services.async_call = AsyncMock(return_value=service_result)
    return hass


_SAMPLE_BLOCKS = {
    "monday": [{"from": "00:00:00", "to": "23:59:59", "data": {"temperature": 19.5}}],
    "tuesday": [{"from": "00:00:00", "to": "23:59:59", "data": {"temperature": 19.5}}],
}


class TestReadScheduleBlocksCurrent:
    """Document current behavior so refactors do not regress it."""

    @pytest.mark.asyncio
    async def test_empty_entity_id_returns_none(self):
        hass = _make_async_hass()
        result = await read_schedule_blocks(hass, "")
        assert result is None
        hass.services.async_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_schedule_entity_returns_none(self):
        hass = _make_async_hass()
        result = await read_schedule_blocks(hass, "binary_sensor.something")
        assert result is None
        hass.services.async_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_call_returns_blocks(self):
        hass = _make_async_hass(service_result={"schedule.heating": _SAMPLE_BLOCKS})
        result = await read_schedule_blocks(hass, "schedule.heating")
        assert result == _SAMPLE_BLOCKS

    @pytest.mark.asyncio
    async def test_service_raises_returns_none(self):
        hass = _make_async_hass(service_raises=RuntimeError("boom"))
        result = await read_schedule_blocks(hass, "schedule.heating")
        assert result is None

    @pytest.mark.asyncio
    async def test_falsy_response_returns_none(self):
        hass = _make_async_hass(service_result=None)
        result = await read_schedule_blocks(hass, "schedule.heating")
        assert result is None


# ---------------------------------------------------------------------------
# read_schedule_blocks — caching behavior (#308)
# ---------------------------------------------------------------------------


class TestReadScheduleBlocksCache:
    """Cache successful reads to recover from transient service failures."""

    @pytest.mark.asyncio
    async def test_success_writes_into_cache(self):
        hass = _make_async_hass(service_result={"schedule.heating": _SAMPLE_BLOCKS})
        cache: dict = {}
        result = await read_schedule_blocks(hass, "schedule.heating", cache=cache)
        assert result == _SAMPLE_BLOCKS
        assert cache["schedule.heating"] == _SAMPLE_BLOCKS

    @pytest.mark.asyncio
    async def test_exception_returns_cached_blocks(self):
        cache: dict = {"schedule.heating": _SAMPLE_BLOCKS}
        hass = _make_async_hass(service_raises=RuntimeError("boom"))
        result = await read_schedule_blocks(hass, "schedule.heating", cache=cache)
        assert result == _SAMPLE_BLOCKS

    @pytest.mark.asyncio
    async def test_empty_response_returns_cached_blocks(self):
        cache: dict = {"schedule.heating": _SAMPLE_BLOCKS}
        hass = _make_async_hass(service_result=None)
        result = await read_schedule_blocks(hass, "schedule.heating", cache=cache)
        assert result == _SAMPLE_BLOCKS

    @pytest.mark.asyncio
    async def test_response_missing_entity_returns_cached_blocks(self):
        """If get_schedule returns data for the wrong entity, treat as failure."""
        cache: dict = {"schedule.heating": _SAMPLE_BLOCKS}
        hass = _make_async_hass(service_result={"schedule.other": _SAMPLE_BLOCKS})
        result = await read_schedule_blocks(hass, "schedule.heating", cache=cache)
        assert result == _SAMPLE_BLOCKS

    @pytest.mark.asyncio
    async def test_failure_without_cache_entry_returns_none(self):
        cache: dict = {}
        hass = _make_async_hass(service_raises=RuntimeError("boom"))
        result = await read_schedule_blocks(hass, "schedule.heating", cache=cache)
        assert result is None

    @pytest.mark.asyncio
    async def test_failure_cache_none_returns_none(self):
        hass = _make_async_hass(service_raises=RuntimeError("boom"))
        result = await read_schedule_blocks(hass, "schedule.heating", cache=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_success_overwrites_stale_cache(self):
        cache: dict = {"schedule.heating": {"monday": []}}
        hass = _make_async_hass(service_result={"schedule.heating": _SAMPLE_BLOCKS})
        result = await read_schedule_blocks(hass, "schedule.heating", cache=cache)
        assert result == _SAMPLE_BLOCKS
        assert cache["schedule.heating"] == _SAMPLE_BLOCKS

    @pytest.mark.asyncio
    async def test_cache_isolated_per_entity(self):
        cache: dict = {"schedule.upstairs": _SAMPLE_BLOCKS}
        hass = _make_async_hass(service_raises=RuntimeError("boom"))
        result = await read_schedule_blocks(hass, "schedule.downstairs", cache=cache)
        assert result is None
        assert "schedule.upstairs" in cache


# ---------------------------------------------------------------------------
# read_schedule_blocks — logging contract (#308)
# ---------------------------------------------------------------------------


class TestReadScheduleBlocksLogging:
    """Verify log levels: WARNING when cache is empty, DEBUG when cache hides failure."""

    @pytest.mark.asyncio
    async def test_warning_when_cache_empty(self, caplog):
        hass = _make_async_hass(service_raises=RuntimeError("boom"))
        cache: dict = {}
        with caplog.at_level(logging.DEBUG, logger="custom_components.roommind.utils.schedule_utils"):
            await read_schedule_blocks(hass, "schedule.heating", cache=cache)
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("schedule.heating" in r.getMessage() for r in warning_records)

    @pytest.mark.asyncio
    async def test_debug_only_when_cache_covers_failure(self, caplog):
        cache: dict = {"schedule.heating": _SAMPLE_BLOCKS}
        hass = _make_async_hass(service_raises=RuntimeError("boom"))
        with caplog.at_level(logging.DEBUG, logger="custom_components.roommind.utils.schedule_utils"):
            await read_schedule_blocks(hass, "schedule.heating", cache=cache)
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert warning_records == []
        assert any("cached blocks" in r.getMessage() for r in debug_records)

    @pytest.mark.asyncio
    async def test_success_logs_nothing(self, caplog):
        hass = _make_async_hass(service_result={"schedule.heating": _SAMPLE_BLOCKS})
        cache: dict = {}
        with caplog.at_level(logging.DEBUG, logger="custom_components.roommind.utils.schedule_utils"):
            await read_schedule_blocks(hass, "schedule.heating", cache=cache)
        relevant = [r for r in caplog.records if r.name == "custom_components.roommind.utils.schedule_utils"]
        assert relevant == []


# ---------------------------------------------------------------------------
# Timezone-aware block resolution
#
# time.time() is a UTC epoch. Schedule blocks are authored in the user's local
# (HA-configured) timezone. The resolver must therefore evaluate the block
# window in HA's timezone, NOT the host/container timezone.
#
# These tests set HA's default timezone to Australia/Sydney (UTC+10) and pick
# UTC instants whose Sydney-local day/time lands on the OPPOSITE
# side of the block boundary from UTC — so they fail on the naive code (in a
# UTC/non-Sydney runner) and pass once the resolver honours HA's timezone.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _ha_time_zone(name: str):
    """Temporarily set HA's default timezone (restored afterwards)."""
    original = dt_util.DEFAULT_TIME_ZONE
    dt_util.set_default_time_zone(ZoneInfo(name))
    try:
        yield
    finally:
        dt_util.set_default_time_zone(original)


def _utc_ts(year, month, day, hour, minute=0):
    """Epoch seconds for an explicit UTC wall-clock instant."""
    return datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp()


class TestTimezoneAwareBlockResolution:
    """Schedule windows are evaluated in HA's timezone, not the host's."""

    # 2024-07-06 is a Saturday; +10h (Sydney AEST) → Sunday. July has no DST.
    _SUNDAY_BLOCK = {
        "sunday": [{"from": "08:00:00", "to": "21:00:00", "data": {"temperature": 22.0}}],
        # deliberately no "saturday" entry — the UTC-naive reading would land here
    }

    def test_find_active_block_uses_ha_timezone_not_host(self):
        """Sat 23:30 UTC is Sun 09:30 in Sydney → matches the Sunday block."""
        ts = _utc_ts(2024, 7, 6, 23, 30)  # Sydney: Sun 2024-07-07 09:30
        with _ha_time_zone("Australia/Sydney"):
            block = find_active_block(self._SUNDAY_BLOCK, ts)
        assert block is not None
        assert block["temperature"] == 22.0

    def test_find_active_block_ha_local_outside_window_returns_none(self):
        """Sun 12:00 UTC is Sun 22:00 in Sydney → past the 21:00 block end → None."""
        ts = _utc_ts(2024, 7, 7, 12, 0)  # Sydney: Sun 2024-07-07 22:00
        with _ha_time_zone("Australia/Sydney"):
            block = find_active_block(self._SUNDAY_BLOCK, ts)
        assert block is None

    def test_overnight_resolves_to_eco_in_ha_timezone(self):
        """Sydney overnight (outside the daytime block) must resolve to eco, not comfort."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        blocks = {
            day: [{"from": "07:00:00", "to": "21:30:00", "data": {}}]
            for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
        }
        # 2024-07-08 16:00 UTC → Sydney Tue 2024-07-09 02:00 (overnight, block is 07:00–21:30)
        ts = _utc_ts(2024, 7, 8, 16, 0)
        with _ha_time_zone("Australia/Sydney"):
            result = resolve_targets_at_time(
                ts=ts,
                schedule_blocks=blocks,
                override_until=None,
                override_heat=None,
                override_cool=None,
                vacation_until=None,
                vacation_temp=None,
                comfort_heat=24.0,
                comfort_cool=26.0,
                eco_heat=19.0,
                eco_cool=29.0,
            )
        assert result == TargetTemps(heat=19.0, cool=29.0)  # eco

    def test_daytime_resolves_to_comfort_in_ha_timezone(self):
        """Sydney daytime (inside the block) must resolve to comfort even when UTC is 'off'."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        blocks = {
            day: [{"from": "07:00:00", "to": "21:30:00", "data": {}}]
            for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
        }
        # 2024-07-08 00:00 UTC → Sydney Mon 2024-07-08 10:00 (inside 07:00–21:30);
        # naive-UTC reading (00:00) would be outside the block → eco.
        ts = _utc_ts(2024, 7, 8, 0, 0)
        with _ha_time_zone("Australia/Sydney"):
            result = resolve_targets_at_time(
                ts=ts,
                schedule_blocks=blocks,
                override_until=None,
                override_heat=None,
                override_cool=None,
                vacation_until=None,
                vacation_temp=None,
                comfort_heat=24.0,
                comfort_cool=26.0,
                eco_heat=19.0,
                eco_cool=29.0,
            )
        assert result == TargetTemps(heat=24.0, cool=26.0)  # comfort


# ---------------------------------------------------------------------------
# sanitize_block_temp / out-of-range block temps (#395)
# ---------------------------------------------------------------------------


def _monday_blocks(data: dict) -> dict:
    """Return schedule blocks with a single Monday 08:00-12:00 block."""
    return {"monday": [{"from": "08:00:00", "to": "12:00:00", "data": data}]}


def _monday_ts() -> float:
    """Monday 2025-01-06 10:00 local time, inside the _monday_blocks() window.

    Must be evaluated inside the test, not at import time: the autouse fixture in
    conftest sets HA's timezone per test, so a module-level constant would be
    built against the wrong zone and fall outside the block (CI TZ sweep).
    """
    return datetime(2025, 1, 6, 10, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE).timestamp()


class TestSanitizeBlockTemp:
    """Tests for sanitize_block_temp (#395)."""

    def test_valid_value_passes_through(self):
        """A plausible value is returned unchanged."""
        from custom_components.roommind.utils.schedule_utils import sanitize_block_temp

        assert sanitize_block_temp(21.0) == 21.0

    @pytest.mark.parametrize("bound", [5.0, 35.0])
    def test_bounds_are_inclusive(self, bound):
        """Both range limits are accepted."""
        from custom_components.roommind.utils.schedule_utils import sanitize_block_temp

        assert sanitize_block_temp(bound) == bound

    @pytest.mark.parametrize("raw", [110, 35.1, 4.9, -5, 0])
    def test_out_of_range_returns_none(self, raw):
        """Implausible values are rejected."""
        from custom_components.roommind.utils.schedule_utils import sanitize_block_temp

        assert sanitize_block_temp(raw) is None

    @pytest.mark.parametrize("raw", ["not_a_number", None, [21.0], {}])
    def test_non_numeric_returns_none(self, raw):
        """Unparseable values are rejected (previous ValueError/TypeError path)."""
        from custom_components.roommind.utils.schedule_utils import sanitize_block_temp

        assert sanitize_block_temp(raw) is None

    def test_nan_returns_none(self):
        """NaN is rejected — it would pass a naive `MIN <= v <= MAX` check."""
        from custom_components.roommind.utils.schedule_utils import sanitize_block_temp

        assert sanitize_block_temp(float("nan")) is None

    def test_infinity_returns_none(self):
        """Infinity is rejected."""
        from custom_components.roommind.utils.schedule_utils import sanitize_block_temp

        assert sanitize_block_temp(float("inf")) is None

    def test_numeric_string_is_accepted(self):
        """Numeric strings still resolve (YAML may deliver them quoted)."""
        from custom_components.roommind.utils.schedule_utils import sanitize_block_temp

        assert sanitize_block_temp("21.5") == 21.5

    def test_range_check_runs_after_converter(self):
        """71.6°F converts to 22°C and is accepted, not rejected as out of range."""
        from custom_components.roommind.utils.schedule_utils import sanitize_block_temp

        converter = lambda v: (v - 32) * 5 / 9  # noqa: E731
        assert sanitize_block_temp(71.6, converter) == pytest.approx(22.0, abs=0.1)

    def test_converted_value_out_of_range_rejected(self):
        """220°F = 104°C is rejected after conversion."""
        from custom_components.roommind.utils.schedule_utils import sanitize_block_temp

        converter = lambda v: (v - 32) * 5 / 9  # noqa: E731
        assert sanitize_block_temp(220.0, converter) is None


class TestOutOfRangeBlockTemps:
    """Out-of-range schedule block temps fall back to comfort (#395)."""

    def test_resolve_target_at_time_falls_back_to_comfort(self):
        """The #395 typo (110 instead of 11) resolves to comfort_temp."""
        result = resolve_target_at_time(
            ts=_monday_ts(),
            schedule_blocks=_monday_blocks({"temperature": 110}),
            override_until=None,
            override_temp=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_temp=21.0,
            eco_temp=18.0,
        )
        assert result == 21.0

    def test_resolve_target_at_time_accepts_intended_value(self):
        """The value the user meant to type (11) is still honoured."""
        result = resolve_target_at_time(
            ts=_monday_ts(),
            schedule_blocks=_monday_blocks({"temperature": 11}),
            override_until=None,
            override_temp=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_temp=21.0,
            eco_temp=18.0,
        )
        assert result == 11.0

    def test_resolve_targets_single_temp_falls_back(self):
        """Out-of-range single `temperature` falls back to both comfort targets."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        result = resolve_targets_at_time(
            ts=_monday_ts(),
            schedule_blocks=_monday_blocks({"temperature": 110}),
            override_until=None,
            override_heat=None,
            override_cool=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=21.0,
            comfort_cool=24.0,
            eco_heat=17.0,
            eco_cool=27.0,
        )
        assert result == TargetTemps(heat=21.0, cool=24.0)

    def test_resolve_targets_bad_heat_keeps_good_cool(self):
        """Only the out-of-range field falls back; the valid one is kept."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        result = resolve_targets_at_time(
            ts=_monday_ts(),
            schedule_blocks=_monday_blocks({"heat_temperature": 200, "cool_temperature": 25.0}),
            override_until=None,
            override_heat=None,
            override_cool=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=21.0,
            comfort_cool=24.0,
            eco_heat=17.0,
            eco_cool=27.0,
        )
        assert result == TargetTemps(heat=21.0, cool=25.0)

    def test_resolve_targets_bad_cool_keeps_good_heat(self):
        """Mirror case: out-of-range cool_temperature falls back to comfort_cool."""
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        result = resolve_targets_at_time(
            ts=_monday_ts(),
            schedule_blocks=_monday_blocks({"heat_temperature": 20.0, "cool_temperature": -40}),
            override_until=None,
            override_heat=None,
            override_cool=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=21.0,
            comfort_cool=24.0,
            eco_heat=17.0,
            eco_cool=27.0,
        )
        assert result == TargetTemps(heat=20.0, cool=24.0)

    def test_make_target_resolver_falls_back(self):
        """The MPC/forecast resolver drops the typo too."""
        resolver = make_target_resolver(
            _monday_blocks({"temperature": 110}),
            {"comfort_heat": 21.0, "comfort_cool": 24.0, "eco_heat": 17.0, "eco_cool": 27.0},
            {},
        )
        assert resolver(_monday_ts()) == TargetTemps(heat=21.0, cool=24.0)


class TestFindRejectedBlockTemps:
    """Whole-week scan for unusable block temps (#395)."""

    def test_none_and_empty_return_empty(self):
        """No schedule blocks → nothing to report."""
        from custom_components.roommind.utils.schedule_utils import find_rejected_block_temps

        assert find_rejected_block_temps(None) == []
        assert find_rejected_block_temps({}) == []

    def test_all_valid_returns_empty(self):
        """A clean schedule produces no warnings."""
        from custom_components.roommind.utils.schedule_utils import find_rejected_block_temps

        blocks = _monday_blocks({"temperature": 21.0})
        assert find_rejected_block_temps(blocks) == []

    def test_finds_typo_in_inactive_block(self):
        """The #395 scenario: a bad value in the 00:00-06:00 block is found at any time."""
        from custom_components.roommind.utils.schedule_utils import find_rejected_block_temps

        blocks = {"monday": [{"from": "00:00:00", "to": "06:00:00", "data": {"temperature": 110}}]}
        assert find_rejected_block_temps(blocks) == [
            {"day": "monday", "from": "00:00:00", "field": "temperature", "value": "110"}
        ]

    def test_reports_split_fields(self):
        """heat_temperature and cool_temperature are scanned too."""
        from custom_components.roommind.utils.schedule_utils import find_rejected_block_temps

        blocks = _monday_blocks({"heat_temperature": 200, "cool_temperature": 24.0})
        result = find_rejected_block_temps(blocks)
        assert [r["field"] for r in result] == ["heat_temperature"]
        assert result[0]["value"] == "200"

    def test_reports_non_numeric_values(self):
        """Unparseable values are reported as-is, not swallowed."""
        from custom_components.roommind.utils.schedule_utils import find_rejected_block_temps

        blocks = _monday_blocks({"temperature": "twenty"})
        assert find_rejected_block_temps(blocks)[0]["value"] == "twenty"

    def test_results_are_in_weekday_order(self):
        """Entries are ordered Monday-first regardless of dict insertion order."""
        from custom_components.roommind.utils.schedule_utils import find_rejected_block_temps

        blocks = {
            "wednesday": [{"from": "01:00:00", "to": "02:00:00", "data": {"temperature": 111}}],
            "monday": [{"from": "01:00:00", "to": "02:00:00", "data": {"temperature": 110}}],
        }
        assert [r["day"] for r in find_rejected_block_temps(blocks)] == ["monday", "wednesday"]

    def test_converter_applied_before_check(self):
        """71.6°F is valid after conversion and must not be reported."""
        from custom_components.roommind.utils.schedule_utils import find_rejected_block_temps

        converter = lambda v: (v - 32) * 5 / 9  # noqa: E731
        assert find_rejected_block_temps(_monday_blocks({"temperature": 71.6}), converter) == []
        assert len(find_rejected_block_temps(_monday_blocks({"temperature": 220}), converter)) == 1

    def test_missing_data_key_is_not_reported(self):
        """A block without temperature fields is fine — it falls back to comfort."""
        from custom_components.roommind.utils.schedule_utils import find_rejected_block_temps

        assert find_rejected_block_temps({"monday": [{"from": "01:00:00", "to": "02:00:00"}]}) == []

    def test_day_with_none_block_list(self):
        """A day key holding None is tolerated."""
        from custom_components.roommind.utils.schedule_utils import find_rejected_block_temps

        assert find_rejected_block_temps({"monday": None}) == []


class TestOverrideBands:
    """Override presets are bands, not single points (#284)."""

    ROOM = {
        "comfort_heat": 21.5,
        "comfort_cool": 24.0,
        "eco_heat": 17.0,
        "eco_cool": 27.5,
    }

    def test_eco_preset_keeps_both_sides(self):
        assert override_preset_band(self.ROOM, "eco") == (17.0, 27.5)

    def test_boost_preset_is_comfort_band(self):
        assert override_preset_band(self.ROOM, "boost") == (21.5, 24.0)

    def test_legacy_single_temps_fall_back(self):
        room = {"eco_temp": 16.0, "comfort_temp": 22.0}
        assert override_preset_band(room, "eco") == (16.0, 27.0)
        assert override_preset_band(room, "boost") == (22.0, 24.0)

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [("auto", (17.0, 27.5)), ("heat_only", (17.0, None)), ("cool_only", (None, 27.5))],
    )
    def test_mask_drops_unusable_side(self, mode, expected):
        assert mask_override_band(17.0, 27.5, mode) == expected

    def test_resolver_keeps_eco_override_band_over_horizon(self):
        """The MPC forecast sees the same band the live path uses."""
        room = {**self.ROOM, "override_heat": 17.0, "override_cool": 27.5, "override_until": None}
        resolver = make_target_resolver(None, room, {})
        for offset in (0, 1800, 7200):
            t = resolver(time.time() + offset)
            assert (t.heat, t.cool) == (17.0, 27.5)


# ---------------------------------------------------------------------------
# Occupancy eco: an empty room caps the schedule at eco (#180)
# ---------------------------------------------------------------------------


def _monday(hour: int) -> float:
    # Evaluated per test: the TZ sweep switches the default zone after import.
    return datetime(2025, 1, 6, hour, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE).timestamp()


class TestRoomVacantCapsAtEco:
    def _resolve(self, ts, blocks, **kwargs):
        from custom_components.roommind.utils.schedule_utils import resolve_targets_at_time

        args = {
            "override_until": None,
            "override_heat": None,
            "override_cool": None,
            "vacation_until": None,
            "vacation_temp": None,
            "comfort_heat": 21.0,
            "comfort_cool": 24.0,
            "eco_heat": 17.0,
            "eco_cool": 27.0,
        }
        args.update(kwargs)
        return resolve_targets_at_time(ts, blocks, **args)

    def test_comfort_block_falls_to_eco(self):
        blocks = {"monday": [{"from": "08:00:00", "to": "12:00:00", "data": {}}]}
        assert self._resolve(_monday(10), blocks, room_vacant=True) == TargetTemps(17.0, 27.0)

    def test_without_schedule_comfort_falls_to_eco(self):
        assert self._resolve(_monday(10), None, room_vacant=True) == TargetTemps(17.0, 27.0)

    def test_block_below_eco_is_not_raised(self):
        blocks = {"monday": [{"from": "08:00:00", "to": "12:00:00", "data": {"temperature": 16.0}}]}
        assert self._resolve(_monday(10), blocks, room_vacant=True) == TargetTemps(16.0, 27.0)

    def test_schedule_off_stays_off(self):
        blocks = {"monday": [{"from": "08:00:00", "to": "12:00:00", "data": {}}]}
        result = self._resolve(_monday(6), blocks, room_vacant=True, schedule_off_action="off")
        assert result == TargetTemps(None, None)

    def test_override_wins(self):
        result = self._resolve(
            _monday(10),
            None,
            room_vacant=True,
            override_heat=23.0,
            override_cool=25.0,
            override_until=_monday(10) + 3600,
        )
        assert result == TargetTemps(23.0, 25.0)

    def test_occupied_room_follows_schedule(self):
        assert self._resolve(_monday(10), None) == TargetTemps(21.0, 24.0)


class TestCapToEco:
    def test_caps_heat_down_and_cool_up(self):
        from custom_components.roommind.utils.schedule_utils import cap_to_eco

        assert cap_to_eco(TargetTemps(21.0, 24.0), 17.0, 27.0) == TargetTemps(17.0, 27.0)

    def test_keeps_lower_demand_and_missing_sides(self):
        from custom_components.roommind.utils.schedule_utils import cap_to_eco

        assert cap_to_eco(TargetTemps(15.0, 28.0), 17.0, 27.0) == TargetTemps(15.0, 28.0)
        assert cap_to_eco(TargetTemps(None, 24.0), 17.0, 27.0) == TargetTemps(None, 27.0)
        assert cap_to_eco(TargetTemps(21.0, None), 17.0, 27.0) == TargetTemps(17.0, None)


def test_make_target_resolver_caps_an_empty_room_over_the_horizon():
    """The MPC horizon must not pre-heat an empty room towards a later comfort block (#180)."""
    from custom_components.roommind.utils.schedule_utils import make_target_resolver

    room = {"comfort_heat": 21.0, "comfort_cool": 24.0, "eco_heat": 17.0, "eco_cool": 27.0}
    resolver = make_target_resolver(None, room, {}, room_vacant=True)
    assert resolver(time.time() + 3600) == TargetTemps(17.0, 27.0)
    assert make_target_resolver(None, room, {})(time.time()) == TargetTemps(21.0, 24.0)


# ---------------------------------------------------------------------------
# band_in_force / set_band_sides (#259)
# ---------------------------------------------------------------------------


def test_band_in_force_prefers_the_active_override():
    room = {"override_heat": 23.0, "override_cool": 25.0, "override_until": None}
    assert band_in_force(room, {"heat_target": 21.0, "cool_target": 24.0}) == (23.0, 25.0)


def test_band_in_force_takes_live_targets_without_mold_delta():
    live = {"heat_target": 22.0, "cool_target": 24.0, "mold_prevention_active": True, "mold_prevention_delta": 1.0}
    assert band_in_force({}, live) == (21.0, 24.0)


def test_band_in_force_falls_back_to_comfort():
    assert band_in_force({"comfort_heat": 20.5, "comfort_cool": 25.0}, {}) == (20.5, 25.0)


def test_set_band_sides_sets_heat_and_pushes_cool_along():
    assert set_band_sides((21.0, 24.0), 23.0, None, "auto") == (23.0, 24.0)
    assert set_band_sides((21.0, 24.0), 25.0, None, "auto") == (25.0, 25.0)


def test_set_band_sides_sets_cool_and_pushes_heat_along():
    assert set_band_sides((21.0, 24.0), None, 20.0, "auto") == (20.0, 20.0)


def test_set_band_sides_sets_both_and_stays_within_bounds():
    assert set_band_sides((21.0, 24.0), 22.0, 25.04, "auto") == (22.0, 25.0)
    assert set_band_sides((21.0, None), 2.0, None, "heat_only") == (5.0, None)
    assert set_band_sides((21.0, 24.0), 40.0, None, "auto") == (35.0, 35.0)


def test_set_band_sides_ignores_the_unused_side():
    assert set_band_sides((21.0, 24.0), 22.0, 26.0, "cool_only") == (None, 26.0)
    assert set_band_sides((21.0, 24.0), 21.5, None, "heat_only") == (21.5, None)
    assert set_band_sides((None, 24.0), 22.0, None, "cool_only") is None
    assert set_band_sides((21.0, None), None, 22.0, "heat_only") is None


# ---------------------------------------------------------------------------
# override_hold_until
# ---------------------------------------------------------------------------

_HOLD_ROOM = {"schedules": [{"entity_id": "schedule.living"}]}


def _hold_hass(state: str | None, next_event=None):
    hass = MagicMock()
    if state is None:
        hass.states.get.return_value = None
    else:
        hass.states.get.return_value = MagicMock(state=state, attributes={"next_event": next_event})
    return hass


def test_hold_ends_at_the_next_schedule_change():
    nxt = datetime.fromtimestamp(time.time() + 1800, UTC)
    assert override_hold_until(_hold_hass("on", nxt), _HOLD_ROOM) == nxt.timestamp()


def test_hold_reads_an_iso_next_event():
    nxt = datetime.fromtimestamp(time.time() + 1800, UTC)
    assert override_hold_until(_hold_hass("on", nxt.isoformat()), _HOLD_ROOM) == nxt.timestamp()


def test_hold_without_schedule_has_no_end():
    assert override_hold_until(_hold_hass(None), {"schedules": []}) is None


def test_hold_under_a_schedule_without_blocks_has_no_end():
    """An empty helper never changes, so there is nothing to wait for."""
    assert override_hold_until(_hold_hass("off", None), _HOLD_ROOM) is None


@pytest.mark.parametrize(
    "hass",
    [
        _hold_hass(None),
        _hold_hass("unavailable"),
        _hold_hass("unknown"),
        _hold_hass("on", datetime.fromtimestamp(time.time() - 60, UTC)),
    ],
    ids=["missing", "unavailable", "unknown", "stale"],
)
def test_hold_under_an_unreadable_schedule_ends_after_the_fallback(hass):
    """Without a readable next change the hold must not run forever."""
    now = time.time()
    until = override_hold_until(hass, _HOLD_ROOM, now=now)
    assert until == now + SCHEDULE_UNREADABLE_HOLD_HOURS * 3600


# ---------------------------------------------------------------------------
# Whole-home plant override (Central HVAC wall thermostat) + price shift modes
# ---------------------------------------------------------------------------


class TestIsPlantOverrideAt:
    """Whole-home plant override window checks."""

    def test_no_plant_override_returns_false(self):
        """No wall setpoint recorded on the Central Thermostat yet."""
        assert is_plant_override_at(1000.0, {}) is False
        assert is_plant_override_at(1000.0, {"plant_override_heat": None, "plant_override_cool": None}) is False

    def test_active_plant_override_returns_true(self):
        """A wall setpoint within its hold window is in force."""
        assert is_plant_override_at(1000.0, {"plant_override_heat": 22.0, "plant_override_cool": None}) is True

    def test_expired_plant_override_returns_false(self):
        """A wall setpoint past its expiry is no longer in force."""
        assert is_plant_override_at(1000.0, {"plant_override_heat": 22.0, "plant_override_until": 500.0}) is False

    def test_cool_only_plant_override_is_active(self):
        """A single-setpoint (cool) wall thermostat is a plant override too."""
        assert is_plant_override_at(1000.0, {"plant_override_heat": None, "plant_override_cool": 20.0}) is True

    def test_presence_away_with_clears_override_holds_it(self):
        """An away home with presence_clears_override set does not apply the override."""
        settings = {"plant_override_heat": 22.0, "presence_clears_override": True}
        assert is_plant_override_at(1000.0, settings, presence_away=True) is False
        assert (
            is_plant_override_at(
                1000.0, {"plant_override_heat": 22.0, "presence_clears_override": False}, presence_away=True
            )
            is True
        )


class TestResolveTargetsAtTimePlantOverride:
    """Whole-home plant overrides outrank the schedule but not a per-room override."""

    def test_plant_override_wins_over_the_schedule(self):
        """A wall setpoint replaces the schedule block until it expires."""
        ts = _monday_ts()
        result = resolve_targets_at_time(
            ts=ts,
            schedule_blocks=_monday_blocks({"temperature": 22.0}),
            override_until=None,
            override_heat=None,
            override_cool=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=21.0,
            comfort_cool=24.0,
            eco_heat=17.0,
            eco_cool=27.0,
            plant_override_until=ts + 7200,
            plant_override_heat=23.0,
            plant_override_cool=None,
        )
        # One-sided wall setpoint: heat follows the cool side, never crossing the band.
        assert result == TargetTemps(heat=23.0, cool=23.0)

    def test_plant_override_wins_over_vacation(self):
        """A wall setpoint outranks an active vacation."""
        ts = _monday_ts()
        result = resolve_targets_at_time(
            ts=ts,
            schedule_blocks=None,
            override_until=None,
            override_heat=None,
            override_cool=None,
            vacation_until=ts + 86400,
            vacation_temp=16.0,
            comfort_heat=21.0,
            comfort_cool=24.0,
            eco_heat=17.0,
            eco_cool=27.0,
            plant_override_until=ts + 7200,
            plant_override_heat=22.0,
            plant_override_cool=26.0,
        )
        assert result == TargetTemps(heat=22.0, cool=26.0)

    def test_plant_override_yields_to_the_per_room_override(self):
        """A per-room override is the tighter, more specific intent and wins."""
        ts = _monday_ts()
        result = resolve_targets_at_time(
            ts=ts,
            schedule_blocks=None,
            override_until=ts + 7200,
            override_heat=25.0,
            override_cool=25.0,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=21.0,
            comfort_cool=24.0,
            eco_heat=17.0,
            eco_cool=27.0,
            plant_override_until=None,
            plant_override_heat=22.0,
            plant_override_cool=22.0,
        )
        assert result == TargetTemps(heat=25.0, cool=25.0)

    def test_plant_override_respects_presence_clears_override(self):
        """An away home with presence_clears_override set holds the wall setpoint."""
        ts = _monday_ts()
        result = resolve_targets_at_time(
            ts=ts,
            schedule_blocks=None,
            override_until=None,
            override_heat=None,
            override_cool=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=21.0,
            comfort_cool=24.0,
            eco_heat=17.0,
            eco_cool=27.0,
            plant_override_until=ts + 7200,
            plant_override_heat=22.0,
            plant_override_cool=22.0,
            presence_away=True,
            presence_clears_override=True,
        )
        assert result == TargetTemps(heat=17.0, cool=27.0)

    def test_expired_plant_override_falls_through_to_the_schedule(self):
        """An expired wall setpoint hands the schedule back."""
        ts = _monday_ts()
        result = resolve_targets_at_time(
            ts=ts,
            schedule_blocks=_monday_blocks({"temperature": 22.0}),
            override_until=None,
            override_heat=None,
            override_cool=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=21.0,
            comfort_cool=24.0,
            eco_heat=17.0,
            eco_cool=27.0,
            plant_override_until=ts - 100,
            plant_override_heat=23.0,
            plant_override_cool=None,
        )
        assert result == TargetTemps(heat=22.0, cool=22.0)


class TestScheduleSetsTargetPlantOverride:
    """The price shift must never move a manual wall-thermostat override."""

    def test_plant_override_is_not_a_price_shift_target(self):
        """A whole-home plant override is the user's own choice, not a price opportunity."""
        room = {"override_heat": None, "override_cool": None}
        settings = {"plant_override_heat": 22.0, "plant_override_cool": None, "plant_override_until": None}
        assert schedule_sets_target(time.time(), room, settings, presence_away=False, room_vacant=False) is False

    def test_room_override_is_not_a_price_shift_target(self):
        room = {"override_heat": 23.0, "override_cool": None, "override_until": None}
        assert schedule_sets_target(time.time(), room, {}, presence_away=False, room_vacant=False) is False

    def test_no_override_says_the_schedule_sets_the_target(self):
        room = {"override_heat": None, "override_cool": None}
        assert schedule_sets_target(time.time(), room, {}, presence_away=False, room_vacant=False) is True


class TestPriceShiftModes:
    """Electricity price shifting: cool-only by default, "both" when opted in."""

    def _signal(self, now: float) -> PriceSignal:
        """A cheap signal: charging (pre-heating / pre-cooling) the schedule target."""
        return PriceSignal(current_level=LEVEL_CHEAP, now=now)

    def test_default_price_shift_is_cool_only(self):
        """Central heating is a fuel furnace: the default leaves the heat side alone."""
        now = time.time()
        shifted, deltas = price_shifted(
            TargetTemps(heat=21.0, cool=24.0),
            now,
            {"price_shift": 1.0},
            {},
            self._signal(now),
            presence_away=False,
            room_vacant=False,
        )
        assert shifted == TargetTemps(heat=21.0, cool=23.0)
        assert deltas == (0.0, -1.0)

    def test_both_mode_shifts_heat_and_cool(self):
        """An explicit Central Plant "both" opt-in shifts the heating target too."""
        now = time.time()
        shifted, deltas = price_shifted(
            TargetTemps(heat=21.0, cool=24.0),
            now,
            {"price_shift": 1.0},
            {"central_plant": {"price_shift_modes": "both"}},
            self._signal(now),
            presence_away=False,
            room_vacant=False,
        )
        assert shifted == TargetTemps(heat=22.0, cool=23.0)
        assert deltas == (1.0, -1.0)

    def test_global_price_shift_modes_both_shifts_both_sides(self):
        """The top-level price_shift_modes key is honoured the same way."""
        now = time.time()
        shifted, deltas = price_shifted(
            TargetTemps(heat=21.0, cool=24.0),
            now,
            {"price_shift": 1.0},
            {"price_shift_modes": "both"},
            self._signal(now),
            presence_away=False,
            room_vacant=False,
        )
        assert shifted == TargetTemps(heat=22.0, cool=23.0)
        assert deltas == (1.0, -1.0)

    def test_cool_only_keeps_the_heat_side_and_clamps(self):
        """Cheap pre-cooling never pushes the cool target under the kept heat target."""
        now = time.time()
        shifted, deltas = price_shifted(
            TargetTemps(heat=21.0, cool=18.0),
            now,
            {"price_shift": 2.0, "central_plant": {"price_shift_modes": "cool_only"}},
            {},
            self._signal(now),
            presence_away=False,
            room_vacant=False,
        )
        assert shifted == TargetTemps(heat=21.0, cool=21.0)

    def test_plant_override_suppresses_the_price_shift(self):
        """A wall setpoint is the user's own choice; the price never shifts it."""
        now = time.time()
        room = {"price_shift": 1.0}
        settings = {"plant_override_heat": 22.0, "plant_override_cool": None, "plant_override_until": None}
        shifted, deltas = price_shifted(
            TargetTemps(heat=21.0, cool=24.0),
            now,
            room,
            settings,
            self._signal(now),
            presence_away=False,
            room_vacant=False,
        )
        assert shifted == TargetTemps(heat=21.0, cool=24.0)
        assert deltas == (0.0, 0.0)


def test_make_target_resolver_applies_the_whole_home_plant_override():
    """The MPC resolver takes the whole-home wall setpoint from the Central Plant settings."""
    room = {"comfort_heat": 21.0, "comfort_cool": 24.0, "eco_heat": 17.0, "eco_cool": 27.0}
    settings = {"plant_override_heat": None, "plant_override_cool": 20.0, "plant_override_until": None}
    resolver = make_target_resolver(None, room, settings)
    assert resolver(time.time()) == TargetTemps(heat=20.0, cool=20.0)


def test_make_target_resolver_never_priceshifts_a_whole_home_plant_override():
    """A wall setpoint survives the price shift untouched."""
    room = {"comfort_heat": 21.0, "comfort_cool": 24.0, "eco_heat": 17.0, "eco_cool": 27.0, "price_shift": 1.0}
    settings = {"plant_override_heat": 22.0, "plant_override_cool": None, "plant_override_until": None}
    resolver = make_target_resolver(
        None,
        room,
        settings,
        price_signal=PriceSignal(current_level=LEVEL_CHEAP, now=time.time()),
    )
    assert resolver(time.time()) == TargetTemps(heat=22.0, cool=24.0)


# ---------------------------------------------------------------------------
# House layer (rungs 5-7 of the target ladder)
# ---------------------------------------------------------------------------


def _house(**overrides):
    """A resolved house settings dict with an active pair and a comfort schedule."""
    house = {
        "enabled": True,
        "heat_c": 21.0,
        "cool_c": 24.0,
        "eco_heat_c": 16.0,
        "eco_cool_c": 28.0,
        "price_shift": 0.0,
        "schedules": [{"entity_id": "schedule.house"}],
        "schedule_selector_entity": "",
    }
    house.update(overrides)
    return house


def _house_ctx(house=None, blocks=None, *, active=True):
    """Build the HouseContext the coordinator hands to the room resolver."""
    house = house or _house()
    return HouseContext(
        active=active,
        heat_c=house.get("heat_c"),
        cool_c=house.get("cool_c"),
        eco_heat_c=house.get("eco_heat_c"),
        eco_cool_c=house.get("eco_cool_c"),
        target_at=make_house_target_resolver(blocks, house),
        price_shift=float(house.get("price_shift") or 0.0),
        hold_active_at=lambda ts: house_override_active_at(ts, house),
    )


class TestResolveHouseTargetsAtTime:
    """The house resolves its own band: override > block > comfort pair, never eco/off."""

    def test_pair_when_no_schedule(self):
        assert resolve_house_targets_at_time(_monday_ts(), None, _house()) == TargetTemps(21.0, 24.0)

    def test_block_heat_and_cool_win(self):
        blocks = _monday_blocks({"heat_temperature": 20.0, "cool_temperature": 25.0})
        assert resolve_house_targets_at_time(_monday_ts(), blocks, _house()) == TargetTemps(20.0, 25.0)

    def test_single_temperature_block_sets_both_sides(self):
        blocks = _monday_blocks({"temperature": 22.0})
        assert resolve_house_targets_at_time(_monday_ts(), blocks, _house()) == TargetTemps(22.0, 22.0)

    def test_block_with_one_side_keeps_house_pair_for_other(self):
        blocks = _monday_blocks({"heat_temperature": 20.0})
        assert resolve_house_targets_at_time(_monday_ts(), blocks, _house()) == TargetTemps(20.0, 24.0)

    def test_house_override_beats_block(self):
        house = _house(override_heat=23.0, override_cool=26.0, override_until=_monday_ts() + 3600)
        blocks = _monday_blocks({"temperature": 22.0})
        assert resolve_house_targets_at_time(_monday_ts(), blocks, house) == TargetTemps(23.0, 26.0)

    def test_one_sided_override_takes_missing_side_from_block_without_crossing(self):
        house = _house(override_heat=23.0, override_cool=None, override_until=_monday_ts() + 3600)
        blocks = _monday_blocks({"temperature": 22.0})
        assert resolve_house_targets_at_time(_monday_ts(), blocks, house) == TargetTemps(23.0, 23.0)

    def test_expired_override_is_ignored(self):
        house = _house(override_heat=23.0, override_cool=26.0, override_until=_monday_ts() - 100)
        blocks = _monday_blocks({"temperature": 22.0})
        assert resolve_house_targets_at_time(_monday_ts(), blocks, house) == TargetTemps(22.0, 22.0)

    def test_block_without_temps_falls_back_to_pair(self):
        assert resolve_house_targets_at_time(_monday_ts(), _monday_blocks({}), _house()) == TargetTemps(21.0, 24.0)

    def test_out_of_range_side_falls_back_to_pair(self):
        blocks = _monday_blocks({"heat_temperature": 110, "cool_temperature": 25.0})
        assert resolve_house_targets_at_time(_monday_ts(), blocks, _house()) == TargetTemps(21.0, 25.0)

    def test_schedule_gap_returns_pair_not_eco(self):
        # Monday 06:00 is outside the 08:00-12:00 block, and a gap never means off.
        gap = datetime(2025, 1, 6, 6, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE).timestamp()
        assert resolve_house_targets_at_time(gap, _monday_blocks({}), _house()) == TargetTemps(21.0, 24.0)

    def test_target_source_override_schedule_target(self):
        ts = _monday_ts()
        assert house_target_source(ts, _monday_blocks({"temperature": 22.0}), _house()) == "schedule"
        assert house_target_source(ts, None, _house()) == "target"
        house = _house(override_heat=23.0, override_until=ts + 3600)
        assert house_target_source(ts, _monday_blocks({"temperature": 22.0}), house) == "override"

    def test_make_house_resolver_binds_the_converter(self):
        resolver = make_house_target_resolver(_monday_blocks({"temperature": 22.0}), _house())
        assert resolver(_monday_ts()) == TargetTemps(22.0, 22.0)


class TestTargetLadderWithHouse:
    """Room override/plant/vacation/schedule outrank the house; a gap falls to it."""

    def _resolve(self, ts, blocks, *, house=None, follow=True, **kwargs):
        args = {
            "override_until": None,
            "override_heat": None,
            "override_cool": None,
            "vacation_until": None,
            "vacation_temp": None,
            "comfort_heat": 21.0,
            "comfort_cool": 24.0,
            "eco_heat": 17.0,
            "eco_cool": 27.0,
        }
        args.update(kwargs)
        return resolve_targets_with_source(ts, blocks, house=house, follow_house=follow, **args)

    def test_room_block_beats_house(self):
        targets, source = self._resolve(_monday_ts(), _monday_blocks({"temperature": 22.0}), house=_house_ctx())
        assert (targets, source) == (TargetTemps(22.0, 22.0), TargetSource.ROOM_SCHEDULE)

    def test_room_schedule_gap_falls_to_house_for_following_room(self):
        gap = datetime(2025, 1, 6, 6, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE).timestamp()
        targets, source = self._resolve(gap, _monday_blocks({}), house=_house_ctx())
        assert (targets, source) == (TargetTemps(21.0, 24.0), TargetSource.HOUSE)

    def test_independent_room_gap_stays_eco(self):
        gap = datetime(2025, 1, 6, 6, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE).timestamp()
        targets, source = self._resolve(gap, _monday_blocks({}), house=_house_ctx(), follow=False)
        assert (targets, source) == (TargetTemps(17.0, 27.0), TargetSource.ROOM_BASE)

    def test_independent_room_gap_can_stay_off(self):
        gap = datetime(2025, 1, 6, 6, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE).timestamp()
        targets, source = self._resolve(
            gap, _monday_blocks({}), house=_house_ctx(), follow=False, schedule_off_action="off"
        )
        assert (targets, source) == (TargetTemps(None, None), TargetSource.ROOM_BASE)

    def test_inactive_house_reproduces_old_behaviour(self):
        ts = _monday_ts()
        targets, source = self._resolve(ts, None, house=_house_ctx(active=False))
        assert (targets, source) == (TargetTemps(21.0, 24.0), TargetSource.ROOM_BASE)

    def test_room_manual_override_beats_house(self):
        targets, source = self._resolve(
            _monday_ts(),
            None,
            house=_house_ctx(),
            override_heat=25.0,
            override_cool=25.0,
            override_until=_monday_ts() + 3600,
        )
        assert (targets, source) == (TargetTemps(25.0, 25.0), TargetSource.OVERRIDE)

    def test_vacation_beats_house(self):
        ts = _monday_ts()
        targets, source = self._resolve(ts, None, house=_house_ctx(), vacation_until=ts + 3600, vacation_temp=16.0)
        assert (targets, source) == (TargetTemps(16.0, 27.0), TargetSource.VACATION)

    def test_plant_override_beats_house(self):
        ts = _monday_ts()
        targets, source = self._resolve(
            ts,
            None,
            house=_house_ctx(),
            plant_override_until=ts + 3600,
            plant_override_heat=23.0,
            plant_override_cool=26.0,
        )
        assert (targets, source) == (TargetTemps(23.0, 26.0), TargetSource.PLANT_OVERRIDE)

    def test_vacant_following_room_caps_to_house_eco(self):
        gap = datetime(2025, 1, 6, 6, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE).timestamp()
        targets, source = self._resolve(gap, _monday_blocks({}), house=_house_ctx(), room_vacant=True)
        assert (targets, source) == (TargetTemps(16.0, 28.0), TargetSource.HOUSE)

    def test_occupied_following_room_gets_house_pair(self):
        gap = datetime(2025, 1, 6, 6, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE).timestamp()
        targets, _ = self._resolve(gap, _monday_blocks({}), house=_house_ctx())
        assert targets == TargetTemps(21.0, 24.0)

    def test_independent_vacant_room_caps_to_its_own_eco(self):
        targets, source = self._resolve(_monday_ts(), None, house=_house_ctx(), follow=False, room_vacant=True)
        assert (targets, source) == (TargetTemps(17.0, 27.0), TargetSource.ROOM_BASE)

    def test_compatibility_wrapper_returns_targets_only(self):
        gap = datetime(2025, 1, 6, 6, 0, 0, tzinfo=dt_util.DEFAULT_TIME_ZONE).timestamp()
        result = resolve_targets_at_time(
            ts=gap,
            schedule_blocks=_monday_blocks({}),
            override_until=None,
            override_heat=None,
            override_cool=None,
            vacation_until=None,
            vacation_temp=None,
            comfort_heat=21.0,
            comfort_cool=24.0,
            eco_heat=17.0,
            eco_cool=27.0,
            house=_house_ctx(),
            follow_house=True,
        )
        assert result == TargetTemps(21.0, 24.0)


class TestScheduleBlockSetsTarget:
    """Only a block carrying a sanitized temperature claims the schedule rung."""

    def test_block_with_temps_is_true(self):
        assert schedule_block_sets_target(_monday_ts(), _monday_blocks({"temperature": 22.0})) is True
        assert schedule_block_sets_target(_monday_ts(), _monday_blocks({"heat_temperature": 22.0})) is True

    def test_block_without_temps_is_false(self):
        assert schedule_block_sets_target(_monday_ts(), _monday_blocks({})) is False

    def test_no_block_is_false(self):
        assert schedule_block_sets_target(_monday_ts(), None) is False
        assert schedule_block_sets_target(_monday_ts(), {}) is False

    def test_out_of_range_only_is_false(self):
        assert schedule_block_sets_target(_monday_ts(), _monday_blocks({"temperature": 110})) is False


class TestHouseOverrideHoldUntil:
    """House holds follow the same schedule boundary rules as room overrides."""

    def test_hold_ends_at_next_schedule_change(self):
        nxt = datetime.fromtimestamp(time.time() + 1800, UTC)
        assert house_override_hold_until(_hold_hass("on", nxt), _house()) == nxt.timestamp()

    def test_no_schedule_returns_none(self):
        assert house_override_hold_until(_hold_hass(None), {"schedules": []}) is None

    def test_unreadable_schedule_uses_the_fallback(self):
        now = time.time()
        until = house_override_hold_until(_hold_hass("unavailable"), _house(), now=now)
        assert until == now + SCHEDULE_UNREADABLE_HOLD_HOURS * 3600

    def test_past_next_event_uses_the_fallback(self):
        now = time.time()
        stale = datetime.fromtimestamp(now - 60, UTC)
        until = house_override_hold_until(_hold_hass("on", stale), _house(), now=now)
        assert until == now + SCHEDULE_UNREADABLE_HOLD_HOURS * 3600

    def test_override_active_at_expiry(self):
        now = time.time()
        house = _house(override_heat=22.0, override_until=now + 60)
        assert house_override_active_at(now, house) is True
        assert house_override_active_at(now + 120, house) is False
        assert house_override_active_at(now, _house(override_heat=22.0, override_until=None)) is True


class TestHousePriceShift:
    """Effective shift comes from the house; a house hold is never price-shifted."""

    def _signal(self, now: float) -> PriceSignal:
        return PriceSignal(current_level=LEVEL_CHEAP, now=now)

    def test_shift_override_replaces_the_room_value(self):
        now = time.time()
        shifted, deltas = price_shifted(
            TargetTemps(21.0, 24.0),
            now,
            {"price_shift": 0.0},
            {},
            self._signal(now),
            presence_away=False,
            room_vacant=False,
            shift_override=1.0,
        )
        assert shifted == TargetTemps(21.0, 23.0)
        assert deltas == (0.0, -1.0)

    def test_zero_shift_override_disables_the_room_value(self):
        now = time.time()
        shifted, deltas = price_shifted(
            TargetTemps(21.0, 24.0),
            now,
            {"price_shift": 2.0},
            {},
            self._signal(now),
            presence_away=False,
            room_vacant=False,
            shift_override=0.0,
        )
        assert shifted == TargetTemps(21.0, 24.0)
        assert deltas == (0.0, 0.0)

    def test_schedule_sets_target_respects_house_hold(self):
        now = time.time()
        assert (
            schedule_sets_target(now, {}, {}, presence_away=False, room_vacant=False, house_override_active=True)
            is False
        )
        assert (
            schedule_sets_target(now, {}, {}, presence_away=False, room_vacant=False, house_override_active=False)
            is True
        )

    def test_house_hold_suppresses_the_price_shift_per_timestamp(self):
        # One minute into an hour slot: the signal's level only covers the slot it
        # was built in, so a wall-clock `now` made the +120 s assertion flip in the
        # last two minutes of every hour.
        now = float(int(time.time() // 3600) * 3600 + 60)
        room = {"comfort_heat": 21.0, "comfort_cool": 24.0, "price_shift": 1.0}
        house = _house(price_shift=1.0, override_heat=22.0, override_until=now + 60)
        ctx = _house_ctx(house)
        resolver = make_target_resolver(
            None,
            room,
            {},
            house=ctx,
            follow_house=True,
            price_signal=self._signal(now),
        )
        # Inside the hold: the house's overridden pair is untouched by the price.
        assert resolver(now) == TargetTemps(22.0, 24.0)
        # After the hold expires, the price shift applies to the house's pair
        # using the house's own price_shift setting.
        assert resolver(now + 120) == TargetTemps(21.0, 23.0)

    def test_following_room_uses_house_shift_even_without_room_shift(self):
        now = time.time()
        room = {"comfort_heat": 21.0, "comfort_cool": 24.0}
        house = _house(price_shift=1.0)
        resolver = make_target_resolver(
            None,
            room,
            {},
            house=_house_ctx(house),
            follow_house=True,
            price_signal=self._signal(now),
        )
        assert resolver(now) == TargetTemps(21.0, 23.0)
