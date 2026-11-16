"""Tests for occupancy_manager: dwell filtering, Occupancy Trends and Room Priority."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import cast
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from homeassistant.util import dt as dt_util

from custom_components.roommind.managers import occupancy_manager as occ
from custom_components.roommind.managers.occupancy_manager import (
    AWAY_PRIORITY_WEIGHT,
    BINS_PER_DAY,
    DEFAULT_SLOT_PROBABILITY,
    FREEZE_PRIORITY_WEIGHT,
    GUARDRAIL_PRIORITY_WEIGHT,
    OCCUPANCY_MODE_FOLLOW_SCHEDULE,
    OCCUPANCY_MODE_GUARDRAIL_ONLY,
    OCCUPANCY_MODE_REACTIVE_TREND,
    OCCUPANCY_MODE_SCHEDULE_ASSUMED,
    OVERRIDE_PRIORITY_WEIGHT,
    TREND_EMA_ALPHA,
    TREND_MATURITY_DAYS,
    VACANCY_GRACE_MINUTES,
    DwellTracker,
    OccupancyManager,
    OccupancyTrendModel,
    RoomOccupancyState,
    read_raw_occupancy,
)

TZ_NAME = "America/New_York"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _hass_with_states(states: dict[str, str | None]) -> MagicMock:
    """Build a MagicMock hass whose state machine returns *states*."""
    hass = MagicMock()
    hass.is_running = True
    hass.data = {}

    def _get(entity_id: str) -> MagicMock | None:
        value = states.get(entity_id)
        if value is None:
            return None
        state = MagicMock()
        state.entity_id = entity_id
        state.state = value
        return state

    hass.states.get = MagicMock(side_effect=_get)
    return hass


def _patch_registry(monkeypatch: pytest.MonkeyPatch, known: dict[str, MagicMock | None]) -> None:
    """Patch the entity registry lookup used by _is_entity_gone."""
    registry = MagicMock()
    registry.async_get = MagicMock(side_effect=known.get)

    def _registry_for(_hass: object) -> MagicMock:
        return registry

    monkeypatch.setattr(occ.presence_utils.er, "async_get", _registry_for)


def _registry_entry(disabled_by: object | None = None) -> MagicMock:
    entry = MagicMock()
    entry.disabled_by = disabled_by
    return entry


@contextmanager
def _tz(name: str) -> Iterator[None]:
    """Temporarily set Home Assistant's configured timezone."""
    original = dt_util.get_default_time_zone()
    dt_util.set_default_time_zone(ZoneInfo(name))
    try:
        yield
    finally:
        dt_util.set_default_time_zone(original)


def _local_ts(year: int, month: int, day: int, hour: int, minute: int = 0) -> float:
    """Return an epoch timestamp for a wall-clock time in the current HA timezone."""
    zone = dt_util.get_default_time_zone()
    return datetime(year, month, day, hour, minute, tzinfo=zone).timestamp()


def _reactive_room(area_id: str = "kitchen") -> dict:
    return {
        "occupancy_sensors": [f"binary_sensor.{area_id}_motion"],
        "occupancy_mode": OCCUPANCY_MODE_REACTIVE_TREND,
    }


# --------------------------------------------------------------------------- #
# read_raw_occupancy
# --------------------------------------------------------------------------- #


def test_read_raw_occupancy_without_sensors() -> None:
    hass = _hass_with_states({})
    assert read_raw_occupancy(hass, {"occupancy_sensors": []}) == (False, False)
    assert read_raw_occupancy(hass, {}) == (False, False)


def test_read_raw_occupancy_any_sensor_on() -> None:
    hass = _hass_with_states({"binary_sensor.a": "off", "binary_sensor.b": "on"})
    room = {"occupancy_sensors": ["binary_sensor.a", "binary_sensor.b"]}
    assert read_raw_occupancy(hass, room) == (True, True)


def test_read_raw_occupancy_all_sensors_off() -> None:
    hass = _hass_with_states({"binary_sensor.a": "off", "binary_sensor.b": "off"})
    room = {"occupancy_sensors": ["binary_sensor.a", "binary_sensor.b"]}
    assert read_raw_occupancy(hass, room) == (False, True)


def test_read_raw_occupancy_unavailable_and_unknown_are_failsafe_occupied() -> None:
    for value in ("unavailable", "unknown"):
        hass = _hass_with_states({"binary_sensor.a": value, "binary_sensor.b": "off"})
        room = {"occupancy_sensors": ["binary_sensor.a", "binary_sensor.b"]}
        assert read_raw_occupancy(hass, room) == (True, True)


def test_read_raw_occupancy_missing_state_counts_occupied(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entity that exists but has no state yet (startup) is occupied by fail-safe."""
    hass = _hass_with_states({})
    _patch_registry(monkeypatch, {"binary_sensor.a": _registry_entry()})
    assert read_raw_occupancy(hass, {"occupancy_sensors": ["binary_sensor.a"]}) == (True, True)


def test_read_raw_occupancy_skips_deleted_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    hass = _hass_with_states({"binary_sensor.b": "off"})
    _patch_registry(monkeypatch, {"binary_sensor.a": None, "binary_sensor.b": _registry_entry()})
    room = {"occupancy_sensors": ["binary_sensor.a", "binary_sensor.b"]}
    assert read_raw_occupancy(hass, room) == (False, True)


def test_read_raw_occupancy_all_entities_gone_has_no_sensors(monkeypatch: pytest.MonkeyPatch) -> None:
    hass = _hass_with_states({})
    _patch_registry(monkeypatch, {"binary_sensor.a": _registry_entry(disabled_by="user")})
    assert read_raw_occupancy(hass, {"occupancy_sensors": ["binary_sensor.a"]}) == (False, False)


def test_read_raw_occupancy_broken_registry_stays_failsafe(monkeypatch: pytest.MonkeyPatch) -> None:
    hass = _hass_with_states({})

    def _broken(_hass: object) -> MagicMock:
        raise RuntimeError("registry not ready")

    monkeypatch.setattr(occ.presence_utils.er, "async_get", _broken)
    assert read_raw_occupancy(hass, {"occupancy_sensors": ["binary_sensor.a"]}) == (True, True)


# --------------------------------------------------------------------------- #
# DwellTracker
# --------------------------------------------------------------------------- #


def test_dwell_90_second_pass_through_does_not_saturate() -> None:
    """A 90-second walk-through only reaches half of a 3-minute dwell requirement."""
    tracker = DwellTracker()
    assert tracker.step("kitchen", True, 0.0) == 0.0
    assert tracker.step("kitchen", True, 30.0) == pytest.approx(1.0 / 6.0)
    assert tracker.step("kitchen", True, 60.0) == pytest.approx(1.0 / 3.0)
    assert tracker.step("kitchen", True, 90.0) == pytest.approx(0.5)
    assert tracker.score("kitchen") == pytest.approx(0.5)


def test_dwell_sustained_presence_saturates() -> None:
    tracker = DwellTracker()
    for second in range(0, 181, 30):
        score = tracker.step("kitchen", True, float(second))
    assert score == 1.0
    # Further occupancy can never exceed 1.0.
    assert tracker.step("kitchen", True, 600.0) == 1.0


def test_dwell_zero_dwell_minutes_is_immediate() -> None:
    tracker = DwellTracker()
    assert tracker.step("kitchen", True, 100.0, dwell_minutes=0.0) == 1.0
    assert tracker.step("kitchen", True, 101.0, dwell_minutes=-2.0) == 1.0


def test_dwell_ignores_clock_going_backwards() -> None:
    tracker = DwellTracker()
    tracker.step("kitchen", True, 100.0)
    # A backwards timestamp never adds dwell, and the next forward step only
    # counts the time since the backwards observation.
    assert tracker.step("kitchen", True, 50.0) == 0.0
    assert tracker.step("kitchen", True, 80.0) == pytest.approx(1.0 / 6.0)


def test_dwell_vacancy_grace_then_linear_decay() -> None:
    tracker = DwellTracker()
    for second in range(0, 181, 30):
        tracker.step("kitchen", True, float(second))
    assert tracker.score("kitchen") == 1.0

    # Within the grace the score is held, even at the exact grace boundary.
    assert tracker.step("kitchen", False, 200.0) == 1.0
    assert tracker.step("kitchen", False, 200.0 + VACANCY_GRACE_MINUTES * 60.0) == 1.0

    # Past the grace it decays at 1 / eco_delay per minute.  The last step was at
    # the grace boundary, so 8 minutes later at eco_delay 10 -> 0.2 left.
    assert tracker.step("kitchen", False, 200.0 + 120.0 + 480.0, eco_delay_minutes=10.0) == pytest.approx(0.2)
    # Never negative.
    assert tracker.step("kitchen", False, 5000.0, eco_delay_minutes=10.0) == 0.0


def test_dwell_reoccupation_clears_vacancy_timer() -> None:
    tracker = DwellTracker()
    tracker.step("kitchen", False, 0.0)
    tracker.step("kitchen", True, 60.0)
    assert tracker.step("kitchen", False, 120.0) == pytest.approx(1.0 / 3.0)
    # Re-occupy before the grace expires: the vacancy timer is forgotten.
    tracker.step("kitchen", True, 150.0)
    assert tracker.step("kitchen", False, 200.0) == pytest.approx(0.5)


def test_dwell_remove_room_and_reset_all() -> None:
    tracker = DwellTracker()
    tracker.step("kitchen", True, 0.0)
    tracker.step("kitchen", True, 60.0)
    tracker.remove_room("kitchen")
    assert tracker.score("kitchen") == 0.0
    tracker.step("living", True, 0.0)
    tracker.reset_all()
    assert tracker.score("living") == 0.0


def test_dwell_persistence_roundtrip_and_malformed_entries() -> None:
    tracker = DwellTracker()
    tracker.step("kitchen", True, 0.0)
    tracker.step("kitchen", True, 90.0)
    data = tracker.to_dict()
    assert data["kitchen"]["last_step_ts"] == 90.0

    restored = DwellTracker()
    restored.load_data(data)
    assert restored.score("kitchen") == pytest.approx(0.5)
    # A restored tracker continues from the persisted timers.
    assert restored.step("kitchen", True, 180.0) == 1.0

    restored.load_data(
        {
            "good": {"score": 2.0, "last_step_ts": 10, "vacant_since_ts": 5},
            "bad": "not-a-dict",
            "empty": {},
            "weird": {"score": "x", "last_step_ts": None, "vacant_since_ts": "y"},
        }
    )
    assert restored.score("good") == 1.0
    assert restored.score("bad") == 0.0
    assert restored.score("weird") == 0.0
    restored.load_data(cast(dict, None))
    assert restored.score("good") == 0.0


# --------------------------------------------------------------------------- #
# OccupancyTrendModel
# --------------------------------------------------------------------------- #


def test_trend_learns_local_weekday_bin() -> None:
    with _tz(TZ_NAME):
        model = OccupancyTrendModel()
        monday_1000 = _local_ts(2026, 1, 5, 10, 0)
        monday_1015 = _local_ts(2026, 1, 5, 10, 15)
        assert monday_1015 > monday_1000
        model.observe(monday_1000, True)
        model.observe(monday_1015, True)

        slot = 10 * 4
        assert model.weekday_bins[slot] == pytest.approx(
            (1.0 - TREND_EMA_ALPHA) * DEFAULT_SLOT_PROBABILITY + TREND_EMA_ALPHA * 1.0
        )
        assert model.weekday_bins[slot + 1] == DEFAULT_SLOT_PROBABILITY
        assert model.weekend_bins[slot] == DEFAULT_SLOT_PROBABILITY
        assert model.days_observed == pytest.approx(1.0 / BINS_PER_DAY)


def test_trend_learns_weekend_separately() -> None:
    with _tz(TZ_NAME):
        model = OccupancyTrendModel()
        model.observe(_local_ts(2026, 1, 10, 10, 0), True)  # Saturday
        model.observe(_local_ts(2026, 1, 10, 10, 15), True)
        slot = 40
        assert model.weekend_bins[slot] == pytest.approx(0.3625)
        assert model.weekday_bins[slot] == DEFAULT_SLOT_PROBABILITY
        assert model.days_observed == pytest.approx(1.0 / BINS_PER_DAY)


def test_trend_local_timezone_decides_bin_and_day_type() -> None:
    """22:30 New York is already the next UTC day; the local day must win."""
    with _tz(TZ_NAME):
        model = OccupancyTrendModel()
        friday_evening = _local_ts(2026, 1, 9, 22, 30)
        friday_evening_later = _local_ts(2026, 1, 9, 22, 45)
        model.observe(friday_evening, True)
        model.observe(friday_evening_later, True)
        # Still Friday local -> weekday bins, slot 22*4+2 = 90.
        assert model.weekday_bins[90] == pytest.approx(0.3625)
        assert model.weekend_bins[90] == DEFAULT_SLOT_PROBABILITY


def test_trend_midnight_smoothing_wrap() -> None:
    with _tz(TZ_NAME):
        model = OccupancyTrendModel()
        weekday = [DEFAULT_SLOT_PROBABILITY] * BINS_PER_DAY
        weekday[95] = 1.0
        weekday[0] = 0.0
        weekday[1] = 0.0
        weekend = [DEFAULT_SLOT_PROBABILITY] * BINS_PER_DAY
        weekend[94] = 0.4
        weekend[95] = 0.2
        weekend[0] = 0.8
        model.load_data({"weekday_bins": weekday, "weekend_bins": weekend})

        # Monday 00:00: left neighbour wraps to weekday bin 95.
        assert model.raw_smoothed_probability(_local_ts(2026, 1, 5, 0, 0)) == pytest.approx(0.25)
        # Friday 23:45: right neighbour wraps to weekday bin 0.
        assert model.raw_smoothed_probability(_local_ts(2026, 1, 9, 23, 45)) == pytest.approx(0.5625)
        # Sunday 23:45 uses the weekend array and wraps to weekend bin 0.
        assert model.raw_smoothed_probability(_local_ts(2026, 1, 11, 23, 45)) == pytest.approx(0.4)


def test_trend_maturity_scaling() -> None:
    with _tz(TZ_NAME):
        model = OccupancyTrendModel()
        monday_noon = _local_ts(2026, 1, 5, 12, 0)
        raw = model.raw_smoothed_probability(monday_noon)
        assert raw == pytest.approx(DEFAULT_SLOT_PROBABILITY)
        assert model.weighted_trend_probability(monday_noon) == 0.0

        model.days_observed = TREND_MATURITY_DAYS / 2.0
        assert model.weighted_trend_probability(monday_noon) == pytest.approx(raw * 0.5)

        model.days_observed = TREND_MATURITY_DAYS
        assert model.weighted_trend_probability(monday_noon) == pytest.approx(raw)

        model.days_observed = TREND_MATURITY_DAYS * 3.0
        assert model.weighted_trend_probability(monday_noon) == pytest.approx(raw)

        model.days_observed = -1.0
        assert model.weighted_trend_probability(monday_noon) == 0.0


def test_trend_vacation_freezes_learning() -> None:
    with _tz(TZ_NAME):
        model = OccupancyTrendModel()
        monday_1000 = _local_ts(2026, 1, 5, 10, 0)
        model.observe(monday_1000, True)
        # Vacation begins before the slot completes: the partial sample is dropped.
        for minute in (15, 30, 45):
            model.observe(_local_ts(2026, 1, 5, 10, minute), True, vacation_active=True)
        assert model.days_observed == 0.0
        assert model.weekday_bins[40] == DEFAULT_SLOT_PROBABILITY

        # Back home: learning resumes and only counts real observations.
        model.observe(_local_ts(2026, 1, 5, 11, 0), True)
        model.observe(_local_ts(2026, 1, 5, 11, 15), True)
        assert model.weekday_bins[44] == pytest.approx(0.3625)
        assert model.days_observed == pytest.approx(1.0 / BINS_PER_DAY)


def test_trend_clock_going_backwards_restarts_sample() -> None:
    with _tz(TZ_NAME):
        model = OccupancyTrendModel()
        model.observe(_local_ts(2026, 1, 5, 10, 15), True)
        model.observe(_local_ts(2026, 1, 5, 10, 0), True)
        assert model.days_observed == 0.0
        assert model.weekday_bins[40] == DEFAULT_SLOT_PROBABILITY
        # The new sample continues from slot 40 and folds normally afterwards.
        model.observe(_local_ts(2026, 1, 5, 10, 15), True)
        assert model.weekday_bins[40] == pytest.approx(0.3625)


def test_trend_step_and_update_aliases() -> None:
    with _tz(TZ_NAME):
        model = OccupancyTrendModel()
        model.step(_local_ts(2026, 1, 5, 9, 0), True)
        model.update(_local_ts(2026, 1, 5, 9, 15), True)
        assert model.weekday_bins[36] == pytest.approx(0.3625)


def test_trend_fold_without_sample_is_a_no_op() -> None:
    model = OccupancyTrendModel()
    model._fold_completed_slot()
    assert model.days_observed == 0.0


def test_trend_persistence_roundtrip_mid_slot() -> None:
    with _tz(TZ_NAME):
        model = OccupancyTrendModel()
        model.observe(_local_ts(2026, 1, 5, 10, 0), True)
        data = model.to_dict()
        assert "sample" in data

        restored = OccupancyTrendModel()
        restored.load_data(data)
        # The partial 10:00 slot survives, so the 10:15 sample completes it.
        restored.observe(_local_ts(2026, 1, 5, 10, 15), True)
        assert restored.weekday_bins[40] == pytest.approx(0.3625)
        assert restored.days_observed == pytest.approx(1.0 / BINS_PER_DAY)


def test_trend_load_data_sanitizes_bad_input() -> None:
    model = OccupancyTrendModel()
    model.load_data(
        {
            "weekday_bins": [2.0, "not-a-number", -1.0],
            "weekend_bins": "not-a-list",
            "days_observed": "not-a-number",
        }
    )
    assert len(model.weekday_bins) == BINS_PER_DAY
    assert model.weekday_bins[0] == 1.0
    assert model.weekday_bins[1] == DEFAULT_SLOT_PROBABILITY
    assert model.weekday_bins[2] == 0.0
    assert model.weekend_bins == [DEFAULT_SLOT_PROBABILITY] * BINS_PER_DAY
    assert model.days_observed == 0.0
    model.load_data(cast(dict, None))
    assert model.weekday_bins == [DEFAULT_SLOT_PROBABILITY] * BINS_PER_DAY


@pytest.mark.parametrize(
    "sample",
    [
        {"day_key": 9, "slot_idx": 0, "sum": 1.0, "count": 1},
        {"day_key": 1, "slot_idx": 200, "sum": 1.0, "count": 1},
        {"day_key": 1, "slot_idx": 0, "sum": "x", "count": 1},
        {"day_key": 1, "slot_idx": 0, "sum": 1.0, "count": 0},
    ],
)
def test_trend_load_data_rejects_bad_sample(sample: dict) -> None:
    model = OccupancyTrendModel()
    model.load_data({"sample": sample})
    assert model._cur_day_key is None
    assert model._cur_slot is None


def test_trend_profile_24h_rounding_and_weekend() -> None:
    model = OccupancyTrendModel()
    weekday = [0.3] * BINS_PER_DAY
    weekday[0] = 0.9
    model.load_data({"weekday_bins": weekday, "weekend_bins": weekday})
    profile = model.profile_24h()
    assert len(profile) == BINS_PER_DAY
    assert profile[1] == pytest.approx(0.45)  # 0.25*0.9 + 0.5*0.3 + 0.25*0.3
    assert profile[95] == pytest.approx(0.45)  # 0.25*0.3 + 0.5*0.3 + 0.25*0.9
    assert model.profile_24h(weekend=True) == profile


# --------------------------------------------------------------------------- #
# OccupancyManager: mode resolution and live state
# --------------------------------------------------------------------------- #


def test_resolve_occupancy_mode_explicit_and_derived() -> None:
    manager = OccupancyManager()
    assert manager.resolve_occupancy_mode({"occupancy_mode": OCCUPANCY_MODE_GUARDRAIL_ONLY}) == (
        OCCUPANCY_MODE_GUARDRAIL_ONLY
    )
    assert (
        manager.resolve_occupancy_mode(
            {"occupancy_mode": OCCUPANCY_MODE_SCHEDULE_ASSUMED, "occupancy_sensors": ["binary_sensor.a"]}
        )
        == OCCUPANCY_MODE_SCHEDULE_ASSUMED
    )
    # Invalid explicit mode falls back to the derivation.
    assert (
        manager.resolve_occupancy_mode({"occupancy_mode": "nonsense", "occupancy_sensors": ["binary_sensor.a"]})
        == OCCUPANCY_MODE_SCHEDULE_ASSUMED
    )
    # Sensors plus occupancy eco enabled -> reactive trend.
    assert (
        manager.resolve_occupancy_mode({"occupancy_sensors": ["binary_sensor.a"], "occupancy_eco_enabled": True})
        == OCCUPANCY_MODE_REACTIVE_TREND
    )
    # Sensors without eco -> schedule assumed (PIR may miss a sleeping person).
    assert manager.resolve_occupancy_mode({"occupancy_sensors": ["binary_sensor.a"]}) == OCCUPANCY_MODE_SCHEDULE_ASSUMED
    # No sensors -> follow the schedule, whatever the legacy eco flag says.
    assert manager.resolve_occupancy_mode({"occupancy_eco_enabled": True}) == OCCUPANCY_MODE_FOLLOW_SCHEDULE
    assert manager.resolve_occupancy_mode({}) == OCCUPANCY_MODE_FOLLOW_SCHEDULE


def test_step_room_reactive_occupancy_and_priority() -> None:
    hass = _hass_with_states({"binary_sensor.kitchen_motion": "on"})
    manager = OccupancyManager()
    room = _reactive_room()

    state = manager.step_room(hass, "kitchen", room, 0.0)
    assert isinstance(state, RoomOccupancyState)
    assert state.raw_occupied is True
    assert state.has_sensors is True
    # The very first sample has no elapsed dwell yet, so it is not "present"
    # for filtering purposes until the dwell has actually built up.
    assert state.is_vacant is True
    for second in range(30, 181, 30):
        state = manager.step_room(hass, "kitchen", room, float(second))
    assert state.dwell_score == 1.0
    assert state.effective_occupancy == 1.0
    assert state.priority_weight == 1.0
    assert manager.last_state("kitchen") is state


def test_step_room_sensor_off_is_vacant_with_guardrail_priority() -> None:
    hass = _hass_with_states({"binary_sensor.kitchen_motion": "off"})
    manager = OccupancyManager()
    state = manager.step_room(hass, "kitchen", _reactive_room(), 100.0)
    assert state.raw_occupied is False
    assert state.has_sensors is True
    assert state.dwell_score == 0.0
    assert state.is_vacant is True
    assert state.effective_occupancy == 0.0
    assert state.priority_weight == GUARDRAIL_PRIORITY_WEIGHT


def test_step_room_sleeping_pir_schedule_assumed_holds_priority() -> None:
    """A sleeping person's PIR is off, but the schedule floor keeps priority at 1.0."""
    hass = _hass_with_states({"binary_sensor.bedroom_motion": "off"})
    manager = OccupancyManager()
    room = {
        "occupancy_sensors": ["binary_sensor.bedroom_motion"],
        "occupancy_mode": OCCUPANCY_MODE_SCHEDULE_ASSUMED,
    }
    state = manager.step_room(hass, "bedroom", room, 1000.0, schedule_active_now=True)
    assert state.raw_occupied is False
    assert state.dwell_score == 0.0
    assert state.effective_occupancy == 1.0
    assert state.priority_weight == 1.0

    state = manager.step_room(hass, "bedroom", room, 1300.0, schedule_active_now=False)
    assert state.effective_occupancy == 0.0
    assert state.priority_weight == GUARDRAIL_PRIORITY_WEIGHT


def test_step_room_guardrail_and_follow_schedule_modes() -> None:
    hass = _hass_with_states({"binary_sensor.kitchen_motion": "on"})
    manager = OccupancyManager()
    guardrail = manager.step_room(hass, "kitchen", {"occupancy_mode": OCCUPANCY_MODE_GUARDRAIL_ONLY}, 0.0)
    assert guardrail.effective_occupancy == 0.0
    assert guardrail.priority_weight == GUARDRAIL_PRIORITY_WEIGHT

    follow_on = manager.step_room(
        hass,
        "living",
        {"occupancy_mode": OCCUPANCY_MODE_FOLLOW_SCHEDULE},
        0.0,
        schedule_active_now=True,
    )
    assert follow_on.effective_occupancy == 1.0
    assert follow_on.priority_weight == 1.0

    follow_off = manager.step_room(
        hass,
        "living",
        {"occupancy_mode": OCCUPANCY_MODE_FOLLOW_SCHEDULE},
        30.0,
        schedule_active_now=False,
    )
    assert follow_off.effective_occupancy == 0.0
    assert follow_off.priority_weight == GUARDRAIL_PRIORITY_WEIGHT


def test_step_room_trend_contributes_when_mature() -> None:
    hass = _hass_with_states({"binary_sensor.kitchen_motion": "off"})
    manager = OccupancyManager()
    manager.load_data(
        {
            "trends": {
                "kitchen": {
                    "weekday_bins": [0.9] * BINS_PER_DAY,
                    "weekend_bins": [0.9] * BINS_PER_DAY,
                    "days_observed": TREND_MATURITY_DAYS,
                }
            }
        }
    )
    with _tz(TZ_NAME):
        now = _local_ts(2026, 1, 5, 10, 0)
        state = manager.step_room(hass, "kitchen", _reactive_room(), now)
    assert state.trend_probability == pytest.approx(0.9)
    assert state.effective_occupancy == pytest.approx(0.9)
    assert state.priority_weight == pytest.approx(round(0.08 + 0.92 * 0.9, 4))


def test_step_room_override_away_vacation_and_window() -> None:
    hass = _hass_with_states({"binary_sensor.kitchen_motion": "on"})
    manager = OccupancyManager()
    room = _reactive_room()

    override = manager.step_room(hass, "kitchen", room, 0.0, override_active=True)
    assert (override.effective_occupancy, override.priority_weight) == (1.0, OVERRIDE_PRIORITY_WEIGHT)

    away = manager.step_room(hass, "kitchen", room, 30.0, presence_away=True)
    assert (away.effective_occupancy, away.priority_weight) == (0.0, AWAY_PRIORITY_WEIGHT)

    ignored = manager.step_room(hass, "kitchen", {**room, "ignore_presence": True}, 60.0, presence_away=True)
    assert ignored.priority_weight > AWAY_PRIORITY_WEIGHT

    vacation = manager.step_room(hass, "kitchen", room, 90.0, vacation_active=True)
    assert (vacation.effective_occupancy, vacation.priority_weight) == (0.0, AWAY_PRIORITY_WEIGHT)

    frozen = manager.step_room(hass, "kitchen", room, 120.0, window_open=True, current_temp_c=5.0)
    assert (frozen.effective_occupancy, frozen.priority_weight) == (0.0, FREEZE_PRIORITY_WEIGHT)

    open_window = manager.step_room(hass, "kitchen", room, 150.0, window_open=True, current_temp_c=12.0)
    assert (open_window.effective_occupancy, open_window.priority_weight) == (0.0, 0.0)

    unknown_temp = manager.step_room(hass, "kitchen", room, 180.0, window_open=True)
    assert unknown_temp.priority_weight == 0.0


def test_step_room_outdoor_and_disabled_room_zero_priority() -> None:
    hass = _hass_with_states({"binary_sensor.kitchen_motion": "on"})
    manager = OccupancyManager()
    outdoor = manager.step_room(hass, "porch", {**_reactive_room("porch"), "is_outdoor": True}, 0.0)
    assert (outdoor.effective_occupancy, outdoor.priority_weight) == (0.0, 0.0)
    disabled = manager.step_room(hass, "kitchen", {**_reactive_room(), "climate_control_enabled": False}, 30.0)
    assert (disabled.effective_occupancy, disabled.priority_weight) == (0.0, 0.0)


def test_step_room_custom_and_malformed_dwell_settings() -> None:
    hass = _hass_with_states({"binary_sensor.kitchen_motion": "on"})
    manager = OccupancyManager()
    fast = {**_reactive_room(), "occupancy_dwell_minutes": 0.5, "occupancy_eco_delay": 5}
    manager.step_room(hass, "kitchen", fast, 0.0)
    state = manager.step_room(hass, "kitchen", fast, 15.0)
    assert state.dwell_score == 0.5

    malformed = {
        **_reactive_room("living"),
        "occupancy_dwell_minutes": "soon",
        "occupancy_eco_delay": None,
    }
    living_hass = _hass_with_states({"binary_sensor.living_motion": "on"})
    manager.step_room(living_hass, "living", malformed, 0.0)
    # Default dwell of 3 minutes still applies.
    assert manager.step_room(living_hass, "living", malformed, 60.0).dwell_score == pytest.approx(1.0 / 3.0)


def test_manager_load_data_accepts_room_keyed_and_flat_shapes() -> None:
    manager = OccupancyManager()
    manager.load_data(
        {
            "kitchen": {
                "dwell": {"score": 0.5, "last_step_ts": 100.0, "vacant_since_ts": None},
                "trend": {
                    "weekday_bins": [0.6] * BINS_PER_DAY,
                    "weekend_bins": [0.7] * BINS_PER_DAY,
                    "days_observed": 3.0,
                },
            },
            "living": {"score": 0.25, "last_step_ts": 50.0},
            "office": {"weekday_bins": [0.4] * BINS_PER_DAY, "days_observed": 1.0},
            "dwell_only": {"dwell": {"score": 0.1}, "trend": {}},
            "broken": "not-a-dict",
            "empty": {},
        }
    )
    assert manager.dwell_score("kitchen") == pytest.approx(0.5)
    assert manager.dwell_score("living") == pytest.approx(0.25)
    assert manager.dwell_score("dwell_only") == pytest.approx(0.1)
    kitchen = manager.trend_model("kitchen")
    office = manager.trend_model("office")
    assert kitchen is not None and kitchen.days_observed == pytest.approx(3.0)
    assert office is not None and office.days_observed == pytest.approx(1.0)
    assert manager.trend_model("living") is None
    assert manager.trend_model("dwell_only") is None
    assert manager.trend_model("broken") is None
    assert manager.trend_model("empty") is None


def test_manager_trend_accessors_and_24h_profile() -> None:
    manager = OccupancyManager()
    assert manager.weighted_trend_probability("kitchen", 0.0) == 0.0
    assert manager.raw_trend_probability("kitchen", 0.0) == 0.0
    assert manager.trend_model("kitchen") is None
    assert manager.get_24h_profile("kitchen") == [DEFAULT_SLOT_PROBABILITY] * BINS_PER_DAY

    manager.load_data(
        {
            "trends": {
                "kitchen": {
                    "weekday_bins": [0.3] * BINS_PER_DAY,
                    "weekend_bins": [0.7] * BINS_PER_DAY,
                    "days_observed": 1.0,
                }
            }
        }
    )
    with _tz(TZ_NAME):
        weekday_ts = _local_ts(2026, 1, 5, 10, 0)
    assert manager.raw_trend_probability("kitchen", weekday_ts) == pytest.approx(0.3)
    assert manager.weighted_trend_probability("kitchen", weekday_ts) == pytest.approx(0.3 / TREND_MATURITY_DAYS)
    assert manager.get_24h_profile("kitchen") == [0.3] * BINS_PER_DAY
    assert manager.get_24h_profile("kitchen", weekend=True) == [0.7] * BINS_PER_DAY


def test_manager_persistence_roundtrip_and_lifecycle() -> None:
    with _tz(TZ_NAME):
        hass = _hass_with_states({"binary_sensor.kitchen_motion": "on"})
        manager = OccupancyManager()
        room = _reactive_room()
        manager.step_room(hass, "kitchen", room, _local_ts(2026, 1, 5, 10, 0))
        manager.step_room(hass, "kitchen", room, _local_ts(2026, 1, 5, 10, 15))
        manager.step_room(hass, "kitchen", room, _local_ts(2026, 1, 5, 10, 30))

        data = manager.to_dict()
        assert "kitchen" in data  # room-keyed, so a single room can be cleared
        restored = OccupancyManager()
        restored.load_data(data)
        model = restored.trend_model("kitchen")
        assert model is not None
        assert restored.dwell_score("kitchen") == pytest.approx(1.0)
        assert model.weekday_bins[40] == pytest.approx(0.3625)
        assert model.days_observed == pytest.approx(2.0 / BINS_PER_DAY)
        assert restored.weighted_trend_probability("kitchen", _local_ts(2026, 1, 5, 10, 30)) > 0.0

        restored.remove_room("kitchen")
        assert restored.trend_model("kitchen") is None
        assert restored.dwell_score("kitchen") == 0.0
        assert restored.last_state("kitchen") is None

        restored.step_room(hass, "kitchen", room, _local_ts(2026, 1, 5, 11, 0))
        restored.reset_room("kitchen")
        reset_model = restored.trend_model("kitchen")
        assert reset_model is not None
        assert reset_model.days_observed == 0.0
        assert restored.dwell_score("kitchen") == 0.0
        assert restored.get_24h_profile("kitchen") == [DEFAULT_SLOT_PROBABILITY] * BINS_PER_DAY
        restored.step_room(hass, "kitchen", room, _local_ts(2026, 1, 5, 11, 30))
        restored.reset_all()
        assert restored.trend_model("kitchen") is None
        restored.load_data(cast(dict, None))
        assert restored.trend_model("kitchen") is None


# --------------------------------------------------------------------------- #
# OccupancyManager.build_priority_series
# --------------------------------------------------------------------------- #


def _series(
    manager: OccupancyManager,
    room: dict,
    now_ts: float,
    n_blocks: int,
    **kwargs: object,
) -> tuple[list[float], list[float]]:
    return manager.build_priority_series("kitchen", room, {}, now_ts, n_blocks, **kwargs)  # type: ignore[arg-type]


def test_build_priority_series_outdoor_and_disabled_are_zero() -> None:
    manager = OccupancyManager()
    for room in ({"is_outdoor": True, "occupancy_sensors": ["binary_sensor.a"]}, {"climate_control_enabled": False}):
        weights, probs = _series(manager, room, 0.0, 4)
        assert weights == [0.0] * 4
        assert probs == [0.0] * 4


def test_build_priority_series_window_open_and_freeze_emergency() -> None:
    manager = OccupancyManager()
    room = _reactive_room()
    weights, probs = _series(manager, room, 0.0, 4, window_open=True, current_temp_c=12.0)
    assert weights == [0.0] * 4 and probs == [0.0] * 4

    # Strictly below 7.0 °C keeps the freeze guardrail alive.
    weights, probs = _series(manager, room, 0.0, 4, window_open=True, current_temp_c=6.9)
    assert weights == [FREEZE_PRIORITY_WEIGHT] * 4 and probs == [0.0] * 4

    weights, _ = _series(manager, room, 0.0, 2, window_open=True, current_temp_c=7.0)
    assert weights == [0.0] * 2

    weights, _ = _series(manager, room, 0.0, 2, window_open=True, current_temp_c=None)
    assert weights == [0.0] * 2


def test_build_priority_series_zero_blocks() -> None:
    manager = OccupancyManager()
    weights, probs = _series(manager, _reactive_room(), 0.0, 0)
    assert weights == [] and probs == []
    weights, probs = _series(manager, _reactive_room(), 0.0, -3)
    assert weights == [] and probs == []


def test_build_priority_series_override_wins_per_block() -> None:
    manager = OccupancyManager()
    room = {"occupancy_mode": OCCUPANCY_MODE_GUARDRAIL_ONLY}
    now = 1000.0
    weights, probs = _series(
        manager,
        room,
        now,
        4,
        override_active_at=lambda ts: ts >= now + 2 * 300.0,
    )
    assert weights[:2] == [GUARDRAIL_PRIORITY_WEIGHT, GUARDRAIL_PRIORITY_WEIGHT]
    assert weights[2:] == [OVERRIDE_PRIORITY_WEIGHT, OVERRIDE_PRIORITY_WEIGHT]
    assert probs == [0.0, 0.0, 1.0, 1.0]


def test_build_priority_series_away_vacation_and_ignore_presence() -> None:
    manager = OccupancyManager()
    room = {"occupancy_mode": OCCUPANCY_MODE_GUARDRAIL_ONLY}
    now = 500.0
    weights, probs = _series(manager, room, now, 2, presence_away=True)
    assert weights == [AWAY_PRIORITY_WEIGHT] * 2 and probs == [0.0, 0.0]

    weights, probs = _series(manager, {**room, "ignore_presence": True}, now, 2, presence_away=True)
    assert weights == [GUARDRAIL_PRIORITY_WEIGHT] * 2 and probs == [0.0, 0.0]

    weights, probs = _series(manager, room, now, 2, vacation_active=True)
    assert weights == [AWAY_PRIORITY_WEIGHT] * 2 and probs == [0.0, 0.0]


def test_build_priority_series_guardrail_mode() -> None:
    manager = OccupancyManager()
    weights, probs = _series(manager, {"occupancy_mode": OCCUPANCY_MODE_GUARDRAIL_ONLY}, 0.0, 3)
    assert weights == [GUARDRAIL_PRIORITY_WEIGHT] * 3
    assert probs == [0.0] * 3


def test_build_priority_series_follows_schedule_per_block() -> None:
    manager = OccupancyManager()
    room = {"occupancy_mode": OCCUPANCY_MODE_FOLLOW_SCHEDULE}
    now = 10_000.0
    weights, probs = _series(
        manager,
        room,
        now,
        4,
        schedule_active_at=lambda ts: int((ts - now) // 300.0) % 2 == 0,
    )
    assert probs == [1.0, 0.0, 1.0, 0.0]
    assert weights[0] == 1.0
    assert weights[1] == GUARDRAIL_PRIORITY_WEIGHT
    assert weights[2] == 1.0
    assert weights[3] == GUARDRAIL_PRIORITY_WEIGHT


def test_build_priority_series_follows_schedule_without_a_callable() -> None:
    manager = OccupancyManager()
    weights, probs = _series(manager, {"occupancy_mode": OCCUPANCY_MODE_FOLLOW_SCHEDULE}, 0.0, 2)
    assert weights == [1.0, 1.0] and probs == [1.0, 1.0]


def test_build_priority_series_reactive_dwell_decays_over_horizon() -> None:
    hass = _hass_with_states({"binary_sensor.kitchen_motion": "on"})
    manager = OccupancyManager()
    room = _reactive_room()
    for second in range(0, 181, 30):
        manager.step_room(hass, "kitchen", room, float(second))

    weights, probs = _series(manager, room, 180.0, 10)
    assert weights[0] == 1.0
    assert probs[0] == 1.0
    expected_k9 = round(GUARDRAIL_PRIORITY_WEIGHT + (1.0 - GUARDRAIL_PRIORITY_WEIGHT) * math.exp(-1.0), 4)
    assert weights[9] == pytest.approx(expected_k9)
    assert weights == sorted(weights, reverse=True)
    assert all(0.0 <= prob <= 1.0 for prob in probs)


def test_build_priority_series_without_sensors_falls_back_to_schedule() -> None:
    manager = OccupancyManager()
    room = {"occupancy_sensors": [], "occupancy_mode": OCCUPANCY_MODE_REACTIVE_TREND}
    weights, probs = _series(manager, room, 0.0, 3, schedule_active_at=lambda _ts: False)
    assert weights == [GUARDRAIL_PRIORITY_WEIGHT] * 3
    assert probs == [0.0] * 3


def test_build_priority_series_schedule_assumed_floor() -> None:
    manager = OccupancyManager()
    room = {
        "occupancy_sensors": ["binary_sensor.bedroom_motion"],
        "occupancy_mode": OCCUPANCY_MODE_SCHEDULE_ASSUMED,
    }
    weights, probs = _series(manager, room, 0.0, 6, schedule_active_at=lambda _ts: True)
    assert weights == [1.0] * 6
    assert probs == [1.0] * 6


def test_build_priority_series_mature_trend_alone_drives_priority() -> None:
    manager = OccupancyManager()
    manager.load_data(
        {
            "trends": {
                "kitchen": {
                    "weekday_bins": [0.9] * BINS_PER_DAY,
                    "weekend_bins": [0.9] * BINS_PER_DAY,
                    "days_observed": TREND_MATURITY_DAYS,
                }
            }
        }
    )
    room = _reactive_room()
    weights, probs = _series(manager, room, 0.0, 3)
    assert probs == pytest.approx([0.9, 0.9, 0.9])
    assert weights == pytest.approx([round(0.08 + 0.92 * 0.9, 4)] * 3)
