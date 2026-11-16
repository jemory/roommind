"""Tests for WetCoilTracker: charge model, RH gate, lockout window and persistence."""

from __future__ import annotations

import math
from typing import cast

import pytest

from custom_components.roommind.managers.wet_coil_manager import (
    FAN_EVAP_TAU_MINUTES,
    HEAT_DRY_TAU_MINUTES,
    IDLE_DRAIN_TAU_MINUTES,
    LOCKOUT_CHARGE_THRESHOLD,
    WET_CHARGE_TAU_MINUTES,
    WetCoilTracker,
)


def test_initial_state_is_dry() -> None:
    tracker = WetCoilTracker()
    assert tracker.charge == 0.0
    assert tracker.last_cooling_ts is None
    assert tracker.state_dirty is False


def test_cooling_charges_the_coil_and_records_timestamp() -> None:
    tracker = WetCoilTracker()
    charge = tracker.step("cooling", WET_CHARGE_TAU_MINUTES, now_ts=1000.0)
    assert charge == pytest.approx(1.0 - math.exp(-1.0))
    assert tracker.last_cooling_ts == 1000.0
    assert tracker.state_dirty is True

    # A second cooling period approaches 1.0 but never exceeds it.
    charge = tracker.step("cooling", 1000.0, now_ts=1060.0)
    assert charge == pytest.approx(1.0, abs=1e-6)
    assert charge <= 1.0
    assert tracker.last_cooling_ts == 1060.0


def test_idle_drain_uses_idle_tau() -> None:
    tracker = WetCoilTracker()
    tracker.step("cooling", 1000.0, now_ts=0.0)
    tracker.state_dirty = False
    charge = tracker.step("idle", IDLE_DRAIN_TAU_MINUTES, now_ts=60.0)
    assert charge == pytest.approx(math.exp(-1.0))
    assert tracker.state_dirty is True


def test_circulation_evaporates_faster_than_idle() -> None:
    tracker = WetCoilTracker()
    tracker.step("cooling", 1000.0, now_ts=0.0)
    charge = tracker.step("circulating", FAN_EVAP_TAU_MINUTES, now_ts=60.0)
    assert charge == pytest.approx(math.exp(-1.0))


def test_heating_dries_the_coil_fastest() -> None:
    tracker = WetCoilTracker()
    tracker.step("cooling", 1000.0, now_ts=0.0)
    charge = tracker.step("heating", HEAT_DRY_TAU_MINUTES, now_ts=60.0)
    assert charge == pytest.approx(math.exp(-1.0))


def test_unknown_or_missing_action_drains_like_idle() -> None:
    tracker = WetCoilTracker()
    tracker.step("cooling", 1000.0, now_ts=0.0)
    charge = tracker.step(None, IDLE_DRAIN_TAU_MINUTES, now_ts=60.0)
    assert charge == pytest.approx(math.exp(-1.0))

    tracker.step("cooling", 1000.0, now_ts=120.0)
    charge = tracker.step("something-else", IDLE_DRAIN_TAU_MINUTES, now_ts=180.0)
    assert charge == pytest.approx(math.exp(-1.0))


def test_charge_below_epsilon_snaps_to_zero() -> None:
    tracker = WetCoilTracker()
    tracker.step("cooling", 1000.0, now_ts=0.0)
    assert tracker.step("idle", 10_000.0, now_ts=60.0) == 0.0


def test_negative_dt_is_ignored() -> None:
    tracker = WetCoilTracker()
    tracker.step("cooling", 10.0, now_ts=0.0)
    before = tracker.charge
    assert tracker.step("cooling", -10.0, now_ts=60.0) == before


def test_state_dirty_only_on_meaningful_change_or_cooling_edge() -> None:
    tracker = WetCoilTracker()
    tracker.step("cooling", 1000.0, now_ts=0.0)
    tracker.step("idle", 10.0, now_ts=60.0)  # cooling stop marks dirty
    tracker.state_dirty = False

    # Tiny idle step: no transition, less than 0.05 change -> not dirty.
    tracker.step("idle", 0.5, now_ts=61.0)
    assert tracker.state_dirty is False

    # Cooling start marks dirty even when the charge barely moves.
    tracker.state_dirty = False
    tracker.step("cooling", 0.1, now_ts=62.0)
    assert tracker.state_dirty is True

    # Cooling stop marks dirty too.
    tracker.state_dirty = False
    tracker.step("idle", 0.1, now_ts=63.0)
    assert tracker.state_dirty is True


def test_lockout_requires_charge_and_positive_window() -> None:
    tracker = WetCoilTracker()
    tracker.step("cooling", 10.0, now_ts=1000.0)
    tracker.charge = LOCKOUT_CHARGE_THRESHOLD - 0.01
    assert tracker.is_circulation_locked(None, {}, now_ts=1001.0) is False

    tracker.charge = 1.0
    assert tracker.is_circulation_locked(None, {"wet_coil_lockout_minutes": 0}, now_ts=1001.0) is False
    assert tracker.is_circulation_locked(None, {"wet_coil_lockout_minutes": -5}, now_ts=1001.0) is False


def test_lockout_without_rh_reading_is_conservative() -> None:
    tracker = WetCoilTracker()
    tracker.step("cooling", 10.0, now_ts=1000.0)
    assert tracker.is_circulation_locked(None, {}, now_ts=1000.0) is True


def test_lockout_rh_gate_and_default_max() -> None:
    tracker = WetCoilTracker()
    tracker.step("cooling", 10.0, now_ts=1000.0)
    # Default max RH is 52 %: dry air allows circulation, humid air blocks it.
    assert tracker.is_circulation_locked(40.0, {}, now_ts=1000.0) is False
    assert tracker.is_circulation_locked(52.0, {}, now_ts=1000.0) is True
    assert tracker.is_circulation_locked(65.0, {}, now_ts=1000.0) is True
    # A custom threshold is honoured.
    assert tracker.is_circulation_locked(55.0, {"wet_coil_max_rh": 60.0}, now_ts=1000.0) is False


def test_lockout_window_expires() -> None:
    tracker = WetCoilTracker()
    tracker.step("cooling", 10.0, now_ts=1000.0)
    settings = {"wet_coil_lockout_minutes": 25}
    assert tracker.is_circulation_locked(80.0, settings, now_ts=1000.0 + 24 * 60.0) is True
    assert tracker.is_circulation_locked(80.0, settings, now_ts=1000.0 + 25 * 60.0) is False
    assert tracker.is_circulation_locked(80.0, settings, now_ts=1000.0 + 10_000.0) is False


def test_lockout_without_last_cooling_timestamp_stays_until_rh_drops() -> None:
    tracker = WetCoilTracker()
    tracker.charge = 0.8
    assert tracker.is_circulation_locked(80.0, {}, now_ts=1000.0) is True
    assert tracker.is_circulation_locked(30.0, {}, now_ts=1000.0) is False


def test_lockout_uses_wall_clock_when_now_is_omitted() -> None:
    tracker = WetCoilTracker()
    tracker.charge = 0.8
    tracker.last_cooling_ts = None
    assert tracker.is_circulation_locked(80.0, {"wet_coil_lockout_minutes": 1}) is True


def test_step_uses_wall_clock_when_now_is_omitted() -> None:
    tracker = WetCoilTracker()
    charge = tracker.step("cooling", 0.0)
    assert charge == 0.0
    assert tracker.last_cooling_ts is not None


def test_persistence_roundtrip() -> None:
    tracker = WetCoilTracker()
    tracker.step("cooling", 10.0, now_ts=1000.0)
    data = tracker.to_dict()
    assert data["charge"] == pytest.approx(tracker.charge)
    assert data["last_cooling_ts"] == 1000.0

    restored = WetCoilTracker()
    restored.load_state(data)
    assert restored.charge == pytest.approx(tracker.charge)
    assert restored.last_cooling_ts == 1000.0
    assert restored.state_dirty is False
    assert restored.is_circulation_locked(80.0, {}, now_ts=1001.0) is True


def test_load_state_clamps_and_ignores_malformed_values() -> None:
    tracker = WetCoilTracker()
    tracker.load_state({"charge": 2.0, "last_cooling_ts": "later"})
    assert tracker.charge == 1.0
    assert tracker.last_cooling_ts is None

    tracker.load_state({"charge": -1.0, "last_cooling_ts": 12.0})
    assert tracker.charge == 0.0
    assert tracker.last_cooling_ts == 12.0

    tracker.load_state({})
    assert tracker.charge == 0.0
    tracker.load_state(cast(dict, None))
    assert tracker.charge == 0.0


def test_reset() -> None:
    tracker = WetCoilTracker()
    tracker.step("cooling", 10.0, now_ts=1000.0)
    tracker.reset()
    assert tracker.charge == 0.0
    assert tracker.last_cooling_ts is None
    assert tracker.state_dirty is False
