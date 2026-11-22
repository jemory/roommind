"""Tests for the Central Plant guards, hardware observation and exterior gate."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from custom_components.roommind.const import MODE_COOLING, MODE_HEATING, MODE_IDLE
from custom_components.roommind.managers.central_plant_manager import (
    MODE_CIRCULATING,
    CentralPlantManager,
    ExteriorGateResult,
    PlantGuardState,
)


def _state(state: str = "heat", **attributes: Any) -> SimpleNamespace:
    return SimpleNamespace(state=state, attributes=attributes)


def _manager(
    *,
    active_action: str = MODE_IDLE,
    started_at: float | None = None,
    last_heat: float | None = None,
    last_cool: float | None = None,
) -> CentralPlantManager:
    manager = CentralPlantManager()
    manager.active_action = active_action
    manager.action_started_at = started_at
    manager.last_heat_ended_at = last_heat
    manager.last_cool_ended_at = last_cool
    return manager


ROOMS: dict[str, dict[str, Any]] = {
    "living": {"climate_control_enabled": True, "is_outdoor": False},
    "kitchen": {"climate_control_enabled": True, "is_outdoor": False},
    "office": {"climate_control_enabled": True, "is_outdoor": False},
    "off_room": {"climate_control_enabled": False, "is_outdoor": False},
    "terrace": {"climate_control_enabled": True, "is_outdoor": True},
}


# --------------------------------------------------------------------------- #
# observe_plant_action
# --------------------------------------------------------------------------- #


class TestObservePlantAction:
    def test_no_state_is_unknown(self) -> None:
        assert CentralPlantManager().observe_plant_action(None, {}, MODE_HEATING) == (None, False)

    @pytest.mark.parametrize("state", ["unavailable", "unknown"])
    def test_unavailable_state_is_unknown(self, state: str) -> None:
        mgr = CentralPlantManager()
        assert mgr.observe_plant_action(_state(state, hvac_action="heating"), {}, MODE_HEATING) == (None, False)

    def test_reported_heating(self) -> None:
        mgr = CentralPlantManager()
        assert mgr.observe_plant_action(_state("heat", hvac_action="heating"), {}, MODE_IDLE) == (MODE_HEATING, True)

    def test_reported_cooling(self) -> None:
        mgr = CentralPlantManager()
        assert mgr.observe_plant_action(_state("cool", hvac_action="cooling"), {}, MODE_IDLE) == (MODE_COOLING, True)

    @pytest.mark.parametrize(
        ("state", "attributes"),
        [
            ("heat", {"hvac_action": "fan"}),
            ("fan_only", {}),
            ("fan_only", {"hvac_action": "idle"}),
        ],
    )
    def test_fan_reports_circulation(self, state: str, attributes: dict[str, Any]) -> None:
        mgr = CentralPlantManager()
        assert mgr.observe_plant_action(SimpleNamespace(state=state, attributes=attributes), {}, MODE_IDLE) == (
            MODE_CIRCULATING,
            True,
        )

    def test_idle_with_circulation_fan(self) -> None:
        mgr = CentralPlantManager()
        state = _state("heat_cool", hvac_action="idle", fan_mode="on")
        assert mgr.observe_plant_action(state, {"circulate_fan_mode": "on"}, MODE_IDLE) == (MODE_CIRCULATING, True)

    def test_custom_circulation_fan_name(self) -> None:
        mgr = CentralPlantManager()
        state = _state("heat_cool", hvac_action="idle", fan_mode="high")
        assert mgr.observe_plant_action(state, {"circulate_fan_mode": "high"}, MODE_IDLE) == (MODE_CIRCULATING, True)

    def test_idle_with_fan_on_while_off_is_idle(self) -> None:
        mgr = CentralPlantManager()
        state = _state("off", hvac_action="off", fan_mode="on")
        assert mgr.observe_plant_action(state, {}, MODE_IDLE) == (MODE_IDLE, False)

    def test_idle_with_default_fan_is_idle(self) -> None:
        mgr = CentralPlantManager()
        state = _state("heat_cool", hvac_action="idle", fan_mode="auto")
        assert mgr.observe_plant_action(state, {}, MODE_IDLE) == (MODE_IDLE, False)

    @pytest.mark.parametrize("action", ["preheating", "defrosting", "drying"])
    def test_transient_equipment_states_are_skipped(self, action: str) -> None:
        mgr = CentralPlantManager()
        assert mgr.observe_plant_action(_state("heat", hvac_action=action), {}, MODE_HEATING) == (None, False)

    @pytest.mark.parametrize(
        ("commanded", "expected"),
        [(MODE_HEATING, (MODE_HEATING, True)), (MODE_COOLING, (MODE_COOLING, True)), (MODE_IDLE, (MODE_IDLE, False))],
    )
    def test_missing_hvac_action_falls_back_to_commanded(self, commanded: str, expected: tuple[str, bool]) -> None:
        mgr = CentralPlantManager()
        assert mgr.observe_plant_action(_state("heat_cool"), {}, commanded) == expected

    def test_unknown_hvac_action_falls_back_to_commanded(self) -> None:
        mgr = CentralPlantManager()
        state = _state("heat", hvac_action="something_new")
        assert mgr.observe_plant_action(state, {}, MODE_CIRCULATING) == (MODE_CIRCULATING, True)

    def test_non_dict_attributes_are_tolerated(self) -> None:
        mgr = CentralPlantManager()
        state = SimpleNamespace(state="heat", attributes=None)
        assert mgr.observe_plant_action(state, {}, MODE_HEATING) == (MODE_HEATING, True)


# --------------------------------------------------------------------------- #
# evaluate_exterior_gate
# --------------------------------------------------------------------------- #


class TestExteriorGate:
    def test_open_windows_pause_unless_door_is_closed(self) -> None:
        mgr = CentralPlantManager()
        result = mgr.evaluate_exterior_gate(
            ROOMS,
            window_states={"living": True, "kitchen": True, "office": True, "off_room": True, "terrace": True},
            door_open_states={"office": False, "living": True},
            rooms_with_door_sensors={"office", "living"},
            plant_settings={},
        )
        assert result == ExteriorGateResult(True, ["kitchen", "living"])

    def test_pause_always_ignores_closed_interior_doors(self) -> None:
        mgr = CentralPlantManager()
        result = mgr.evaluate_exterior_gate(
            ROOMS,
            window_states={"office": True},
            door_open_states={"office": False},
            rooms_with_door_sensors={"office"},
            plant_settings={"exterior_open_policy": "pause_always"},
        )
        assert result == ExteriorGateResult(True, ["office"])

    def test_exclude_room_only_never_pauses(self) -> None:
        mgr = CentralPlantManager()
        result = mgr.evaluate_exterior_gate(
            ROOMS,
            window_states={"living": True, "kitchen": True},
            door_open_states={},
            rooms_with_door_sensors=set(),
            plant_settings={"exterior_open_policy": "exclude_room_only"},
        )
        assert result == ExteriorGateResult(False, [])

    def test_any_override_keeps_the_plant_running(self) -> None:
        mgr = CentralPlantManager()
        result = mgr.evaluate_exterior_gate(
            ROOMS,
            window_states={"living": True},
            door_open_states={},
            rooms_with_door_sensors=set(),
            plant_settings={},
            any_override_active=True,
        )
        assert result == ExteriorGateResult(False, [])

    def test_door_sensor_without_state_counts_as_open(self) -> None:
        mgr = CentralPlantManager()
        result = mgr.evaluate_exterior_gate(
            ROOMS,
            window_states={"office": True},
            door_open_states={},
            rooms_with_door_sensors={"office"},
            plant_settings={},
        )
        assert result == ExteriorGateResult(True, ["office"])

    def test_door_open_and_window_open_pauses(self) -> None:
        mgr = CentralPlantManager()
        result = mgr.evaluate_exterior_gate(
            ROOMS,
            window_states={"office": True},
            door_open_states={"office": True},
            rooms_with_door_sensors={"office"},
            plant_settings={},
        )
        assert result == ExteriorGateResult(True, ["office"])

    def test_no_open_windows_means_no_pause(self) -> None:
        mgr = CentralPlantManager()
        result = mgr.evaluate_exterior_gate(ROOMS, {}, {}, set(), {})
        assert result == ExteriorGateResult(False, [])

    def test_malformed_room_entry_is_ignored(self) -> None:
        mgr = CentralPlantManager()
        rooms: dict[str, dict[str, Any]] = {"broken": None}  # type: ignore[dict-item]
        result = mgr.evaluate_exterior_gate(
            rooms,
            window_states={"broken": True},
            door_open_states={},
            rooms_with_door_sensors=set(),
            plant_settings={},
        )
        assert result == ExteriorGateResult(False, [])


# --------------------------------------------------------------------------- #
# allowed_actions
# --------------------------------------------------------------------------- #


class TestAllowedActions:
    def test_force_off_only_allows_idle(self) -> None:
        state = CentralPlantManager().allowed_actions({}, "auto", None, {}, force_off=True, now_ts=1000)
        assert state == PlantGuardState({MODE_IDLE}, MODE_IDLE, False, None, None)

    def test_global_control_off_only_allows_idle(self) -> None:
        state = CentralPlantManager().allowed_actions({}, "auto", None, {"climate_control_active": False}, now_ts=1000)
        assert state == PlantGuardState({MODE_IDLE}, MODE_IDLE, False, None, None)

    def test_exterior_pause_keeps_conditioning_out_but_allows_circulation(self) -> None:
        state = CentralPlantManager().allowed_actions({}, "auto", None, {}, exterior_paused=True, now_ts=1000)
        assert state == PlantGuardState({MODE_IDLE, MODE_CIRCULATING}, None, True, "exterior_open", None)

    def test_a_wet_coil_keeps_circulation_out_of_an_exterior_pause(self) -> None:
        state = CentralPlantManager().allowed_actions(
            {}, "auto", None, {}, exterior_paused=True, wet_coil_locked=True, now_ts=1000
        )
        assert state == PlantGuardState({MODE_IDLE}, None, True, "exterior_open", None)

    def test_auto_mode_supports_all_actions(self) -> None:
        state = CentralPlantManager().allowed_actions({}, "auto", None, {}, now_ts=1000)
        assert state.legal_actions == {MODE_IDLE, MODE_HEATING, MODE_COOLING, MODE_CIRCULATING}
        assert state.forced_action is None
        assert state.guard_active is False
        assert state.guard_reason is None
        assert state.guard_until is None

    def test_heat_only_cannot_cool(self) -> None:
        state = CentralPlantManager().allowed_actions({}, "heat_only", None, {}, now_ts=1000)
        assert state.legal_actions == {MODE_IDLE, MODE_HEATING, MODE_CIRCULATING}

    def test_cool_only_cannot_heat(self) -> None:
        state = CentralPlantManager().allowed_actions({}, "cool_only", None, {}, now_ts=1000)
        assert state.legal_actions == {MODE_IDLE, MODE_COOLING, MODE_CIRCULATING}

    def test_cold_outdoors_block_cooling(self) -> None:
        state = CentralPlantManager().allowed_actions({}, "auto", 10.0, {}, now_ts=1000)
        assert MODE_COOLING not in state.legal_actions
        assert MODE_HEATING in state.legal_actions

    def test_warm_outdoors_block_heating(self) -> None:
        state = CentralPlantManager().allowed_actions({}, "auto", 25.0, {}, now_ts=1000)
        assert MODE_HEATING not in state.legal_actions
        assert MODE_COOLING in state.legal_actions

    def test_override_lifts_outdoor_gates(self) -> None:
        state = CentralPlantManager().allowed_actions({}, "auto", 25.0, {}, any_override=True, now_ts=1000)
        assert MODE_HEATING in state.legal_actions

    def test_custom_outdoor_limits(self) -> None:
        settings = {"outdoor_cooling_min": 18, "outdoor_heating_max": 24}
        cold = CentralPlantManager().allowed_actions(settings, "auto", 17.0, {}, now_ts=1000)
        assert MODE_COOLING not in cold.legal_actions
        warm = CentralPlantManager().allowed_actions(settings, "auto", 25.0, {}, now_ts=1000)
        assert MODE_HEATING not in warm.legal_actions

    def test_the_global_settings_supply_the_outdoor_limits(self) -> None:
        """The panel saves the limits in the global settings, not the plant block.

        Reading only the plant block pinned both to their defaults, so a Caller who
        lowered the cooling floor still had cooling withheld above it.
        """
        state = CentralPlantManager().allowed_actions({}, "auto", 12.0, {"outdoor_cooling_min": 10.0}, now_ts=1000)
        assert MODE_COOLING in state.legal_actions
        assert state.outdoor_blocked == ()

    def test_the_outdoor_lockout_reports_the_direction_it_withheld(self) -> None:
        state = CentralPlantManager().allowed_actions({}, "auto", 10.0, {}, now_ts=1000)
        assert state.outdoor_blocked == (MODE_COOLING,)
        # A standing weather limit, not an equipment protection: the Guard chip
        # keeps meaning "the plant is waiting something out".
        assert state.guard_active is False
        assert state.guard_reason is None

    def test_a_mode_restriction_is_not_blamed_on_the_outdoor_lockout(self) -> None:
        state = CentralPlantManager().allowed_actions({}, "heat_only", 10.0, {}, now_ts=1000)
        assert MODE_COOLING not in state.legal_actions
        assert state.outdoor_blocked == ()

    def test_an_override_lifts_the_outdoor_lockout_report(self) -> None:
        state = CentralPlantManager().allowed_actions({}, "auto", 10.0, {}, any_override=True, now_ts=1000)
        assert MODE_COOLING in state.legal_actions
        assert state.outdoor_blocked == ()

    def test_wet_coil_lockout_drops_circulation(self) -> None:
        state = CentralPlantManager().allowed_actions({}, "auto", None, {}, wet_coil_locked=True, now_ts=1000)
        assert MODE_CIRCULATING not in state.legal_actions
        assert state.guard_active is True
        assert state.guard_reason == "wet_coil"

    # -- min_run -------------------------------------------------------

    def test_min_run_holds_an_ongoing_heating_run(self) -> None:
        mgr = _manager(active_action=MODE_HEATING, started_at=1000.0)
        state = mgr.allowed_actions({"min_run_minutes": 10}, "auto", None, {}, now_ts=1300.0)
        assert state == PlantGuardState({MODE_HEATING}, MODE_HEATING, True, "min_run", 1600.0)

    def test_min_run_expires(self) -> None:
        mgr = _manager(active_action=MODE_HEATING, started_at=1000.0)
        state = mgr.allowed_actions({"min_run_minutes": 10}, "auto", None, {}, now_ts=1600.0)
        # Still heating, so the changeover guard keeps cooling blocked.
        assert state.legal_actions == {MODE_IDLE, MODE_HEATING, MODE_CIRCULATING}
        assert state.guard_reason is None

    def test_min_run_does_not_hold_an_action_outside_legal_actions(self) -> None:
        mgr = _manager(active_action=MODE_HEATING, started_at=1000.0)
        state = mgr.allowed_actions({}, "cool_only", None, {}, now_ts=1100.0)
        assert MODE_HEATING not in state.legal_actions
        assert state.legal_actions == {MODE_IDLE, MODE_CIRCULATING}
        assert state.guard_active is False

    def test_corrupt_min_run_setting_uses_default(self) -> None:
        mgr = _manager(active_action=MODE_HEATING, started_at=1000.0)
        state = mgr.allowed_actions({"min_run_minutes": "not-a-number"}, "auto", None, {}, now_ts=1100.0)
        assert state.guard_reason == "min_run"
        assert state.guard_until == 1600.0

    # -- min_fan -------------------------------------------------------

    def test_min_fan_blocks_idle_but_allows_upgrade(self) -> None:
        mgr = _manager(active_action=MODE_CIRCULATING, started_at=1000.0)
        state = mgr.allowed_actions({"min_fan_minutes": 10}, "auto", None, {}, now_ts=1300.0)
        assert state == PlantGuardState({MODE_CIRCULATING, MODE_HEATING, MODE_COOLING}, None, True, "min_fan", 1600.0)

    def test_min_fan_without_an_upgrade_forces_circulation(self) -> None:
        mgr = _manager(active_action=MODE_CIRCULATING, started_at=1000.0)
        settings = {"outdoor_cooling_min": 30, "outdoor_heating_max": 10}
        state = mgr.allowed_actions(settings, "auto", 20.0, {}, now_ts=1300.0)
        assert state == PlantGuardState({MODE_CIRCULATING}, MODE_CIRCULATING, True, "min_fan", 1600.0)

    def test_min_fan_only_allows_heating_in_a_heat_only_home(self) -> None:
        mgr = _manager(active_action=MODE_CIRCULATING, started_at=1000.0)
        state = mgr.allowed_actions({}, "heat_only", None, {}, now_ts=1300.0)
        assert state.legal_actions == {MODE_CIRCULATING, MODE_HEATING}
        assert state.forced_action is None

    def test_min_fan_expires(self) -> None:
        mgr = _manager(active_action=MODE_CIRCULATING, started_at=1000.0)
        state = mgr.allowed_actions({"min_fan_minutes": 10}, "auto", None, {}, now_ts=1600.0)
        assert state.guard_active is False
        assert MODE_IDLE in state.legal_actions

    def test_wet_coil_supersedes_the_min_fan_hold(self) -> None:
        mgr = _manager(active_action=MODE_CIRCULATING, started_at=1000.0)
        state = mgr.allowed_actions({}, "auto", None, {}, wet_coil_locked=True, now_ts=1100.0)
        assert MODE_CIRCULATING not in state.legal_actions
        assert state.legal_actions == {MODE_IDLE, MODE_HEATING, MODE_COOLING}
        assert state.guard_reason == "wet_coil"

    # -- min_off -------------------------------------------------------

    def test_min_off_blocks_a_restart(self) -> None:
        mgr = _manager(last_heat=1000.0)
        state = mgr.allowed_actions({"min_off_minutes": 5}, "auto", None, {}, now_ts=1100.0)
        assert state == PlantGuardState({MODE_IDLE, MODE_CIRCULATING}, None, True, "min_off", 1300.0)

    def test_min_off_expires(self) -> None:
        mgr = _manager(last_heat=1000.0)
        state = mgr.allowed_actions({"min_off_minutes": 5, "changeover_minutes": 0}, "auto", None, {}, now_ts=1300.0)
        assert MODE_HEATING in state.legal_actions
        assert MODE_COOLING in state.legal_actions
        assert state.guard_reason is None

    def test_min_off_uses_the_latest_end(self) -> None:
        mgr = _manager(last_heat=1000.0, last_cool=1200.0)
        state = mgr.allowed_actions({"min_off_minutes": 5}, "auto", None, {}, now_ts=1300.0)
        assert state.guard_reason == "min_off"
        assert state.guard_until == 1500.0

    def test_min_off_is_not_applied_during_an_active_run(self) -> None:
        mgr = _manager(active_action=MODE_HEATING, started_at=0.0, last_heat=1000.0)
        state = mgr.allowed_actions({"min_off_minutes": 5, "changeover_minutes": 0}, "auto", None, {}, now_ts=1100.0)
        assert MODE_HEATING in state.legal_actions
        assert state.guard_reason is None

    # -- changeover ----------------------------------------------------

    def test_changeover_after_heating_blocks_cooling(self) -> None:
        mgr = _manager(last_heat=1000.0)
        state = mgr.allowed_actions({"min_off_minutes": 0, "changeover_minutes": 30}, "auto", None, {}, now_ts=1100.0)
        assert MODE_COOLING not in state.legal_actions
        assert state == PlantGuardState({MODE_IDLE, MODE_HEATING, MODE_CIRCULATING}, None, True, "changeover", 2800.0)

    def test_changeover_after_cooling_blocks_heating(self) -> None:
        mgr = _manager(last_cool=1000.0)
        state = mgr.allowed_actions({"min_off_minutes": 0, "changeover_minutes": 30}, "auto", None, {}, now_ts=1100.0)
        assert MODE_HEATING not in state.legal_actions
        assert state.guard_reason == "changeover"
        assert state.guard_until == 2800.0

    def test_changeover_does_not_override_min_off(self) -> None:
        mgr = _manager(last_heat=1000.0)
        state = mgr.allowed_actions({"min_off_minutes": 5, "changeover_minutes": 30}, "auto", None, {}, now_ts=1100.0)
        assert state.guard_reason == "min_off"
        assert state.guard_until == 1300.0

    def test_active_heating_will_not_flip_to_cooling(self) -> None:
        mgr = _manager(active_action=MODE_HEATING, started_at=0.0)
        state = mgr.allowed_actions({"min_run_minutes": 1}, "auto", None, {}, now_ts=1000.0)
        assert MODE_COOLING not in state.legal_actions
        assert state.guard_active is False

    def test_active_cooling_will_not_flip_to_heating(self) -> None:
        mgr = _manager(active_action=MODE_COOLING, started_at=0.0)
        state = mgr.allowed_actions({"min_run_minutes": 1}, "auto", None, {}, now_ts=1000.0)
        assert MODE_HEATING not in state.legal_actions
        assert state.guard_active is False

    def test_both_changeover_timers_restrict_both_sides(self) -> None:
        mgr = _manager(last_heat=1000.0, last_cool=1100.0)
        state = mgr.allowed_actions({"min_off_minutes": 0, "changeover_minutes": 30}, "auto", None, {}, now_ts=1200.0)
        assert state.legal_actions == {MODE_IDLE, MODE_CIRCULATING}
        assert state.guard_reason == "changeover"
        assert state.guard_until == 2800.0


# --------------------------------------------------------------------------- #
# record_commanded_action
# --------------------------------------------------------------------------- #


class TestRecordCommandedAction:
    def test_first_idle_command_starts_the_clock(self) -> None:
        mgr = CentralPlantManager()
        mgr.record_commanded_action(MODE_IDLE, now_ts=1000.0)
        assert mgr.active_action == MODE_IDLE
        assert mgr.action_started_at == 1000.0
        assert mgr.state_dirty is True
        assert mgr.last_heat_ended_at is None
        assert mgr.last_cool_ended_at is None

    def test_repeating_the_same_action_keeps_the_run_start(self) -> None:
        mgr = _manager(active_action=MODE_HEATING, started_at=1000.0)
        mgr.state_dirty = False
        mgr.record_commanded_action(MODE_HEATING, now_ts=1300.0)
        assert mgr.action_started_at == 1000.0
        assert mgr.state_dirty is False

    def test_same_action_without_a_start_stamps_one(self) -> None:
        mgr = _manager(active_action=MODE_HEATING, started_at=None)
        mgr.record_commanded_action(MODE_HEATING, now_ts=1300.0)
        assert mgr.action_started_at == 1300.0
        assert mgr.state_dirty is True

    def test_leaving_heating_stamps_its_end(self) -> None:
        mgr = _manager(active_action=MODE_HEATING, started_at=1000.0)
        mgr.record_commanded_action(MODE_IDLE, now_ts=1200.0)
        assert mgr.active_action == MODE_IDLE
        assert mgr.action_started_at == 1200.0
        assert mgr.last_heat_ended_at == 1200.0
        assert mgr.last_cool_ended_at is None

    def test_leaving_cooling_stamps_its_end(self) -> None:
        mgr = _manager(active_action=MODE_COOLING, started_at=1000.0)
        mgr.record_commanded_action(MODE_HEATING, now_ts=1200.0)
        assert mgr.active_action == MODE_HEATING
        assert mgr.last_cool_ended_at == 1200.0
        assert mgr.last_heat_ended_at is None

    def test_entering_heating_from_idle_stamps_no_end(self) -> None:
        mgr = CentralPlantManager()
        mgr.record_commanded_action(MODE_HEATING, now_ts=1200.0)
        assert mgr.action_started_at == 1200.0
        assert mgr.last_heat_ended_at is None


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_round_trip(self) -> None:
        mgr = _manager(
            active_action=MODE_COOLING,
            started_at=1000.0,
            last_heat=900.0,
            last_cool=800.0,
        )
        mgr.state_dirty = True

        restored = CentralPlantManager()
        restored.load_state(mgr.to_dict())

        assert restored.active_action == MODE_COOLING
        assert restored.action_started_at == 1000.0
        assert restored.last_heat_ended_at == 900.0
        assert restored.last_cool_ended_at == 800.0
        assert restored.state_dirty is True

    @pytest.mark.parametrize("data", [None, {}, {"active_action": "bogus"}])
    def test_missing_or_unknown_action_resets_to_idle(self, data: dict[str, Any] | None) -> None:
        mgr = _manager(active_action=MODE_HEATING, started_at=1.0, last_heat=2.0)
        mgr.load_state(data)
        assert mgr.active_action == MODE_IDLE
        assert mgr.action_started_at is None
        assert mgr.last_heat_ended_at is None
        assert mgr.last_cool_ended_at is None
        assert mgr.state_dirty is False

    def test_malformed_timestamps_are_dropped(self) -> None:
        mgr = CentralPlantManager()
        mgr.load_state(
            {
                "active_action": MODE_HEATING,
                "action_started_at": "yesterday",
                "last_heat_ended_at": [],
                "last_cool_ended_at": None,
                "state_dirty": "yes",
            }
        )
        assert mgr.active_action == MODE_HEATING
        assert mgr.action_started_at is None
        assert mgr.last_heat_ended_at is None
        assert mgr.last_cool_ended_at is None
        assert mgr.state_dirty is True

    def test_round_trip_with_none_timestamps(self) -> None:
        mgr = CentralPlantManager()
        restored = CentralPlantManager()
        restored.load_state(mgr.to_dict())
        assert restored.to_dict() == {
            "active_action": MODE_IDLE,
            "action_started_at": None,
            "last_heat_ended_at": None,
            "last_cool_ended_at": None,
            "state_dirty": False,
        }
