"""Tests for const.py helper functions."""

from __future__ import annotations

import time

from custom_components.roommind.const import (
    AWAY_PRIORITY_WEIGHT,
    DEFAULT_CENTRAL_PLANT_SETTINGS,
    DEFAULT_DEHUMIDIFY_RH_THRESHOLD,
    DEFAULT_DOOR_CLOSED_MIX_FACTOR,
    DEFAULT_FALLBACK_COOL_C,
    DEFAULT_FALLBACK_HEAT_C,
    DEFAULT_INTERIOR_DOOR_DELAY_S,
    DEFAULT_MAX_NUDGE_HEADROOM_C,
    DEFAULT_NUDGE_DELTA_C,
    DEFAULT_OCCUPANCY_DWELL_MINUTES,
    DEFAULT_WET_COIL_MAX_RH,
    EXTERIOR_OPEN_PAUSE_UNLESS_DOOR_CLOSED,
    EXTERIOR_OPEN_POLICIES,
    FAN_ENERGY_FRACTION,
    FREEZE_PROTECTION_TEMP_C,
    GUARDRAIL_PRIORITY_WEIGHT,
    MODE_CIRCULATING,
    OCCUPANCY_MODES,
    OVERRIDE_PRIORITY_WEIGHT,
    PLANT_MODES,
    PRICE_SHIFT_MODES,
    TARGET_REASON_PLANT_OVERRIDE,
    build_override_live,
    get_central_plant_settings,
    is_override_active,
    is_override_suppressed,
    is_plant_override_active,
    make_roommind_context,
)


class TestIsOverrideActive:
    """Unit tests for is_override_active."""

    def test_no_override_fields(self):
        """No override fields → inactive."""
        assert is_override_active({}) is False

    def test_both_targets_none(self):
        """override_heat=None and override_cool=None → inactive."""
        assert is_override_active({"override_heat": None, "override_cool": None}) is False

    def test_heat_only_active(self):
        """Only override_heat set → active."""
        assert is_override_active({"override_heat": 21.0, "override_cool": None}) is True

    def test_cool_only_active(self):
        """Only override_cool set → active."""
        assert is_override_active({"override_heat": None, "override_cool": 24.0}) is True

    def test_permanent_override(self):
        """Targets set, override_until=None → permanent, active."""
        assert is_override_active({"override_heat": 20.0, "override_cool": 24.0, "override_until": None}) is True

    def test_future_override(self):
        """override_until in the future → active."""
        future = time.time() + 3600
        assert is_override_active({"override_heat": 20.0, "override_cool": 24.0, "override_until": future}) is True

    def test_expired_override(self):
        """override_until in the past → inactive."""
        past = time.time() - 3600
        assert is_override_active({"override_heat": 20.0, "override_until": past}) is False


class TestIsOverrideSuppressed:
    """Unit tests for is_override_suppressed (#306)."""

    def test_presence_home_never_suppresses(self):
        assert is_override_suppressed({}, {"presence_clears_override": True}, False) is False

    def test_setting_disabled_never_suppresses(self):
        assert is_override_suppressed({}, {"presence_clears_override": False}, True) is False

    def test_setting_missing_defaults_off(self):
        assert is_override_suppressed({}, {}, True) is False

    def test_ignore_presence_room_never_suppressed(self):
        room = {"ignore_presence": True}
        assert is_override_suppressed(room, {"presence_clears_override": True}, True) is False

    def test_suppressed_when_all_conditions_met(self):
        assert is_override_suppressed({}, {"presence_clears_override": True}, True) is True


class TestBuildOverrideLive:
    """Unit tests for build_override_live (#306)."""

    def test_inactive_override(self):
        result = build_override_live({})
        assert result["override_active"] is False
        assert result["override_suppressed"] is False

    def test_active_override_not_suppressed(self):
        room = {"override_heat": 22.0, "override_cool": 25.0, "override_until": None, "override_type": "boost"}
        result = build_override_live(room, suppressed=False)
        assert result["override_active"] is True
        assert result["override_type"] == "boost"
        assert result["override_suppressed"] is False

    def test_active_override_suppressed(self):
        room = {"override_heat": 22.0, "override_cool": 25.0, "override_until": None, "override_type": "boost"}
        result = build_override_live(room, suppressed=True)
        assert result["override_active"] is True
        assert result["override_suppressed"] is True

    def test_inactive_override_never_suppressed(self):
        """Suppressed flag is meaningless when override is not active."""
        result = build_override_live({}, suppressed=True)
        assert result["override_active"] is False
        assert result["override_suppressed"] is False

    def test_exposes_split_targets(self):
        room = {"override_type": "custom", "override_heat": 21.0, "override_cool": 24.0, "override_until": None}
        live = build_override_live(room)
        assert live["override_active"] is True
        assert live["override_heat"] == 21.0
        assert live["override_cool"] == 24.0
        assert "override_temp" not in live

    def test_inactive_nulls_targets(self):
        live = build_override_live({"override_heat": None, "override_cool": None})
        assert live["override_active"] is False
        assert live["override_heat"] is None
        assert live["override_cool"] is None


class TestCentralPlantSettings:
    """Unit tests for get_central_plant_settings."""

    def test_missing_central_plant_returns_defaults(self):
        assert get_central_plant_settings({}) == DEFAULT_CENTRAL_PLANT_SETTINGS

    def test_returns_a_fresh_dict(self):
        resolved = get_central_plant_settings({})
        assert resolved is not DEFAULT_CENTRAL_PLANT_SETTINGS
        resolved["thermostat_entity"] = "climate.mutated"
        assert DEFAULT_CENTRAL_PLANT_SETTINGS["thermostat_entity"] == ""

    def test_merges_overrides_over_defaults(self):
        resolved = get_central_plant_settings(
            {"central_plant": {"thermostat_entity": "climate.central", "nudge_delta_c": 2.0}}
        )
        assert resolved["thermostat_entity"] == "climate.central"
        assert resolved["nudge_delta_c"] == 2.0
        assert resolved["changeover_minutes"] == DEFAULT_CENTRAL_PLANT_SETTINGS["changeover_minutes"]

    def test_ignores_a_non_dict_central_plant(self):
        assert get_central_plant_settings({"central_plant": "broken"}) == DEFAULT_CENTRAL_PLANT_SETTINGS

    def test_does_not_mutate_the_input(self):
        settings = {"central_plant": {"nudge_delta_c": 2.0}}
        get_central_plant_settings(settings)
        assert settings == {"central_plant": {"nudge_delta_c": 2.0}}

    def test_keeps_unknown_keys(self):
        resolved = get_central_plant_settings({"central_plant": {"future_key": 1}})
        assert resolved["future_key"] == 1


class TestIsPlantOverrideActive:
    """Unit tests for is_plant_override_active."""

    def test_no_override_fields(self):
        assert is_plant_override_active({}) is False

    def test_both_targets_none(self):
        assert is_plant_override_active({"plant_override_heat": None, "plant_override_cool": None}) is False

    def test_permanent_heat_override(self):
        assert is_plant_override_active({"plant_override_heat": 20.0, "plant_override_until": None}) is True

    def test_future_cool_override(self):
        future = time.time() + 3600
        assert is_plant_override_active({"plant_override_cool": 26.0, "plant_override_until": future}) is True

    def test_expired_override(self):
        past = time.time() - 3600
        assert is_plant_override_active({"plant_override_heat": 20.0, "plant_override_until": past}) is False

    def test_accepts_a_resolved_central_plant_dict(self):
        resolved = get_central_plant_settings({"central_plant": {"plant_override_heat": 20.0}})
        assert is_plant_override_active(resolved) is True

    def test_reads_the_nested_central_plant(self):
        assert is_plant_override_active({"central_plant": {"plant_override_cool": 26.0}}) is True

    def test_falls_through_to_the_nested_central_plant(self):
        settings = {
            "plant_override_heat": None,
            "plant_override_cool": None,
            "central_plant": {"plant_override_cool": 26.0},
        }
        assert is_plant_override_active(settings) is True

    def test_defaults_contain_no_override(self):
        assert is_plant_override_active(get_central_plant_settings({})) is False


class TestCentralHvacConstants:
    """Sanity checks on the Central HVAC constant block."""

    def test_circulating_mode(self):
        assert MODE_CIRCULATING == "circulating"

    def test_plant_modes_cover_every_action(self):
        assert PLANT_MODES == ["idle", "heating", "cooling", "circulating"]

    def test_target_reason_plant_override(self):
        assert TARGET_REASON_PLANT_OVERRIDE == "plant_override"

    def test_occupancy_modes_are_unique(self):
        assert len(OCCUPANCY_MODES) == len(set(OCCUPANCY_MODES)) == 4

    def test_exterior_open_policies(self):
        assert EXTERIOR_OPEN_POLICIES == ["pause_unless_door_closed", "pause_always", "exclude_room_only"]

    def test_price_shift_modes(self):
        assert PRICE_SHIFT_MODES == ["cool_only", "both"]

    def test_default_plant_settings_are_consistent(self):
        defaults = DEFAULT_CENTRAL_PLANT_SETTINGS
        assert defaults["thermostat_entity"] == ""
        assert defaults["thermostat_area_id"] == ""
        assert defaults["follow_thermostat_setpoint"] is True
        assert defaults["exterior_open_policy"] in EXTERIOR_OPEN_POLICIES
        assert defaults["price_shift_modes"] in PRICE_SHIFT_MODES
        assert defaults["fallback_heat_c"] < defaults["fallback_cool_c"]

    def test_central_thermostat_defaults_are_positive(self):
        assert DEFAULT_NUDGE_DELTA_C > 0
        assert DEFAULT_MAX_NUDGE_HEADROOM_C >= DEFAULT_NUDGE_DELTA_C
        assert DEFAULT_FALLBACK_HEAT_C < DEFAULT_FALLBACK_COOL_C

    def test_room_physics_defaults_are_positive(self):
        assert DEFAULT_OCCUPANCY_DWELL_MINUTES > 0
        assert DEFAULT_INTERIOR_DOOR_DELAY_S > 0
        assert 0 < DEFAULT_DOOR_CLOSED_MIX_FACTOR <= 1
        assert 0 < FAN_ENERGY_FRACTION < 1

    def test_priority_weights_are_ordered(self):
        assert OVERRIDE_PRIORITY_WEIGHT > AWAY_PRIORITY_WEIGHT > GUARDRAIL_PRIORITY_WEIGHT > 0

    def test_freeze_protection_sits_below_the_fallback_band(self):
        assert FREEZE_PROTECTION_TEMP_C < DEFAULT_FALLBACK_HEAT_C

    def test_wet_coil_lockout_precedes_dehumidification(self):
        assert DEFAULT_WET_COIL_MAX_RH < DEFAULT_DEHUMIDIFY_RH_THRESHOLD

    def test_exterior_open_default_policy(self):
        assert EXTERIOR_OPEN_PAUSE_UNLESS_DOOR_CLOSED in EXTERIOR_OPEN_POLICIES


class TestMakeRoommindContext:
    """Unit tests for make_roommind_context."""

    def test_tags_the_context(self):
        context = make_roommind_context()
        assert context.parent_id == "roommind"
