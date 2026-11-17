"""Tests for the joint whole-home MPC, cold-start controller and controller facade."""

from __future__ import annotations

import pytest

from custom_components.roommind.const import (
    BANGBANG_COOL_HYSTERESIS,
    BANGBANG_HEAT_HYSTERESIS,
    MODE_CIRCULATING,
    MODE_COOLING,
    MODE_HEATING,
    MODE_IDLE,
)
from custom_components.roommind.control.central_mpc import (
    ARRIVAL_DEADLINE_MARGIN_C,
    DEFAULT_GUARDRAIL_COOL,
    DEFAULT_GUARDRAIL_HEAT,
    DEFAULT_HOUSE_TEMP_C,
    FAN_ENERGY_FRACTION,
    GUARDRAIL_WEIGHT,
    LOOKAHEAD_BLOCKS,
    MAX_HORIZON_BLOCKS,
    MIN_HORIZON_BLOCKS,
    MPC_MAX_PREDICTION_STD,
    MPC_MIN_ACTIVE_UPDATES,
    MPC_MIN_IDLE_UPDATES,
    OPPOSITE_BOUND_TOLERANCE_C,
    PLAN_DT_MINUTES,
    PREHEAT_MAX_LEAD_HOURS,
    START_HYSTERESIS_C,
    WRONG_WAY_PENALTY,
    CentralBangBang,
    CentralMPCController,
    CentralMPCOptimizer,
    CentralMPCPlan,
    RoomHorizonInput,
    _blocks_from_minutes,
    _optional_at,
    _PlantRunState,
    _series_at,
    bounded_dehumidified_cool_target,
    compute_house_temperature,
    compute_room_reference_temperatures,
    driving_rooms,
    guardrail_breach_direction,
    is_room_mpc_ready,
    preheat_max_blocks,
    start_discomfort_reached,
    step_house_block,
)
from custom_components.roommind.control.thermal_model import RCModel, RoomModelManager, ThermalEKF

N_BLOCKS = 32
LONG_AGO = 1_000_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_room(
    area_id: str = "room",
    T_init: float = 20.0,
    *,
    heat: float | None = 21.0,
    cool: float | None = 23.0,
    priority: float = 1.0,
    occupied: float = 1.0,
    rh: float | None = None,
    model: RCModel | None = None,
    blocks: int = N_BLOCKS,
    **overrides: object,
) -> RoomHorizonInput:
    """A room whose priority, occupancy and dual targets are flat over the horizon."""
    heat_series: list[float | None] = [heat] * blocks if heat is not None else []
    cool_series: list[float | None] = [cool] * blocks if cool is not None else []
    room = RoomHorizonInput(
        area_id=area_id,
        model=model or RCModel(),
        T_init=T_init,
        rh_init=rh,
        priority_series=[priority] * blocks,
        occupied_prob_series=[occupied] * blocks,
        comfort_heat_series=heat_series,
        comfort_cool_series=cool_series,
    )
    for name, value in overrides.items():
        setattr(room, name, value)
    return room


def arrival_room(
    *,
    area_id: str = "office",
    T_init: float = 19.0,
    jump_block: int = 9,
    heat_before: float = 20.0,
    heat_after: float = 21.0,
    priority_before: float = 0.1,
    priority_after: float = 0.85,
    occupied_before: float = 0.1,
    occupied_after: float = 0.85,
    model: RCModel | None = None,
    blocks: int = N_BLOCKS,
) -> RoomHorizonInput:
    """An office whose occupancy- and target-step arrives at ``jump_block``."""
    heat_series: list[float | None] = []
    heat_series.extend([heat_before] * jump_block)
    heat_series.extend([heat_after] * (blocks - jump_block))
    cool_series: list[float | None] = []
    cool_series.extend([23.0] * blocks)
    return RoomHorizonInput(
        area_id=area_id,
        model=model or RCModel(Q_heat=4.4, U_house=0.1),
        T_init=T_init,
        priority_series=[priority_before] * jump_block + [priority_after] * (blocks - jump_block),
        occupied_prob_series=[occupied_before] * jump_block + [occupied_after] * (blocks - jump_block),
        comfort_heat_series=heat_series,
        comfort_cool_series=cool_series,
    )


def _optional_series(first: float, first_blocks: int, rest: float, blocks: int = N_BLOCKS) -> list[float | None]:
    """A dual-target series that steps from *first* to *rest* after *first_blocks*."""
    series: list[float | None] = []
    series.extend([first] * first_blocks)
    series.extend([rest] * (blocks - first_blocks))
    return series


def ready_estimator(
    *,
    idle: int = 45,
    heating: int = 20,
    cooling: int = 0,
    circulating: int = 0,
) -> ThermalEKF:
    """A ThermalEKF fed physically consistent idle/active samples."""
    estimator = ThermalEKF()
    model = RCModel()
    T = 20.0
    for _ in range(idle):
        T = model.predict(T, 10.0, 0.0, PLAN_DT_MINUTES, mode=MODE_IDLE)
        estimator.update(T, 10.0, MODE_IDLE, PLAN_DT_MINUTES)
    for _ in range(heating):
        T = model.predict(T, 10.0, model.Q_heat, PLAN_DT_MINUTES, mode=MODE_HEATING)
        estimator.update(T, 10.0, MODE_HEATING, PLAN_DT_MINUTES)
    for _ in range(cooling):
        T = model.predict(T, 25.0, -model.Q_cool, PLAN_DT_MINUTES, mode=MODE_COOLING)
        estimator.update(T, 25.0, MODE_COOLING, PLAN_DT_MINUTES)
    for _ in range(circulating):
        T = model.predict(T, 10.0, 0.0, PLAN_DT_MINUTES, mode=MODE_CIRCULATING, u_fan=1.0)
        estimator.update(T, 10.0, MODE_CIRCULATING, PLAN_DT_MINUTES)
    return estimator


def action_costs(
    optimizer: CentralMPCOptimizer,
    temperatures: dict[str, float],
    humidities: dict[str, float | None],
    charges: dict[str, float],
) -> dict[str, float]:
    return {
        action: optimizer._rollout_cost(action, 0, temperatures, humidities, charges)
        for action in (MODE_IDLE, MODE_CIRCULATING, MODE_HEATING, MODE_COOLING)
    }


# ---------------------------------------------------------------------------
# Constants and dataclasses
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """The published constants carry the specified values."""

    def test_plan_constants(self) -> None:
        assert PLAN_DT_MINUTES == 5.0
        assert LOOKAHEAD_BLOCKS == 8
        assert MIN_HORIZON_BLOCKS == 24
        assert MAX_HORIZON_BLOCKS == 144

    def test_weight_constants(self) -> None:
        assert FAN_ENERGY_FRACTION == 0.08
        assert GUARDRAIL_WEIGHT == 0.08
        assert WRONG_WAY_PENALTY == 2.0

    def test_preheat_lead(self) -> None:
        """The pre-heat lead follows the comfort slider and the room switch."""
        assert preheat_max_blocks(70) is None
        assert preheat_max_blocks(95) is None
        assert preheat_max_blocks(70, enabled=False) == 0
        assert preheat_max_blocks(0) == 0
        assert preheat_max_blocks(-20) == 0
        assert preheat_max_blocks(35) == int(PREHEAT_MAX_LEAD_HOURS * 35 / 70 * 60 / PLAN_DT_MINUTES)
        assert preheat_max_blocks(35, 10.0) == int(PREHEAT_MAX_LEAD_HOURS * 35 / 70 * 60 / 10.0)

    def test_hysteresis_and_gates(self) -> None:
        assert START_HYSTERESIS_C == 0.15
        assert MPC_MIN_IDLE_UPDATES == 40
        assert MPC_MIN_ACTIVE_UPDATES == 12
        assert MPC_MAX_PREDICTION_STD == 0.50
        assert ARRIVAL_DEADLINE_MARGIN_C == 0.2
        assert OPPOSITE_BOUND_TOLERANCE_C == 0.5
        assert DEFAULT_GUARDRAIL_HEAT == 16.0
        assert DEFAULT_GUARDRAIL_COOL == 28.0
        assert DEFAULT_HOUSE_TEMP_C == 21.0


class TestRoomHorizonInput:
    """The per-room input dataclass defaults to a safe, neutral room."""

    def test_defaults(self) -> None:
        room = RoomHorizonInput(area_id="office", model=RCModel(), T_init=20.0)
        assert room.rh_init is None
        assert room.rebound_init == 0.0
        assert room.door_mix_factor == 1.0
        assert room.door_supply_factor == 1.0
        assert room.priority_series == []
        assert room.occupied_prob_series == []
        assert room.comfort_heat_series == []
        assert room.comfort_cool_series == []
        assert room.guardrail_heat == 16.0
        assert room.guardrail_cool == 28.0
        assert room.solar_series == []
        assert room.occupancy_gain_series == []
        assert room.preheat_enabled is True
        assert room.include_in_house_mean is True
        assert room.mpc_ready is True


class TestCentralMPCPlan:
    """The plan exposes the current action and defaults to an idle plan."""

    def test_current_action(self) -> None:
        plan = CentralMPCPlan(
            actions=[MODE_HEATING, MODE_IDLE],
            room_temperatures={"a": [20.0, 20.2, 20.1]},
            house_temperatures=[20.0, 20.1, 20.1],
        )
        assert plan.current_action() == MODE_HEATING
        assert plan.dt_minutes == PLAN_DT_MINUTES
        assert plan.preheat_info == {}

    def test_empty_plan_is_idle(self) -> None:
        plan = CentralMPCPlan(actions=[], room_temperatures={}, house_temperatures=[20.0])
        assert plan.current_action() == MODE_IDLE


# ---------------------------------------------------------------------------
# Series helpers
# ---------------------------------------------------------------------------


class TestSeriesHelpers:
    """Horizon series hold their last value and fall back when empty."""

    def test_series_at(self) -> None:
        assert _series_at([], 0, 7.5) == 7.5
        assert _series_at([1.0, 2.0, 3.0], 0, 0.0) == 1.0
        assert _series_at([1.0, 2.0, 3.0], 2, 0.0) == 3.0
        assert _series_at([1.0, 2.0, 3.0], 9, 0.0) == 3.0

    def test_optional_at(self) -> None:
        assert _optional_at([], 0, None) is None
        assert _optional_at([], 0, 21.0) == 21.0
        assert _optional_at([None, 20.0], 0, 99.0) is None
        assert _optional_at([None, 20.0], 1, 99.0) == 20.0
        assert _optional_at([None, 20.0], 5, 99.0) == 20.0

    def test_blocks_from_minutes(self) -> None:
        assert _blocks_from_minutes(10, 5.0, 10.0) == 2
        assert _blocks_from_minutes(0, 5.0, 10.0) == 1
        assert _blocks_from_minutes(-5, 5.0, 10.0) == 1
        assert _blocks_from_minutes("corrupt", 5.0, 15.0) == 3
        assert _blocks_from_minutes(None, 0.0, 10.0) == 1


# ---------------------------------------------------------------------------
# Coupled multi-room physics
# ---------------------------------------------------------------------------


class TestComputeHouseTemperature:
    """The House Reference Temperature only counts the conditioned envelope."""

    def test_mean_of_included_rooms(self) -> None:
        rooms = [make_room("a", 20.0), make_room("b", 24.0)]
        assert compute_house_temperature(rooms, {"a": 20.0, "b": 24.0}) == pytest.approx(22.0)

    def test_excluded_room_is_left_out(self) -> None:
        rooms = [make_room("a", 20.0), make_room("b", 24.0, include_in_house_mean=False)]
        assert compute_house_temperature(rooms, {"a": 20.0, "b": 24.0}) == pytest.approx(20.0)

    def test_extra_hallway_reading(self) -> None:
        rooms = [make_room("a", 20.0)]
        assert compute_house_temperature(rooms, {"a": 20.0}, extra_house_temp=24.0) == pytest.approx(22.0)

    def test_fallback_to_all_rooms(self) -> None:
        rooms = [make_room("a", 20.0, include_in_house_mean=False), make_room("b", 24.0)]
        assert compute_house_temperature(rooms, {"a": 20.0, "b": 24.0}) == pytest.approx(24.0)

    def test_no_rooms_falls_back_to_default(self) -> None:
        assert compute_house_temperature([], {}) == DEFAULT_HOUSE_TEMP_C


class TestRoomReferenceTemperatures:
    """A room's local reference falls back to the whole-house mean when unconfigured."""

    def test_isolated_room_uses_the_house_mean_exactly(self) -> None:
        rooms = [make_room("a", 20.0), make_room("b", 24.0)]
        temps = {"a": 20.0, "b": 24.0}

        refs = compute_room_reference_temperatures(rooms, temps)

        fallback = compute_house_temperature(rooms, temps)
        assert refs == {"a": fallback, "b": fallback}

    def test_single_warm_neighbour_becomes_the_reference(self) -> None:
        rooms = [make_room("a", 20.0, neighbours={"b": 1.0}), make_room("b", 24.0)]
        temps = {"a": 20.0, "b": 24.0}

        refs = compute_room_reference_temperatures(rooms, temps)

        assert refs["a"] == 24.0  # not the 22 C house mean
        assert refs["b"] == 22.0

    def test_window_open_neighbour_is_excluded(self) -> None:
        rooms = [
            make_room("a", 20.0, neighbours={"b": 1.0}),
            make_room("b", 24.0, include_in_house_mean=False),
        ]
        temps = {"a": 20.0, "b": 24.0}

        refs = compute_room_reference_temperatures(rooms, temps)

        assert refs["a"] == compute_house_temperature(rooms, temps) == 20.0

    def test_step_house_block_uses_each_rooms_own_reference(self) -> None:
        model = RCModel(C=1.0, U=0.0, U_house=0.2, Q_solar=0.0, Q_heat=0.0, Q_cool=0.0)
        other = make_room("b", 26.0, model=model)
        local = make_room("a", 20.0, neighbours={"b": 1.0}, model=model)
        global_ref = make_room("a", 20.0, model=model)
        temps = {"a": 20.0, "b": 26.0}

        local_next, _, _ = step_house_block([local, other], temps, {}, {}, MODE_IDLE, 20.0, 0)
        global_next, _, _ = step_house_block([global_ref, other], temps, {}, {}, MODE_IDLE, 20.0, 0)

        # The warm neighbour warms the local room faster than the 23 C house mean.
        assert local_next["a"] > global_next["a"]


class TestStepHouseBlock:
    """One coupled step moves every room with the shared plant action."""

    def test_heating_raises_every_room_and_charges_the_air(self) -> None:
        rooms = [make_room("a", 20.0), make_room("b", 21.0)]
        temperatures, humidities, charges = step_house_block(
            rooms,
            {"a": 20.0, "b": 21.0},
            {"a": None, "b": None},
            {"a": 0.0, "b": 0.0},
            MODE_HEATING,
            5.0,
            0,
        )
        assert temperatures["a"] > 20.0
        assert temperatures["b"] > 21.0
        assert charges["a"] > 0.0
        assert humidities["a"] is None

    def test_cooling_lowers_every_room_and_charges_negative(self) -> None:
        rooms = [make_room("a", 24.0), make_room("b", 23.0)]
        temperatures, _humidities, charges = step_house_block(
            rooms,
            {"a": 24.0, "b": 23.0},
            {"a": None, "b": None},
            {"a": 0.0, "b": 0.0},
            MODE_COOLING,
            30.0,
            0,
        )
        assert temperatures["a"] < 24.0
        assert temperatures["b"] < 23.0
        assert charges["a"] < 0.0

    def test_circulation_mixes_toward_the_house_mean(self) -> None:
        rooms = [make_room("warm", 25.0), make_room("cool", 20.0)]
        temperatures, _humidities, charges = step_house_block(
            rooms,
            {"warm": 25.0, "cool": 20.0},
            {"warm": None, "cool": None},
            {"warm": 0.0, "cool": 0.0},
            MODE_CIRCULATING,
            15.0,
            0,
        )
        assert temperatures["warm"] < 25.0
        assert temperatures["cool"] > 20.0
        assert charges["warm"] == pytest.approx(0.0)

    def test_idle_relaxes_a_forced_air_charge(self) -> None:
        rooms = [make_room("a", 20.0, model=RCModel(Q_fast=20.0))]
        temperatures, _humidities, charges = step_house_block(
            rooms,
            {"a": 20.0},
            {"a": None},
            {"a": -0.8},
            MODE_IDLE,
            10.0,
            0,
        )
        assert temperatures["a"] > 20.0
        assert abs(charges["a"]) < 0.8

    def test_humidity_is_tracked_while_cooling(self) -> None:
        rooms = [make_room("bath", 23.5, rh=70.0)]
        temperatures, humidities, _charges = step_house_block(
            rooms,
            {"bath": 23.5},
            {"bath": 70.0},
            {"bath": 0.0},
            MODE_COOLING,
            30.0,
            0,
        )
        assert temperatures["bath"] < 23.5
        assert humidities["bath"] is not None
        assert humidities["bath"] < 100.0

    def test_prediction_is_clamped_near_the_start(self) -> None:
        hot_model = RCModel(Q_heat=1000.0)
        rooms = [make_room("a", 20.0, model=hot_model)]
        temperatures, _humidities, _charges = step_house_block(
            rooms,
            {"a": 20.0},
            {"a": None},
            {"a": 0.0},
            MODE_HEATING,
            5.0,
            0,
        )
        assert temperatures["a"] <= 45.0

    def test_room_above_the_ceiling_stays_put(self) -> None:
        rooms = [make_room("a", 46.0)]
        temperatures, _humidities, _charges = step_house_block(
            rooms,
            {"a": 46.0},
            {"a": None},
            {"a": 0.0},
            MODE_IDLE,
            5.0,
            0,
        )
        assert temperatures["a"] <= 46.0

    def test_door_factors_and_disturbances_enter_the_step(self) -> None:
        rooms = [
            make_room(
                "a",
                20.0,
                door_mix_factor=0.4,
                door_supply_factor=0.75,
                solar_series=[0.6],
                occupancy_gain_series=[1.0],
            )
        ]
        temperatures, _humidities, _charges = step_house_block(
            rooms,
            {"a": 20.0},
            {"a": None},
            {"a": 0.0},
            MODE_HEATING,
            5.0,
            0,
            extra_house_temp=24.0,
        )
        assert temperatures["a"] > 20.0


class TestBoundedCoolTarget:
    """The dehumidification offset never reverses the plant inside one room."""

    def test_no_cool_target(self) -> None:
        assert bounded_dehumidified_cool_target(20.0, None, 16.0, 0.5) is None

    def test_offset_lowers_the_target(self) -> None:
        assert bounded_dehumidified_cool_target(20.0, 24.0, 16.0, 0.5) == pytest.approx(23.5)

    def test_floor_holds_above_the_heat_target(self) -> None:
        assert bounded_dehumidified_cool_target(20.0, 20.6, 16.0, 0.8) == pytest.approx(20.5)

    def test_floor_uses_the_guardrail_without_a_heat_target(self) -> None:
        assert bounded_dehumidified_cool_target(None, 16.2, 16.0, 0.8) == pytest.approx(16.5)

    def test_negative_offset_is_ignored(self) -> None:
        assert bounded_dehumidified_cool_target(20.0, 24.0, 16.0, -0.4) == pytest.approx(24.0)


# ---------------------------------------------------------------------------
# Whole-home judgements
# ---------------------------------------------------------------------------


class TestGuardrailBreach:
    """Guardrail breaches are found regardless of Room Priority."""

    def test_below_the_heat_guardrail(self) -> None:
        rooms = [make_room("guest", 15.0, priority=0.0, occupied=0.0)]
        assert guardrail_breach_direction(rooms, {"guest": 15.0}) == MODE_HEATING

    def test_above_the_cool_guardrail(self) -> None:
        rooms = [make_room("attic", 30.0, priority=0.0, occupied=0.0)]
        assert guardrail_breach_direction(rooms, {"attic": 30.0}) == MODE_COOLING

    def test_no_breach(self) -> None:
        rooms = [make_room("a", 21.0)]
        assert guardrail_breach_direction(rooms, {"a": 21.0}) is None

    def test_worst_breach_wins(self) -> None:
        rooms = [
            make_room("a", 15.4, priority=0.0, occupied=0.0),
            make_room("b", 30.5, priority=0.0, occupied=0.0, guardrail_cool=28.0),
        ]
        assert guardrail_breach_direction(rooms, {"a": 15.4, "b": 30.5}) == MODE_COOLING

    def test_missing_temperature_is_skipped(self) -> None:
        rooms = [make_room("a", 21.0)]
        assert guardrail_breach_direction(rooms, {}) is None


class TestStartDiscomfort:
    """The start hysteresis needs a measurable error or a guardrail breach."""

    def test_heating_discomfort_above_hysteresis(self) -> None:
        rooms = [make_room("a", 20.5, heat=21.0)]
        assert start_discomfort_reached(rooms, MODE_HEATING, {"a": 20.5}, {"a": None}) is True

    def test_heating_discomfort_below_hysteresis(self) -> None:
        rooms = [make_room("a", 20.9, heat=21.0)]
        assert start_discomfort_reached(rooms, MODE_HEATING, {"a": 20.9}, {"a": None}) is False

    def test_guardrail_breach_clears_hysteresis_at_priority_zero(self) -> None:
        rooms = [make_room("a", 15.0, heat=21.0, priority=0.0, occupied=0.0)]
        assert start_discomfort_reached(rooms, MODE_HEATING, {"a": 15.0}, {"a": None}) is True

    def test_priority_zero_comfort_error_does_not_start_the_plant(self) -> None:
        rooms = [make_room("a", 18.0, heat=21.0, priority=0.0, occupied=0.0)]
        assert start_discomfort_reached(rooms, MODE_HEATING, {"a": 18.0}, {"a": None}) is False

    def test_cooling_mirror(self) -> None:
        rooms = [make_room("a", 24.0, cool=23.0)]
        assert start_discomfort_reached(rooms, MODE_COOLING, {"a": 24.0}, {"a": None}) is True
        rooms_ok = [make_room("a", 23.1, cool=23.0)]
        assert start_discomfort_reached(rooms_ok, MODE_COOLING, {"a": 23.1}, {"a": None}) is False

    def test_missing_target_does_not_start(self) -> None:
        rooms = [make_room("a", 18.0, heat=None, cool=None)]
        assert start_discomfort_reached(rooms, MODE_HEATING, {"a": 18.0}, {"a": None}) is False
        assert start_discomfort_reached(rooms, MODE_COOLING, {"a": 18.0}, {"a": None}) is False

    def test_missing_temperature_never_starts(self) -> None:
        rooms = [make_room("a", 18.0, heat=21.0, cool=23.0)]
        assert start_discomfort_reached(rooms, MODE_HEATING, {}, {"a": None}) is False

    def test_cooling_guardrail_breach_clears_hysteresis_at_priority_zero(self) -> None:
        rooms = [make_room("a", 29.0, heat=21.0, cool=23.0, priority=0.0, occupied=0.0)]
        assert start_discomfort_reached(rooms, MODE_COOLING, {"a": 29.0}, {"a": None}) is True

    def test_unoccupied_room_without_cooling_discomfort_does_not_start(self) -> None:
        rooms = [make_room("a", 22.0, heat=20.0, cool=23.0, priority=0.0, occupied=0.0)]
        assert start_discomfort_reached(rooms, MODE_COOLING, {"a": 22.0}, {"a": None}) is False

    def test_idle_never_starts(self) -> None:
        rooms = [make_room("a", 18.0, heat=21.0)]
        assert start_discomfort_reached(rooms, MODE_IDLE, {"a": 18.0}, {"a": None}) is False

    def test_humidity_makes_the_room_feel_warmer(self) -> None:
        rooms = [make_room("bath", 23.5, cool=24.0, rh=70.0)]
        assert start_discomfort_reached(rooms, MODE_COOLING, {"bath": 23.5}, {"bath": 70.0}) is True


class TestDrivingRooms:
    """Driving rooms rank the discomfort that selected the plant action."""

    def test_idle_has_no_driving_room(self) -> None:
        rooms = [make_room("a", 25.0)]
        assert driving_rooms(rooms, MODE_IDLE, {"a": 25.0}, {"a": None}, 22.0) == []

    def test_heating_ranks_the_coldest_room_first(self) -> None:
        rooms = [make_room("a", 20.4, heat=21.0), make_room("b", 19.0, heat=21.0)]
        result = driving_rooms(rooms, MODE_HEATING, {"a": 20.4, "b": 19.0}, {"a": None, "b": None}, 20.0)
        assert result == ["b", "a"]

    def test_cooling_ranks_the_warmest_room_first(self) -> None:
        rooms = [make_room("a", 23.4, cool=23.0), make_room("b", 25.0, cool=23.0)]
        result = driving_rooms(rooms, MODE_COOLING, {"a": 23.4, "b": 25.0}, {"a": None, "b": None}, 24.0)
        assert result == ["b", "a"]

    def test_circulation_ranks_by_distance_from_the_house(self) -> None:
        rooms = [make_room("a", 21.0), make_room("b", 25.0)]
        result = driving_rooms(rooms, MODE_CIRCULATING, {"a": 21.0, "b": 25.0}, {"a": None, "b": None}, 22.0)
        assert result == ["b", "a"]

    def test_guardrail_breach_counts_as_discomfort(self) -> None:
        rooms = [make_room("guest", 15.0, heat=21.0, priority=0.0, occupied=0.0)]
        result = driving_rooms(rooms, MODE_HEATING, {"guest": 15.0}, {"guest": None}, 21.0)
        assert result == ["guest"]

    def test_cooling_guardrail_breach_counts_as_discomfort(self) -> None:
        rooms = [make_room("attic", 30.0, heat=20.0, cool=28.0, priority=0.0, occupied=0.0)]
        result = driving_rooms(rooms, MODE_COOLING, {"attic": 30.0}, {"attic": None}, 24.0)
        assert result == ["attic"]

    def test_comfortable_rooms_are_not_listed(self) -> None:
        rooms = [make_room("a", 21.0, heat=20.0, cool=23.0)]
        assert driving_rooms(rooms, MODE_HEATING, {"a": 21.0}, {"a": None}, 21.0) == []

    def test_missing_temperature_is_skipped(self) -> None:
        rooms = [make_room("a", 25.0)]
        assert driving_rooms(rooms, MODE_CIRCULATING, {}, {"a": None}, 21.0) == []


# ---------------------------------------------------------------------------
# MPC readiness gates
# ---------------------------------------------------------------------------


class TestRoomMpcReadiness:
    """A room's model must have seen enough excitation before the joint MPC."""

    def test_fresh_estimator_is_not_ready(self) -> None:
        assert is_room_mpc_ready(ThermalEKF(), True, True) is False

    def test_trained_estimator_is_ready(self) -> None:
        assert is_room_mpc_ready(ready_estimator(), True, False) is True

    def test_heating_only_room_does_not_need_cooling_data(self) -> None:
        estimator = ready_estimator(cooling=0)
        assert estimator._n_cooling == 0
        assert is_room_mpc_ready(estimator, True, True) is True

    def test_too_few_idle_samples(self) -> None:
        estimator = ready_estimator(idle=30, heating=20)
        assert is_room_mpc_ready(estimator, True, True) is False

    def test_too_few_active_samples(self) -> None:
        estimator = ready_estimator(idle=45, heating=5)
        assert is_room_mpc_ready(estimator, True, True) is False

    def test_circulation_counts_as_active_excitation(self) -> None:
        estimator = ready_estimator(idle=45, heating=0, circulating=15)
        assert is_room_mpc_ready(estimator, True, False) is True

    def test_large_prediction_std_blocks_readiness(self) -> None:
        estimator = ready_estimator()
        estimator._P[0][0] = 10.0
        assert estimator.prediction_std(0.0, 20.0, 15.0, PLAN_DT_MINUTES) > MPC_MAX_PREDICTION_STD
        assert is_room_mpc_ready(estimator, True, True) is False

    def test_cooling_mode_std_is_gated_when_learned(self) -> None:
        estimator = ready_estimator(cooling=5)
        estimator._P[ThermalEKF._STATE_BETA_C][ThermalEKF._STATE_BETA_C] = 100.0
        assert estimator.prediction_std(-estimator._x[ThermalEKF._STATE_BETA_C], 20.0, 25.0, PLAN_DT_MINUTES) > (
            MPC_MAX_PREDICTION_STD
        )
        assert is_room_mpc_ready(estimator, True, True) is False


# ---------------------------------------------------------------------------
# Equipment-guard simulation
# ---------------------------------------------------------------------------


class TestPlantRunState:
    """The simulated guards replay the plant's protection rules per block."""

    def test_min_run_forces_the_running_action(self) -> None:
        state = _PlantRunState(last_action=MODE_HEATING, run_blocks=1)
        allowed = state.candidates(
            {MODE_IDLE, MODE_HEATING, MODE_COOLING},
            min_run_blocks=2,
            min_fan_blocks=2,
            min_off_blocks=0,
            changeover_blocks=0,
            wet_coil_locked_blocks=0,
        )
        assert allowed == {MODE_HEATING}

    def test_forced_action_falls_back_to_idle_when_illegal(self) -> None:
        state = _PlantRunState(last_action=MODE_COOLING, run_blocks=0)
        allowed = state.candidates(
            {MODE_IDLE, MODE_HEATING},
            min_run_blocks=2,
            min_fan_blocks=2,
            min_off_blocks=0,
            changeover_blocks=0,
            wet_coil_locked_blocks=0,
        )
        assert allowed == {MODE_IDLE}

    def test_min_fan_run_forces_circulation(self) -> None:
        state = _PlantRunState(last_action=MODE_CIRCULATING, run_blocks=1)
        allowed = state.candidates(
            {MODE_IDLE, MODE_CIRCULATING, MODE_HEATING},
            min_run_blocks=2,
            min_fan_blocks=3,
            min_off_blocks=0,
            changeover_blocks=0,
            wet_coil_locked_blocks=0,
        )
        assert allowed == {MODE_CIRCULATING}

    def test_min_off_blocks_a_restart(self) -> None:
        state = _PlantRunState(last_action=MODE_IDLE, run_blocks=3, off_blocks=0)
        allowed = state.candidates(
            {MODE_IDLE, MODE_HEATING, MODE_COOLING, MODE_CIRCULATING},
            min_run_blocks=2,
            min_fan_blocks=2,
            min_off_blocks=2,
            changeover_blocks=0,
            wet_coil_locked_blocks=0,
        )
        assert MODE_HEATING not in allowed
        assert MODE_COOLING not in allowed
        assert MODE_CIRCULATING in allowed

    def test_min_off_does_not_stop_a_running_conditioning_run(self) -> None:
        state = _PlantRunState(last_action=MODE_HEATING, run_blocks=4, off_blocks=0)
        allowed = state.candidates(
            {MODE_IDLE, MODE_HEATING},
            min_run_blocks=2,
            min_fan_blocks=2,
            min_off_blocks=2,
            changeover_blocks=0,
            wet_coil_locked_blocks=0,
        )
        assert MODE_HEATING in allowed

    def test_changeover_blocks_a_direction_reversal(self) -> None:
        state = _PlantRunState(last_action=MODE_IDLE, run_blocks=1, off_blocks=5, since_heat=1, since_cool=5)
        allowed = state.candidates(
            {MODE_IDLE, MODE_HEATING, MODE_COOLING},
            min_run_blocks=1,
            min_fan_blocks=1,
            min_off_blocks=0,
            changeover_blocks=3,
            wet_coil_locked_blocks=0,
        )
        assert MODE_COOLING not in allowed
        assert MODE_HEATING in allowed

    def test_wet_coil_locks_circulation_out(self) -> None:
        state = _PlantRunState(last_action=MODE_IDLE, run_blocks=2, off_blocks=3, since_heat=9, since_cool=1)
        allowed = state.candidates(
            {MODE_IDLE, MODE_COOLING, MODE_CIRCULATING},
            min_run_blocks=1,
            min_fan_blocks=1,
            min_off_blocks=0,
            changeover_blocks=0,
            wet_coil_locked_blocks=4,
        )
        assert MODE_CIRCULATING not in allowed
        assert MODE_COOLING in allowed

    def test_observe_tracks_the_run_and_the_timers(self) -> None:
        state = _PlantRunState()
        state.observe(MODE_HEATING)
        assert state.last_action == MODE_HEATING
        assert state.run_blocks == 1
        assert state.since_heat == 0
        state.observe(MODE_HEATING)
        assert state.run_blocks == 2
        state.observe(MODE_IDLE)
        assert state.last_action == MODE_IDLE
        assert state.run_blocks == 1
        assert state.since_heat == 1
        assert state.off_blocks == 1
        state.observe(MODE_COOLING)
        assert state.since_cool == 0
        assert state.since_heat == 2
        state.observe(MODE_CIRCULATING)
        assert state.since_cool == 1
        assert state.since_heat == 3


# ---------------------------------------------------------------------------
# Acceptance: stratified house chooses circulation
# ---------------------------------------------------------------------------


class TestStratifiedHouseCirculation:
    """A warm office and a cool living room are best served by the blower."""

    def test_circulation_beats_cooling_and_idle(self) -> None:
        office = make_room("office", 25.0, heat=20.0, cool=23.0)
        living = make_room("living", 20.2, heat=20.0, cool=23.0)
        optimizer = CentralMPCOptimizer(
            rooms=[office, living],
            T_outdoor_series=[15.0] * N_BLOCKS,
        )
        plan = optimizer.optimize()
        assert plan.current_action() == MODE_CIRCULATING
        assert plan.decision_reason == "circulate_mix"
        assert plan.mpc_active is True
        assert plan.driving_rooms == ["living", "office"]

        costs = action_costs(
            optimizer,
            {"office": 25.0, "living": 20.2},
            {"office": None, "living": None},
            {"office": 0.0, "living": 0.0},
        )
        assert costs[MODE_CIRCULATING] < costs[MODE_IDLE]
        assert costs[MODE_CIRCULATING] < costs[MODE_COOLING]
        assert costs[MODE_CIRCULATING] < costs[MODE_HEATING]

    def test_circulation_does_not_chill_the_living_room(self) -> None:
        office = make_room("office", 25.0, heat=20.0, cool=23.0)
        living = make_room("living", 20.2, heat=20.0, cool=23.0)
        plan = CentralMPCOptimizer(
            rooms=[office, living],
            T_outdoor_series=[15.0] * N_BLOCKS,
        ).optimize()
        assert plan.actions[0] == MODE_CIRCULATING
        assert min(plan.room_temperatures["living"]) >= 20.0
        assert min(plan.room_temperatures["office"]) < 25.0
        assert max(plan.room_temperatures["living"]) <= 23.0

    def test_cooling_would_chill_the_cool_room(self) -> None:
        office = make_room("office", 25.0, heat=20.0, cool=23.0)
        living = make_room("living", 20.2, heat=20.0, cool=23.0)
        optimizer = CentralMPCOptimizer(
            rooms=[office, living],
            T_outdoor_series=[15.0] * N_BLOCKS,
        )
        temperatures = {"office": 25.0, "living": 20.2}
        humidities: dict[str, float | None] = {"office": None, "living": None}
        charges = {"office": 0.0, "living": 0.0}
        for _ in range(LOOKAHEAD_BLOCKS):
            temperatures, humidities, charges = step_house_block(
                optimizer.rooms,
                temperatures,
                humidities,
                charges,
                MODE_COOLING,
                15.0,
                0,
            )
        assert temperatures["living"] < 20.0

    def test_circulation_run_is_limited_by_the_fan_min_run(self) -> None:
        office = make_room("office", 25.0, heat=20.0, cool=23.0)
        living = make_room("living", 20.2, heat=20.0, cool=23.0)
        plan = CentralMPCOptimizer(
            rooms=[office, living],
            T_outdoor_series=[15.0] * N_BLOCKS,
            min_fan_blocks=2,
        ).optimize()
        assert plan.actions[:2] == [MODE_CIRCULATING, MODE_CIRCULATING]


# ---------------------------------------------------------------------------
# Acceptance: occupied priority beats unoccupied drift
# ---------------------------------------------------------------------------


class TestOccupiedPriorityBeatsUnoccupiedDrift:
    """An empty room drifts inside its Guardrail Band without starting the plant."""

    def test_empty_guest_room_does_not_heat(self) -> None:
        guest = make_room(
            "guest",
            18.0,
            heat=21.0,
            cool=24.0,
            priority=0.0,
            occupied=0.0,
            guardrail_heat=16.0,
            guardrail_cool=28.0,
        )
        plan = CentralMPCOptimizer(rooms=[guest], T_outdoor_series=[5.0] * N_BLOCKS).optimize()
        assert plan.current_action() == MODE_IDLE
        assert plan.decision_reason == "idle"

    def test_guest_below_the_guardrail_heats(self) -> None:
        guest = make_room(
            "guest",
            15.5,
            heat=21.0,
            cool=24.0,
            priority=0.0,
            occupied=0.0,
            guardrail_heat=16.0,
        )
        plan = CentralMPCOptimizer(rooms=[guest], T_outdoor_series=[5.0] * N_BLOCKS).optimize()
        assert plan.current_action() == MODE_HEATING
        assert plan.decision_reason == "guardrail"
        assert plan.driving_rooms == ["guest"]

    def test_occupied_guest_room_does_heat(self) -> None:
        guest = make_room(
            "guest",
            18.0,
            heat=21.0,
            cool=24.0,
            priority=1.0,
            occupied=1.0,
            guardrail_heat=16.0,
        )
        plan = CentralMPCOptimizer(rooms=[guest], T_outdoor_series=[5.0] * N_BLOCKS).optimize()
        assert plan.current_action() == MODE_HEATING
        assert plan.decision_reason == "mpc"

    def test_occupied_room_outranks_the_guest_drift(self) -> None:
        guest = make_room(
            "guest",
            18.0,
            heat=21.0,
            cool=23.0,
            priority=0.08,
            occupied=0.0,
            guardrail_heat=16.0,
        )
        # The occupied living room is already comfortable: heating for the guest
        # would push it past its cool bound, so the plant stays off.
        living = make_room("living", 22.6, heat=20.0, cool=23.0, priority=1.0, occupied=1.0)
        plan = CentralMPCOptimizer(
            rooms=[guest, living],
            T_outdoor_series=[18.0] * N_BLOCKS,
        ).optimize()
        assert plan.current_action() != MODE_HEATING


# ---------------------------------------------------------------------------
# Acceptance: anticipatory pre-heating from an occupancy trend
# ---------------------------------------------------------------------------


class TestArrivalPreheat:
    """An arriving room is warmed ahead of time, not after it is late."""

    def test_preheats_ahead_of_the_occupancy_trend(self) -> None:
        office = arrival_room()
        plan = CentralMPCOptimizer(rooms=[office], T_outdoor_series=[5.0] * N_BLOCKS).optimize()
        assert plan.current_action() == MODE_HEATING
        assert plan.decision_reason == "preheat"
        assert plan.preheat_info == {"office": {"mode": MODE_HEATING, "target": 21.0, "at": 45.0}}
        assert plan.room_temperatures["office"][9] >= 21.0 - ARRIVAL_DEADLINE_MARGIN_C

    def test_without_preheat_the_arrival_is_late(self) -> None:
        office = arrival_room()
        office.preheat_enabled = False
        plan = CentralMPCOptimizer(rooms=[office], T_outdoor_series=[5.0] * N_BLOCKS).optimize()
        assert plan.decision_reason != "preheat"
        assert plan.preheat_info == {}
        assert plan.room_temperatures["office"][9] < 21.0 - ARRIVAL_DEADLINE_MARGIN_C

    def test_the_slider_caps_the_preheat_lead(self) -> None:
        office = arrival_room()
        plan = CentralMPCOptimizer(
            rooms=[office],
            T_outdoor_series=[5.0] * N_BLOCKS,
            comfort_weight=0.0,
        ).optimize()
        assert plan.decision_reason != "preheat"
        assert plan.preheat_info == {}

    def test_precool_prepares_a_falling_cool_target(self) -> None:
        falling_cool: list[float | None] = []
        falling_cool.extend([25.0] * 6)
        falling_cool.extend([23.0] * (N_BLOCKS - 6))
        bath = RoomHorizonInput(
            area_id="bath",
            model=RCModel(),
            T_init=25.5,
            priority_series=[0.1] * 6 + [0.85] * (N_BLOCKS - 6),
            occupied_prob_series=[0.1] * 6 + [0.85] * (N_BLOCKS - 6),
            comfort_heat_series=[19.0] * N_BLOCKS,
            comfort_cool_series=falling_cool,
        )
        plan = CentralMPCOptimizer(rooms=[bath], T_outdoor_series=[30.0] * N_BLOCKS).optimize()
        assert plan.current_action() == MODE_COOLING
        assert plan.decision_reason == "precool"
        assert plan.preheat_info["bath"]["mode"] == MODE_COOLING
        assert plan.preheat_info["bath"]["target"] == pytest.approx(23.0)
        baseline = CentralMPCOptimizer(
            rooms=[bath],
            T_outdoor_series=[30.0] * N_BLOCKS,
            initial_legal_actions={MODE_IDLE, MODE_CIRCULATING},
        ).optimize()
        assert plan.room_temperatures["bath"][6] < baseline.room_temperatures["bath"][6]

    def test_a_deadline_the_guard_cannot_honour_is_not_a_badge(self) -> None:
        bath = RoomHorizonInput(
            area_id="bath",
            model=RCModel(),
            T_init=25.5,
            priority_series=[0.1] * 6 + [0.85] * (N_BLOCKS - 6),
            occupied_prob_series=[0.1] * 6 + [0.85] * (N_BLOCKS - 6),
            comfort_heat_series=[19.0] * N_BLOCKS,
            comfort_cool_series=_optional_series(25.0, 6, 23.0),
        )
        # Outdoor air below the cooling gate: the precool deadline cannot run.
        plan = CentralMPCOptimizer(rooms=[bath], T_outdoor_series=[10.0] * N_BLOCKS).optimize()
        assert plan.current_action() != MODE_COOLING
        assert plan.preheat_info == {}

    def test_a_rising_heat_target_alone_preheats(self) -> None:
        office = make_room("office", 19.0, heat=20.0, priority=1.0, occupied=1.0)
        rising: list[float | None] = []
        rising.extend([20.0] * 6)
        rising.extend([22.0] * (N_BLOCKS - 6))
        office.comfort_heat_series = rising
        plan = CentralMPCOptimizer(rooms=[office], T_outdoor_series=[5.0] * N_BLOCKS).optimize()
        assert plan.current_action() == MODE_HEATING
        assert plan.decision_reason == "preheat"
        assert plan.preheat_info["office"]["target"] == pytest.approx(22.0)

    def test_a_warm_occupied_room_blocks_the_preheat(self) -> None:
        office = arrival_room(jump_block=6, T_init=18.0, heat_before=19.0)
        living = make_room("living", 23.0, heat=20.0, cool=23.0, priority=1.0, occupied=1.0)
        optimizer = CentralMPCOptimizer(
            rooms=[office, living],
            T_outdoor_series=[5.0] * N_BLOCKS,
        )
        direction, info = optimizer._check_arrival_deadlines(
            {"office": 18.0, "living": 23.0},
            {"office": None, "living": None},
            {"office": 0.0, "living": 0.0},
        )
        assert direction is None
        assert info == {}
        alone = CentralMPCOptimizer(rooms=[office], T_outdoor_series=[5.0] * N_BLOCKS).optimize()
        assert alone.decision_reason == "preheat"

    def test_preheat_info_lists_every_driving_room(self) -> None:
        first = arrival_room(area_id="office", jump_block=5)
        second = arrival_room(area_id="study", jump_block=6)
        optimizer = CentralMPCOptimizer(rooms=[first, second], T_outdoor_series=[5.0] * N_BLOCKS)
        _direction, info = optimizer._check_arrival_deadlines(
            {"office": 19.0, "study": 19.0},
            {"office": None, "study": None},
            {"office": 0.0, "study": 0.0},
        )
        assert set(info) == {"office", "study"}

    def test_no_arrival_event_means_no_deadline(self) -> None:
        steady = make_room("office", 21.0, heat=21.0)
        optimizer = CentralMPCOptimizer(rooms=[steady], T_outdoor_series=[5.0] * N_BLOCKS)
        assert optimizer._check_arrival_deadlines(
            {"office": 21.0},
            {"office": None},
            {"office": 0.0},
        ) == (None, {})


# ---------------------------------------------------------------------------
# Acceptance: humidity effective temperature and dehumidification offset
# ---------------------------------------------------------------------------


class TestHumidityComfort:
    """A muggy room is cooled even when its dry-bulb temperature is in band."""

    def test_humid_room_cools_within_the_dry_bulb_band(self) -> None:
        bath = make_room("bath", 23.5, heat=21.0, cool=24.0, rh=70.0)
        plan = CentralMPCOptimizer(rooms=[bath], T_outdoor_series=[30.0] * N_BLOCKS).optimize()
        assert plan.current_action() == MODE_COOLING
        assert plan.decision_reason == "mpc"

    def test_dry_room_stays_idle(self) -> None:
        bath = make_room("bath", 23.5, heat=21.0, cool=24.0, rh=70.0)
        plan = CentralMPCOptimizer(
            rooms=[bath],
            T_outdoor_series=[30.0] * N_BLOCKS,
            humidity_comfort_enabled=False,
        ).optimize()
        assert plan.current_action() == MODE_IDLE

    def test_dehumidification_never_cools_below_the_heat_floor(self) -> None:
        bath = make_room("bath", 23.5, heat=21.0, cool=24.0, rh=70.0)
        plan = CentralMPCOptimizer(rooms=[bath], T_outdoor_series=[30.0] * N_BLOCKS).optimize()
        assert min(plan.room_temperatures["bath"]) >= 21.0 + 0.5

    def test_weighted_humidity_skips_zero_priority_rooms(self) -> None:
        bath = make_room("bath", 23.5, rh=70.0, priority=0.0, occupied=0.0)
        optimizer = CentralMPCOptimizer(rooms=[bath], T_outdoor_series=[30.0] * N_BLOCKS)
        assert optimizer._weighted_rh(0, {"bath": 70.0}) is None

    def test_weighted_humidity_is_priority_weighted(self) -> None:
        dry = make_room("dry", 21.0, rh=40.0, priority=1.0, occupied=1.0)
        wet = make_room("wet", 23.0, rh=80.0, priority=3.0, occupied=1.0)
        optimizer = CentralMPCOptimizer(rooms=[dry, wet], T_outdoor_series=[30.0] * N_BLOCKS)
        assert optimizer._weighted_rh(0, {"dry": 40.0, "wet": 80.0}) == pytest.approx(70.0)


# ---------------------------------------------------------------------------
# Optimizer control surface
# ---------------------------------------------------------------------------


class TestOptimizerControl:
    """The optimizer honours the guard state, the climate mode and the horizon."""

    def test_invalid_dt_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            CentralMPCOptimizer(rooms=[make_room()], T_outdoor_series=[5.0], dt_minutes=0.0)

    def test_horizon_is_capped_at_the_maximum(self) -> None:
        plan = CentralMPCOptimizer(
            rooms=[make_room()],
            T_outdoor_series=[15.0] * (MAX_HORIZON_BLOCKS + 40),
            horizon_blocks=10 * MAX_HORIZON_BLOCKS,
        ).optimize()
        assert len(plan.actions) == MAX_HORIZON_BLOCKS
        assert len(plan.house_temperatures) == MAX_HORIZON_BLOCKS + 1
        assert len(plan.room_temperatures["room"]) == MAX_HORIZON_BLOCKS + 1

    def test_default_horizon_is_two_hours(self) -> None:
        plan = CentralMPCOptimizer(
            rooms=[make_room()],
            T_outdoor_series=[15.0] * 200,
        ).optimize()
        assert len(plan.actions) == MIN_HORIZON_BLOCKS

    def test_horizon_blocks_limit(self) -> None:
        plan = CentralMPCOptimizer(
            rooms=[make_room()],
            T_outdoor_series=[15.0] * N_BLOCKS,
            horizon_blocks=10,
        ).optimize()
        assert len(plan.actions) == 10

    def test_missing_outdoor_series_yields_an_empty_plan(self) -> None:
        plan = CentralMPCOptimizer(rooms=[make_room()], T_outdoor_series=[]).optimize()
        assert plan.actions == []
        assert plan.house_temperatures == [20.0]
        assert plan.room_temperatures["room"] == [20.0]
        assert plan.current_action() == MODE_IDLE

    def test_no_rooms_yields_an_empty_plan(self) -> None:
        plan = CentralMPCOptimizer(rooms=[], T_outdoor_series=[5.0] * N_BLOCKS).optimize()
        assert plan.actions == []
        assert plan.mpc_active is False

    def test_forced_action_wins(self) -> None:
        cold = make_room("a", 18.0, heat=21.0)
        plan = CentralMPCOptimizer(
            rooms=[cold],
            T_outdoor_series=[5.0] * N_BLOCKS,
            forced_action=MODE_IDLE,
        ).optimize()
        assert plan.current_action() == MODE_IDLE
        assert plan.decision_reason == "hold_guard"

    def test_forced_action_outside_the_legal_set_falls_back_to_idle(self) -> None:
        cold = make_room("a", 18.0, heat=21.0)
        plan = CentralMPCOptimizer(
            rooms=[cold],
            T_outdoor_series=[5.0] * N_BLOCKS,
            forced_action=MODE_HEATING,
            initial_legal_actions={MODE_IDLE, MODE_CIRCULATING},
        ).optimize()
        assert plan.current_action() == MODE_IDLE
        assert plan.decision_reason == "hold_guard"

    def test_a_run_inside_its_min_run_is_held(self) -> None:
        warm = make_room("a", 22.5, heat=21.0, cool=23.0)
        plan = CentralMPCOptimizer(
            rooms=[warm],
            T_outdoor_series=[5.0] * N_BLOCKS,
            initial_action=MODE_HEATING,
            initial_run_blocks=0,
            min_run_blocks=2,
        ).optimize()
        assert plan.current_action() == MODE_HEATING
        assert plan.decision_reason == "hold_guard"
        assert plan.actions[:2] == [MODE_HEATING, MODE_HEATING]

    def test_min_off_blocks_a_restart_at_block_zero(self) -> None:
        cold = make_room("a", 18.0, heat=21.0)
        plan = CentralMPCOptimizer(
            rooms=[cold],
            T_outdoor_series=[5.0] * N_BLOCKS,
            initial_off_blocks=0,
            min_off_blocks=2,
        ).optimize()
        assert plan.current_action() != MODE_HEATING

    def test_initial_legal_actions_are_respected(self) -> None:
        cold = make_room("a", 18.0, heat=21.0)
        plan = CentralMPCOptimizer(
            rooms=[cold],
            T_outdoor_series=[5.0] * N_BLOCKS,
            initial_legal_actions={MODE_IDLE, MODE_CIRCULATING},
        ).optimize()
        assert plan.current_action() != MODE_HEATING
        assert plan.current_action() != MODE_COOLING

    def test_hot_outdoor_air_gates_heating(self) -> None:
        optimizer = CentralMPCOptimizer(rooms=[make_room()], T_outdoor_series=[25.0] * N_BLOCKS)
        assert MODE_HEATING not in optimizer._base_actions(0)
        assert MODE_COOLING in optimizer._base_actions(0)

    def test_cold_outdoor_air_gates_cooling(self) -> None:
        optimizer = CentralMPCOptimizer(rooms=[make_room()], T_outdoor_series=[10.0] * N_BLOCKS)
        assert MODE_COOLING not in optimizer._base_actions(0)
        assert MODE_HEATING in optimizer._base_actions(0)

    def test_override_bypasses_the_outdoor_gates(self) -> None:
        optimizer = CentralMPCOptimizer(
            rooms=[make_room()],
            T_outdoor_series=[10.0] * N_BLOCKS,
            override_active=True,
        )
        assert {MODE_HEATING, MODE_COOLING, MODE_CIRCULATING, MODE_IDLE} <= optimizer._base_actions(0)

    def test_climate_mode_restricts_the_plant(self) -> None:
        optimizer = CentralMPCOptimizer(
            rooms=[make_room()],
            T_outdoor_series=[5.0] * N_BLOCKS,
            can_heat=False,
        )
        assert MODE_HEATING not in optimizer._base_actions(0)
        optimizer_cool = CentralMPCOptimizer(
            rooms=[make_room()],
            T_outdoor_series=[25.0] * N_BLOCKS,
            can_cool=False,
        )
        assert MODE_COOLING not in optimizer_cool._base_actions(0)

    def test_stage_cost_ignores_rooms_without_a_temperature(self) -> None:
        optimizer = CentralMPCOptimizer(
            rooms=[make_room("a", 21.0), make_room("b", 25.0)],
            T_outdoor_series=[5.0] * N_BLOCKS,
        )
        with_room = optimizer._stage_cost(0, MODE_IDLE, {"a": 21.0, "b": 25.0}, {"a": None, "b": None})
        without_room = optimizer._stage_cost(0, MODE_IDLE, {"a": 21.0}, {"a": None})
        assert without_room < with_room

    def test_comfort_weight_extremes_are_clamped(self) -> None:
        for comfort_weight in (-50.0, 150.0):
            plan = CentralMPCOptimizer(
                rooms=[make_room("a", 20.8, heat=21.0)],
                T_outdoor_series=[5.0] * N_BLOCKS,
                comfort_weight=comfort_weight,
            ).optimize()
            assert plan.current_action() in (MODE_IDLE, MODE_HEATING)

    def test_sub_hysteresis_noise_does_not_start_the_plant(self) -> None:
        almost = make_room("a", 20.9, heat=21.0)
        plan = CentralMPCOptimizer(rooms=[almost], T_outdoor_series=[5.0] * N_BLOCKS).optimize()
        assert plan.current_action() == MODE_IDLE
        assert plan.decision_reason == "idle"

    def test_measurable_discomfort_starts_the_plant(self) -> None:
        cold = make_room("a", 20.4, heat=21.0)
        plan = CentralMPCOptimizer(rooms=[cold], T_outdoor_series=[5.0] * N_BLOCKS).optimize()
        assert plan.current_action() == MODE_HEATING

    def test_wet_coil_lockout_survives_the_rollout(self) -> None:
        plan = CentralMPCOptimizer(
            rooms=[make_room("a", 22.0)],
            T_outdoor_series=[15.0] * N_BLOCKS,
            initial_action=MODE_COOLING,
            initial_run_blocks=2,
            wet_coil_locked_blocks=10,
        ).optimize()
        assert len(plan.actions) == MIN_HORIZON_BLOCKS

    def test_min_run_for_circulation(self) -> None:
        optimizer = CentralMPCOptimizer(
            rooms=[make_room()],
            T_outdoor_series=[15.0] * N_BLOCKS,
            min_fan_blocks=4,
        )
        assert optimizer._min_run_for(MODE_CIRCULATING) == 4
        assert optimizer._min_run_for(MODE_HEATING) == optimizer.min_run_blocks


# ---------------------------------------------------------------------------
# Acceptance: cold-start bang-bang fallback
# ---------------------------------------------------------------------------


class TestCentralBangBang:
    """The cold-start rule keeps the home safe without a learned model."""

    def test_start_hysteresis(self) -> None:
        bangbang = CentralBangBang()
        assert bangbang.decide(rooms=[make_room("a", 20.5, heat=21.0)]) == MODE_HEATING
        assert bangbang.decide(rooms=[make_room("a", 20.9, heat=21.0)]) == MODE_IDLE
        assert bangbang.decide(rooms=[make_room("a", 24.5, cool=23.0)]) == MODE_COOLING
        assert bangbang.decide(rooms=[make_room("a", 23.1, cool=23.0)]) == MODE_IDLE

    def test_continues_until_the_target_is_reached(self) -> None:
        bangbang = CentralBangBang(active_action=MODE_HEATING)
        assert bangbang.decide(rooms=[make_room("a", 20.5, heat=21.0)]) == MODE_HEATING
        assert bangbang.decide(rooms=[make_room("a", 21.0, heat=21.0)]) == MODE_IDLE
        cooling = CentralBangBang(active_action=MODE_COOLING)
        assert cooling.decide(rooms=[make_room("a", 23.5, cool=23.0)]) == MODE_COOLING
        assert cooling.decide(rooms=[make_room("a", 23.0, cool=23.0)]) == MODE_IDLE

    def test_opposing_occupied_rooms_circulate(self) -> None:
        office = make_room("office", 25.0, heat=20.0, cool=23.0)
        living = make_room("living", 20.2, heat=20.0, cool=23.0)
        assert CentralBangBang().decide(rooms=[office, living]) == MODE_CIRCULATING

    def test_spread_with_mild_error_circulates(self) -> None:
        warm = make_room("warm", 23.5, heat=21.0, cool=23.0)
        cool = make_room("cool", 20.3, heat=20.0, cool=23.0)
        assert CentralBangBang().decide(rooms=[warm, cool]) == MODE_CIRCULATING

    def test_spread_without_circulation_legal_falls_back(self) -> None:
        office = make_room("office", 25.0, heat=20.0, cool=23.0)
        living = make_room("living", 20.2, heat=20.0, cool=23.0)
        action = CentralBangBang().decide(
            rooms=[office, living],
            legal_actions={MODE_IDLE, MODE_COOLING},
        )
        assert action == MODE_COOLING

    def test_forced_action_is_held(self) -> None:
        bangbang = CentralBangBang()
        cold = make_room("a", 18.0, heat=21.0)
        assert bangbang.decide(rooms=[cold], forced_action=MODE_IDLE) == MODE_IDLE
        assert bangbang.decide(rooms=[cold], forced_action=MODE_CIRCULATING) == MODE_CIRCULATING
        assert (
            bangbang.decide(
                rooms=[cold],
                forced_action=MODE_HEATING,
                legal_actions={MODE_IDLE, MODE_CIRCULATING},
            )
            == MODE_IDLE
        )

    def test_legal_actions_are_respected(self) -> None:
        cold = make_room("a", 18.0, heat=21.0)
        assert CentralBangBang().decide(rooms=[cold], legal_actions={MODE_IDLE, MODE_COOLING}) == MODE_IDLE
        assert CentralBangBang().decide(rooms=[cold], legal_actions={MODE_IDLE, MODE_HEATING}) == MODE_HEATING

    def test_guardrail_breach_heats_at_priority_zero(self) -> None:
        guest = make_room("guest", 15.5, heat=21.0, priority=0.0, occupied=0.0)
        assert CentralBangBang().decide(rooms=[guest]) == MODE_HEATING

    def test_no_targets_means_idle(self) -> None:
        room = make_room("a", 20.0, heat=None, cool=None)
        assert CentralBangBang().decide(rooms=[room]) == MODE_IDLE

    def test_humidity_effective_temperature_cools(self) -> None:
        bath = make_room("bath", 23.5, heat=21.0, cool=24.0, rh=70.0)
        assert CentralBangBang().decide(rooms=[bath]) == MODE_COOLING

    def test_single_occupied_room_is_no_spread(self) -> None:
        room = make_room("a", 22.5, heat=21.0, cool=23.0)
        assert CentralBangBang().decide(rooms=[room]) == MODE_IDLE

    def test_zero_priority_rooms_do_not_weight_the_errors(self) -> None:
        room = make_room("a", 18.0, heat=21.0, priority=0.0, occupied=0.0)
        assert CentralBangBang().decide(rooms=[room]) == MODE_IDLE

    def test_cooling_falls_back_when_heating_is_illegal(self) -> None:
        room = make_room("a", 21.0, heat=21.5, cool=20.7)
        action = CentralBangBang().decide(rooms=[room], legal_actions={MODE_IDLE, MODE_COOLING})
        assert action == MODE_COOLING

    def test_hysteresis_constants_are_the_start_threshold(self) -> None:
        room = make_room("a", 21.0 - BANGBANG_HEAT_HYSTERESIS - 0.01, heat=21.0)
        assert CentralBangBang().decide(rooms=[room]) == MODE_HEATING
        room_cool = make_room("a", 23.0 + BANGBANG_COOL_HYSTERESIS + 0.01, cool=23.0)
        assert CentralBangBang().decide(rooms=[room_cool]) == MODE_COOLING

    def test_mixable_allows_undeclared_or_shared_groups_only(self) -> None:
        undeclared = make_room("a")
        group_0 = make_room("b", mixing_group=0)
        other_0 = make_room("c", mixing_group=0)
        group_1 = make_room("d", mixing_group=1)
        assert CentralBangBang._mixable(undeclared, group_0)
        assert CentralBangBang._mixable(group_0, undeclared)
        assert CentralBangBang._mixable(group_0, other_0)
        assert not CentralBangBang._mixable(group_0, group_1)

    def test_different_declared_groups_do_not_circulate(self) -> None:
        office = make_room("office", 25.0, heat=20.0, cool=23.0, mixing_group=0)
        living = make_room("living", 20.2, heat=20.0, cool=23.0, mixing_group=1)
        assert CentralBangBang().decide(rooms=[office, living]) != MODE_CIRCULATING

    def test_same_declared_group_still_circulates(self) -> None:
        office = make_room("office", 25.0, heat=20.0, cool=23.0, mixing_group=0)
        living = make_room("living", 20.2, heat=20.0, cool=23.0, mixing_group=0)
        assert CentralBangBang().decide(rooms=[office, living]) == MODE_CIRCULATING

    def test_spread_across_different_groups_does_not_circulate(self) -> None:
        warm = make_room("warm", 23.5, heat=21.0, cool=23.0, mixing_group=0)
        cool = make_room("cool", 20.3, heat=20.0, cool=23.0, mixing_group=1)
        assert CentralBangBang().decide(rooms=[warm, cool]) != MODE_CIRCULATING

    def test_undeclared_rooms_keep_the_old_circulation_behaviour(self) -> None:
        # Both mixing_group=None: one bucket, identical to the pre-feature spread.
        warm = make_room("warm", 23.5, heat=21.0, cool=23.0)
        cool = make_room("cool", 20.3, heat=20.0, cool=23.0)
        assert CentralBangBang().decide(rooms=[warm, cool]) == MODE_CIRCULATING


# ---------------------------------------------------------------------------
# Controller facade
# ---------------------------------------------------------------------------


class TestCentralMPCController:
    """The facade picks the joint MPC only for models that cleared the gates."""

    def test_empty_inputs_yield_an_idle_plan(self) -> None:
        controller = CentralMPCController()
        plan = controller.evaluate(rooms=[], T_outdoor_series=[5.0])
        assert plan.actions == []
        assert plan.house_temperatures == [DEFAULT_HOUSE_TEMP_C]
        no_outdoor = controller.evaluate(rooms=[make_room()], T_outdoor_series=[])
        assert no_outdoor.actions == []
        assert no_outdoor.mpc_active is False

    def test_cold_start_uses_bangbang_and_rolls_trajectories(self) -> None:
        controller = CentralMPCController()
        cold = make_room("a", 19.0, heat=21.0, mpc_ready=False)
        plan = controller.evaluate(rooms=[cold], T_outdoor_series=[5.0] * N_BLOCKS)
        assert plan.current_action() == MODE_HEATING
        assert plan.decision_reason == "bangbang"
        assert plan.mpc_active is False
        assert len(plan.actions) == MIN_HORIZON_BLOCKS
        assert len(plan.room_temperatures["a"]) == MIN_HORIZON_BLOCKS + 1
        assert plan.driving_rooms == ["a"]

    def test_prediction_disabled_still_keeps_the_plan_invariant(self) -> None:
        controller = CentralMPCController()
        cold = make_room("a", 19.0, heat=21.0, mpc_ready=False)
        plan = controller.evaluate(
            rooms=[cold],
            T_outdoor_series=[5.0] * N_BLOCKS,
            prediction_enabled=False,
        )
        assert len(plan.actions) == 1
        assert len(plan.room_temperatures["a"]) == 2
        assert len(plan.house_temperatures) == 2

    def test_control_mode_bangbang_forces_the_fallback(self) -> None:
        controller = CentralMPCController()
        estimator = ready_estimator()
        room = make_room("a", 19.0, heat=21.0, mpc_ready=True)
        plan = controller.evaluate(
            rooms=[room],
            T_outdoor_series=[5.0] * N_BLOCKS,
            settings={"control_mode": "bangbang"},
            estimators={"a": estimator},
        )
        assert plan.decision_reason == "bangbang"
        assert plan.mpc_active is False

    def test_ready_room_uses_the_joint_mpc(self) -> None:
        controller = CentralMPCController()
        estimator = ready_estimator()
        room = make_room("a", 19.0, heat=21.0, mpc_ready=True)
        plan = controller.evaluate(
            rooms=[room],
            T_outdoor_series=[5.0] * N_BLOCKS,
            estimators={"a": estimator},
        )
        assert plan.mpc_active is True
        assert plan.decision_reason in ("mpc", "preheat", "guardrail")
        assert len(plan.actions) == MIN_HORIZON_BLOCKS

    def test_caller_readiness_is_verified_against_the_estimator(self) -> None:
        controller = CentralMPCController()
        room = make_room("a", 19.0, heat=21.0, mpc_ready=True)
        plan = controller.evaluate(
            rooms=[room],
            T_outdoor_series=[5.0] * N_BLOCKS,
            estimators={"a": ThermalEKF()},
        )
        assert plan.decision_reason == "bangbang"

    def test_model_manager_supplies_estimators_and_priors(self) -> None:
        manager = RoomModelManager()
        manager._estimators["a"] = ready_estimator()
        controller = CentralMPCController(manager)
        room = make_room("a", 19.0, heat=21.0, mpc_ready=True)
        plan = controller.evaluate(rooms=[room], T_outdoor_series=[5.0] * N_BLOCKS)
        assert plan.mpc_active is True

    def test_not_ready_room_gets_the_forced_air_priors(self) -> None:
        manager = RoomModelManager()
        manager._estimators["ready"] = ready_estimator()
        controller = CentralMPCController(manager)
        ready_room = make_room("ready", 19.0, heat=21.0, mpc_ready=True)
        cold_room = make_room("cold", 19.0, heat=21.0, mpc_ready=False)
        plan = controller.evaluate(
            rooms=[ready_room, cold_room],
            T_outdoor_series=[5.0] * N_BLOCKS,
        )
        assert plan.mpc_active is True
        assert set(plan.room_temperatures) == {"ready", "cold"}

    def test_legal_actions_and_forced_action_pass_through(self) -> None:
        controller = CentralMPCController()
        estimator = ready_estimator()
        room = make_room("a", 19.0, heat=21.0, mpc_ready=True)
        plan = controller.evaluate(
            rooms=[room],
            T_outdoor_series=[5.0] * N_BLOCKS,
            estimators={"a": estimator},
            legal_actions={MODE_IDLE, MODE_CIRCULATING, MODE_HEATING},
            forced_action=MODE_IDLE,
        )
        assert plan.current_action() == MODE_IDLE
        assert plan.decision_reason == "hold_guard"

    def test_plant_timings_come_from_the_central_settings(self) -> None:
        controller = CentralMPCController()
        estimator = ready_estimator()
        room = make_room("a", 22.0, heat=21.0, cool=23.0, mpc_ready=True)
        plan = controller.evaluate(
            rooms=[room],
            T_outdoor_series=[5.0] * N_BLOCKS,
            settings={"central_plant": {"min_run_minutes": 30}},
            estimators={"a": estimator},
            initial_action=MODE_HEATING,
            initial_run_blocks=0,
        )
        assert plan.actions[:6] == [MODE_HEATING] * 6

    def test_corrupt_central_settings_fall_back_to_defaults(self) -> None:
        controller = CentralMPCController()
        estimator = ready_estimator()
        room = make_room("a", 19.0, heat=21.0, mpc_ready=True)
        plan = controller.evaluate(
            rooms=[room],
            T_outdoor_series=[5.0] * N_BLOCKS,
            settings={"central_plant": {"min_run_minutes": "corrupt"}, "comfort_weight": "bad"},
            estimators={"a": estimator},
        )
        assert plan.mpc_active is True

    def test_climate_mode_gates_reach_the_optimizer(self) -> None:
        controller = CentralMPCController()
        estimator = ready_estimator()
        warm = make_room("a", 25.0, heat=20.0, cool=23.0, mpc_ready=True)
        plan = controller.evaluate(
            rooms=[warm],
            T_outdoor_series=[5.0] * N_BLOCKS,
            estimators={"a": estimator},
            can_heat=False,
        )
        assert plan.current_action() != MODE_HEATING

    def test_priority_zero_rooms_do_not_arm_the_mpc(self) -> None:
        controller = CentralMPCController()
        estimator = ready_estimator()
        room = make_room("a", 19.0, heat=21.0, priority=0.0, occupied=0.0, mpc_ready=True)
        plan = controller.evaluate(
            rooms=[room],
            T_outdoor_series=[5.0] * N_BLOCKS,
            estimators={"a": estimator},
        )
        assert plan.mpc_active is False

    def test_explicit_legal_actions_respect_the_climate_mode(self) -> None:
        controller = CentralMPCController()
        room = make_room("a", 19.0, heat=21.0, mpc_ready=False)
        plan = controller.evaluate(
            rooms=[room],
            T_outdoor_series=[5.0] * N_BLOCKS,
            legal_actions={MODE_IDLE},
        )
        assert plan.current_action() == MODE_IDLE

    def test_bangbang_seeds_the_run_in_progress(self) -> None:
        controller = CentralMPCController()
        room = make_room("a", 20.5, heat=21.0, mpc_ready=False)
        plan = controller.evaluate(
            rooms=[room],
            T_outdoor_series=[5.0] * N_BLOCKS,
            initial_action=MODE_HEATING,
        )
        assert plan.current_action() == MODE_HEATING
