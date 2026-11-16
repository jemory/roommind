"""Tests for the Central HVAC analytics prediction simulator."""

from __future__ import annotations

import time
from typing import Any

import pytest

from custom_components.roommind.const import (
    MODE_CIRCULATING,
    MODE_COOLING,
    MODE_HEATING,
    MODE_IDLE,
)
from custom_components.roommind.control.analytics_simulator import (
    DEFAULT_OUTDOOR_TEMP_FALLBACK,
    OBSERVED_DRIFT_MULTIPLIER,
    WINDOW_OPEN_CEILING_C,
    WINDOW_OPEN_FLOOR_C,
    _cap_idle_drift,
    _climate_capabilities,
    _legal_actions,
    _plan_curve,
    _series_at,
    build_forecast_outdoor_series,
    build_forecast_solar_series,
    build_target_room_input,
    compute_observed_idle_rate,
    simulate_bangbang_prediction,
    simulate_house_prediction,
    simulate_prediction,
    simulate_window_open_prediction,
)
from custom_components.roommind.control.central_mpc import (
    DEFAULT_GUARDRAIL_COOL,
    DEFAULT_GUARDRAIL_HEAT,
    CentralMPCOptimizer,
    CentralMPCPlan,
    RoomHorizonInput,
)
from custom_components.roommind.control.thermal_model import RCModel, ThermalEKF

N_BLOCKS = 12
HEAT_ONLY = {"area_id": "office", "climate_mode": "heat_only"}
COOL_ONLY = {"area_id": "office", "climate_mode": "cool_only"}
AUTO = {"area_id": "office", "climate_mode": "auto"}


def _forecast(
    n_blocks: int = N_BLOCKS,
    heat: float | None = 21.0,
    cool: float | None = 23.0,
    target: float | None = None,
) -> list[dict]:
    if target is None:
        target = (heat + cool) / 2.0 if heat is not None and cool is not None else (heat if heat is not None else cool)
    entries: list[dict] = []
    for _ in range(n_blocks):
        entry: dict[str, Any] = {"target_temp": target}
        if heat is not None:
            entry["heat_target"] = heat
        if cool is not None:
            entry["cool_target"] = cool
        entries.append(entry)
    return entries


def _target(
    *,
    model: RCModel | None = None,
    current_temp: float = 19.5,
    heat: float | None = 21.0,
    cool: float | None = 23.0,
    rh: float | None = None,
    area_id: str = "office",
    blocks: int = N_BLOCKS,
    mpc_ready: bool = True,
    **overrides: Any,
) -> RoomHorizonInput:
    target = build_target_room_input(
        model=model or RCModel(),
        current_temp=current_temp,
        room_config={"area_id": area_id},
        target_forecast=_forecast(blocks, heat=heat, cool=cool),
        current_rh_pct=rh,
        mpc_ready=mpc_ready,
    )
    for name, value in overrides.items():
        setattr(target, name, value)
    return target


def _sibling_room(
    T_init: float = 25.0,
    area_id: str = "living",
    *,
    heat: float = 20.0,
    cool: float = 23.0,
    blocks: int = N_BLOCKS,
) -> RoomHorizonInput:
    heat_series: list[float | None] = [heat] * blocks
    cool_series: list[float | None] = [cool] * blocks
    return RoomHorizonInput(
        area_id=area_id,
        model=RCModel(),
        T_init=T_init,
        priority_series=[1.0] * blocks,
        occupied_prob_series=[1.0] * blocks,
        comfort_heat_series=heat_series,
        comfort_cool_series=cool_series,
    )


def _kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        model=RCModel(),
        estimator=ThermalEKF(),
        target_forecast=_forecast(),
        outdoor_series=[5.0] * N_BLOCKS,
        current_temp=19.5,
        window_open=False,
        mpc_active=True,
        room_config=dict(HEAT_ONLY),
        settings={},
        all_points=[],
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# Outdoor forecast series
# ---------------------------------------------------------------------------


class TestBuildForecastOutdoorSeries:
    """The outdoor series expands hourly forecast entries onto the plan grid."""

    def test_with_forecast_data(self) -> None:
        forecast = [{"temperature": 5.0}, {"temperature": 6.0}]
        assert build_forecast_outdoor_series(forecast, 10.0, 3) == [10.0, 5.0, 5.0]

    def test_without_forecast_uses_the_reading(self) -> None:
        assert build_forecast_outdoor_series(None, 8.0, 4) == [8.0] * 4

    def test_without_forecast_or_reading_uses_the_placeholder(self) -> None:
        assert build_forecast_outdoor_series([], None, 3) == [DEFAULT_OUTDOOR_TEMP_FALLBACK] * 3

    def test_forecast_shorter_than_n_blocks_is_padded(self) -> None:
        assert build_forecast_outdoor_series([{"temperature": 6.0}], None, 15) == [6.0] * 15

    def test_forecast_longer_than_n_blocks_is_truncated(self) -> None:
        forecast = [{"temperature": 3.0}, {"temperature": 4.0}, {"temperature": 9.0}]
        assert build_forecast_outdoor_series(forecast, None, 12) == [3.0] * 12

    def test_hourly_entries_expand_to_blocks(self) -> None:
        forecast = [{"temperature": 1.0}, {"temperature": 2.0}]
        series = build_forecast_outdoor_series(forecast, 3.0, 15)
        assert series[0] == 3.0
        assert series[1:12] == [1.0] * 11
        assert series[12:] == [2.0] * 3

    def test_missing_temperature_key_uses_the_reading(self) -> None:
        forecast: list[dict] = [{"temperature": 5.0}, {"condition": "cloudy"}, {"temperature": 7.0}]
        assert build_forecast_outdoor_series(forecast, 10.0, 3) == [10.0, 5.0, 5.0]

    def test_missing_temperature_key_without_reading_uses_the_placeholder(self) -> None:
        forecast: list[dict] = [{"condition": "cloudy"}]
        assert build_forecast_outdoor_series(forecast, None, 2) == [
            DEFAULT_OUTDOOR_TEMP_FALLBACK,
            DEFAULT_OUTDOOR_TEMP_FALLBACK,
        ]

    def test_none_forecast_same_as_empty(self) -> None:
        assert build_forecast_outdoor_series(None, 8.0, 4) == build_forecast_outdoor_series([], 8.0, 4)

    def test_single_block(self) -> None:
        assert build_forecast_outdoor_series([{"temperature": 3.0}], 4.0, 1) == [4.0]

    def test_zero_blocks(self) -> None:
        assert build_forecast_outdoor_series([{"temperature": 3.0}], 4.0, 0) == []


# ---------------------------------------------------------------------------
# Solar forecast series
# ---------------------------------------------------------------------------


class TestBuildForecastSolarSeries:
    """The solar series expands hourly cloud cover for the solar module."""

    def test_zero_coordinates_return_none(self) -> None:
        assert build_forecast_solar_series(0.0, 0.0, [], 8) is None

    def test_valid_location_returns_one_value_per_block(self) -> None:
        series = build_forecast_solar_series(52.0, 13.0, [], 8)
        assert series is not None
        assert len(series) == 8
        assert all(value >= 0.0 for value in series)

    def test_cloud_coverage_is_expanded(self) -> None:
        from custom_components.roommind.control.solar import build_solar_series

        overcast = build_forecast_solar_series(52.0, 13.0, [{"cloud_coverage": 100}], 8)
        clear = build_forecast_solar_series(52.0, 13.0, [{"cloud_coverage": 0}], 8)
        assert overcast is not None and clear is not None
        # One hourly overcast entry is expanded to 12 plan blocks before the solar module runs.
        expected = build_solar_series(52.0, 13.0, 8, 5.0, cloud_series=[100.0] * 8)
        assert overcast == pytest.approx(expected)
        assert overcast[0] <= clear[0]

    def test_shading_factor_scales_the_series(self) -> None:
        full = build_forecast_solar_series(52.0, 13.0, [], 6)
        shaded = build_forecast_solar_series(52.0, 13.0, [], 6, shading_factor=0.5)
        assert full is not None and shaded is not None
        assert shaded[0] == pytest.approx(full[0] * 0.5)

    def test_short_cloud_series_is_padded(self) -> None:
        series = build_forecast_solar_series(52.0, 13.0, [{"cloud_coverage": 50}], 20)
        assert series is not None
        assert len(series) == 20


# ---------------------------------------------------------------------------
# Observed idle rate
# ---------------------------------------------------------------------------


class TestComputeObservedIdleRate:
    """The idle-rate cap reads recent idle points only."""

    def test_empty_points_return_none(self) -> None:
        assert compute_observed_idle_rate([]) is None

    def test_insufficient_idle_points_return_none(self) -> None:
        now = time.time()
        points = [{"ts": now - 100, "room_temp": 20.0, "mode": MODE_IDLE}]
        assert compute_observed_idle_rate(points) is None

    def test_no_idle_points_return_none(self) -> None:
        now = time.time()
        points = [
            {"ts": now - 600, "room_temp": 20.0, "mode": MODE_HEATING},
            {"ts": now - 300, "room_temp": 21.0, "mode": MODE_HEATING},
        ]
        assert compute_observed_idle_rate(points) is None

    def test_normal_idle_rate(self) -> None:
        now = time.time()
        points = [
            {"ts": now - 600, "room_temp": 21.0, "mode": MODE_IDLE},
            {"ts": now - 300, "room_temp": 20.5, "mode": MODE_IDLE},
            {"ts": now, "room_temp": 20.0, "mode": MODE_IDLE},
        ]
        rate = compute_observed_idle_rate(points)
        assert rate is not None
        assert rate == pytest.approx(-0.5, abs=0.01)

    def test_rising_idle_rate(self) -> None:
        now = time.time()
        points = [
            {"ts": now - 600, "room_temp": 20.0, "mode": MODE_IDLE},
            {"ts": now, "room_temp": 21.0, "mode": MODE_IDLE},
        ]
        rate = compute_observed_idle_rate(points)
        assert rate is not None
        assert rate > 0.0

    def test_points_older_than_an_hour_are_ignored(self) -> None:
        now = time.time()
        points = [
            {"ts": now - 7200, "room_temp": 25.0, "mode": MODE_IDLE},
            {"ts": now - 300, "room_temp": 20.0, "mode": MODE_IDLE},
        ]
        assert compute_observed_idle_rate(points) is None

    def test_empty_mode_counts_as_idle(self) -> None:
        now = time.time()
        points = [
            {"ts": now - 600, "room_temp": 21.0, "mode": ""},
            {"ts": now, "room_temp": 20.0, "mode": ""},
        ]
        assert compute_observed_idle_rate(points) is not None

    def test_very_close_timestamps_return_none(self) -> None:
        now = time.time()
        points = [
            {"ts": now - 30, "room_temp": 20.0, "mode": MODE_IDLE},
            {"ts": now, "room_temp": 19.9, "mode": MODE_IDLE},
        ]
        assert compute_observed_idle_rate(points) is None


# ---------------------------------------------------------------------------
# Target room input
# ---------------------------------------------------------------------------


class TestBuildTargetRoomInput:
    """The analytics feed maps onto a RoomHorizonInput without losing detail."""

    def test_maps_targets_and_room_config(self) -> None:
        forecast = [
            {"target_temp": 21.0, "heat_target": 20.0, "cool_target": 23.0},
            {"target_temp": 22.0},
        ]
        room = build_target_room_input(
            model=RCModel(),
            current_temp=19.5,
            room_config={
                "id": "study",
                "door_mix_factor": 0.4,
                "door_supply_factor": 0.75,
                "guardrail_heat": 15.0,
                "guardrail_cool": 27.0,
                "preheat_enabled": False,
                "include_in_house_mean": False,
            },
            target_forecast=forecast,
            solar_series=[0.5, 0.6],
            q_occupancy=1.0,
            q_rebound=-0.4,
            current_rh_pct=55.0,
            mpc_ready=False,
        )
        assert room.area_id == "study"
        assert room.T_init == 19.5
        assert room.rh_init == 55.0
        assert room.rebound_init == -0.4
        assert room.door_mix_factor == 0.4
        assert room.door_supply_factor == 0.75
        assert room.guardrail_heat == 15.0
        assert room.guardrail_cool == 27.0
        assert room.preheat_enabled is False
        assert room.include_in_house_mean is False
        assert room.comfort_heat_series == [20.0, 22.0]
        assert room.comfort_cool_series == [23.0, 22.0]
        assert room.priority_series == [1.0, 1.0]
        assert room.occupied_prob_series == [1.0, 1.0]
        assert room.solar_series == [0.5, 0.6]
        assert room.occupancy_gain_series == [1.0, 1.0]
        assert room.mpc_ready is False

    def test_defaults_when_the_room_config_is_sparse(self) -> None:
        room = build_target_room_input(
            model=RCModel(),
            current_temp=20.0,
            room_config={},
            target_forecast=[{"target_temp": 21.0}],
        )
        assert room.area_id == "target"
        assert room.guardrail_heat == DEFAULT_GUARDRAIL_HEAT
        assert room.guardrail_cool == DEFAULT_GUARDRAIL_COOL
        assert room.rh_init is None
        assert room.rebound_init == 0.0
        assert room.solar_series == []
        assert room.preheat_enabled is True
        assert room.include_in_house_mean is True
        assert room.mpc_ready is True


# ---------------------------------------------------------------------------
# Climate capabilities and legal actions
# ---------------------------------------------------------------------------


class TestClimateCapabilities:
    """The room's climate mode and an outdoor room decide what is possible."""

    def test_auto(self) -> None:
        assert _climate_capabilities({"climate_mode": "auto"}) == (True, True)

    def test_heat_only(self) -> None:
        assert _climate_capabilities({"climate_mode": "heat_only"}) == (True, False)

    def test_cool_only(self) -> None:
        assert _climate_capabilities({"climate_mode": "cool_only"}) == (False, True)

    def test_outdoor_room_is_not_conditioned(self) -> None:
        assert _climate_capabilities({"climate_mode": "auto", "is_outdoor": True}) == (False, False)


class TestLegalActions:
    """Outdoor gates, overrides and climate modes shape the plant action set."""

    def test_auto_room_has_every_mode(self) -> None:
        legal = _legal_actions(dict(AUTO), {}, 18.0, True)
        assert legal == {MODE_IDLE, MODE_HEATING, MODE_COOLING, MODE_CIRCULATING}

    def test_hot_outdoor_air_gates_heating(self) -> None:
        legal = _legal_actions(dict(AUTO), {"outdoor_heating_max": 22.0}, 25.0, True)
        assert MODE_HEATING not in legal
        assert MODE_COOLING in legal

    def test_cold_outdoor_air_gates_cooling(self) -> None:
        legal = _legal_actions(dict(AUTO), {"outdoor_cooling_min": 16.0}, 10.0, True)
        assert MODE_COOLING not in legal
        assert MODE_HEATING in legal

    def test_unknown_outdoor_reading_gates_nothing(self) -> None:
        legal = _legal_actions(dict(AUTO), {}, 30.0, False)
        assert MODE_HEATING in legal
        assert MODE_COOLING in legal

    def test_override_bypasses_the_gates(self) -> None:
        legal = _legal_actions({"climate_mode": "auto", "override_heat": 21.0}, {}, 10.0, True)
        assert MODE_COOLING in legal

    def test_heat_only_room_never_cools(self) -> None:
        legal = _legal_actions(dict(HEAT_ONLY), {}, 20.0, True)
        assert MODE_COOLING not in legal
        assert MODE_HEATING in legal

    def test_outdoor_room_only_idles(self) -> None:
        legal = _legal_actions({"climate_mode": "auto", "is_outdoor": True}, {}, 20.0, True)
        assert legal == {MODE_IDLE, MODE_CIRCULATING}


# ---------------------------------------------------------------------------
# Window-open path
# ---------------------------------------------------------------------------


class TestSimulateWindowOpenPrediction:
    """With an Exterior Opening open the chart shows pure heat exchange."""

    def test_cools_toward_the_outdoor_air(self) -> None:
        model = RCModel(C=1.0, U=0.1, Q_heat=1.0, Q_cool=1.0, Q_solar=0.0)
        predictions = simulate_window_open_prediction(
            model=model,
            estimator=ThermalEKF(),
            target_forecast=_forecast(),
            outdoor_series=[5.0] * N_BLOCKS,
            current_temp=21.0,
        )
        assert len(predictions) == N_BLOCKS
        assert predictions[0] < 21.0
        assert predictions[-1] < predictions[0]

    def test_warms_when_the_outdoor_air_is_hotter(self) -> None:
        model = RCModel(C=1.0, U=0.1, Q_heat=1.0, Q_cool=1.0, Q_solar=0.0)
        predictions = simulate_window_open_prediction(
            model=model,
            estimator=ThermalEKF(),
            target_forecast=_forecast(),
            outdoor_series=[35.0] * N_BLOCKS,
            current_temp=21.0,
        )
        assert predictions[0] > 21.0

    def test_uses_the_learned_window_multiplier(self) -> None:
        model = RCModel(C=1.0, U=0.1, Q_heat=1.0, Q_cool=1.0, Q_solar=0.0)
        estimator = ThermalEKF()
        estimate_window = simulate_window_open_prediction(
            model=model,
            estimator=estimator,
            target_forecast=_forecast(3),
            outdoor_series=[5.0] * 3,
            current_temp=21.0,
        )
        estimator._k_window = 1.0
        small_window = simulate_window_open_prediction(
            model=model,
            estimator=estimator,
            target_forecast=_forecast(3),
            outdoor_series=[5.0] * 3,
            current_temp=21.0,
        )
        assert small_window[0] > estimate_window[0]

    def test_temperatures_stay_inside_the_plausible_band(self) -> None:
        model = RCModel(C=1.0, U=10.0, Q_heat=1.0, Q_cool=1.0, Q_solar=0.0)
        cold = simulate_window_open_prediction(
            model=model,
            estimator=ThermalEKF(),
            target_forecast=_forecast(3),
            outdoor_series=[-20.0] * 3,
            current_temp=20.0,
        )
        hot = simulate_window_open_prediction(
            model=model,
            estimator=ThermalEKF(),
            target_forecast=_forecast(3),
            outdoor_series=[50.0] * 3,
            current_temp=20.0,
        )
        assert all(value >= WINDOW_OPEN_FLOOR_C for value in cold)
        assert all(value <= WINDOW_OPEN_CEILING_C for value in hot)


# ---------------------------------------------------------------------------
# One-room / whole-house Central MPC forecast
# ---------------------------------------------------------------------------


class TestSimulateHousePrediction:
    """The joint rollout returns the target room's Central MPC curve."""

    def test_one_room_curve_follows_the_plan(self) -> None:
        target = _target(current_temp=19.5)
        curve = simulate_house_prediction(
            target=target,
            siblings=[],
            outdoor_series=[5.0] * N_BLOCKS,
            settings={},
        )
        assert curve is not None
        assert len(curve) == N_BLOCKS
        assert all(value == round(value, 2) for value in curve)
        assert curve[0] > 19.5

    def test_empty_forecast_cannot_be_simulated(self) -> None:
        target = _target(blocks=0)
        assert simulate_house_prediction(target=target, siblings=[], outdoor_series=[5.0], settings={}) is None

    def test_missing_outdoor_series_cannot_be_simulated(self) -> None:
        target = _target(blocks=1)
        assert simulate_house_prediction(target=target, siblings=[], outdoor_series=[], settings={}) is None

    def test_duplicate_sibling_is_ignored(self) -> None:
        target = _target()
        curve = simulate_house_prediction(
            target=target,
            siblings=[_sibling_room(25.0, area_id="office"), _sibling_room(23.0, area_id="hall")],
            outdoor_series=[5.0] * N_BLOCKS,
            settings={},
        )
        assert curve is not None
        assert len(curve) == N_BLOCKS

    def test_short_outdoor_series_is_padded(self) -> None:
        target = _target()
        curve = simulate_house_prediction(
            target=target,
            siblings=[_sibling_room()],
            outdoor_series=[5.0] * 3,
            settings={},
        )
        assert curve is not None
        assert len(curve) == N_BLOCKS
        assert curve[-1] == curve[2]

    def test_plant_and_outdoor_settings_reach_the_optimizer(self) -> None:
        target = _target()
        curve = simulate_house_prediction(
            target=target,
            siblings=[_sibling_room()],
            outdoor_series=[5.0] * N_BLOCKS,
            settings={
                "comfort_weight": 90,
                "outdoor_cooling_min": 10.0,
                "outdoor_heating_max": 18.0,
                "central_plant": {
                    "humidity_comfort_enabled": False,
                    "dehumidify_rh_threshold": 55.0,
                    "max_dehumidify_overcool_c": 0.4,
                },
            },
        )
        assert curve is not None
        assert len(curve) == N_BLOCKS

    def test_a_run_in_progress_is_held(self) -> None:
        target = _target(current_temp=20.9)
        cold = simulate_house_prediction(
            target=target,
            siblings=[],
            outdoor_series=[5.0] * N_BLOCKS,
            settings={},
        )
        holding = simulate_house_prediction(
            target=target,
            siblings=[],
            outdoor_series=[5.0] * N_BLOCKS,
            settings={},
            previous_mode=MODE_HEATING,
            mode_elapsed_minutes=5.0,
        )
        assert cold is not None and holding is not None
        assert cold[0] < 20.9
        assert holding[0] > 20.9

    def test_climate_mode_gates_the_plan(self) -> None:
        cooling = _target(current_temp=24.5, heat=20.0, cool=23.0)
        gated = simulate_house_prediction(
            target=cooling,
            siblings=[],
            outdoor_series=[30.0] * N_BLOCKS,
            settings={},
            can_cool=False,
        )
        assert gated is not None
        assert gated[0] > 24.5

    def test_override_bypasses_the_outdoor_gates(self) -> None:
        cooling = _target(current_temp=24.5, heat=20.0, cool=23.0)
        gated = simulate_house_prediction(
            target=cooling,
            siblings=[],
            outdoor_series=[10.0] * N_BLOCKS,
            settings={},
        )
        overridden = simulate_house_prediction(
            target=cooling,
            siblings=[],
            outdoor_series=[10.0] * N_BLOCKS,
            settings={},
            override_active=True,
        )
        assert gated is not None and overridden is not None
        assert overridden[0] < gated[0]


# ---------------------------------------------------------------------------
# Cold-start bang-bang forecast
# ---------------------------------------------------------------------------


class TestSimulateBangbangPrediction:
    """The cold-start forecast re-decides every block with CentralBangBang."""

    def test_heating_raises_the_temperature(self) -> None:
        curve = simulate_bangbang_prediction(
            target=_target(current_temp=19.5),
            outdoor_series=[5.0] * N_BLOCKS,
            settings={},
            room_config=dict(HEAT_ONLY),
        )
        assert len(curve) == N_BLOCKS
        assert curve[0] > 19.5
        assert all(value == round(value, 2) for value in curve)

    def test_cooling_lowers_the_temperature(self) -> None:
        curve = simulate_bangbang_prediction(
            target=_target(current_temp=26.0, heat=20.0, cool=23.0),
            outdoor_series=[32.0] * N_BLOCKS,
            settings={},
            room_config=dict(COOL_ONLY),
        )
        assert curve[0] < 26.0

    def test_heat_only_room_never_cools(self) -> None:
        # A warm room with a hot outdoor reading would cool under AUTO: HEAT_ONLY may not.
        curve = simulate_bangbang_prediction(
            target=_target(current_temp=26.0, heat=20.0, cool=23.0),
            outdoor_series=[30.0] * N_BLOCKS,
            settings={},
            room_config=dict(HEAT_ONLY),
        )
        assert curve[0] > 26.0

    def test_cool_only_room_never_heats(self) -> None:
        # A cold room with a warm outdoor reading would heat under AUTO: COOL_ONLY may not.
        curve = simulate_bangbang_prediction(
            target=_target(current_temp=17.0, heat=21.0, cool=26.0),
            outdoor_series=[20.0] * N_BLOCKS,
            settings={},
            room_config=dict(COOL_ONLY),
        )
        assert curve[0] > 17.0

    def test_hot_outdoor_air_gates_heating(self) -> None:
        gated = simulate_bangbang_prediction(
            target=_target(current_temp=19.0),
            outdoor_series=[25.0] * N_BLOCKS,
            settings={},
            room_config=dict(AUTO),
        )
        # Heating is gated by the hot outdoor air, so the room drifts towards it.
        assert gated[0] > 19.0

    def test_override_bypasses_the_outdoor_gates(self) -> None:
        overridden = simulate_bangbang_prediction(
            target=_target(current_temp=19.0),
            outdoor_series=[25.0] * N_BLOCKS,
            settings={},
            room_config={"climate_mode": "auto", "override_heat": 21.0},
        )
        assert overridden[0] > 19.0

    def test_unknown_outdoor_reading_gates_nothing(self) -> None:
        curve = simulate_bangbang_prediction(
            target=_target(current_temp=19.0),
            outdoor_series=[25.0] * N_BLOCKS,
            settings={},
            room_config=dict(AUTO),
            outdoor_known=False,
        )
        assert curve[0] > 19.0

    def test_a_run_in_progress_continues(self) -> None:
        target = _target(current_temp=20.5)
        fresh = simulate_bangbang_prediction(
            target=target,
            outdoor_series=[5.0] * N_BLOCKS,
            settings={},
            room_config=dict(HEAT_ONLY),
        )
        running = simulate_bangbang_prediction(
            target=target,
            outdoor_series=[5.0] * N_BLOCKS,
            settings={},
            room_config=dict(HEAT_ONLY),
            previous_mode=MODE_HEATING,
        )
        assert fresh[0] > 20.5
        assert running[0] > 20.5

    def test_idle_drift_is_capped_by_the_observed_rate(self) -> None:
        model = RCModel(C=1.0, U=0.5, Q_heat=0.0, Q_cool=0.0, Q_solar=0.0)
        target = _target(model=model, current_temp=21.0, heat=21.0, cool=23.0)
        now = time.time()
        points: list[dict] = [
            {"ts": now - 600, "room_temp": 21.0, "mode": MODE_IDLE},
            {"ts": now, "room_temp": 20.8, "mode": MODE_IDLE},
        ]
        capped = simulate_bangbang_prediction(
            target=target,
            outdoor_series=[5.0] * N_BLOCKS,
            settings={},
            room_config=dict(HEAT_ONLY),
            all_points=points,
        )
        uncapped = simulate_bangbang_prediction(
            target=target,
            outdoor_series=[5.0] * N_BLOCKS,
            settings={},
            room_config=dict(HEAT_ONLY),
        )
        observed_rate = compute_observed_idle_rate(points)
        assert observed_rate is not None
        assert capped[0] == pytest.approx(21.0 + observed_rate * OBSERVED_DRIFT_MULTIPLIER)
        assert uncapped[0] < capped[0]

    def test_a_forced_air_charge_warms_the_idle_forecast(self) -> None:
        model = RCModel(C=1.0, U=0.02, Q_heat=3.0, Q_cool=2.0, Q_fast=20.0)
        target = _target(model=model, current_temp=24.0, heat=20.0, cool=30.0, rh=None)
        target.rebound_init = -0.8
        charged = simulate_bangbang_prediction(
            target=target,
            outdoor_series=[24.0] * N_BLOCKS,
            settings={},
            room_config=dict(COOL_ONLY),
        )
        target.rebound_init = 0.0
        plain = simulate_bangbang_prediction(
            target=target,
            outdoor_series=[24.0] * N_BLOCKS,
            settings={},
            room_config=dict(COOL_ONLY),
        )
        assert charged[0] > plain[0]

    def test_humidity_inside_the_dry_bulb_band_cools(self) -> None:
        target = _target(current_temp=23.5, heat=21.0, cool=24.0, rh=70.0)
        humid = simulate_bangbang_prediction(
            target=target,
            outdoor_series=[30.0] * N_BLOCKS,
            settings={},
            room_config=dict(COOL_ONLY),
        )
        target.rh_init = None
        dry = simulate_bangbang_prediction(
            target=target,
            outdoor_series=[30.0] * N_BLOCKS,
            settings={},
            room_config=dict(COOL_ONLY),
        )
        assert humid[0] < 23.5
        assert dry[0] > 23.5

    def test_guardrail_breach_heats_an_empty_room(self) -> None:
        target = _target(current_temp=15.5, heat=21.0, cool=24.0)
        target.priority_series = [0.0] * N_BLOCKS
        target.occupied_prob_series = [0.0] * N_BLOCKS
        curve = simulate_bangbang_prediction(
            target=target,
            outdoor_series=[5.0] * N_BLOCKS,
            settings={},
            room_config=dict(HEAT_ONLY),
        )
        assert curve[0] > 15.5

    def test_empty_forecast_returns_no_predictions(self) -> None:
        target = _target(blocks=0)
        assert (
            simulate_bangbang_prediction(
                target=target,
                outdoor_series=[5.0],
                settings={},
                room_config=dict(HEAT_ONLY),
            )
            == []
        )

    def test_no_outdoor_horizon_returns_no_predictions(self) -> None:
        assert (
            simulate_bangbang_prediction(
                target=_target(),
                outdoor_series=[],
                settings={},
                room_config=dict(HEAT_ONLY),
            )
            == []
        )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestSimulatePredictionDispatch:
    """simulate_prediction routes the request to the right Central HVAC path."""

    def test_mpc_path_matches_the_one_room_rollout(self) -> None:
        kwargs = _kwargs()
        expected = simulate_house_prediction(
            target=build_target_room_input(
                model=kwargs["model"],
                current_temp=kwargs["current_temp"],
                room_config=kwargs["room_config"],
                target_forecast=kwargs["target_forecast"],
                mpc_ready=True,
            ),
            siblings=[],
            outdoor_series=kwargs["outdoor_series"],
            settings=kwargs["settings"],
            horizon_blocks=N_BLOCKS,
        )
        assert simulate_prediction(**kwargs) == expected

    def test_bangbang_path_matches_the_cold_start_rollout(self) -> None:
        kwargs = _kwargs(mpc_active=False)
        expected = simulate_bangbang_prediction(
            target=build_target_room_input(
                model=kwargs["model"],
                current_temp=kwargs["current_temp"],
                room_config=kwargs["room_config"],
                target_forecast=kwargs["target_forecast"],
                mpc_ready=False,
            ),
            outdoor_series=kwargs["outdoor_series"],
            settings=kwargs["settings"],
            room_config=kwargs["room_config"],
            horizon_blocks=N_BLOCKS,
        )
        assert simulate_prediction(**kwargs) == expected

    def test_window_open_takes_precedence(self) -> None:
        kwargs = _kwargs(window_open=True)
        assert simulate_prediction(**kwargs) == simulate_window_open_prediction(
            model=kwargs["model"],
            estimator=kwargs["estimator"],
            target_forecast=kwargs["target_forecast"],
            outdoor_series=kwargs["outdoor_series"],
            current_temp=kwargs["current_temp"],
        )

    def test_window_open_ignores_siblings(self) -> None:
        kwargs = _kwargs(window_open=True)
        assert simulate_prediction(**kwargs, sibling_rooms=[_sibling_room(25.0)]) == simulate_prediction(**kwargs)

    def test_joint_house_curve_uses_the_siblings(self) -> None:
        single = simulate_prediction(**_kwargs())
        joint = simulate_prediction(**_kwargs(), sibling_rooms=[_sibling_room(25.0)])
        assert len(joint) == len(single)
        assert joint != single

    def test_joint_house_curve_mixes_toward_a_warm_sibling(self) -> None:
        joint = simulate_prediction(**_kwargs(), sibling_rooms=[_sibling_room(25.0)])
        assert joint[0] > 19.5

    def test_bangbang_ignores_siblings(self) -> None:
        kwargs = _kwargs(mpc_active=False)
        with_siblings = simulate_prediction(**kwargs, sibling_rooms=[_sibling_room(25.0)])
        assert with_siblings == simulate_prediction(**kwargs)

    def test_empty_sibling_list_keeps_the_one_room_path(self) -> None:
        assert simulate_prediction(**_kwargs(), sibling_rooms=[]) == simulate_prediction(**_kwargs())

    def test_empty_forecast_returns_no_predictions(self) -> None:
        assert simulate_prediction(**_kwargs(target_forecast=[])) == []

    def test_missing_plan_falls_back_to_the_cold_start_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_optimize(self: CentralMPCOptimizer) -> CentralMPCPlan:
            return CentralMPCPlan(actions=[], room_temperatures={}, house_temperatures=[20.0])

        monkeypatch.setattr(CentralMPCOptimizer, "optimize", fake_optimize)
        kwargs = _kwargs()
        expected = simulate_bangbang_prediction(
            target=build_target_room_input(
                model=kwargs["model"],
                current_temp=kwargs["current_temp"],
                room_config=kwargs["room_config"],
                target_forecast=kwargs["target_forecast"],
                mpc_ready=True,
            ),
            outdoor_series=kwargs["outdoor_series"],
            settings=kwargs["settings"],
            room_config=kwargs["room_config"],
            horizon_blocks=N_BLOCKS,
        )
        assert simulate_prediction(**kwargs) == expected

    def test_humidity_seeds_the_cold_start_curve(self) -> None:
        kwargs = _kwargs(
            current_temp=23.5,
            target_forecast=_forecast(heat=21.0, cool=24.0),
            outdoor_series=[30.0] * N_BLOCKS,
            mpc_active=False,
            room_config=dict(COOL_ONLY),
        )
        humid = simulate_prediction(**kwargs, current_rh_pct=70.0)
        dry = simulate_prediction(**kwargs)
        assert humid[0] < 23.5
        assert dry[0] > 23.5

    def test_previous_mode_seeds_the_mpc_plan(self) -> None:
        kwargs = _kwargs(current_temp=20.9)
        idle = simulate_prediction(**kwargs)
        running = simulate_prediction(**kwargs, previous_mode=MODE_HEATING, mode_elapsed_minutes=5.0)
        assert idle[0] < 20.9
        assert running[0] > 20.9

    def test_charge_seeds_the_curve(self) -> None:
        model = RCModel(C=1.0, U=0.02, Q_heat=3.0, Q_cool=2.0, Q_fast=20.0)
        kwargs = _kwargs(
            model=model,
            current_temp=24.0,
            target_forecast=_forecast(heat=20.0, cool=30.0),
            outdoor_series=[24.0] * N_BLOCKS,
            mpc_active=False,
            room_config=dict(COOL_ONLY),
        )
        charged = simulate_prediction(**kwargs, q_rebound=-0.8)
        plain = simulate_prediction(**kwargs)
        assert charged[0] > plain[0]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


class TestSeriesAt:
    """The series helper holds its last value past the end."""

    def test_empty_series(self) -> None:
        assert _series_at([], 3, 7.5) == 7.5

    def test_index_inside(self) -> None:
        assert _series_at([1.0, 2.0], 0, 0.0) == 1.0
        assert _series_at([1.0, 2.0], 1, 0.0) == 2.0

    def test_index_past_the_end(self) -> None:
        assert _series_at([1.0, 2.0], 9, 0.0) == 2.0


class TestPlanCurve:
    """The plan curve slices, pads and rejects an unusable trajectory."""

    def test_missing_room(self) -> None:
        plan = CentralMPCPlan(actions=[], room_temperatures={}, house_temperatures=[20.0])
        assert _plan_curve(plan, "office", 4) is None

    def test_initial_value_only(self) -> None:
        plan = CentralMPCPlan(actions=[], room_temperatures={"office": [20.0]}, house_temperatures=[20.0])
        assert _plan_curve(plan, "office", 4) is None

    def test_pads_and_truncates(self) -> None:
        plan = CentralMPCPlan(
            actions=[MODE_IDLE] * 2,
            room_temperatures={"office": [20.0, 20.1, 20.2]},
            house_temperatures=[20.0, 20.1, 20.2],
        )
        assert _plan_curve(plan, "office", 4) == [20.1, 20.2, 20.2, 20.2]
        assert _plan_curve(plan, "office", 1) == [20.1]

    def test_rounds_values(self) -> None:
        plan = CentralMPCPlan(
            actions=[MODE_IDLE],
            room_temperatures={"office": [20.0, 20.123456]},
            house_temperatures=[20.0, 20.12],
        )
        assert _plan_curve(plan, "office", 1) == [20.12]


class TestCapIdleDrift:
    """The idle-drift cap never lets the chart outrun the observed rate."""

    def test_no_observed_rate(self) -> None:
        assert _cap_idle_drift(20.0, 19.0, None) == 19.0

    def test_caps_a_drop(self) -> None:
        assert _cap_idle_drift(20.0, 19.0, -0.1) == pytest.approx(19.8)

    def test_caps_a_rise(self) -> None:
        assert _cap_idle_drift(20.0, 21.0, 0.1) == pytest.approx(20.2)

    def test_inside_the_cap_is_unchanged(self) -> None:
        assert _cap_idle_drift(20.0, 19.95, -0.1) == 19.95

    def test_opposite_direction_is_unchanged(self) -> None:
        assert _cap_idle_drift(20.0, 20.05, -0.1) == 20.05
