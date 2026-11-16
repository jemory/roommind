"""Tests for analytics_service.py."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.util import dt as dt_util

from custom_components.roommind.control.thermal_model import ThermalEKF
from custom_components.roommind.services.analytics_service import (
    _ALPHA_HOUSE_STATE,
    _BETA_FAN_STATE,
    _FAST_STATE,
    _OCCUPANCY_STATE,
    _SOLAR_STATE,
    DEFAULT_OUTDOOR_TEMP_FALLBACK,
    _central_can_heat_cool,
    _compute_target_forecast,
    _csv_to_points,
    _current_humidity,
    _observed,
    _occupancy_trend,
    _rebound_charge,
    _safe_float,
    _safe_int,
    build_analytics_data,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _detail_row(ts: float | str, **kw) -> dict:
    row = {
        "timestamp": str(ts),
        "room_temp": "20.0",
        "outdoor_temp": "10",
        "target_temp": "21",
        "mode": "idle",
        "predicted_temp": "",
        "window_open": "",
        "heating_power": "",
    }
    row.update(kw)
    return row


def _estimator() -> ThermalEKF:
    est = ThermalEKF()
    est._n_updates = 200
    est._n_idle = 120
    est._n_heating = 60
    est._n_cooling = 20
    est._n_circulating = 30
    return est


def _make_coordinator(
    *,
    area_id: str = "room1",
    estimator: ThermalEKF | None = None,
    detail_rows: list[dict] | None = None,
    history_rows: list[dict] | None = None,
    rooms: dict | None = None,
    outdoor_temp: float | None = 18.0,
    forecast: list[dict] | None = None,
    humidity: float | None = None,
    charge: float | None = None,
    occupancy_manager: object | None = None,
) -> MagicMock:
    """A coordinator with the attributes build_analytics_data reads."""
    now = time.time()
    est = estimator
    coordinator = MagicMock()
    coordinator._model_manager._estimators = {area_id: est} if est else {}
    coordinator._model_manager.get_model.return_value = est.get_model() if est else MagicMock()
    coordinator.outdoor_temp_effective = outdoor_temp
    live = {"current_humidity": humidity} if humidity is not None else {}
    coordinator.rooms = {area_id: live} if rooms is None else rooms
    coordinator._weather_manager._outdoor_forecast = forecast if forecast is not None else []
    coordinator._current_q_solar = 0.5
    coordinator.price_signal = None
    coordinator._schedule_blocks_cache = None
    coordinator._previous_modes = {}
    coordinator._mode_on_since = {}
    coordinator._window_manager._paused = {}
    coordinator._occupancy_manager = occupancy_manager
    if charge is None:
        coordinator._rebound_tracker = None
    else:
        coordinator._rebound_tracker.charge.return_value = charge

    history_store = MagicMock()
    history_store.read_detail.return_value = detail_rows if detail_rows is not None else [_detail_row(now - 60)]
    history_store.read_history.return_value = history_rows if history_rows is not None else []
    coordinator._history_store = history_store
    return coordinator


def _run_executor(hass: MagicMock) -> None:
    async def mock_executor(fn, *args, **kwargs):  # noqa: ANN001, ANN202
        return fn(*args, **kwargs)

    hass.async_add_executor_job = mock_executor


def _forecast_point(now: float) -> dict:
    return {"ts": now, "target_temp": 21.0, "heat_target": 21.0, "cool_target": 24.0}


def _mock_forecast(value: float = 21.0) -> MagicMock:
    mock = MagicMock(return_value=[value])
    return mock


# ---------------------------------------------------------------------------
# _compute_target_forecast -- mold delta with heat_target=None
# ---------------------------------------------------------------------------


class TestComputeTargetForecast:
    """Tests for _compute_target_forecast edge cases."""

    @pytest.mark.asyncio
    async def test_mold_delta_applied_when_heat_target_none(self):
        """When heat_target is None and mold_prevention_delta > 0, eco_heat + delta used."""
        hass = MagicMock()
        hass.config.units.temperature_unit = "°C"
        room = {
            "comfort_heat": 21.0,
            "comfort_cool": 24.0,
            "eco_heat": 17.0,
            "eco_cool": 27.0,
            "climate_mode": "auto",
        }
        settings = {}

        from custom_components.roommind.const import TargetTemps

        with (
            patch(
                "custom_components.roommind.utils.presence_utils.is_presence_away",
                return_value=False,
            ),
            patch(
                "custom_components.roommind.utils.schedule_utils.get_active_schedule_entity",
                return_value=None,
            ),
            patch(
                "custom_components.roommind.utils.schedule_utils.resolve_targets_at_time",
                return_value=TargetTemps(heat=None, cool=24.0),
            ),
        ):
            result = await _compute_target_forecast(
                hass,
                room,
                settings,
                mold_prevention_delta=2.0,
                hours=0.0,
                interval_minutes=5,
            )
            assert len(result) == 1
            assert result[0]["heat_target"] == round(17.0 + 2.0, 1)

    @pytest.mark.asyncio
    async def test_cool_only_mode_returns_cool_target(self):
        """climate_mode=cool_only -> target = cool_target."""
        hass = MagicMock()
        hass.config.units.temperature_unit = "°C"
        room = {
            "comfort_heat": 21.0,
            "comfort_cool": 24.0,
            "eco_heat": 17.0,
            "eco_cool": 27.0,
            "climate_mode": "cool_only",
        }
        settings = {}

        from custom_components.roommind.const import TargetTemps

        with (
            patch(
                "custom_components.roommind.utils.presence_utils.is_presence_away",
                return_value=False,
            ),
            patch(
                "custom_components.roommind.utils.schedule_utils.get_active_schedule_entity",
                return_value=None,
            ),
            patch(
                "custom_components.roommind.utils.schedule_utils.resolve_targets_at_time",
                return_value=TargetTemps(heat=21.0, cool=24.0),
            ),
        ):
            result = await _compute_target_forecast(
                hass,
                room,
                settings,
                hours=0.0,
                interval_minutes=5,
            )
            assert len(result) == 1
            assert result[0]["target_temp"] == 24.0

    @pytest.mark.asyncio
    async def test_heat_only_mode_returns_heat_target(self):
        """climate_mode=heat_only -> target = heat_target."""
        hass = MagicMock()
        hass.config.units.temperature_unit = "°C"
        room = {
            "comfort_heat": 21.0,
            "comfort_cool": 24.0,
            "eco_heat": 17.0,
            "eco_cool": 27.0,
            "climate_mode": "heat_only",
        }
        settings = {}

        from custom_components.roommind.const import TargetTemps

        with (
            patch(
                "custom_components.roommind.utils.presence_utils.is_presence_away",
                return_value=False,
            ),
            patch(
                "custom_components.roommind.utils.schedule_utils.get_active_schedule_entity",
                return_value=None,
            ),
            patch(
                "custom_components.roommind.utils.schedule_utils.resolve_targets_at_time",
                return_value=TargetTemps(heat=21.0, cool=24.0),
            ),
        ):
            result = await _compute_target_forecast(
                hass,
                room,
                settings,
                hours=0.0,
                interval_minutes=5,
            )
            assert len(result) == 1
            assert result[0]["target_temp"] == 21.0

    @pytest.mark.asyncio
    async def test_override_targets_passed_to_resolver(self):
        """override_heat/override_cool from the room flow into the resolver call."""
        hass = MagicMock()
        hass.config.units.temperature_unit = "°C"
        room = {
            "comfort_heat": 21.0,
            "comfort_cool": 24.0,
            "eco_heat": 17.0,
            "eco_cool": 27.0,
            "climate_mode": "auto",
            "override_heat": 19.5,
            "override_cool": 23.5,
            "override_until": None,
        }
        settings = {}

        from custom_components.roommind.const import TargetTemps

        with (
            patch(
                "custom_components.roommind.utils.presence_utils.is_presence_away",
                return_value=False,
            ),
            patch(
                "custom_components.roommind.utils.schedule_utils.get_active_schedule_entity",
                return_value=None,
            ),
            patch(
                "custom_components.roommind.utils.schedule_utils.resolve_targets_at_time",
                return_value=TargetTemps(heat=19.5, cool=23.5),
            ) as mock_resolve,
        ):
            await _compute_target_forecast(
                hass,
                room,
                settings,
                hours=0.0,
                interval_minutes=5,
            )
            args = mock_resolve.call_args[0]
            assert args[3] == 19.5
            assert args[4] == 23.5

    @pytest.mark.asyncio
    async def test_cache_keeps_forecast_when_service_fails(self):
        """#308: when schedule.get_schedule raises but a cache entry exists,
        the forecast must use the cached block temperature instead of
        falling back to comfort/eco."""
        hass = MagicMock()
        hass.config.units.temperature_unit = "°C"
        # No state for the schedule selector -> first schedule wins
        hass.states.get = MagicMock(return_value=None)
        hass.services.async_call = AsyncMock(side_effect=RuntimeError("boom"))

        room = {
            "area_id": "living_room",
            "schedules": [{"entity_id": "schedule.heating"}],
            "schedule_selector_entity": "",
            "comfort_heat": 21.0,
            "comfort_cool": 24.0,
            "eco_heat": 17.0,
            "eco_cool": 27.0,
            "climate_mode": "heat_only",
        }
        settings = {"presence_away_action": "eco", "schedule_off_action": "eco"}

        all_day = {"from": "00:00:00", "to": "23:59:59", "data": {"temperature": 16.5}}
        schedule_data = {
            day: [all_day] for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        }
        cache = {"schedule.heating": schedule_data}

        with patch(
            "custom_components.roommind.utils.presence_utils.is_presence_away",
            return_value=False,
        ):
            result = await _compute_target_forecast(
                hass,
                room,
                settings,
                hours=0.1,
                interval_minutes=5,
                schedule_blocks_cache=cache,
            )

        assert result, "forecast should not be empty"
        heat_targets = {point["heat_target"] for point in result if point.get("heat_target") is not None}
        assert heat_targets == {16.5}

    @pytest.mark.asyncio
    async def test_empty_room_forecast_stays_at_eco(self):
        """An empty room (occupancy eco, #180) shows eco over the horizon, not the comfort schedule."""
        hass = MagicMock()
        hass.config.units.temperature_unit = "°C"
        room = {"comfort_heat": 21.0, "comfort_cool": 24.0, "eco_heat": 17.0, "eco_cool": 27.0}

        with patch(
            "custom_components.roommind.utils.schedule_utils.get_active_schedule_entity",
            return_value=None,
        ):
            result = await _compute_target_forecast(hass, room, {}, hours=0.5, room_vacant=True)
        assert {p["heat_target"] for p in result} == {17.0}


# ---------------------------------------------------------------------------
# Central Plant sides of a Thermal Zone
# ---------------------------------------------------------------------------


class TestCentralCanHeatCool:
    """The shared plant serves every room: only mode, outdoor gates and overrides decide."""

    def test_auto_room_has_both_sides(self):
        # Inside both default gates (cooling from 16 °C, heating up to 22 °C)
        assert _central_can_heat_cool({}, {}, 18.0) == (True, True)

    @pytest.mark.parametrize(
        ("climate_mode", "expected"),
        [("heat_only", (True, False)), ("cool_only", (False, True)), ("auto", (True, True))],
    )
    def test_climate_mode_removes_one_side(self, climate_mode, expected):
        assert _central_can_heat_cool({"climate_mode": climate_mode}, {}, 18.0) == expected

    def test_outdoor_room_is_conditioned_by_the_weather(self):
        assert _central_can_heat_cool({"is_outdoor": True}, {}, 20.0) == (False, False)

    def test_warm_outside_closes_heating(self):
        settings = {"outdoor_heating_max": 22.0, "outdoor_cooling_min": 16.0}
        assert _central_can_heat_cool({}, settings, 25.0) == (False, True)

    def test_cold_outside_closes_cooling(self):
        settings = {"outdoor_heating_max": 22.0, "outdoor_cooling_min": 16.0}
        assert _central_can_heat_cool({}, settings, 5.0) == (True, False)

    def test_no_outdoor_reading_keeps_both_sides(self):
        settings = {"outdoor_heating_max": 22.0, "outdoor_cooling_min": 16.0}
        assert _central_can_heat_cool({}, settings, None) == (True, True)

    def test_an_override_bypasses_the_outdoor_gates(self):
        settings = {"outdoor_heating_max": 22.0, "outdoor_cooling_min": 16.0}
        room = {"override_heat": 21.0, "override_until": None}
        assert _central_can_heat_cool(room, settings, 30.0) == (True, True)

    def test_an_expired_override_does_not_bypass_the_gates(self):
        settings = {"outdoor_heating_max": 22.0, "outdoor_cooling_min": 16.0}
        room = {"override_heat": 21.0, "override_until": time.time() - 60}
        assert _central_can_heat_cool(room, settings, 30.0) == (False, True)

    def test_settings_defaults_apply(self):
        # 10 °C is inside both default gates (cooling min 16, heating max 22)
        assert DEFAULT_OUTDOOR_TEMP_FALLBACK == 10.0
        assert _central_can_heat_cool({}, {}, 0.0) == (True, False)


# ---------------------------------------------------------------------------
# Central Plant model figures
# ---------------------------------------------------------------------------


class TestModelFigures:
    """The model card shows the forced-air mixing and house coupling of the EKF."""

    @staticmethod
    async def _payload(est: ThermalEKF, **coordinator_kwargs: object) -> dict:
        coordinator = _make_coordinator(estimator=est, **coordinator_kwargs)  # type: ignore[arg-type]
        hass = MagicMock()
        hass.config.latitude = 48.0
        hass.config.longitude = 11.0
        _run_executor(hass)
        store = MagicMock()
        store.get_settings.return_value = {}
        store.get_room.return_value = {}
        with (
            patch(
                "custom_components.roommind.services.analytics_service._compute_target_forecast",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "custom_components.roommind.services.analytics_service.is_room_mpc_ready",
                return_value=False,
            ),
        ):
            return await build_analytics_data(hass, "room1", "12h", store, coordinator)

    @pytest.mark.asyncio
    async def test_fan_and_house_figures_come_from_the_model(self):
        est = _estimator()
        est._P[_BETA_FAN_STATE][_BETA_FAN_STATE] = 0.1 * ThermalEKF._p_init_diag()[_BETA_FAN_STATE]
        data = await self._payload(est)

        model = data["model"]
        rc = est.get_model()
        assert model["fan_mix_rate"] == round(rc.Q_fan, 3)
        assert model["fan_mix_learned"] is True
        assert model["house_coupling"] == round(rc.U_house, 3)
        assert model["house_coupling_learned"] is False
        # Without a manager there is no trend to show
        assert model["occupancy_trend_24h"] == []
        assert "afterglow_k" not in model

    @pytest.mark.asyncio
    async def test_learned_gains_use_the_9_state_indices(self):
        est = _estimator()
        p_init = ThermalEKF._p_init_diag()
        # Solar (6) and occupancy (7) shrink below the learned share, house coupling (2) does not
        est._P[_SOLAR_STATE][_SOLAR_STATE] = 0.1 * p_init[_SOLAR_STATE]
        est._P[_OCCUPANCY_STATE][_OCCUPANCY_STATE] = 0.1 * p_init[_OCCUPANCY_STATE]
        assert _SOLAR_STATE == 6
        assert _OCCUPANCY_STATE == 7
        assert _ALPHA_HOUSE_STATE == 2
        assert _BETA_FAN_STATE == 5
        data = await self._payload(est)

        model = data["model"]
        assert model["solar_learned"] is True
        assert model["occupancy_learned"] is True
        assert model["house_coupling_learned"] is False

    @pytest.mark.asyncio
    async def test_occupancy_trend_is_the_24h_profile_of_the_weekday_type(self):
        manager = MagicMock()
        profile = [i / 96 for i in range(96)]
        manager.get_24h_profile.return_value = profile
        data = await self._payload(_estimator(), occupancy_manager=manager)

        assert data["model"]["occupancy_trend_24h"] == profile
        assert len(data["model"]["occupancy_trend_24h"]) == 96
        kwargs = manager.get_24h_profile.call_args.kwargs
        assert kwargs["weekend"] is (dt_util.now().weekday() >= 5)

    def test_occupancy_trend_helper_without_manager(self):
        coordinator = MagicMock()
        coordinator._occupancy_manager = None
        assert _occupancy_trend(coordinator, "room1") == []

    def test_occupancy_trend_helper_casts_the_profile_to_floats(self):
        manager = MagicMock()
        manager.get_24h_profile.return_value = [0, 0.25, 1]
        coordinator = MagicMock()
        coordinator._occupancy_manager = manager
        assert _occupancy_trend(coordinator, "room1") == [0.0, 0.25, 1.0]
        assert all(isinstance(value, float) for value in _occupancy_trend(coordinator, "room1"))


# ---------------------------------------------------------------------------
# build_analytics_data -- edge cases
# ---------------------------------------------------------------------------


class TestBuildAnalyticsData:
    """Tests for build_analytics_data edge cases."""

    @pytest.mark.asyncio
    async def test_forecast_gets_the_live_occupancy_vacancy(self):
        hass = MagicMock()
        _run_executor(hass)
        store = MagicMock()
        store.get_settings.return_value = {}
        store.get_room.return_value = {}
        coordinator = _make_coordinator(rooms={"living_room": {"occupancy_vacant": True}})
        coordinator._history_store.read_detail.return_value = []
        coordinator._history_store.read_history.return_value = []

        with patch(
            "custom_components.roommind.services.analytics_service._compute_target_forecast",
            new_callable=AsyncMock,
            return_value=[],
        ) as forecast:
            await build_analytics_data(hass, "living_room", "12h", store, coordinator)
        assert forecast.call_args.kwargs["room_vacant"] is True

    @pytest.mark.asyncio
    async def test_target_forecast_exception_returns_empty(self):
        """Exception in _compute_target_forecast -> empty forecast."""
        hass = MagicMock()
        hass.async_add_executor_job = AsyncMock(return_value=[])
        store = MagicMock()
        store.get_settings.return_value = {}
        store.get_room.return_value = {}
        coordinator = None

        with patch(
            "custom_components.roommind.services.analytics_service._compute_target_forecast",
            side_effect=RuntimeError("boom"),
        ):
            result = await build_analytics_data(hass, "living_room", "12h", store, coordinator)
        assert result["forecast"] == []

    @pytest.mark.asyncio
    async def test_a_custom_range_reads_both_files_from_its_bounds(self):
        hass = MagicMock()
        _run_executor(hass)
        store = MagicMock()
        store.get_settings.return_value = {}
        store.get_room.return_value = {}
        coordinator = _make_coordinator()

        with patch(
            "custom_components.roommind.services.analytics_service._compute_target_forecast",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await build_analytics_data(hass, "room1", "12h", store, coordinator, custom_start=100.0, custom_end=200.0)

        # Both files are read from the selected bounds, not from a range preset
        coordinator._history_store.read_detail.assert_any_call("room1", None, 100.0, 200.0)
        coordinator._history_store.read_history.assert_any_call("room1", None, 100.0, 200.0)

    @pytest.mark.asyncio
    async def test_no_history_store_returns_empty_ranges(self):
        hass = MagicMock()
        _run_executor(hass)
        store = MagicMock()
        store.get_settings.return_value = {}
        store.get_room.return_value = {}
        coordinator = _make_coordinator()
        coordinator._history_store = None

        with patch(
            "custom_components.roommind.services.analytics_service._compute_target_forecast",
            new_callable=AsyncMock,
            return_value=[],
        ):
            data = await build_analytics_data(hass, "room1", "12h", store, coordinator)
        assert data["detail"] == []
        assert data["rhythm"] is None

    @pytest.mark.asyncio
    async def test_current_temp_from_reversed_points(self):
        """Last room_temp is found by iterating reversed all_points."""
        hass = MagicMock()
        hass.config.latitude = 48.0
        hass.config.longitude = 11.0
        _run_executor(hass)
        store = MagicMock()
        store.get_settings.return_value = {"prediction_enabled": True}
        store.get_room.return_value = {"temperature_sensor": "sensor.temp"}

        now = time.time()
        est = _estimator()
        detail_rows = [_detail_row(1000, room_temp=""), _detail_row(2000, room_temp="20.5")]
        coordinator = _make_coordinator(estimator=est, detail_rows=detail_rows)

        simulate = _mock_forecast(21.0)
        with (
            patch(
                "custom_components.roommind.services.analytics_service._compute_target_forecast",
                new_callable=AsyncMock,
                return_value=[_forecast_point(now)],
            ),
            patch(
                "custom_components.roommind.services.analytics_service.is_room_mpc_ready",
                return_value=False,
            ),
            patch(
                "custom_components.roommind.control.analytics_simulator.simulate_prediction",
                simulate,
            ),
        ):
            await build_analytics_data(hass, "room1", "12h", store, coordinator)
            simulate.assert_called_once()
            assert simulate.call_args.kwargs["current_temp"] == 20.5

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("outdoor", "forecast", "known"),
        [(None, [], False), (None, [{"temperature": 25.0}], True), (12.0, [], True)],
    )
    async def test_prediction_knows_whether_the_outdoor_temp_is_a_placeholder(self, outdoor, forecast, known):
        """No reading and no forecast: the prediction must not gate on the placeholder (#464)."""
        hass = MagicMock()
        hass.config.latitude = 48.0
        hass.config.longitude = 11.0
        _run_executor(hass)
        store = MagicMock()
        store.get_settings.return_value = {"prediction_enabled": True}
        store.get_room.return_value = {"temperature_sensor": "sensor.temp"}

        est = _estimator()
        coordinator = _make_coordinator(
            estimator=est,
            outdoor_temp=outdoor,
            forecast=forecast,
            detail_rows=[_detail_row(2000, room_temp="26.0", outdoor_temp="", target_temp="24")],
        )

        simulate = _mock_forecast(24.0)
        with (
            patch(
                "custom_components.roommind.services.analytics_service._compute_target_forecast",
                new_callable=AsyncMock,
                return_value=[_forecast_point(time.time())],
            ),
            patch(
                "custom_components.roommind.services.analytics_service.is_room_mpc_ready",
                return_value=False,
            ),
            patch(
                "custom_components.roommind.control.analytics_simulator.simulate_prediction",
                simulate,
            ),
        ):
            await build_analytics_data(hass, "room1", "12h", store, coordinator)
        assert simulate.call_args.kwargs["outdoor_known"] is known
        # With neither a reading nor a forecast the placeholder of the control path is used
        assert simulate.call_args.kwargs["outdoor_series"][0] == (
            DEFAULT_OUTDOOR_TEMP_FALLBACK
            if outdoor is None and not forecast
            else outdoor_series_first(forecast, outdoor)
        )

    @pytest.mark.asyncio
    async def test_mpc_readiness_gates_the_forecast_path(self):
        """The joint plant plans a room whose EKF is ready; the forecast simulates that path."""
        hass = MagicMock()
        hass.config.latitude = 48.0
        hass.config.longitude = 11.0
        _run_executor(hass)
        store = MagicMock()
        store.get_settings.return_value = {"prediction_enabled": True}
        store.get_room.return_value = {"temperature_sensor": "sensor.temp"}
        coordinator = _make_coordinator(estimator=_estimator())

        simulate = _mock_forecast(20.5)
        ready = MagicMock(return_value=True)
        with (
            patch(
                "custom_components.roommind.services.analytics_service._compute_target_forecast",
                new_callable=AsyncMock,
                return_value=[_forecast_point(time.time())],
            ),
            patch("custom_components.roommind.services.analytics_service.is_room_mpc_ready", ready),
            patch(
                "custom_components.roommind.control.analytics_simulator.simulate_prediction",
                simulate,
            ),
        ):
            data = await build_analytics_data(hass, "room1", "12h", store, coordinator)

        assert data["model"]["mpc_active"] is True
        assert simulate.call_args.kwargs["mpc_active"] is True
        estimator_arg, can_heat, can_cool = ready.call_args.args
        assert estimator_arg is coordinator._model_manager._estimators["room1"]
        assert (can_heat, can_cool) == (True, True)

    @pytest.mark.asyncio
    async def test_bangbang_control_mode_never_reports_the_joint_plan(self):
        """A ready model under bang-bang control still simulates the fallback path."""
        hass = MagicMock()
        hass.config.latitude = 48.0
        hass.config.longitude = 11.0
        _run_executor(hass)
        store = MagicMock()
        store.get_settings.return_value = {"prediction_enabled": True, "control_mode": "bangbang"}
        store.get_room.return_value = {"temperature_sensor": "sensor.temp"}
        coordinator = _make_coordinator(estimator=_estimator())

        simulate = _mock_forecast(20.5)
        with (
            patch(
                "custom_components.roommind.services.analytics_service._compute_target_forecast",
                new_callable=AsyncMock,
                return_value=[_forecast_point(time.time())],
            ),
            patch("custom_components.roommind.services.analytics_service.is_room_mpc_ready", return_value=True),
            patch(
                "custom_components.roommind.control.analytics_simulator.simulate_prediction",
                simulate,
            ),
        ):
            data = await build_analytics_data(hass, "room1", "12h", store, coordinator)

        assert data["model"]["mpc_active"] is False
        assert simulate.call_args.kwargs["mpc_active"] is False

    @pytest.mark.asyncio
    async def test_cool_only_room_in_winter_asks_for_no_sides(self):
        """A cool-only room below the cooling minimum cannot condition at all."""
        hass = MagicMock()
        hass.config.latitude = 48.0
        hass.config.longitude = 11.0
        _run_executor(hass)
        store = MagicMock()
        store.get_settings.return_value = {"prediction_enabled": True}
        store.get_room.return_value = {"temperature_sensor": "sensor.temp", "climate_mode": "cool_only"}
        coordinator = _make_coordinator(estimator=_estimator(), outdoor_temp=5.0)

        ready = MagicMock(return_value=False)
        with (
            patch(
                "custom_components.roommind.services.analytics_service._compute_target_forecast",
                new_callable=AsyncMock,
                return_value=[_forecast_point(time.time())],
            ),
            patch("custom_components.roommind.services.analytics_service.is_room_mpc_ready", ready),
            patch(
                "custom_components.roommind.control.analytics_simulator.simulate_prediction",
                return_value=[14.0],
            ) as simulate,
        ):
            data = await build_analytics_data(hass, "room1", "12h", store, coordinator)

        assert data["model"]["mpc_active"] is False
        assert ready.call_args.args[1:] == (False, False)
        assert simulate.call_args.kwargs["mpc_active"] is False

    @pytest.mark.asyncio
    async def test_an_outdoor_room_is_never_planned_by_the_plant(self):
        hass = MagicMock()
        hass.config.latitude = 48.0
        hass.config.longitude = 11.0
        _run_executor(hass)
        store = MagicMock()
        store.get_settings.return_value = {"prediction_enabled": True}
        store.get_room.return_value = {"temperature_sensor": "sensor.temp", "is_outdoor": True}
        coordinator = _make_coordinator(estimator=_estimator())

        ready = MagicMock(return_value=False)
        with (
            patch(
                "custom_components.roommind.services.analytics_service._compute_target_forecast",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("custom_components.roommind.services.analytics_service.is_room_mpc_ready", ready),
        ):
            await build_analytics_data(hass, "room1", "12h", store, coordinator)
        assert ready.call_args.args[1:] == (False, False)

    @pytest.mark.asyncio
    async def test_no_room_temp_skips_prediction(self):
        """If no room_temp in any point, prediction is skipped (current_t stays None)."""
        hass = MagicMock()
        hass.config.latitude = 48.0
        hass.config.longitude = 11.0
        _run_executor(hass)
        store = MagicMock()
        store.get_settings.return_value = {"prediction_enabled": True}
        store.get_room.return_value = {"temperature_sensor": "sensor.temp"}

        coordinator = _make_coordinator(estimator=_estimator(), detail_rows=[_detail_row(1000, room_temp="")])

        with (
            patch(
                "custom_components.roommind.services.analytics_service._compute_target_forecast",
                new_callable=AsyncMock,
                return_value=[_forecast_point(time.time())],
            ),
            patch("custom_components.roommind.services.analytics_service.is_room_mpc_ready", return_value=False),
        ):
            result = await build_analytics_data(hass, "room1", "12h", store, coordinator)
        assert len(result["forecast"]) == 1
        assert result["forecast"][0]["predicted_temp"] is None

    @pytest.mark.asyncio
    async def test_prediction_disabled_skips_the_simulation(self):
        hass = MagicMock()
        _run_executor(hass)
        store = MagicMock()
        store.get_settings.return_value = {"prediction_enabled": False}
        store.get_room.return_value = {}
        coordinator = _make_coordinator(estimator=_estimator())

        with (
            patch(
                "custom_components.roommind.services.analytics_service._compute_target_forecast",
                new_callable=AsyncMock,
                return_value=[_forecast_point(time.time())],
            ),
            patch("custom_components.roommind.services.analytics_service.is_room_mpc_ready", return_value=True),
            patch(
                "custom_components.roommind.control.analytics_simulator.simulate_prediction",
            ) as simulate,
        ):
            result = await build_analytics_data(hass, "room1", "12h", store, coordinator)
        simulate.assert_not_called()
        assert result["forecast"][0]["predicted_temp"] is None

    @pytest.mark.asyncio
    async def test_the_rebound_charge_of_the_run_seeds_the_simulation(self):
        """The signed forced-air charge of the run in progress reaches the forecast."""
        hass = MagicMock()
        hass.config.latitude = 48.0
        hass.config.longitude = 11.0
        _run_executor(hass)
        store = MagicMock()
        store.get_settings.return_value = {"prediction_enabled": True}
        store.get_room.return_value = {"temperature_sensor": "sensor.temp"}
        coordinator = _make_coordinator(estimator=_estimator(), charge=0.62, humidity=55.0)

        simulate = _mock_forecast(21.0)
        with (
            patch(
                "custom_components.roommind.services.analytics_service._compute_target_forecast",
                new_callable=AsyncMock,
                return_value=[_forecast_point(time.time())],
            ),
            patch("custom_components.roommind.services.analytics_service.is_room_mpc_ready", return_value=False),
            patch(
                "custom_components.roommind.control.analytics_simulator.simulate_prediction",
                simulate,
            ),
        ):
            await build_analytics_data(hass, "room1", "12h", store, coordinator)

        kwargs = simulate.call_args.kwargs
        assert kwargs["q_rebound"] == 0.62
        assert kwargs["current_rh_pct"] == 55.0
        assert "residual_scale" not in kwargs
        assert "heating_duration_minutes" not in kwargs

    @pytest.mark.asyncio
    async def test_without_a_tracker_the_simulation_starts_uncharged(self):
        hass = MagicMock()
        hass.config.latitude = 48.0
        hass.config.longitude = 11.0
        _run_executor(hass)
        store = MagicMock()
        store.get_settings.return_value = {"prediction_enabled": True}
        store.get_room.return_value = {"temperature_sensor": "sensor.temp"}
        coordinator = _make_coordinator(estimator=_estimator())

        simulate = _mock_forecast(21.0)
        with (
            patch(
                "custom_components.roommind.services.analytics_service._compute_target_forecast",
                new_callable=AsyncMock,
                return_value=[_forecast_point(time.time())],
            ),
            patch("custom_components.roommind.services.analytics_service.is_room_mpc_ready", return_value=False),
            patch(
                "custom_components.roommind.control.analytics_simulator.simulate_prediction",
                simulate,
            ),
        ):
            await build_analytics_data(hass, "room1", "12h", store, coordinator)

        assert simulate.call_args.kwargs["q_rebound"] == 0.0
        # No live humidity reading either: the recorded one is not there, so no seed
        assert simulate.call_args.kwargs["current_rh_pct"] is None


def outdoor_series_first(forecast: list[dict], outdoor: float | None) -> float:
    """First block of the outdoor series when the reading (or the forecast) exists."""
    if outdoor is not None:
        return outdoor
    return forecast[0]["temperature"]


# ---------------------------------------------------------------------------
# _current_humidity / _rebound_charge
# ---------------------------------------------------------------------------


def test_current_humidity_prefers_the_live_reading():
    coordinator = MagicMock()
    coordinator.rooms = {"room1": {"current_humidity": 48.5}}
    assert _current_humidity(coordinator, "room1", [{"humidity": 30.0}]) == 48.5


def test_current_humidity_falls_back_to_the_last_recorded_value():
    coordinator = MagicMock()
    coordinator.rooms = {"room1": {}}
    points = [{"humidity": None}, {"humidity": 41.0}, {"humidity": None}]
    assert _current_humidity(coordinator, "room1", points) == 41.0


def test_current_humidity_without_any_reading():
    coordinator = MagicMock()
    coordinator.rooms = {}
    assert _current_humidity(coordinator, "room1", [{"humidity": None}]) is None


def test_rebound_charge_reads_the_tracker():
    coordinator = MagicMock()
    coordinator._rebound_tracker.charge.return_value = -0.4
    assert _rebound_charge(coordinator, "room1") == -0.4


def test_rebound_charge_without_tracker():
    coordinator = MagicMock()
    coordinator._rebound_tracker = None
    assert _rebound_charge(coordinator, "room1") == 0.0


def test_rebound_charge_ignores_a_non_numeric_tracker():
    coordinator = MagicMock()
    coordinator._rebound_tracker.charge.return_value = "not a charge"
    assert _rebound_charge(coordinator, "room1") == 0.0


# ---------------------------------------------------------------------------
# _csv_to_points -- recorded columns
# ---------------------------------------------------------------------------


class TestCsvToPoints:
    """The recorded row becomes a typed point, Central Plant columns included."""

    def test_device_setpoint_converted_to_float(self):
        points = _csv_to_points([_detail_row(1000, device_setpoint="24.5")])
        assert len(points) == 1
        assert points[0]["device_setpoint"] == 24.5

    def test_missing_device_setpoint_returns_none(self):
        points = _csv_to_points([_detail_row(1000)])
        assert points[0]["device_setpoint"] is None

    def test_central_hvac_columns_are_typed(self):
        row = _detail_row(
            1000,
            effective_temp="21.4",
            priority_weight="0.75",
            door_open="True",
            fan_circulating="1",
            circulate_starts="2",
            model_house_temp="20.1",
            model_power="-0.5",
            model_fan="0.8",
        )
        point = _csv_to_points([row])[0]
        assert point["effective_temp"] == 21.4
        assert point["priority_weight"] == 0.75
        assert point["door_open"] is True
        assert point["fan_circulating"] is True
        assert point["circulate_starts"] == 2
        # The House Reference Temperature under the name the page draws
        assert point["house_temp"] == 20.1
        assert point["model_house_temp"] == 20.1
        assert point["model_power"] == -0.5
        assert point["model_fan"] == 0.8

    @pytest.mark.parametrize(("raw", "expected"), [("false", False), ("", False), ("true", True)])
    def test_binary_columns_accept_the_recorded_spellings(self, raw, expected):
        point = _csv_to_points([_detail_row(1000, door_open=raw, fan_circulating=raw)])[0]
        assert point["door_open"] is expected
        assert point["fan_circulating"] is expected

    def test_a_row_without_timestamp_is_skipped(self):
        assert _csv_to_points([_detail_row("")]) == []


# ---------------------------------------------------------------------------
# build_analytics_data -- shading factor / occupancy sensors
# ---------------------------------------------------------------------------


class TestBuildAnalyticsShadingFactor:
    @pytest.mark.asyncio
    async def test_blind_position_triggers_shading_factor(self):
        hass = MagicMock()
        hass.config.latitude = 48.0
        hass.config.longitude = 11.0
        _run_executor(hass)
        store = MagicMock()
        store.get_settings.return_value = {"prediction_enabled": True}
        store.get_room.return_value = {"temperature_sensor": "sensor.temp"}
        coordinator = _make_coordinator(estimator=_estimator(), rooms={"room1": {"blind_position": 50}})

        with (
            patch(
                "custom_components.roommind.services.analytics_service._compute_target_forecast",
                new_callable=AsyncMock,
                return_value=[_forecast_point(time.time())],
            ),
            patch("custom_components.roommind.services.analytics_service.is_room_mpc_ready", return_value=False),
            patch(
                "custom_components.roommind.managers.cover_manager.compute_shading_factor",
                return_value=0.5,
            ) as mock_shading,
            patch(
                "custom_components.roommind.control.analytics_simulator.simulate_prediction",
                _mock_forecast(21.0),
            ) as mock_sim,
        ):
            await build_analytics_data(hass, "room1", "12h", store, coordinator)
            mock_shading.assert_called_once_with([50])
            mock_sim.assert_called_once()


class TestBuildAnalyticsOccupancy:
    @pytest.mark.asyncio
    async def test_occupancy_sensor_on_sets_q_occupancy(self):
        hass = MagicMock()
        hass.config.latitude = 48.0
        hass.config.longitude = 11.0
        _run_executor(hass)
        occ_state = MagicMock()
        occ_state.state = "on"
        hass.states.get = MagicMock(return_value=occ_state)

        store = MagicMock()
        store.get_settings.return_value = {"prediction_enabled": True}
        store.get_room.return_value = {
            "temperature_sensor": "sensor.temp",
            "occupancy_sensors": ["binary_sensor.occ1"],
        }
        coordinator = _make_coordinator(estimator=_estimator())

        with (
            patch(
                "custom_components.roommind.services.analytics_service._compute_target_forecast",
                new_callable=AsyncMock,
                return_value=[_forecast_point(time.time())],
            ),
            patch("custom_components.roommind.services.analytics_service.is_room_mpc_ready", return_value=False),
            patch(
                "custom_components.roommind.control.analytics_simulator.simulate_prediction",
                _mock_forecast(21.0),
            ) as mock_sim,
        ):
            await build_analytics_data(hass, "room1", "12h", store, coordinator)
            mock_sim.assert_called_once()
            assert mock_sim.call_args.kwargs["q_occupancy"] == 1.0

    @pytest.mark.asyncio
    async def test_occupancy_sensor_off_keeps_q_occupancy_zero(self):
        hass = MagicMock()
        hass.config.latitude = 48.0
        hass.config.longitude = 11.0
        _run_executor(hass)
        occ_state = MagicMock()
        occ_state.state = "off"
        hass.states.get = MagicMock(return_value=occ_state)

        store = MagicMock()
        store.get_settings.return_value = {"prediction_enabled": True}
        store.get_room.return_value = {
            "temperature_sensor": "sensor.temp",
            "occupancy_sensors": ["binary_sensor.occ1"],
        }
        coordinator = _make_coordinator(estimator=_estimator())

        with (
            patch(
                "custom_components.roommind.services.analytics_service._compute_target_forecast",
                new_callable=AsyncMock,
                return_value=[_forecast_point(time.time())],
            ),
            patch("custom_components.roommind.services.analytics_service.is_room_mpc_ready", return_value=False),
            patch(
                "custom_components.roommind.control.analytics_simulator.simulate_prediction",
                _mock_forecast(21.0),
            ) as mock_sim,
        ):
            await build_analytics_data(hass, "room1", "12h", store, coordinator)
            mock_sim.assert_called_once()
            assert mock_sim.call_args.kwargs["q_occupancy"] == 0.0


# ---------------------------------------------------------------------------
# build_analytics_data -- response payload
# ---------------------------------------------------------------------------


class TestBuildAnalyticsPayload:
    @pytest.mark.asyncio
    async def test_points_carry_the_control_variables_but_not_the_model_inputs(self):
        hass = MagicMock()
        hass.config.latitude = 48.0
        hass.config.longitude = 11.0
        _run_executor(hass)
        store = MagicMock()
        store.get_settings.return_value = {}
        store.get_room.return_value = {}
        rows = [
            _detail_row(
                1000,
                effective_temp="21.4",
                priority_weight="0.75",
                door_open="true",
                fan_circulating="True",
                circulate_starts="1",
                heat_starts="1",
                model_house_temp="20.1",
                model_power="0.4",
            )
        ]
        coordinator = _make_coordinator(detail_rows=rows)

        with patch(
            "custom_components.roommind.services.analytics_service._compute_target_forecast",
            new_callable=AsyncMock,
            return_value=[],
        ):
            data = await build_analytics_data(hass, "room1", "12h", store, coordinator)

        point = data["detail"][0]
        assert point["effective_temp"] == 21.4
        assert point["priority_weight"] == 0.75
        assert point["door_open"] is True
        assert point["fan_circulating"] is True
        assert point["house_temp"] == 20.1
        # Bucket metadata and model inputs stay out of the payload
        for key in ("heat_starts", "cool_starts", "circulate_starts", "room_temp_max", "room_temp_min"):
            assert key not in point
        for key in ("model_power", "model_house_temp", "model_solar"):
            assert key not in point

    @pytest.mark.asyncio
    async def test_stats_and_rhythm_carry_the_circulation_figures(self):
        hass = MagicMock()
        hass.config.latitude = 48.0
        hass.config.longitude = 11.0
        _run_executor(hass)
        store = MagicMock()
        store.get_settings.return_value = {}
        store.get_room.return_value = {}
        now = time.time()
        rows = [
            _detail_row(now - 7200, mode="circulating", circulate_starts="1"),
            _detail_row(now - 6900, mode="circulating"),
        ]
        coordinator = _make_coordinator(detail_rows=[], history_rows=rows)

        with patch(
            "custom_components.roommind.services.analytics_service._compute_target_forecast",
            new_callable=AsyncMock,
            return_value=[],
        ):
            data = await build_analytics_data(hass, "room1", "12h", store, coordinator)

        assert data["stats"]["circulating_hours"] > 0
        assert data["stats"]["circulate_starts"] == 1


# ---------------------------------------------------------------------------
# _observed / _safe_float
# ---------------------------------------------------------------------------


def test_observed_rejects_a_state_a_legacy_covariance_does_not_carry():
    """A covariance from an older store stops short of the occupancy and fast-gain states."""
    est = MagicMock()
    est._P = [[0.01 for _ in range(7)] for _ in range(7)]
    assert _observed(est, _BETA_FAN_STATE) is True  # the state exists, its variance is learned
    assert _observed(est, _OCCUPANCY_STATE) is False
    assert _observed(est, _FAST_STATE) is False


def test_safe_float_handles_empty_and_invalid_values():
    assert _safe_float("") is None
    assert _safe_float(None) is None
    assert _safe_float("not a number") is None
    assert _safe_float("-1.5") == -1.5


# ---------------------------------------------------------------------------
# _safe_int
# ---------------------------------------------------------------------------


class TestSafeInt:
    def test_valid_integer_string(self):
        assert _safe_int("42") == 42

    def test_valid_float_string_truncates(self):
        assert _safe_int("3.7") == 3

    def test_invalid_string_returns_none(self):
        assert _safe_int("abc") is None

    def test_none_value_returns_none(self):
        assert _safe_int(None) is None
