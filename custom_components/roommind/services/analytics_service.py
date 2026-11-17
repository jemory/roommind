"""Analytics data assembly service for RoomMind."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, cast

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from ..const import (
    CLIMATE_MODE_COOL_ONLY,
    CLIMATE_MODE_HEAT_ONLY,
    CONTROL_MODE_BANGBANG,
    DEFAULT_COMFORT_COOL,
    DEFAULT_COMFORT_HEAT,
    DEFAULT_CONTROL_MODE,
    DEFAULT_ECO_COOL,
    DEFAULT_ECO_HEAT,
    DEFAULT_OUTDOOR_COOLING_MIN,
    DEFAULT_OUTDOOR_HEATING_MAX,
    MODE_IDLE,
    is_override_active,
)
from ..control.central_mpc import is_room_mpc_ready
from ..control.solar import compute_q_solar_norm
from ..control.thermal_model import ThermalEKF
from ..utils.history_store import MODEL_INPUT_FIELDS
from ..utils.price_utils import PriceSignal
from ..utils.room_utils import room_solar_azimuths
from .analytics_stats import compute_period_stats, compute_rhythm, rhythm_window
from .model_backtest import backtest

_LOGGER = logging.getLogger(__name__)

SOLAR_PEAK_STEP_S = 600
# Placeholder for a missing outdoor reading, the same one the control path uses.
DEFAULT_OUTDOOR_TEMP_FALLBACK = 10.0


def clear_sky_peak_today(latitude: float, longitude: float, now: float) -> float:
    """Highest clear-sky q_solar (GHI/1000) of the local calendar day containing *now*."""
    start = dt_util.start_of_local_day(dt_util.utc_from_timestamp(now)).timestamp()
    return max(
        compute_q_solar_norm(latitude, longitude, start + i * SOLAR_PEAK_STEP_S)
        for i in range(86400 // SOLAR_PEAK_STEP_S)
    )


def _central_can_heat_cool(room_config: dict, settings: dict, outdoor_temp: float | None) -> tuple[bool, bool]:
    """The heating and cooling sides a Thermal Zone can use under the Central Plant.

    One shared forced-air plant serves every room, so only the room's climate mode and
    the outdoor gates decide (there is no per-room device to check): a heat-only room
    never cools, a cool-only room never heats, and an outdoor reading past a gate closes
    that side. An active override bypasses the gates because the user asked for that
    temperature now, and an outdoor room is conditioned by the weather, not the plant.
    """
    if room_config.get("is_outdoor", False):
        return False, False
    climate_mode = room_config.get("climate_mode", "auto")
    can_heat = climate_mode != CLIMATE_MODE_COOL_ONLY
    can_cool = climate_mode != CLIMATE_MODE_HEAT_ONLY
    if outdoor_temp is not None and not is_override_active(room_config):
        if outdoor_temp > settings.get("outdoor_heating_max", DEFAULT_OUTDOOR_HEATING_MAX):
            can_heat = False
        if outdoor_temp < settings.get("outdoor_cooling_min", DEFAULT_OUTDOOR_COOLING_MIN):
            can_cool = False
    return can_heat, can_cool


def _occupancy_trend(coordinator: Any, area_id: str) -> list[float]:
    """The room's learned 24 h Occupancy Trend (96 quarter-hour values), empty without one."""
    manager = getattr(coordinator, "_occupancy_manager", None)
    if manager is None:
        return []
    profile = manager.get_24h_profile(area_id, weekend=dt_util.now().weekday() >= 5)
    return [float(value) for value in profile]


def _rebound_charge(coordinator: Any, area_id: str) -> float:
    """Signed air/mass charge of the room's forced-air run (0.0 without a tracker)."""
    tracker = getattr(coordinator, "_rebound_tracker", None)
    charge = tracker.charge(area_id) if tracker is not None else 0.0
    return float(charge) if isinstance(charge, int | float) else 0.0


def _current_humidity(coordinator: Any, area_id: str, points: list[dict]) -> float | None:
    """Relative humidity of the room now: its live reading, else the last recorded one."""
    live: dict = coordinator.rooms.get(area_id) or {}
    humidity = live.get("current_humidity")
    if isinstance(humidity, int | float):
        return float(humidity)
    for point in reversed(points):
        value = point.get("humidity")
        if value is not None:
            return float(value)
    return None


# EKF states of the Central Forced-Air model (thermal_model.ThermalEKF)
# x = [T, alpha_out, alpha_house, beta_h, beta_c, beta_fan, beta_s, beta_o, beta_f]
_ALPHA_OUT_STATE = 1
_ALPHA_HOUSE_STATE = 2
_BETA_H_STATE = 3
_BETA_C_STATE = 4
_BETA_FAN_STATE = 5
_SOLAR_STATE = 6
_OCCUPANCY_STATE = 7
_FAST_STATE = 8
# A gain counts as learned once its variance fell below this share of its start value
LEARNED_VARIANCE_SHARE = 0.25


def _observed(est: Any, state: int) -> bool:
    """Whether the estimator has seen this gain in its data, not just its start value.

    With gain gating a state that was never excited (no sun, no occupancy) gets no Kalman gain
    and no drift noise, so its variance stays exactly at the start value until it is observed.
    """
    P = est._P
    if len(P) <= state:
        return False
    return bool(P[state][state] < LEARNED_VARIANCE_SHARE * ThermalEKF._p_init_diag()[state])


def _safe_float(value: str) -> float | None:
    """Convert CSV string to float, or None for empty/invalid values."""
    if not value:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_int(value: str) -> int | None:
    """Convert CSV string to int, or None for empty/invalid values."""
    if not value:
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def _csv_to_points(rows: list[dict]) -> list[dict]:
    """Convert CSV rows (string values, 'timestamp' key) to typed points ('ts' key)."""
    result = []
    for row in rows:
        ts = _safe_float(row.get("timestamp", ""))
        if ts is None:
            continue
        result.append(
            {
                "ts": ts,
                "room_temp": _safe_float(row.get("room_temp", "")),
                "outdoor_temp": _safe_float(row.get("outdoor_temp", "")),
                "target_temp": _safe_float(row.get("target_temp", "")),
                "mode": row.get("mode", ""),
                "predicted_temp": _safe_float(row.get("predicted_temp", "")),
                "window_open": row.get("window_open", "") in ("True", "true", "1"),
                "heating_power": _safe_float(row.get("heating_power", "")),
                "solar_irradiance": _safe_float(row.get("solar_irradiance", "")),
                "blind_position": _safe_int(row.get("blind_position", "")),
                "cover_reason": row.get("cover_reason", ""),
                "device_setpoint": _safe_float(row.get("device_setpoint", "")),
                "humidity": _safe_float(row.get("humidity", "")),
                "heat_target": _safe_float(row.get("heat_target", "")),
                "cool_target": _safe_float(row.get("cool_target", "")),
                "target_reason": row.get("target_reason", ""),
                "mold_boost": _safe_float(row.get("mold_boost", "")),
                "price": _safe_float(row.get("price", "")),
                "price_level": row.get("price_level", ""),
                "price_shift": _safe_float(row.get("price_shift", "")),
                "heat_starts": _safe_int(row.get("heat_starts", "")),
                "cool_starts": _safe_int(row.get("cool_starts", "")),
                "circulate_starts": _safe_int(row.get("circulate_starts", "")),
                "room_temp_max": _safe_float(row.get("room_temp_max", "")),
                "room_temp_min": _safe_float(row.get("room_temp_min", "")),
                "co2": _safe_float(row.get("co2", "")),
                # Central HVAC: the resolved control variables behind the mode and the two
                # binary inputs the forced-air model integrates
                "effective_temp": _safe_float(row.get("effective_temp", "")),
                "priority_weight": _safe_float(row.get("priority_weight", "")),
                "door_open": row.get("door_open", "") in ("True", "true", "1"),
                "fan_circulating": row.get("fan_circulating", "") in ("True", "true", "1"),
                # The House Reference Temperature the model learned from, under its own name
                "house_temp": _safe_float(row.get("model_house_temp", "")),
                **{f: _safe_float(row.get(f, "")) for f in MODEL_INPUT_FIELDS},
            }
        )
    return result


async def _compute_target_forecast(
    hass: HomeAssistant,
    room: dict,
    settings: dict,
    mold_prevention_delta: float = 0.0,
    hours: float = 3.0,
    interval_minutes: int = 5,
    schedule_blocks_cache: dict[str, dict] | None = None,
    room_vacant: bool = False,
    price_signal: PriceSignal | None = None,
) -> list[dict]:
    """Compute target temperature forecast for the next N hours.

    Each point contains ``target_temp`` (chart display, mode-aware),
    ``heat_target`` and ``cool_target`` (for MPC simulator).
    """
    from ..utils.presence_utils import is_presence_away
    from ..utils.schedule_utils import (
        cap_cool_only_heat,
        get_active_schedule_entity,
        price_shifted,
        read_schedule_blocks,
        resolve_targets_at_time,
    )
    from ..utils.temp_utils import ha_temp_to_celsius

    comfort_heat = room.get("comfort_heat", room.get("comfort_temp", DEFAULT_COMFORT_HEAT))
    comfort_cool = room.get("comfort_cool", DEFAULT_COMFORT_COOL)
    eco_heat = room.get("eco_heat", room.get("eco_temp", DEFAULT_ECO_HEAT))
    eco_cool = room.get("eco_cool", DEFAULT_ECO_COOL)
    override_until = room.get("override_until")
    override_heat = room.get("override_heat")
    override_cool = room.get("override_cool")
    vacation_until = settings.get("vacation_until")
    vacation_temp = settings.get("vacation_temp")
    climate_mode = room.get("climate_mode", "auto")

    presence_away = not room.get("ignore_presence", False) and is_presence_away(hass, room, settings)

    entity_id = get_active_schedule_entity(hass, room)
    schedule_blocks = await read_schedule_blocks(hass, entity_id, cache=schedule_blocks_cache) if entity_id else None

    _hass = hass
    converter = lambda v: ha_temp_to_celsius(_hass, v)  # noqa: E731

    # Generate forecast points
    now = time.time()
    end_ts = now + hours * 3600
    result: list[dict] = []
    ts = now
    while ts <= end_ts:
        targets = resolve_targets_at_time(
            ts,
            schedule_blocks,
            override_until,
            override_heat,
            override_cool,
            vacation_until,
            vacation_temp,
            comfort_heat,
            comfort_cool,
            eco_heat,
            eco_cool,
            presence_away=presence_away,
            block_temp_converter=converter,
            presence_away_action=settings.get("presence_away_action", "eco"),
            schedule_off_action=settings.get("schedule_off_action", "eco"),
            presence_clears_override=bool(settings.get("presence_clears_override", False)),
            room_vacant=room_vacant,
        )
        targets, (shift_heat, shift_cool) = price_shifted(
            targets, ts, room, settings, price_signal, presence_away=presence_away, room_vacant=room_vacant
        )
        heat_target = targets.heat
        cool_target = targets.cool

        # Apply mold prevention delta to heat target only
        if heat_target is not None:
            heat_target = round(heat_target + mold_prevention_delta, 1)
        elif mold_prevention_delta > 0:
            heat_target = round(eco_heat + mold_prevention_delta, 1)
        heat_target, cool_target = cap_cool_only_heat(heat_target, cool_target, climate_mode)

        # Chart display: mode-aware single value
        if climate_mode == CLIMATE_MODE_COOL_ONLY:
            target = cool_target
        elif climate_mode == CLIMATE_MODE_HEAT_ONLY:
            target = heat_target
        else:
            # Auto mode: show heat target (primary for chart line)
            target = heat_target

        result.append(
            {
                "ts": round(ts, 1),
                "target_temp": target,
                "heat_target": heat_target,
                "cool_target": cool_target,
                # For the prediction's pre-heat deadline, which ignores the price shift
                "price_shift_heat": shift_heat,
                "price_shift_cool": shift_cool,
            }
        )
        ts += interval_minutes * 60
    return result


async def build_analytics_data(
    hass: HomeAssistant,
    area_id: str,
    range_key: str,
    store: Any,
    coordinator: Any,
    custom_start: float | None = None,
    custom_end: float | None = None,
) -> dict:
    """Build analytics response data for a room.

    This is the core data assembly extracted from websocket_get_analytics.
    """
    settings = store.get_settings()
    history_store = getattr(coordinator, "_history_store", None)

    # Read history data -- custom timestamps take precedence over range preset
    detail: list = []
    history: list = []
    now = time.time()
    range_start = custom_start if custom_start is not None else now - 43200
    if history_store:
        if custom_start is not None:
            detail = _csv_to_points(
                await hass.async_add_executor_job(history_store.read_detail, area_id, None, custom_start, custom_end)
            )
            history = _csv_to_points(
                await hass.async_add_executor_job(history_store.read_history, area_id, None, custom_start, custom_end)
            )
        else:
            max_age_map = {
                "12h": 43200,
                "24h": 86400,
                "2d": 172800,
                "7d": 604800,
                "14d": 1209600,
                "30d": 2592000,
                "90d": 7776000,
            }
            max_age = max_age_map.get(range_key, 43200)
            range_start = now - max_age
            detail = _csv_to_points(await hass.async_add_executor_job(history_store.read_detail, area_id, max_age))
            history = _csv_to_points(await hass.async_add_executor_job(history_store.read_history, area_id, max_age))

    room_config = store.get_room(area_id) or {}
    climate_mode = room_config.get("climate_mode", "auto")
    stats = compute_period_stats(sorted(history + detail, key=lambda p: p["ts"]), climate_mode)
    # The model's error one hour ahead on this range; the line replaces the stored 3-min prediction
    backtest_figures: dict | None = None
    if coordinator and area_id in coordinator._model_manager._estimators:
        est = coordinator._model_manager._estimators[area_id]
        merged = sorted(history + detail, key=lambda p: p["ts"])
        result = await hass.async_add_executor_job(backtest, merged, est.get_model(), est.k_window)
        backtest_figures = result.figures()
        for point in merged:
            point["predicted_temp"] = result.predicted.get(point["ts"])
    else:
        for point in history + detail:
            point["predicted_temp"] = None
    # Only the figures need the run starts, bucket extremes and model inputs; 90 days of
    # points would carry them. The control variables (effective temp, priority, openings,
    # house temperature) stay: they are one number per point and the page draws them.
    for point in history + detail:
        del point["heat_starts"], point["cool_starts"], point["circulate_starts"]
        del point["room_temp_max"], point["room_temp_min"]
        for f in MODEL_INPUT_FIELDS:
            del point[f]

    rhythm: dict | None = None
    if history_store:
        rhythm_days, rhythm_end = rhythm_window(range_start, custom_end if custom_end is not None else now, now)
        rhythm_start = rhythm_end - (rhythm_days + 1) * 86400
        rhythm_rows = await hass.async_add_executor_job(
            history_store.read_history, area_id, None, rhythm_start, rhythm_end
        )
        rhythm_rows += await hass.async_add_executor_job(
            history_store.read_detail, area_id, None, rhythm_start, rhythm_end
        )
        rhythm = compute_rhythm(
            sorted(_csv_to_points(rhythm_rows), key=lambda p: p["ts"]), climate_mode, rhythm_end, rhythm_days
        )

    # Model info (only if estimator exists -- avoid auto-creating for unknown rooms)
    model_info: dict = {}
    # The joint whole-home MPC advances every Thermal Zone together, so the room's own gate
    # is its EKF: ready for the plant modes it may use means its curve is planned, and the
    # forecast simulates that path (Circulation is available to a room whose sides are closed).
    mpc_ready = False
    if coordinator:
        mgr = coordinator._model_manager
        if area_id in mgr._estimators:
            est = mgr._estimators[area_id]
            rc_model = est.get_model()
            pred_std_idle = est.prediction_std(0.0, 20.0, 15.0, 5.0)
            pred_std_heat = est.prediction_std(rc_model.Q_heat, 20.0, 10.0, 5.0)
            can_heat, can_cool = _central_can_heat_cool(room_config, settings, coordinator.outdoor_temp_effective)
            # Bang-bang control never runs the joint plan, however ready the model is
            mpc_ready = settings.get("control_mode", DEFAULT_CONTROL_MODE) != CONTROL_MODE_BANGBANG and (
                is_room_mpc_ready(est, can_heat, can_cool)
            )
            # EKF uncertainty: sqrt(P[0][0]) as proxy for sigma_e
            sigma_proxy = math.sqrt(max(est._P[0][0], 0.0))
            has_occupancy_sensors = len(room_config.get("occupancy_sensors", [])) > 0
            model_info = {
                "confidence": est.confidence,
                "daily_history": [
                    {
                        "date": e[0],
                        "confidence": e[1],
                        "mpc_active": e[3],
                        "error": e[4],
                        "error_steps": e[5],
                    }
                    for e in est.confidence_history
                ],
                "model": rc_model.to_dict(),
                "n_samples": est._n_updates,
                "n_observations": est._n_updates,
                "last_update_at": est._last_update_at,
                "n_heating": est._n_heating,
                "n_cooling": est._n_cooling,
                "applicable_modes": sorted(est._applicable_modes),
                "mpc_active": mpc_ready,
                "sigma_e": round(sigma_proxy, 4),
                "prediction_std_idle": round(pred_std_idle, 4),
                "prediction_std_heating": round(pred_std_heat, 4),
                "has_occupancy_sensors": has_occupancy_sensors,
                "solar_learned": _observed(est, _SOLAR_STATE),
                "occupancy_learned": _observed(est, _OCCUPANCY_STATE),
                # Central Forced-Air model: how fast the blower mixes the room with the house
                # (1/h) and how strongly it couples to it passively, both learned
                "fan_mix_rate": round(rc_model.Q_fan, 3),
                "fan_mix_learned": _observed(est, _BETA_FAN_STATE),
                "house_coupling": round(rc_model.U_house, 3),
                "house_coupling_learned": _observed(est, _ALPHA_HOUSE_STATE),
                # When the room tends to be occupied, for the comfort card's outlook
                "occupancy_trend_24h": _occupancy_trend(coordinator, area_id),
                # Q_solar is the gain at full sun (1000 W/m²); these are what it means now and today
                "solar_gain_now": round(
                    rc_model.Q_solar
                    * coordinator._current_q_solar
                    * coordinator.rooms.get(area_id, {}).get("shading_factor", 1.0),
                    3,
                ),
                "solar_gain_today_max": round(
                    rc_model.Q_solar * clear_sky_peak_today(hass.config.latitude, hass.config.longitude, time.time()),
                    3,
                ),
                "backtest": backtest_figures,
            }

    # Build merged forecast: same format as history points, on a shared 5-min grid
    mold_delta = 0.0
    room_vacant = False
    if coordinator:
        live = coordinator.rooms.get(area_id, {})
        mold_delta = live.get("mold_prevention_delta", 0.0)
        room_vacant = bool(live.get("occupancy_vacant", False))
    try:
        target_forecast = await _compute_target_forecast(
            hass,
            room_config,
            settings,
            mold_prevention_delta=mold_delta,
            schedule_blocks_cache=getattr(coordinator, "_schedule_blocks_cache", None),
            room_vacant=room_vacant,
            price_signal=coordinator.price_signal if coordinator else None,
        )
    except Exception:  # noqa: BLE001
        _LOGGER.debug("Target forecast computation failed for '%s'", area_id)
        target_forecast = []

    # Forward-simulate temperature prediction for the forecast period.
    from ..control.analytics_simulator import (
        build_forecast_outdoor_series,
        build_forecast_solar_series,
        simulate_prediction,
    )

    pred_temps: list[float | None] = list()
    prediction_enabled = settings.get("prediction_enabled", True)
    if prediction_enabled and target_forecast and coordinator:
        mgr = coordinator._model_manager
        if area_id in mgr._estimators:
            model = mgr.get_model(area_id)
            est = mgr._estimators[area_id]
            all_points = detail if detail else history
            current_t: float | None = None
            for p in reversed(all_points):
                if p.get("room_temp") is not None:
                    current_t = p["room_temp"]
                    break
            if current_t is not None:
                outdoor_now = coordinator.outdoor_temp_effective
                if outdoor_now is None and not coordinator._weather_manager._outdoor_forecast:
                    # Neither a reading nor a forecast: the placeholder of the control path
                    outdoor_now = DEFAULT_OUTDOOR_TEMP_FALLBACK
                outdoor_series = build_forecast_outdoor_series(
                    coordinator._weather_manager._outdoor_forecast,
                    outdoor_now,
                    len(target_forecast),
                )
                # Shading factor from current cover positions
                live = coordinator.rooms.get(area_id, {})
                _shading = 1.0
                if live.get("blind_position") is not None:
                    from ..managers.cover_manager import compute_shading_factor

                    _shading = compute_shading_factor([live["blind_position"]])
                solar_series = build_forecast_solar_series(
                    hass.config.latitude,
                    hass.config.longitude,
                    coordinator._weather_manager._outdoor_forecast,
                    len(target_forecast),
                    shading_factor=_shading,
                    # The backtest must replay the same regressor the EKF trained on
                    azimuths=room_solar_azimuths(room_config),
                )
                sim_q_occupancy = 0.0
                for occ_eid in room_config.get("occupancy_sensors", []):
                    occ_state = hass.states.get(occ_eid)
                    if occ_state and occ_state.state == "on":
                        sim_q_occupancy = 1.0
                        break

                # Seed the run in progress: without it the prediction starts idle, and a
                # room inside the start hysteresis predicts a stop RoomMind never makes.
                sim_prev_mode = coordinator._previous_modes.get(area_id, MODE_IDLE)
                mode_on_since = coordinator._mode_on_since.get(area_id)
                sim_mode_elapsed = (time.time() - mode_on_since) / 60.0 if mode_on_since is not None else None

                pred_temps = cast(
                    list[float | None],
                    simulate_prediction(
                        model=model,
                        estimator=est,
                        target_forecast=target_forecast,
                        outdoor_series=outdoor_series,
                        current_temp=current_t,
                        window_open=coordinator._window_manager._paused.get(area_id, False),
                        mpc_active=mpc_ready,
                        room_config=room_config,
                        settings=settings,
                        all_points=all_points,
                        solar_series=solar_series,
                        q_occupancy=sim_q_occupancy,
                        # Signed forced-air charge of the run in progress: the air leads the
                        # mass during the run and relaxes back after it, so the forecast shows it
                        q_rebound=_rebound_charge(coordinator, area_id),
                        # The room's humidity seeds the Effective Temperature of the roll-out
                        current_rh_pct=_current_humidity(coordinator, area_id, all_points),
                        previous_mode=sim_prev_mode,
                        mode_elapsed_minutes=sim_mode_elapsed,
                        outdoor_known=(
                            coordinator.outdoor_temp_effective is not None
                            or bool(coordinator._weather_manager._outdoor_forecast)
                        ),
                    ),
                )

    # Merge into unified forecast points on shared 5-min grid
    forecast: list[dict] = []
    grid = 300  # 5 minutes
    for i, tf in enumerate(target_forecast):
        snapped = round(tf["ts"] / grid) * grid
        forecast.append(
            {
                "ts": snapped,
                "room_temp": None,
                "outdoor_temp": None,
                "target_temp": tf["target_temp"],
                "heat_target": tf["heat_target"],
                "cool_target": tf["cool_target"],
                "mode": "forecast",
                "predicted_temp": pred_temps[i] if i < len(pred_temps) else None,
                "window_open": False,
                "device_setpoint": None,
            }
        )

    return {
        "detail": detail,
        "history": history,
        "model": model_info,
        "forecast": forecast,
        "stats": stats,
        "rhythm": rhythm,
        "price_unit": _price_unit(hass, settings),
    }


def _price_unit(hass: HomeAssistant, settings: dict) -> str:
    """Unit of the price source (ct/kWh, EUR/kWh, ...), empty for a switch or none."""
    entity_id = settings.get("price_entity") or ""
    state = hass.states.get(entity_id) if entity_id else None
    return str(state.attributes.get("unit_of_measurement") or "") if state else ""
