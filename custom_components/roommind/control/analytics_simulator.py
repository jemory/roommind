"""Analytics temperature prediction simulator for RoomMind's Central HVAC stack.

The analytics chart shows where a room's temperature is heading under the plan the
coordinator would choose.  The forecast therefore runs the *same* Central HVAC
machinery the control loop uses -- no airflow or cost maths is duplicated here:

* :class:`~.central_mpc.CentralMPCOptimizer` for a room whose model cleared the
  joint-MPC data gates (the single-room forecast is a one-room Central MPC run,
  and passing the sibling rooms makes it a whole-house run),
* :class:`~.central_mpc.CentralBangBang` with the coupled
  :class:`~.thermal_model.RCModel` and the signed forced-air relaxation of
  :mod:`.rebound` while the EKF is still cold,
* :class:`~.thermal_model.RCModel.predict_window_open` while an Exterior Opening
  pauses the plant.

Only the target room's curve is returned, one prediction per forecast block, on
the ``PLAN_DT_MINUTES`` grid the analytics service uses.
"""

from __future__ import annotations

import time
from dataclasses import replace

from ..const import (
    CLIMATE_MODE_COOL_ONLY,
    CLIMATE_MODE_HEAT_ONLY,
    DEFAULT_COMFORT_WEIGHT,
    DEFAULT_DEHUMIDIFY_RH_THRESHOLD,
    DEFAULT_MAX_DEHUMIDIFY_OVERCOOL_C,
    DEFAULT_OUTDOOR_COOLING_MIN,
    DEFAULT_OUTDOOR_HEATING_MAX,
    MODE_CIRCULATING,
    MODE_COOLING,
    MODE_HEATING,
    MODE_IDLE,
    get_central_plant_settings,
    is_override_active,
)
from ..utils.humidity_utils import predict_rh_at_temp
from .central_mpc import (
    DEFAULT_GUARDRAIL_COOL,
    DEFAULT_GUARDRAIL_HEAT,
    PLAN_DT_MINUTES,
    TEMP_CLAMP_CEILING_C,
    TEMP_CLAMP_FLOOR_C,
    CentralBangBang,
    CentralMPCOptimizer,
    CentralMPCPlan,
    RoomHorizonInput,
)
from .rebound import advance_rebound, drive_for_mode, mean_rebound
from .thermal_model import RCModel, ThermalEKF

# Placeholder for a missing outdoor reading, the same one the control path uses.
DEFAULT_OUTDOOR_TEMP_FALLBACK = 10.0
# The window-open simulation keeps the chart inside a physically plausible band.
WINDOW_OPEN_FLOOR_C = 5.0
WINDOW_OPEN_CEILING_C = 40.0
# A model idle drift is capped at this multiple of the rate the room actually showed.
OBSERVED_DRIFT_MULTIPLIER = 2.0


# ---------------------------------------------------------------------------
# Forecast disturbance series
# ---------------------------------------------------------------------------


def build_forecast_outdoor_series(
    forecast: list[dict] | None,
    current_outdoor: float | None,
    n_blocks: int,
) -> list[float]:
    """Build the outdoor temperature series for the analytics prediction.

    Block 0 uses ``current_outdoor`` (the real-time sensor reading). Subsequent
    blocks expand each hourly forecast entry to ``60 // PLAN_DT_MINUTES`` blocks.
    Padded with the last value when the horizon exceeds the forecast.  Without a
    reading (None) block 0 keeps the forecast and only gaps get the placeholder.
    """
    fallback = current_outdoor if current_outdoor is not None else DEFAULT_OUTDOOR_TEMP_FALLBACK
    if not forecast:
        return [fallback] * n_blocks

    blocks_per_hour = int(60 // PLAN_DT_MINUTES)
    series: list[float] = []
    for entry in forecast:
        temp = entry.get("temperature", fallback)
        series.extend([temp] * blocks_per_hour)
        if len(series) >= n_blocks:
            break

    if current_outdoor is not None:
        series[0] = current_outdoor

    while len(series) < n_blocks:
        series.append(series[-1])

    return series[:n_blocks]


def build_forecast_solar_series(
    latitude: float,
    longitude: float,
    forecast: list[dict],
    n_blocks: int,
    shading_factor: float = 1.0,
) -> list[float] | None:
    """Build the solar gain series for the analytics prediction.

    Expands hourly ``cloud_coverage`` entries to ``60 // PLAN_DT_MINUTES`` blocks
    before passing them to the solar module, matching the control path's series.
    Returns None when the home has no coordinates.
    """
    if latitude == 0.0 and longitude == 0.0:
        return None
    from .solar import build_solar_series

    cloud_series: list[float | None] | None = None
    if forecast:
        blocks_per_hour = int(60 // PLAN_DT_MINUTES)
        expanded: list[float | None] = []
        for entry in forecast:
            cloud = entry.get("cloud_coverage")
            expanded.extend([cloud] * blocks_per_hour)
        cloud_series = expanded[:n_blocks]
        while len(cloud_series) < n_blocks:
            cloud_series.append(cloud_series[-1] if cloud_series else None)

    series = build_solar_series(latitude, longitude, n_blocks, PLAN_DT_MINUTES, cloud_series=cloud_series)
    if shading_factor != 1.0:
        series = [value * shading_factor for value in series]
    return series


def compute_observed_idle_rate(all_points: list[dict]) -> float | None:
    """Observed idle drift from recent history in degC per plan block.

    Returns None when there is not enough recent idle data.  The cold-start
    forecast caps the model's idle drift at twice this rate, so the chart never
    predicts a faster change than the room actually showed recently.
    """
    if not all_points:
        return None
    now_ts = time.time()
    idle_points: list[tuple[float, float]] = []
    for point in reversed(all_points):
        if now_ts - point["ts"] > 3600:
            break
        mode = point.get("mode") or ""
        room_temp = point.get("room_temp")
        if room_temp is not None and mode in ("", MODE_IDLE):
            idle_points.append((point["ts"], room_temp))
    if len(idle_points) < 2:
        return None
    idle_points.sort(key=lambda item: item[0])
    dt_seconds = idle_points[-1][0] - idle_points[0][0]
    if dt_seconds <= 60:
        return None
    rate_per_second = (idle_points[-1][1] - idle_points[0][1]) / dt_seconds
    return rate_per_second * PLAN_DT_MINUTES * 60


# ---------------------------------------------------------------------------
# Target room input
# ---------------------------------------------------------------------------


def build_target_room_input(
    *,
    model: RCModel,
    current_temp: float,
    room_config: dict,
    target_forecast: list[dict],
    solar_series: list[float] | None = None,
    q_occupancy: float = 0.0,
    q_rebound: float | None = None,
    current_rh_pct: float | None = None,
    mpc_ready: bool = True,
) -> RoomHorizonInput:
    """Build the target room's :class:`RoomHorizonInput` from the analytics feed.

    The analytics forecast is the target room's own comfort curve, so the room is
    treated as priority-weighted and occupied for the whole horizon: its resolved
    dual targets are authoritative and the wrong-way penalties protect it.  The
    interior-door coupling factors, the pre-heat switch and the exterior-opening
    state come from the room configuration; the Guardrail Band defaults to the
    plant's 16..28 degC.
    """
    n_blocks = len(target_forecast)
    heat_targets: list[float | None] = []
    cool_targets: list[float | None] = []
    for forecast_block in target_forecast:
        block_target = forecast_block.get("target_temp")
        heat_targets.append(forecast_block.get("heat_target", block_target))
        cool_targets.append(forecast_block.get("cool_target", block_target))
    return RoomHorizonInput(
        area_id=str(room_config.get("area_id") or room_config.get("id") or "target"),
        model=model,
        T_init=current_temp,
        rh_init=current_rh_pct,
        rebound_init=0.0 if q_rebound is None else q_rebound,
        door_mix_factor=float(room_config.get("door_mix_factor", 1.0)),
        door_supply_factor=float(room_config.get("door_supply_factor", 1.0)),
        priority_series=[1.0] * n_blocks,
        occupied_prob_series=[1.0] * n_blocks,
        comfort_heat_series=heat_targets,
        comfort_cool_series=cool_targets,
        guardrail_heat=float(room_config.get("guardrail_heat", DEFAULT_GUARDRAIL_HEAT)),
        guardrail_cool=float(room_config.get("guardrail_cool", DEFAULT_GUARDRAIL_COOL)),
        solar_series=list(solar_series or []),
        occupancy_gain_series=[q_occupancy] * n_blocks,
        preheat_enabled=bool(room_config.get("preheat_enabled", True)),
        include_in_house_mean=bool(room_config.get("include_in_house_mean", True)),
        mpc_ready=mpc_ready,
    )


def _climate_capabilities(room_config: dict) -> tuple[bool, bool]:
    """The heating and cooling sides the room's climate mode allows at all."""
    if room_config.get("is_outdoor", False):
        return False, False
    climate_mode = room_config.get("climate_mode", "auto")
    return climate_mode != CLIMATE_MODE_COOL_ONLY, climate_mode != CLIMATE_MODE_HEAT_ONLY


def _legal_actions(
    room_config: dict,
    settings: dict,
    T_outdoor: float,
    outdoor_known: bool,
) -> set[str]:
    """Plant actions the room and the outdoor gates leave available for one block."""
    legal = {MODE_IDLE, MODE_CIRCULATING}
    can_heat, can_cool = _climate_capabilities(room_config)
    if outdoor_known and not is_override_active(room_config):
        if T_outdoor > settings.get("outdoor_heating_max", DEFAULT_OUTDOOR_HEATING_MAX):
            can_heat = False
        if T_outdoor < settings.get("outdoor_cooling_min", DEFAULT_OUTDOOR_COOLING_MIN):
            can_cool = False
    if can_heat:
        legal.add(MODE_HEATING)
    if can_cool:
        legal.add(MODE_COOLING)
    return legal


def _series_at(series: list[float], k: int, default: float) -> float:
    """Value of *series* at block *k*, holding its last value past the end."""
    if not series:
        return default
    return float(series[k] if k < len(series) else series[-1])


def _plan_curve(plan: CentralMPCPlan, area_id: str, n_blocks: int) -> list[float] | None:
    """The target room's predicted curve from a plan, one value per forecast block."""
    trajectory = plan.room_temperatures.get(area_id)
    if not trajectory:
        return None
    predictions = [round(value, 2) for value in trajectory[1 : n_blocks + 1]]
    if not predictions:
        return None
    while len(predictions) < n_blocks:
        predictions.append(predictions[-1])
    return predictions


# ---------------------------------------------------------------------------
# Simulation paths
# ---------------------------------------------------------------------------


def simulate_window_open_prediction(
    *,
    model: RCModel,
    estimator: ThermalEKF,
    target_forecast: list[dict],
    outdoor_series: list[float],
    current_temp: float,
    dt_minutes: float = PLAN_DT_MINUTES,
) -> list[float]:
    """Forecast with an Exterior Opening open: HVAC paused, pure heat exchange."""
    k_window = estimator.k_window
    temperature = current_temp
    predictions: list[float] = []
    for i in range(len(target_forecast)):
        temperature = model.predict_window_open(
            temperature,
            _series_at(outdoor_series, i, DEFAULT_OUTDOOR_TEMP_FALLBACK),
            k_window,
            dt_minutes,
        )
        temperature = max(WINDOW_OPEN_FLOOR_C, min(WINDOW_OPEN_CEILING_C, temperature))
        predictions.append(round(temperature, 2))
    return predictions


def simulate_house_prediction(
    *,
    target: RoomHorizonInput,
    siblings: list[RoomHorizonInput],
    outdoor_series: list[float],
    settings: dict,
    dt_minutes: float = PLAN_DT_MINUTES,
    horizon_blocks: int | None = None,
    can_heat: bool = True,
    can_cool: bool = True,
    override_active: bool = False,
    previous_mode: str = MODE_IDLE,
    mode_elapsed_minutes: float | None = None,
) -> list[float] | None:
    """Joint whole-home MPC rollout of the analytics forecast for the *target* room.

    With no siblings this is the one-room Central MPC run the control loop would
    make for this room: the plan's first block is the action to apply now and the
    curve follows the same coupled physics.  With siblings, their heat capacity,
    floor coupling and open windows change what the plant does to the target room,
    and Circulation can flatten opposing room errors without firing the plant.

    ``None`` when no plan can be built (no forecast blocks, no outdoor horizon),
    so the caller falls back to the cold-start forecast.
    """
    n_blocks = len(target.comfort_heat_series) if horizon_blocks is None else horizon_blocks
    if n_blocks <= 0 or not outdoor_series:
        return None
    room_ids = {target.area_id}
    house_rooms = [target]
    for sibling in siblings:
        if sibling.area_id in room_ids:
            continue
        room_ids.add(sibling.area_id)
        house_rooms.append(sibling)
    run_blocks = (
        max(0, int(mode_elapsed_minutes // dt_minutes))
        if previous_mode != MODE_IDLE and mode_elapsed_minutes is not None
        else 0
    )
    plant = get_central_plant_settings(settings)
    optimizer = CentralMPCOptimizer(
        rooms=house_rooms,
        T_outdoor_series=list(outdoor_series),
        dt_minutes=dt_minutes,
        comfort_weight=float(settings.get("comfort_weight", DEFAULT_COMFORT_WEIGHT)),
        horizon_blocks=n_blocks,
        can_heat=can_heat,
        can_cool=can_cool,
        override_active=override_active,
        humidity_comfort_enabled=bool(plant.get("humidity_comfort_enabled", True)),
        dehum_threshold=float(plant.get("dehumidify_rh_threshold", DEFAULT_DEHUMIDIFY_RH_THRESHOLD)),
        max_dehum_overcool_c=float(plant.get("max_dehumidify_overcool_c", DEFAULT_MAX_DEHUMIDIFY_OVERCOOL_C)),
        outdoor_cooling_min=float(settings.get("outdoor_cooling_min", DEFAULT_OUTDOOR_COOLING_MIN)),
        outdoor_heating_max=float(settings.get("outdoor_heating_max", DEFAULT_OUTDOOR_HEATING_MAX)),
        initial_action=previous_mode,
        initial_run_blocks=run_blocks,
    )
    plan = optimizer.optimize()
    return _plan_curve(plan, target.area_id, n_blocks)


def simulate_bangbang_prediction(
    *,
    target: RoomHorizonInput,
    outdoor_series: list[float],
    settings: dict,
    room_config: dict,
    dt_minutes: float = PLAN_DT_MINUTES,
    horizon_blocks: int | None = None,
    previous_mode: str = MODE_IDLE,
    all_points: list[dict] | None = None,
    outdoor_known: bool = True,
) -> list[float]:
    """Cold-start forecast: the room advanced block by block under CentralBangBang.

    Until the EKF is trustworthy the plan cannot be optimised, so every block
    re-decides with :class:`CentralBangBang` on the simulated temperature and the
    coupled :class:`RCModel` advances the air node with the signed forced-air
    charge of :mod:`.rebound` (``drive_for_mode``/``mean_rebound``/
    ``advance_rebound``).  An idle block's model drift is capped at twice the rate
    the room actually showed recently, so the chart never predicts a faster change
    than the observations support.
    """
    blocks = len(target.comfort_heat_series) if horizon_blocks is None else horizon_blocks
    blocks = min(blocks, len(outdoor_series))
    if blocks <= 0:
        return []
    observed_idle_rate = compute_observed_idle_rate(all_points or [])
    bangbang = CentralBangBang()
    bangbang.active_action = previous_mode
    temperature = target.T_init
    humidity = target.rh_init
    charge = target.rebound_init
    predictions: list[float] = []
    for k in range(blocks):
        T_outdoor = _series_at(outdoor_series, k, DEFAULT_OUTDOOR_TEMP_FALLBACK)
        legal = _legal_actions(room_config, settings, T_outdoor, outdoor_known)
        current = replace(target, T_init=temperature, rh_init=humidity)
        action = bangbang.decide(rooms=[current], k=k, legal_actions=legal)
        bangbang.active_action = action
        drive = drive_for_mode(action)
        if action == MODE_HEATING:
            Q_active = target.model.Q_heat
        elif action == MODE_COOLING:
            Q_active = -target.model.Q_cool
        else:
            Q_active = 0.0
        T_next = target.model.predict(
            temperature,
            T_outdoor,
            Q_active,
            dt_minutes,
            T_house=temperature,
            u_fan=1.0 if action == MODE_CIRCULATING else 0.0,
            door_mix_factor=target.door_mix_factor,
            door_supply_factor=target.door_supply_factor,
            q_solar=_series_at(target.solar_series, k, 0.0),
            q_occupancy=_series_at(target.occupancy_gain_series, k, 0.0),
            q_rebound=mean_rebound(charge, drive, dt_minutes),
            mode=action,
        )
        T_next = max(
            min(temperature, TEMP_CLAMP_FLOOR_C),
            min(T_next, max(temperature, TEMP_CLAMP_CEILING_C)),
        )
        if action == MODE_IDLE:
            T_next = _cap_idle_drift(temperature, T_next, observed_idle_rate)
        if humidity is not None:
            humidity = predict_rh_at_temp(
                temperature,
                humidity,
                T_next,
                cooling_active=(action == MODE_COOLING),
                dt_minutes=dt_minutes,
            )
        charge = advance_rebound(charge, drive, dt_minutes)
        temperature = T_next
        predictions.append(round(temperature, 2))
    return predictions


def _cap_idle_drift(current: float, predicted: float, observed_rate: float | None) -> float:
    """Cap a model idle drift at twice the rate the room actually showed recently."""
    if observed_rate is None:
        return predicted
    max_delta = observed_rate * OBSERVED_DRIFT_MULTIPLIER
    delta = predicted - current
    if delta < 0.0 and max_delta < 0.0 and delta < max_delta:
        return current + max_delta
    if delta > 0.0 and max_delta > 0.0 and delta > max_delta:
        return current + max_delta
    return predicted


def simulate_prediction(
    *,
    model: RCModel,
    estimator: ThermalEKF,
    target_forecast: list[dict],
    outdoor_series: list[float],
    current_temp: float,
    window_open: bool,
    mpc_active: bool,
    room_config: dict,
    settings: dict,
    all_points: list[dict],
    solar_series: list[float] | None = None,
    q_occupancy: float = 0.0,
    q_rebound: float | None = None,
    current_rh_pct: float | None = None,
    previous_mode: str = MODE_IDLE,
    mode_elapsed_minutes: float | None = None,
    outdoor_known: bool = True,
    sibling_rooms: list[RoomHorizonInput] | None = None,
) -> list[float]:
    """Simulate the analytics forecast curve for one Thermal Zone.

    Dispatch:
    - Window open: pure heat exchange, HVAC paused.
    - ``mpc_active``: a Central MPC run (one room, or the whole house when
      ``sibling_rooms`` is passed) via :class:`CentralMPCOptimizer`.
    - Otherwise (cold EKF, bangbang control mode): the block-by-block
      :class:`CentralBangBang` forecast with the coupled RC model and the signed
      forced-air relaxation of :mod:`.rebound`.

    ``previous_mode``/``mode_elapsed_minutes`` seed the run in progress so a room
    inside the start hysteresis keeps conditioning in the forecast; the equipment
    guards' min-run timers do not otherwise shape the curve.  ``estimator`` supplies
    the learned window multiplier for the window-open path.
    """
    if not target_forecast:
        return []
    if window_open:
        return simulate_window_open_prediction(
            model=model,
            estimator=estimator,
            target_forecast=target_forecast,
            outdoor_series=outdoor_series,
            current_temp=current_temp,
        )

    target_room = build_target_room_input(
        model=model,
        current_temp=current_temp,
        room_config=room_config,
        target_forecast=target_forecast,
        solar_series=solar_series,
        q_occupancy=q_occupancy,
        q_rebound=q_rebound,
        current_rh_pct=current_rh_pct,
        mpc_ready=mpc_active,
    )
    if mpc_active:
        siblings = list(sibling_rooms or [])
        can_heat, can_cool = _climate_capabilities(room_config)
        curve = simulate_house_prediction(
            target=target_room,
            siblings=siblings,
            outdoor_series=outdoor_series,
            settings=settings,
            horizon_blocks=len(target_forecast),
            # The whole house may need a side the target room itself cannot use.
            can_heat=True if siblings else can_heat,
            can_cool=True if siblings else can_cool,
            override_active=is_override_active(room_config),
            previous_mode=previous_mode,
            mode_elapsed_minutes=mode_elapsed_minutes,
        )
        if curve is not None:
            return curve
    return simulate_bangbang_prediction(
        target=target_room,
        outdoor_series=outdoor_series,
        settings=settings,
        room_config=room_config,
        horizon_blocks=len(target_forecast),
        previous_mode=previous_mode,
        all_points=all_points,
        outdoor_known=outdoor_known,
    )
