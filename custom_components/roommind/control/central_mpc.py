"""Joint whole-home MPC and cold-start controller for RoomMind.

The Central Plant is one ducted forced-air system -- a furnace, a heat pump and a
shared blower -- serving every Thermal Zone at once.  A per-room controller cannot
see that heating one room also heats the room next to it, that cooling an already
cool occupied room to help a warm one is a losing trade, or that running the
blower alone can equalize two rooms' opposing errors for a fraction of the
compressor's or furnace's energy.  This module plans the plant as a whole:

* :class:`CentralMPCOptimizer` advances every room *and* the House Reference
  Temperature together over the discrete plant actions (``idle``, ``heating``,
  ``cooling``, ``circulating``), scoring each step by Room Priority weighted
  comfort, Guardrail Band safety, wrong-way over-conditioning and energy.
* :func:`step_house_block` is the coupled multi-room step built on the 9-state
  :class:`~.thermal_model.RCModel` (door coupling, blower mixing and the signed
  forced-air air-vs-mass relaxation of :mod:`.rebound`), so the plan uses the
  same physics the EKFs learn.
* :class:`CentralBangBang` keeps the home safe while the EKFs are still cold: it
  works from priority-weighted errors and the occupied temperature spread, and
  chooses Circulation when two occupied rooms disagree in opposite directions.
* :class:`CentralMPCController` picks between the two paths, hands rooms whose
  model is not MPC-ready the forced-air priors of
  :class:`~.thermal_model.RoomModelManager`, and rolls the plan forward for the
  forecast charts.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ..const import (
    BANGBANG_COOL_HYSTERESIS,
    BANGBANG_HEAT_HYSTERESIS,
    CONTROL_MODE_BANGBANG,
    CONTROL_MODE_MPC,
    DEFAULT_CHANGEOVER_MINUTES,
    DEFAULT_COMFORT_WEIGHT,
    DEFAULT_DEHUMIDIFY_RH_THRESHOLD,
    DEFAULT_MAX_DEHUMIDIFY_OVERCOOL_C,
    DEFAULT_MIN_FAN_MINUTES,
    DEFAULT_MIN_OFF_MINUTES,
    DEFAULT_MIN_RUN_MINUTES,
    DEFAULT_OUTDOOR_COOLING_MIN,
    DEFAULT_OUTDOOR_HEATING_MAX,
    MODE_CIRCULATING,
    MODE_COOLING,
    MODE_HEATING,
    MODE_IDLE,
    get_central_plant_settings,
)
from ..utils.humidity_utils import (
    dehumidify_cool_offset_c,
    effective_temperature_c,
    predict_rh_at_temp,
)
from .rebound import advance_rebound, drive_for_mode, mean_rebound
from .thermal_model import RCModel, RoomModelManager, ThermalEKF
from .topology import reference_temperature

# Pre-heat lead at the default comfort weight; the slider scales it to none at
# the efficiency end.  The deadline planner never looks further ahead than this.
PREHEAT_MAX_LEAD_HOURS = 12.0
# One plan block: the control loop's cadence.
PLAN_DT_MINUTES = 5.0
# Lookahead a single decision is judged over: long enough to see the forced-air
# relaxation after a run (tau ~10 min) and the blower mixing the house air.
LOOKAHEAD_BLOCKS = 8
# Planning horizon bounds: two hours by default, half a day at most.
MIN_HORIZON_BLOCKS = 24
MAX_HORIZON_BLOCKS = 144
# A blower run costs a fraction of a compressor/furnace run (~300 W vs ~4 kW).
FAN_ENERGY_FRACTION = 0.08
# Weight of a Guardrail Band breach relative to the priority-weighted comfort cost.
GUARDRAIL_WEIGHT = 0.08
# Extra quadratic weight when an active run pushes an occupied room past its
# opposite comfort bound (heating a room above its cool limit, or cooling one
# below its heat limit).
WRONG_WAY_PENALTY = 2.0
# Sub-tenth noise must not short-start the compressor or furnace.
START_HYSTERESIS_C = 0.15
# Data gates for the joint MPC to trust a room's model (9-state EKF samples).
MPC_MIN_IDLE_UPDATES = 40
MPC_MIN_ACTIVE_UPDATES = 12
MPC_MAX_PREDICTION_STD = 0.50
# An arrival pre-heat may miss the comfort target by this much and still count as "on time".
ARRIVAL_DEADLINE_MARGIN_C = 0.2
# Pre-heating/pre-cooling for an arriving room may push an already occupied room
# at most this far past its opposite comfort bound.
OPPOSITE_BOUND_TOLERANCE_C = 0.5
# Dehumidification may lower the effective cool target at most to here above the heat bound.
DEHUMIDIFY_FLOOR_MARGIN_C = 0.5
# Occupancy probability that marks a room as occupied for penalties and spreads.
OCCUPIED_PROB_THRESHOLD = 0.5
# A cooling room and a room this close to its heat floor disagree: mix the air.
WARM_FLOOR_MARGIN_C = 0.3
# Occupancy probability that marks a room's arrival for the pre-heat deadline.
ARRIVAL_OCCUPANCY_THRESHOLD = 0.6
# Guardrail Band defaults (freeze protection / overheat protection).
DEFAULT_GUARDRAIL_HEAT = 16.0
DEFAULT_GUARDRAIL_COOL = 28.0
# Fallback House Reference Temperature when no room reports one.
DEFAULT_HOUSE_TEMP_C = 21.0
# Numerical safety clamp of a simulated room temperature around its start.
TEMP_CLAMP_FLOOR_C = -5.0
TEMP_CLAMP_CEILING_C = 45.0
# Equipment protection in blocks, from the Central Plant defaults.
MIN_RUN_BLOCKS = round(DEFAULT_MIN_RUN_MINUTES / PLAN_DT_MINUTES)
MIN_OFF_BLOCKS = round(DEFAULT_MIN_OFF_MINUTES / PLAN_DT_MINUTES)
MIN_FAN_BLOCKS = round(DEFAULT_MIN_FAN_MINUTES / PLAN_DT_MINUTES)
CHANGEOVER_BLOCKS = round(DEFAULT_CHANGEOVER_MINUTES / PLAN_DT_MINUTES)
# "Long ago" for the simulated guard timers.
_LONG_AGO = 1_000_000
# Cost comparison tolerance (float noise between equal candidates).
_COST_EPS = 1e-9
# Discomfort below this is model noise, not a reason to act.
_DISCOMFORT_EPS = 1e-9


# ---------------------------------------------------------------------------
# Per-room inputs and the plan
# ---------------------------------------------------------------------------


@dataclass
class RoomHorizonInput:
    """One Thermal Zone's initial state and per-block horizon series.

    Every series is indexed by plan block and holds its last value past its end
    (an empty series falls back to the model's default for that block).  The
    per-block series come from the coordinator: Room Priority and Occupancy Trend
    (:meth:`~.managers.occupancy_manager.OccupancyManager.build_priority_series`),
    the resolved dual targets, the solar forecast and the interior door state.
    """

    area_id: str
    model: RCModel
    T_init: float
    rh_init: float | None = None
    #: Signed forced-air charge in [-1, +1] (:mod:`~.control.rebound`).
    rebound_init: float = 0.0
    #: Scaling of the door coupling: 1.0 = open door, lower = closed interior door.
    door_mix_factor: float = 1.0
    #: Scaling of the HVAC supply into the room air node (duct losses, mixing).
    door_supply_factor: float = 1.0
    priority_series: list[float] = field(default_factory=list)
    occupied_prob_series: list[float] = field(default_factory=list)
    comfort_heat_series: list[float | None] = field(default_factory=list)
    comfort_cool_series: list[float | None] = field(default_factory=list)
    guardrail_heat: float = DEFAULT_GUARDRAIL_HEAT
    guardrail_cool: float = DEFAULT_GUARDRAIL_COOL
    solar_series: list[float] = field(default_factory=list)
    occupancy_gain_series: list[float] = field(default_factory=list)
    preheat_enabled: bool = True
    #: False when the room's exterior window is open: it would poison the mean.
    include_in_house_mean: bool = True
    #: Neighbour weights for this room's local reference, restricted to rooms
    #: contributing to the house mean. Empty means use the global mean.
    neighbours: Mapping[str, float] = field(default_factory=dict)
    #: Connected component of the topology graph, None when undeclared.
    mixing_group: int | None = None
    #: Whether this room's model passed the joint-MPC data gates.
    mpc_ready: bool = True


@dataclass
class CentralMPCPlan:
    """The joint whole-home plan: one plant action per block and its trajectories."""

    actions: list[str]
    room_temperatures: dict[str, list[float]]
    house_temperatures: list[float]
    driving_rooms: list[str] = field(default_factory=list)
    decision_reason: str = "idle"
    mpc_active: bool = False
    dt_minutes: float = PLAN_DT_MINUTES
    #: ``{area_id: {"mode": "heating"|"cooling", "target": degC, "at": minutes}}``
    #: for the rooms the current block pre-heats or pre-cools for (UI badges).
    preheat_info: dict[str, dict[str, Any]] = field(default_factory=dict)

    def current_action(self) -> str:
        """Plant action of the current (first) block."""
        if not self.actions:
            return MODE_IDLE
        return self.actions[0]


# ---------------------------------------------------------------------------
# Small series helpers
# ---------------------------------------------------------------------------


def _series_at(series: Sequence[float], k: int, default: float) -> float:
    """Value of *series* at block *k*, holding its last value past the end."""
    if not series:
        return default
    index = k if k < len(series) else len(series) - 1
    return float(series[index])


def _optional_at(series: Sequence[float | None], k: int, default: float | None) -> float | None:
    """Value of an optional *series* at block *k*, holding its last value past the end."""
    if not series:
        return default
    value = series[k] if k < len(series) else series[-1]
    return None if value is None else float(value)


def _as_float(value: object, default: float) -> float:
    """Coerce a settings value to float, falling back for missing/corrupt entries."""
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


def _blocks_from_minutes(value: object, dt_minutes: float, default_minutes: float) -> int:
    """Convert an equipment-protection duration to whole plan blocks (at least one)."""
    minutes = _as_float(value, default_minutes)
    if minutes <= 0.0 or dt_minutes <= 0.0:
        return 1
    return max(1, round(minutes / dt_minutes))


def preheat_max_blocks(
    comfort_weight: float,
    dt_minutes: float = PLAN_DT_MINUTES,
    *,
    enabled: bool = True,
) -> int | None:
    """Longest pre-heat lead the comfort/efficiency slider allows, in plan blocks.

    At the default comfort weight and above the arrival deadline may plan up to the
    whole horizon; towards the efficiency end the allowed lead shrinks to none.
    ``enabled`` is the room's pre-heat switch: off means no lead at all.
    """
    if not enabled:
        return 0
    if comfort_weight >= DEFAULT_COMFORT_WEIGHT:
        return None
    hours = PREHEAT_MAX_LEAD_HOURS * max(0.0, comfort_weight) / DEFAULT_COMFORT_WEIGHT
    return int(hours * 60 / dt_minutes)


# ---------------------------------------------------------------------------
# Coupled multi-room physics
# ---------------------------------------------------------------------------


def compute_house_temperature(
    rooms: Sequence[RoomHorizonInput],
    temperatures: Mapping[str, float],
    extra_house_temp: float | None = None,
) -> float:
    """House Reference Temperature from the rooms that mix into the house air.

    Rooms with ``include_in_house_mean=False`` (an open exterior window) are left
    out: they are not part of the conditioned envelope.  ``extra_house_temp``
    adds the thermostat's own hallway reading when it senses a place no monitored
    room covers.  Without usable rooms the mean of all rooms, else ``21.0``.
    """
    included = [
        temperatures[room.area_id] for room in rooms if room.include_in_house_mean and room.area_id in temperatures
    ]
    if extra_house_temp is not None:
        included.append(float(extra_house_temp))
    if not included:
        included = [temperatures[room.area_id] for room in rooms if room.area_id in temperatures]
    if not included:
        return DEFAULT_HOUSE_TEMP_C
    return sum(included) / len(included)


def compute_room_reference_temperatures(
    rooms: Sequence[RoomHorizonInput],
    temperatures: Mapping[str, float],
    extra_house_temp: float | None = None,
) -> dict[str, float]:
    """Per-room local reference temperatures, falling back to the house mean.

    The include set is the rooms with ``include_in_house_mean`` present in
    *temperatures*; each room's declared neighbours are restricted to it (and
    never include the room itself). A room with no usable neighbours gets
    ``compute_house_temperature(...)`` exactly, so an unconfigured house behaves
    bit-identically to the previous single-reference model. The global house
    trajectory is still published separately from :func:`compute_house_temperature`.
    """
    fallback = compute_house_temperature(rooms, temperatures, extra_house_temp)
    included = {room.area_id for room in rooms if room.include_in_house_mean and room.area_id in temperatures}
    result: dict[str, float] = {}
    for room in rooms:
        neighbours = {
            neighbour: weight
            for neighbour, weight in room.neighbours.items()
            if neighbour in included and neighbour != room.area_id
        }
        result[room.area_id] = reference_temperature(neighbours, temperatures, fallback)
    return result


def step_house_block(
    rooms: Sequence[RoomHorizonInput],
    temperatures: Mapping[str, float],
    humidities: Mapping[str, float | None],
    charges: Mapping[str, float],
    action: str,
    T_outdoor: float,
    k: int,
    *,
    dt_minutes: float = PLAN_DT_MINUTES,
    extra_house_temp: float | None = None,
) -> tuple[dict[str, float], dict[str, float | None], dict[str, float]]:
    """Advance every room and the shared house air one block under *action*.

    The plant action drives every room at once: the furnace or heat pump adds its
    learned rate to each ``door_supply_factor``-scaled air node, the blower mixes
    each room with the House Reference Temperature, and the signed forced-air
    charge relaxes toward the mode's drive (see :mod:`~.control.rebound`).  The
    returned dictionaries are keyed by ``area_id``.

    Args:
        rooms: The rooms participating in the joint step.
        temperatures: Current room temperatures [degC].
        humidities: Current room relative humidities [%], None where unmeasured.
        charges: Current signed forced-air charges [-1, +1].
        action: Plant action for this block.
        T_outdoor: Outdoor temperature [degC].
        k: Block index into each room's disturbance series.
        dt_minutes: Block duration in minutes.
        extra_house_temp: Thermostat hallway reading added to the house mean.

    Returns:
        ``(temperatures, humidities, charges)`` after the block.
    """
    drive = drive_for_mode(action)
    u_fan = 1.0 if action == MODE_CIRCULATING else 0.0
    house_temp = compute_house_temperature(rooms, temperatures, extra_house_temp)
    room_refs = compute_room_reference_temperatures(rooms, temperatures, extra_house_temp)
    next_temperatures: dict[str, float] = {}
    next_humidities: dict[str, float | None] = {}
    next_charges: dict[str, float] = {}
    for room in rooms:
        T_i = temperatures[room.area_id]
        rh_i = humidities.get(room.area_id)
        charge = charges.get(room.area_id, room.rebound_init)
        if action == MODE_HEATING:
            Q_act = room.model.Q_heat
        elif action == MODE_COOLING:
            Q_act = -room.model.Q_cool
        else:
            Q_act = 0.0
        q_rebound = mean_rebound(charge, drive, dt_minutes)
        charge_next = advance_rebound(charge, drive, dt_minutes)
        T_next = room.model.predict(
            T_i,
            T_outdoor,
            Q_act,
            dt_minutes,
            T_house=room_refs.get(room.area_id, house_temp),
            u_fan=u_fan,
            door_mix_factor=room.door_mix_factor,
            door_supply_factor=room.door_supply_factor,
            q_solar=_series_at(room.solar_series, k, 0.0),
            q_occupancy=_series_at(room.occupancy_gain_series, k, 0.0),
            q_rebound=q_rebound,
            mode=action,
        )
        next_temperatures[room.area_id] = max(
            min(T_i, TEMP_CLAMP_FLOOR_C),
            min(T_next, max(T_i, TEMP_CLAMP_CEILING_C)),
        )
        if rh_i is not None:
            next_humidities[room.area_id] = predict_rh_at_temp(
                T_i,
                rh_i,
                next_temperatures[room.area_id],
                cooling_active=(action == MODE_COOLING),
                dt_minutes=dt_minutes,
            )
        else:
            next_humidities[room.area_id] = None
        next_charges[room.area_id] = charge_next
    return next_temperatures, next_humidities, next_charges


def bounded_dehumidified_cool_target(
    heat_target: float | None,
    cool_target: float | None,
    guardrail_heat: float,
    dehumidify_offset_c: float,
) -> float | None:
    """The room's effective cool target after the bounded dehumidification offset.

    Dehumidifying over-cools a room slightly to condense moisture out of its air.
    The floor keeps the target at least ``DEHUMIDIFY_FLOOR_MARGIN_C`` above the
    room's heat target (or above the Guardrail Band's heating floor when the room
    has no heat target), so a dehumidify run can never flip the plant's direction
    inside one room.
    """
    if cool_target is None:
        return None
    floor = (
        heat_target + DEHUMIDIFY_FLOOR_MARGIN_C
        if heat_target is not None
        else guardrail_heat + DEHUMIDIFY_FLOOR_MARGIN_C
    )
    return max(floor, cool_target - max(0.0, dehumidify_offset_c))


# ---------------------------------------------------------------------------
# Whole-home judgements shared by both controllers
# ---------------------------------------------------------------------------


def guardrail_breach_direction(
    rooms: Sequence[RoomHorizonInput],
    temperatures: Mapping[str, float],
) -> str | None:
    """Direction that repairs the worst Guardrail Band breach, None when all safe.

    Guardrails are the wide background limits that keep empty rooms from freezing
    or overheating; a breach outranks energy and is acted on even at priority 0.
    """
    worst = 0.0
    direction: str | None = None
    for room in rooms:
        if room.area_id not in temperatures:
            continue
        T_i = temperatures[room.area_id]
        if T_i < room.guardrail_heat:
            breach = room.guardrail_heat - T_i
            if breach > worst:
                worst, direction = breach, MODE_HEATING
        elif T_i > room.guardrail_cool:
            breach = T_i - room.guardrail_cool
            if breach > worst:
                worst, direction = breach, MODE_COOLING
    return direction


def start_discomfort_reached(
    rooms: Sequence[RoomHorizonInput],
    action: str,
    temperatures: Mapping[str, float],
    humidities: Mapping[str, float | None],
    k: int = 0,
    *,
    humidity_comfort_enabled: bool = True,
) -> bool:
    """Whether a HEATING/COOLING start clears the start hysteresis in one room.

    At least one priority-weighted room must be measurably uncomfortable in the
    direction *action* would move it (or breach its Guardrail Band).  Sub-tenth
    drift must not short-start the compressor or furnace.
    """
    for room in rooms:
        if room.area_id not in temperatures:
            continue
        T_i = temperatures[room.area_id]
        if action == MODE_HEATING:
            if T_i < room.guardrail_heat:
                return True
            if _series_at(room.priority_series, k, 1.0) <= 0.0:
                continue
            target = _optional_at(room.comfort_heat_series, k, None)
            if target is None:
                continue
            T_eff = effective_temperature_c(
                T_i,
                humidities.get(room.area_id),
                enabled=humidity_comfort_enabled,
            )
            if target - T_eff >= START_HYSTERESIS_C:
                return True
        elif action == MODE_COOLING:
            if T_i > room.guardrail_cool:
                return True
            if _series_at(room.priority_series, k, 1.0) <= 0.0:
                continue
            target = _optional_at(room.comfort_cool_series, k, None)
            if target is None:
                continue
            T_eff = effective_temperature_c(
                T_i,
                humidities.get(room.area_id),
                enabled=humidity_comfort_enabled,
            )
            if T_eff - target >= START_HYSTERESIS_C:
                return True
    return False


def driving_rooms(
    rooms: Sequence[RoomHorizonInput],
    action: str,
    temperatures: Mapping[str, float],
    humidities: Mapping[str, float | None],
    house_temp: float,
    k: int = 0,
    *,
    humidity_comfort_enabled: bool = True,
) -> list[str]:
    """Rooms contributing the largest discomfort in the chosen direction.

    Heating and cooling rank the rooms by their positive error in that direction
    (a Guardrail Band breach counts as its own error); circulation ranks by the
    largest distance from the House Reference Temperature.  Idle has no direction.
    """
    if action == MODE_IDLE:
        return []
    scored: list[tuple[str, float]] = []
    for room in rooms:
        if room.area_id not in temperatures:
            continue
        T_i = temperatures[room.area_id]
        if action == MODE_CIRCULATING:
            score = abs(house_temp - T_i)
        else:
            T_eff = effective_temperature_c(
                T_i,
                humidities.get(room.area_id),
                enabled=humidity_comfort_enabled,
            )
            if action == MODE_HEATING:
                target = _optional_at(room.comfort_heat_series, k, None)
                score = max(0.0, target - T_eff) if target is not None else 0.0
                if T_i < room.guardrail_heat:
                    score = max(score, room.guardrail_heat - T_i)
            else:
                target = _optional_at(room.comfort_cool_series, k, None)
                score = max(0.0, T_eff - target) if target is not None else 0.0
                if T_i > room.guardrail_cool:
                    score = max(score, T_i - room.guardrail_cool)
        if score > _DISCOMFORT_EPS:
            scored.append((room.area_id, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [area_id for area_id, _score in scored]


def is_room_mpc_ready(estimator: ThermalEKF, can_heat: bool, can_cool: bool) -> bool:
    """Whether a room's EKF has seen enough excitation for the joint MPC.

    The joint rollout is only as good as the model it advances: a room needs
    ``MPC_MIN_IDLE_UPDATES`` idle samples (the house coupling and outdoor loss),
    ``MPC_MIN_ACTIVE_UPDATES`` samples of the plant modes it can actually use,
    and a prediction uncertainty below ``MPC_MAX_PREDICTION_STD``.  A mode with
    fewer than two samples is skipped rather than failing the room: a heating-only
    season must not require cooling data.
    """
    n_idle = int(getattr(estimator, "_n_idle", 0))
    if n_idle < MPC_MIN_IDLE_UPDATES:
        return False
    n_heating = int(getattr(estimator, "_n_heating", 0))
    n_cooling = int(getattr(estimator, "_n_cooling", 0))
    n_circulating = int(getattr(estimator, "_n_circulating", 0))
    active = n_circulating
    if can_heat:
        active += n_heating
    if can_cool:
        active += n_cooling
    if active < MPC_MIN_ACTIVE_UPDATES:
        return False
    stds = [estimator.prediction_std(0.0, 20.0, 15.0, PLAN_DT_MINUTES)]
    if can_heat and n_heating >= 2:
        stds.append(
            estimator.prediction_std(
                estimator._x[ThermalEKF._STATE_BETA_H],
                20.0,
                10.0,
                PLAN_DT_MINUTES,
            )
        )
    if can_cool and n_cooling >= 2:
        stds.append(
            estimator.prediction_std(
                -estimator._x[ThermalEKF._STATE_BETA_C],
                20.0,
                25.0,
                PLAN_DT_MINUTES,
            )
        )
    return max(stds) <= MPC_MAX_PREDICTION_STD


# ---------------------------------------------------------------------------
# Equipment-guard simulation during the rollout
# ---------------------------------------------------------------------------


@dataclass
class _PlantRunState:
    """Simulated equipment-protection state while the rollout advances the plan.

    The Central Plant's min-run, min-off, min-fan and changeover guards outlive a
    single block.  Block 0 uses the real guard state (:attr:`CentralMPCOptimizer.
    initial_legal_actions`); later blocks replay the same rules so the plan never
    shows a plant move the equipment could not actually make.
    """

    last_action: str = MODE_IDLE
    run_blocks: int = 0
    off_blocks: int = _LONG_AGO
    since_heat: int = _LONG_AGO
    since_cool: int = _LONG_AGO

    def candidates(
        self,
        base: set[str],
        *,
        min_run_blocks: int,
        min_fan_blocks: int,
        min_off_blocks: int,
        changeover_blocks: int,
        wet_coil_locked_blocks: int,
    ) -> set[str]:
        """Plant actions the guards allow for the next block."""
        required = self._required_run(min_run_blocks, min_fan_blocks)
        if required is not None and self.run_blocks < required:
            return {self.last_action} if self.last_action in base else {MODE_IDLE}
        allowed = set(base)
        allowed.add(MODE_IDLE)
        running = self.last_action in (MODE_HEATING, MODE_COOLING)
        if not running and self.off_blocks < min_off_blocks:
            allowed.discard(MODE_HEATING)
            allowed.discard(MODE_COOLING)
        if self.since_heat < changeover_blocks:
            allowed.discard(MODE_COOLING)
        if self.since_cool < changeover_blocks:
            allowed.discard(MODE_HEATING)
        if wet_coil_locked_blocks > 0 and self.since_cool < wet_coil_locked_blocks:
            allowed.discard(MODE_CIRCULATING)
        return allowed

    def observe(self, action: str) -> None:
        """Record one completed block of *action*."""
        if action == self.last_action:
            self.run_blocks += 1
        else:
            self.last_action = action
            self.run_blocks = 1
        if action == MODE_HEATING:
            self.since_heat = 0
            self.since_cool = min(self.since_cool + 1, _LONG_AGO)
            self.off_blocks = 0
        elif action == MODE_COOLING:
            self.since_cool = 0
            self.since_heat = min(self.since_heat + 1, _LONG_AGO)
            self.off_blocks = 0
        else:
            self.since_heat = min(self.since_heat + 1, _LONG_AGO)
            self.since_cool = min(self.since_cool + 1, _LONG_AGO)
            self.off_blocks = min(self.off_blocks + 1, _LONG_AGO)

    def _required_run(self, min_run_blocks: int, min_fan_blocks: int) -> int | None:
        """Blocks the current action must still run, None when it may change."""
        if self.last_action in (MODE_HEATING, MODE_COOLING):
            return max(1, min_run_blocks)
        if self.last_action == MODE_CIRCULATING:
            return max(1, min_fan_blocks)
        return None


# ---------------------------------------------------------------------------
# Joint whole-home MPC
# ---------------------------------------------------------------------------


@dataclass
class CentralMPCOptimizer:
    """Joint whole-home MPC over the Central Plant's discrete actions.

    Each block is chosen by replaying every legal candidate action over
    ``LOOKAHEAD_BLOCKS`` blocks of the coupled multi-room model and taking the
    cheapest; the chosen action then advances the whole house one block for real
    before the next decision.  The horizon is a rolling plan: block 0 is the
    action the controller applies now.
    """

    rooms: list[RoomHorizonInput]
    T_outdoor_series: list[float]
    dt_minutes: float = PLAN_DT_MINUTES
    comfort_weight: float = float(DEFAULT_COMFORT_WEIGHT)
    horizon_blocks: int = MIN_HORIZON_BLOCKS
    can_heat: bool = True
    can_cool: bool = True
    humidity_comfort_enabled: bool = True
    dehum_threshold: float = float(DEFAULT_DEHUMIDIFY_RH_THRESHOLD)
    max_dehum_overcool_c: float = float(DEFAULT_MAX_DEHUMIDIFY_OVERCOOL_C)
    outdoor_cooling_min: float = float(DEFAULT_OUTDOOR_COOLING_MIN)
    outdoor_heating_max: float = float(DEFAULT_OUTDOOR_HEATING_MAX)
    override_active: bool = False
    extra_house_temp: float | None = None
    #: Actions the Central Plant guard allows right now (block 0 only).
    initial_legal_actions: set[str] | None = None
    #: Plant guard hold: the action the equipment state forces right now.
    forced_action: str | None = None
    initial_action: str = MODE_IDLE
    initial_run_blocks: int = 0
    #: Blocks the plant has been off after a conditioning run (``_LONG_AGO`` = not recently).
    initial_off_blocks: int = _LONG_AGO
    min_run_blocks: int = MIN_RUN_BLOCKS
    min_off_blocks: int = MIN_OFF_BLOCKS
    min_fan_blocks: int = MIN_FAN_BLOCKS
    changeover_blocks: int = CHANGEOVER_BLOCKS
    #: Blocks Circulation stays locked out while the cooling coil is still wet.
    wet_coil_locked_blocks: int = 0

    def __post_init__(self) -> None:
        if self.dt_minutes <= 0.0:
            raise ValueError("dt_minutes must be positive")
        self.horizon_blocks = max(1, min(int(self.horizon_blocks), MAX_HORIZON_BLOCKS))

    # -- public API ----------------------------------------------------------

    def optimize(self) -> CentralMPCPlan:
        """Build the joint plan, block 0 being the action to apply now."""
        rooms = list(self.rooms)
        n_blocks = self._horizon()
        if not rooms or n_blocks <= 0:
            return self._empty_plan(rooms)

        temperatures = {room.area_id: room.T_init for room in rooms}
        humidities = {room.area_id: room.rh_init for room in rooms}
        charges = {room.area_id: room.rebound_init for room in rooms}
        initial_temperatures = dict(temperatures)
        initial_humidities = dict(humidities)
        initial_house = compute_house_temperature(rooms, temperatures, self.extra_house_temp)
        room_temperatures: dict[str, list[float]] = {room.area_id: [room.T_init] for room in rooms}
        house_temperatures = [initial_house]

        run_state = _PlantRunState(
            last_action=self.initial_action,
            run_blocks=max(0, self.initial_run_blocks),
            off_blocks=max(0, self.initial_off_blocks),
            since_heat=0 if self.initial_action == MODE_HEATING else _LONG_AGO,
            since_cool=0 if self.initial_action == MODE_COOLING else _LONG_AGO,
        )
        actions: list[str] = []
        decision_reason = "idle"
        preheat_info: dict[str, dict[str, Any]] = {}

        for k in range(n_blocks):
            if k == 0:
                action, decision_reason, preheat_info = self._decide_first_block(
                    temperatures,
                    humidities,
                    charges,
                )
            else:
                candidates = run_state.candidates(
                    self._base_actions(k),
                    min_run_blocks=self.min_run_blocks,
                    min_fan_blocks=self.min_fan_blocks,
                    min_off_blocks=self.min_off_blocks,
                    changeover_blocks=self.changeover_blocks,
                    wet_coil_locked_blocks=self.wet_coil_locked_blocks,
                )
                action = self._choose(k, candidates, temperatures, humidities, charges)
                breach = guardrail_breach_direction(rooms, temperatures)
                if breach is not None and breach in candidates:
                    action = breach

            next_temperatures, next_humidities, next_charges = step_house_block(
                rooms,
                temperatures,
                humidities,
                charges,
                action,
                self._outdoor_at(k),
                k,
                dt_minutes=self.dt_minutes,
                extra_house_temp=self.extra_house_temp,
            )
            actions.append(action)
            for room in rooms:
                room_temperatures[room.area_id].append(next_temperatures[room.area_id])
            house_temperatures.append(compute_house_temperature(rooms, next_temperatures, self.extra_house_temp))
            temperatures = next_temperatures
            humidities = next_humidities
            charges = next_charges
            run_state.observe(action)

        first_action = actions[0] if actions else MODE_IDLE
        return CentralMPCPlan(
            actions=actions,
            room_temperatures=room_temperatures,
            house_temperatures=house_temperatures,
            driving_rooms=driving_rooms(
                rooms,
                first_action,
                initial_temperatures,
                initial_humidities,
                initial_house,
                humidity_comfort_enabled=self.humidity_comfort_enabled,
            ),
            decision_reason=decision_reason,
            mpc_active=True,
            dt_minutes=self.dt_minutes,
            preheat_info=preheat_info,
        )

    # -- block 0 decision ----------------------------------------------------

    def _decide_first_block(
        self,
        temperatures: Mapping[str, float],
        humidities: Mapping[str, float | None],
        charges: Mapping[str, float],
    ) -> tuple[str, str, dict[str, dict[str, Any]]]:
        """Current plant action plus its UI reason and pre-heat badges."""
        legal = set(self.initial_legal_actions) if self.initial_legal_actions is not None else self._base_actions(0)
        legal.add(MODE_IDLE)
        if self.initial_legal_actions is None and self.initial_off_blocks < self.min_off_blocks:
            # The plant just stopped a conditioning run: without the guard's own
            # legal set, replay the min-off rule so block 0 cannot short-cycle.
            legal.discard(MODE_HEATING)
            legal.discard(MODE_COOLING)
        if self.forced_action is not None:
            action = self.forced_action if self.forced_action in legal else MODE_IDLE
            return action, "hold_guard", {}
        if (
            self.initial_action in (MODE_HEATING, MODE_COOLING, MODE_CIRCULATING)
            and self.initial_run_blocks < self._min_run_for(self.initial_action)
            and self.initial_action in legal
        ):
            return self.initial_action, "hold_guard", {}

        action = self._choose(0, legal, temperatures, humidities, charges)
        breach = guardrail_breach_direction(self.rooms, temperatures)
        if breach is not None and breach in legal:
            return breach, "guardrail", {}

        direction, preheat_info = self._check_arrival_deadlines(temperatures, humidities, charges)
        if direction is not None and direction in legal:
            action = direction
            reason = "preheat" if direction == MODE_HEATING else "precool"
        else:
            # A deadline the equipment guard cannot honour is not a badge either.
            preheat_info = {}
            reason = self._reason_for(action)
            if action in (MODE_HEATING, MODE_COOLING) and self.initial_action == MODE_IDLE:
                if not start_discomfort_reached(
                    self.rooms,
                    action,
                    temperatures,
                    humidities,
                    humidity_comfort_enabled=self.humidity_comfort_enabled,
                ):
                    action = MODE_IDLE
                    reason = "idle"
        return action, reason, preheat_info

    @staticmethod
    def _reason_for(action: str) -> str:
        if action == MODE_CIRCULATING:
            return "circulate_mix"
        if action == MODE_HEATING or action == MODE_COOLING:
            return "mpc"
        return "idle"

    # -- horizon and candidate actions ---------------------------------------

    def _horizon(self) -> int:
        """Number of blocks to plan, bounded by the outdoor series and the maximum."""
        available = len(self.T_outdoor_series)
        if available <= 0:
            return 0
        return min(available, self.horizon_blocks, MAX_HORIZON_BLOCKS)

    def _outdoor_at(self, k: int) -> float:
        return _series_at(self.T_outdoor_series, k, DEFAULT_HOUSE_TEMP_C)

    def _base_actions(self, k: int) -> set[str]:
        """Plant actions physics and the climate mode leave available at block *k*."""
        actions = {MODE_IDLE, MODE_CIRCULATING}
        T_out = self._outdoor_at(k)
        if self.can_heat and (self.override_active or T_out <= self.outdoor_heating_max):
            actions.add(MODE_HEATING)
        if self.can_cool and (self.override_active or T_out >= self.outdoor_cooling_min):
            actions.add(MODE_COOLING)
        return actions

    def _min_run_for(self, action: str) -> int:
        if action == MODE_CIRCULATING:
            return max(1, self.min_fan_blocks)
        return max(1, self.min_run_blocks)

    def _empty_plan(self, rooms: Sequence[RoomHorizonInput]) -> CentralMPCPlan:
        temperatures = {room.area_id: room.T_init for room in rooms}
        house_temp = compute_house_temperature(rooms, temperatures, self.extra_house_temp)
        return CentralMPCPlan(
            actions=[],
            room_temperatures={room.area_id: [room.T_init] for room in rooms},
            house_temperatures=[house_temp],
            driving_rooms=[],
            decision_reason="idle",
            mpc_active=False,
            dt_minutes=self.dt_minutes,
            preheat_info={},
        )

    # -- cost ----------------------------------------------------------------

    def _choose(
        self,
        k: int,
        candidates: Iterable[str],
        temperatures: Mapping[str, float],
        humidities: Mapping[str, float | None],
        charges: Mapping[str, float],
    ) -> str:
        """Cheapest candidate action at block *k* (idle wins exact ties)."""
        allowed = set(candidates)
        best_action = MODE_IDLE if MODE_IDLE in allowed else next(iter(sorted(allowed)))
        best_cost = math.inf
        for action in (MODE_IDLE, MODE_CIRCULATING, MODE_HEATING, MODE_COOLING):
            if action not in allowed:
                continue
            cost = self._rollout_cost(action, k, temperatures, humidities, charges)
            if cost < best_cost - _COST_EPS:
                best_action, best_cost = action, cost
        return best_action

    def _rollout_cost(
        self,
        action: str,
        k: int,
        temperatures: Mapping[str, float],
        humidities: Mapping[str, float | None],
        charges: Mapping[str, float],
    ) -> float:
        """Cost of holding *action* over the decision lookahead from block *k*."""
        horizon = min(LOOKAHEAD_BLOCKS, self._horizon() - k)
        temps = dict(temperatures)
        rhs = dict(humidities)
        zs = dict(charges)
        total = 0.0
        for step in range(horizon):
            block = k + step
            total += self._stage_cost(block, action, temps, rhs)
            if step + 1 < horizon:
                temps, rhs, zs = step_house_block(
                    self.rooms,
                    temps,
                    rhs,
                    zs,
                    action,
                    self._outdoor_at(block),
                    block,
                    dt_minutes=self.dt_minutes,
                    extra_house_temp=self.extra_house_temp,
                )
        return total

    def _stage_cost(
        self,
        k: int,
        action: str,
        temperatures: Mapping[str, float],
        humidities: Mapping[str, float | None],
    ) -> float:
        """One block's priority-weighted comfort, guardrail, wrong-way and energy cost."""
        cw = max(0.0, min(100.0, self.comfort_weight)) / 100.0
        w_comfort = 0.2 + 1.8 * cw
        w_energy = 1.5 * (1.0 - 0.85 * cw)
        dehumidify_offset = 0.0
        if self.humidity_comfort_enabled:
            dehumidify_offset = dehumidify_cool_offset_c(
                self._weighted_rh(k, humidities),
                self.dehum_threshold,
                self.max_dehum_overcool_c,
            )
        total = 0.0
        for room in self.rooms:
            if room.area_id not in temperatures:
                continue
            T_i = temperatures[room.area_id]
            # Guardrails protect every room, including empty ones at priority 0.
            breach = max(0.0, room.guardrail_heat - T_i, T_i - room.guardrail_cool)
            total += GUARDRAIL_WEIGHT * breach * breach * (4.0 if breach > 1.0 else 1.0)
            w_i = _series_at(room.priority_series, k, 1.0)
            if w_i <= 0.0:
                continue
            T_eff = effective_temperature_c(
                T_i,
                humidities.get(room.area_id),
                enabled=self.humidity_comfort_enabled,
            )
            heat_target = _optional_at(room.comfort_heat_series, k, None)
            cool_target_raw = _optional_at(room.comfort_cool_series, k, None)
            cool_target = bounded_dehumidified_cool_target(
                heat_target,
                cool_target_raw,
                room.guardrail_heat,
                dehumidify_offset,
            )
            err_heat = max(0.0, heat_target - T_eff) if heat_target is not None else 0.0
            err_cool = max(0.0, T_eff - cool_target) if cool_target is not None else 0.0
            total += w_comfort * w_i * (err_heat * err_heat + err_cool * err_cool)
            occupied = _series_at(room.occupied_prob_series, k, 1.0) >= OCCUPIED_PROB_THRESHOLD
            if occupied and action == MODE_HEATING and cool_target_raw is not None and T_eff > cool_target_raw:
                total += WRONG_WAY_PENALTY * w_i * (T_eff - cool_target_raw) ** 2
            elif occupied and action == MODE_COOLING and heat_target is not None and T_eff < heat_target:
                total += WRONG_WAY_PENALTY * w_i * (heat_target - T_eff) ** 2
        if action == MODE_CIRCULATING:
            total += w_energy * FAN_ENERGY_FRACTION
        elif action == MODE_HEATING or action == MODE_COOLING:
            total += w_energy
        return total

    def _weighted_rh(self, k: int, humidities: Mapping[str, float | None]) -> float | None:
        """Room Priority weighted relative humidity across the measured rooms."""
        weighted = 0.0
        total_weight = 0.0
        for room in self.rooms:
            rh = humidities.get(room.area_id)
            if rh is None:
                continue
            w_i = _series_at(room.priority_series, k, 1.0)
            if w_i <= 0.0:
                continue
            weighted += w_i * rh
            total_weight += w_i
        if total_weight <= 0.0:
            return None
        return weighted / total_weight

    # -- anticipatory pre-heat / pre-cool ------------------------------------

    def _check_arrival_deadlines(
        self,
        temperatures: Mapping[str, float],
        humidities: Mapping[str, float | None],
        charges: Mapping[str, float],
    ) -> tuple[str | None, dict[str, dict[str, Any]]]:
        """Direction (and UI badges) when an arriving room needs the plant now.

        A room whose comfort step arrives later (Occupancy Trend crossing, or a
        schedule target rise/fall) must be warm or cool *on time*.  Waiting one
        more block and then running at full power would miss it by more than
        ``ARRIVAL_DEADLINE_MARGIN_C``: start now.  The run is refused when it would
        push an already occupied room more than ``OPPOSITE_BOUND_TOLERANCE_C``
        past its opposite comfort bound.
        """
        horizon = self._horizon()
        limit = preheat_max_blocks(self.comfort_weight, self.dt_minutes)
        window = horizon - 1 if limit is None else min(limit, horizon - 1)
        if window < 1:
            return None, {}
        house_temp = compute_house_temperature(self.rooms, temperatures, self.extra_house_temp)
        heat = self._direction_deadlines(
            MODE_HEATING,
            window,
            temperatures,
            humidities,
            charges,
            house_temp,
        )
        cool = self._direction_deadlines(
            MODE_COOLING,
            window,
            temperatures,
            humidities,
            charges,
            house_temp,
        )
        if not heat and not cool:
            return None, {}
        heat_best = max(heat, key=lambda item: (item[0], -item[2])) if heat else None
        cool_best = max(cool, key=lambda item: (item[0], -item[2])) if cool else None
        if cool_best is None or (heat_best is not None and heat_best[0] >= cool_best[0]):
            direction, candidates = MODE_HEATING, heat
        else:
            direction, candidates = MODE_COOLING, cool
        info: dict[str, dict[str, Any]] = {}
        for _priority, area_id, j, target in candidates:
            info[area_id] = {
                "mode": direction,
                "target": target,
                "at": j * self.dt_minutes,
            }
        return direction, info

    def _direction_deadlines(
        self,
        direction: str,
        window: int,
        temperatures: Mapping[str, float],
        humidities: Mapping[str, float | None],
        charges: Mapping[str, float],
        house_temp: float,
    ) -> list[tuple[float, str, int, float]]:
        """Rooms whose arrival *direction* would miss if the plant waited a block."""
        heat = direction == MODE_HEATING
        prefix = self._opposite_bound_prefix(
            direction,
            window,
            temperatures,
            humidities,
            charges,
        )
        candidates: list[tuple[float, str, int, float]] = []
        for room in self.rooms:
            if not room.preheat_enabled:
                continue
            delayed = self._delayed_room_temperatures(
                room,
                direction,
                window,
                house_temp,
                temperatures,
                charges,
            )
            for j in range(1, window + 1):
                target = self._arrival_target(room, j, heat)
                if target is None:
                    continue
                if heat and delayed[j] >= target - ARRIVAL_DEADLINE_MARGIN_C:
                    continue
                if not heat and delayed[j] <= target + ARRIVAL_DEADLINE_MARGIN_C:
                    continue
                if prefix[j] > OPPOSITE_BOUND_TOLERANCE_C:
                    continue
                candidates.append((_series_at(room.priority_series, j, 1.0), room.area_id, j, target))
        return candidates

    @staticmethod
    def _arrival_target(room: RoomHorizonInput, j: int, heat: bool) -> float | None:
        """Comfort target a room's arrival at block *j* asks for, None when none."""
        occupied = _series_at(room.occupied_prob_series, j, 1.0)
        occupied_before = _series_at(room.occupied_prob_series, j - 1, 1.0)
        arrival = occupied >= ARRIVAL_OCCUPANCY_THRESHOLD and occupied_before < ARRIVAL_OCCUPANCY_THRESHOLD
        if heat:
            target = _optional_at(room.comfort_heat_series, j, None)
            previous = _optional_at(room.comfort_heat_series, j - 1, None)
            rise = target is not None and previous is not None and target > previous + _COST_EPS
            return target if target is not None and (arrival or rise) else None
        target = _optional_at(room.comfort_cool_series, j, None)
        previous = _optional_at(room.comfort_cool_series, j - 1, None)
        fall = target is not None and previous is not None and target < previous - _COST_EPS
        return target if target is not None and (arrival or fall) else None

    def _delayed_room_temperatures(
        self,
        room: RoomHorizonInput,
        direction: str,
        window: int,
        house_temp: float,
        temperatures: Mapping[str, float],
        charges: Mapping[str, float],
    ) -> list[float]:
        """Temperatures at each block if the plant idles now and runs from block 1.

        ``result[j]`` is the temperature at the start of block *j* (after one idle
        block and ``j - 1`` full-power blocks).  Starting the run one block later
        is what the deadline compares against running now.
        """
        T = temperatures[room.area_id]
        charge = charges.get(room.area_id, room.rebound_init)
        result = [T]
        # Predict against this room's own reference when it has usable
        # neighbours; otherwise keep the existing whole-house temperature.
        ref_temp = house_temp
        if room.neighbours:
            included = {
                other.area_id for other in self.rooms if other.include_in_house_mean and other.area_id in temperatures
            }
            neighbours = {
                neighbour: weight
                for neighbour, weight in room.neighbours.items()
                if neighbour in included and neighbour != room.area_id
            }
            if neighbours:
                ref_temp = reference_temperature(neighbours, temperatures, house_temp)
        T_idle = room.model.predict(
            T,
            self._outdoor_at(0),
            0.0,
            self.dt_minutes,
            T_house=ref_temp,
            u_fan=0.0,
            door_mix_factor=room.door_mix_factor,
            door_supply_factor=room.door_supply_factor,
            q_solar=_series_at(room.solar_series, 0, 0.0),
            q_occupancy=_series_at(room.occupancy_gain_series, 0, 0.0),
            q_rebound=mean_rebound(charge, 0.0, self.dt_minutes),
            mode=MODE_IDLE,
        )
        result.append(T_idle)
        T = T_idle
        charge = advance_rebound(charge, 0.0, self.dt_minutes)
        drive = drive_for_mode(direction)
        Q = room.model.Q_heat if direction == MODE_HEATING else -room.model.Q_cool
        for j in range(1, window + 1):
            T = room.model.predict(
                T,
                self._outdoor_at(j),
                Q,
                self.dt_minutes,
                T_house=ref_temp,
                u_fan=0.0,
                door_mix_factor=room.door_mix_factor,
                door_supply_factor=room.door_supply_factor,
                q_solar=_series_at(room.solar_series, j, 0.0),
                q_occupancy=_series_at(room.occupancy_gain_series, j, 0.0),
                q_rebound=mean_rebound(charge, drive, self.dt_minutes),
                mode=direction,
            )
            charge = advance_rebound(charge, drive, self.dt_minutes)
            result.append(T)
        return result

    def _opposite_bound_prefix(
        self,
        direction: str,
        steps: int,
        temperatures: Mapping[str, float],
        humidities: Mapping[str, float | None],
        charges: Mapping[str, float],
    ) -> list[float]:
        """Worst overshoot past an occupied room's opposite bound, cumulative.

        ``result[j]`` is the worst overshoot over blocks ``0 .. j-1`` of running
        *direction* now, so a deadline at block *j* can refuse a run that would
        over-condition a room for the sake of another.
        """
        occupied = [
            room for room in self.rooms if _series_at(room.occupied_prob_series, 0, 1.0) >= OCCUPIED_PROB_THRESHOLD
        ]
        if not occupied:
            return [0.0] * (steps + 1)
        prefix = [0.0]
        temps = dict(temperatures)
        rhs = dict(humidities)
        zs = dict(charges)
        worst = 0.0
        for k in range(steps):
            temps, rhs, zs = step_house_block(
                self.rooms,
                temps,
                rhs,
                zs,
                direction,
                self._outdoor_at(k),
                k,
                dt_minutes=self.dt_minutes,
                extra_house_temp=self.extra_house_temp,
            )
            for room in occupied:
                T_eff = effective_temperature_c(
                    temps[room.area_id],
                    rhs.get(room.area_id),
                    enabled=self.humidity_comfort_enabled,
                )
                if direction == MODE_HEATING:
                    bound = _optional_at(room.comfort_cool_series, k, None)
                    if bound is not None:
                        worst = max(worst, T_eff - bound)
                else:
                    bound = _optional_at(room.comfort_heat_series, k, None)
                    if bound is not None:
                        worst = max(worst, bound - T_eff)
            prefix.append(worst)
        return prefix


# ---------------------------------------------------------------------------
# Cold-start bang-bang fallback
# ---------------------------------------------------------------------------


@dataclass
class CentralBangBang:
    """Priority-weighted bang-bang fallback for the cold-start phase.

    Every EKF needs hours of excitation before the joint rollout can be trusted.
    Until then this keeps the home safe with rules that cannot diverge: it starts
    a run ``0.2 degC`` past the weighted target, continues it until the target is
    reached, and prefers Circulation when two occupied rooms disagree in opposite
    directions (over-conditioning one room to satisfy the other is a losing trade).
    """

    #: Plant action running right now (seeded by the controller each block).
    active_action: str = MODE_IDLE
    humidity_comfort_enabled: bool = True
    #: Occupied-room spread that makes mixing air worthwhile.
    circulation_spread_c: float = 2.0
    #: Weighted error below this is a mild disagreement Circulation may arbitrate.
    mild_error_c: float = 1.0

    @staticmethod
    def _mixable(a: RoomHorizonInput, b: RoomHorizonInput) -> bool:
        """Two rooms can trade air when either has no declared group, or both share one."""
        return a.mixing_group is None or b.mixing_group is None or a.mixing_group == b.mixing_group

    def decide(
        self,
        *,
        rooms: Sequence[RoomHorizonInput],
        legal_actions: Iterable[str] | None = None,
        forced_action: str | None = None,
        k: int = 0,
    ) -> str:
        """Choose the plant action for the current block."""
        legal = set(legal_actions) if legal_actions is not None else set(_ALL_PLANT_MODES)
        if forced_action is not None:
            return forced_action if (legal_actions is None or forced_action in legal) else MODE_IDLE

        heat_err = 0.0
        cool_err = 0.0
        heat_weight = 0.0
        cool_weight = 0.0
        occupied_rooms: list[tuple[RoomHorizonInput, float, float | None, float | None]] = []
        for room in rooms:
            T_eff = effective_temperature_c(
                room.T_init,
                room.rh_init,
                enabled=self.humidity_comfort_enabled,
            )
            occupied = _series_at(room.occupied_prob_series, k, 1.0) >= OCCUPIED_PROB_THRESHOLD
            heat_target = _optional_at(room.comfort_heat_series, k, None)
            cool_target = _optional_at(room.comfort_cool_series, k, None)
            if occupied:
                occupied_rooms.append((room, T_eff, heat_target, cool_target))
            w_i = _series_at(room.priority_series, k, 1.0)
            if w_i <= 0.0:
                continue
            if heat_target is not None:
                heat_err += w_i * max(0.0, heat_target - T_eff)
                heat_weight += w_i
            if cool_target is not None:
                cool_err += w_i * max(0.0, T_eff - cool_target)
                cool_weight += w_i
        heat_error = heat_err / heat_weight if heat_weight > 0.0 else 0.0
        cool_error = cool_err / cool_weight if cool_weight > 0.0 else 0.0
        # Circulation only arbitrates rooms that can actually exchange air:
        # undeclared rooms form one bucket, each declared group its own bucket.
        buckets: dict[int | None, list[float]] = {}
        for room, T_eff, _heat, _cool in occupied_rooms:
            buckets.setdefault(room.mixing_group, []).append(T_eff)
        group_spreads = [max(temps) - min(temps) for temps in buckets.values() if len(temps) >= 2]
        spread = max(group_spreads, default=0.0)

        breach = self._guardrail_direction(rooms)
        if breach is not None and breach in legal:
            return breach

        cooling_rooms = [
            room
            for room, T_eff, _heat, cool_target in occupied_rooms
            if cool_target is not None and T_eff - cool_target > BANGBANG_COOL_HYSTERESIS
        ]
        near_floor_rooms = [
            room
            for room, T_eff, heat_target, _cool in occupied_rooms
            if heat_target is not None and T_eff <= heat_target + WARM_FLOOR_MARGIN_C
        ]
        # Circulation only arbitrates a genuine disagreement between two rooms
        # that can trade air; a cross-group pair cannot equalise by mixing.
        opposite_errors = any(
            cooling.area_id != warm.area_id and self._mixable(cooling, warm)
            for cooling in cooling_rooms
            for warm in near_floor_rooms
        )
        mild_spread = spread >= self.circulation_spread_c and (
            heat_error < self.mild_error_c and cool_error < self.mild_error_c
        )
        if MODE_CIRCULATING in legal and (opposite_errors or mild_spread):
            return MODE_CIRCULATING

        if self.active_action == MODE_HEATING and heat_error > 0.0 and MODE_HEATING in legal:
            return MODE_HEATING
        if self.active_action == MODE_COOLING and cool_error > 0.0 and MODE_COOLING in legal:
            return MODE_COOLING
        if cool_error > BANGBANG_COOL_HYSTERESIS and cool_error > heat_error and MODE_COOLING in legal:
            return MODE_COOLING
        if heat_error > BANGBANG_HEAT_HYSTERESIS and MODE_HEATING in legal:
            return MODE_HEATING
        if cool_error > BANGBANG_COOL_HYSTERESIS and MODE_COOLING in legal:
            return MODE_COOLING
        return MODE_IDLE

    @staticmethod
    def _guardrail_direction(rooms: Sequence[RoomHorizonInput]) -> str | None:
        """Direction repairing a Guardrail Band breach (freeze/overheat protection)."""
        return guardrail_breach_direction(
            rooms,
            {room.area_id: room.T_init for room in rooms},
        )


_ALL_PLANT_MODES = (MODE_IDLE, MODE_HEATING, MODE_COOLING, MODE_CIRCULATING)


# ---------------------------------------------------------------------------
# Controller facade
# ---------------------------------------------------------------------------


class CentralMPCController:
    """Chooses the Central Plant action for the whole home.

    The joint MPC runs whenever at least one priority-weighted room's model passed
    the data gates; rooms that are not ready yet keep their Caller-supplied priors
    from :class:`~.thermal_model.RoomModelManager` and still participate in the
    coupled rollout, so a single cold room cannot blind the plan.  With no ready
    room, or with ``control_mode == "bangbang"``, the cold-start controller takes
    over and still rolls the room trajectories forward for the forecast charts.
    """

    def __init__(self, model_manager: RoomModelManager | None = None) -> None:
        self._model_manager = model_manager
        self._bangbang = CentralBangBang()

    def evaluate(
        self,
        *,
        rooms: Sequence[RoomHorizonInput],
        T_outdoor_series: Sequence[float],
        settings: Mapping[str, Any] | None = None,
        estimators: Mapping[str, ThermalEKF] | None = None,
        legal_actions: Iterable[str] | None = None,
        forced_action: str | None = None,
        initial_action: str = MODE_IDLE,
        initial_run_blocks: int = 0,
        initial_off_blocks: int = 0,
        can_heat: bool = True,
        can_cool: bool = True,
        prediction_enabled: bool = True,
        dt_minutes: float = PLAN_DT_MINUTES,
        horizon_blocks: int | None = None,
        wet_coil_locked_blocks: int = 0,
    ) -> CentralMPCPlan:
        """Build the plan for this control cycle.

        Args:
            rooms: The Thermal Zones with their initial state and horizon series.
            T_outdoor_series: Outdoor temperature per block.
            settings: Global settings (``control_mode``, ``comfort_weight``, ...).
            estimators: Per-room EKFs for the MPC readiness gates; the
                ``model_manager`` supplies them when omitted.
            legal_actions: Plant actions the equipment guard allows right now.
            forced_action: Action the equipment guard forces right now.
            initial_action: Plant action running at the start of this block.
            initial_run_blocks: Blocks the running action has already run.
            initial_off_blocks: Blocks the plant has been off after a conditioning run.
            can_heat: Whether the climate mode allows heating at all.
            can_cool: Whether the climate mode allows cooling at all.
            prediction_enabled: Roll the room trajectories forward (forecast charts).
            dt_minutes: Plan block duration.
            horizon_blocks: Plan horizon; None uses ``MIN_HORIZON_BLOCKS``.
            wet_coil_locked_blocks: Circulation lockout left after a cooling run.

        Returns:
            The plan for this cycle; ``plan.current_action()`` is the action to apply.
        """
        settings = dict(settings or {})
        plant = get_central_plant_settings(settings)
        room_list = list(rooms)
        outdoor = list(T_outdoor_series)
        if not room_list or not outdoor:
            return CentralMPCController._empty_plan(room_list, dt_minutes)

        prepared = [self._prepare_room(room, estimators, can_heat=can_heat, can_cool=can_cool) for room in room_list]
        legal = self._legal_actions(legal_actions, can_heat=can_heat, can_cool=can_cool)
        control_mode = str(settings.get("control_mode", CONTROL_MODE_MPC))
        weighted_ready = any(room.mpc_ready and _series_at(room.priority_series, 0, 1.0) > 0.0 for room in prepared)

        if control_mode != CONTROL_MODE_BANGBANG and weighted_ready:
            optimizer = CentralMPCOptimizer(
                rooms=prepared,
                T_outdoor_series=outdoor,
                dt_minutes=dt_minutes,
                comfort_weight=_as_float(settings.get("comfort_weight"), float(DEFAULT_COMFORT_WEIGHT)),
                horizon_blocks=MIN_HORIZON_BLOCKS if horizon_blocks is None else horizon_blocks,
                can_heat=can_heat,
                can_cool=can_cool,
                humidity_comfort_enabled=bool(plant.get("humidity_comfort_enabled", True)),
                dehum_threshold=_as_float(
                    plant.get("dehumidify_rh_threshold"),
                    float(DEFAULT_DEHUMIDIFY_RH_THRESHOLD),
                ),
                max_dehum_overcool_c=_as_float(
                    plant.get("max_dehumidify_overcool_c"),
                    float(DEFAULT_MAX_DEHUMIDIFY_OVERCOOL_C),
                ),
                outdoor_cooling_min=_as_float(
                    settings.get("outdoor_cooling_min"),
                    float(DEFAULT_OUTDOOR_COOLING_MIN),
                ),
                outdoor_heating_max=_as_float(
                    settings.get("outdoor_heating_max"),
                    float(DEFAULT_OUTDOOR_HEATING_MAX),
                ),
                initial_legal_actions=legal,
                forced_action=forced_action,
                initial_action=initial_action,
                initial_run_blocks=initial_run_blocks,
                initial_off_blocks=initial_off_blocks,
                min_run_blocks=_blocks_from_minutes(
                    plant.get("min_run_minutes"),
                    dt_minutes,
                    DEFAULT_MIN_RUN_MINUTES,
                ),
                min_off_blocks=_blocks_from_minutes(
                    plant.get("min_off_minutes"),
                    dt_minutes,
                    DEFAULT_MIN_OFF_MINUTES,
                ),
                min_fan_blocks=_blocks_from_minutes(
                    plant.get("min_fan_minutes"),
                    dt_minutes,
                    DEFAULT_MIN_FAN_MINUTES,
                ),
                changeover_blocks=_blocks_from_minutes(
                    plant.get("changeover_minutes"),
                    dt_minutes,
                    DEFAULT_CHANGEOVER_MINUTES,
                ),
                wet_coil_locked_blocks=wet_coil_locked_blocks,
            )
            return optimizer.optimize()
        return self._bangbang_plan(
            prepared,
            outdoor,
            legal=legal,
            forced_action=forced_action,
            initial_action=initial_action,
            prediction_enabled=prediction_enabled,
            dt_minutes=dt_minutes,
            horizon_blocks=horizon_blocks,
        )

    # -- helpers -------------------------------------------------------------

    def _prepare_room(
        self,
        room: RoomHorizonInput,
        estimators: Mapping[str, ThermalEKF] | None,
        *,
        can_heat: bool,
        can_cool: bool,
    ) -> RoomHorizonInput:
        """Resolve readiness and hand a not-ready room its forced-air priors."""
        estimator = self._estimator_for(room.area_id, estimators)
        ready = room.mpc_ready
        if estimator is not None:
            ready = ready and is_room_mpc_ready(estimator, can_heat, can_cool)
        model = room.model
        if not ready and self._model_manager is not None:
            model = self._model_manager.get_model(room.area_id)
        if ready is room.mpc_ready and model is room.model:
            return room
        return replace(room, model=model, mpc_ready=ready)

    def _estimator_for(
        self,
        area_id: str,
        estimators: Mapping[str, ThermalEKF] | None,
    ) -> ThermalEKF | None:
        """The room's EKF for the readiness gate (a fresh one reads as not ready)."""
        if estimators is not None and area_id in estimators:
            return estimators[area_id]
        manager = self._model_manager
        if manager is not None:
            return manager.get_estimator(area_id)
        return None

    @staticmethod
    def _legal_actions(
        legal_actions: Iterable[str] | None,
        *,
        can_heat: bool,
        can_cool: bool,
    ) -> set[str]:
        if legal_actions is not None:
            legal = set(legal_actions)
            legal.add(MODE_IDLE)
            return legal
        legal = {MODE_IDLE, MODE_CIRCULATING}
        if can_heat:
            legal.add(MODE_HEATING)
        if can_cool:
            legal.add(MODE_COOLING)
        return legal

    def _bangbang_plan(
        self,
        rooms: Sequence[RoomHorizonInput],
        outdoor: Sequence[float],
        *,
        legal: set[str],
        forced_action: str | None,
        initial_action: str,
        prediction_enabled: bool,
        dt_minutes: float,
        horizon_blocks: int | None,
    ) -> CentralMPCPlan:
        """Cold-start decision plus the trajectory roll-forward for the charts."""
        self._bangbang.active_action = initial_action
        action = self._bangbang.decide(
            rooms=rooms,
            legal_actions=legal,
            forced_action=forced_action,
        )
        self._bangbang.active_action = action
        blocks = min(len(outdoor), MAX_HORIZON_BLOCKS)
        requested = MIN_HORIZON_BLOCKS if horizon_blocks is None else max(1, horizon_blocks)
        blocks = min(blocks, requested)
        steps = blocks if prediction_enabled else 1

        temperatures = {room.area_id: room.T_init for room in rooms}
        humidities = {room.area_id: room.rh_init for room in rooms}
        charges = {room.area_id: room.rebound_init for room in rooms}
        initial_temperatures = dict(temperatures)
        initial_humidities = dict(humidities)
        house_temp = compute_house_temperature(rooms, temperatures)
        room_temperatures: dict[str, list[float]] = {room.area_id: [room.T_init] for room in rooms}
        house_temperatures = [house_temp]
        actions: list[str] = []
        for k in range(max(1, steps)):
            temperatures, humidities, charges = step_house_block(
                rooms,
                temperatures,
                humidities,
                charges,
                action,
                _series_at(outdoor, k, DEFAULT_HOUSE_TEMP_C),
                k,
                dt_minutes=dt_minutes,
            )
            actions.append(action)
            for room in rooms:
                room_temperatures[room.area_id].append(temperatures[room.area_id])
            house_temperatures.append(compute_house_temperature(rooms, temperatures))
        return CentralMPCPlan(
            actions=actions,
            room_temperatures=room_temperatures,
            house_temperatures=house_temperatures,
            driving_rooms=driving_rooms(
                rooms,
                action,
                initial_temperatures,
                initial_humidities,
                house_temp,
                humidity_comfort_enabled=self._bangbang.humidity_comfort_enabled,
            ),
            decision_reason="bangbang",
            mpc_active=False,
            dt_minutes=dt_minutes,
        )

    @staticmethod
    def _empty_plan(rooms: Sequence[RoomHorizonInput], dt_minutes: float) -> CentralMPCPlan:
        temperatures = {room.area_id: room.T_init for room in rooms}
        return CentralMPCPlan(
            actions=[],
            room_temperatures={room.area_id: [room.T_init] for room in rooms},
            house_temperatures=[compute_house_temperature(rooms, temperatures)],
            driving_rooms=[],
            decision_reason="idle",
            mpc_active=False,
            dt_minutes=dt_minutes,
            preheat_info={},
        )
