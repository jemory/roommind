"""Live occupancy, learned Occupancy Trends and Room Priority trajectories for RoomMind.

Three signals are combined here:

* a live dwell filter, so a 30-second walk-through does not count as occupancy
  while a sustained presence saturates within a few minutes,
* a time-of-week trend model that learns the recurring occupied probability in
  15-minute bins, separately for weekdays and weekends, and
* Room Priority weights that fold both signals together with schedule
  assumptions, away/vacation states, overrides and window-freeze protection.

Times are wall-clock (``time.time()``) because dwell and trends survive a Home
Assistant restart.  Bin learning is local-time based: Home Assistant's
configured timezone decides the day type and the 15-minute bin index, so a
house whose HA timezone differs from UTC still learns the right hours.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from ..utils import presence_utils

# Occupancy modes (per room, ``room["occupancy_mode"]``).
OCCUPANCY_MODE_REACTIVE_TREND = "reactive_and_trend"
OCCUPANCY_MODE_SCHEDULE_ASSUMED = "schedule_assumed_when_on"
OCCUPANCY_MODE_FOLLOW_SCHEDULE = "follow_schedule"
OCCUPANCY_MODE_GUARDRAIL_ONLY = "guardrail_only"
OCCUPANCY_MODES = frozenset(
    {
        OCCUPANCY_MODE_REACTIVE_TREND,
        OCCUPANCY_MODE_SCHEDULE_ASSUMED,
        OCCUPANCY_MODE_FOLLOW_SCHEDULE,
        OCCUPANCY_MODE_GUARDRAIL_ONLY,
    }
)

# Dwell filter.
DEFAULT_DWELL_MINUTES = 3.0
DEFAULT_ECO_DELAY_MINUTES = 15.0
VACANCY_GRACE_MINUTES = 2.0

# Room Priority weights.
OVERRIDE_PRIORITY_WEIGHT = 3.0
AWAY_PRIORITY_WEIGHT = 0.25
GUARDRAIL_PRIORITY_WEIGHT = 0.08
FREEZE_PROTECTION_TEMP_C = 7.0
FREEZE_PRIORITY_WEIGHT = 5.0

# Trend model.
LIVE_PERSIST_TAU_MINUTES = 45.0
TREND_EMA_ALPHA = 0.15
TREND_MATURITY_DAYS = 7.0
BINS_PER_DAY = 96
DEFAULT_SLOT_PROBABILITY = 0.25


@dataclass
class RoomOccupancyState:
    """Live occupancy and priority judgement for one room."""

    raw_occupied: bool = False
    has_sensors: bool = False
    dwell_score: float = 0.0
    trend_probability: float = 0.0
    effective_occupancy: float = 0.0
    priority_weight: float = 0.0
    is_vacant: bool = False


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp *value* into ``[low, high]``."""
    return max(low, min(high, value))


def _as_float(value: object, default: float) -> float:
    """Read a numeric setting, falling back to *default* for malformed data."""
    if isinstance(value, int | float):
        return float(value)
    return default


def _safe_is_entity_gone(hass: HomeAssistant, entity_id: str) -> bool:
    """Return ``True`` only when the entity registry positively says it is gone.

    A missing registry (or a registry lookup that cannot answer, as in a bare
    test harness) must stay fail-safe: not gone means the caller treats the
    missing state as "not started yet" and counts the sensor as occupied.
    """
    try:
        return bool(presence_utils._is_entity_gone(hass, entity_id))
    except Exception:  # noqa: BLE001 - a broken registry must never starve comfort
        return False


def read_raw_occupancy(hass: HomeAssistant, room: dict) -> tuple[bool, bool]:
    """Aggregate a room's occupancy sensors into ``(raw_occupied, has_sensors)``.

    RoomMind's fail-safe rule applies: a live sensor that reports
    ``unavailable``/``unknown`` (or has no state object yet right after a
    restart) counts as ``"on"``.  A broken sensor must never starve a room of
    comfort.  Deleted or disabled entities are skipped instead, so removing one
    from the configuration cannot pin a room to occupied forever.
    """
    sensors = room.get("occupancy_sensors") or []
    live = 0
    for entity_id in sensors:
        state = hass.states.get(entity_id)
        if state is None:
            if _safe_is_entity_gone(hass, entity_id):
                continue
            # Not started yet (or missing while HA is still booting): fail-safe.
            return True, True
        live += 1
        if state.state != "off":
            return True, True
    if live == 0:
        return False, False
    return False, True


class DwellTracker:
    """Per-room dwell score that filters motion spikes and decays on vacancy."""

    def __init__(self) -> None:
        self._scores: dict[str, float] = {}
        self._last_step_ts: dict[str, float] = {}
        self._vacant_since_ts: dict[str, float] = {}

    def step(
        self,
        area_id: str,
        raw_occupied: bool,
        now_ts: float,
        dwell_minutes: float = DEFAULT_DWELL_MINUTES,
        eco_delay_minutes: float = DEFAULT_ECO_DELAY_MINUTES,
    ) -> float:
        """Advance one room and return its dwell score in ``[0, 1]``.

        Occupancy adds the elapsed time divided by *dwell_minutes*, so a
        60-second visit with a 3-minute dwell only reaches 0.33.  Vacancy holds
        the score for ``VACANCY_GRACE_MINUTES`` (a sensor retrigger gap) and
        then decays linearly, reaching zero after *eco_delay_minutes*.
        """
        score = self._scores.get(area_id, 0.0)
        last = self._last_step_ts.get(area_id)

        if raw_occupied:
            self._vacant_since_ts.pop(area_id, None)
            elapsed_min = 0.0 if last is None else max(0.0, now_ts - last) / 60.0
            if dwell_minutes <= 0:
                score = 1.0
            else:
                score = min(1.0, score + elapsed_min / dwell_minutes)
        else:
            vacant_since = self._vacant_since_ts.setdefault(area_id, now_ts)
            grace_end = vacant_since + VACANCY_GRACE_MINUTES * 60.0
            if now_ts > grace_end:
                decay_from = max(last if last is not None else grace_end, grace_end)
                decay_min = max(0.0, now_ts - decay_from) / 60.0
                score = max(0.0, score - decay_min / max(eco_delay_minutes, 1.0))

        score = _clamp(score)
        # Six 30-second steps add up to 0.9999999999999999 in binary floating
        # point; a full dwell requirement must read as exactly 1.0.
        if score > 1.0 - 1e-12:
            score = 1.0
        self._scores[area_id] = score
        self._last_step_ts[area_id] = now_ts
        return score

    def score(self, area_id: str) -> float:
        """Return the last dwell score for *area_id* (0.0 when unknown)."""
        return self._scores.get(area_id, 0.0)

    def remove_room(self, area_id: str) -> None:
        """Drop all dwell state for a removed room."""
        self._scores.pop(area_id, None)
        self._last_step_ts.pop(area_id, None)
        self._vacant_since_ts.pop(area_id, None)

    def reset_all(self) -> None:
        """Drop all rooms."""
        self._scores.clear()
        self._last_step_ts.clear()
        self._vacant_since_ts.clear()

    def to_dict(self) -> dict:
        """Persistable state: score plus the timers that make it continuous."""
        areas = set(self._scores) | set(self._last_step_ts) | set(self._vacant_since_ts)
        return {
            area_id: {
                "score": self._scores.get(area_id, 0.0),
                "last_step_ts": self._last_step_ts.get(area_id),
                "vacant_since_ts": self._vacant_since_ts.get(area_id),
            }
            for area_id in areas
        }

    def load_data(self, data: dict) -> None:
        """Restore dwell state, skipping malformed entries."""
        self.reset_all()
        if not isinstance(data, dict):
            return
        for area_id, entry in data.items():
            if not isinstance(entry, dict):
                continue
            key = str(area_id)
            score = entry.get("score")
            if isinstance(score, int | float):
                self._scores[key] = _clamp(float(score))
            last = entry.get("last_step_ts")
            if isinstance(last, int | float):
                self._last_step_ts[key] = float(last)
            vacant = entry.get("vacant_since_ts")
            if isinstance(vacant, int | float):
                self._vacant_since_ts[key] = float(vacant)


class OccupancyTrendModel:
    """One room's learned time-of-week occupancy probability.

    ``weekday_bins`` and ``weekend_bins`` each hold 96 probabilities (15-minute
    bins) initialised to ``DEFAULT_SLOT_PROBABILITY``.  A completed slot updates
    its bin with an EMA and advances ``days_observed`` by ``1 / 96``; the
    weighted probability is scaled by that maturity so an untrained model
    cannot claim a confident trend.
    """

    def __init__(self) -> None:
        self.weekday_bins: list[float] = [DEFAULT_SLOT_PROBABILITY] * BINS_PER_DAY
        self.weekend_bins: list[float] = [DEFAULT_SLOT_PROBABILITY] * BINS_PER_DAY
        self.days_observed: float = 0.0
        # Sample accumulators for the slot currently being observed.
        self._cur_day_key: int | None = None
        self._cur_slot: int | None = None
        self._slot_sum: float = 0.0
        self._slot_count: int = 0
        self._last_ts: float | None = None

    @staticmethod
    def _local_bin(ts: float) -> tuple[int, int]:
        """Return ``(day_key, slot_idx)`` in Home Assistant's local timezone."""
        local = dt_util.as_local(dt_util.utc_from_timestamp(ts))
        return local.weekday(), local.hour * 4 + local.minute // 15

    def _bins(self, day_key: int) -> list[float]:
        """Weekend bins for Saturday/Sunday, weekday bins otherwise."""
        return self.weekend_bins if day_key >= 5 else self.weekday_bins

    def observe(self, ts: float, occupied: bool, vacation_active: bool = False) -> None:
        """Fold one occupancy sample in.

        While *vacation_active* the model is frozen: no sample is accumulated
        and any partial slot is discarded, so the holiday does not dilute the
        learned routine.
        """
        if vacation_active:
            self._clear_sample()
            return

        day_key, slot_idx = self._local_bin(ts)
        if self._last_ts is not None and ts < self._last_ts:
            # Clock went backwards: do not fold a future sample into an old bin.
            self._clear_sample()
        if self._cur_day_key is None or self._cur_slot is None:
            self._start_sample(day_key, slot_idx)
        elif (day_key, slot_idx) != (self._cur_day_key, self._cur_slot):
            self._fold_completed_slot()
            self._start_sample(day_key, slot_idx)

        self._slot_sum += 1.0 if occupied else 0.0
        self._slot_count += 1
        self._last_ts = ts

    def step(self, ts: float, occupied: bool, vacation_active: bool = False) -> None:
        """Alias for :meth:`observe`."""
        self.observe(ts, occupied, vacation_active)

    def update(self, ts: float, occupied: bool, vacation_active: bool = False) -> None:
        """Alias for :meth:`observe`."""
        self.observe(ts, occupied, vacation_active)

    def _start_sample(self, day_key: int, slot_idx: int) -> None:
        self._cur_day_key = day_key
        self._cur_slot = slot_idx
        self._slot_sum = 0.0
        self._slot_count = 0

    def _clear_sample(self) -> None:
        self._cur_day_key = None
        self._cur_slot = None
        self._slot_sum = 0.0
        self._slot_count = 0
        self._last_ts = None

    def _fold_completed_slot(self) -> None:
        """Write the finished slot's mean occupancy into its bin via EMA."""
        if self._cur_day_key is None or self._cur_slot is None or self._slot_count <= 0:
            return
        slot_mean = _clamp(self._slot_sum / self._slot_count)
        bins = self._bins(self._cur_day_key)
        bins[self._cur_slot] = _clamp((1.0 - TREND_EMA_ALPHA) * bins[self._cur_slot] + TREND_EMA_ALPHA * slot_mean)
        self.days_observed += 1.0 / BINS_PER_DAY

    def raw_smoothed_probability(self, ts: float) -> float:
        """Smoothed, unmatured probability for *ts* (used by the 24 h UI profile).

        The circular 3-bin kernel wraps within the day type: bin 0 uses bin 95
        as its left neighbour and bin 95 uses bin 0 as its right neighbour, so
        midnight needs no special case.
        """
        day_key, slot_idx = self._local_bin(ts)
        bins = self._bins(day_key)
        left = bins[slot_idx - 1]
        right = bins[(slot_idx + 1) % BINS_PER_DAY]
        return _clamp(0.25 * left + 0.50 * bins[slot_idx] + 0.25 * right)

    def weighted_trend_probability(self, ts: float) -> float:
        """Raw smoothed probability scaled by how many days were observed."""
        maturity = min(1.0, max(0.0, self.days_observed) / TREND_MATURITY_DAYS)
        return _clamp(maturity * self.raw_smoothed_probability(ts))

    def profile_24h(self, weekend: bool = False) -> list[float]:
        """Return the 96-bin raw profile for one day type, rounded for UI use."""
        bins = self.weekend_bins if weekend else self.weekday_bins
        profile: list[float] = []
        for slot_idx in range(BINS_PER_DAY):
            smoothed = 0.25 * bins[slot_idx - 1] + 0.50 * bins[slot_idx]
            smoothed += 0.25 * bins[(slot_idx + 1) % BINS_PER_DAY]
            profile.append(round(_clamp(smoothed), 4))
        return profile

    def to_dict(self) -> dict:
        """Persistable state, including the partially observed slot."""
        data: dict = {
            "weekday_bins": list(self.weekday_bins),
            "weekend_bins": list(self.weekend_bins),
            "days_observed": self.days_observed,
        }
        if self._cur_day_key is not None and self._cur_slot is not None and self._slot_count > 0:
            data["sample"] = {
                "day_key": self._cur_day_key,
                "slot_idx": self._cur_slot,
                "sum": self._slot_sum,
                "count": self._slot_count,
            }
        return data

    def load_data(self, data: dict) -> None:
        """Restore persisted bins, maturity and a partially observed slot."""
        self.weekday_bins = [DEFAULT_SLOT_PROBABILITY] * BINS_PER_DAY
        self.weekend_bins = [DEFAULT_SLOT_PROBABILITY] * BINS_PER_DAY
        self.days_observed = 0.0
        self._cur_day_key = None
        self._cur_slot = None
        self._slot_sum = 0.0
        self._slot_count = 0
        self._last_ts = None
        if not isinstance(data, dict):
            return
        weekday = data.get("weekday_bins")
        if isinstance(weekday, list):
            self.weekday_bins = _sanitize_bins(weekday)
        weekend = data.get("weekend_bins")
        if isinstance(weekend, list):
            self.weekend_bins = _sanitize_bins(weekend)
        days = data.get("days_observed")
        if isinstance(days, int | float):
            self.days_observed = max(0.0, float(days))
        sample = data.get("sample")
        if isinstance(sample, dict):
            self._load_sample(sample)

    def _load_sample(self, sample: dict) -> None:
        day_key = sample.get("day_key")
        slot_idx = sample.get("slot_idx")
        total = sample.get("sum")
        count = sample.get("count")
        if not isinstance(day_key, int) or not 0 <= day_key <= 6:
            return
        if not isinstance(slot_idx, int) or not 0 <= slot_idx < BINS_PER_DAY:
            return
        if not isinstance(total, int | float) or not isinstance(count, int | float) or count <= 0:
            return
        self._cur_day_key = day_key
        self._cur_slot = slot_idx
        self._slot_sum = float(total)
        self._slot_count = int(count)


def _sanitize_bins(values: list) -> list[float]:
    """Clamp a persisted bin list to 96 valid probabilities."""
    result = [DEFAULT_SLOT_PROBABILITY] * BINS_PER_DAY
    for index in range(min(BINS_PER_DAY, len(values))):
        value = values[index]
        if isinstance(value, int | float):
            result[index] = _clamp(float(value))
    return result


class OccupancyManager:
    """Combine live dwell filtering with per-room Occupancy Trends."""

    def __init__(self) -> None:
        self._dwell = DwellTracker()
        self._trends: dict[str, OccupancyTrendModel] = {}
        self._states: dict[str, RoomOccupancyState] = {}

    def resolve_occupancy_mode(self, room: dict) -> str:
        """Return the room's explicit mode, or derive one from its legacy flags.

        A room without occupancy sensors follows its schedule.  With sensors it
        uses live occupancy when occupancy eco is enabled, and otherwise assumes
        the schedule's "on" blocks mean someone is present (a sleeping person's
        PIR stays off while the schedule still holds the room).
        """
        mode = room.get("occupancy_mode")
        if mode in OCCUPANCY_MODES:
            return str(mode)
        if not room.get("occupancy_sensors"):
            return OCCUPANCY_MODE_FOLLOW_SCHEDULE
        if room.get("occupancy_eco_enabled", False):
            return OCCUPANCY_MODE_REACTIVE_TREND
        return OCCUPANCY_MODE_SCHEDULE_ASSUMED

    def _trend_for(self, area_id: str) -> OccupancyTrendModel:
        model = self._trends.get(area_id)
        if model is None:
            model = OccupancyTrendModel()
            self._trends[area_id] = model
        return model

    def weighted_trend_probability(self, area_id: str, ts: float) -> float:
        """Matured trend probability for *area_id* at *ts* (0.0 when untrained)."""
        model = self._trends.get(area_id)
        if model is None:
            return 0.0
        return model.weighted_trend_probability(ts)

    def raw_trend_probability(self, area_id: str, ts: float) -> float:
        """Unmatured smoothed trend probability, for UI visualisation."""
        model = self._trends.get(area_id)
        if model is None:
            return 0.0
        return model.raw_smoothed_probability(ts)

    def _mode_values(
        self,
        mode: str,
        k: int,
        dt_minutes: float,
        dwell_now: float,
        trend_now: float,
        schedule_on: bool,
        has_sensors: bool,
    ) -> tuple[float, float]:
        """Return ``(priority_weight, occupied_probability)`` for one block."""
        if mode == OCCUPANCY_MODE_GUARDRAIL_ONLY:
            return GUARDRAIL_PRIORITY_WEIGHT, 0.0

        if mode == OCCUPANCY_MODE_FOLLOW_SCHEDULE or (
            not has_sensors and mode in (OCCUPANCY_MODE_REACTIVE_TREND, OCCUPANCY_MODE_FOLLOW_SCHEDULE)
        ):
            occupied = 1.0 if schedule_on else 0.0
            weight = GUARDRAIL_PRIORITY_WEIGHT + (1.0 - GUARDRAIL_PRIORITY_WEIGHT) * occupied
            return weight, occupied

        live = dwell_now * math.exp(-(k * dt_minutes) / LIVE_PERSIST_TAU_MINUTES)
        floor = 1.0 if mode == OCCUPANCY_MODE_SCHEDULE_ASSUMED and schedule_on else 0.0
        occupied = max(live, trend_now, floor)
        weight = round(GUARDRAIL_PRIORITY_WEIGHT + (1.0 - GUARDRAIL_PRIORITY_WEIGHT) * occupied, 4)
        return weight, occupied

    def step_room(
        self,
        hass: HomeAssistant,
        area_id: str,
        room: dict,
        now_ts: float,
        *,
        schedule_active_now: bool = True,
        presence_away: bool = False,
        vacation_active: bool = False,
        override_active: bool = False,
        window_open: bool = False,
        current_temp_c: float | None = None,
    ) -> RoomOccupancyState:
        """Read live occupancy and return the room's current occupancy state."""
        raw_occupied, has_sensors = read_raw_occupancy(hass, room)
        mode = self.resolve_occupancy_mode(room)

        dwell_minutes = _as_float(room.get("occupancy_dwell_minutes"), DEFAULT_DWELL_MINUTES)
        eco_delay_minutes = _as_float(room.get("occupancy_eco_delay"), DEFAULT_ECO_DELAY_MINUTES)
        dwell_score = self._dwell.step(area_id, raw_occupied, now_ts, dwell_minutes, eco_delay_minutes)

        trend_probability = 0.0
        if has_sensors:
            model = self._trend_for(area_id)
            model.observe(now_ts, raw_occupied, vacation_active)
            trend_probability = model.weighted_trend_probability(now_ts)

        if room.get("is_outdoor", False) or not room.get("climate_control_enabled", True):
            effective, weight = 0.0, 0.0
        elif window_open:
            effective = 0.0
            freezing = current_temp_c is not None and current_temp_c < FREEZE_PROTECTION_TEMP_C
            weight = FREEZE_PRIORITY_WEIGHT if freezing else 0.0
        elif override_active:
            effective, weight = 1.0, OVERRIDE_PRIORITY_WEIGHT
        elif vacation_active or (presence_away and not room.get("ignore_presence", False)):
            effective, weight = 0.0, AWAY_PRIORITY_WEIGHT
        else:
            weight, effective = self._mode_values(
                mode,
                0,
                0.0,
                dwell_score,
                trend_probability,
                bool(schedule_active_now),
                has_sensors,
            )

        is_vacant = has_sensors and dwell_score <= 0.01 and mode == OCCUPANCY_MODE_REACTIVE_TREND
        state = RoomOccupancyState(
            raw_occupied=raw_occupied,
            has_sensors=has_sensors,
            dwell_score=dwell_score,
            trend_probability=trend_probability,
            effective_occupancy=effective,
            priority_weight=weight,
            is_vacant=is_vacant,
        )
        self._states[area_id] = state
        return state

    def build_priority_series(
        self,
        area_id: str,
        room: dict,
        settings: dict,
        now_ts: float,
        n_blocks: int,
        dt_minutes: float = 5.0,
        *,
        schedule_active_at: Callable[[float], bool] | None = None,
        override_active_at: Callable[[float], bool] | None = None,
        presence_away: bool = False,
        vacation_active: bool = False,
        window_open: bool = False,
        current_temp_c: float | None = None,
    ) -> tuple[list[float], list[float]]:
        """Build Room Priority weights and occupancy probabilities for the horizon.

        Returns ``(priority_weights, occupied_probs)`` with one entry per block.
        Excluded rooms (outdoor, climate control off) and open-window rooms get
        zero priority, except the sub-7 °C freeze emergency which keeps the
        room's guardrail alive at ``FREEZE_PRIORITY_WEIGHT``.
        """
        count = max(0, int(n_blocks))
        zeros = [0.0] * count
        if room.get("is_outdoor", False) or not room.get("climate_control_enabled", True):
            return zeros, list(zeros)
        if window_open:
            if current_temp_c is not None and current_temp_c < FREEZE_PROTECTION_TEMP_C:
                return [FREEZE_PRIORITY_WEIGHT] * count, list(zeros)
            return zeros, list(zeros)

        mode = self.resolve_occupancy_mode(room)
        has_sensors = bool(room.get("occupancy_sensors") or [])
        dwell_now = self._dwell.score(area_id)
        model = self._trends.get(area_id)
        step_seconds = float(dt_minutes) * 60.0

        weights: list[float] = []
        probabilities: list[float] = []
        for k in range(count):
            t_k = now_ts + k * step_seconds
            schedule_on = True if schedule_active_at is None else bool(schedule_active_at(t_k))
            if override_active_at is not None and override_active_at(t_k):
                weight, occupied = OVERRIDE_PRIORITY_WEIGHT, 1.0
            elif vacation_active or (presence_away and not room.get("ignore_presence", False)):
                weight, occupied = AWAY_PRIORITY_WEIGHT, 0.0
            else:
                trend_now = 0.0 if model is None else model.weighted_trend_probability(t_k)
                weight, occupied = self._mode_values(
                    mode, k, dt_minutes, dwell_now, trend_now, schedule_on, has_sensors
                )
            weights.append(weight)
            probabilities.append(occupied)
        return weights, probabilities

    def get_24h_profile(self, area_id: str, *, weekend: bool = False) -> list[float]:
        """Return 96 rounded raw trend probabilities for the UI profile chart."""
        model = self._trends.get(area_id)
        if model is None:
            return [DEFAULT_SLOT_PROBABILITY] * BINS_PER_DAY
        return model.profile_24h(weekend)

    def last_state(self, area_id: str) -> RoomOccupancyState | None:
        """Return the state produced by the last :meth:`step_room` for *area_id*."""
        return self._states.get(area_id)

    def dwell_score(self, area_id: str) -> float:
        """Return the live dwell score for *area_id*."""
        return self._dwell.score(area_id)

    def trend_model(self, area_id: str) -> OccupancyTrendModel | None:
        """Return the trend model for *area_id* (None when never observed)."""
        return self._trends.get(area_id)

    def to_dict(self) -> dict:
        """Persistable manager state, keyed by room so a room can be cleared alone."""
        dwell_data = self._dwell.to_dict()
        result: dict = {}
        for area_id in set(dwell_data) | set(self._trends):
            result[area_id] = {
                "dwell": dwell_data.get(area_id, {}),
                "trend": self._trends[area_id].to_dict() if area_id in self._trends else {},
            }
        return result

    def load_data(self, data: dict) -> None:
        """Restore persisted dwell and trend state.

        Two shapes are accepted: the current room-keyed mapping and the older
        ``{"dwell": ..., "trends": ...}`` grouping, so data written by either
        layout keeps loading.
        """
        self._states.clear()
        self._trends.clear()
        if not isinstance(data, dict):
            self._dwell.load_data({})
            return
        if "dwell" in data or "trends" in data:
            self._load_grouped_data(data)
            return
        dwell_payload: dict = {}
        for area_id, payload in data.items():
            if not isinstance(payload, dict):
                continue
            key = str(area_id)
            dwell_entry = payload.get("dwell")
            if isinstance(dwell_entry, dict):
                dwell_payload[key] = dwell_entry
            elif any(field in payload for field in ("score", "last_step_ts", "vacant_since_ts")):
                dwell_payload[key] = payload
            trend_entry = payload.get("trend")
            if isinstance(trend_entry, dict) and trend_entry:
                model = OccupancyTrendModel()
                model.load_data(trend_entry)
                self._trends[key] = model
            elif any(field in payload for field in ("weekday_bins", "weekend_bins", "days_observed")):
                model = OccupancyTrendModel()
                model.load_data(payload)
                self._trends[key] = model
        self._dwell.load_data(dwell_payload)

    def _load_grouped_data(self, data: dict) -> None:
        """Restore the legacy grouped ``{"dwell": ..., "trends": ...}`` layout."""
        dwell = data.get("dwell")
        self._dwell.load_data(dwell if isinstance(dwell, dict) else {})
        trends = data.get("trends")
        if isinstance(trends, dict):
            for area_id, payload in trends.items():
                model = OccupancyTrendModel()
                if isinstance(payload, dict):
                    model.load_data(payload)
                self._trends[str(area_id)] = model

    def remove_room(self, area_id: str) -> None:
        """Drop every dwell and trend trace of a removed room."""
        self._dwell.remove_room(area_id)
        self._trends.pop(area_id, None)
        self._states.pop(area_id, None)

    def reset_room(self, area_id: str) -> None:
        """Reset a room to its initial learning state, keeping other rooms intact."""
        self._dwell.remove_room(area_id)
        self._trends[area_id] = OccupancyTrendModel()
        self._states.pop(area_id, None)

    def reset_all(self) -> None:
        """Clear all learned occupancy state."""
        self._dwell.reset_all()
        self._trends.clear()
        self._states.clear()
