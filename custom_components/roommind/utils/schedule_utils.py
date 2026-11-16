"""Schedule utilities for resolving future target temperatures."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

from homeassistant.util import dt as dt_util

from ..const import (
    CLIMATE_MODE_COOL_ONLY,
    CLIMATE_MODE_HEAT_ONLY,
    DEFAULT_COMFORT_COOL,
    DEFAULT_COMFORT_HEAT,
    DEFAULT_ECO_COOL,
    DEFAULT_ECO_HEAT,
    MAX_TARGET_TEMP,
    MIN_TARGET_TEMP,
    OVERRIDE_BOOST,
    OVERRIDE_ECO,
    SCHEDULE_UNREADABLE_HOLD_HOURS,
    TargetTemps,
    is_override_active,
)
from .price_utils import PriceSignal, apply_price_shift

_LOGGER = logging.getLogger(__name__)


class TargetSource(StrEnum):
    """Which rung of the target ladder produced a room's resolved targets."""

    OVERRIDE = "override"
    PLANT_OVERRIDE = "plant_override"
    VACATION = "vacation"
    AWAY = "away"
    ROOM_SCHEDULE = "room_schedule"
    HOUSE = "house"
    ROOM_BASE = "room_base"


@dataclass(frozen=True)
class HouseContext:
    """Resolved house-layer inputs passed into a room's target resolver.

    ``target_at`` resolves the house's own targets at a timestamp (override /
    active schedule block / comfort pair), so the room resolver stays a pure
    function of its inputs. ``eco_heat_c`` / ``eco_cool_c`` are the band used to
    cap an empty following room; ``price_shift`` is the house's own opt-in and
    replaces the room's when the room follows an active house.
    ``hold_active_at`` reports, per timestamp, whether the house's manual hold
    is in force so a later MPC block cannot inherit a stale hold flag.
    """

    active: bool
    heat_c: float | None
    cool_c: float | None
    eco_heat_c: float | None
    eco_cool_c: float | None
    target_at: Callable[[float], TargetTemps]
    price_shift: float = 0.0
    hold_active_at: Callable[[float], bool] | None = None


def find_active_block(schedule_blocks: dict, ts: float) -> dict[str, Any] | None:
    """Find the schedule block active at the given timestamp.

    Returns the block's ``data`` dict, or None if no block matches.
    Caller must check ``schedule_blocks is None`` before calling.
    """
    dt = dt_util.as_local(datetime.fromtimestamp(ts, tz=UTC))
    day_name = dt.strftime("%A").lower()
    current_time = dt.time()
    day_blocks = schedule_blocks.get(day_name, [])
    for block in day_blocks:
        from_raw = block.get("from", "00:00:00")
        to_raw = block.get("to", "00:00:00")
        from_time = from_raw if hasattr(from_raw, "hour") else datetime.strptime(str(from_raw), "%H:%M:%S").time()
        to_time = to_raw if hasattr(to_raw, "hour") else datetime.strptime(str(to_raw), "%H:%M:%S").time()
        if from_time <= current_time < to_time:
            return dict(block.get("data", {}))
    return None


def sanitize_block_temp(
    raw: Any,
    converter: Callable[[float], float] | None = None,
) -> float | None:
    """Convert a schedule block temperature to °C, or None if unusable.

    Schedule blocks are free-form YAML, so a typo (110 instead of 11) reaches us
    unchecked and would be heated against for hours (#395). Values outside
    ``MIN_TARGET_TEMP``..``MAX_TARGET_TEMP`` are rejected so the caller can fall
    back to comfort/eco.

    The range check runs *after* ``converter``, so the bounds always apply to
    Celsius regardless of the HA display unit (on a °F system the accepted
    window is 41–95 °F).
    """
    try:
        val = float(raw)
    except (ValueError, TypeError):
        return None
    if converter is not None:
        val = converter(val)
    # Negated comparison also rejects NaN, which would pass `MIN <= v <= MAX`.
    if not MIN_TARGET_TEMP <= val <= MAX_TARGET_TEMP:
        return None
    return val


# Weekday order used when reporting rejected blocks, so the panel lists them
# in calendar order rather than dict insertion order.
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

# Block data keys that carry a target temperature.
_BLOCK_TEMP_FIELDS = ("temperature", "heat_temperature", "cool_temperature")


def find_rejected_block_temps(
    schedule_blocks: dict | None,
    converter: Callable[[float], float] | None = None,
) -> list[dict[str, Any]]:
    """Scan *every* block of a schedule for temperatures the resolver would drop.

    ``sanitize_block_temp`` only sees the block that is running right now, so a
    typo in a block scheduled for 03:00 stays invisible until it fires (#395).
    This walks the whole week so the panel can surface it at any time of day.

    Returns one entry per rejected value, each with ``day``, ``from``, ``field``
    and the offending ``value`` as a string (it may not be numeric at all).
    Only values that are present but unusable are reported — an absent field is
    not a problem, it simply falls back to comfort.
    """
    if not schedule_blocks:
        return []

    rejected: list[dict[str, Any]] = []
    for day in _WEEKDAYS:
        for block in schedule_blocks.get(day, []) or []:
            data = block.get("data") or {}
            for field in _BLOCK_TEMP_FIELDS:
                raw = data.get(field)
                if raw is None or sanitize_block_temp(raw, converter) is not None:
                    continue
                rejected.append(
                    {
                        "day": day,
                        "from": str(block.get("from", "")),
                        "field": field,
                        "value": str(raw),
                    }
                )
    return rejected


def override_preset_band(room: dict, override_type: str) -> tuple[float, float]:
    """Return the (heat, cool) band a boost/eco override stands for.

    Both presets are bands in auto rooms, never a single point: collapsing eco
    to ``eco_heat`` on both sides makes an AC cool toward the heating target (#284).
    """
    if override_type == OVERRIDE_ECO:
        return (
            room.get("eco_heat", room.get("eco_temp", DEFAULT_ECO_HEAT)),
            room.get("eco_cool", DEFAULT_ECO_COOL),
        )
    return (
        room.get("comfort_heat", room.get("comfort_temp", DEFAULT_COMFORT_HEAT)),
        room.get("comfort_cool", DEFAULT_COMFORT_COOL),
    )


def mask_override_band(heat: float | None, cool: float | None, climate_mode: str) -> tuple[float | None, float | None]:
    """Drop the side a heat_only/cool_only room cannot act on."""
    if climate_mode == CLIMATE_MODE_HEAT_ONLY:
        return heat, None
    if climate_mode == CLIMATE_MODE_COOL_ONLY:
        return None, cool
    return heat, cool


def band_in_force(room: dict, live: dict) -> tuple[float | None, float | None]:
    """The (heat, cool) band a room is held at: its active override, else the targets resolved last cycle.

    Mold prevention is taken off again (not part of the user's band). Falls back to
    the comfort band when no live targets exist yet or the room is forced off.
    """
    if is_override_active(room):
        return room.get("override_heat"), room.get("override_cool")
    heat = live.get("heat_target")
    cool = live.get("cool_target")
    if heat is None and cool is None:
        return override_preset_band(room, OVERRIDE_BOOST)
    return without_temporary_shifts(heat, cool, live)


def without_temporary_shifts(heat: float | None, cool: float | None, live: dict) -> tuple[float | None, float | None]:
    """Live targets without mold prevention and the electricity price shift: neither is part of
    the user's band, a slider or wall display mirroring them would echo them back as an override."""
    if heat is not None:
        heat -= live.get("price_shift_heat") or 0
        if live.get("mold_prevention_active"):
            heat -= live.get("mold_prevention_delta") or 0
    if cool is not None:
        cool -= live.get("price_shift_cool") or 0
    return heat, cool


def set_band_sides(
    band: tuple[float | None, float | None], heat: float | None, cool: float | None, climate_mode: str
) -> tuple[float | None, float | None] | None:
    """*band* with the given sides (°C, None = unchanged) set within the target bounds.

    A side the room's climate mode does not use is ignored; None if no usable side is
    given. A set side pushes the other along instead of crossing it, so heat never
    ends above cool.
    """
    cur_heat, cur_cool = mask_override_band(*band, climate_mode)
    heat, cool = mask_override_band(heat, cool, climate_mode)
    if heat is None and cool is None:
        return None

    def bounded(value: float) -> float:
        return round(min(MAX_TARGET_TEMP, max(MIN_TARGET_TEMP, value)), 1)

    if heat is not None:
        cur_heat = bounded(heat)
        if cool is None and cur_cool is not None and cur_cool < cur_heat:
            cur_cool = cur_heat
    if cool is not None:
        cur_cool = bounded(cool)
        if cur_heat is not None and cur_cool < cur_heat:
            cur_heat = cur_cool
    return cur_heat, cur_cool


def cap_cool_only_heat(heat: float | None, cool: float | None, climate_mode: str) -> tuple[float | None, float | None]:
    """Keep the unused heat side of a cool_only room from rising above the cool target.

    A cool_only room configured with the legacy ``comfort_temp`` (API, old configs)
    carried that value on the heat side next to the default ``comfort_cool``, e.g.
    heat 24.5 / cool 24.0. The optimizer lifts the cool target to the heat side and
    the cover target takes the higher one, while guard, hysteresis and the panel use
    the cool target. The cool target is what the panel shows for cool_only rooms.
    """
    if climate_mode == CLIMATE_MODE_COOL_ONLY and heat is not None and cool is not None and heat > cool:
        return cool, cool
    return heat, cool


def is_plant_override_at(ts: float, settings: dict, *, presence_away: bool = False) -> bool:
    """Return True when a whole-home plant override is in force at ``ts``.

    A manual setpoint change on the Central Thermostat — the shared wall thermostat of the
    Central HVAC plant — becomes a plant override that holds until its expiry; it outranks
    the schedule, vacation and presence branches of the resolver. While the home is
    presence-away and ``presence_clears_override`` is set, the override is held, not applied.
    """
    p_heat = settings.get("plant_override_heat")
    p_cool = settings.get("plant_override_cool")
    if p_heat is None and p_cool is None:
        return False
    p_until = settings.get("plant_override_until")
    if p_until is not None and ts >= p_until:
        return False
    if presence_away and bool(settings.get("presence_clears_override", False)):
        return False
    return True


def resolve_target_at_time(
    ts: float,
    schedule_blocks: dict | None,
    override_until: float | None,
    override_temp: float | None,
    vacation_until: float | None,
    vacation_temp: float | None,
    comfort_temp: float,
    eco_temp: float,
    presence_away: bool = False,
    block_temp_converter: Callable[[float], float] | None = None,
    presence_away_action: str = "eco",
    schedule_off_action: str = "eco",
) -> float | None:
    """Resolve what the target temp would be at a specific timestamp.

    Returns None when the action is "off" (devices should be turned off).
    """
    # 1. Override
    if override_temp is not None and (override_until is None or ts < override_until):
        return float(override_temp)
    # 2. Vacation
    if vacation_until is not None and ts < vacation_until and vacation_temp is not None:
        return float(vacation_temp)
    # 2.5 Presence
    if presence_away:
        if presence_away_action == "off":
            return None
        return eco_temp
    # 3. Schedule blocks
    if schedule_blocks is None:
        return comfort_temp
    data = find_active_block(schedule_blocks, ts)
    if data is not None:
        block_temp = data.get("temperature")
        if block_temp is not None:
            val = sanitize_block_temp(block_temp, block_temp_converter)
            if val is not None:
                return val
        return comfort_temp
    # Not in any block → eco or off
    if schedule_off_action == "off":
        return None
    return eco_temp


def resolve_targets_with_source(
    ts: float,
    schedule_blocks: dict | None,
    override_until: float | None,
    override_heat: float | None,
    override_cool: float | None,
    vacation_until: float | None,
    vacation_temp: float | None,
    comfort_heat: float,
    comfort_cool: float,
    eco_heat: float,
    eco_cool: float,
    presence_away: bool = False,
    block_temp_converter: Callable[[float], float] | None = None,
    presence_away_action: str = "eco",
    schedule_off_action: str = "eco",
    presence_clears_override: bool = False,
    room_vacant: bool = False,
    plant_override_until: float | None = None,
    plant_override_heat: float | None = None,
    plant_override_cool: float | None = None,
    *,
    house: HouseContext | None = None,
    follow_house: bool = False,
) -> tuple[TargetTemps, TargetSource]:
    """Resolve dual heat/cool targets and the ladder rung that produced them.

    Rungs, highest first: room override, plant (Central Thermostat) override,
    vacation, presence-away, a room schedule block carrying a sanitized
    temperature, an active followed house, then the room's own base schedule /
    comfort / eco / off. ``None`` sides mean "force off". The occupancy eco cap
    applies only to the last three rungs; when a following room is empty it
    uses the house's eco pair, never the room's.

    A deliberately held house target is reported as ``TargetSource.HOUSE``;
    ``schedule_sets_target(house_override_active=...)`` is the companion guard
    that keeps such a hold out of the electricity-price shift.
    """
    # 1. Override — split heat/cool dead-band (skipped when presence-away suppresses it)
    if (override_heat is not None or override_cool is not None) and (override_until is None or ts < override_until):
        if not (presence_away and presence_clears_override):
            return TargetTemps(heat=override_heat, cool=override_cool), TargetSource.OVERRIDE
    # 2. Whole-home plant override — the Central Thermostat wall setpoint wins over the schedule
    if (
        (plant_override_heat is not None or plant_override_cool is not None)
        and (plant_override_until is None or ts < plant_override_until)
        and not (presence_away and presence_clears_override)
    ):
        sched = _schedule_targets_at_time(
            ts,
            schedule_blocks,
            comfort_heat,
            comfort_cool,
            eco_heat,
            eco_cool,
            block_temp_converter,
            schedule_off_action,
        )
        h = plant_override_heat if plant_override_heat is not None else sched.heat
        c = plant_override_cool if plant_override_cool is not None else sched.cool
        if h is not None and c is not None and c < h:
            if plant_override_cool is None:
                c = h
            else:
                h = c
        return TargetTemps(heat=h, cool=c), TargetSource.PLANT_OVERRIDE
    # 3. Vacation — heat setback, cooling stays at eco_cool
    if vacation_until is not None and ts < vacation_until and vacation_temp is not None:
        t = float(vacation_temp)
        return TargetTemps(heat=t, cool=max(t, eco_cool)), TargetSource.VACATION
    # 4. Presence
    if presence_away:
        if presence_away_action == "off":
            return TargetTemps(heat=None, cool=None), TargetSource.AWAY
        return TargetTemps(heat=eco_heat, cool=eco_cool), TargetSource.AWAY

    active_house = house if house is not None and follow_house and house.active else None
    if active_house is not None:
        cap_heat = active_house.eco_heat_c if active_house.eco_heat_c is not None else eco_heat
        cap_cool = active_house.eco_cool_c if active_house.eco_cool_c is not None else eco_cool
    else:
        cap_heat, cap_cool = eco_heat, eco_cool

    # 5. A room schedule block with a usable temperature outranks the house
    if schedule_block_sets_target(ts, schedule_blocks, block_temp_converter):
        targets = _schedule_targets_at_time(
            ts,
            schedule_blocks,
            comfort_heat,
            comfort_cool,
            eco_heat,
            eco_cool,
            block_temp_converter,
            schedule_off_action,
        )
        return (cap_to_eco(targets, cap_heat, cap_cool) if room_vacant else targets), TargetSource.ROOM_SCHEDULE
    # 6. A following room in a schedule gap inherits the house's band
    if active_house is not None:
        targets = active_house.target_at(ts)
        return (cap_to_eco(targets, cap_heat, cap_cool) if room_vacant else targets), TargetSource.HOUSE
    # 7. Room base: comfort, a block without temperatures, eco or off
    targets = _schedule_targets_at_time(
        ts, schedule_blocks, comfort_heat, comfort_cool, eco_heat, eco_cool, block_temp_converter, schedule_off_action
    )
    return (cap_to_eco(targets, eco_heat, eco_cool) if room_vacant else targets), TargetSource.ROOM_BASE


def resolve_targets_at_time(
    ts: float,
    schedule_blocks: dict | None,
    override_until: float | None,
    override_heat: float | None,
    override_cool: float | None,
    vacation_until: float | None,
    vacation_temp: float | None,
    comfort_heat: float,
    comfort_cool: float,
    eco_heat: float,
    eco_cool: float,
    presence_away: bool = False,
    block_temp_converter: Callable[[float], float] | None = None,
    presence_away_action: str = "eco",
    schedule_off_action: str = "eco",
    presence_clears_override: bool = False,
    room_vacant: bool = False,
    plant_override_until: float | None = None,
    plant_override_heat: float | None = None,
    plant_override_cool: float | None = None,
    *,
    house: HouseContext | None = None,
    follow_house: bool = False,
) -> TargetTemps:
    """Compatibility wrapper around ``resolve_targets_with_source`` for targets only.

    With no house (or with ``follow_house`` False) the result is byte-for-byte
    the v3.0.0 resolution. See ``resolve_targets_with_source`` for the ladder.
    """
    targets, _ = resolve_targets_with_source(
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
        presence_away,
        block_temp_converter,
        presence_away_action,
        schedule_off_action,
        presence_clears_override,
        room_vacant,
        plant_override_until,
        plant_override_heat,
        plant_override_cool,
        house=house,
        follow_house=follow_house,
    )
    return targets


def cap_to_eco(targets: TargetTemps, eco_heat: float, eco_cool: float) -> TargetTemps:
    """Lower the demand of *targets* to at most eco; never raise it, "off" sides stay off."""
    return TargetTemps(
        heat=min(targets.heat, eco_heat) if targets.heat is not None else None,
        cool=max(targets.cool, eco_cool) if targets.cool is not None else None,
    )


def _schedule_targets_at_time(
    ts: float,
    schedule_blocks: dict | None,
    comfort_heat: float,
    comfort_cool: float,
    eco_heat: float,
    eco_cool: float,
    block_temp_converter: Callable[[float], float] | None,
    schedule_off_action: str,
) -> TargetTemps:
    if schedule_blocks is None:
        return TargetTemps(heat=comfort_heat, cool=comfort_cool)
    data = find_active_block(schedule_blocks, ts)
    if data is not None:
        heat_temp_raw = data.get("heat_temperature")
        cool_temp_raw = data.get("cool_temperature")
        if heat_temp_raw is not None or cool_temp_raw is not None:
            h = comfort_heat
            c = comfort_cool
            if heat_temp_raw is not None:
                val = sanitize_block_temp(heat_temp_raw, block_temp_converter)
                if val is not None:
                    h = val
            if cool_temp_raw is not None:
                val = sanitize_block_temp(cool_temp_raw, block_temp_converter)
                if val is not None:
                    c = val
            return TargetTemps(heat=h, cool=c)
        block_temp = data.get("temperature")
        if block_temp is not None:
            # Distinct name from the vacation branch's `t`, which is a plain float.
            single = sanitize_block_temp(block_temp, block_temp_converter)
            if single is not None:
                return TargetTemps(heat=single, cool=single)
        return TargetTemps(heat=comfort_heat, cool=comfort_cool)
    # Not in any block → eco or off
    if schedule_off_action == "off":
        return TargetTemps(heat=None, cool=None)
    return TargetTemps(heat=eco_heat, cool=eco_cool)


def schedule_block_sets_target(
    ts: float,
    schedule_blocks: dict | None,
    block_temp_converter: Callable[[float], float] | None = None,
) -> bool:
    """Whether the block active at ``ts`` carries at least one usable temperature.

    A block without temperatures (or with only implausible ones) leaves the
    target to comfort/eco, so it must not win a rung of the ladder on its own.
    Shared with the coordinator for arbitration and the room's reason text.
    """
    if not schedule_blocks:
        return False
    data = find_active_block(schedule_blocks, ts)
    if data is None:
        return False
    for raw in (data.get("heat_temperature"), data.get("cool_temperature"), data.get("temperature")):
        if raw is not None and sanitize_block_temp(raw, block_temp_converter) is not None:
            return True
    return False


def _house_block_targets_at_time(
    ts: float,
    house_blocks: dict | None,
    house: dict,
    block_temp_converter: Callable[[float], float] | None,
) -> TargetTemps:
    """House targets from the active block or, lacking one, the comfort pair.

    Never eco, never "off": the house layer holds its comfort band even in a
    schedule gap. A side the block leaves out (or that fails sanitization)
    comes from the house comfort pair; a single ``temperature`` sets both.
    """
    heat_c = house.get("heat_c")
    cool_c = house.get("cool_c")
    if house_blocks:
        data = find_active_block(house_blocks, ts)
        if data is not None:
            h_raw = data.get("heat_temperature")
            c_raw = data.get("cool_temperature")
            if h_raw is not None or c_raw is not None:
                h = heat_c
                c = cool_c
                if h_raw is not None:
                    val = sanitize_block_temp(h_raw, block_temp_converter)
                    if val is not None:
                        h = val
                if c_raw is not None:
                    val = sanitize_block_temp(c_raw, block_temp_converter)
                    if val is not None:
                        c = val
                return TargetTemps(heat=h, cool=c)
            single = data.get("temperature")
            if single is not None:
                val = sanitize_block_temp(single, block_temp_converter)
                if val is not None:
                    return TargetTemps(heat=val, cool=val)
    return TargetTemps(heat=heat_c, cool=cool_c)


def house_override_active_at(ts: float, house: dict) -> bool:
    """Whether the house's manual hold is in force at ``ts``.

    Same expiry semantics as a plant override, without presence suppression:
    an open-ended hold (``override_until`` None) is always active once a side
    is set, otherwise the hold ends at its timestamp. Exported so the
    coordinator can build ``HouseContext.hold_active_at`` from the house dict.
    """
    if house.get("override_heat") is None and house.get("override_cool") is None:
        return False
    until = house.get("override_until")
    return until is None or ts < until


def _house_override_in_force(ts: float, house: dict) -> bool:
    """Whether the house's manual hold is active at ``ts``."""
    return house_override_active_at(ts, house)


def resolve_house_targets_at_time(
    ts: float,
    house_blocks: dict | None,
    house: dict,
    block_temp_converter: Callable[[float], float] | None = None,
) -> TargetTemps:
    """Resolve the house's own heat/cool targets at ``ts``.

    A manual house override wins; a one-sided override takes the missing side
    from the active block / comfort pair without crossing the band. Otherwise
    an active block with temperatures replaces the corresponding sides of the
    house comfort pair, and a schedule gap simply falls back to that pair.
    Eco and "off" never appear at the house layer.
    """
    house = house or {}
    if _house_override_in_force(ts, house):
        sched = _house_block_targets_at_time(ts, house_blocks, house, block_temp_converter)
        h = house.get("override_heat")
        c = house.get("override_cool")
        h = h if h is not None else sched.heat
        c = c if c is not None else sched.cool
        if h is not None and c is not None and c < h:
            if house.get("override_cool") is None:
                c = h
            else:
                h = c
        return TargetTemps(heat=h, cool=c)
    return _house_block_targets_at_time(ts, house_blocks, house, block_temp_converter)


def house_target_source(
    ts: float,
    house_blocks: dict | None,
    house: dict,
    block_temp_converter: Callable[[float], float] | None = None,
) -> str:
    """Reason string for a house target: "override", "schedule" or "target"."""
    house = house or {}
    if _house_override_in_force(ts, house):
        return "override"
    if schedule_block_sets_target(ts, house_blocks, block_temp_converter):
        return "schedule"
    return "target"


def make_house_target_resolver(
    house_blocks: dict | None,
    house: dict,
    hass: HomeAssistant | None = None,
) -> Callable[[float], TargetTemps]:
    """Create a sync house target resolver with the temperature converter built once."""
    converter: Callable[[float], float] | None = None
    if hass is not None:
        from .temp_utils import ha_temp_to_celsius

        _hass = hass
        converter = lambda v: ha_temp_to_celsius(_hass, v)  # noqa: E731
    return lambda ts: resolve_house_targets_at_time(ts, house_blocks, house, converter)


def house_override_hold_until(hass: HomeAssistant, house: dict, now: float | None = None) -> float | None:
    """End of a house hold-until-schedule-change; identical rules to a room's.

    The house uses the same ``schedules`` / ``schedule_selector_entity`` keys as
    a room, so the room helper resolves its schedule entity directly.
    """
    return override_hold_until(hass, house, now=now)


def resolve_schedule_index(
    hass: HomeAssistant,
    room: dict,
    *,
    schedules_key: str = "schedules",
    selector_key: str = "schedule_selector_entity",
) -> int:
    """Return the 0-based index of the active schedule, or -1 if none.

    This is the single source of truth for schedule selector resolution,
    used by both the coordinator and schedule_utils helpers.

    Supports custom key names for reuse with different schedule types
    (e.g. cover schedules).
    """
    schedules = room.get(schedules_key, [])
    if not schedules:
        return -1

    selector_entity = room.get(selector_key, "")
    if not selector_entity:
        return 0

    state = hass.states.get(selector_entity)
    if state is None or state.state in ("unavailable", "unknown"):
        return 0

    if selector_entity.startswith("input_boolean."):
        return 1 if state.state == "on" else 0

    if selector_entity.startswith("input_number."):
        try:
            idx = int(float(state.state)) - 1  # 1-based → 0-based
        except (ValueError, TypeError):
            return 0
        if 0 <= idx < len(schedules):
            return idx
        return -1

    # Fallback for unknown entity domains
    return 0


def get_active_schedule_entity(
    hass: HomeAssistant,
    room: dict,
) -> str | None:
    """Return the entity_id of the currently active schedule, or None."""
    schedules = room.get("schedules", [])
    idx = resolve_schedule_index(hass, room)
    if 0 <= idx < len(schedules):
        return schedules[idx].get("entity_id", "") or None
    return None


def next_schedule_change(hass: HomeAssistant, room: dict) -> float | None:
    """Return the timestamp of the active schedule's next block boundary, or None without one."""
    entity_id = get_active_schedule_entity(hass, room)
    state = hass.states.get(entity_id) if entity_id else None
    return _next_event(state)


def _next_event(state: Any) -> float | None:
    raw = state.attributes.get("next_event") if state else None
    if isinstance(raw, str):
        raw = dt_util.parse_datetime(raw)
    if not isinstance(raw, datetime):
        return None
    return raw.timestamp()


def override_hold_until(hass: HomeAssistant, room: dict, now: float | None = None) -> float | None:
    """End of an override held until the schedule changes; None holds until changed.

    None without a schedule and under a schedule without blocks (it never changes).
    A schedule that cannot be read (missing, unavailable, a past `next_event`) ends
    the hold after `SCHEDULE_UNREADABLE_HOLD_HOURS`, else it would never end.
    """
    entity_id = get_active_schedule_entity(hass, room)
    if not entity_id:
        return None
    now = time.time() if now is None else now
    state = hass.states.get(entity_id)
    if state is not None and state.state not in ("unavailable", "unknown"):
        nxt = _next_event(state)
        if nxt is None and not state.attributes.get("next_event"):
            return None
        if nxt is not None and nxt > now:
            return nxt
    return now + SCHEDULE_UNREADABLE_HOLD_HOURS * 3600


async def read_schedule_blocks(
    hass: HomeAssistant,
    schedule_entity_id: str,
    cache: dict[str, dict] | None = None,
) -> dict | None:
    """Read weekly schedule blocks via schedule.get_schedule service.

    When ``cache`` is provided, successful reads are stored under the entity ID
    so subsequent failures fall back to the last good blocks. Without a cache,
    transient service failures cause the caller's target resolution to silently
    revert to comfort_temp (see #308).
    """
    if not schedule_entity_id or not schedule_entity_id.startswith("schedule."):
        return None

    blocks: dict | None = None
    error: Exception | None = None
    try:
        response = await hass.services.async_call(
            "schedule",
            "get_schedule",
            {"entity_id": schedule_entity_id},
            blocking=True,
            return_response=True,
        )
        if response and schedule_entity_id in response:
            result = response[schedule_entity_id]
            if isinstance(result, dict):
                blocks = dict(result)
    except Exception as err:  # noqa: BLE001
        error = err

    if blocks is not None:
        if cache is not None:
            cache[schedule_entity_id] = blocks
        return blocks

    cached = cache.get(schedule_entity_id) if cache is not None else None
    if cached is not None:
        _LOGGER.debug(
            "schedule.get_schedule unavailable for %s; using cached blocks (error=%r)",
            schedule_entity_id,
            error,
        )
        return cached

    _LOGGER.warning(
        "schedule.get_schedule unavailable for %s and no cached blocks; "
        "target will fall back to comfort/eco (error=%r)",
        schedule_entity_id,
        error,
    )
    return None


def schedule_sets_target(
    ts: float,
    room: dict,
    settings: dict,
    *,
    presence_away: bool,
    room_vacant: bool,
    house_override_active: bool = False,
) -> bool:
    """Whether the schedule (or comfort/eco without one) sets the target at ``ts``.

    Only then may the electricity price shift it: an override, a whole-home plant override,
    a deliberate house hold and vacation/absence lower the target on purpose, and an empty
    room never gets more than eco (#180). ``house_override_active`` is evaluated per
    timestamp by the caller, so a hold ending inside the horizon stops suppressing the
    price shift at the right block.
    """
    if presence_away or room_vacant or house_override_active:
        return False
    vacation_until = settings.get("vacation_until")
    if vacation_until is not None and settings.get("vacation_temp") is not None and ts < vacation_until:
        return False
    if is_plant_override_at(ts, settings, presence_away=presence_away):
        # A manual Central Thermostat wall setpoint is the user's own choice, not a price
        # opportunity: never price-shift a plant override.
        return False
    if room.get("override_heat") is None and room.get("override_cool") is None:
        return True
    override_until = room.get("override_until")
    return override_until is not None and ts >= override_until


def price_shifted(
    targets: TargetTemps,
    ts: float,
    room: dict,
    settings: dict,
    price_signal: PriceSignal | None,
    *,
    presence_away: bool,
    room_vacant: bool,
    shift_override: float | None = None,
    house_override_active: bool = False,
) -> tuple[TargetTemps, tuple[float, float]]:
    """Targets moved with the price where the room opted in, and the (heat, cool) shift applied.

    Central heating is a fuel furnace while central cooling is an electric heat pump, so the
    electricity price shifts the cooling side by default; the Central Plant setting
    ``price_shift_modes`` of "both" opts into shifting the heating target as well.
    ``shift_override`` replaces the room's own ``price_shift`` when following an active
    house, and ``house_override_active`` suppresses the shift during a house hold.
    """
    shift = float((room.get("price_shift") or 0.0) if shift_override is None else shift_override)
    if price_signal is None or shift <= 0 or room.get("is_outdoor"):
        return targets, (0.0, 0.0)
    if not schedule_sets_target(
        ts,
        room,
        settings,
        presence_away=presence_away,
        room_vacant=room_vacant,
        house_override_active=house_override_active,
    ):
        return targets, (0.0, 0.0)
    shifted = apply_price_shift(targets, price_signal.direction_at(ts), shift)

    def delta(new: float | None, old: float | None) -> float:
        return round(new - old, 3) if new is not None and old is not None else 0.0

    # Central heating is a fuel furnace while central cooling is an electric heat pump, so
    # an electricity price only ever charges the cooling side by default; "both" opts into
    # shifting the heating target as well.
    plant_cfg = settings.get("central_plant")
    mode = str((plant_cfg or {}).get("price_shift_modes", settings.get("price_shift_modes", "cool_only")))
    if mode == "cool_only":
        new_heat = targets.heat
        new_cool = shifted.cool
        if new_heat is not None and new_cool is not None and new_cool < new_heat:
            new_cool = new_heat
        shifted = TargetTemps(heat=new_heat, cool=new_cool)
    return shifted, (delta(shifted.heat, targets.heat), delta(shifted.cool, targets.cool))


def make_target_resolver(
    schedule_blocks: dict | None,
    room: dict,
    settings: dict,
    hass: HomeAssistant | None = None,
    presence_away: bool = False,
    mold_prevention_delta: float = 0.0,
    room_vacant: bool = False,
    price_signal: PriceSignal | None = None,
    house: HouseContext | None = None,
    follow_house: bool = False,
) -> Callable[[float], TargetTemps]:
    """Create a sync target resolver function (schedule blocks pre-fetched).

    Returns TargetTemps with None values for timestamps where devices should
    be turned off. With ``house`` (and ``follow_house``) set, the resolved
    ladder can stop at the house rung; the price shift and the hold suppression
    are then taken from the house context instead of the room. The house hold
    is checked per timestamp so a hold expiring inside the MPC horizon stops
    suppressing the shift at the right block.
    """
    comfort_heat = room.get("comfort_heat", room.get("comfort_temp", DEFAULT_COMFORT_HEAT))
    comfort_cool = room.get("comfort_cool", DEFAULT_COMFORT_COOL)
    eco_heat = room.get("eco_heat", room.get("eco_temp", DEFAULT_ECO_HEAT))
    eco_cool = room.get("eco_cool", DEFAULT_ECO_COOL)
    override_until = room.get("override_until")
    override_heat = room.get("override_heat")
    override_cool = room.get("override_cool")
    vacation_until = settings.get("vacation_until")
    vacation_temp = settings.get("vacation_temp")
    presence_away_action = settings.get("presence_away_action", "eco")
    schedule_off_action = settings.get("schedule_off_action", "eco")
    presence_clears_override = bool(settings.get("presence_clears_override", False))
    plant_override_until = settings.get("plant_override_until")
    plant_override_heat = settings.get("plant_override_heat")
    plant_override_cool = settings.get("plant_override_cool")
    climate_mode = room.get("climate_mode", "auto")
    active_house = house if house is not None and follow_house and house.active else None

    converter: Callable[[float], float] | None = None
    if hass is not None:
        from .temp_utils import ha_temp_to_celsius

        _hass = hass
        converter = lambda v: ha_temp_to_celsius(_hass, v)  # noqa: E731

    def resolver(ts: float) -> TargetTemps:
        targets, _ = resolve_targets_with_source(
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
            presence_away_action=presence_away_action,
            schedule_off_action=schedule_off_action,
            presence_clears_override=presence_clears_override,
            room_vacant=room_vacant,
            plant_override_until=plant_override_until,
            plant_override_heat=plant_override_heat,
            plant_override_cool=plant_override_cool,
            house=house,
            follow_house=follow_house,
        )
        if targets.heat is None and targets.cool is None:
            return targets
        hold_active = (
            bool(active_house.hold_active_at(ts))
            if active_house is not None and active_house.hold_active_at is not None
            else False
        )
        targets, _ = price_shifted(
            targets,
            ts,
            room,
            settings,
            price_signal,
            presence_away=presence_away,
            room_vacant=room_vacant,
            shift_override=active_house.price_shift if active_house is not None else None,
            house_override_active=hold_active,
        )
        return TargetTemps(
            *cap_cool_only_heat(
                targets.heat + mold_prevention_delta if targets.heat is not None else None,
                targets.cool,
                climate_mode,
            )
        )

    return resolver
