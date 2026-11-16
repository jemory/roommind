"""Constants for the RoomMind integration."""

import time
from typing import NamedTuple

from homeassistant.const import Platform
from homeassistant.core import Context

DOMAIN = "roommind"
VERSION = "3.0.0"

# Platforms
PLATFORMS = [Platform.SENSOR, Platform.SWITCH, Platform.BINARY_SENSOR, Platform.CLIMATE]

# Climate modes
CLIMATE_MODE_AUTO = "auto"
CLIMATE_MODE_HEAT_ONLY = "heat_only"
CLIMATE_MODE_COOL_ONLY = "cool_only"
CLIMATE_MODES = [CLIMATE_MODE_AUTO, CLIMATE_MODE_HEAT_ONLY, CLIMATE_MODE_COOL_ONLY]

# Override types
OVERRIDE_BOOST = "boost"
OVERRIDE_ECO = "eco"
OVERRIDE_CUSTOM = "custom"
OVERRIDE_TYPES = [OVERRIDE_BOOST, OVERRIDE_ECO, OVERRIDE_CUSTOM]
# Grace period after the override entity is switched on before the seeded
# override is evaluated; the caller's real targets normally arrive within it (#447).
OVERRIDE_TURN_ON_REFRESH_DELAY_S = 3.0
# A hold "until the schedule changes" while the schedule cannot be read ends after this
# (h, same as the dashboard card's fallback without a schedule) instead of never
SCHEDULE_UNREADABLE_HOLD_HOURS = 4

# Room modes
MODE_IDLE = "idle"
MODE_HEATING = "heating"
MODE_COOLING = "cooling"

# Global control strategy (settings["control_mode"]). Missing value = MPC, which
# is what every install ran before the setting was read by the controller.
CONTROL_MODE_MPC = "mpc"
CONTROL_MODE_BANGBANG = "bangbang"
DEFAULT_CONTROL_MODE = CONTROL_MODE_MPC

# Split heat/cool defaults
DEFAULT_COMFORT_HEAT = 21.0
DEFAULT_COMFORT_COOL = 24.0
DEFAULT_ECO_HEAT = 17.0

# What set a room's targets, recorded with the history for the analytics chart; "" = schedule
TARGET_REASON_OVERRIDE = "override"
TARGET_REASON_VACATION = "vacation"
TARGET_REASON_AWAY = "away"
TARGET_REASON_VACANT = "vacant"

# Per room: how far the schedule target may move with the electricity price (0 = off)
MAX_PRICE_SHIFT_K = 2.0
DEFAULT_ECO_COOL = 27.0

# Minutes all occupancy sensors must report "off" before an empty room drops to eco (#180)
DEFAULT_OCCUPANCY_ECO_DELAY_MINUTES = 15
# Shorter delays flip the target on every motion-sensor edge and flood battery TRVs with setpoints
MIN_OCCUPANCY_ECO_DELAY_MINUTES = 5

# Air quality (#267). CO2: UBA rates < 1000 ppm harmless, 1000-2000 ppm ventilate more
DEFAULT_CO2_THRESHOLD = 1000
MIN_CO2_THRESHOLD = 800
MAX_CO2_THRESHOLD = 5000
# Good again below 80 % of the threshold (1000 -> 800 ppm); own hysteresis choice, not a UBA value
AIR_QUALITY_CLEAR_FACTOR = 0.8
VENTILATION_DOMAINS = ("fan", "switch", "input_boolean")
VENTILATION_MIN_RUN_MINUTES = 15

# Context identifier for RoomMind-initiated service calls.
# Automations can check: trigger.context.parent_id == "roommind"
ROOMMIND_CONTEXT_ID = "roommind"


def make_roommind_context() -> Context:
    """Create a HA Context tagged as originating from RoomMind."""
    return Context(parent_id=ROOMMIND_CONTEXT_ID)


class TargetTemps(NamedTuple):
    """Dual-target temperatures for heating and cooling."""

    heat: float | None = None  # None = don't heat / force off
    cool: float | None = None  # None = don't cool / force off


# Plausibility bounds for resolved room target temperatures (°C).
# Schedule blocks are free-form YAML, so a typo (110 instead of 11) would
# otherwise be heated against for hours (#395). Same range as the override
# climate entity in climate.py, which imports these.
MIN_TARGET_TEMP = 5.0
MAX_TARGET_TEMP = 35.0

# Cold-start bang-bang fallback and the plant's outdoor temperature gates.
BANGBANG_HEAT_HYSTERESIS = 0.2  # °C below target → start heating (bang-bang fallback)
BANGBANG_COOL_HYSTERESIS = 0.2  # °C above target → start cooling (bang-bang fallback)
DEFAULT_OUTDOOR_COOLING_MIN = 16  # Hard block: NEVER cool if outdoor < this
DEFAULT_OUTDOOR_HEATING_MAX = 22  # Don't heat if outdoor > this
# Default comfort/efficiency slider value; the joint MPC weighs comfort against energy with it.
DEFAULT_COMFORT_WEIGHT = 70

# Update interval in seconds
UPDATE_INTERVAL = 30

# Sensor dropout: keep using cached temperature for this many seconds
# before falling back to idle (~10 coordinator cycles at 30s).
MAX_SENSOR_STALENESS = 300

# Coordinator throttle intervals (in cycles of UPDATE_INTERVAL)
HISTORY_WRITE_CYCLES = 6  # ~3 min at 30s cycle
THERMAL_SAVE_CYCLES = 30  # ~15 min
HISTORY_ROTATE_CYCLES = 360  # ~3 hours

# EKF update: accumulate observations before updating (better signal-to-noise)
EKF_UPDATE_MIN_DT = 3.0  # minutes — matches HISTORY_WRITE_CYCLES

# Outdoor sensor watchdog: notify when no valid outdoor temperature for this
# many coordinator cycles (60 × 30 s = 30 min).
OUTDOOR_UNAVAILABLE_NOTIFY_CYCLES = 60
OUTDOOR_UNAVAILABLE_NOTIFICATION_ID = "roommind_outdoor_unavailable"

# Mold risk detection & prevention
MOLD_RISK_OK = "ok"
MOLD_RISK_WARNING = "warning"
MOLD_RISK_CRITICAL = "critical"
MOLD_SURFACE_RH_WARNING = 70.0  # estimated surface RH % — warning threshold
MOLD_SURFACE_RH_CRITICAL = 80.0  # estimated surface RH % — critical threshold
DEFAULT_MOLD_HUMIDITY_THRESHOLD = 70.0  # room air RH % — notification trigger
DEFAULT_MOLD_SUSTAINED_MINUTES = 30  # minutes risk must persist before notification
DEFAULT_MOLD_COOLDOWN_MINUTES = 60  # minutes between repeated notifications per room
MOLD_PREVENTION_DELTAS = {"light": 1.0, "medium": 2.0, "strong": 3.0}
MOLD_HYSTERESIS = 5.0  # surface RH must drop this much below warning to clear
# ...and stay there this long: the estimate follows the air, which a heater warms in minutes
MOLD_PREVENTION_RELEASE_MINUTES = 30
DEFAULT_MOLD_F_RSI = 0.80  # DIN 4108-2 temperature factor; realistic for standard existing buildings


# Blind/cover control
COVER_SOLAR_MIN: float = 0.15
DEFAULT_COVERS_OUTDOOR_MIN_TEMP: float = 10.0  # °C outdoor below which the sun is not shaded (None = this)
COVER_HYSTERESIS: float = 1.0
COVER_MIN_HOLD_SECONDS: int = 900
COVER_POS_SCALE: float = 50.0
COVER_MAX_EFFECTIVENESS: float = 0.85
COVER_USER_CONFLICT_THRESHOLD: int = 15
COVER_USER_OVERRIDE_MINUTES: int = 60
COVER_TRANSITION_SETTLE_S: int = 90  # seconds after commanding before override detection activates
COVER_DEFAULT_BETA_S: float = 3.0  # °C/h per unit q_solar (default for rooms without learned data)
COVER_LINEAR_LOOKAHEAD_H: float = 1.0  # linear fallback: 1h (no heat-loss correction → keep short)
COVER_COOLING_LOOKAHEAD_H: float = 3.0  # rooms that cool: solar input the learned model counts (unit removes it all)
COVER_COOLING_LINEAR_H: float = 2.0  # same with the default gain (~2x a learned one) while the model is not confident
COVER_DIRECT_SUN_ANGLE_DEG: float = 60.0  # sun within this angle of the facade normal counts fully
COVER_PREDICTION_DT_MINUTES: float = 5.0  # time step for RC trajectory simulation
COVER_MAX_PREDICTION_STD: float = 0.5  # max idle+solar prediction_std to activate RC tier
COVER_CONFIDENCE_REFERENCE_SOLAR: float = 0.5  # reference q_solar for confidence check
COVER_MIN_IDLE_FOR_LEARNED: int = 30  # Min idle observations before trusting EKF's beta_s
COVER_POS_DEADBAND: int = 20  # min position change (%) to trigger motor movement
COVER_REOPEN_STEP: int = 40  # min partial reopening (%) within a shading episode
COVER_REOPEN_HOLD_SECONDS: int = 3600  # min time since the last move before a partial reopening
COVER_REOPEN_SNAP_PCT: int = 5  # a reopening ending this close to open (or the baseline) goes all the way
COVER_DAILY_LOOKAHEAD_H: float = 8.0  # hours ahead to search for daily solar peak (Tier 2)
COVER_SMOOTHING_MINUTES: float = 30.0  # EMA time constant for room temp and shading excess (AC cycles)
COVER_SMOOTHING_RISE_MINUTES: float = 10.0  # faster EMA while the shading excess climbs


# Far-future sentinel: vacation active indefinitely (year 2999)
VACATION_SENTINEL_UNTIL = 32503680000.0


def is_override_active(room: dict) -> bool:
    """Return True when a manual override is currently active."""
    if room.get("override_heat") is None and room.get("override_cool") is None:
        return False
    override_until = room.get("override_until")
    return override_until is None or time.time() < override_until


def is_override_suppressed(room: dict, settings: dict, presence_away: bool) -> bool:
    """Return True when an active override should be ignored due to presence."""
    if not presence_away:
        return False
    if room.get("ignore_presence", False):
        return False
    return bool(settings.get("presence_clears_override", False))


# ---------------------------------------------------------------------------
# Central HVAC adaptation
# ---------------------------------------------------------------------------
# RoomMind v2 drives one shared forced-air plant through a single Central
# Thermostat. Rooms contribute targets, openings and priorities; they no longer
# own devices. The per-device constants above stay for the legacy modules that
# are still imported during the migration waves.
MODE_CIRCULATING = "circulating"
# Every action the joint whole-home optimizer can choose for the plant.
PLANT_MODES = [MODE_IDLE, MODE_HEATING, MODE_COOLING, MODE_CIRCULATING]
TARGET_REASON_PLANT_OVERRIDE = "plant_override"

# How a room's occupancy drives its target resolution and Room Priority.
OCCUPANCY_MODE_REACTIVE_TREND = "reactive_and_trend"
OCCUPANCY_MODE_SCHEDULE_ASSUMED = "schedule_assumed_when_on"
OCCUPANCY_MODE_FOLLOW_SCHEDULE = "follow_schedule"
OCCUPANCY_MODE_GUARDRAIL_ONLY = "guardrail_only"
OCCUPANCY_MODES = [
    OCCUPANCY_MODE_REACTIVE_TREND,
    OCCUPANCY_MODE_SCHEDULE_ASSUMED,
    OCCUPANCY_MODE_FOLLOW_SCHEDULE,
    OCCUPANCY_MODE_GUARDRAIL_ONLY,
]
# How long a room must stay empty before it counts as unoccupied: shorter dwell
# lets one doorway crossing drop a room to its guardrail band.
DEFAULT_OCCUPANCY_DWELL_MINUTES = 3.0

# What the whole plant does while an Exterior Opening reports open.
EXTERIOR_OPEN_PAUSE_UNLESS_DOOR_CLOSED = "pause_unless_door_closed"
EXTERIOR_OPEN_PAUSE_ALWAYS = "pause_always"
EXTERIOR_OPEN_EXCLUDE_ROOM_ONLY = "exclude_room_only"
EXTERIOR_OPEN_POLICIES = [
    EXTERIOR_OPEN_PAUSE_UNLESS_DOOR_CLOSED,
    EXTERIOR_OPEN_PAUSE_ALWAYS,
    EXTERIOR_OPEN_EXCLUDE_ROOM_ONLY,
]

# Which targets the electricity-price shift may widen.
PRICE_SHIFT_COOL_ONLY = "cool_only"
PRICE_SHIFT_BOTH = "both"
PRICE_SHIFT_MODES = [PRICE_SHIFT_COOL_ONLY, PRICE_SHIFT_BOTH]

# Interior doors cut a room off from the return path: the supply still heats or
# cools it, but the room no longer mixes with the House Reference Temperature.
DEFAULT_INTERIOR_DOOR_DELAY_S = 60
DEFAULT_DOOR_CLOSED_MIX_FACTOR = 0.40
DOOR_CLOSED_SUPPLY_FACTOR = 0.75
# Air-vs-mass relaxation of the forced-air rebound after a plant run stops.
FORCED_AIR_REBOUND_TAU_MINUTES = 10.0
FORCED_AIR_REBOUND_CUTOFF = 0.01

# Central Thermostat actuation: the setpoint nudge sits just past the
# thermostat's own reading rather than at a remote room's target.
DEFAULT_NUDGE_DELTA_C = 1.2
DEFAULT_MAX_NUDGE_HEADROOM_C = 2.5
# Safe band parked on the thermostat while the plant is idle or circulating so
# it protects the home without Home Assistant.
DEFAULT_FALLBACK_HEAT_C = 16.5
DEFAULT_FALLBACK_COOL_C = 26.5
DEFAULT_MIN_DEADBAND_C = 1.7
DEFAULT_CIRCULATE_FAN_MODE = "on"
DEFAULT_PLANT_IDLE_FAN_MODE = "auto"
# Equipment protection: short-cycling guards for the compressor, the blower and
# the heat/cool changeover.
DEFAULT_MIN_RUN_MINUTES = 10
DEFAULT_MIN_OFF_MINUTES = 5
DEFAULT_MIN_FAN_MINUTES = 10
DEFAULT_CHANGEOVER_MINUTES = 30
# An override on the Central Thermostat itself keeps plant priority this long.
DEFAULT_THERMOSTAT_OVERRIDE_HOURS = 2.0
# Dehumidification may over-cool slightly; a wet coil locks Circulation out
# until it has drained to avoid raising indoor humidity.
DEFAULT_DEHUMIDIFY_RH_THRESHOLD = 60.0
DEFAULT_MAX_DEHUMIDIFY_OVERCOOL_C = 0.8
DEFAULT_WET_COIL_LOCKOUT_MINUTES = 25
DEFAULT_WET_COIL_MAX_RH = 52.0
# Room Priority weights: an active override outranks occupancy, occupancy
# outranks the guardrail band that keeps empty rooms from drifting.
OVERRIDE_PRIORITY_WEIGHT = 3.0
AWAY_PRIORITY_WEIGHT = 0.25
GUARDRAIL_PRIORITY_WEIGHT = 0.08
# Never let the plant idle below this: freeze protection outranks energy.
FREEZE_PROTECTION_TEMP_C = 7.0
# Share of a conditioning run's energy that the blower alone accounts for.
FAN_ENERGY_FRACTION = 0.08

DEFAULT_CENTRAL_PLANT_SETTINGS: dict[str, object] = {
    "thermostat_entity": "",
    "thermostat_area_id": "",
    "nudge_delta_c": DEFAULT_NUDGE_DELTA_C,
    "max_nudge_headroom_c": DEFAULT_MAX_NUDGE_HEADROOM_C,
    "fallback_heat_c": DEFAULT_FALLBACK_HEAT_C,
    "fallback_cool_c": DEFAULT_FALLBACK_COOL_C,
    "min_deadband_c": DEFAULT_MIN_DEADBAND_C,
    "circulate_fan_mode": DEFAULT_CIRCULATE_FAN_MODE,
    "idle_fan_mode": DEFAULT_PLANT_IDLE_FAN_MODE,
    "min_run_minutes": DEFAULT_MIN_RUN_MINUTES,
    "min_off_minutes": DEFAULT_MIN_OFF_MINUTES,
    "min_fan_minutes": DEFAULT_MIN_FAN_MINUTES,
    "changeover_minutes": DEFAULT_CHANGEOVER_MINUTES,
    "follow_thermostat_setpoint": True,
    "thermostat_override_hours": DEFAULT_THERMOSTAT_OVERRIDE_HOURS,
    "exterior_open_policy": EXTERIOR_OPEN_PAUSE_UNLESS_DOOR_CLOSED,
    "humidity_comfort_enabled": True,
    "dehumidify_rh_threshold": DEFAULT_DEHUMIDIFY_RH_THRESHOLD,
    "max_dehumidify_overcool_c": DEFAULT_MAX_DEHUMIDIFY_OVERCOOL_C,
    "wet_coil_lockout_minutes": DEFAULT_WET_COIL_LOCKOUT_MINUTES,
    "wet_coil_max_rh": DEFAULT_WET_COIL_MAX_RH,
    "price_shift_modes": PRICE_SHIFT_COOL_ONLY,
}


def get_central_plant_settings(settings: dict) -> dict:
    """Return the central plant settings with every unspecified key defaulted.

    Always a fresh dict: callers may mutate the result without touching either
    the global settings or the module-level defaults.
    """
    merged = dict(DEFAULT_CENTRAL_PLANT_SETTINGS)
    central = settings.get("central_plant")
    if isinstance(central, dict):
        merged.update(central)
    return merged


def is_plant_override_active(settings: dict) -> bool:
    """Return True when a manual override on the Central Thermostat is active.

    Accepts either the full global settings dict or an already resolved central
    plant settings dict, so callers need not unwrap ``central_plant`` first.
    """
    central = settings.get("central_plant")
    sources = [settings, central] if isinstance(central, dict) else [settings]
    for source in sources:
        if source.get("plant_override_heat") is None and source.get("plant_override_cool") is None:
            continue
        override_until = source.get("plant_override_until")
        return override_until is None or time.time() < override_until
    return False


def build_override_live(room: dict, suppressed: bool = False) -> dict:
    """Build override fields for live data from a room config dict.

    When *suppressed* is True the override is held in the store but currently
    has no effect (e.g. presence-away with presence_clears_override enabled).
    Live data still reports the underlying intent so the UI can render a
    "paused" indicator.
    """
    active = is_override_active(room)
    return {
        "override_active": active,
        "override_type": room.get("override_type") if active else None,
        "override_heat": room.get("override_heat") if active else None,
        "override_cool": room.get("override_cool") if active else None,
        "override_until": room.get("override_until") if active else None,
        "override_suppressed": active and suppressed,
    }
