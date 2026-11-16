"""Device models: TRV/radiator, underfloor zone, split AC, central heat pump, heater, boiler.

``central_hvac`` is the North American forced-air plant built on the ``heat_pump``
machinery: furnace heating, heat-pump cooling with a wet evaporator coil, and a blower
that keeps recirculating the served rooms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..physics.psychro import LATENT_HEAT, abs_humidity
from ..physics.zone import Gains
from .base import CommandResult, DeviceEnv, DuctState, SimDevice, approach, clamp

GROUND_TEMP_C = 12.0
VALVE_TRAVEL_S = 120.0
UFH_ACTUATOR_TRAVEL_S = 180.0
COIL_TEMP_C = 10.0
MAX_SUPPLY_C = 75.0
# Forced-air defaults; a profile overrides any of them per device.
CIRCULATION_ACH = 2.5  # blower circulation while the plant heats or cools
FAN_CIRCULATION_ACH = 1.8  # blower circulation with neither burner nor compressor
DOOR_CLOSED_SUPPLY_FACTOR = 0.75  # duct supply of a room behind a shut door
DOOR_CLOSED_RETURN_FACTOR = 0.35  # its share of the return-plenum loop
MAX_COIL_WATER_KG = 0.6  # film the evaporator holds before the rest drips to the pan
COIL_EVAP_TAU_S = 600.0  # wet-coil re-evaporation into the duct air
COIL_DRAIN_TAU_S = 1500.0  # passive drain into the condensate pan
_VALVE_SERVICES = {"set_valve_opening_degree": "opening", "set_valve_closing_degree": "closing"}


def _gains(heat_w: float, radiant_fraction: float) -> Gains:
    return Gains(conv_w=heat_w * (1 - radiant_fraction), rad_w=heat_w * radiant_fraction)


def frost_rate(p: dict[str, Any], t_out: float, rh: float) -> float:
    """Share of a full frost load the outdoor coil gathers per second of heating.

    Frost needs a coil below 0 °C in humid air: none above ~6 °C outdoor, most around
    -5..+3 °C, less in dry deep cold. ``defrost_interval_s`` is the shortest cycle (worst
    conditions); 30-90 min between defrosts is the usual range. Assumed shape, not fitted."""
    interval = float(p.get("defrost_interval_s", 2700))
    if interval <= 0:
        return 0.0
    if t_out >= 6.0 or t_out <= -20.0:
        f_t = 0.0
    elif t_out > 3.0:
        f_t = (6.0 - t_out) / 3.0
    elif t_out >= -5.0:
        f_t = 1.0
    else:
        f_t = (t_out + 20.0) / 15.0
    return f_t * clamp((rh - 60.0) / 30.0, 0.0, 1.0) / interval


def advance_frost(dev: Any, dt: float, env: DeviceEnv, heating: bool) -> bool:
    """Frost and defrost bookkeeping of an air-source unit; True while it defrosts.

    A defrost reverses the cycle for ``defrost_s``: no heat reaches the room, the
    compressor keeps running. Frost thaws off a stopped coil above +2 °C outdoor."""
    p = dev.spec.params
    if heating:
        if env.now < dev.defrost_until:
            return True
        dev.frost += dt * frost_rate(p, env.outdoor.temp_c, env.outdoor.rh)
        if dev.frost >= 1.0:
            dev.frost = 0.0
            dev.defrost_until = env.now + float(p.get("defrost_s", 300))
            return True
    elif env.outdoor.temp_c > 2.0:
        dev.frost = max(0.0, dev.frost - dt / 1800.0)
    return False


def curve_supply_c(p: dict[str, Any], t_out: float, design_default: float) -> float:
    """Supply temperature of a weather-compensated heating curve (straight line from
    ``supply_foot_c`` at +20 °C outdoor to ``supply_temp_c`` at ``design_outdoor_c``).
    ``heating_curve: false`` keeps the supply fixed at ``supply_temp_c``."""
    design = float(p.get("supply_temp_c", design_default))
    if not p.get("heating_curve", True):
        return design
    foot = float(p.get("supply_foot_c", 25.0))
    design_out = float(p.get("design_outdoor_c", -12.0))
    share = (20.0 - t_out) / (20.0 - design_out)
    return clamp(foot + (design - foot) * share, foot, MAX_SUPPLY_C)


@dataclass
class Trv(SimDevice):
    """TRV on a hot-water radiator. The head measures air + bias toward the radiator and
    runs a P controller, so a proportional setpoint from RoomMind acts like real life."""

    model = "trv"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.t_rad: float | None = None
        self.valve = 0.0
        self.valve_opening_deg = 100
        self.valve_closing_deg = self._default_closing_deg()
        self._next_report = 0.0

    def _idle_position(self) -> bool:
        """TRV-ZBT: the second number is the opening while idle, not a closing degree."""
        return bool(self.spec.capabilities.get("valve_idle_position"))

    def _default_closing_deg(self) -> int:
        """Untouched TRV: closed while idle (TRVZB closing degree 100, TRV-ZBT idle position 0)."""
        return 0 if self._idle_position() else 100

    def _apply(self, service: str, data: dict[str, Any]) -> CommandResult:
        kind = _VALVE_SERVICES.get(service)
        if kind is None:
            return super()._apply(service, data)
        if kind not in self.spec.valve_numbers():
            return CommandResult(False, f"unsupported_service:{service}")
        value = round(float(data["value"]))
        if not 0 <= value <= 100:
            return CommandResult(False, "out_of_range")
        attr = f"valve_{kind}_deg"
        changed = getattr(self, attr) != value
        setattr(self, attr, value)
        return CommandResult(True, changed=changed)

    def _valve_position(self, demand: float) -> float:
        """TRVZB semantics: open position = opening degree, closed position = 100 - closing
        degree; the P controller's demand moves the valve between both. TRV-ZBT: closed
        position = idle position. Crossed degrees (open position below closed position)
        clamp the demand into their range instead of inverting it."""
        lo = self.valve_closing_deg / 100.0 if self._idle_position() else 1.0 - self.valve_closing_deg / 100.0
        hi = self.valve_opening_deg / 100.0
        if hi >= lo:
            return lo + demand * (hi - lo)
        return clamp(demand, hi, lo)

    def flow_share(self) -> float:
        """Share of full water flow at the current valve position (dead zone, then position ** exponent)."""
        p = self.spec.params
        dead = float(p.get("valve_dead_zone", 0.0))
        if self.valve <= dead:
            return 0.0
        return ((self.valve - dead) / (1.0 - dead)) ** float(p.get("valve_flow_exponent", 1.0))

    def _design_dt(self) -> float:
        """Mean radiator excess over a 20 °C room at the design supply (steady state, valve open)."""
        p = self.spec.params
        ratio = float(p.get("flow_per_h", 1.5))
        supply = float(p.get("supply_temp_c", 55))
        return (ratio * supply + 20.0) / (ratio + 1.0) - 20.0

    def _size(self, env: DeviceEnv) -> float:
        """Radiator W/K at the design point: explicit, or sized to the room's heat load."""
        p = self.spec.params
        if "h_radiator_w_k" in p:
            return float(p["h_radiator_w_k"])
        ua = env.zone.params.h_air_out + env.zone.params.h_mass_out
        design_w = float(p.get("design_factor", 1.3)) * ua * (20.0 - float(p.get("design_outdoor_c", -12)))
        return design_w / self._design_dt()

    def _physics(self, dt: float, env: DeviceEnv) -> dict[str, Gains]:
        p = self.spec.params
        z = env.zone
        h_rad = self._size(env)
        if self.t_rad is None:
            self.t_rad = z.t_air
        reading = z.t_air + float(p.get("sensor_bias", 0.04)) * (self.t_rad - z.t_air)
        if env.now >= self._next_report:
            self.sensor_c = round(reading, 1)
            self._next_report = env.now + float(p.get("report_interval_s", 60))
        demand = 0.0
        if self.hvac_mode in ("heat", "auto", "heat_cool") and self.available:
            demand = clamp((self.target_c - reading) / float(p.get("p_band_k", 1.0)), 0.0, 1.0)
        self.valve = approach(self.valve, self._valve_position(demand), 1.0 / VALVE_TRAVEL_S, dt)
        supply = env.supply_temp_c if env.supply_temp_c is not None else curve_supply_c(p, env.outdoor.temp_c, 55.0)
        flow = float(p.get("flow_w_k", h_rad * float(p.get("flow_per_h", 1.5))))
        capacity_j_k = float(p.get("radiator_capacity_kj_k", h_rad * float(p.get("capacity_kj_per_h", 1.75)))) * 1000
        q_in = self.flow_share() * flow * max(0.0, supply - self.t_rad)
        # EN 442: output grows with the excess temperature to the power n (~1.3 for panels)
        dt_design = self._design_dt()
        excess = self.t_rad - z.t_air
        n = float(p.get("radiator_exponent", 1.3))
        q_out = h_rad * dt_design * math.copysign((abs(excess) / dt_design) ** n, excess)
        self.t_rad += dt * (q_in - q_out) / capacity_j_k
        self.active = self.valve > 0.05 and q_in > 1.0
        self.hvac_action = "off" if self.hvac_mode == "off" else ("heating" if self.active else "idle")
        self.water_draw_w = q_in
        self._account(dt, q_out, 0.0)
        return {self.room: _gains(q_out, float(p.get("radiant_fraction", 0.3)))}

    def _model_state(self) -> dict[str, Any]:
        return {
            "t_rad": self.t_rad,
            "valve": self.valve,
            "valve_opening_deg": self.valve_opening_deg,
            "valve_closing_deg": self.valve_closing_deg,
        }

    def _restore_model_state(self, data: dict[str, Any]) -> None:
        self.t_rad = data.get("t_rad")
        self.valve = data.get("valve", 0.0)
        self.valve_opening_deg = data.get("valve_opening_deg", 100)
        self.valve_closing_deg = data.get("valve_closing_deg", self._default_closing_deg())


@dataclass
class Ufh(SimDevice):
    """Underfloor zone: on/off room thermostat, thermal actuator, screed slab node."""

    model = "ufh"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.t_slab: float | None = None
        self.actuator = 0.0
        self.calling = False

    def _physics(self, dt: float, env: DeviceEnv) -> dict[str, Gains]:
        p = self.spec.params
        z = env.zone
        area = z.params.floor_area_m2 * float(p.get("coverage", 0.9))
        if self.t_slab is None:
            self.t_slab = z.t_air
        reading = z.t_air + float(p.get("sensor_bias_k", 0.0))
        self.sensor_c = round(reading, 1)
        hyst = float(p.get("hysteresis_k", 0.3))
        if self.hvac_mode != "heat" or not self.available:
            self.calling = False
        elif reading <= self.target_c - hyst:
            self.calling = True
        elif reading >= self.target_c + hyst:
            self.calling = False
        self.actuator = approach(self.actuator, 1.0 if self.calling else 0.0, 1.0 / UFH_ACTUATOR_TRAVEL_S, dt)
        supply = env.supply_temp_c if env.supply_temp_c is not None else curve_supply_c(p, env.outdoor.temp_c, 35.0)
        q_in = self.actuator * float(p.get("flow_w_per_m2k", 8)) * area * max(0.0, supply - self.t_slab)
        q_up = float(p.get("h_slab_air_w_per_m2k", 10.8)) * area * (self.t_slab - z.t_air)
        q_down = float(p.get("h_slab_ground_w_per_m2k", 0.5)) * area * (self.t_slab - GROUND_TEMP_C)
        cap = float(p.get("slab_capacity_kj_k_per_m2", 90)) * 1000 * area
        self.t_slab += dt * (q_in - q_up - q_down) / cap
        self.active = self.actuator > 0.5
        self.hvac_action = "off" if self.hvac_mode == "off" else ("heating" if self.active else "idle")
        self.water_draw_w = q_in
        self._account(dt, q_up, 0.0)
        return {self.room: _gains(q_up, float(p.get("radiant_fraction", 0.6)))}

    def _model_state(self) -> dict[str, Any]:
        return {"t_slab": self.t_slab, "actuator": self.actuator, "calling": self.calling}

    def _restore_model_state(self, data: dict[str, Any]) -> None:
        self.t_slab = data.get("t_slab")
        self.actuator = data.get("actuator", 0.0)
        self.calling = data.get("calling", False)


@dataclass
class SplitAc(SimDevice):
    """Inverter split AC. Controls on its return-air sensor (offset while heating),
    modulates, has its own deadband and minimum run, dehumidifies while cooling.

    The return-air offset builds up and fades with ``sensor_offset_tau_s``: warm air
    collects at the intake while the compressor runs and is still there right after
    it stops. With the compressor stopped it settles at ``sensor_offset_idle_k``
    (default 0): a wall unit under the ceiling reads the warm layer once no air
    moves past it.

    ``capacity_tau_s`` lags the delivered output behind the compressor (inverter soft
    start, refrigerant and coil pull-down; the coil also keeps working briefly after a
    stop). ``min_off_s`` is the restart delay after a stop (anti-short-cycle timer).

    Real units ramp up with a time constant of 40-76 s (DOE/AHRI cycling tests). The
    profile keeps ``capacity_tau_s`` at 0: with a 3.5 kW unit at full power the lag
    turned into short cycles and a mislearned model. The earlier explanation (air node
    2-4x faster than real rooms) rested on a throttled real unit (ECO mode) and is void.

    ``max_output`` caps the delivered share (ECO/quiet modes run the inverter well below
    its rating). In heat mode the outdoor coil frosts and defrosts (``frost_rate``)."""

    model = "ac"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.running_mode: str | None = None  # "heat" | "cool" while the compressor runs
        self.on_since = 0.0
        self.modulation = 0.0
        self.output = 0.0  # delivered share of capacity, lags the compressor
        self.output_mode: str | None = None
        self.off_since = -1e12
        self.coil_water_kg = 0.0
        self.sensor_offset_k = 0.0
        self.frost = 0.0  # outdoor coil, 1.0 = defrost due
        self.defrost_until = -1e12

    def _update_sensor_offset(self, dt: float) -> None:
        p = self.spec.params
        goal = float(p.get("sensor_offset_idle_k", 0.0))
        if self.running_mode == "heat":
            goal = float(p.get("sensor_offset_heat_k", 1.5))
        elif self.running_mode == "cool":
            goal = float(p.get("sensor_offset_cool_k", 0.0))
        tau = float(p.get("sensor_offset_tau_s", 0.0))
        if tau <= 0:
            self.sensor_offset_k = goal
        else:
            self.sensor_offset_k += (goal - self.sensor_offset_k) * (1 - math.exp(-dt / tau))

    def _sensor(self, t_air: float) -> float:
        return t_air + self.sensor_offset_k

    def _wanted(self, reading: float) -> str | None:
        p = self.spec.params
        dead = float(p.get("deadband_k", 0.5))
        restart = float(p.get("restart_k", 0.5))
        mode = self.hvac_mode
        if mode in ("heat_cool", "auto"):
            low = self.to_c(self.target_low) if self.target_low is not None else self.target_c - 1
            high = self.to_c(self.target_high) if self.target_high is not None else self.target_c + 1
            if self.running_mode == "heat":
                return "heat" if reading < low + dead else None
            if self.running_mode == "cool":
                return "cool" if reading > high - dead else None
            if reading <= low - restart:
                return "heat"
            if reading >= high + restart:
                return "cool"
            return None
        sp = self.target_c
        if mode == "heat":
            if self.running_mode == "heat":
                return "heat" if reading < sp + dead else None
            return "heat" if reading <= sp - restart else None
        if mode in ("cool", "dry"):
            if self.running_mode == "cool":
                return "cool" if reading > sp - dead else None
            return "cool" if reading >= sp + restart else None
        return None

    def _physics(self, dt: float, env: DeviceEnv) -> dict[str, Gains]:
        p = self.spec.params
        z = env.zone
        t_out = env.outdoor.temp_c
        self._update_sensor_offset(dt)
        reading = self._sensor(z.t_air)
        self.sensor_c = round(reading * 2) / 2
        wanted = self._wanted(reading) if self.available and self.hvac_mode != "off" else None
        min_run = float(p.get("internal_min_run_s", 180))
        if self.running_mode and wanted != self.running_mode:
            # Mode change or "off" commanded stops at once; reaching the setpoint waits for min run.
            commanded_stop = self.hvac_mode == "off" or not self.available or (wanted is not None)
            if commanded_stop or self.hvac_mode in ("fan_only",) or env.now - self.on_since >= min_run:
                self.running_mode = None
                self.off_since = env.now
        if (
            self.running_mode is None
            and wanted is not None
            and env.now - self.off_since >= float(p.get("min_off_s", 0))
        ):
            self.running_mode = wanted
            self.on_since = env.now
        heat_w = 0.0
        electric = 0.0
        moisture = 0.0
        if self.running_mode:
            if self.running_mode == "heat":
                sp = self.to_c(self.target_low) if self.target_low is not None else self.target_c
                err = max(0.0, sp - reading)
            else:
                err = max(0.0, reading - self.target_c)
            lo = float(p.get("min_modulation", 0.3))
            top = float(p.get("max_output", 1.0))
            self.modulation = (
                0.4 if self.hvac_mode == "dry" else clamp(lo + float(p.get("kp", 0.6)) * err, min(lo, top), top)
            )
        else:
            self.modulation = 0.0
        if self.running_mode and self.output_mode not in (None, self.running_mode):
            self.output = 0.0  # reversing the cycle starts from scratch
        defrosting = advance_frost(self, dt, env, self.running_mode == "heat")
        tau = float(p.get("capacity_tau_s", 0.0))
        if defrosting:
            self.output = 0.0  # heating restarts from scratch after the defrost
        elif tau <= 0:
            self.output = self.modulation
        else:
            self.output += (self.modulation - self.output) * (1 - math.exp(-dt / tau))
        if self.running_mode:
            self.output_mode = self.running_mode
        elif self.output < 1e-3 or self.hvac_mode == "off" or not self.available:
            # with the indoor fan stopped the cold coil no longer reaches the room
            self.output, self.output_mode = 0.0, None
        out_mode = self.output_mode
        if out_mode and self.output > 0:
            if out_mode == "heat":
                cap = float(p.get("heat_capacity_w", 3500)) * max(
                    0.3, 1 - float(p.get("heat_derate_per_k", 0.02)) * max(0.0, 7 - t_out)
                )
                heat_w = cap * self.output
                a, b = p.get("cop_heat", [3.2, 0.06])
                electric = heat_w / max(1.5, a + b * t_out)
            else:
                q = float(p.get("cool_capacity_w", 3500)) * self.output
                shr = 0.5 if self.hvac_mode == "dry" else float(p.get("shr", 0.75))
                electric = q / float(p.get("eer", 3.5))
                if z.x > abs_humidity(COIL_TEMP_C, 100):
                    latent = q * (1 - shr)
                    moisture = -latent / LATENT_HEAT
                    self.coil_water_kg += -moisture * dt
                    heat_w = -q * shr
                else:
                    heat_w = -q
        if defrosting:
            electric = float(p.get("heat_capacity_w", 3500)) * 0.1
        if not self.running_mode:
            electric = 0.0  # what the coil still delivers after a stop costs no compressor power
            if self.coil_water_kg > 0:
                rate = 3e-5 if self.hvac_mode == "fan_only" else 3e-6  # kg/s re-evaporation / drain
                evap = min(self.coil_water_kg, rate * dt)
                self.coil_water_kg -= evap
                if self.hvac_mode == "fan_only":
                    moisture += evap / dt
        fan_on = self.hvac_mode != "off" and self.available
        electric += 25.0 if fan_on else 2.0
        self.active = self.running_mode is not None
        self.hvac_action = {
            # most integrations keep showing "heating" through a defrost; some report it
            "heat": str(p.get("defrost_action", "heating")) if defrosting else "heating",
            "cool": "drying" if self.hvac_mode == "dry" else "cooling",
        }.get(
            self.running_mode or "",
            "fan" if self.hvac_mode == "fan_only" else ("off" if self.hvac_mode == "off" else "idle"),
        )
        self._account(dt, heat_w, electric)
        return {self.room: Gains(conv_w=heat_w, moisture_kg_s=moisture)}

    def _model_state(self) -> dict[str, Any]:
        return {
            "running_mode": self.running_mode,
            "on_since": self.on_since,
            "off_since": self.off_since,
            "output": self.output,
            "output_mode": self.output_mode,
            "coil_water_kg": self.coil_water_kg,
            "sensor_offset_k": self.sensor_offset_k,
            "frost": self.frost,
            "defrost_until": self.defrost_until,
        }

    def _restore_model_state(self, data: dict[str, Any]) -> None:
        self.running_mode = data.get("running_mode")
        self.on_since = data.get("on_since", 0.0)
        self.off_since = data.get("off_since", -1e12)
        self.output = data.get("output", 0.0)
        self.output_mode = data.get("output_mode")
        self.coil_water_kg = data.get("coil_water_kg", 0.0)
        self.sensor_offset_k = data.get("sensor_offset_k", 0.0)
        self.frost = data.get("frost", 0.0)
        self.defrost_until = data.get("defrost_until", -1e12)


@dataclass
class HeatPump(SimDevice):
    """Central ducted heat pump behind a thermostat (Ecobee-like), range setpoint.

    The air handler is part of the model: a blower moves the same air through the served
    rooms (``serves_rooms`` besides its own room), the supply is split by
    ``supply_shares``, and every room whose interior door is shut is throttled
    (``door_closed_supply_factor``, default 0.75) while it is served. The evaporator
    stays wet after a cooling run (``coil_water_kg``, capped at ``max_coil_water_kg``,
    default 0.6 kg); with the compressor and the burner off, the blower re-evaporates that
    film into the duct air (time constant ``coil_evap_tau_s``, 600 s) instead of draining
    it (``coil_drain_tau_s``, 1500 s). That is the physics behind RoomMind's
    ``WetCoilTracker``: humidity rises in every served room while only the fan runs.

    ``furnace_heating`` swaps the compressor for a furnace on the heating side: constant
    ``heat_capacity_w`` independent of outdoor temperature, no defrost cycle, and no COP
    (the electric figure is the fuel-equivalent, ``furnace_efficiency``).
    """

    model = "heat_pump"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.running_mode: str | None = None  # "heat" | "cool" while burner/compressor runs
        self.on_since = 0.0
        self.frost = 0.0
        self.defrost_until = -1e12
        self.coil_water_kg = 0.0  # evaporator water film from condensation
        self.coil_evap_kg_s = 0.0  # last tick's wet-coil re-evaporation
        self.coil_drip_kg_s = 0.0  # last tick's condensate overflowing to the drain

    # --- central air handler -------------------------------------------------------------

    @property
    def furnace_heating(self) -> bool:
        """A furnace instead of the heat pump's compressor for heating."""
        return bool(self.spec.params.get("furnace_heating", self.model == "central_hvac"))

    @property
    def duct_rooms(self) -> list[str]:
        rooms = [self.room, *(self.spec.params.get("serves_rooms") or [])]
        return list(dict.fromkeys(str(r) for r in rooms))

    def blower_on(self) -> bool:
        if not self.available or self.hvac_mode == "off":
            return False
        if self.running_mode is not None:
            return True
        return self.hvac_mode == "fan_only" or self.fan_mode == "on"

    def circulating(self) -> bool:
        """The blower runs, but neither the compressor nor the burner does."""
        return self.blower_on() and self.running_mode is None

    def _design_shares(self) -> dict[str, float]:
        """Normalized design split over ``duct_rooms``; unlisted rooms share equally."""
        rooms = self.duct_rooms
        if not rooms:
            return {}
        configured = self.spec.params.get("supply_shares") or {}
        equal = 1.0 / len(rooms)
        raw = {r: max(0.0, float(configured.get(r, equal))) for r in rooms}
        total = sum(raw.values())
        if total <= 0:
            return dict.fromkeys(rooms, equal)
        return {r: v / total for r, v in raw.items()}

    def supply_weights(self, env: DeviceEnv) -> dict[str, float]:
        """Share of the unit's output each served room gets this tick.

        A closed interior door throttles its room (``door_closed_supply_factor``); the
        rest is renormalized, so shutting a door pushes the air to the other rooms
        instead of losing it.
        """
        shares = self._design_shares()
        factor = float(self.spec.params.get("door_closed_supply_factor", DOOR_CLOSED_SUPPLY_FACTOR))
        raw = {r: s * (factor if r in env.closed_rooms else 1.0) for r, s in shares.items() if r in env.zones}
        total = sum(raw.values())
        if total <= 0:
            return {}
        return {r: v / total for r, v in raw.items()}

    def duct_state(self) -> DuctState | None:
        p = self.spec.params
        if not self.blower_on():
            return None
        if len(self.duct_rooms) < 2 and "circulation_ach" not in p and self.model != "central_hvac":
            return None  # one room, no configured circulation: nothing to mix
        ach = (
            float(p.get("fan_circulation_ach", FAN_CIRCULATION_ACH))
            if self.circulating()
            else float(p.get("circulation_ach", CIRCULATION_ACH))
        )
        return DuctState(
            rooms=self.duct_rooms,
            shares=self._design_shares(),
            circulation_ach=ach,
            circulating=self.circulating(),
            door_closed_return_factor=float(p.get("door_closed_return_factor", DOOR_CLOSED_RETURN_FACTOR)),
        )

    # --- evaporator water film ------------------------------------------------------------

    def _condense(self, kg: float, dt: float) -> None:
        """Moisture pulled out of the duct air onto the coil; the overflow drips away."""
        cap = float(self.spec.params.get("max_coil_water_kg", MAX_COIL_WATER_KG))
        before = self.coil_water_kg
        self.coil_water_kg = max(0.0, min(cap, before + kg))
        self.coil_drip_kg_s = max(0.0, before + kg - self.coil_water_kg) / dt if dt > 0 else 0.0

    def _wet_coil_evap(self, dt: float) -> float:
        """Blow the retained film back into the duct air; returns kg/s re-evaporated."""
        if self.coil_water_kg <= 0 or dt <= 0:
            return 0.0
        tau = max(1.0, float(self.spec.params.get("coil_evap_tau_s", COIL_EVAP_TAU_S)))
        kg = min(self.coil_water_kg, self.coil_water_kg / tau * dt)
        self.coil_water_kg -= kg
        return kg / dt

    def _coil_drain(self, dt: float) -> None:
        """A stopped or dry coil lets the film run into the condensate pan."""
        tau = max(1.0, float(self.spec.params.get("coil_drain_tau_s", COIL_DRAIN_TAU_S)))
        self.coil_water_kg *= math.exp(-dt / tau)

    # --- physics --------------------------------------------------------------------------

    def _wanted(self, reading: float, mode: str, low: float, high: float) -> str | None:
        h = float(self.spec.params.get("hysteresis_k", 0.5))
        want: str | None = None
        if mode in ("heat", "heat_cool", "auto"):
            if reading <= low - h or (self.running_mode == "heat" and reading < low + h):
                want = "heat"
        if want is None and mode in ("cool", "heat_cool", "auto"):
            if reading >= high + h or (self.running_mode == "cool" and reading > high - h):
                want = "cool"
        return want

    def _call_stage(self, want: str | None, mode: str, env: DeviceEnv) -> None:
        """Start/stop the burner or compressor stage.

        ``off`` and ``fan_only`` stop at once: the thermostat was told to stop
        conditioning, and ``fan_only`` means the air handler alone. Any other change of
        the wanted stage waits for ``min_run_s`` (compressor/heat-exchanger protection).
        """
        if self.running_mode == want:
            return
        ran = env.now - self.on_since
        commanded_stop = mode in ("off", "fan_only")
        if self.running_mode is None or commanded_stop or ran >= float(self.spec.params.get("min_run_s", 300)):
            self.running_mode = want
            self.on_since = env.now

    def _furnace_stage(self, dt: float, env: DeviceEnv) -> tuple[float, float, bool]:
        """Heat and electric power of the furnace/burner stage, plus whether it defrosts."""
        p = self.spec.params
        cap = float(p.get("heat_capacity_w", 8000))
        if self.furnace_heating:
            # Constant output: a furnace does not lose capacity to outdoor temperature and
            # has no outdoor coil, so no defrost cycle either.
            return cap, cap / max(0.1, float(p.get("furnace_efficiency", 0.95))), False
        defrosting = advance_frost(self, dt, env, True)
        if defrosting:
            return 0.0, 0.1 * cap, True
        a, b = p.get("cop_heat", [2.8, 0.05])
        return cap, cap / max(1.5, a + b * env.outdoor.temp_c), False

    def _action(self, mode: str, defrosting: bool, circulating: bool) -> str:
        p = self.spec.params
        if mode == "off":
            return "off"
        if self.running_mode == "heat":
            return str(p.get("defrost_action", "heating")) if defrosting else "heating"
        if self.running_mode == "cool":
            return "drying" if mode == "dry" else "cooling"
        return "fan" if circulating else "idle"

    def _physics(self, dt: float, env: DeviceEnv) -> dict[str, Gains]:
        p = self.spec.params
        z = env.zone
        reading = z.t_air
        self.sensor_c = reading
        mode = self.hvac_mode if self.available else "off"
        low = (
            self.to_c(self.target_low)
            if mode in ("heat_cool", "auto") and self.target_low is not None
            else self.target_c
        )
        high = (
            self.to_c(self.target_high)
            if mode in ("heat_cool", "auto") and self.target_high is not None
            else self.target_c
        )
        self._call_stage(self._wanted(reading, mode, low, high), mode, env)
        heat_w = 0.0
        electric = 0.0
        moisture = 0.0
        defrosting = False
        if self.running_mode == "heat":
            heat_w, electric, defrosting = self._furnace_stage(dt, env)
        elif self.running_mode == "cool":
            q = float(p.get("cool_capacity_w", 7000))
            shr = float(p.get("shr", 0.7))
            electric = q / float(p.get("eer", 3.2))
            heat_w = -q * shr
            moisture = -q * (1 - shr) / LATENT_HEAT
            self._condense(q * (1 - shr) / LATENT_HEAT * dt, dt)
        elif not self.furnace_heating:
            advance_frost(self, dt, env, False)  # the outdoor coil thaws while stopped
        self.coil_evap_kg_s = 0.0
        circulating = self.circulating()
        if circulating and self.coil_water_kg > 0:
            # Wet-coil blowback: the film evaporates into the supply air, taking its
            # latent heat out of the duct air and putting the water back into the rooms.
            self.coil_evap_kg_s = self._wet_coil_evap(dt)
            heat_w -= self.coil_evap_kg_s * LATENT_HEAT
            moisture += self.coil_evap_kg_s
        elif self.running_mode != "cool":
            self._coil_drain(dt)  # blower off or a dry heating cycle: the pan, not the air
        self.active = self.running_mode is not None
        self.hvac_action = self._action(mode, defrosting, circulating)
        blower_w = float(p.get("blower_w", 40.0)) if self.blower_on() else float(p.get("standby_w", 3.0))
        self._account(dt, heat_w, electric + blower_w)
        weights = self.supply_weights(env)
        if not weights:
            return {self.room: Gains(conv_w=heat_w, moisture_kg_s=moisture)}
        gains: dict[str, Gains] = {}
        for room, share in weights.items():
            g = gains.setdefault(room, Gains())
            g.conv_w += heat_w * share
            g.moisture_kg_s += moisture * share
        return gains

    def _model_state(self) -> dict[str, Any]:
        return {
            "running_mode": self.running_mode,
            "on_since": self.on_since,
            "frost": self.frost,
            "defrost_until": self.defrost_until,
            "coil_water_kg": self.coil_water_kg,
        }

    def _restore_model_state(self, data: dict[str, Any]) -> None:
        self.running_mode = data.get("running_mode")
        self.on_since = data.get("on_since", 0.0)
        self.frost = data.get("frost", 0.0)
        self.defrost_until = data.get("defrost_until", -1e12)
        self.coil_water_kg = data.get("coil_water_kg", 0.0)


@dataclass
class CentralHVAC(HeatPump):
    """North American central forced-air plant: furnace heat, heat-pump cool, blower.

    One thermostat for the whole house. Heating is a furnace (constant output, no
    defrost), cooling a heat pump with a wet evaporator coil, and the blower recirculates
    the served rooms whether it conditions or not. ``heat_pump`` already carries the
    mechanics; this class only supplies the forced-air defaults (fan modes, circulation
    and wet-coil constants) so a profile can stay short.
    """

    model = "central_hvac"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.spec.capabilities.get("fan_modes"):
            # Every forced-air thermostat can run its air handler on its own.
            self.fan_modes = ["auto", "on"]
            self.fan_mode = "auto"


@dataclass
class Heater(SimDevice):
    """Electric heater with body mass and on/off thermostat; "cool" is a fan only (Dyson)."""

    model = "heater"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.t_body: float | None = None
        self.element = False

    def _physics(self, dt: float, env: DeviceEnv) -> dict[str, Gains]:
        p = self.spec.params
        z = env.zone
        if self.t_body is None:
            self.t_body = z.t_air
        power = float(p.get("power_w", 1500))
        h_body = power / 40.0
        reading = z.t_air + (float(p.get("sensor_bias_k", 0.5)) if self.element else 0.0)
        self.sensor_c = round(reading, 1)
        hyst = float(p.get("hysteresis_k", 0.4))
        if self.hvac_mode != "heat" or not self.available:
            self.element = False
        elif reading <= self.target_c - hyst:
            self.element = True
        elif reading >= self.target_c + hyst:
            self.element = False
        q_el = power if self.element else 0.0
        q_out = h_body * (self.t_body - z.t_air)
        self.t_body += dt * (q_el - q_out) / (float(p.get("capacity_kj_k", 15)) * 1000)
        self.active = self.element
        fan = 30.0 if self.hvac_mode == "cool" else 0.0
        self.hvac_action = (
            "heating"
            if self.element
            else ("cooling" if self.hvac_mode == "cool" else ("off" if self.hvac_mode == "off" else "idle"))
        )
        self._account(dt, q_out, q_el + fan)
        return {self.room: _gains(q_out, float(p.get("radiant_fraction", 0.3)))}

    def _model_state(self) -> dict[str, Any]:
        return {"t_body": self.t_body, "element": self.element}

    def _restore_model_state(self, data: dict[str, Any]) -> None:
        self.t_body = data.get("t_body")
        self.element = data.get("element", False)


@dataclass
class Boiler(SimDevice):
    """Central heat source. Supplies hot water to the TRV/UFH devices it serves; the
    burner fires while switched on and someone draws heat."""

    model = "boiler"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.t_water: float | None = None
        self.load_w = 0.0  # last tick's draw from served devices (set by the world)

    def serves(self, entity_id: str) -> bool:
        serves = self.spec.params.get("serves", "all")
        return serves == "all" or entity_id in (serves or [])

    def supply_temp(self, room_temp: float) -> float:
        return self.t_water if self.t_water is not None else room_temp

    def _physics(self, dt: float, env: DeviceEnv) -> dict[str, Gains]:
        p = self.spec.params
        if self.t_water is None:
            self.t_water = env.zone.t_air
        on = self.hvac_mode == "heat" and self.available
        target = curve_supply_c(p, env.outdoor.temp_c, 60.0) if on else env.zone.t_air
        warmup = float(p.get("warmup_s", 120))
        self.t_water += (target - self.t_water) * min(1.0, dt / max(warmup, dt))
        self.sensor_c = round(self.t_water, 1)
        self.active = on and self.load_w > 50
        self.hvac_action = "heating" if self.active else ("off" if not on else "idle")
        self._account(dt, 0.0, self.load_w / float(p.get("efficiency", 0.92)) if self.active else 0.0)
        return {}

    def _model_state(self) -> dict[str, Any]:
        return {"t_water": self.t_water}

    def _restore_model_state(self, data: dict[str, Any]) -> None:
        self.t_water = data.get("t_water")


MODELS: dict[str, type[SimDevice]] = {
    "trv": Trv,
    "ufh": Ufh,
    "ac": SplitAc,
    "heat_pump": HeatPump,
    "central_hvac": CentralHVAC,
    "heater": Heater,
    "boiler": Boiler,
}


def create_device(spec, rng) -> SimDevice:  # noqa: ANN001
    return MODELS[spec.model](spec=spec, rng=rng)
