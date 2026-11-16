# Central Plant Guide

This page explains how to set up and tune RoomMind's Central HVAC control: the Central Thermostat, the Setpoint Nudge, the Fallback Band, the plant guards, and the room-level options that decide where the plant's air goes.

Everything here lives in two places: `Settings → Central HVAC` for the plant, and the room page (`Sensors`, `Schedule & temperatures`, `Covers & blinds`) for the Thermal Zones.

## 1. The Central Thermostat

RoomMind drives a ducted forced-air system through **one** `climate.*` entity — the wall thermostat wired to the air handler. Set it in `Settings → Central HVAC → Central thermostat`.

What RoomMind needs from that entity:

| Attribute | Why |
|---|---|
| `hvac_modes` with `heat`, `cool` and preferably `heat_cool` | Heating, cooling, and a neutral Fallback Band as a range |
| `hvac_action` | To know whether the furnace or compressor is really running (not just commanded) |
| `fan_modes` with `auto` and `on` | `on` is used for Circulation, `auto` for idle |
| `current_temperature` | The reference the nudge is relative to |
| `min_temp` / `max_temp` | RoomMind never sends a setpoint outside them |
| `target_temp_step` | Whole-degree or half-degree setpoints are respected |

RoomMind only ever calls `set_hvac_mode`, `set_temperature` (single value or `target_temp_low`/`target_temp_high`) and `set_fan_mode`. It does not call vendor-specific services and it does not need the thermostat's own schedule.

Set **Thermostat area** as well: the area the thermostat is mounted in tells RoomMind where the wall unit measures the House Reference Temperature. If the thermostat sits in a hallway without a sensor, give the hallway one — the whole plan is built on that reading.

> The Central Thermostat must be the entity that runs the equipment. A smart TRV, a minisplit remote or an IR blaster cannot serve as the Central Thermostat.

## 2. Range Setpoint (`heat_cool`) or Single Setpoint

**Thermostats that support `heat_cool` with a range** get the Fallback Band as a range (`target_temp_low` / `target_temp_high`); while conditioning, the nudge moves the active side of that range past the thermostat's reading. This is the cleanest setup: idle is genuinely neutral, and the thermostat's own deadband protects the house between calls.

**Single-setpoint thermostats** (heat or cool only, or a `heat_cool` mode without range support) get one idle setpoint derived from the Fallback Band and the room targets. Keep the Minimum deadband setting generous enough that the idle setpoint cannot sit within the thermostat's own hysteresis of the room temperature, or the unit will cycle on its own.

### °F and whole-degree thermostats

RoomMind works in °C internally and converts to the thermostat's unit when it sends a setpoint, so `min_temp` / `max_temp` are always evaluated in the device's own unit. On a **°F thermostat with whole-degree steps** (typical for Ecobee-style units) the nudge is rounded to the device's step: a 1.2 °C nudge becomes a 2 °F shift, which is still enough to start a call.

Whole-degree snapping can produce a value that is identical to the one already set. RoomMind tracks what it sent, so an unchanged snapped setpoint is not re-sent every cycle — that is what keeps the command rate on such a thermostat low. If you see frequent commands, raise the Minimum deadband and lengthen the Minimum run; a very small nudge delta makes calls marginal and can make the unit start and stop more often.

## 3. The Fallback Band

The **Fallback Band** is what is left on the thermostat whenever the plant is idle or circulating: a **Heating floor** and a **Cooling ceiling**. RoomMind does not park the thermostat in `hvac_mode: off` for idle, so the house stays protected if Home Assistant, Wi-Fi or the integration is unreachable. Only an explicit off does that: the presence-away action "Turn off devices", or `hvac_mode: off` on the room climate entity when that room is the only one the plant serves.

- **Heating floor** — low enough that idle never heats (a few degrees under your coldest eco target), high enough to keep pipes and plants safe. The default is 16.5 °C.
- **Cooling ceiling** — high enough that idle never cools, low enough that a hot house is not left to bake. The default is 26.5 °C.
- **Minimum deadband** — the smallest gap kept between the two. With a range thermostat this is the band the unit will never cross on its own; with a single-setpoint thermostat it keeps the idle setpoint from hunting.

Rules of thumb:

- The floor must be below every room's eco heating target, otherwise the Fallback Band would heat rooms you set to eco.
- The ceiling must be above every room's eco cooling target for the same reason.
- On a heat-pump house, leave the band wide: a narrow band turns idle into unintended conditioning.

## 4. Setpoint Nudge

RoomMind starts and sustains a call by nudging the thermostat's active setpoint **just past its own reading**:

- **Nudge delta** — how far past the reading the setpoint goes to start a call (default 1.2 °C; 1.0–1.2 on a responsive heat pump, up to 2.0 on a sluggish one). Too small: the thermostat ignores the call or stops it immediately. Too large: the room overshoots before the plan can end the run.
- **Max headroom** (the safety headroom) — the largest shift away from the Fallback Band, i.e. how far beyond the active room targets the nudge may go. This is what prevents a stuck call from running the furnace flat out. The default is 2.5 °C; keep it modest and never so large that a lost connection leaves an extreme setpoint behind.
- **Minimum deadband** — also applies to the nudged setpoints: heating and cooling setpoints never come closer than this.

You should not need to touch the nudge once it works. If a room is consistently a little late, give it its own temperature sensor and let the plan pre-heat earlier rather than raising the nudge — the nudge is a switch, not a thermostat.

## 5. Plant Guards

`Settings → Central HVAC → Plant guards` protects the equipment:

| Setting | What it does | Default | Typical |
|---|---|---|---|
| **Minimum run** | Minutes a heating or cooling run must last before it may stop | 10 min | 10–15 min (furnace), 10 min (heat pump) |
| **Minimum off** | Minutes the plant rests after a conditioning run | 5 min | 5 min |
| **Minimum fan** | Minutes a Circulation run must last | 10 min | 10–30 min |
| **Changeover lockout** | Minutes the heat pump rests before reversing between heating and cooling | 30 min | 20–30 min |

Set the minimum run at or above the thermostat's own minimum run time if it has one — otherwise the plan can finish a run and the thermostat can keep it going, or vice versa. Minimum fan matters most: a blower that starts and stops every few minutes is audible and buys nothing.

While a guard holds a run, the plant status shows it and the affected rooms report why they are idle. Guards never override a safety decision: an open Exterior Opening or an explicit off still stops the plant.

## 6. Wall-Thermostat Override Adoption

`Follow thermostat setpoint` decides what happens when someone turns the setpoint in on the wall thermostat itself:

- **On** — the new setpoint becomes a manual override and is respected for **Override duration** hours (default 2 h). While it lasts, the plant serves it like any other override and no Setpoint Nudge moves it.
- **Off** — RoomMind treats the thermostat's setpoint as its own and continues nudging; a hand-set value is overwritten with the next cycle.

Turn it on for households where people walk up to the thermostat. Turn it off for a thermostat that runs its own program, is written by other automations, or is shared with a landlord's system — RoomMind cannot tell those changes from a hand on the dial.

A manual thermostat change is never price-shifted and outranks vacation, presence away and the schedule, like every override.

## 7. Exterior Openings and the Open Window Policy

An **Exterior Opening** is a window or exterior door to the outdoors. Configure the sensors per room (`Sensors → Windows and exterior doors`), with open and close delays.

`Settings → Central HVAC → Open window policy` decides how much an open opening matters:

| Policy | Behaviour | When to use |
|---|---|---|
| **Pause unless door closed** (default) | An open opening pauses the whole plant, *unless* the room's own Interior Door is confirmed closed — then only that room leaves the plan | Homes where a room can be closed off (office, guest room) |
| **Pause always** | Any open opening pauses the whole plant | Small homes without interior door sensors, or where every room is open to the hallway |
| **Exclude room only** | The room leaves the plan, the plant keeps running for the others | Homes where windows are opened for a few minutes while the rest of the house is occupied |

An active override always keeps the plant running. After a restart RoomMind waits up to 5 minutes for the sensors to report; a sensor that no longer exists counts as closed with a warning in the log.

## 8. Occupancy Modes

Per room (`Sensors → Occupancy mode`), the mode decides how occupancy drives Room Priority:

| Mode | Behaviour | Use for |
|---|---|---|
| **Reactive + trend** | Uses the motion sensors and the learned Occupancy Trend. An empty room drops to eco after the delay and returns as soon as someone is detected | Living rooms, offices, kitchens |
| **Assumed while schedule on** | Treats the room as occupied while its schedule is "on", whatever the sensors say; motion only refines it | Bedrooms and nurseries with a PIR that cannot see someone sleeping |
| **Follow schedule** | Occupancy follows the schedule only; motion sensors are ignored | Rooms with a reliable schedule and no useful sensor |
| **Guardrail only** | Never assumes occupancy; the room is only kept inside its Guardrail Band | Guest rooms, storage rooms, spare bedrooms |

Two more room settings tune the behaviour:

- **Dwell** (`occupancy_dwell_minutes`) — how long motion must persist before the room counts as occupied, and how long a short absence is bridged. The default (3 minutes) is right for most rooms; a hallway or a kitchen benefits from a little more.
- **Eco delay** (`occupancy_eco_delay`) — how long the room must be empty before it drops to eco. 10–20 minutes avoids reacting to every trip to the kitchen.

The Guardrail Band itself (16..28 °C by default) is what protects empty rooms: they never drive comfort cycles, but they never drift to extremes either.

## 9. Interior Doors: Sensor, Delay and Airflow

An **Interior Door** is a door between two rooms (`Sensors → Interior door`): the binary sensor that reports it and the delay a changed state must hold before it counts (default 60 s; 30–120 s keeps a door that is briefly opened from flapping the plant).

Two things change when a room's Interior Door is reported closed:

- **Policy** — with the default `pause unless door closed`, the room's open window no longer pauses the whole home; the room leaves the plan and the rest of the house keeps heating.
- **Airflow** — the room's share of the blower's air is throttled on the supply side, and its return path is throttled harder, because a closed door blocks the return-plenum loop. RoomMind learns the resulting slower circulation coupling (`beta_fan`) per room, and the `door_closed_mix_factor` room setting (default 0.40) says how much of the house's air a shut room still sees.

Practical guidance:

- Mount the door sensor where it reports the *door*, not the frame: a door that is ajar must read open.
- Give a closed-off room its own heating target that it can reach with the reduced airflow — or leave its door open and let Circulation do the work.
- Do not use an Interior Door sensor as an Exterior Opening sensor. They answer different questions: "is this room part of the house?" versus "is the envelope open?".

## 10. Rooms, Sensors and Shades

A room is a **Thermal Zone** as soon as it has its own temperature sensor. Add:

- **Temperature sensor** (recommended) — unlocks the learned model and predictive control for that room. Without it the room gets targets but no model.
- **Humidity sensor** — enables Effective Temperature, the dehumidify decision and the mold risk.
- **Occupancy sensor(s)** — motion or presence, see the modes above.
- **CO₂ / VOC sensor** — see [Air quality](features.md#air-quality).
- **Covers** — driven by RoomMind (sun protection, night close, schedules, window hold), and **read-only covers** — manual blinds or numeric position sensors — that are only read for shading and never moved.

## 11. Control Mode and Priority

- `Settings → Control → Control mode` — **Predictive (MPC)** plans the plant with the learned room models. **Simple (on/off)** keeps the plant on hysteresis only; the models keep learning for the analytics, and covers use a simpler solar estimate. Use Simple only while diagnosing a problem.
- `Settings → Control → Priority` — the Comfort ↔ Efficiency slider. Toward Comfort the plan reacts earlier and works harder to hold the band; toward Efficiency it allows more drift and saves runtime. This does not change schedule targets, overrides, comfort or eco values — only how aggressively the plan chases them.

## 12. Troubleshooting the Plant

| Symptom | Look at |
|---|---|
| Plant stays idle although a room is cold | Is the Heating floor below the room's target (and the Cooling ceiling above it)? Is the thermostat entity correct and reporting `hvac_action`? Is the outdoor heating limit (`Settings → Control`) not exceeded? |
| Plant runs, rooms do not get warm | Airflow: is the room served by the ducts and is its Interior Door reported open? Is its `beta_fan` small because it is throttled? |
| Short cycling on the furnace or compressor | Raise Minimum run and Minimum off; raise Minimum deadband; check the thermostat's own minimum run time |
| Room overshoots after a call | Lower Nudge delta, lower Max headroom, or give the room a sensor so pre-heating is planned |
| VOC/CO₂ remains high with the blower running | Circulation recirculates air; it does not ventilate. Use a dedicated ventilation entity |
| House feels damp right after cooling | Wet-Coil Lockout too short or Wet-coil max RH too high; see `binary_sensor.roommind_wet_coil_lockout` |
| Plant pauses whenever a window opens somewhere | Switch the policy to `pause unless door closed` and configure the Interior Door sensors, or use `exclude room only` |
| A setpoint changed at the wall is overwritten | Turn on `Follow thermostat setpoint` and set the override duration |

## 13. The House Layer

The house layer is one comfort pair, one eco pair, an optional schedule bound to a `schedule.*` helper, a climate mode, preheat and a price shift, held in global settings under `house`. Following rooms inherit all of it; rooms with their own settings keep them.

The house climate entity is `climate.roommind_house`. Its modes are not the same as the room entity's:

- **`off`** pauses all control (`climate_control_active`). The house layer stays configured, its schedule keeps running in Home Assistant, and nothing is served until the mode is turned back on. This is a pause, not a switch-off of the equipment, and it is not the same as the house layer being disabled.
- **`heat` / `cool` / `heat_cool`** set the house climate mode that following rooms inherit. A room in Heat only takes part in heating plans only, as with the room entity.

Writing to the entity for the first time enables the house layer on an install that was upgraded from an earlier version; an upgrade starts with the house layer disabled (`enabled=false`) so nothing changes until a house target is set. A fresh install starts enabled at 21/24 °C with 17/27 °C eco.

A target or a preset change on the entity is a temporary hold, the **House Override**: it holds until the next house schedule change, and the preset `none` ("Resume schedule") ends it. A hold outranks the house schedule block but not a room schedule block with temperatures, a presence-away state, a vacation or an override higher up the ladder.

Precedence for a following room, top to bottom:

```
Room Manual Override > Central Thermostat Override > Vacation > Presence Away > Room schedule block with temperatures > House Override > House schedule block > House Target > the room's comfort/eco band
```

An empty following room runs on the house eco pair, not the house comfort pair. When no room has a setting of its own, occupancy decides which room gets the house target, and empty rooms stay on their eco/guardrail band.
