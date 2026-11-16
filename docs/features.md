# RoomMind features in detail

The [README](../README.md) gives the overview. This page explains what RoomMind does with the Central Plant, how each feature behaves and where its limits are. Setting up the thermostat, nudging, plant guards, occupancy modes and door factors is covered in the [Central Plant Guide](control-and-devices.md).

RoomMind controls exactly one thing: the **Central Thermostat** of a ducted forced-air system. Every room is a **Thermal Zone** that informs the decision; none of them has a thermostat of its own.

## Central Plant actuation

### The four plant actions

Every control cycle RoomMind picks one action for the whole house:

| Action | What happens at the hardware |
|---|---|
| `idle` | The thermostat is parked in the Fallback Band. Neither burner nor compressor may run; the blower is off (or on the idle fan mode you picked). |
| `heating` | The heating setpoint is nudged past the thermostat's reading until the furnace fires and the rooms that need it warm up. |
| `cooling` | The cooling setpoint is nudged below the thermostat's reading until the heat pump runs and dehumidifies. |
| `circulating` | Only the blower runs: furnace and compressor stay off while the air of all served rooms is mixed through the ducts. |

`sensor.roommind_central_action` shows what the plant is really doing (not just what was commanded), which rooms drove the decision and the reason behind it, plus any active guard. A room's own `hvac_action` tells you what the plant is doing *for that room*.

### Whole-home planning

The actions are chosen by a joint rollout over all Thermal Zones and the House Reference Temperature: the plan advances every room of the home at once, so it can weigh over-conditioning one occupied room against under-conditioning another, and it can recognize when Circulation equalizes two rooms' opposing errors for a fraction of the furnace's or compressor's energy. There is no per-room controller and no post-hoc device group — one ducted plant conditions every open register simultaneously.

### Cold start

While the per-room filters are still learning, RoomMind runs the plant with a priority-weighted on/off controller: it reacts to the occupied rooms' errors, keeps the Guardrail Band safe, and circulates when two occupied rooms disagree. As rooms become model-ready they join the predictive plan, usually within the first day.

## Bounded Setpoint Nudging

RoomMind never sends a room's target temperature to the thermostat. It sends a **Setpoint Nudge**: a temporary shift of the thermostat's active setpoint *just past the thermostat's own temperature reading*, bounded by the nudge delta and the maximum headroom beyond the active room targets.

- The nudge is relative to the thermostat's reading, so a call starts even when the room the thermostat is mounted in is already satisfied while a remote room is not.
- It is bounded, so the plant cannot be pushed to an extreme setpoint that would keep running if Home Assistant went offline mid-call.
- On a whole-degree thermostat (°F) the nudge is rounded to the device's step; RoomMind keeps the resulting command rate low instead of re-sending a value that snaps back.

### Fallback Band

Whenever the plant is idle or circulating, the thermostat is parked in a neutral **Fallback Band**: a heating floor and a cooling ceiling far enough apart to guarantee nothing runs, but close enough that the thermostat still protects the house on its own if Home Assistant or Wi-Fi is unreachable. On a thermostat that supports `heat_cool` the band is sent as a range; on a single-setpoint thermostat the idle setpoint is derived from the band. `hvac_mode: off` is never used for normal idle.

### Plant guards

Four guards keep the equipment and the house safe:

- **Minimum run** — a heating or cooling run lasts at least the configured minutes before it may stop.
- **Minimum off** — the plant rests after a conditioning run before the next one.
- **Minimum fan** — a Circulation run lasts at least the configured minutes, so the blower is not toggled for a single cycle.
- **Changeover lockout** — the heat pump rests before reversing between heating and cooling.

A guard that holds a run shows on the plant status, and the rooms report why they are idle.

## Rooms and the model

### Self-learning model per Thermal Zone

Each room with its own temperature sensor gets an Extended Kalman Filter. It learns the room's heat loss against outdoor air, its response to furnace heating and heat-pump cooling, its solar gain, internal gains from people, and its coupling to the **House Reference Temperature**. Because every room is modelled against one live house temperature, the filters stay decoupled and identifiable from room thermistors alone — no floorplan, no pairwise conductance matrix.

Rooms without their own sensor still get targets and shades, but no learned model; they cannot drive the plan with the same confidence, which is why a temperature sensor per room is the single most useful addition.

### Air Circulation mixing (`beta_fan`)

Circulation is not just "fan on": while the blower runs, every served room exchanges air with the rest of the house. RoomMind learns that coupling per room as `beta_fan` — the rate at which a room's temperature is pulled toward the House Reference Temperature while the blower runs — separately from the passive coupling `beta_f` that applies when the plant is off. A room that mixes quickly with the house is cheap to fix by circulating; a room behind a shut Interior Door mixes slowly, and its `beta_fan` reflects the throttled duct supply and return.

That is what makes the circulation decision quantitative: the plan compares the blower's small energy draw against the furnace or compressor energy that heating or cooling both rooms would cost.

### Pre-heating and pre-cooling

Before a schedule block raises a room's target, the plan starts heating early enough for the room to be warm when the block begins; rooms that need cooling pre-cool the same way when a cooler block cannot be reached by starting shortly before it. The lead is limited by the `Priority` slider (Comfort ↔ Efficiency) and can be switched off per room ("Pre-heat before schedule blocks"). A room that rarely heats takes longer to reach its block, and the plan learns that.

### Solar gain

RoomMind estimates solar irradiance from sun position and weather data, per window orientation and cover position, and learns each room's solar response. A sunny room is heated by the sun instead of the furnace, and a south-facing room may need shading or cooling hours before the thermostat's hallway does.

## Occupancy and Room Priority

### Room Priority

Comfort is weighted by **Room Priority**: an occupied room, an upcoming Occupancy Trend and an active override raise a room's weight; an empty room falls back to a small guardrail weight. The plan spends the plant's output where it buys the most comfort, and it will not run the furnace for a room nobody is in.

### Occupancy dwell

Motion sensors are noisy and people walk through rooms. RoomMind filters raw occupancy through a dwell score: a room counts as occupied after a short dwell and stays occupied through brief absences, and it only drops to eco after the room has been empty for the configured delay (default 15 minutes). Presence sensors, motion sensors and schedule blocks all feed the same score.

### Occupancy Trends

Next to live occupancy, RoomMind learns an **Occupancy Trend** per room: the recurring likelihood of the room being occupied by day type (weekday/weekend) and time of day. Trends let the plan anticipate comfort before someone arrives — the office is coming up to temperature when the working day starts, not twenty minutes later. Trends mature over days and are visible in the room's occupancy profile.

### Sleeping PIR: `schedule_assumed_when_on`

A PIR sensor cannot see someone sleeping. In the `schedule_assumed_when_on` occupancy mode, the room's schedule "on" blocks act as the floor of its occupancy: while the sleep block runs, the bedroom is treated as occupied even though the sensor reports no motion, so it keeps its sleep comfort target all night and only drops to eco when the block ends. This is the mode to use for bedrooms, nurseries and any room where stillness is normal.

### Guardrail Band

Every controlled room keeps a **Guardrail Band** (16..28 °C by default). An empty room is not comfort-controlled, but it never drifts to an extreme temperature either: if it leaves the band, the plan acts. That protects pipes, plants and furniture without paying for comfort in unused rooms.

### Presence, vacation and eco when empty

- **Presence away**: when every tracked person is away, rooms go to their eco targets (or are switched off, if that is the configured away action), and the plant is not run for comfort.
- **Vacation**: one fixed setback temperature for all rooms until an end date.
- **Eco when the room is empty**: with live occupancy in charge, an empty room drops to eco once the delay has passed and returns to its schedule the moment someone is detected.

### Manual override

A comfort, eco or custom-temperature override holds until the next schedule change, and the plan re-prioritizes the room while it lasts. An override lifts the outdoor temperature limits for the plant while it is active.

### Changes on the central thermostat

A setpoint turned in on the wall thermostat itself can be adopted as a manual override ("Follow thermostat setpoint", with its own duration). This is how a household member who walks up to the thermostat gets their way without fighting the automation — and how RoomMind avoids overwriting a change it cannot distinguish from its own nudge.

## Comfort and humidity

### Effective Temperature

Comfort is judged by **Effective Temperature**, which combines dry-bulb temperature and relative humidity. A humid 24 °C room feels warmer than a dry one, and the plan treats it that way: it may cool a little further in a humid room, and it will not chase a dry-bulb number that no one can feel. Rooms with a humidity sensor get the full computation; it is exposed as `effective_temperature` on the room climate entity and plotted on the analytics chart.

### Dehumidifying

When an occupied room is above the dehumidify threshold, the plan dehumidifies by cooling slightly past the target, bounded by the maximum overcool so it never turns comfort cooling into a cold house. The evaporator coil condenses the moisture out of the air; that is the latent half of the heat pump's work.

### Wet-Coil Lockout

After a cooling run the evaporator coil is wet. Running the blower then re-evaporates that water film into the supply air — the rooms get warmer *and* more humid, right after they felt dry. **Wet-Coil Lockout** holds Circulation for the configured time after a cooling run while the rooms are still humid, until the coil has drained (or the air is dry enough that evaporating the rest does no harm). `binary_sensor.roommind_wet_coil_lockout` shows the hold, and the plant status shows the guard reason.

Lockout does not block heating or cooling: a returning call always wins over a circulation hold.

## Openings

### Exterior Openings

A window or exterior door to the outdoors is an **Exterior Opening**. While one is open the plant pauses — heating a house with an open window wastes energy, and cooling one with an open door never reaches its target. The pause starts after the configured open delay and ends after the close delay, with a short grace period so a sensor that flickers does not stop the plant.

After a restart, RoomMind waits until the opening sensors report a state (at most 5 minutes; a sensor that is still unavailable then counts as closed). A sensor that no longer exists at all (deleted or renamed) counts as closed right away once Home Assistant has finished starting, with a warning in the log. A sensor that drops out briefly keeps its last state.

Three policies decide how much an open opening matters:

- **Pause unless door closed** (default) — an open opening pauses the whole home, unless the room's own Interior Door is confirmed closed. Then the opening is a local problem: the room loses its priority and the plant keeps serving the rest of the house.
- **Pause always** — any open opening pauses the whole plant, whatever the doors say.
- **Exclude room only** — the room leaves the plan, the plant keeps running for the others.

An active override always keeps the plant running.

### Interior Doors

An **Interior Door** is a door between two rooms. It matters in two ways:

- **As a sensor**, it tells RoomMind whether a room is really cut off from the rest of the home. Combined with the default exterior policy, a closed door turns an open window from a whole-home problem into a local one. A changed door state must hold for the door delay before it counts, so a door that is briefly opened for the cat does not flap the plant.
- **As physics**, a shut door reduces the room's air exchange with the house *and* its share of the duct air: the supply to a closed room is throttled, and its return path is throttled harder, so the blower's mixing pulls it toward the house more slowly (`beta_fan`). A room behind a shut door therefore needs its own comfort call more than an open one.

Interior doors are configured per room (sensor, delay, airflow factor), not per door pair.

## Shades

### Covers and read-only covers

RoomMind coordinates the shades of the whole home with the plant, because a shaded window is cooling the house for free. Per room you can pick the covers RoomMind may drive (`covers`), and **read-only covers** (`read_only_covers`) — manual blinds or numeric position sensors — that it only reads for shading and never moves.

What the automation does:

- **Sun protection**: if the sun would heat the room past its cooling target, the covers go down beforehand, only as far as needed, and only while the sun is actually on that facade when an orientation is set. The deploy threshold sets how far past the target the estimate has to be.
- **Night close**: covers close at sunset (or at a configured sun elevation and offset) and open again at sunrise, to a configurable night position.
- **Cover schedules**: a schedule helper can force positions, or allow sun protection only while its block is on.
- **Window hold**: an optional per-room hold keeps covers from going down while one of the room's openings is reported open; it catches up once it is closed. A cover already moving is not stopped.
- **Moved by hand**: a manual move pauses the automation for the configured time and the manual position is kept.
- **Priority**: moved by hand > schedule > night close > sun protection. If none applies, covers stay open.
- The **outdoor temperature gate** keeps covers open in the cold season, when the sun's warmth is welcome.

Cover decisions also use the plant's plan: if shading changes how much the plan expects to heat or cool, the covers react to that, not only to the raw solar estimate. A south-facing room can therefore be shaded *before* the compressor starts, instead of cooling the sun's heat with electricity.

## Schedules and targets

### Schedules

Create and edit schedules right in RoomMind: on a week grid on a computer, as a list of days on a phone. They stay normal Home Assistant schedule helpers, and the Schedules tab lists every helper with the rooms that use it. Every block can carry its own heating and cooling temperature; cover schedules carry a position. Multiple `schedule.*` entities per room with selector switching via `input_boolean` or `input_number`.

### Separate heat/cool targets

Independent comfort and eco temperatures for heating and cooling in auto mode, creating a natural dead-band. In a heat-only room the single comfort/eco value is the heating target; in a cool-only room it is the cooling target.

### Climate control per room

Disable climate control for individual rooms while the plant keeps serving the rest of the home. A room with climate control off still reports its readings and still learns, but takes no part in plant decisions.

### Outdoor areas

Mark rooms as outdoor (e.g. balcony, terrace) to take them out of the plan while keeping monitoring.

## Mold risk

Surface humidity estimation using the DIN 4108-2 method: RoomMind estimates the coldest wall spot of a room from indoor humidity, indoor temperature and outdoor temperature and reports a risk level with configurable notifications.

**Mold prevention** raises the room's heating target a little while the risk persists, which in a forced-air home means the plan calls the furnace for that room. Prevention only runs where the plant can actually act: a hallway without a register reports the risk but gets no prevention, because a raised target it cannot reach would only produce notifications.

Prevention has hysteresis — it holds until the surface humidity is clearly below the warning — so a prevention phase that ends right under the threshold cannot switch on and off every few minutes.

## Air quality

Per room, optional. Add a CO₂ and/or VOC sensor in the room's sensor settings. From the threshold on (CO₂: 1000 ppm by default, the level from which the German Environment Agency asks for more fresh air) the room page and its tile on the RoomMind dashboard show a hint and `binary_sensor.roommind_{area_id}_air_quality` turns on; it turns off again below 80 % of the threshold (800 ppm). VOC sensors report an index, ppb or µg/m³, so the VOC threshold is entered in the sensor's own unit; without one the reading is only shown.

Optionally pick a ventilation entity (`fan.*`, `switch.*` or `input_boolean.*`, e.g. an extractor fan or the boost of a ventilation unit): RoomMind switches it on when the air turns bad and off again once it is good, after at least 15 minutes. It only switches off what it switched on itself, leaves a fan alone that someone switched off during a bad spell, switches nothing on while climate control is off and switches its own fan off then. The central blower is never used for this — Circulation recirculates the air and does not lower the CO₂.

## Electricity price (TOU cooling shift)

Optional. Pick a price source (Tibber, Nordpool, EPEX Spot, aWATTar, ENTSO-e, Octopus, a template, or a switch that is on during PV surplus or off-peak) and let chosen rooms move their schedule target by up to 2 °C.

With a Central Plant the price shifts the **cooling** side by default, because central cooling runs on the electricity price while central heating may be gas: a room pre-cools while power is cheap before an expensive stretch and coasts a little warmer while it is expensive. The heating side is shifted only if you set `price_shift_modes` to `both` (e.g. an all-electric heat pump). Overrides, vacation and absence are never shifted, and a manual thermostat override is never price-shifted.

The analytics page shows the price and how long the target was shifted by it.

## Analytics

![Analytics](images/analytics.png)

Select a room and time range (24h / 7d / 30d / 90d / custom) to view temperature, **effective temperature** and humidity history, heating/cooling power, solar irradiance, the plant's action (including **circulating** periods), and model predictions. Export as CSV or diagnostics report.

- **This period**: time on target (within 0.5° of the wanted band, open windows left out), too cold/too warm shares, mean deviation, heating and cooling hours with the number of runs, circulation runs, and window-open hours.
- **Daily rhythm**: an overview of at least 14 days by hour, with the selected range marked. It stays on the latest days, grows with a longer range (up to 31 days) and moves back for an older one. One view shows when the plant heated, cooled or circulated and when windows were open, the other when the room missed its target. Hover or tap an hour for its figures, tap a day to show it in the chart above.
- **Heat loss** in the model card: how many degrees per hour the room loses without heating at 10° difference to outside, for comparing rooms or a renovation before and after. Degrees per hour, not watts, because RoomMind does not know the size of the room.
- **Model card**: what has been learned per room — heat loss, heating and cooling response, solar gain and the circulation coupling `beta_fan`, next to the plan's prediction accuracy.

Humidity needs a humidity sensor on the room. Rooms with a CO₂ sensor get a CO₂ plot below the temperature, from the first reading on.
