<div align="center">

# RoomMind

**Whole-home predictive Central HVAC control for Home Assistant — one thermostat, every room.**

[![Home Assistant Marketplace](https://img.shields.io/badge/Marketplace-Default-orange.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=snazzybean&repository=roommind&category=integration)
[![Home Assistant](https://img.shields.io/badge/Home%20Assistant-2026.5%2B-blue.svg)](https://www.home-assistant.io/)
[![GitHub Release](https://img.shields.io/github/v/release/snazzybean/roommind)](https://github.com/snazzybean/roommind/releases/latest)
[![Tests](https://github.com/snazzybean/roommind/actions/workflows/ci.yml/badge.svg)](https://github.com/snazzybean/roommind/actions/workflows/ci.yml)
![Coverage](https://raw.githubusercontent.com/snazzybean/roommind/python-coverage-comment-action-data/badge.svg)
[![License](https://img.shields.io/github/license/snazzybean/roommind)](https://github.com/snazzybean/roommind/blob/main/LICENSE)
[![Support on Ko-fi](https://img.shields.io/badge/Ko--fi-Support%20RoomMind-ff5e5b?logo=ko-fi&logoColor=white)](https://ko-fi.com/Y8Y31VP2VK)

<img src="docs/images/roommind-2.0-preview.gif" alt="RoomMind panel: room overview, room page, schedule editor and analytics" width="900">

</div>

RoomMind runs the **Central Plant** of a ducted forced-air home from Home Assistant: a gas or electric **furnace** for heating, an air-source **heat pump** for cooling, and the central blower for Circulation — all behind one Wi-Fi **Central Thermostat**. No TRVs, no minisplits: RoomMind learns each **Thermal Zone** with its own Extended Kalman Filter, weighs rooms by **Room Priority**, and decides block by block whether the plant should heat, cool, circulate or idle.

- 🎯 **The right room, on time.** Occupancy-driven **Room Priority** and learned **Occupancy Trends** warm the office before you sit down and leave the guest room on its **Guardrail Band**.
- 🌀 **Mix instead of over-condition.** When one room is too warm and another too cool, **Circulation** with the blower alone can equalize them for a fraction of the furnace's or compressor's energy.
- 🌡️ **Comfort, not just dry-bulb.** **Effective Temperature** accounts for humidity; **Wet-Coil Lockout** keeps a wet evaporator from blowing moisture back into the rooms.
- 🚪 **Openings that make sense.** An open window pauses the whole home; a closed **Interior Door** turns that into a local problem so the rest of the house keeps heating.
- 🧭 **Set up in minutes.** Pick the Central Thermostat, add rooms and sensors. Everything runs locally, no cloud.

## Screenshots

<table>
  <tr>
    <td width="50%"><img src="docs/images/dashboard.png" alt="Room overview with live state, plant action and quick override per room"></td>
    <td width="50%"><img src="docs/images/room.png" alt="Room page with target, override, schedule, sensors and shades"></td>
  </tr>
  <tr>
    <td align="center"><b>Rooms</b><br>Every Thermal Zone at a glance, with the plant's action and a quick override</td>
    <td align="center"><b>Room page</b><br>Target, schedule, sensors and shades in one place</td>
  </tr>
  <tr>
    <td><img src="docs/images/schedule.png" alt="Schedule editor with a week grid"></td>
    <td><img src="docs/images/analytics.png" alt="Analytics chart with temperature, target, outdoor temperature, humidity, price and effective temperature"></td>
  </tr>
  <tr>
    <td align="center"><b>Schedule editor</b><br>Draw blocks on a week grid</td>
    <td align="center"><b>Analytics</b><br>History, time on target, daily rhythm, circulation, learned model</td>
  </tr>
</table>

<p align="center"><img src="docs/images/mobile.png" alt="RoomMind on a phone: room overview and room page" width="520"></p>

## Features

<table>
  <tr>
    <td>🏠 <b>Central Plant</b></td>
    <td>One Central Thermostat drives furnace heating, heat-pump cooling and blower Circulation · bounded Setpoint Nudging just past the thermostat's own reading · Fallback Band left on the thermostat while idle · plant guards for minimum run, minimum off, minimum fan and changeover</td>
  </tr>
  <tr>
    <td>🧠 <b>Per-room models</b></td>
    <td>An Extended Kalman Filter per Thermal Zone against the House Reference Temperature: heat loss, furnace and heat-pump response, solar gain and learned Circulation mixing (<code>beta_fan</code>) · predictive whole-home MPC over <code>idle</code>, <code>heating</code>, <code>cooling</code> and <code>circulating</code> · on/off control while the models are still learning</td>
  </tr>
  <tr>
    <td>👥 <b>Occupancy &amp; priority</b></td>
    <td>Occupancy-Driven Room Priority from motion, presence and schedule · learned Occupancy Trends by day type and time of day · dwell filtering · the sleeping-PIR mode <code>schedule_assumed_when_on</code> · Guardrail Band for empty rooms</td>
  </tr>
  <tr>
    <td>💧 <b>Humidity comfort</b></td>
    <td>Effective Temperature instead of dry-bulb alone · dehumidifying by cooling slightly past the target · Wet-Coil Lockout so Circulation never re-evaporates the wet evaporator into the supply air</td>
  </tr>
  <tr>
    <td>🚪 <b>Openings &amp; shades</b></td>
    <td>Whole-home gate for Exterior Openings with a policy that respects a shut Interior Door · interior-door airflow factors · Whole-Home Coordinated Window Shades: solar shading, night close, cover schedules and read-only covers</td>
  </tr>
  <tr>
    <td>📊 <b>Insight</b></td>
    <td>Analytics page with effective temperature, circulation periods, comfort, price and the learned model · <a href="#dashboard-card">dashboard card</a> · <a href="#room-climate-entity">room climate entity</a> for automations, thermostat cards and voice assistants · English, German, French and Italian</td>
  </tr>
</table>

Every feature in detail: [docs/features.md](docs/features.md). Configuring the thermostat, nudging and plant guards: [Central Plant Guide](docs/control-and-devices.md).

## Installation

[![Open your Home Assistant instance and open RoomMind in the Marketplace.](https://my.home-assistant.io/redirect/hacs_repository/?owner=snazzybean&repository=roommind&category=integration)](https://my.home-assistant.io/redirect/hacs_repository/?owner=snazzybean&repository=roommind&category=integration)

1. Open the **Marketplace** in Home Assistant (formerly HACS), search for **RoomMind** and install it
2. Restart Home Assistant
3. Go to **Settings → Devices & Services → Add Integration → RoomMind**

<details>
<summary>Manual installation</summary>

1. Copy `custom_components/roommind/` to your `config/custom_components/` directory
2. Restart Home Assistant
3. Go to **Settings → Devices & Services → Add Integration → RoomMind**

</details>

## Getting Started

RoomMind appears as a panel in the sidebar (for administrators) and shows every Home Assistant area as a room.

1. **Pick the Central Thermostat**: `Settings → Central HVAC → Central thermostat`. It must be the `climate.*` entity that really runs the furnace, the heat pump and the blower. RoomMind nudges its setpoint and reads its `hvac_action` and `fan_mode`; it never needs a second thermostat.
2. **Set the thermostat's area** so RoomMind knows where the wall unit measures the House Reference Temperature.
3. **Add a temperature sensor per room** (recommended). A room with its own sensor is a full Thermal Zone: it gets the learned model, occupancy priority and its own share of the plant's air.
4. **Add a schedule per room**, or pick an existing schedule helper. Blocks decide when comfort applies.
5. **Set comfort and eco** for heating and cooling, and set the **Fallback Band** (heating floor / cooling ceiling) that stays on the thermostat while the plant is idle.

RoomMind starts controlling right away.

**The first hours.** Until the per-room filters know the rooms, RoomMind runs the plant in a safe on/off pattern and learns in the background. It needs about three hours of data without and one hour with heating or cooling per room, and hands rooms to the predictive plan one by one as their forecasts become reliable, usually on the first day. The model card on the analytics page shows how far each room is.

> [!TIP]
> **Which thermostat?** The one wired to the air handler — not a smart TRV and not a minisplit remote. If your furnace is controlled by a plain relay, create a [Generic Thermostat](https://www.home-assistant.io/integrations/generic_thermostat/) for it and pick that `climate` entity.

## Dashboard Card

Put a RoomMind room on any dashboard: Edit dashboard → Add card → **RoomMind room**. The card ships with the integration, no resource needs to be added. After installing or updating RoomMind, reload the browser tab (or restart the companion app) once so the card is found.

It shows the room's temperature, target, humidity and what the plant is doing with that room, plus:

- **Temporary override**: Comfort, Eco or a custom temperature for 1, 2 or 4 hours, with the time left. Tap the active one to end it.
- **Mode**: Auto, Heat only or Cool only, same as in the panel
- **Climate control switch**: turns RoomMind's control of the room on or off, same as the switch in the panel
- **Comfort and eco temperatures**: change the room's comfort and eco targets

Each part can be hidden in the card editor, e.g. a card for the family that only offers the override. **Compact layout** makes the card smaller: icon-only buttons side by side and a smaller status block; with every part hidden it is a plain status card. Hiding only changes what the card shows: the room climate entity still offers the same controls. The RoomMind panel and every other setting are for Home Assistant administrators only; other users can use the card but cannot open the panel. Administrators tap the room name to open the room in the panel. Outdoor areas show their readings only.

<details>
<summary>YAML</summary>

```yaml
type: custom:roommind-room-card
area: living_room           # area ID of the room
compact: false
hide_override: false
hide_mode: false
hide_climate_control: false
hide_temperatures: false
```

</details>

## How It Works

### One Central Plant, many Thermal Zones

A ducted forced-air system conditions every open register at once. RoomMind therefore plans the plant as a whole: each block it advances every Thermal Zone and the House Reference Temperature together over the four plant actions — `idle`, `heating`, `cooling` and `circulating` — and scores them by Room Priority weighted comfort, Guardrail Band safety, wrong-way over-conditioning and energy. That is how the plan can tell that heating one more degree for a warm room would push a cool one over its ceiling, or that the blower alone would fix both.

Rooms are never commanded individually. There is one lever — the Central Thermostat — and RoomMind moves it.

### Target Temperature Priority

```
Manual Override > Thermostat Override > Vacation > Presence Away > Schedule Block > Comfort / Eco  (+ Mold Delta)
```

"Eco when the room is empty" lowers the result to eco while the room is empty; override, vacation and presence away still take priority. The electricity price can move the **cooling** target by up to 2 °C (heating too if you opt in), never an override, vacation or absence.

### Setpoint Nudge and Fallback Band

RoomMind does not send a room's target temperature to the thermostat. It sends a bounded **Setpoint Nudge**: a temporary shift of the thermostat's active heating or cooling setpoint just past the thermostat's own reading, enough to start or sustain a call for the rooms further down the duct. When the plant is idle or only circulating, the thermostat is parked in a neutral **Fallback Band** (a `heat_cool` range where supported) so it keeps protecting the home on its own if Home Assistant or Wi-Fi goes offline.

Why not just send the target: the thermostat senses temperature in one hallway. Sending a remote room's target fails to start a call whenever the hallway is already satisfied, and extreme setpoint shifts or `hvac_mode: off` risk thermal runaway or freeze damage.

### Per-room models and the House Reference Temperature

Each Thermal Zone has its own Extended Kalman Filter. It learns the room's heat loss against outdoor air, its response to furnace heating and heat-pump cooling, its solar gain — and its coupling to the **House Reference Temperature**, both passive (`beta_f`) and while the blower mixes air through the ducts (`beta_fan`). Coupling every room to one live house temperature keeps estimation decoupled and identifiable from room thermistors alone.

### Occupancy-Driven Room Priority and Occupancy Trends

Motion and presence sensors say who is where right now; a learned **Occupancy Trend** says where people usually are at this hour on this kind of day. Together they set a room's Room Priority weight: an occupied or soon-to-be-occupied room drives comfort cycles, an empty room only carries its Guardrail Band. Dwell filtering bridges short absences, and the sleeping-PIR mode `schedule_assumed_when_on` lets a bedroom keep its sleep target while the PIR sees no motion.

### Effective Temperature and Wet-Coil Lockout

Two rooms at 24 °C do not feel the same at 40 % and 65 % relative humidity. RoomMind judges comfort by **Effective Temperature**, and when the air is too humid it dehumidifies by cooling slightly past the target (bounded by the maximum overcool). Right after a cooling run the evaporator coil is wet; running the blower then would re-evaporate that film into the supply air and raise the humidity again. **Wet-Coil Lockout** holds Circulation until the coil has drained or the rooms are dry enough.

### Exterior Openings and Interior Doors

An **Exterior Opening** is a window or door to the outdoors: while it is open the plant pauses, because the envelope is no longer a closed system. An **Interior Door** is a door between two rooms: RoomMind uses it twice. As a sensor it can tell the plant that an open window is a local problem — with the default policy `pause_unless_door_closed`, a room whose own Interior Door is confirmed closed no longer pauses the whole home. And as physics, a shut door throttles the room's duct supply and its return path, so a closed room receives less of the blower's air.

### Whole-Home Coordinated Window Shades

Shades are decided for the whole home, not per device: RoomMind lowers covers before the sun would push a room past its cooling target, closes them at night, leaves them up while a window is open (if configured) and pauses automatic control when someone moves one by hand. Read-only covers (manual blinds, numeric position sensors) are read for shading and never driven.

### Predictive Control vs. Simple

`Settings → Control → Control Mode` decides whether the whole-home plan is used at all. **Predictive (MPC)** is the default and uses the learned room models. **Simple (on/off)** keeps the plant on plain hysteresis; the models still learn and feed the analytics prediction, and covers fall back to a simpler solar estimate.

## Entities

| Entity | Description |
|--------|-------------|
| `climate.roommind_{area_id}` | The room as one thermostat, see [Room climate entity](#room-climate-entity) |
| `sensor.roommind_central_action` | What the plant is doing: `idle`, `heating`, `cooling` or `circulating`, with the driving rooms, decision reason and active guards as attributes |
| `sensor.roommind_house_temperature` | The House Reference Temperature the plan works against |
| `binary_sensor.roommind_wet_coil_lockout` | On while Wet-Coil Lockout holds Circulation after a cooling run |
| `switch.roommind_vacation` | Global vacation mode toggle |
| `switch.roommind_climate_control` | Global climate control toggle, the same as `Settings → General`. Off stops RoomMind controlling the plant and leaves the thermostat as it is, e.g. for a summer/winter automation |
| `switch.roommind_{area_id}_cover_auto` | Per-room automatic cover control toggle |
| `binary_sensor.roommind_{area_id}_cover_paused` | On when manual cover override is detected |
| `binary_sensor.roommind_{area_id}_air_quality` | On while the room's CO₂ or VOC is past its threshold (rooms with an air quality sensor); attributes `co2`, `voc`, `reason`, `ventilation_active` |

### Room climate entity

`climate.roommind_{area_id}` shows and controls a room the way a thermostat card, an automation or a voice assistant expects. Changing its temperature sets an override until the next schedule change, like a thermostat's temporary hold.

<details>
<summary>What each climate feature does in RoomMind</summary>

| Climate feature | In RoomMind |
|---|---|
| Target temperature (range in Auto mode) | The targets currently in force for that room. Changing it sets an override that holds until the next change of the active schedule, like a thermostat's temporary hold. Without a schedule it holds until you pick the preset `none`; if the schedule helper cannot be read at that moment (unavailable, deleted), it holds for 4 hours. Sending the value already in force changes nothing. A single temperature in Auto mode (e.g. from a voice assistant) sets the heating side and keeps the cooling side at least at comfort cooling. |
| Presets Comfort / Eco | Comfort or eco override, also until the next schedule change |
| Preset Manual | Shown while a custom temperature is held (set here, in the panel or on the old override entity); picking it holds the targets in force until the next schedule change |
| Preset Schedule (`none`) | Ends the override and returns to the schedule |
| HVAC mode `heat` / `cool` / `heat_cool` | Climate mode Heat only / Cool only / Auto for the room, and turns control on. `climate.turn_on` turns control on and keeps the climate mode. A room in Heat only takes part in heating plans only; the plant itself stays free to cool the other rooms. |
| HVAC mode `off` | Stops RoomMind controlling the room; it takes no part in plant decisions until it is turned back on. It does not switch the shared furnace off for the rest of the home. |
| `hvac_action` | What the plant is doing with the room: `heating`, `cooling`, `circulating`, `idle` or `off` |
| Current temperature and humidity | The room's readings. The attribute `effective_temperature` is what the room feels like, i.e. the dry-bulb reading adjusted for humidity. |

Attributes: `override_until`, `window_open`, `presence_away`, `occupancy_vacant`, `vacation`, `price_level`, `effective_temperature`, `priority_weight`, `occupancy_probability`, `interior_door_open`, `exterior_window_open`. An open window shows as `idle` with `window_open: true`.

The entity is placed in the room's area once when it is created, so it shows up on area cards and Assist finds it by the area ("set the temperature in the living room to 22"). An area you change or remove later is kept. For voice control, hide the room's own sensors from Assist where that matters.

Like every override, one set here also lifts the outdoor temperature limits for heating and cooling while it lasts.

A wall display or a thermostat RoomMind does not control can be linked with an automation: its setpoint → `climate.set_temperature` on the room entity, and the room entity's target → the display, so the display always shows what applies.

</details>

<details>
<summary>Deprecated entities</summary>

These per-room entities still work but will be removed in a future release. Rooms added since 2.0 get them disabled; rooms set up before keep them as they are. If automations or scripts reference them directly, Home Assistant shows a repair notice listing them; uses inside templates or dashboards are not detected, but every service call to an old entity logs a warning once.

| Old entity | Use instead |
|---|---|
| `sensor.roommind_{area_id}_target_temp` | Target temperature of `climate.roommind_{area_id}` |
| `sensor.roommind_{area_id}_mode` | `hvac_action` of `climate.roommind_{area_id}` |
| `climate.roommind_{area_id}_override` | `climate.set_temperature` or a preset on `climate.roommind_{area_id}`. Unlike the old entity, an override there ends at the next schedule change. |

`switch.roommind_{area_id}_climate_control` is not deprecated and stays. Its off only stops RoomMind controlling the room; the plant keeps running for the rest of the home.

</details>

## Troubleshooting

<details>
<summary><b>A room still uses on/off control after a few days</b></summary>

The model needs about three hours of data without and one hour with heating or cooling, and forecasts better than 0.5 °C. Rooms that rarely see a plant cycle, or that are disturbed often (open windows, a fireplace), take longer. The model card on the analytics page shows what is still missing.

</details>

<details>
<summary><b>The plant does not heat or cool when expected</b></summary>

Check `Settings → Central HVAC` first: the Central Thermostat entity, the Heating floor and Cooling ceiling of the Fallback Band, and the plant guards. Then check the outdoor limits in `Settings → Control` — by default no cooling below 16 °C and no heating above 22 °C outside; an active manual override ignores these limits. A plant that reports `circulating` instead of heating is usually equalizing two rooms on purpose; `sensor.roommind_central_action` shows the decision reason and the driving rooms.

</details>

<details>
<summary><b>The house feels damp right after cooling stops</b></summary>

That is exactly what Wet-Coil Lockout prevents. Check `binary_sensor.roommind_wet_coil_lockout` and `Settings → Central HVAC → Humidity comfort`: a wet-coil lockout time that is too short (or a Wet-coil max RH that is too high) lets the blower start while the evaporator is still wet. In a very humid climate, lower the dehumidify threshold and allow a little more overcool instead of longer circulation.

</details>

<details>
<summary><b>One room is always too cold (or too warm)</b></summary>

A ducted system cannot serve every room equally. Give the room its own temperature sensor so it becomes a real Thermal Zone, check that its Interior Door is reported (a shut door throttles its supply and return), and give the thermostat's own area a sensor too — the House Reference Temperature is only as good as the readings behind it. Rooms far from the thermostat benefit most from Circulation.

</details>

<details>
<summary><b>A setpoint changed on the wall thermostat was overwritten</b></summary>

Turn on `Follow thermostat setpoint` in `Settings → Central HVAC` with an override duration that suits the household. RoomMind then adopts a setpoint changed on the Central Thermostat itself as a manual override and leaves the plant alone for that long. Like every override, it lifts the outdoor temperature limits while it lasts.

</details>

<details>
<summary><b>The model seems wrong after changes to the room</b></summary>

If you changed insulation, windows, or moved sensors, use "Re-learn faster" in `Settings → Model training`, or reset the model in `Settings → Reset model`.

</details>

<details>
<summary><b>The panel does not update after an update</b></summary>

Hard-refresh the browser: **Cmd+Shift+R** (Mac) or **Ctrl+Shift+R** (Windows/Linux).

</details>

## Requirements

- **Home Assistant** 2026.5 or newer
- A ducted Central Plant behind **one** `climate.*` Central Thermostat (furnace, heat pump, or both) that reports `hvac_action`, and ideally `fan_mode`
- At least one Home Assistant area with a temperature sensor
- Optional: humidity, window, door, occupancy and CO₂/VOC sensors, weather entity, electricity price sensor, person entities, covers

No cloud services, no account: everything runs locally.

## Feedback & Contributing

- 🐛 **Bug reports**: [Open an issue](https://github.com/snazzybean/roommind/issues/new?template=bug_report.yml)
- 💡 **Feature requests & ideas**: [Post in Discussions](https://github.com/snazzybean/roommind/discussions/categories/feature-requests) so the community can vote and discuss
- 🙋 **Questions & usage help**: [Ask in Q&A](https://github.com/snazzybean/roommind/discussions/categories/q-a)
- 📣 **Announcements**: [Follow Announcements](https://github.com/snazzybean/roommind/discussions/categories/announcements)

If RoomMind keeps your home comfortable, you can support its development:

[![ko-fi](https://img.shields.io/badge/Ko--fi-Support%20RoomMind-ff5e5b?logo=ko-fi&logoColor=white)](https://ko-fi.com/Y8Y31VP2VK)
