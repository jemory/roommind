# RoomMind Central HVAC

Whole-home comfort optimization for a single ducted forced-air HVAC system using room-by-room sensing, occupancy-driven priority, and predictive thermal modeling.

## Language

### Plant & Actuation

**Central Plant**:
The home's single shared forced-air conditioning system—a furnace for heating, a heat pump for cooling, and a central blower for air circulation—operated through one thermostat.
_Avoid_: Compressor group, master device, boiler, zone equipment

**Central Thermostat**:
The wall-mounted Wi-Fi climate controller through which Home Assistant observes equipment state and issues heating, cooling, and circulation calls.
_Avoid_: Room thermostat, TRV, head unit

**Circulation**:
Running the central blower without firing the furnace or heat pump to mix air across rooms through the supply and return ducts.
_Avoid_: Ventilation, coil dry, fan-only idle

**Setpoint Nudge**:
A bounded temporary shift of the Central Thermostat's setpoint just past its own temperature reading to trigger or sustain an active heating or cooling call for remote rooms.
_Avoid_: Boost setpoint, proportional valve opening

**Fallback Band**:
The safe heating floor and cooling ceiling left on the Central Thermostat whenever the Central Plant is idle or circulating, ensuring the thermostat protects the home on its own if Home Assistant is unreachable.
_Avoid_: Idle off, setback mode

**Wet-Coil Lockout**:
A temporary hold preventing Circulation shortly after a cooling run when evaporating moisture off the wet indoor cooling coil would raise indoor humidity to uncomfortable levels.
_Avoid_: Evaporator anti-odor dry, coil dry run

### Rooms, Coupling & Comfort

**Thermal Zone**:
An indoor monitored room (Home Assistant Area) whose sensors, shades, openings, schedules, and occupancy inform Central Plant decisions.
_Avoid_: Device group, managed room

**House Reference Temperature**:
The representative indoor air temperature of the conditioned home as a whole, against which individual rooms exchange heat passively and during Circulation.
_Avoid_: Return plenum temperature, master temperature

> The **House Reference Temperature** is a measured quantity, the mean of the conditioned rooms. It is not a setting and it is not the **House Target**, which is a household comfort setting chosen by the occupant. The plan works against the first and follows the second.

**House Target**:
The household comfort setting for the whole home, a heating and a cooling target that every following room inherits unless it carries a setting of its own.
_Avoid_: Whole-home setpoint, master thermostat, house temperature

**House Schedule**:
A home assistant schedule helper, or a set of them selected by a selector entity, whose blocks move the House Target through the day for following rooms.
_Avoid_: House program, master schedule, thermostat timetable

**House Eco**:
The household eco pair, the heating and cooling targets a following room runs on while it is empty, standing in place of the House Target for that room.
_Avoid_: House setback, whole-home away mode, household eco mode

**Room Target**:
The comfort heating and cooling targets in force for one Thermal Zone, inherited from the House Target when the room follows the house and taken from the room's own settings otherwise.
_Avoid_: Zone setpoint, per-room reference, local house target

**Follow House**:
The per-room setting that binds a Thermal Zone to the House Target, the House Eco pair, the house climate mode, the House Schedule, preheat and price shift.
_Avoid_: House lock, whole-home sync, master follow

**Independent Room**:
A Thermal Zone set to keep its own comfort and eco targets, its own schedule blocks and its own gaps, with no inheritance from the house layer.
_Avoid_: Detached zone, orphan room, standalone room

**Exterior Opening**:
A window or exterior door connecting a Thermal Zone to outdoor air, causing rapid envelope loss when open.
_Avoid_: Contact sensor, door/window sensor

**Interior Door**:
A door separating a Thermal Zone from adjacent indoor spaces and central return pathways, reducing room-to-house air exchange and duct airflow when closed.
_Avoid_: Window sensor, damper

**Effective Temperature**:
How warm or cool a room feels after accounting for both dry-bulb air temperature and relative humidity.
_Avoid_: Raw temperature, heat index

**Room Priority**:
The dynamic weight given to a Thermal Zone's comfort in whole-home decisions, driven by current occupancy, upcoming Occupancy Trends, and active overrides.
_Avoid_: Comfort weight

**Occupancy Trend**:
A room's learned recurring likelihood of being occupied by day type and time of day, used to anticipate comfort needs before people arrive.
_Avoid_: Presence schedule, motion state

**Guardrail Band**:
Wide background heating and cooling limits maintained for unoccupied rooms so empty spaces neither drive normal plant cycles nor drift to extreme temperatures.
_Avoid_: Vacation mode, frost protection
