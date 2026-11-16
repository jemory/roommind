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
