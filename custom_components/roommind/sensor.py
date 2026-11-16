"""Sensor platform for RoomMind."""

from __future__ import annotations

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import RoomMindCoordinator
from .utils.area_utils import rooms_with_area

# The Central Plant publishes one shared state block per coordinator cycle
CENTRAL_ACTION_ENTITY_ID = "sensor.roommind_central_action"
HOUSE_TEMPERATURE_ENTITY_ID = "sensor.roommind_house_temperature"


def _central_plant_data(coordinator: RoomMindCoordinator) -> dict:
    """Return the live Central Plant block, or an empty dict before the first cycle."""
    data = coordinator.data or {}
    plant = data.get("central_plant") if isinstance(data, dict) else None
    return plant if isinstance(plant, dict) else {}


def _create_room_entities(coordinator: RoomMindCoordinator, area_id: str) -> list[SensorEntity]:
    """Create the standard set of sensor entities for a room."""
    return [
        RoomMindTargetTemperatureSensor(coordinator, area_id),
        RoomMindModeSensor(coordinator, area_id),
    ]


def _create_global_entities(coordinator: RoomMindCoordinator) -> list[SensorEntity]:
    """Create the whole-home Central Plant sensors (enabled by default)."""
    return [RoomMindCentralActionSensor(coordinator), RoomMindHouseTemperatureSensor(coordinator)]


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up RoomMind sensor entities from a config entry."""
    coordinator: RoomMindCoordinator = hass.data[DOMAIN][entry.entry_id]
    store = hass.data[DOMAIN]["store"]

    # Store the callback on the coordinator so dynamic entity creation works
    coordinator.async_add_entities = async_add_entities

    # The Central Plant sensors exist once per installation, independent of any room
    entities: list[SensorEntity] = _create_global_entities(coordinator)

    # Create entities for rooms that already exist in the store
    rooms = rooms_with_area(hass, store.get_rooms())
    for area_id in rooms:
        entities.extend(_create_room_entities(coordinator, area_id))
        coordinator._entity_areas.add(area_id)
    async_add_entities(entities)


class _RoomMindBaseSensor(CoordinatorEntity, SensorEntity):
    """Base class for all RoomMind room sensors."""

    _attr_has_entity_name = True
    # Deprecated (#238): only rooms set up before 2.0 keep them enabled; HA applies this to new registry entries only
    _attr_entity_registry_enabled_default = False
    _data_key: str  # Key in the room state dict (e.g. "current_temp")

    def __init__(
        self,
        coordinator: RoomMindCoordinator,
        area_id: str,
        suffix: str,
        name_label: str,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._area_id = area_id
        self._attr_unique_id = f"{DOMAIN}_{area_id}_{suffix}"
        self._attr_name = f"{area_id} {name_label}"
        self.entity_id = f"sensor.{DOMAIN}_{area_id}_{suffix}"

    @property
    def native_value(self) -> float | str | None:
        """Return the sensor value from the coordinator data."""
        room = self.coordinator.data.get("rooms", {}).get(self._area_id)
        if room:
            val = room.get(self._data_key)
            return val if isinstance(val, (float, int, str)) else None
        return None


class RoomMindTargetTemperatureSensor(_RoomMindBaseSensor):
    """Sensor showing the target temperature for a RoomMind room."""

    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _data_key = "target_temp"

    def __init__(self, coordinator: RoomMindCoordinator, area_id: str) -> None:
        super().__init__(coordinator, area_id, "target_temp", "Target Temperature")


class RoomMindModeSensor(_RoomMindBaseSensor):
    """Sensor showing the current mode for a RoomMind room."""

    _data_key = "mode"

    def __init__(self, coordinator: RoomMindCoordinator, area_id: str) -> None:
        super().__init__(coordinator, area_id, "mode", "Mode")

    @property
    def native_value(self) -> str | None:
        """Return the current mode, defaulting to 'idle'."""
        room = self.coordinator.data.get("rooms", {}).get(self._area_id)
        if room:
            val = room.get("mode", "idle")
            return str(val) if val is not None else "idle"
        return "idle"


class RoomMindCentralActionSensor(CoordinatorEntity, SensorEntity):
    """The plant action the whole-home optimizer is executing (idle/heating/cooling/circulating)."""

    _attr_has_entity_name = True
    _attr_entity_registry_enabled_default = True
    _attr_icon = "mdi:hvac"
    _attr_translation_key = "central_action"

    def __init__(self, coordinator: RoomMindCoordinator) -> None:
        """Initialize the Central HVAC Action sensor."""
        super().__init__(coordinator)
        self._attr_unique_id = f"{DOMAIN}_central_action"
        self.entity_id = CENTRAL_ACTION_ENTITY_ID

    @property
    def native_value(self) -> str:
        """Return the plant's current action, defaulting to idle."""
        value = _central_plant_data(self.coordinator).get("action", "idle")
        return str(value) if value is not None else "idle"

    @property
    def extra_state_attributes(self) -> dict:
        """Expose the decision behind the action for the panel and automations."""
        plant = _central_plant_data(self.coordinator)
        return {
            # The controller publishes the commanded action as "action"; older shapes differ
            "commanded_action": plant.get("commanded_action", plant.get("action")),
            "driving_rooms": plant.get("driving_rooms", []),
            "decision_reason": plant.get("decision_reason"),
            "guard_active": bool(plant.get("guard_active", False)),
            "guard_reason": plant.get("guard_reason"),
            "wet_coil_locked": bool(plant.get("wet_coil_locked", False)),
            "exterior_paused_rooms": plant.get("exterior_paused_rooms", plant.get("blocking_rooms", [])),
            "plant_override_active": bool(plant.get("plant_override_active", plant.get("any_override", False))),
        }


class RoomMindHouseTemperatureSensor(CoordinatorEntity, SensorEntity):
    """The House Reference Temperature the Central Plant compares the rooms against.

    ``SensorDeviceClass.TEMPERATURE`` with a Celsius native unit makes Home
    Assistant convert the value into the installation's configured display unit.
    """

    _attr_has_entity_name = True
    _attr_entity_registry_enabled_default = True
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _attr_translation_key = "house_temperature"

    def __init__(self, coordinator: RoomMindCoordinator) -> None:
        """Initialize the House Reference Temperature sensor."""
        super().__init__(coordinator)
        self._attr_unique_id = f"{DOMAIN}_house_temperature"
        self.entity_id = HOUSE_TEMPERATURE_ENTITY_ID

    @property
    def native_value(self) -> float | None:
        """Return the House Reference Temperature, or None while it is unknown."""
        value = _central_plant_data(self.coordinator).get("house_ref_temp")
        return float(value) if isinstance(value, (int, float)) else None
