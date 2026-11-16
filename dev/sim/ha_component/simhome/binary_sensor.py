"""Window contacts, interior doors and occupancy sensors."""

from __future__ import annotations

from typing import Any

from homeassistant.components.binary_sensor import BinarySensorDeviceClass, BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .entity import SimEntity


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, add: AddEntitiesCallback) -> None:
    rt = hass.data[DOMAIN]
    entities: list[BinarySensorEntity] = []
    for eid, window in rt.world.window_sensors.items():
        entities.append(SimWindowSensor(rt, eid, window.spec.name or eid.split(".", 1)[1].replace("_", " ").title()))
    for eid in rt.world.doors:
        entities.append(SimDoorSensor(rt, eid, eid.split(".", 1)[1].replace("_", " ").title()))
    for eid, occ in rt.world.occupancy.items():
        entities.append(SimOccupancySensor(rt, eid, occ.spec.name))
    add(entities)


class SimWindowSensor(SimEntity, BinarySensorEntity):
    _attr_device_class = BinarySensorDeviceClass.WINDOW

    @property
    def available(self) -> bool:
        return self.rt.world.is_available(self.entity_id, self.now)

    @property
    def is_on(self) -> bool:
        return self.rt.world.window_sensors[self.entity_id].open


class SimDoorSensor(SimEntity, BinarySensorEntity):
    """An interior door between two rooms: ``on`` = open, ``off`` = closed."""

    _attr_device_class = BinarySensorDeviceClass.DOOR

    def __init__(self, rt: Any, eid: str, name: str) -> None:
        super().__init__(rt, eid, name)

    @property
    def available(self) -> bool:
        return self.rt.world.is_available(self.entity_id, self.now)

    @property
    def is_on(self) -> bool:
        return self.rt.world.doors[self.entity_id].open

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        spec = self.rt.world.doors[self.entity_id].spec
        return {"room": spec.room, "neighbor": spec.neighbor}


class SimOccupancySensor(SimEntity, BinarySensorEntity):
    _attr_device_class = BinarySensorDeviceClass.OCCUPANCY

    def __init__(self, rt: Any, eid: str, name: str) -> None:
        super().__init__(rt, eid, name)

    @property
    def available(self) -> bool:
        return self.rt.world.is_available(self.entity_id, self.now)

    @property
    def is_on(self) -> bool:
        return self.rt.world.occupancy[self.entity_id].on
