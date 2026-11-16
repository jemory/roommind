"""Fixtures for RoomMind tests."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
from homeassistant.core import CoreState
from homeassistant.util import dt as dt_util

from custom_components.roommind.const import DOMAIN
from custom_components.roommind.coordinator import RoomMindCoordinator
from custom_components.roommind.store import RoomMindStore


@pytest.fixture(autouse=True)
def _ha_default_timezone():
    """Set HA's configured timezone from the ``ROOMMIND_TEST_TZ`` env var.

    Autouse + function-scoped so every test starts from a known HA timezone and
    the previous value is restored afterwards. This isolates tests that mutate
    ``dt_util.DEFAULT_TIME_ZONE`` themselves (e.g. the schedule-resolution tests
    in ``test_schedule_utils.py``, which pin non-UTC zones via a context manager).
    """
    tz_name = os.environ.get("ROOMMIND_TEST_TZ", "UTC")
    original = dt_util.get_default_time_zone()
    dt_util.set_default_time_zone(ZoneInfo(tz_name))
    try:
        yield
    finally:
        dt_util.set_default_time_zone(original)


def make_mock_states_get(
    temp="18.0",
    humidity="55.0",
    schedule_state="on",
    schedule_attrs=None,
    outdoor_temp="5.0",
    outdoor_temp_entity="sensor.outdoor_temp",
    window_sensors=None,
    selector_state=None,
    selector_entity="input_boolean.schedule_toggle",
    person_states=None,
    extra=None,
    temp_unit=None,
):
    """Create a mock ``hass.states.get`` with configurable return values.

    Parameters
    ----------
    temp:
        Value for ``sensor.living_room_temp``.  Set to ``None`` to return
        ``None`` (entity missing).  ``"unavailable"`` is also supported.
    humidity:
        Value for ``sensor.living_room_humidity``.
    schedule_state:
        State for ``schedule.living_room_heating``.
    schedule_attrs:
        Attributes dict for the schedule entity (default ``{}``).
    outdoor_temp:
        If given, ``outdoor_temp_entity`` will return this value.
    outdoor_temp_entity:
        Entity id for the outdoor sensor (default ``sensor.outdoor_temp``).
    window_sensors:
        Dict mapping ``binary_sensor.*`` entity ids to state strings
        (e.g. ``{"binary_sensor.window1": "on"}``).
    selector_state:
        If given, ``selector_entity`` will return this state string.
    selector_entity:
        Entity id for the schedule selector (default ``input_boolean.schedule_toggle``).
    person_states:
        Dict mapping ``person.*`` entity ids to state strings
        (e.g. ``{"person.kevin": "home"}``).
    extra:
        Dict mapping arbitrary entity ids to ``(state_str, attrs_dict | None)``
        tuples for any entities not covered by the above parameters.
    temp_unit:
        If given (e.g. ``"°F"``), temperature sensor mocks will include
        ``unit_of_measurement`` in their attributes.
    """
    if schedule_attrs is None:
        schedule_attrs = {}
    if window_sensors is None:
        window_sensors = {}
    if person_states is None:
        person_states = {}
    if extra is None:
        extra = {}

    def _temp_attrs():
        return {"unit_of_measurement": temp_unit} if temp_unit else {}

    def _mock(entity_id):
        # Temperature sensor
        if entity_id == "sensor.living_room_temp":
            if temp is None:
                return None
            s = MagicMock()
            s.state = temp
            s.attributes = _temp_attrs()
            return s

        # Humidity sensor
        if entity_id == "sensor.living_room_humidity":
            if humidity is None:
                return None
            s = MagicMock()
            s.state = humidity
            s.attributes = {}
            return s

        # Schedule entity
        if entity_id == "schedule.living_room_heating":
            s = MagicMock()
            s.state = schedule_state
            s.attributes = schedule_attrs
            return s

        # Outdoor temperature
        if outdoor_temp is not None and entity_id == outdoor_temp_entity:
            s = MagicMock()
            s.state = outdoor_temp
            s.attributes = _temp_attrs()
            return s

        # Window / door sensors
        if entity_id in window_sensors:
            s = MagicMock()
            s.state = window_sensors[entity_id]
            return s

        # Schedule selector
        if selector_state is not None and entity_id == selector_entity:
            s = MagicMock()
            s.state = selector_state
            return s

        # Person entities
        if entity_id in person_states:
            s = MagicMock()
            s.state = person_states[entity_id]
            return s

        # Arbitrary extras
        if entity_id in extra:
            val = extra[entity_id]
            s = MagicMock()
            if isinstance(val, tuple):
                s.state = val[0]
                s.attributes = val[1] if val[1] is not None else {}
            else:
                s.state = val
            return s

        return None

    return _mock


@pytest.fixture
def hass():
    """Return a mocked Home Assistant instance."""
    hass = MagicMock()
    hass.data = {}
    hass.config.path = MagicMock(side_effect=lambda *p: "/".join(("/config",) + p))
    hass.config_entries = MagicMock()
    hass.config_entries.async_forward_entry_setups = AsyncMock()
    hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)
    hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *args: fn(*args))
    return hass


@pytest.fixture
def mock_config_entry():
    """Return a mocked config entry."""
    entry = MagicMock()
    entry.entry_id = "test_entry_id"
    entry.data = {}
    entry.options = {}
    return entry


@pytest.fixture
def store(hass):
    """Return a RoomMindStore with mocked HA storage backend."""
    s = RoomMindStore(hass)
    s._store = AsyncMock()
    s._store.async_load = AsyncMock(return_value=None)
    s._store.async_save = AsyncMock()
    return s


# A Central Thermostat that reports its own reading and accepts a heat/cool band.
THERMOSTAT_ENTITY = "climate.hall_thermostat"

DEFAULT_THERMOSTAT_ATTRIBUTES: dict[str, Any] = {
    "current_temperature": 20.0,
    "hvac_modes": ["off", "heat", "cool", "heat_cool", "fan_only"],
    "supported_features": 2,
    "min_temp": 5.0,
    "max_temp": 35.0,
    "target_temp_step": 0.5,
    "temperature": None,
    "target_temp_low": 18.0,
    "target_temp_high": 24.0,
    "fan_modes": ["auto", "on"],
    "fan_mode": "auto",
    "hvac_action": "idle",
}


class FakeState:
    """A minimal HA state object."""

    def __init__(self, entity_id: str, state: str, attributes: dict[str, Any] | None = None) -> None:
        self.entity_id = entity_id
        self.state = state
        self.attributes = dict(attributes or {})
        self.last_changed = dt_util.utcnow()


class FakeStates:
    """A dict-backed ``hass.states``."""

    def __init__(self) -> None:
        self._states: dict[str, FakeState] = {}

    def set(self, entity_id: str, state: str, attributes: dict[str, Any] | None = None) -> FakeState:
        """Store one entity state and return it."""
        self._states[entity_id] = FakeState(entity_id, state, attributes)
        return self._states[entity_id]

    def get(self, entity_id: str) -> FakeState | None:
        """Return the stored state, or None."""
        return self._states.get(entity_id)

    def remove(self, entity_id: str) -> None:
        """Forget one entity."""
        self._states.pop(entity_id, None)


class FakeRegistry:
    """A dict-backed entity registry."""

    def __init__(self) -> None:
        self.entities: dict[str, Any] = {}
        self.removed: list[str] = []

    def async_get(self, entity_id: str) -> Any:
        """Return the registered entry, or None."""
        return self.entities.get(entity_id)

    def async_remove(self, entity_id: str) -> None:
        """Record and drop one entity."""
        self.removed.append(entity_id)
        self.entities.pop(entity_id, None)


class Harness:
    """One coordinator with a real store, wired to the fake Home Assistant."""

    def __init__(self, hass: Any) -> None:
        self.hass = hass
        self.store = RoomMindStore(hass)
        self.store._store = AsyncMock()
        self.store._store.async_load = AsyncMock(return_value=None)
        self.store._store.async_save = AsyncMock()
        self.coordinator: RoomMindCoordinator | None = None

    async def seed(
        self,
        rooms: dict[str, dict] | None = None,
        settings: dict | None = None,
        thermal_data: dict | None = None,
        occupancy_trends: dict | None = None,
    ) -> RoomMindCoordinator:
        """Fill the store and build the coordinator (not yet refreshed)."""
        await self.store.async_load()
        for area_id, room in (rooms or {}).items():
            await self.store.async_save_room(area_id, {"area_id": area_id, **room})
        base_settings = {
            "outdoor_temp_sensor": "sensor.outdoor_temp",
            "central_plant": {"thermostat_entity": THERMOSTAT_ENTITY},
        }
        base_settings.update(settings or {})
        await self.store.async_save_settings(base_settings)
        await self.store.async_save_thermal_data(thermal_data or {})
        await self.store.async_save_occupancy_trends(occupancy_trends or {})
        self.hass.data[DOMAIN] = {"store": self.store}
        with patch("homeassistant.helpers.frame.report_usage"):
            self.coordinator = RoomMindCoordinator(self.hass, MagicMock())
        self._wire_history(self.coordinator)
        return self.coordinator

    @staticmethod
    def _wire_history(coordinator: RoomMindCoordinator) -> None:
        """A history store that keeps its rows in memory (no /config files)."""
        history = MagicMock()
        history.read_detail.return_value = []
        history.read_history.return_value = []
        history.record = MagicMock()
        history.rotate = MagicMock()
        history.remove_room = MagicMock()
        coordinator._history_store = history

    async def cycle(self) -> dict:
        """Run one control cycle."""
        assert self.coordinator is not None
        return await self.coordinator._async_update_data()


def make_room(overrides: dict | None = None, **kwargs: Any) -> dict:
    """A complete, valid indoor room configuration."""
    room: dict[str, Any] = {
        "temperature_sensor": "sensor.room_temp",
        "humidity_sensor": "sensor.room_humidity",
        "climate_mode": "auto",
        "schedules": [],
        "window_sensors": [],
        "exterior_window_sensors": [],
        "interior_door_sensors": [],
        "occupancy_sensors": [],
        "covers": [],
        "read_only_covers": [],
        "comfort_heat": 21.0,
        "comfort_cool": 24.0,
        "eco_heat": 17.0,
        "eco_cool": 27.0,
        "presence_persons": [],
        "is_outdoor": False,
        "climate_control_enabled": True,
        "preheat_enabled": True,
        "co2_sensor": "",
        "voc_sensor": "",
        "ventilation_entity": "",
    }
    room.update(overrides or {})
    room.update(kwargs)
    return room


@pytest.fixture
def fake_registry(monkeypatch) -> FakeRegistry:
    """Install a dict-backed entity registry for the integration's helpers."""
    registry = FakeRegistry()
    monkeypatch.setattr("homeassistant.helpers.entity_registry.async_get", lambda hass: registry)
    return registry


@pytest.fixture
def fake_states(hass) -> FakeStates:
    """Install the dict-backed state machine on the mocked hass."""
    states = FakeStates()
    hass.states = states
    return states


@pytest.fixture
def plant_hass(hass, fake_states, fake_registry) -> Any:
    """A mocked hass that can run the whole Central Plant cycle."""
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock(return_value={})
    hass.bus = MagicMock()
    hass.state = CoreState.running
    hass.config.units = MagicMock()
    hass.config.units.temperature_unit = "°C"
    hass.config.latitude = 52.0
    hass.config.longitude = 13.0
    hass.config.time_zone = "UTC"
    hass.async_create_task = MagicMock(side_effect=lambda coro: coro.close())
    fake_states.set("sensor.outdoor_temp", "5.0", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.room_temp", "18.0", {"unit_of_measurement": "°C"})
    fake_states.set("sensor.room_humidity", "55.0")
    fake_states.set(THERMOSTAT_ENTITY, "heat", dict(DEFAULT_THERMOSTAT_ATTRIBUTES))
    return hass


@pytest.fixture
def harness(plant_hass) -> Harness:
    """A coordinator harness sharing the fake hass."""
    return Harness(plant_hass)
