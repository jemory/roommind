"""Tests for RoomMind climate platform."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.climate import (
    ATTR_TARGET_TEMP_HIGH,
    ATTR_TARGET_TEMP_LOW,
    PRESET_COMFORT,
    PRESET_ECO,
    PRESET_NONE,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.const import UnitOfTemperature

from custom_components.roommind.climate import (
    OVERRIDE_TURN_ON_REFRESH_DELAY_S,
    PRESET_MANUAL,
    RoomMindHouseClimate,
    RoomMindOverrideClimate,
    RoomMindRoomClimate,
    _create_room_climates,
    async_setup_entry,
)
from custom_components.roommind.const import (
    DEFAULT_COMFORT_COOL,
    DEFAULT_COMFORT_HEAT,
    DOMAIN,
    OVERRIDE_BOOST,
    OVERRIDE_CUSTOM,
    OVERRIDE_ECO,
)


@pytest.fixture
def mock_coordinator():
    coordinator = MagicMock()
    coordinator.hass = MagicMock()
    coordinator.async_request_refresh = AsyncMock()
    coordinator.is_override_seed_pending = MagicMock(return_value=False)
    store = MagicMock()
    coordinator.hass.data = {DOMAIN: {"store": store}}
    coordinator.data = {}
    return coordinator, store


@pytest.fixture(autouse=True)
def call_later():
    """Capture the deferred refresh and stub state writes (entities are never added to hass here)."""
    cancel = MagicMock()
    with (
        patch("custom_components.roommind.climate.async_call_later", return_value=cancel) as later,
        patch.object(RoomMindOverrideClimate, "async_write_ha_state"),
    ):
        later.cancel = cancel
        yield later


def _active_auto_room(**overrides):
    room = {
        "climate_mode": "auto",
        "override_heat": 21.0,
        "override_cool": 24.0,
        "override_until": None,
        "override_type": "custom",
    }
    room.update(overrides)
    return room


def test_create_room_climates(mock_coordinator):
    """Factory creates the deprecated override entity next to the room climate entity (#238)."""
    coordinator, _ = mock_coordinator
    climates = _create_room_climates(coordinator, "living_room")
    assert [type(c) for c in climates] == [RoomMindOverrideClimate, RoomMindRoomClimate]


def test_unique_id_and_entity_id(mock_coordinator):
    """Climate entity has correct unique_id and entity_id."""
    coordinator, _ = mock_coordinator
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    assert entity.unique_id == "roommind_living_room_override"
    assert entity.entity_id == "climate.roommind_living_room_override"


def test_deprecated_override_entity_is_disabled_for_new_rooms(mock_coordinator):
    """New registry entries of the deprecated override entity start disabled (#238)."""
    coordinator, _ = mock_coordinator
    assert RoomMindOverrideClimate(coordinator, "living_room").entity_registry_enabled_default is False


def test_hvac_mode_off_when_no_override(mock_coordinator):
    """hvac_mode returns OFF when no override is set."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = {
        "override_heat": None,
        "override_cool": None,
        "override_until": None,
        "override_type": None,
    }
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    assert entity.hvac_mode == HVACMode.OFF


def test_hvac_mode_heat_cool_when_permanent_override(mock_coordinator):
    """hvac_mode returns HEAT_COOL when permanent override is active in auto mode."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = _active_auto_room()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    assert entity.hvac_mode == HVACMode.HEAT_COOL


def test_hvac_mode_heat_cool_when_timed_override_active(mock_coordinator):
    """hvac_mode returns HEAT_COOL when timed override is still active."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = _active_auto_room(override_until=time.time() + 3600, override_type="boost")
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    assert entity.hvac_mode == HVACMode.HEAT_COOL


def test_hvac_mode_off_when_timed_override_expired(mock_coordinator):
    """hvac_mode returns OFF when timed override has expired."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = _active_auto_room(override_until=time.time() - 100, override_type="boost")
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    assert entity.hvac_mode == HVACMode.OFF


def test_hvac_mode_heat_when_heat_only(mock_coordinator):
    """hvac_mode returns HEAT for active override in heat_only room."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = _active_auto_room(climate_mode="heat_only", override_cool=None)
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    assert entity.hvac_mode == HVACMode.HEAT


def test_hvac_mode_cool_when_cool_only(mock_coordinator):
    """hvac_mode returns COOL for active override in cool_only room."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = _active_auto_room(climate_mode="cool_only", override_heat=None)
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    assert entity.hvac_mode == HVACMode.COOL


@pytest.mark.parametrize("entity_class", [RoomMindOverrideClimate, RoomMindRoomClimate, RoomMindHouseClimate])
@pytest.mark.parametrize(
    "name",
    ["current_temperature", "target_temperature", "target_temperature_low", "target_temperature_high"],
)
def test_temperatures_are_served_under_both_names(entity_class, name):
    """HA 2026.11 reads native_*, older releases the old names; both must be the same property."""
    assert getattr(entity_class, name) is getattr(entity_class, f"native_{name}")


@pytest.mark.parametrize("entity_class", [RoomMindOverrideClimate, RoomMindRoomClimate, RoomMindHouseClimate])
def test_temperature_unit_is_celsius_under_both_names(entity_class, mock_coordinator):
    if entity_class is RoomMindHouseClimate:
        entity = entity_class(mock_coordinator[0])
    else:
        entity = entity_class(mock_coordinator[0], "living_room")
    assert entity._attr_native_temperature_unit == UnitOfTemperature.CELSIUS
    assert entity.temperature_unit == UnitOfTemperature.CELSIUS


def test_override_entity_native_range(mock_coordinator):
    coordinator, store = mock_coordinator
    store.get_room.return_value = _active_auto_room()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    assert entity.native_target_temperature_low == 21.0
    assert entity.native_target_temperature_high == 24.0


def test_override_entity_auto_reports_range(mock_coordinator):
    """Auto mode exposes HEAT_COOL with range support and low/high targets."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = _active_auto_room()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    assert HVACMode.HEAT_COOL in entity.hvac_modes
    assert entity.supported_features & ClimateEntityFeature.TARGET_TEMPERATURE_RANGE
    assert entity.target_temperature_low == 21.0
    assert entity.target_temperature_high == 24.0


def test_override_entity_auto_accepts_single_temperature(mock_coordinator):
    """Auto mode also accepts a bare temperature, HA rejects it without the feature (#471)."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = _active_auto_room()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    assert entity.supported_features & ClimateEntityFeature.TARGET_TEMPERATURE
    # None keeps the dual slider in the thermostat card
    assert entity.target_temperature is None


def test_override_entity_heat_only_single_target(mock_coordinator):
    """heat_only mode exposes single-target HEAT with override_heat value."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = _active_auto_room(climate_mode="heat_only", override_cool=None)
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    assert entity.hvac_modes == [HVACMode.OFF, HVACMode.HEAT]
    assert entity.supported_features & ClimateEntityFeature.TARGET_TEMPERATURE
    assert not entity.supported_features & ClimateEntityFeature.TARGET_TEMPERATURE_RANGE
    assert entity.target_temperature == 21.0


def test_override_entity_cool_only_single_target(mock_coordinator):
    """cool_only mode exposes single-target COOL with override_cool value."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = _active_auto_room(climate_mode="cool_only", override_heat=None)
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    assert entity.hvac_modes == [HVACMode.OFF, HVACMode.COOL]
    assert entity.target_temperature == 24.0


def test_target_temperatures_none_when_no_override(mock_coordinator):
    """All target properties return None when no override is active."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = {
        "climate_mode": "auto",
        "override_heat": None,
        "override_cool": None,
        "override_until": None,
        "override_type": None,
    }
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    assert entity.target_temperature is None
    assert entity.target_temperature_low is None
    assert entity.target_temperature_high is None


def test_current_temperature_from_coordinator_data(mock_coordinator):
    """current_temperature reads from coordinator.data."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = {"override_heat": None, "override_cool": None}
    coordinator.data = {"rooms": {"living_room": {"current_temp": 20.5}}}
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    assert entity.current_temperature == 20.5


def test_current_temperature_none_when_no_data(mock_coordinator):
    """current_temperature returns None when coordinator has no data."""
    coordinator, store = mock_coordinator
    coordinator.data = None
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    assert entity.current_temperature is None


def test_current_temperature_none_when_room_not_in_data(mock_coordinator):
    """current_temperature returns None when room not in coordinator data."""
    coordinator, store = mock_coordinator
    coordinator.data = {"rooms": {"other_room": {"current_temp": 20.0}}}
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    assert entity.current_temperature is None


@pytest.mark.asyncio
async def test_override_entity_set_range_writes_split(mock_coordinator):
    """Setting low/high in auto mode writes split override_heat/override_cool."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = {"climate_mode": "auto"}
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_temperature(target_temp_low=20.0, target_temp_high=25.0)
    store.async_update_room.assert_awaited_once_with(
        "living_room",
        {
            "override_heat": 20.0,
            "override_cool": 25.0,
            "override_until": None,
            "override_type": OVERRIDE_CUSTOM,
        },
    )
    coordinator.async_request_refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_set_single_temperature_auto_derives_dead_band(mock_coordinator):
    """Bare temperature in auto mode derives a dead-band from comfort_cool."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = {"climate_mode": "auto", "comfort_cool": 24.0}
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_temperature(temperature=22.0)
    store.async_update_room.assert_awaited_once_with(
        "living_room",
        {
            "override_heat": 22.0,
            "override_cool": 24.0,
            "override_until": None,
            "override_type": OVERRIDE_CUSTOM,
        },
    )


@pytest.mark.asyncio
async def test_set_single_temperature_auto_above_comfort_cool(mock_coordinator):
    """Bare temperature above comfort_cool collapses band to the higher value."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = {"climate_mode": "auto", "comfort_cool": 24.0}
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_temperature(temperature=26.0)
    written = store.async_update_room.await_args[0][1]
    assert written["override_heat"] == 26.0
    assert written["override_cool"] == 26.0


@pytest.mark.asyncio
async def test_set_single_temperature_heat_only(mock_coordinator):
    """Bare temperature in heat_only mode writes heat target only."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = {"climate_mode": "heat_only"}
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_temperature(temperature=22.0)
    written = store.async_update_room.await_args[0][1]
    assert written["override_heat"] == 22.0
    assert written["override_cool"] is None


@pytest.mark.asyncio
async def test_set_single_temperature_cool_only(mock_coordinator):
    """Bare temperature in cool_only mode writes cool target only."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = {"climate_mode": "cool_only"}
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_temperature(temperature=23.0)
    written = store.async_update_room.await_args[0][1]
    assert written["override_heat"] is None
    assert written["override_cool"] == 23.0


@pytest.mark.asyncio
async def test_set_temperature_no_temp_kwarg(mock_coordinator):
    """set_temperature does nothing when no temperature kwargs are given."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = {"climate_mode": "auto"}
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_temperature()
    store.async_update_room.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_hvac_mode_off_clears_override(mock_coordinator):
    """Setting hvac_mode to OFF clears override and refreshes."""
    coordinator, store = mock_coordinator
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_hvac_mode(HVACMode.OFF)
    store.async_update_room.assert_awaited_once_with(
        "living_room",
        {
            "override_heat": None,
            "override_cool": None,
            "override_until": None,
            "override_type": None,
        },
    )
    coordinator.async_request_refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_set_hvac_mode_heat_cool_activates_with_comfort_defaults(mock_coordinator):
    """Setting HEAT_COOL without active override activates with comfort targets."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = {
        "climate_mode": "auto",
        "override_heat": None,
        "override_cool": None,
        "override_until": None,
        "override_type": None,
    }
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_hvac_mode(HVACMode.HEAT_COOL)
    store.async_update_room.assert_awaited_once_with(
        "living_room",
        {
            "override_heat": DEFAULT_COMFORT_HEAT,
            "override_cool": DEFAULT_COMFORT_COOL,
            "override_until": None,
            "override_type": OVERRIDE_CUSTOM,
        },
    )
    coordinator.async_request_refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_hvac_mode_heat_activates_heat_only(mock_coordinator):
    """Setting HEAT in heat_only room activates heat target only."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = {
        "climate_mode": "heat_only",
        "comfort_heat": 22.0,
        "override_heat": None,
        "override_cool": None,
    }
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_hvac_mode(HVACMode.HEAT)
    written = store.async_update_room.await_args[0][1]
    assert written["override_heat"] == 22.0
    assert written["override_cool"] is None


@pytest.mark.asyncio
async def test_set_hvac_mode_cool_activates_cool_only(mock_coordinator):
    """Setting COOL in cool_only room activates cool target only."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = {
        "climate_mode": "cool_only",
        "comfort_cool": 25.0,
        "override_heat": None,
        "override_cool": None,
    }
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_hvac_mode(HVACMode.COOL)
    written = store.async_update_room.await_args[0][1]
    assert written["override_heat"] is None
    assert written["override_cool"] == 25.0


@pytest.mark.asyncio
async def test_set_hvac_mode_noop_when_override_exists(mock_coordinator):
    """Setting a non-OFF mode does not update store if override already active."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = _active_auto_room(override_heat=23.0, override_cool=25.0)
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_hvac_mode(HVACMode.HEAT_COOL)
    store.async_update_room.assert_not_awaited()
    coordinator.clear_override_seed.assert_called_once_with("living_room")
    coordinator.async_request_refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_repeated_turn_on_on_pending_seed_keeps_masking(mock_coordinator, call_later):
    """A second turn_on while the fresh seed is masked neither unmasks nor refreshes nor re-arms the timer (#447)."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = _active_auto_room()
    store.async_update_room = AsyncMock()
    coordinator.is_override_seed_pending.return_value = True
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_hvac_mode(HVACMode.HEAT_COOL)

    coordinator.is_override_seed_pending.assert_called_once_with("living_room")
    coordinator.clear_override_seed.assert_not_called()
    coordinator.note_override_seed.assert_not_called()
    coordinator.async_request_refresh.assert_not_awaited()
    store.async_update_room.assert_not_awaited()
    call_later.assert_not_called()
    call_later.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_turn_off_on_pending_seed_still_clears_and_refreshes(mock_coordinator):
    coordinator, store = mock_coordinator
    store.get_room.return_value = _active_auto_room()
    store.async_update_room = AsyncMock()
    coordinator.is_override_seed_pending.return_value = True
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_hvac_mode(HVACMode.OFF)

    coordinator.clear_override_seed.assert_called_once_with("living_room")
    coordinator.async_request_refresh.assert_awaited_once()


@pytest.mark.parametrize(
    ("live_mode", "expected"),
    [
        ("heating", HVACAction.HEATING),
        ("cooling", HVACAction.COOLING),
        ("circulating", HVACAction.FAN),
        ("idle", HVACAction.IDLE),
        (None, HVACAction.IDLE),
    ],
)
def test_override_entity_hvac_action_follows_the_plant(mock_coordinator, live_mode, expected):
    """The deprecated entity mirrors what the Central Plant is doing for its room."""
    coordinator, _ = mock_coordinator
    coordinator.data = {"rooms": {"living_room": {"mode": live_mode}}}
    assert RoomMindOverrideClimate(coordinator, "living_room").hvac_action == expected


def test_override_entity_hvac_action_without_live_data(mock_coordinator):
    coordinator, _ = mock_coordinator
    coordinator.data = {"rooms": {}}
    assert RoomMindOverrideClimate(coordinator, "living_room").hvac_action == HVACAction.IDLE


def test_hvac_mode_off_when_room_missing(mock_coordinator):
    """hvac_mode returns OFF when room doesn't exist in store."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = None
    entity = RoomMindOverrideClimate(coordinator, "nonexistent")
    assert entity.hvac_mode == HVACMode.OFF


@pytest.mark.asyncio
async def test_async_setup_entry_creates_entities_for_all_rooms():
    """async_setup_entry creates climate entities for all rooms."""
    coordinator = MagicMock()
    coordinator._climate_entity_areas = set()

    store = MagicMock()
    store.get_rooms.return_value = {
        "living_room": {"thermostats": ["climate.living"]},
        "bedroom": {},
    }

    entry = MagicMock()
    entry.entry_id = "test_entry"

    hass = MagicMock()
    hass.data = {DOMAIN: {entry.entry_id: coordinator, "store": store}}

    async_add_entities = MagicMock()

    await async_setup_entry(hass, entry, async_add_entities)

    assert coordinator.async_add_climate_entities is async_add_entities
    async_add_entities.assert_called_once()
    entities = async_add_entities.call_args[0][0]
    assert sum(isinstance(e, RoomMindOverrideClimate) for e in entities) == 2
    assert sum(isinstance(e, RoomMindRoomClimate) for e in entities) == 2
    assert sum(isinstance(e, RoomMindHouseClimate) for e in entities) == 1
    assert "living_room" in coordinator._climate_entity_areas
    assert "bedroom" in coordinator._climate_entity_areas


@pytest.mark.asyncio
async def test_async_setup_entry_no_rooms():
    """With no rooms, async_setup_entry still adds the single global house entity."""
    coordinator = MagicMock()
    coordinator._climate_entity_areas = set()

    store = MagicMock()
    store.get_rooms.return_value = {}

    entry = MagicMock()
    entry.entry_id = "test_entry"

    hass = MagicMock()
    hass.data = {DOMAIN: {entry.entry_id: coordinator, "store": store}}

    async_add_entities = MagicMock()

    await async_setup_entry(hass, entry, async_add_entities)

    async_add_entities.assert_called_once()
    entities = async_add_entities.call_args[0][0]
    assert len(entities) == 1
    assert isinstance(entities[0], RoomMindHouseClimate)


def _live(coordinator, **fields):
    coordinator.data = {"rooms": {"living_room": fields}}


@pytest.mark.asyncio
async def test_set_hvac_mode_seeds_override_from_effective_targets(mock_coordinator):
    """Switching on seeds the override from what is in force, not from comfort (#447)."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = {
        "climate_mode": "auto",
        "comfort_heat": 22.5,
        "comfort_cool": 24.0,
        "override_heat": None,
        "override_cool": None,
    }
    _live(coordinator, heat_target=24.5, cool_target=26.0)
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_hvac_mode(HVACMode.HEAT_COOL)
    written = store.async_update_room.await_args[0][1]
    assert written["override_heat"] == 24.5
    assert written["override_cool"] == 26.0
    assert written["override_type"] == OVERRIDE_CUSTOM


@pytest.mark.asyncio
async def test_set_hvac_mode_seed_excludes_mold_prevention_delta(mock_coordinator):
    """The mold prevention boost is not part of the band the override freezes."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = {"climate_mode": "auto"}
    _live(
        coordinator,
        heat_target=23.0,
        cool_target=26.0,
        mold_prevention_active=True,
        mold_prevention_delta=2.0,
    )
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_hvac_mode(HVACMode.HEAT_COOL)
    assert store.async_update_room.await_args[0][1]["override_heat"] == 21.0


@pytest.mark.asyncio
async def test_set_hvac_mode_seed_falls_back_to_comfort_when_forced_off(mock_coordinator):
    """Force-off rooms have no live band, so the comfort band is used."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = {"climate_mode": "auto", "comfort_heat": 22.0, "comfort_cool": 25.0}
    _live(coordinator, heat_target=None, cool_target=None)
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_hvac_mode(HVACMode.HEAT_COOL)
    written = store.async_update_room.await_args[0][1]
    assert (written["override_heat"], written["override_cool"]) == (22.0, 25.0)


@pytest.mark.asyncio
async def test_set_hvac_mode_seed_masks_unusable_side(mock_coordinator):
    """heat_only rooms never get a cool side from the seed."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = {"climate_mode": "heat_only"}
    _live(coordinator, heat_target=21.0, cool_target=26.0)
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_hvac_mode(HVACMode.HEAT)
    written = store.async_update_room.await_args[0][1]
    assert (written["override_heat"], written["override_cool"]) == (21.0, None)


@pytest.mark.asyncio
async def test_set_range_low_only_keeps_active_cool_side(mock_coordinator):
    """A low-only call keeps the cool side of the active override."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = _active_auto_room(override_heat=21.0, override_cool=26.0)
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_temperature(target_temp_low=22.0)
    written = store.async_update_room.await_args[0][1]
    assert (written["override_heat"], written["override_cool"]) == (22.0, 26.0)


@pytest.mark.asyncio
async def test_set_range_high_only_keeps_active_heat_side(mock_coordinator):
    """A high-only call keeps the heat side of the active override."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = _active_auto_room(override_heat=21.0, override_cool=26.0)
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_temperature(target_temp_high=28.5)
    written = store.async_update_room.await_args[0][1]
    assert (written["override_heat"], written["override_cool"]) == (21.0, 28.5)


@pytest.mark.asyncio
async def test_set_range_half_without_override_uses_effective_band(mock_coordinator):
    """Without an override the missing side comes from the live targets."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = {"climate_mode": "auto"}
    _live(coordinator, heat_target=22.5, cool_target=26.0)
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_temperature(target_temp_high=28.5)
    written = store.async_update_room.await_args[0][1]
    assert (written["override_heat"], written["override_cool"]) == (22.5, 28.5)


@pytest.mark.asyncio
async def test_set_range_half_never_inverts_band(mock_coordinator):
    """A half-range value past the kept side pulls that side along."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = _active_auto_room(override_heat=21.0, override_cool=24.0)
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_temperature(target_temp_low=26.0)
    written = store.async_update_room.await_args[0][1]
    assert (written["override_heat"], written["override_cool"]) == (26.0, 26.0)
    await entity.async_set_temperature(target_temp_high=19.0)
    written = store.async_update_room.await_args[0][1]
    assert (written["override_heat"], written["override_cool"]) == (19.0, 19.0)


@pytest.mark.asyncio
async def test_set_hvac_mode_seed_ignores_room_temperature(mock_coordinator):
    """A bare turn-on keeps the current targets even when the room sits outside the band."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = {"climate_mode": "auto"}
    _live(coordinator, heat_target=21.0, cool_target=24.0, current_temp=14.0, commanded_mode="idle")
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_hvac_mode(HVACMode.HEAT_COOL)
    written = store.async_update_room.await_args[0][1]
    assert (written["override_heat"], written["override_cool"]) == (21.0, 24.0)


@pytest.mark.asyncio
async def test_turn_on_defers_refresh_and_updates_entity_state(mock_coordinator, call_later):
    """Creating the override writes state right away but refreshes only after the grace period."""
    coordinator, store = mock_coordinator
    store.get_room.return_value = {"climate_mode": "auto"}
    _live(coordinator, heat_target=21.0, cool_target=24.0)
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_hvac_mode(HVACMode.HEAT_COOL)

    entity.async_write_ha_state.assert_called_once()
    coordinator.note_override_seed.assert_called_once_with("living_room")
    coordinator.async_request_refresh.assert_not_awaited()
    call_later.assert_called_once()
    assert call_later.call_args[0][1] == OVERRIDE_TURN_ON_REFRESH_DELAY_S

    await call_later.call_args[0][2](None)
    coordinator.clear_override_seed.assert_called_once_with("living_room")
    coordinator.async_request_refresh.assert_awaited_once()
    entity._drop_deferred_refresh()
    call_later.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_turn_on_then_set_temperature_refreshes_once_with_final_values(mock_coordinator, call_later):
    """Reported #447 sequence: only the final targets are ever evaluated."""
    coordinator, store = mock_coordinator
    rooms = {"climate_mode": "auto", "override_heat": None, "override_cool": None, "override_until": None}
    store.get_room.side_effect = lambda _a: dict(rooms)

    async def _update(_area, changes):
        rooms.update(changes)

    store.async_update_room = AsyncMock(side_effect=_update)
    _live(coordinator, heat_target=22.5, cool_target=24.0)
    entity = RoomMindOverrideClimate(coordinator, "living_room")

    await entity.async_set_hvac_mode(HVACMode.HEAT_COOL)
    coordinator.async_request_refresh.assert_not_awaited()
    await entity.async_set_temperature(target_temp_low=24.0, target_temp_high=28.5)

    coordinator.async_request_refresh.assert_awaited_once()
    coordinator.clear_override_seed.assert_called_with("living_room")
    call_later.cancel.assert_called_once()
    assert (rooms["override_heat"], rooms["override_cool"]) == (24.0, 28.5)


@pytest.mark.asyncio
async def test_repeated_turn_on_cancels_previous_timer(mock_coordinator, call_later):
    coordinator, store = mock_coordinator
    store.get_room.return_value = {"climate_mode": "auto"}
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_hvac_mode(HVACMode.HEAT_COOL)
    await entity.async_set_hvac_mode(HVACMode.HEAT_COOL)
    call_later.cancel.assert_called_once()
    assert call_later.call_count == 2


@pytest.mark.asyncio
async def test_turn_off_cancels_pending_refresh_and_refreshes(mock_coordinator, call_later):
    coordinator, store = mock_coordinator
    store.get_room.return_value = {"climate_mode": "auto"}
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_hvac_mode(HVACMode.HEAT_COOL)
    await entity.async_set_hvac_mode(HVACMode.OFF)
    call_later.cancel.assert_called_once()
    coordinator.async_request_refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_removal_cancels_pending_refresh(mock_coordinator, call_later):
    coordinator, store = mock_coordinator
    store.get_room.return_value = {"climate_mode": "auto"}
    store.async_update_room = AsyncMock()
    entity = RoomMindOverrideClimate(coordinator, "living_room")
    await entity.async_set_hvac_mode(HVACMode.HEAT_COOL)
    with patch("custom_components.roommind.climate.CoordinatorEntity.async_will_remove_from_hass", AsyncMock()):
        await entity.async_will_remove_from_hass()
    call_later.cancel.assert_called_once()


# ---------------------------------------------------------------------------
# House climate entity (#T6)
# ---------------------------------------------------------------------------


HOUSE_NEXT_EVENT = time.time() + 6 * 3600


def _live_house(**overrides):
    """The frozen coordinator.data['house'] shape (resolved settings + live fields)."""
    house = {
        "enabled": True,
        "heat_c": 21.0,
        "cool_c": 24.0,
        "eco_heat_c": 17.0,
        "eco_cool_c": 27.0,
        "climate_mode": "auto",
        "schedules": [{"entity_id": "schedule.house"}],
        "schedule_selector_entity": "",
        "preheat_enabled": True,
        "price_shift": 0.0,
        "override_heat": None,
        "override_cool": None,
        "override_until": None,
        "override_type": None,
        "active": True,
        "band": {"heat_c": 21.0, "cool_c": 24.0},
        "source": "target",
        "override_active": False,
        "schedule_entity": "schedule.house",
        "schedule_name": "House schedule",
        "schedule_block": {"from": "08:00:00", "to": "17:00:00"},
        "next_change": HOUSE_NEXT_EVENT,
        "following_rooms": 3,
    }
    house.update(overrides)
    return house


@pytest.fixture
def house_setup():
    coordinator = MagicMock()
    coordinator.hass = MagicMock()
    coordinator.hass.config.units.temperature_unit = UnitOfTemperature.CELSIUS
    coordinator.async_request_refresh = AsyncMock()
    store = MagicMock()
    store.get_settings.return_value = {}
    store.async_save_settings = AsyncMock()
    coordinator.hass.data = {DOMAIN: {"store": store}}
    coordinator.data = {
        "house": _live_house(),
        "central_plant": {"action": "heating", "house_ref_temp": 22.3},
        "rooms": {},
    }
    states = {}
    schedule = MagicMock()
    schedule.attributes = {"next_event": datetime.fromtimestamp(HOUSE_NEXT_EVENT, tz=UTC)}
    states["schedule.house"] = schedule
    coordinator.hass.states.get = states.get

    def make(house=None, plant=None, settings=None, data_missing_house=False):
        if settings is not None:
            store.get_settings.return_value = settings
        if data_missing_house:
            coordinator.data = {} if plant is None else {"central_plant": plant}
        else:
            coordinator.data = {
                "house": _live_house() if house is None else house,
                "central_plant": plant if plant is not None else {"action": "heating", "house_ref_temp": 22.3},
                "rooms": {},
            }
        entity = RoomMindHouseClimate(coordinator)
        entity.hass = coordinator.hass
        return entity

    with patch.object(RoomMindHouseClimate, "async_write_ha_state"):
        yield make, coordinator, store, states


def test_house_entity_ids_and_name(house_setup):
    make, *_ = house_setup
    entity = make()
    assert entity.unique_id == "roommind_house_climate"
    assert entity.entity_id == "climate.roommind_house"
    assert entity.name == "RoomMind House"
    assert entity.has_entity_name is False
    assert entity.icon == "mdi:home-thermometer"


def test_house_hvac_modes_and_features(house_setup):
    make, *_ = house_setup
    entity = make()
    assert entity.hvac_modes == [HVACMode.OFF, HVACMode.HEAT, HVACMode.COOL, HVACMode.HEAT_COOL]
    assert entity.supported_features & ClimateEntityFeature.TARGET_TEMPERATURE_RANGE
    assert entity.supported_features & ClimateEntityFeature.PRESET_MODE
    single = make(_live_house(climate_mode="heat_only"))
    assert single.supported_features & ClimateEntityFeature.TARGET_TEMPERATURE
    assert not single.supported_features & ClimateEntityFeature.TARGET_TEMPERATURE_RANGE


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("auto", HVACMode.HEAT_COOL), ("heat_only", HVACMode.HEAT), ("cool_only", HVACMode.COOL)],
)
def test_house_hvac_mode_maps_climate_mode(house_setup, mode, expected):
    make, *_ = house_setup
    assert make(_live_house(climate_mode=mode)).hvac_mode == expected


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("heating", HVACAction.HEATING),
        ("cooling", HVACAction.COOLING),
        ("circulating", HVACAction.FAN),
        ("idle", HVACAction.IDLE),
        (None, None),
    ],
)
def test_house_hvac_action_from_plant(house_setup, action, expected):
    make, *_ = house_setup
    assert make(plant={"action": action, "house_ref_temp": 22.3}).hvac_action == expected


def test_house_current_temperature_is_reference(house_setup):
    make, *_ = house_setup
    assert make(plant={"action": "idle", "house_ref_temp": 22.3}).current_temperature == 22.3


def test_house_target_band_from_live_data(house_setup):
    make, *_ = house_setup
    entity = make(_live_house(band={"heat_c": 20.5, "cool_c": 23.5}))
    assert entity.target_temperature_low == 20.5
    assert entity.target_temperature_high == 23.5
    assert entity.target_temperature is None  # auto keeps the dual slider


def test_house_single_target_by_mode(house_setup):
    make, *_ = house_setup
    assert make(_live_house(climate_mode="heat_only")).target_temperature == 21.0
    assert make(_live_house(climate_mode="cool_only")).target_temperature == 24.0


def test_house_preset_reflects_override_type(house_setup):
    make, *_ = house_setup
    assert make().preset_mode == PRESET_NONE
    assert (
        make(_live_house(override_heat=21.0, override_cool=24.0, override_type="boost")).preset_mode == PRESET_COMFORT
    )
    assert make(_live_house(override_heat=17.0, override_cool=27.0, override_type="eco")).preset_mode == PRESET_ECO
    assert (
        make(_live_house(override_heat=21.0, override_cool=24.0, override_type="custom")).preset_mode == PRESET_MANUAL
    )


@pytest.mark.asyncio
async def test_house_set_temperature_writes_hold_and_enables(house_setup):
    make, _, store, _ = house_setup
    entity = make()
    await entity.async_set_temperature(**{ATTR_TARGET_TEMP_LOW: 22.0, ATTR_TARGET_TEMP_HIGH: 25.0})
    written = store.async_save_settings.await_args[0][0]
    assert written["house"]["enabled"] is True
    assert written["house"]["override_heat"] == 22.0
    assert written["house"]["override_cool"] == 25.0
    assert written["house"]["override_until"] == pytest.approx(HOUSE_NEXT_EVENT)
    assert written["house"]["override_type"] == OVERRIDE_CUSTOM


@pytest.mark.asyncio
async def test_house_set_temperature_echo_is_noop(house_setup):
    make, _, store, _ = house_setup
    await make().async_set_temperature(**{ATTR_TARGET_TEMP_LOW: 21.0, ATTR_TARGET_TEMP_HIGH: 24.0})
    store.async_save_settings.assert_not_awaited()


@pytest.mark.parametrize(
    ("preset", "override_type", "band"),
    [
        (PRESET_COMFORT, OVERRIDE_BOOST, (21.0, 24.0)),
        (PRESET_ECO, OVERRIDE_ECO, (17.0, 27.0)),
    ],
)
@pytest.mark.asyncio
async def test_house_preset_writes_override(house_setup, preset, override_type, band):
    make, _, store, _ = house_setup
    await make().async_set_preset_mode(preset)
    written = store.async_save_settings.await_args[0][0]["house"]
    assert written["override_heat"] == band[0]
    assert written["override_cool"] == band[1]
    assert written["override_until"] == pytest.approx(HOUSE_NEXT_EVENT)
    assert written["override_type"] == override_type
    assert written["enabled"] is True


@pytest.mark.asyncio
async def test_house_manual_preset_holds_band_in_force(house_setup):
    make, _, store, _ = house_setup
    await make(_live_house(band={"heat_c": 20.0, "cool_c": 25.0})).async_set_preset_mode(PRESET_MANUAL)
    written = store.async_save_settings.await_args[0][0]["house"]
    assert (written["override_heat"], written["override_cool"]) == (20.0, 25.0)
    assert written["override_type"] == OVERRIDE_CUSTOM


@pytest.mark.asyncio
async def test_house_preset_none_clears_override(house_setup):
    make, _, store, _ = house_setup
    await make(_live_house(override_heat=21.0, override_cool=24.0, override_type="boost")).async_set_preset_mode(
        PRESET_NONE
    )
    assert store.async_save_settings.await_args[0][0] == {
        "house": {
            "enabled": True,
            "override_heat": None,
            "override_cool": None,
            "override_until": None,
            "override_type": None,
        }
    }


@pytest.mark.asyncio
async def test_house_off_only_pauses_control(house_setup):
    make, _, store, _ = house_setup
    entity = make()
    await entity.async_set_hvac_mode(HVACMode.OFF)
    store.async_save_settings.assert_awaited_once_with({"climate_control_active": False})


@pytest.mark.parametrize(
    ("hvac_mode", "climate_mode"),
    [(HVACMode.HEAT, "heat_only"), (HVACMode.COOL, "cool_only"), (HVACMode.HEAT_COOL, "auto")],
)
@pytest.mark.asyncio
async def test_house_mode_sets_climate_mode_and_enables(house_setup, hvac_mode, climate_mode):
    make, _, store, _ = house_setup
    await make().async_set_hvac_mode(hvac_mode)
    store.async_save_settings.assert_awaited_once_with(
        {"climate_control_active": True, "house": {"enabled": True, "climate_mode": climate_mode}}
    )


@pytest.mark.asyncio
async def test_house_turn_on_off_delegate(house_setup):
    make, _, store, _ = house_setup
    entity = make(_live_house(climate_mode="cool_only"))
    await entity.async_turn_on()
    assert store.async_save_settings.await_args[0][0]["house"]["climate_mode"] == "cool_only"
    await entity.async_turn_off()
    assert store.async_save_settings.await_args[0][0] == {"climate_control_active": False}


def test_house_control_paused_reports_off(house_setup):
    make, _, _, _ = house_setup
    entity = make(settings={"climate_control_active": False})
    assert entity.hvac_mode == HVACMode.OFF
    assert entity.hvac_action == HVACAction.OFF


def test_house_disabled_reports_off_without_raising(house_setup):
    make, *_ = house_setup
    entity = make(_live_house(enabled=False, active=False))
    assert entity.hvac_mode == HVACMode.OFF
    assert entity.hvac_action == HVACAction.OFF


def test_house_empty_coordinator_data_falls_back_to_store(house_setup):
    make, _, _, _ = house_setup
    entity = make(data_missing_house=True, settings={"house": _live_house()})
    # No central plant / house band in data: must not raise, targets fall back to the pair.
    assert entity.hvac_mode == HVACMode.HEAT_COOL
    assert entity.current_temperature is None
    assert entity.target_temperature_low == 21.0
    assert entity.target_temperature_high == 24.0


def test_house_extra_state_attributes(house_setup):
    make, *_ = house_setup
    attrs = make(
        _live_house(
            override_heat=21.0,
            override_cool=24.0,
            override_until=HOUSE_NEXT_EVENT,
            override_type="boost",
        ),
        plant={"action": "heating", "house_ref_temp": 22.3},
    ).extra_state_attributes
    assert attrs["enabled"] is True
    assert attrs["active"] is True
    assert attrs["heat_c"] == 21.0
    assert attrs["cool_c"] == 24.0
    assert attrs["eco_heat_c"] == 17.0
    assert attrs["eco_cool_c"] == 27.0
    assert attrs["climate_mode"] == "auto"
    assert attrs["override_until"] is not None
    assert attrs["source"] == "target"
    assert attrs["schedule_entity"] == "schedule.house"
    assert attrs["schedule_name"] == "House schedule"
    assert attrs["schedule_block"] == {"from": "08:00:00", "to": "17:00:00"}
    assert attrs["next_change"] == HOUSE_NEXT_EVENT
    assert attrs["following_rooms"] == 3
    assert attrs["house_reference_temperature"] == 22.3
    assert set(attrs) == {
        "enabled",
        "active",
        "heat_c",
        "cool_c",
        "eco_heat_c",
        "eco_cool_c",
        "climate_mode",
        "override_until",
        "source",
        "schedule_entity",
        "schedule_name",
        "schedule_block",
        "next_change",
        "following_rooms",
        "house_reference_temperature",
    }


# ---------------------------------------------------------------------------
# Following-room behavior (#T6)
# ---------------------------------------------------------------------------


def _following_room(**overrides):
    room = {
        "area_id": "living_room",
        "climate_mode": "auto",
        "follow_house": True,
        "comfort_heat": 21.0,
        "comfort_cool": 24.0,
        "override_heat": None,
        "override_cool": None,
        "override_until": None,
        "override_type": None,
        "climate_control_enabled": True,
    }
    room.update(overrides)
    return room


@pytest.fixture
def following_setup():
    coordinator = MagicMock()
    coordinator.hass = MagicMock()
    coordinator.hass.config.units.temperature_unit = UnitOfTemperature.CELSIUS
    coordinator.async_request_refresh = AsyncMock()
    coordinator.async_turn_room_off = AsyncMock()
    store = MagicMock()
    store.async_update_room = AsyncMock()
    store.get_settings.return_value = {}
    coordinator.hass.data = {DOMAIN: {"store": store}}
    coordinator.data = {
        "rooms": {"living_room": {"current_temp": 20.4}},
        "house": _live_house(),
    }

    def make(room=None, house=None):
        store.get_room.return_value = _following_room() if room is None else room
        if house is not None:
            coordinator.data = dict(coordinator.data, house=house)
        entity = RoomMindRoomClimate(coordinator, "living_room")
        entity.hass = coordinator.hass
        return entity

    with patch.object(RoomMindRoomClimate, "async_write_ha_state"):
        yield make, coordinator, store


def test_following_room_reports_house_mode(following_setup):
    make, *_ = following_setup
    entity = make(house=_live_house(climate_mode="heat_only"))
    assert entity.hvac_mode == HVACMode.HEAT


def test_following_room_but_inactive_house_uses_own_mode(following_setup):
    make, *_ = following_setup
    entity = make(house=_live_house(active=False, enabled=False, climate_mode="heat_only"))
    assert entity.hvac_mode == HVACMode.HEAT_COOL  # own dormant mode, house inactive


def test_independent_room_uses_own_mode_even_with_active_house(following_setup):
    make, *_ = following_setup
    entity = make(_following_room(follow_house=False), house=_live_house(climate_mode="cool_only"))
    assert entity.hvac_mode == HVACMode.HEAT_COOL


@pytest.mark.asyncio
async def test_following_room_mode_change_flips_follow_house(following_setup):
    make, _, store = following_setup
    await make().async_set_hvac_mode(HVACMode.COOL)
    written = store.async_update_room.await_args[0][1]
    assert written["climate_mode"] == "cool_only"
    assert written["follow_house"] is False
    assert written["climate_control_enabled"] is True


@pytest.mark.asyncio
async def test_independent_room_mode_change_does_not_write_follow_house(following_setup):
    make, _, store = following_setup
    await make(_following_room(follow_house=False)).async_set_hvac_mode(HVACMode.HEAT)
    written = store.async_update_room.await_args[0][1]
    assert written["climate_mode"] == "heat_only"
    assert "follow_house" not in written


@pytest.mark.asyncio
async def test_following_room_off_still_turns_room_off(following_setup):
    make, coordinator, store = following_setup
    await make().async_set_hvac_mode(HVACMode.OFF)
    coordinator.async_turn_room_off.assert_awaited_once_with("living_room")
    store.async_update_room.assert_not_awaited()


@pytest.mark.parametrize("follow", [True, False])
def test_room_follow_house_attribute_reflects_stored_flag(following_setup, follow):
    make, *_ = following_setup
    entity = make(_following_room(follow_house=follow))
    assert entity.extra_state_attributes["follow_house"] is follow
