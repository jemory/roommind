"""Tests for RoomMind WebSocket API."""

from __future__ import annotations

import time
from datetime import UTC
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.roommind.const import (
    DOMAIN,
    get_central_plant_settings,
    get_house_settings,
)
from custom_components.roommind.websocket_api import (
    _csv_to_points,
    _safe_float,
    websocket_covers_clear_override,
    websocket_delete_room,
    websocket_get_analytics,
    websocket_get_diagnostics,
    websocket_get_settings,
    websocket_house_clear_override,
    websocket_house_save,
    websocket_list_rooms,
    websocket_override_clear,
    websocket_override_set,
    websocket_plant_clear_override,
    websocket_save_room,
    websocket_save_settings,
    websocket_thermal_reset,
    websocket_thermal_reset_all,
)

# The HA @async_response decorator wraps async handlers into synchronous
# schedulers.  Access the original coroutine via ``__wrapped__`` so we can
# ``await`` them directly in tests without needing a running event-loop task
# factory on the mock hass object.
_list_rooms = websocket_list_rooms.__wrapped__
_save_room = websocket_save_room.__wrapped__
_delete_room = websocket_delete_room.__wrapped__.__wrapped__
_override_set = websocket_override_set.__wrapped__
_override_clear = websocket_override_clear.__wrapped__
_get_settings = websocket_get_settings.__wrapped__
_save_settings = websocket_save_settings.__wrapped__.__wrapped__
_house_save = websocket_house_save.__wrapped__
_house_clear_override = websocket_house_clear_override.__wrapped__
_thermal_reset = websocket_thermal_reset.__wrapped__.__wrapped__
_thermal_reset_all = websocket_thermal_reset_all.__wrapped__.__wrapped__
_get_analytics = websocket_get_analytics.__wrapped__
_get_diagnostics = websocket_get_diagnostics.__wrapped__.__wrapped__
_covers_clear_override = websocket_covers_clear_override.__wrapped__


@pytest.fixture
def connection():
    """Return a mocked WebSocket connection."""
    conn = MagicMock()
    conn.send_result = MagicMock()
    conn.send_error = MagicMock()
    return conn


@pytest.fixture
def ws_hass(hass, store):
    """Return a hass instance with the store wired into hass.data."""
    hass.data[DOMAIN] = {"store": store}
    return hass


@pytest.mark.asyncio
async def test_list_rooms_empty(ws_hass, store, connection):
    """Listing rooms on a fresh store returns an empty dict."""
    await store.async_load()

    msg = {"id": 1, "type": "roommind/rooms/list"}
    with patch("custom_components.roommind.websocket_api.time.time", return_value=1_700_000_000.0):
        await _list_rooms(ws_hass, connection, msg)

    connection.send_result.assert_called_once_with(
        1,
        {
            "rooms": {},
            "rooms_without_area": [],
            "server_time": 1_700_000_000.0,
            "outdoor_temp": None,
            "outdoor_humidity": None,
            "vacation_active": False,
            "vacation_temp": None,
            "vacation_until": None,
            "hidden_rooms": [],
            "room_order": [],
            "group_by_floor": False,
            "control_mode": "mpc",
            "climate_control_active": True,
            "presence_enabled": False,
            "price_entity": "",
            "price_status": None,
            "presence_persons": [],
            "presence_away_action": "eco",
            "presence_clears_override": False,
            "schedule_off_action": "eco",
            "anyone_home": True,
            "central_plant": get_central_plant_settings({}),
            "house": get_house_settings({"house": {"enabled": True, "heat_c": 21.0, "cool_c": 24.0}}),
        },
    )


@pytest.mark.asyncio
async def test_save_room_creates_new(ws_hass, store, connection):
    """Saving a room with a new area_id creates the room with defaults."""
    await store.async_load()

    msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "living_room",
        "temperature_sensor": "sensor.living_room_temp",
        "occupancy_sensors": ["binary_sensor.living_room_motion"],
        "occupancy_mode": "follow_schedule",
    }
    await _save_room(ws_hass, connection, msg)

    connection.send_result.assert_called_once()
    call_args = connection.send_result.call_args
    assert call_args[0][0] == 2
    room = call_args[0][1]["room"]
    assert room["area_id"] == "living_room"
    assert room["temperature_sensor"] == "sensor.living_room_temp"
    assert room["occupancy_mode"] == "follow_schedule"
    # Defaults for fields not provided
    assert room["climate_mode"] == "auto"
    assert room["schedules"] == []
    assert room["schedule_selector_entity"] == ""
    assert room["comfort_temp"] == 21.0
    assert room["eco_temp"] == 17.0


@pytest.mark.asyncio
async def test_save_room_updates_existing(ws_hass, store, connection):
    """Saving a room with an existing area_id updates only the provided fields."""
    await store.async_load()

    # First create a room
    create_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "office",
        "temperature_sensor": "sensor.office_temp",
        "climate_mode": "heat_only",
    }
    await _save_room(ws_hass, connection, create_msg)
    connection.send_result.reset_mock()

    # Now update it - only change the temperature sensor
    update_msg = {
        "id": 3,
        "type": "roommind/rooms/save",
        "area_id": "office",
        "temperature_sensor": "sensor.office_temp_2",
    }
    await _save_room(ws_hass, connection, update_msg)

    connection.send_result.assert_called_once()
    call_args = connection.send_result.call_args
    assert call_args[0][0] == 3
    room = call_args[0][1]["room"]
    assert room["area_id"] == "office"
    assert room["temperature_sensor"] == "sensor.office_temp_2"
    # Fields not in update_msg should be preserved
    assert room["climate_mode"] == "heat_only"


@pytest.mark.asyncio
async def test_list_rooms_after_save(ws_hass, store, connection):
    """After saving a room, list_rooms includes it with live state."""
    await store.async_load()

    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "kitchen",
        "temperature_sensor": "sensor.kitchen_temp",
    }
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    # Add coordinator mock before listing so live state can be merged
    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {}  # No live data yet
    mock_coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    list_msg = {"id": 3, "type": "roommind/rooms/list"}
    await _list_rooms(ws_hass, connection, list_msg)

    connection.send_result.assert_called_once()
    call_args = connection.send_result.call_args
    assert call_args[0][0] == 3
    rooms = call_args[0][1]["rooms"]
    assert len(rooms) == 1
    assert "kitchen" in rooms
    room = rooms["kitchen"]
    assert room["temperature_sensor"] == "sensor.kitchen_temp"
    assert "live" in room
    assert room["live"]["mode"] == "idle"
    # No coordinator cycle yet: the plant block falls back to the stored settings
    assert call_args[0][1]["central_plant"] == get_central_plant_settings({})


@pytest.mark.asyncio
async def test_list_rooms_learning_paused_when_outdoor_unavailable(ws_hass, store, connection):
    """list_rooms surfaces learning_paused_reason='outdoor_unavailable' when
    the coordinator has no effective outdoor temperature (see #301)."""
    await store.async_load()
    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "kitchen",
        "temperature_sensor": "sensor.kitchen_temp",
    }
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {}
    mock_coordinator.async_request_refresh = AsyncMock()
    mock_coordinator.outdoor_temp = None
    mock_coordinator.outdoor_temp_effective = None
    mock_coordinator.outdoor_humidity = None
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    await _list_rooms(ws_hass, connection, {"id": 3, "type": "roommind/rooms/list"})

    rooms = connection.send_result.call_args[0][1]["rooms"]
    assert rooms["kitchen"]["live"]["learning_paused_reason"] == "outdoor_unavailable"


@pytest.mark.asyncio
async def test_list_rooms_learning_paused_none_when_outdoor_available(ws_hass, store, connection):
    """learning_paused_reason is None as long as the coordinator has any
    effective outdoor temperature (sensor or weather fallback)."""
    await store.async_load()
    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "kitchen",
        "temperature_sensor": "sensor.kitchen_temp",
    }
    await _save_room(ws_hass, connection, save_msg)
    await store.async_save_settings({"boost_applied_at": {"kitchen": 300}})
    connection.send_result.reset_mock()

    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {}
    mock_coordinator.async_request_refresh = AsyncMock()
    mock_coordinator.outdoor_temp = 5.0
    mock_coordinator.outdoor_temp_effective = 5.0
    mock_coordinator.outdoor_humidity = None
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    await _list_rooms(ws_hass, connection, {"id": 3, "type": "roommind/rooms/list"})

    rooms = connection.send_result.call_args[0][1]["rooms"]
    assert rooms["kitchen"]["live"]["learning_paused_reason"] is None
    assert rooms["kitchen"]["live"]["boost_applied_at"] == 300


@pytest.mark.asyncio
async def test_list_rooms_learning_paused_respects_learning_disabled(ws_hass, store, connection):
    """Rooms in ``learning_disabled_rooms`` do not get a paused reason — they
    are intentionally not learning regardless of outdoor availability."""
    await store.async_load()
    await store.async_save_settings({"learning_disabled_rooms": ["kitchen"]})
    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "kitchen",
        "temperature_sensor": "sensor.kitchen_temp",
    }
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {}
    mock_coordinator.async_request_refresh = AsyncMock()
    mock_coordinator.outdoor_temp = None
    mock_coordinator.outdoor_temp_effective = None
    mock_coordinator.outdoor_humidity = None
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    await _list_rooms(ws_hass, connection, {"id": 3, "type": "roommind/rooms/list"})

    rooms = connection.send_result.call_args[0][1]["rooms"]
    assert rooms["kitchen"]["live"]["learning_paused_reason"] is None


@pytest.mark.asyncio
async def test_list_rooms_learning_paused_for_managed_mode(ws_hass, store, connection):
    """Managed Mode rooms (no temperature_sensor) train on the device reading, so they
    pause without an outdoor temperature like every other room."""
    await store.async_load()
    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "kitchen",
        # no temperature_sensor → Managed Mode
    }
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {}
    mock_coordinator.async_request_refresh = AsyncMock()
    mock_coordinator.outdoor_temp = None
    mock_coordinator.outdoor_temp_effective = None
    mock_coordinator.outdoor_humidity = None
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    await _list_rooms(ws_hass, connection, {"id": 3, "type": "roommind/rooms/list"})

    rooms = connection.send_result.call_args[0][1]["rooms"]
    assert rooms["kitchen"]["live"]["learning_paused_reason"] == "outdoor_unavailable"


@pytest.mark.asyncio
async def test_list_rooms_learning_paused_none_for_outdoor_area(ws_hass, store, connection):
    """is_outdoor rooms disable learning entirely, so learning_paused_reason
    is None even when the outdoor temperature is missing."""
    await store.async_load()
    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "balcony",
        "temperature_sensor": "sensor.balcony_temp",
        "is_outdoor": True,
    }
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {}
    mock_coordinator.async_request_refresh = AsyncMock()
    mock_coordinator.outdoor_temp = None
    mock_coordinator.outdoor_temp_effective = None
    mock_coordinator.outdoor_humidity = None
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    await _list_rooms(ws_hass, connection, {"id": 3, "type": "roommind/rooms/list"})

    rooms = connection.send_result.call_args[0][1]["rooms"]
    assert rooms["balcony"]["live"]["learning_paused_reason"] is None


@pytest.mark.asyncio
async def test_list_rooms_outdoor_temp_uses_effective(ws_hass, store, connection):
    """The rooms/list response surfaces the effective outdoor temperature
    (sensor → weather fallback) rather than only the raw sensor reading."""
    await store.async_load()
    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {}
    mock_coordinator.async_request_refresh = AsyncMock()
    mock_coordinator.outdoor_temp = None
    mock_coordinator.outdoor_temp_effective = 8.5
    mock_coordinator.outdoor_humidity = None
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    await _list_rooms(ws_hass, connection, {"id": 1, "type": "roommind/rooms/list"})

    payload = connection.send_result.call_args[0][1]
    assert payload["outdoor_temp"] == 8.5


@pytest.mark.asyncio
async def test_save_room_display_name_roundtrip(ws_hass, store, connection):
    """display_name is persisted through save and returned in list."""
    await store.async_load()

    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "bedroom",
        "display_name": "Schlafzimmer OG",
    }
    await _save_room(ws_hass, connection, save_msg)

    call_args = connection.send_result.call_args
    room = call_args[0][1]["room"]
    assert room["display_name"] == "Schlafzimmer OG"
    connection.send_result.reset_mock()

    # Verify it comes back in list_rooms
    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {}
    mock_coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    list_msg = {"id": 3, "type": "roommind/rooms/list"}
    await _list_rooms(ws_hass, connection, list_msg)

    rooms = connection.send_result.call_args[0][1]["rooms"]
    assert rooms["bedroom"]["display_name"] == "Schlafzimmer OG"


@pytest.mark.asyncio
async def test_save_room_display_name_defaults_empty(ws_hass, store, connection):
    """Rooms created without display_name default to empty string."""
    await store.async_load()

    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "kitchen",
    }
    await _save_room(ws_hass, connection, save_msg)

    room = connection.send_result.call_args[0][1]["room"]
    assert room["display_name"] == ""


@pytest.mark.asyncio
async def test_save_room_with_schedules(ws_hass, store, connection):
    """Saving a room with schedules persists the reference."""
    await store.async_load()

    msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "bedroom",
        "temperature_sensor": "sensor.bedroom_temp",
        "schedules": [{"entity_id": "schedule.bedroom_heating"}],
        "schedule_selector_entity": "",
        "comfort_temp": 22.0,
        "eco_temp": 18.0,
    }
    await _save_room(ws_hass, connection, msg)

    connection.send_result.assert_called_once()
    room = connection.send_result.call_args[0][1]["room"]
    assert room["schedules"] == [{"entity_id": "schedule.bedroom_heating"}]
    assert room["schedule_selector_entity"] == ""
    assert room["comfort_temp"] == 22.0
    assert room["eco_temp"] == 18.0


@pytest.mark.asyncio
async def test_list_rooms_reports_resting_rooms_separately(ws_hass, store, connection):
    """A room whose area was deleted is not listed as a room, only as one without area (#494)."""
    await store.async_load()
    await store.async_save_room("kitchen", {})
    await store.async_save_room("old_room", {"display_name": "Old room"})
    registry = MagicMock()
    registry.async_list_areas.return_value = [SimpleNamespace(id="kitchen")]

    with patch("custom_components.roommind.utils.area_utils.ar.async_get", return_value=registry):
        await _list_rooms(ws_hass, connection, {"id": 1, "type": "roommind/rooms/list"})

    result = connection.send_result.call_args[0][1]
    assert set(result["rooms"]) == {"kitchen"}
    assert result["rooms_without_area"] == [{"area_id": "old_room", "display_name": "Old room"}]


@pytest.mark.asyncio
async def test_delete_room(ws_hass, store, connection):
    """Deleting a room removes it from the store."""
    await store.async_load()

    # First create a room
    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "garage",
        "temperature_sensor": "sensor.garage_temp",
    }
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    # Now delete it
    delete_msg = {
        "id": 3,
        "type": "roommind/rooms/delete",
        "area_id": "garage",
    }
    await _delete_room(ws_hass, connection, delete_msg)

    connection.send_result.assert_called_once_with(3, {"success": True})

    # Verify room is gone
    assert store.get_rooms() == {}


@pytest.mark.asyncio
async def test_delete_nonexistent_room_sends_error(ws_hass, store, connection):
    """Deleting a room that doesn't exist sends an error."""
    await store.async_load()

    delete_msg = {
        "id": 4,
        "type": "roommind/rooms/delete",
        "area_id": "nonexistent_area",
    }
    await _delete_room(ws_hass, connection, delete_msg)

    connection.send_error.assert_called_once()
    call_args = connection.send_error.call_args
    assert call_args[0][0] == 4
    assert call_args[0][1] == "not_found"


@pytest.mark.asyncio
async def test_save_room_minimal_only_area_id(ws_hass, store, connection):
    """Saving with only area_id creates a room with all defaults."""
    await store.async_load()

    msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "hallway",
    }
    await _save_room(ws_hass, connection, msg)

    connection.send_result.assert_called_once()
    room = connection.send_result.call_args[0][1]["room"]
    assert room["area_id"] == "hallway"
    assert room["temperature_sensor"] == ""
    assert room["humidity_sensor"] == ""
    assert room["climate_mode"] == "auto"
    assert room["schedules"] == []
    assert room["schedule_selector_entity"] == ""
    assert room["comfort_temp"] == 21.0
    assert room["eco_temp"] == 17.0


@pytest.mark.asyncio
async def test_save_room_notifies_coordinator(ws_hass, store, connection):
    """Saving a room notifies the coordinator via async_room_added."""
    await store.async_load()

    mock_coordinator = MagicMock()
    mock_coordinator.async_room_added = AsyncMock()
    # hasattr check used by _get_coordinator
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "balcony",
    }
    await _save_room(ws_hass, connection, msg)

    mock_coordinator.async_room_added.assert_called_once()
    room = mock_coordinator.async_room_added.call_args[0][0]
    assert room["area_id"] == "balcony"


@pytest.mark.asyncio
async def test_delete_room_notifies_coordinator(ws_hass, store, connection):
    """Deleting a room notifies the coordinator via async_room_removed."""
    await store.async_load()

    # First create the room
    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "cellar",
    }
    await _save_room(ws_hass, connection, save_msg)

    mock_coordinator = MagicMock()
    mock_coordinator.async_room_removed = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    delete_msg = {
        "id": 3,
        "type": "roommind/rooms/delete",
        "area_id": "cellar",
    }
    await _delete_room(ws_hass, connection, delete_msg)

    mock_coordinator.async_room_removed.assert_called_once_with("cellar")


@pytest.mark.asyncio
async def test_override_set_boost(ws_hass, store, connection):
    """Setting a boost override uses the room's comfort_temp."""
    await store.async_load()

    # Create room first
    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "living",
        "temperature_sensor": "sensor.living_temp",
        "comfort_temp": 22.0,
        "eco_temp": 17.0,
    }
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    msg = {
        "id": 3,
        "type": "roommind/override/set",
        "area_id": "living",
        "override_type": "boost",
        "duration": 2.0,
    }
    await _override_set(ws_hass, connection, msg)

    connection.send_result.assert_called_once_with(3, {"success": True})

    room = store.get_room("living")
    assert room["override_heat"] == 22.0
    assert room["override_cool"] == 24.0
    assert room["override_type"] == "boost"
    assert room["override_until"] is not None


@pytest.mark.asyncio
async def test_override_set_eco(ws_hass, store, connection):
    """Setting an eco override uses the room's eco_temp."""
    await store.async_load()

    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "bed",
        "comfort_temp": 22.0,
        "eco_temp": 16.0,
    }
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    msg = {
        "id": 3,
        "type": "roommind/override/set",
        "area_id": "bed",
        "override_type": "eco",
        "duration": 4.0,
    }
    await _override_set(ws_hass, connection, msg)

    connection.send_result.assert_called_once_with(3, {"success": True})
    room = store.get_room("bed")
    assert room["override_heat"] == 16.0
    assert room["override_cool"] == 27.0
    assert room["override_type"] == "eco"


@pytest.mark.asyncio
async def test_override_set_custom(ws_hass, store, connection):
    """Setting a custom override uses the provided temperature."""
    await store.async_load()

    save_msg = {"id": 2, "type": "roommind/rooms/save", "area_id": "office"}
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    msg = {
        "id": 3,
        "type": "roommind/override/set",
        "area_id": "office",
        "override_type": "custom",
        "heat": 21.0,
        "cool": 24.5,
        "duration": 1.0,
    }
    await _override_set(ws_hass, connection, msg)

    connection.send_result.assert_called_once_with(3, {"success": True})
    room = store.get_room("office")
    assert room["override_heat"] == 21.0
    assert room["override_cool"] == 24.5
    assert room["override_type"] == "custom"


@pytest.mark.asyncio
async def test_override_set_custom_without_temp_errors(ws_hass, store, connection):
    """Custom override without temperature sends an error."""
    await store.async_load()

    save_msg = {"id": 2, "type": "roommind/rooms/save", "area_id": "hall"}
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()
    connection.send_error.reset_mock()

    msg = {
        "id": 3,
        "type": "roommind/override/set",
        "area_id": "hall",
        "override_type": "custom",
        "duration": 1.0,
    }
    await _override_set(ws_hass, connection, msg)

    connection.send_error.assert_called_once()
    assert connection.send_error.call_args[0][1] == "invalid"


@pytest.mark.asyncio
async def test_override_clear(ws_hass, store, connection):
    """Clearing an override removes override fields."""
    await store.async_load()

    save_msg = {"id": 2, "type": "roommind/rooms/save", "area_id": "bath"}
    await _save_room(ws_hass, connection, save_msg)

    # Set override
    set_msg = {
        "id": 3,
        "type": "roommind/override/set",
        "area_id": "bath",
        "override_type": "boost",
        "duration": 2.0,
    }
    await _override_set(ws_hass, connection, set_msg)
    connection.send_result.reset_mock()

    # Clear it
    clear_msg = {
        "id": 4,
        "type": "roommind/override/clear",
        "area_id": "bath",
    }
    await _override_clear(ws_hass, connection, clear_msg)

    connection.send_result.assert_called_once_with(4, {"success": True})
    room = store.get_room("bath")
    assert room.get("override_heat") is None
    assert room.get("override_cool") is None
    assert room.get("override_until") is None
    assert room.get("override_type") is None


@pytest.mark.asyncio
async def test_override_set_without_duration_permanent(ws_hass, store, connection):
    """Setting override without duration creates a permanent override."""
    await store.async_load()

    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "perm",
        "comfort_temp": 22.0,
        "eco_temp": 17.0,
    }
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    msg = {
        "id": 3,
        "type": "roommind/override/set",
        "area_id": "perm",
        "override_type": "custom",
        "heat": 24.0,
    }
    await _override_set(ws_hass, connection, msg)

    connection.send_result.assert_called_once_with(3, {"success": True})
    room = store.get_room("perm")
    assert room["override_heat"] == 24.0
    assert room["override_cool"] is None
    assert room["override_until"] is None
    assert room["override_type"] == "custom"


@pytest.mark.asyncio
async def test_override_set_nonexistent_room_errors(ws_hass, store, connection):
    """Setting override on nonexistent room sends an error."""
    await store.async_load()

    msg = {
        "id": 2,
        "type": "roommind/override/set",
        "area_id": "nope",
        "override_type": "boost",
        "duration": 1.0,
    }
    await _override_set(ws_hass, connection, msg)

    connection.send_error.assert_called_once()
    assert connection.send_error.call_args[0][1] == "not_found"


@pytest.mark.asyncio
async def test_override_set_rejects_cool_below_heat(ws_hass, store, connection):
    """Custom override with cool < heat sends an error."""
    await store.async_load()

    save_msg = {"id": 2, "type": "roommind/rooms/save", "area_id": "study"}
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()
    connection.send_error.reset_mock()

    msg = {
        "id": 3,
        "type": "roommind/override/set",
        "area_id": "study",
        "override_type": "custom",
        "heat": 24.0,
        "cool": 21.0,
    }
    await _override_set(ws_hass, connection, msg)

    connection.send_error.assert_called_once()
    assert connection.send_error.call_args[0][1] == "invalid"
    room = store.get_room("study")
    assert room.get("override_heat") is None
    assert room.get("override_cool") is None


@pytest.mark.asyncio
async def test_override_set_boost_heat_only_no_cool(ws_hass, store, connection):
    """Boost in a heat_only room sets only the heat target."""
    await store.async_load()

    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "cellar",
        "climate_mode": "heat_only",
        "comfort_temp": 22.0,
    }
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    msg = {
        "id": 3,
        "type": "roommind/override/set",
        "area_id": "cellar",
        "override_type": "boost",
    }
    await _override_set(ws_hass, connection, msg)

    connection.send_result.assert_called_once_with(3, {"success": True})
    room = store.get_room("cellar")
    assert room["override_heat"] == 22.0
    assert room["override_cool"] is None


@pytest.mark.asyncio
async def test_override_set_custom_cool_only_forces_heat_none(ws_hass, store, connection):
    """Custom override in a cool_only room drops any provided heat target."""
    await store.async_load()

    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "server",
        "climate_mode": "cool_only",
    }
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    msg = {
        "id": 3,
        "type": "roommind/override/set",
        "area_id": "server",
        "override_type": "custom",
        "heat": 20.0,
        "cool": 23.0,
    }
    await _override_set(ws_hass, connection, msg)

    connection.send_result.assert_called_once_with(3, {"success": True})
    room = store.get_room("server")
    assert room["override_heat"] is None
    assert room["override_cool"] == 23.0


@pytest.mark.asyncio
async def test_save_room_with_multiple_schedules_and_selector(ws_hass, store, connection):
    """Saving with 2 schedules and a selector entity persists correctly."""
    await store.async_load()

    msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "wohnzimmer",
        "temperature_sensor": "sensor.wz_temp",
        "schedules": [
            {"entity_id": "schedule.morning"},
            {"entity_id": "schedule.evening"},
        ],
        "schedule_selector_entity": "input_boolean.schedule_toggle",
    }
    await _save_room(ws_hass, connection, msg)

    connection.send_result.assert_called_once()
    room = connection.send_result.call_args[0][1]["room"]
    assert room["schedules"] == [
        {"entity_id": "schedule.morning"},
        {"entity_id": "schedule.evening"},
    ]
    assert room["schedule_selector_entity"] == "input_boolean.schedule_toggle"


@pytest.mark.asyncio
async def test_list_rooms_includes_active_schedule_index(ws_hass, store, connection):
    """Verify active_schedule_index appears in live data from list_rooms."""
    await store.async_load()

    # Create a room
    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "buero",
        "temperature_sensor": "sensor.buero_temp",
        "schedules": [{"entity_id": "schedule.buero"}],
    }
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    # Add coordinator mock with live data including active_schedule_index
    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {
        "buero": {
            "current_temp": 20.0,
            "target_temp": 21.0,
            "mode": "heating",
            "active_schedule_index": 0,
        }
    }
    mock_coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    list_msg = {"id": 3, "type": "roommind/rooms/list"}
    await _list_rooms(ws_hass, connection, list_msg)

    connection.send_result.assert_called_once()
    call_args = connection.send_result.call_args
    rooms = call_args[0][1]["rooms"]
    assert "buero" in rooms
    live = rooms["buero"]["live"]
    assert "active_schedule_index" in live
    assert live["active_schedule_index"] == 0


@pytest.mark.asyncio
async def test_list_rooms_includes_cover_override_until(ws_hass, store, connection):
    """Verify cover_override_until appears in live data from list_rooms."""
    await store.async_load()

    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "schlafzimmer",
        "temperature_sensor": "sensor.sz_temp",
        "covers": ["cover.sz_rollo"],
    }
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {
        "schlafzimmer": {
            "current_temp": 20.0,
            "target_temp": 21.0,
            "mode": "idle",
            "cover_auto_paused": True,
            "cover_override_until": 1751700000.0,
            "cover_reason": "window_hold",
        }
    }
    mock_coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    list_msg = {"id": 3, "type": "roommind/rooms/list"}
    await _list_rooms(ws_hass, connection, list_msg)

    connection.send_result.assert_called_once()
    live = connection.send_result.call_args[0][1]["rooms"]["schlafzimmer"]["live"]
    assert live["cover_auto_paused"] is True
    assert live["cover_override_until"] == 1751700000.0
    assert live["cover_reason"] == "window_hold"


@pytest.mark.asyncio
async def test_save_room_with_window_sensors(ws_hass, store, connection):
    """Saving a room with window_sensors persists them; default is empty list."""
    await store.async_load()

    # Save a room WITH window_sensors
    msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "kitchen",
        "temperature_sensor": "sensor.kitchen_temp",
        "window_sensors": ["binary_sensor.kitchen_window"],
    }
    await _save_room(ws_hass, connection, msg)

    connection.send_result.assert_called_once()
    room = connection.send_result.call_args[0][1]["room"]
    assert room["window_sensors"] == ["binary_sensor.kitchen_window"]
    connection.send_result.reset_mock()

    # Save a room WITHOUT window_sensors — should default to []
    msg2 = {
        "id": 3,
        "type": "roommind/rooms/save",
        "area_id": "hallway",
    }
    await _save_room(ws_hass, connection, msg2)

    connection.send_result.assert_called_once()
    room2 = connection.send_result.call_args[0][1]["room"]
    assert room2["window_sensors"] == []


@pytest.mark.asyncio
async def test_list_rooms_includes_window_open(ws_hass, store, connection):
    """Verify window_open appears in live data from list_rooms."""
    await store.async_load()

    # Create a room
    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "wohnzimmer",
        "temperature_sensor": "sensor.wz_temp",
        "window_sensors": ["binary_sensor.wz_window"],
    }
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    # Add coordinator mock with live data including window_open
    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {
        "wohnzimmer": {
            "current_temp": 20.0,
            "target_temp": 21.0,
            "mode": "idle",
            "window_open": True,
            "active_schedule_index": 0,
        }
    }
    mock_coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    list_msg = {"id": 3, "type": "roommind/rooms/list"}
    await _list_rooms(ws_hass, connection, list_msg)

    connection.send_result.assert_called_once()
    call_args = connection.send_result.call_args
    rooms = call_args[0][1]["rooms"]
    assert "wohnzimmer" in rooms
    live = rooms["wohnzimmer"]["live"]
    assert "window_open" in live
    assert live["window_open"] is True


@pytest.mark.asyncio
async def test_get_settings_empty(ws_hass, store, connection):
    """Getting settings on a fresh store returns empty dict."""
    await store.async_load()

    msg = {"id": 10, "type": "roommind/settings/get"}
    await _get_settings(ws_hass, connection, msg)

    connection.send_result.assert_called_once_with(
        10,
        {
            "settings": {
                "house": get_house_settings({"house": {"enabled": True, "heat_c": 21.0, "cool_c": 24.0}}),
                "central_plant": get_central_plant_settings({}),
            }
        },
    )


@pytest.mark.asyncio
async def test_save_settings(ws_hass, store, connection):
    """Saving outdoor_temp_sensor persists and returns updated settings."""
    await store.async_load()

    msg = {
        "id": 11,
        "type": "roommind/settings/save",
        "outdoor_temp_sensor": "sensor.outdoor",
    }
    await _save_settings(ws_hass, connection, msg)

    connection.send_result.assert_called_once()
    result = connection.send_result.call_args[0][1]
    assert result["settings"]["outdoor_temp_sensor"] == "sensor.outdoor"


@pytest.mark.asyncio
async def test_save_settings_climate_control_refreshes_at_once(ws_hass, store, connection):
    """The panel toggle stops or resumes control and updates the switch entity now (#291)."""
    await store.async_load()
    coordinator = MagicMock()
    coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    await _save_settings(
        ws_hass, connection, {"id": 13, "type": "roommind/settings/save", "climate_control_active": False}
    )
    coordinator.async_request_refresh.assert_awaited_once()

    await _save_settings(ws_hass, connection, {"id": 14, "type": "roommind/settings/save", "outdoor_temp_sensor": "x"})
    coordinator.async_request_refresh.assert_awaited_once()


# ---------------------------------------------------------------------------
# Vacation mode tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_settings_vacation(ws_hass, store, connection):
    """Saving vacation fields persists and returns updated settings."""
    await store.async_load()

    until = 1771900000.0
    msg = {
        "id": 12,
        "type": "roommind/settings/save",
        "vacation_temp": 15.0,
        "vacation_until": until,
    }
    await _save_settings(ws_hass, connection, msg)

    connection.send_result.assert_called_once()
    result = connection.send_result.call_args[0][1]
    assert result["settings"]["vacation_temp"] == 15.0
    assert result["settings"]["vacation_until"] == until


@pytest.mark.asyncio
async def test_save_settings_vacation_clear(ws_hass, store, connection):
    """Setting vacation_until to None clears vacation mode."""
    await store.async_load()

    msg = {
        "id": 13,
        "type": "roommind/settings/save",
        "vacation_until": None,
    }
    await _save_settings(ws_hass, connection, msg)

    connection.send_result.assert_called_once()
    result = connection.send_result.call_args[0][1]
    assert result["settings"]["vacation_until"] is None


# ---------------------------------------------------------------------------
# Thermal model reset tests
# ---------------------------------------------------------------------------


def _make_coordinator_with_model(ws_hass):
    """Create a coordinator mock with public thermal API methods."""
    mock_coordinator = MagicMock()
    mock_coordinator.reset_thermal_room = MagicMock()
    mock_coordinator.reset_thermal_all = MagicMock(return_value=["room_a", "room_b"])
    mock_coordinator.boost_learning = MagicMock(return_value=42)
    mock_history = MagicMock()
    mock_history.remove_room = MagicMock()
    mock_coordinator.history_store = mock_history
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator
    return mock_coordinator


@pytest.mark.asyncio
async def test_thermal_reset_room(ws_hass, store, connection):
    """Resetting one room clears its model but keeps others."""
    await store.async_load()
    await store.async_save_room("room_a", {"thermostats": ["climate.a"]})
    await store.async_save_thermal_data({"room_a": {"n_samples": 10}, "room_b": {"n_samples": 5}})

    coordinator = _make_coordinator_with_model(ws_hass)

    msg = {"id": 20, "type": "roommind/thermal/reset", "area_id": "room_a"}
    await _thermal_reset(ws_hass, connection, msg)

    connection.send_result.assert_called_once_with(20, {"success": True})

    # Public API called for room_a
    coordinator.reset_thermal_room.assert_called_once_with("room_a")
    # History cleared for room_a
    coordinator.history_store.remove_room.assert_called_once_with("room_a")
    # Persisted thermal data cleared for room_a
    assert "room_a" not in store.get_thermal_data()
    assert "room_b" in store.get_thermal_data()


@pytest.mark.asyncio
async def test_thermal_reset_all(ws_hass, store, connection):
    """Resetting all rooms clears all models and history."""
    await store.async_load()
    await store.async_save_thermal_data({"room_a": {"n_samples": 10}, "room_b": {"n_samples": 5}})

    coordinator = _make_coordinator_with_model(ws_hass)

    msg = {"id": 21, "type": "roommind/thermal/reset_all"}
    await _thermal_reset_all(ws_hass, connection, msg)

    connection.send_result.assert_called_once_with(21, {"success": True})

    # Public API called
    coordinator.reset_thermal_all.assert_called_once()
    # History cleared for all rooms returned by reset_thermal_all
    assert coordinator.history_store.remove_room.call_count == 2
    # Persisted thermal data empty
    assert store.get_thermal_data() == {}


@pytest.mark.asyncio
async def test_thermal_reset_all_clears_history_of_rooms_without_model(ws_hass, store, connection):
    """An outdoor area learns no model but keeps a history: reset all deletes it too."""
    await store.async_load()
    await store.async_save_room("terrace", {"is_outdoor": True})

    coordinator = _make_coordinator_with_model(ws_hass)

    msg = {"id": 22, "type": "roommind/thermal/reset_all"}
    await _thermal_reset_all(ws_hass, connection, msg)

    removed = sorted(c.args[0] for c in coordinator.history_store.remove_room.call_args_list)
    assert removed == ["room_a", "room_b", "terrace"]


@pytest.mark.asyncio
async def test_thermal_reset_room_without_model(ws_hass, store, connection):
    """Resetting a stored room that has no model data still succeeds (idempotent)."""
    await store.async_load()
    await store.async_save_room("fresh", {"thermostats": ["climate.fresh"]})

    _make_coordinator_with_model(ws_hass)

    msg = {"id": 22, "type": "roommind/thermal/reset", "area_id": "fresh"}
    await _thermal_reset(ws_hass, connection, msg)

    connection.send_result.assert_called_once_with(22, {"success": True})


@pytest.mark.asyncio
@pytest.mark.parametrize("area_id", ["nonexistent", "../../tmp/x", "/config/x"])
async def test_thermal_reset_unknown_room_touches_nothing(ws_hass, store, connection, area_id):
    """An id that is no stored room is rejected before any model or file is touched."""
    await store.async_load()
    await store.async_save_thermal_data({"room_a": {"n_samples": 10}})

    coordinator = _make_coordinator_with_model(ws_hass)

    await _thermal_reset(ws_hass, connection, {"id": 23, "type": "roommind/thermal/reset", "area_id": area_id})

    connection.send_error.assert_called_once()
    assert connection.send_error.call_args.args[1] == "not_found"
    coordinator.reset_thermal_room.assert_not_called()
    coordinator.history_store.remove_room.assert_not_called()
    assert "room_a" in store.get_thermal_data()


# --- Mold risk settings tests ---


@pytest.mark.asyncio
async def test_save_settings_mold_fields(ws_hass, store, connection):
    """Mold detection/prevention settings should be accepted and persisted."""
    await store.async_load()

    msg = {
        "id": 30,
        "type": "roommind/settings/save",
        "mold_detection_enabled": True,
        "mold_humidity_threshold": 65.0,
        "mold_sustained_minutes": 15,
        "mold_notification_cooldown": 30,
        "mold_notifications_enabled": True,
        "mold_notification_targets": [
            {"entity_id": "notify.mobile_app_kevin", "person_entity": "person.kevin", "notify_when": "always"},
        ],
        "mold_prevention_enabled": True,
        "mold_prevention_intensity": "strong",
        "mold_prevention_notify_enabled": True,
        "mold_prevention_notify_targets": [],
    }
    await _save_settings(ws_hass, connection, msg)

    connection.send_result.assert_called_once()
    result = connection.send_result.call_args[0][1]
    settings = result["settings"]
    assert settings["mold_detection_enabled"] is True
    assert settings["mold_humidity_threshold"] == 65.0
    assert settings["mold_sustained_minutes"] == 15
    assert settings["mold_notification_cooldown"] == 30
    assert settings["mold_notifications_enabled"] is True
    assert len(settings["mold_notification_targets"]) == 1
    assert settings["mold_prevention_enabled"] is True
    assert settings["mold_prevention_intensity"] == "strong"


@pytest.mark.asyncio
async def test_save_settings_mold_f_rsi_range(ws_hass, store, connection):
    """mold_f_rsi is stored and limited to the realistic 0.6-0.95."""
    import voluptuous as vol

    await store.async_load()
    await _save_settings(ws_hass, connection, {"id": 30, "type": "roommind/settings/save", "mold_f_rsi": 0.65})
    assert connection.send_result.call_args[0][1]["settings"]["mold_f_rsi"] == 0.65

    schema = websocket_save_settings._ws_schema
    base = {"id": 1, "type": "roommind/settings/save"}
    assert schema({**base, "mold_f_rsi": "0.9"})["mold_f_rsi"] == 0.9
    with pytest.raises(vol.Invalid):
        schema({**base, "mold_f_rsi": 0.5})
    with pytest.raises(vol.Invalid):
        schema({**base, "mold_f_rsi": 1.0})


@pytest.mark.asyncio
async def test_save_settings_mold_partial_update(ws_hass, store, connection):
    """Updating only one mold field should not affect others (merge behavior)."""
    await store.async_load()

    # First save all fields
    msg1 = {
        "id": 31,
        "type": "roommind/settings/save",
        "mold_detection_enabled": True,
        "mold_humidity_threshold": 75.0,
    }
    await _save_settings(ws_hass, connection, msg1)
    connection.send_result.reset_mock()

    # Now save only one field
    msg2 = {
        "id": 32,
        "type": "roommind/settings/save",
        "mold_prevention_enabled": True,
    }
    await _save_settings(ws_hass, connection, msg2)

    result = connection.send_result.call_args[0][1]
    settings = result["settings"]
    # Original fields should still be there
    assert settings["mold_detection_enabled"] is True
    assert settings["mold_humidity_threshold"] == 75.0
    # New field should be set
    assert settings["mold_prevention_enabled"] is True


@pytest.mark.asyncio
async def test_compute_target_forecast_includes_mold_delta(ws_hass):
    """_compute_target_forecast should add mold_prevention_delta to all targets."""
    from custom_components.roommind.websocket_api import _compute_target_forecast

    room = {"comfort_temp": 21.0, "eco_temp": 17.0, "schedules": []}
    settings: dict = {}

    # Without delta
    forecast_base = await _compute_target_forecast(ws_hass, room, settings)
    assert forecast_base[0]["target_temp"] == 21.0

    # With delta
    forecast_mold = await _compute_target_forecast(
        ws_hass,
        room,
        settings,
        mold_prevention_delta=2.0,
    )
    assert forecast_mold[0]["target_temp"] == 23.0

    # All forecast points should have the delta applied
    for point in forecast_mold:
        assert point["target_temp"] == 23.0


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


def test_safe_float_valid():
    assert _safe_float("21.5") == 21.5


def test_safe_float_empty():
    assert _safe_float("") is None


def test_safe_float_none():
    assert _safe_float(None) is None


def test_safe_float_invalid():
    assert _safe_float("abc") is None


def test_csv_to_points_normal():
    """Converts CSV rows with string values to typed points."""
    rows = [
        {
            "timestamp": "1000.0",
            "room_temp": "21.5",
            "outdoor_temp": "5.0",
            "target_temp": "21.0",
            "mode": "heating",
            "predicted_temp": "21.3",
            "window_open": "False",
            "heating_power": "75.0",
        },
    ]
    points = _csv_to_points(rows)
    assert len(points) == 1
    p = points[0]
    assert p["ts"] == 1000.0
    assert p["room_temp"] == 21.5
    assert p["outdoor_temp"] == 5.0
    assert p["target_temp"] == 21.0
    assert p["mode"] == "heating"
    assert p["predicted_temp"] == 21.3
    assert p["window_open"] is False
    assert p["heating_power"] == 75.0
    # Rows from before the start columns have none, a counted zero stays zero
    assert p["heat_starts"] is None
    assert _csv_to_points([{**rows[0], "heat_starts": "0", "cool_starts": "2"}])[0]["cool_starts"] == 2
    assert p["room_temp_max"] is None
    bucket = _csv_to_points([{**rows[0], "room_temp_max": "21.9", "room_temp_min": "21.1"}])[0]
    assert (bucket["room_temp_max"], bucket["room_temp_min"]) == (21.9, 21.1)


def test_csv_to_points_window_open_true():
    rows = [
        {
            "timestamp": "1000",
            "room_temp": "21",
            "outdoor_temp": "5",
            "target_temp": "21",
            "mode": "idle",
            "predicted_temp": "",
            "window_open": "True",
            "heating_power": "",
        },
    ]
    points = _csv_to_points(rows)
    assert points[0]["window_open"] is True


def test_csv_to_points_skips_bad_timestamp():
    rows = [
        {
            "timestamp": "bad",
            "room_temp": "21",
            "outdoor_temp": "5",
            "target_temp": "21",
            "mode": "idle",
            "predicted_temp": "",
            "window_open": "",
            "heating_power": "",
        },
        {
            "timestamp": "1000",
            "room_temp": "21",
            "outdoor_temp": "5",
            "target_temp": "21",
            "mode": "idle",
            "predicted_temp": "",
            "window_open": "",
            "heating_power": "",
        },
    ]
    points = _csv_to_points(rows)
    assert len(points) == 1


def test_csv_to_points_empty():
    assert _csv_to_points([]) == []


# ---------------------------------------------------------------------------
# Analytics handler tests
# ---------------------------------------------------------------------------


def _make_mock_estimator(**overrides):
    """Build a mock ThermalEKF estimator with sensible defaults."""
    est = MagicMock()
    est._n_updates = overrides.get("n_updates", 200)
    est._last_update_at = overrides.get("last_update_at", 1_762_000_000.0)
    est._n_idle = overrides.get("n_idle", 120)
    est._n_heating = overrides.get("n_heating", 60)
    est._n_cooling = overrides.get("n_cooling", 20)
    est._applicable_modes = overrides.get("applicable_modes", {"idle", "heating"})
    est._P = overrides.get("P", [[0.01 * (i == j) for j in range(6)] for i in range(6)])
    est.confidence = overrides.get("confidence", 0.85)
    est.confidence_history = overrides.get(
        "confidence_history", [["2026-11-10", 0.8, 150, False, None, 0], ["2026-11-11", 0.85, 200, True, 0.061, 412]]
    )
    est.prediction_std.return_value = overrides.get("prediction_std", 0.3)
    rc = MagicMock()
    rc.Q_heat = overrides.get("Q_heat", 100.0)
    rc.to_dict.return_value = overrides.get("model_dict", {"alpha": 0.5})
    est.get_model.return_value = rc
    return est


def _make_analytics_coordinator(history_rows=None, estimator=None, rooms_live=None):
    """Build a mock coordinator for analytics tests."""
    coordinator = MagicMock()
    coordinator.rooms = rooms_live or {}
    coordinator.outdoor_temp = 5.0
    coordinator.outdoor_temp_effective = 5.0
    coordinator.outdoor_humidity = 60
    coordinator._weather_manager._outdoor_forecast = []
    coordinator._window_manager._paused = {}

    if history_rows is not None:
        hs = MagicMock()
        hs.read_detail.return_value = history_rows
        hs.read_history.return_value = []
        coordinator._history_store = hs
    else:
        coordinator._history_store = None

    from custom_components.roommind.control.thermal_model import RoomModelManager

    mgr = RoomModelManager()
    if estimator:
        mgr._estimators["room_a"] = estimator
    coordinator._model_manager = mgr

    return coordinator


@pytest.mark.asyncio
async def test_analytics_no_history_store(ws_hass, store, connection):
    """Analytics returns empty data when no history store exists."""
    await store.async_load()
    await store.async_save_room("room_a", {"thermostats": ["climate.trv1"]})

    coordinator = _make_analytics_coordinator(history_rows=None)
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    msg = {"id": 50, "type": "roommind/analytics/get", "area_id": "room_a"}
    await _get_analytics(ws_hass, connection, msg)

    result = connection.send_result.call_args[0][1]
    assert result["detail"] == []
    assert result["history"] == []


@pytest.mark.asyncio
async def test_analytics_with_range_key(ws_hass, store, connection):
    """Analytics reads history with max_age based on range key."""
    await store.async_load()
    await store.async_save_room("room_a", {"thermostats": ["climate.trv1"]})

    csv_rows = [
        {
            "timestamp": "1000",
            "room_temp": "21.0",
            "outdoor_temp": "5.0",
            "target_temp": "21.0",
            "mode": "idle",
            "predicted_temp": "",
            "window_open": "",
            "heating_power": "",
        },
    ]
    coordinator = _make_analytics_coordinator(history_rows=csv_rows)
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    msg = {"id": 51, "type": "roommind/analytics/get", "area_id": "room_a", "range": "24h"}
    await _get_analytics(ws_hass, connection, msg)

    result = connection.send_result.call_args[0][1]
    assert len(result["detail"]) == 1
    assert result["detail"][0]["ts"] == 1000.0
    assert "stats" in result
    assert "rhythm" in result
    # The run starts feed the figures only, the chart points do not carry them
    assert "heat_starts" not in result["detail"][0]
    assert "room_temp_max" not in result["detail"][0]
    # read_detail was called with max_age=86400
    coordinator._history_store.read_detail.assert_any_call("room_a", 86400)


@pytest.mark.asyncio
async def test_analytics_with_custom_timestamps(ws_hass, store, connection):
    """Analytics reads history with custom start/end timestamps."""
    await store.async_load()
    await store.async_save_room("room_a", {"thermostats": ["climate.trv1"]})

    csv_rows = [
        {
            "timestamp": "1500",
            "room_temp": "21.0",
            "outdoor_temp": "5.0",
            "target_temp": "21.0",
            "mode": "idle",
            "predicted_temp": "",
            "window_open": "",
            "heating_power": "",
        },
    ]
    coordinator = _make_analytics_coordinator(history_rows=csv_rows)
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    msg = {
        "id": 52,
        "type": "roommind/analytics/get",
        "area_id": "room_a",
        "start_ts": 1000.0,
        "end_ts": 2000.0,
    }
    await _get_analytics(ws_hass, connection, msg)

    result = connection.send_result.call_args[0][1]
    assert len(result["detail"]) == 1
    coordinator._history_store.read_detail.assert_any_call(
        "room_a",
        None,
        1000.0,
        2000.0,
    )


@pytest.mark.asyncio
async def test_analytics_no_estimator(ws_hass, store, connection):
    """Analytics returns empty model info when no estimator exists."""
    await store.async_load()
    await store.async_save_room("room_a", {"thermostats": ["climate.trv1"]})

    coordinator = _make_analytics_coordinator(history_rows=[])
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    msg = {"id": 53, "type": "roommind/analytics/get", "area_id": "room_a"}
    await _get_analytics(ws_hass, connection, msg)

    result = connection.send_result.call_args[0][1]
    assert result["model"] == {}


@pytest.mark.asyncio
async def test_analytics_with_estimator(ws_hass, store, connection):
    """Analytics includes model info when estimator exists."""
    await store.async_load()
    await store.async_save_room(
        "room_a",
        {
            "thermostats": ["climate.trv1"],
            "temperature_sensor": "sensor.temp",
        },
    )

    est = _make_mock_estimator()

    coordinator = _make_analytics_coordinator(history_rows=[], estimator=est)
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    msg = {"id": 54, "type": "roommind/analytics/get", "area_id": "room_a"}
    await _get_analytics(ws_hass, connection, msg)

    result = connection.send_result.call_args[0][1]
    model = result["model"]
    assert model["confidence"] == 0.85
    assert model["daily_history"] == [
        {"date": "2026-11-10", "confidence": 0.8, "mpc_active": False, "error": None, "error_steps": 0},
        {"date": "2026-11-11", "confidence": 0.85, "mpc_active": True, "error": 0.061, "error_steps": 412},
    ]
    assert model["n_samples"] == 200
    assert model["last_update_at"] == 1_762_000_000.0
    assert model["n_heating"] == 60
    assert model["n_cooling"] == 20
    assert "mpc_active" in model


@pytest.mark.asyncio
@pytest.mark.parametrize(("control_mode", "expected"), [("bangbang", False), ("mpc", True), (None, True)])
async def test_analytics_mpc_active_follows_control_mode(ws_hass, store, connection, control_mode, expected):
    """A ready model only counts as MPC-active when control_mode allows MPC (missing = mpc)."""
    await store.async_load()
    if control_mode is not None:
        await store.async_save_settings({"control_mode": control_mode})
    await store.async_save_room("room_a", {"temperature_sensor": "sensor.temp"})

    coordinator = _make_analytics_coordinator(history_rows=[], estimator=_make_mock_estimator(n_cooling=0))
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    await _get_analytics(ws_hass, connection, {"id": 57, "type": "roommind/analytics/get", "area_id": "room_a"})

    assert connection.send_result.call_args[0][1]["model"]["mpc_active"] is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("estimator_overrides", "expected"),
    [
        ({}, True),  # enough idle/active observations and a confident prediction
        ({"n_idle": 5}, False),  # the idle dynamics are not learned yet
        ({"prediction_std": 0.9}, False),  # prediction uncertainty above the MPC gate
    ],
)
async def test_analytics_mpc_active_follows_the_room_model_readiness(
    ws_hass, store, connection, estimator_overrides, expected
):
    """mpc_active tells the page whether the room's EKF is ready for the joint MPC."""
    await store.async_load()
    await store.async_save_room("room_a", {"temperature_sensor": "sensor.temp"})

    est = _make_mock_estimator(n_cooling=0, **estimator_overrides)
    coordinator = _make_analytics_coordinator(history_rows=[], estimator=est)
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    await _get_analytics(ws_hass, connection, {"id": 56, "type": "roommind/analytics/get", "area_id": "room_a"})

    assert connection.send_result.call_args[0][1]["model"]["mpc_active"] is expected


@pytest.mark.asyncio
async def test_analytics_managed_room_is_mpc_ready(ws_hass, store, connection):
    """A room without an external sensor trains on the plant's room reading, so it can be MPC-ready."""
    await store.async_load()
    await store.async_save_room("room_a", {"temperature_sensor": ""})  # Managed Mode

    coordinator = _make_analytics_coordinator(history_rows=[], estimator=_make_mock_estimator(n_cooling=0))
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    msg = {"id": 55, "type": "roommind/analytics/get", "area_id": "room_a"}
    await _get_analytics(ws_hass, connection, msg)

    assert connection.send_result.call_args[0][1]["model"]["mpc_active"] is True


@pytest.mark.asyncio
async def test_analytics_prediction_disabled(ws_hass, store, connection):
    """With prediction_enabled=False, pred_temps is empty."""
    await store.async_load()
    await store.async_save_room("room_a", {"thermostats": ["climate.trv1"]})
    await store.async_save_settings({"prediction_enabled": False})

    csv_rows = [
        {
            "timestamp": "1000",
            "room_temp": "21.0",
            "outdoor_temp": "5.0",
            "target_temp": "21.0",
            "mode": "idle",
            "predicted_temp": "",
            "window_open": "",
            "heating_power": "",
        },
    ]
    coordinator = _make_analytics_coordinator(history_rows=csv_rows)
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    msg = {"id": 56, "type": "roommind/analytics/get", "area_id": "room_a"}
    await _get_analytics(ws_hass, connection, msg)

    result = connection.send_result.call_args[0][1]
    # Forecast should have target_temp but predicted_temp should be None
    for f in result["forecast"]:
        assert f["predicted_temp"] is None


@pytest.mark.asyncio
async def test_analytics_forecast_grid_alignment(ws_hass, store, connection):
    """Forecast timestamps are snapped to 5-min grid."""
    await store.async_load()
    await store.async_save_room("room_a", {"thermostats": ["climate.trv1"]})

    coordinator = _make_analytics_coordinator(history_rows=[])
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    msg = {"id": 57, "type": "roommind/analytics/get", "area_id": "room_a"}
    await _get_analytics(ws_hass, connection, msg)

    result = connection.send_result.call_args[0][1]
    for f in result["forecast"]:
        assert f["ts"] % 300 == 0  # all timestamps on 5-min grid
        assert f["mode"] == "forecast"
        assert f["room_temp"] is None
        assert f["window_open"] is False


@pytest.mark.asyncio
async def test_analytics_mold_delta_from_live(ws_hass, store, connection):
    """Mold prevention delta is read from coordinator live state."""
    await store.async_load()
    await store.async_save_room(
        "room_a",
        {
            "thermostats": ["climate.trv1"],
            "comfort_temp": 21.0,
        },
    )

    coordinator = _make_analytics_coordinator(
        history_rows=[],
        rooms_live={"room_a": {"mold_prevention_delta": 2.0}},
    )
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    msg = {"id": 58, "type": "roommind/analytics/get", "area_id": "room_a"}
    await _get_analytics(ws_hass, connection, msg)

    result = connection.send_result.call_args[0][1]
    # Target forecast should include the mold delta
    assert result["forecast"][0]["target_temp"] == 23.0
    # Both sides go along so the chart can draw the target band ahead of now
    assert result["forecast"][0]["heat_target"] == 23.0
    assert result["forecast"][0]["cool_target"] is not None


@pytest.mark.asyncio
async def test_analytics_model_has_occupancy_sensors_true(ws_hass, store, connection):
    """Analytics model_info reports has_occupancy_sensors=True when room has occupancy sensors."""
    await store.async_load()
    await store.async_save_room(
        "room_a",
        {
            "thermostats": ["climate.trv1"],
            "occupancy_sensors": ["binary_sensor.room_a_occupancy"],
        },
    )

    est = _make_mock_estimator(model_dict={"Q_heat": 3.0, "Q_solar": 0.5, "Q_occupancy": 0.3})
    coordinator = _make_analytics_coordinator(history_rows=[], estimator=est)
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    msg = {"id": 60, "type": "roommind/analytics/get", "area_id": "room_a"}
    await _get_analytics(ws_hass, connection, msg)

    result = connection.send_result.call_args[0][1]
    assert result["model"]["has_occupancy_sensors"] is True
    assert "Q_occupancy" in result["model"]["model"]


@pytest.mark.asyncio
async def test_analytics_model_has_occupancy_sensors_false(ws_hass, store, connection):
    """Analytics model_info reports has_occupancy_sensors=False when no occupancy sensors configured."""
    await store.async_load()
    await store.async_save_room("room_a", {"thermostats": ["climate.trv1"]})

    est = _make_mock_estimator(model_dict={"Q_heat": 3.0, "Q_solar": 0.5, "Q_occupancy": 0.3})
    coordinator = _make_analytics_coordinator(history_rows=[], estimator=est)
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    msg = {"id": 61, "type": "roommind/analytics/get", "area_id": "room_a"}
    await _get_analytics(ws_hass, connection, msg)

    result = connection.send_result.call_args[0][1]
    assert result["model"]["has_occupancy_sensors"] is False
    assert "Q_occupancy" in result["model"]["model"]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_websocket_commands(hass):
    """async_register_websocket_commands registers all 16 commands."""
    from unittest.mock import patch

    from custom_components.roommind.websocket_api import async_register_websocket_commands

    with patch("custom_components.roommind.websocket_api.websocket_api.async_register_command") as mock_reg:
        async_register_websocket_commands(hass)
        assert mock_reg.call_count == 16
        registered = {call.args[1].__name__ for call in mock_reg.call_args_list}
        assert "websocket_house_save" in registered
        assert "websocket_house_clear_override" in registered


# ---------------------------------------------------------------------------
# Override set: cool_only climate mode paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_override_set_boost_cool_only_uses_comfort_cool(ws_hass, store, connection):
    """Boost override in cool_only room uses comfort_cool temperature."""
    await store.async_load()
    await store.async_save_room("room1", {"climate_mode": "cool_only", "comfort_cool": 26.0})
    connection.send_result.reset_mock()

    msg = {
        "id": 1,
        "type": "roommind/override/set",
        "area_id": "room1",
        "override_type": "boost",
        "duration": 1.0,
    }
    await _override_set(ws_hass, connection, msg)

    room = store.get_room("room1")
    assert room["override_heat"] is None
    assert room["override_cool"] == 26.0


@pytest.mark.asyncio
async def test_override_set_eco_cool_only_uses_eco_cool(ws_hass, store, connection):
    """Eco override in cool_only room uses eco_cool temperature."""
    await store.async_load()
    await store.async_save_room("room1", {"climate_mode": "cool_only", "eco_cool": 29.0})
    connection.send_result.reset_mock()

    msg = {
        "id": 1,
        "type": "roommind/override/set",
        "area_id": "room1",
        "override_type": "eco",
        "duration": 1.0,
    }
    await _override_set(ws_hass, connection, msg)

    room = store.get_room("room1")
    assert room["override_heat"] is None
    assert room["override_cool"] == 29.0


@pytest.mark.asyncio
async def test_override_set_triggers_coordinator_refresh(ws_hass, store, connection):
    """override/set notifies coordinator via async_request_refresh."""
    await store.async_load()
    await store.async_save_room("kitchen", {})

    mock_coordinator = MagicMock()
    mock_coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    msg = {
        "id": 1,
        "type": "roommind/override/set",
        "area_id": "kitchen",
        "override_type": "boost",
        "duration": 1.0,
    }
    await _override_set(ws_hass, connection, msg)

    mock_coordinator.async_request_refresh.assert_called_once()
    mock_coordinator.clear_override_seed.assert_called_once_with("kitchen")


@pytest.mark.asyncio
async def test_override_clear_nonexistent_room_errors(ws_hass, store, connection):
    """Clearing override on non-existent room sends an error."""
    await store.async_load()

    msg = {"id": 1, "type": "roommind/override/clear", "area_id": "does_not_exist"}
    await _override_clear(ws_hass, connection, msg)

    connection.send_error.assert_called_once()
    assert connection.send_error.call_args[0][1] == "not_found"


@pytest.mark.asyncio
async def test_override_clear_triggers_coordinator_refresh(ws_hass, store, connection):
    """override/clear notifies coordinator via async_request_refresh."""
    await store.async_load()
    await store.async_save_room("hall", {})

    mock_coordinator = MagicMock()
    mock_coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    msg = {"id": 1, "type": "roommind/override/clear", "area_id": "hall"}
    await _override_clear(ws_hass, connection, msg)

    mock_coordinator.async_request_refresh.assert_called_once()
    mock_coordinator.clear_override_seed.assert_called_once_with("hall")


# ---------------------------------------------------------------------------
# Boost learning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_boost_learning_no_coordinator_errors(ws_hass, store, connection):
    """boost_learning without coordinator sends an error."""
    from custom_components.roommind.websocket_api import websocket_boost_learning

    _boost_learning = websocket_boost_learning.__wrapped__.__wrapped__

    await store.async_load()
    # No coordinator in hass.data
    msg = {"id": 1, "type": "roommind/model/boost_learning", "area_id": "living_room"}
    await _boost_learning(ws_hass, connection, msg)

    connection.send_error.assert_called_once()
    assert connection.send_error.call_args[0][1] == "no_coordinator"


@pytest.mark.asyncio
async def test_boost_learning_success(ws_hass, store, connection):
    """boost_learning with coordinator boosts EKF and persists cooldown."""
    from custom_components.roommind.websocket_api import websocket_boost_learning

    _boost_learning = websocket_boost_learning.__wrapped__.__wrapped__

    await store.async_load()

    mock_coordinator = MagicMock()
    mock_coordinator.boost_learning = MagicMock(return_value=42)
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    msg = {"id": 1, "type": "roommind/model/boost_learning", "area_id": "living_room"}
    await _boost_learning(ws_hass, connection, msg)

    mock_coordinator.boost_learning.assert_called_once_with("living_room")
    connection.send_result.assert_called_once_with(1, {"success": True, "n_observations": 42})
    assert store.get_settings()["boost_applied_at"] == {"living_room": 42}


@pytest.mark.asyncio
async def test_boost_learning_without_model_errors_and_stores_nothing(ws_hass, store, connection):
    """A room without a model has nothing to boost: error, no anchor."""
    from custom_components.roommind.websocket_api import websocket_boost_learning

    await store.async_load()
    mock_coordinator = MagicMock()
    mock_coordinator.boost_learning = MagicMock(return_value=None)
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    msg = {"id": 1, "type": "roommind/model/boost_learning", "area_id": "living_room"}
    await websocket_boost_learning.__wrapped__.__wrapped__(ws_hass, connection, msg)

    assert connection.send_error.call_args[0][1] == "no_model"
    assert "boost_applied_at" not in store.get_settings()


@pytest.mark.asyncio
async def test_resets_and_delete_drop_the_boost_anchor(ws_hass, store, connection):
    """The anchor counts on the model; a reset or delete takes it along."""
    from custom_components.roommind.websocket_api import (
        websocket_delete_room,
        websocket_thermal_reset,
        websocket_thermal_reset_all,
    )

    await store.async_load()
    for area in ("a", "b", "c"):
        await store.async_save_room(area, {"thermostats": [f"climate.{area}"]})
    await store.async_save_settings({"boost_applied_at": {"a": 10, "b": 20, "c": 30}})
    ws_hass.data[DOMAIN]["coordinator"] = None

    await websocket_thermal_reset.__wrapped__.__wrapped__(ws_hass, connection, {"id": 1, "area_id": "b"})
    assert store.get_settings()["boost_applied_at"] == {"a": 10, "c": 30}
    await websocket_delete_room.__wrapped__.__wrapped__(ws_hass, connection, {"id": 2, "area_id": "a"})
    assert store.get_settings()["boost_applied_at"] == {"c": 30}
    await websocket_thermal_reset_all.__wrapped__.__wrapped__(ws_hass, connection, {"id": 3})
    assert store.get_settings()["boost_applied_at"] == {}
    # Nothing left: no further write
    await websocket_thermal_reset.__wrapped__.__wrapped__(ws_hass, connection, {"id": 4, "area_id": "c"})
    assert store.get_settings()["boost_applied_at"] == {}


# ── Cover schedule WS validation ───────────────────────────────────────


@pytest.mark.asyncio
async def test_save_room_with_cover_schedules(ws_hass, store, connection):
    """Cover schedules with valid entity_id are persisted."""
    await store.async_load()
    msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "sunroom",
        "cover_schedules": [{"entity_id": "schedule.cover_day"}],
        "cover_schedule_selector_entity": "input_boolean.cover_mode",
        "covers_night_close": True,
        "covers_night_position": 10,
        "covers_window_hold": True,
    }
    await _save_room(ws_hass, connection, msg)
    connection.send_result.assert_called_once()
    room = connection.send_result.call_args[0][1]["room"]
    assert room["covers_window_hold"] is True
    assert room["cover_schedules"] == [{"entity_id": "schedule.cover_day"}]
    assert room["cover_schedule_selector_entity"] == "input_boolean.cover_mode"
    assert room["covers_night_close"] is True
    assert room["covers_night_position"] == 10


def test_save_room_cover_night_position_validation():
    """covers_night_position validated by schema: 0-100 range."""
    import voluptuous as vol

    validator = vol.All(vol.Coerce(int), vol.Range(min=0, max=100))
    assert validator(0) == 0
    assert validator(100) == 100
    assert validator(50) == 50
    with pytest.raises(vol.Invalid):
        validator(150)
    with pytest.raises(vol.Invalid):
        validator(-1)


@pytest.mark.asyncio
async def test_save_room_with_is_outdoor(ws_hass, store, connection):
    """Round-trip: save a room with is_outdoor=True and verify it persists."""
    await store.async_load()

    msg = {
        "id": 10,
        "type": "roommind/rooms/save",
        "area_id": "terrasse",
        "is_outdoor": True,
    }
    await _save_room(ws_hass, connection, msg)

    connection.send_result.assert_called_once()
    call_args = connection.send_result.call_args
    room = call_args[0][1]["room"]
    assert room["is_outdoor"] is True

    # Verify it persists in the store
    stored = store.get_room("terrasse")
    assert stored["is_outdoor"] is True


@pytest.mark.asyncio
async def test_save_room_with_climate_control_enabled(ws_hass, store, connection):
    """Round-trip: climate_control_enabled persists through WS save."""
    await store.async_load()

    msg = {
        "id": 10,
        "type": "roommind/rooms/save",
        "area_id": "bedroom",
        "climate_control_enabled": False,
    }
    await _save_room(ws_hass, connection, msg)

    connection.send_result.assert_called_once()
    room = connection.send_result.call_args[0][1]["room"]
    assert room["climate_control_enabled"] is False

    stored = store.get_room("bedroom")
    assert stored["climate_control_enabled"] is False


def test_save_room_cover_deploy_threshold_rejects_negative():
    """covers_deploy_threshold rejects negative values."""
    import voluptuous as vol

    validator = vol.All(vol.Coerce(float), vol.Range(min=0))
    assert validator(0) == 0.0
    assert validator(1.5) == 1.5
    assert validator(5.0) == 5.0
    with pytest.raises(vol.Invalid):
        validator(-1.0)
    with pytest.raises(vol.Invalid):
        validator(-0.1)


# ---------------------------------------------------------------------------
# Self-assignment guard (#86)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("exterior_window_sensors", ["binary_sensor.roommind_test"]),
        ("window_sensors", ["binary_sensor.roommind_test"]),
        ("interior_door_sensors", ["binary_sensor.roommind_living_room_contact"]),
        ("temperature_sensor", "sensor.roommind_living_room_target_temp"),
        ("humidity_sensor", "sensor.roommind_living_room_mode"),
        ("covers", ["cover.roommind_living_room_auto"]),
        ("read_only_covers", ["cover.roommind_living_room_auto"]),
        ("schedules", [{"entity_id": "schedule.roommind_living_room"}]),
        ("cover_schedules", [{"entity_id": "schedule.roommind_living_room"}]),
        ("schedule_selector_entity", "input_boolean.roommind_living_room"),
        ("cover_schedule_selector_entity", "input_boolean.roommind_living_room"),
    ],
)
async def test_save_room_rejects_own_entities(ws_hass, store, connection, field, value):
    """Assigning RoomMind's own entities to a room is rejected."""
    await store.async_load()
    msg = {"id": 2, "type": "roommind/rooms/save", "area_id": "living_room", field: value}
    await _save_room(ws_hass, connection, msg)
    connection.send_error.assert_called_once()
    assert connection.send_error.call_args[0][1] == "invalid_entity"


@pytest.mark.asyncio
async def test_save_room_allows_normal_entities(ws_hass, store, connection):
    """Normal (non-RoomMind) entities are accepted."""
    await store.async_load()
    msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "living_room",
        "temperature_sensor": "sensor.living_room_temp",
    }
    await _save_room(ws_hass, connection, msg)
    connection.send_result.assert_called_once()
    connection.send_error.assert_not_called()


# ---------------------------------------------------------------------------
# Diagnostics WS endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_diagnostics_returns_full_structure(ws_hass, store, connection):
    """roommind/diagnostics/get returns full integration diagnostics."""
    await store.async_load()

    # Mock config_entries.async_entries to return a fake config entry
    mock_entry = MagicMock()
    mock_entry.entry_id = "test_entry_id"
    ws_hass.config_entries = MagicMock()
    ws_hass.config_entries.async_entries = MagicMock(return_value=[mock_entry])

    # Wire up coordinator
    coordinator = MagicMock()
    coordinator.rooms = {}
    coordinator.outdoor_temp = 10.0
    coordinator.outdoor_humidity = 60.0
    coordinator._previous_modes = {}
    coordinator._mode_on_since = {}
    coordinator._last_valid_temps = {}
    coordinator._residual_tracker = MagicMock()
    coordinator._model_manager = MagicMock()
    coordinator._model_manager._estimators = {}
    coordinator._window_manager = MagicMock()
    coordinator._window_manager._states = {}
    coordinator._cover_orchestrator = MagicMock()
    coordinator._cover_orchestrator._cover_managers = {}
    coordinator._heat_source_states = {}
    coordinator._weather_manager = MagicMock()
    coordinator._weather_manager._outdoor_forecast = []
    coordinator._history_store = None
    coordinator._compressor_manager = MagicMock()
    coordinator._compressor_manager._groups = {}
    coordinator._compressor_manager._member_states = {}
    coordinator._valve_manager = MagicMock()
    coordinator._valve_manager._cycling = {}
    coordinator._valve_manager._last_actuation = {}
    ws_hass.data[DOMAIN]["coordinator"] = coordinator
    ws_hass.config = MagicMock()
    ws_hass.config.units = MagicMock()
    ws_hass.config.units.temperature_unit = "°C"

    await _get_diagnostics(ws_hass, connection, {"id": 1, "type": "roommind/diagnostics/get"})

    connection.send_result.assert_called_once()
    result = connection.send_result.call_args[0][1]
    assert "integration" in result
    assert "settings" in result
    assert "rooms" in result
    assert "outdoor" in result
    assert "presence" in result
    assert result["integration"]["domain"] == "roommind"


@pytest.mark.asyncio
async def test_get_diagnostics_no_config_entry(ws_hass, store, connection):
    """roommind/diagnostics/get returns error when no config entry exists."""
    ws_hass.config_entries = MagicMock()
    ws_hass.config_entries.async_entries = MagicMock(return_value=[])

    await _get_diagnostics(ws_hass, connection, {"id": 1, "type": "roommind/diagnostics/get"})

    connection.send_error.assert_called_once()
    assert connection.send_error.call_args[0][1] == "not_found"


@pytest.mark.asyncio
async def test_covers_clear_override_unknown_room(ws_hass, store, connection):
    await store.async_load()
    msg = {"id": 7, "type": "roommind/covers/clear_override", "area_id": "nope"}
    await _covers_clear_override(ws_hass, connection, msg)
    connection.send_error.assert_called_once()
    assert connection.send_error.call_args[0][1] == "not_found"


@pytest.mark.asyncio
async def test_covers_clear_override_success(ws_hass, store, connection):
    await store.async_load()
    await store.async_save_room("lr", {"covers": ["cover.lr"]})
    coordinator = MagicMock()
    coordinator.clear_cover_override = MagicMock()
    coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = coordinator
    msg = {"id": 8, "type": "roommind/covers/clear_override", "area_id": "lr"}
    await _covers_clear_override(ws_hass, connection, msg)
    coordinator.clear_cover_override.assert_called_once_with("lr")
    coordinator.async_request_refresh.assert_awaited_once()
    connection.send_result.assert_called_once_with(8, {"success": True})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("override_type", "climate_mode", "expected"),
    [
        ("eco", "auto", (17.0, 27.5)),
        ("boost", "auto", (21.5, 24.0)),
        ("eco", "heat_only", (17.0, None)),
        ("eco", "cool_only", (None, 27.5)),
        ("boost", "heat_only", (21.5, None)),
    ],
)
async def test_override_preset_resolves_to_band_per_climate_mode(
    ws_hass, store, connection, override_type, climate_mode, expected
):
    """Presets keep both sides in auto (never a single point, #284) and mask the unusable side."""
    await store.async_load()
    await _save_room(
        ws_hass,
        connection,
        {
            "id": 2,
            "type": "roommind/rooms/save",
            "area_id": "bed",
            "climate_mode": climate_mode,
            "comfort_heat": 21.5,
            "comfort_cool": 24.0,
            "eco_heat": 17.0,
            "eco_cool": 27.5,
        },
    )
    connection.send_result.reset_mock()

    await _override_set(
        ws_hass,
        connection,
        {"id": 3, "type": "roommind/override/set", "area_id": "bed", "override_type": override_type},
    )

    room = store.get_room("bed")
    assert (room["override_heat"], room["override_cool"]) == expected


@pytest.mark.asyncio
async def test_analytics_model_reports_solar_gain_now_and_today(ws_hass, store, connection):
    """Q_solar is a full-sun value; the panel also gets the current and today's clear-sky effect."""
    await store.async_load()
    await store.async_save_room("room_a", {"thermostats": ["climate.trv1"]})

    est = _make_mock_estimator(model_dict={"Q_heat": 3.0, "Q_solar": 2.0, "Q_occupancy": 0.3})
    est.get_model.return_value.Q_solar = 2.0
    coordinator = _make_analytics_coordinator(
        history_rows=[], estimator=est, rooms_live={"room_a": {"shading_factor": 0.5}}
    )
    coordinator._current_q_solar = 0.4
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    msg = {"id": 62, "type": "roommind/analytics/get", "area_id": "room_a"}
    with patch("custom_components.roommind.services.analytics_service.clear_sky_peak_today", return_value=0.3):
        await _get_analytics(ws_hass, connection, msg)

    model = connection.send_result.call_args[0][1]["model"]
    assert model["solar_gain_now"] == pytest.approx(0.4)
    assert model["solar_gain_today_max"] == pytest.approx(0.6)


def test_clear_sky_peak_today_follows_the_season():
    from datetime import datetime

    from custom_components.roommind.services.analytics_service import clear_sky_peak_today

    summer = clear_sky_peak_today(52.5, 13.4, datetime(2026, 6, 21, 9, tzinfo=UTC).timestamp())
    winter = clear_sky_peak_today(52.5, 13.4, datetime(2026, 1, 12, 9, tzinfo=UTC).timestamp())
    assert 0.7 < summer < 1.0
    assert winter < 0.4 * summer


def test_occupancy_eco_schema_range():
    """occupancy_eco_delay is limited to 5..240 minutes (#180)."""
    import voluptuous as vol

    schema = websocket_save_room._ws_schema
    base = {"type": "roommind/rooms/save", "id": 1, "area_id": "office"}
    result = schema({**base, "occupancy_eco_enabled": True, "occupancy_eco_delay": "30"})
    assert result["occupancy_eco_enabled"] is True
    assert result["occupancy_eco_delay"] == 30
    for bad in (0, 4, 241):
        with pytest.raises(vol.Invalid):
            schema({**base, "occupancy_eco_delay": bad})


@pytest.mark.asyncio
async def test_save_room_occupancy_eco_roundtrip(ws_hass, store, connection):
    await store.async_load()
    msg = {
        "id": 10,
        "type": "roommind/rooms/save",
        "area_id": "office",
        "occupancy_eco_enabled": True,
        "occupancy_eco_delay": 5,
    }
    await _save_room(ws_hass, connection, msg)
    stored = store.get_room("office")
    assert stored["occupancy_eco_enabled"] is True
    assert stored["occupancy_eco_delay"] == 5


@pytest.mark.asyncio
async def test_list_rooms_forwards_occupancy_vacant(ws_hass, store, connection):
    """The dashboard shows the empty-room icon from rooms/list (#180)."""
    await store.async_load()
    await _save_room(ws_hass, connection, {"id": 2, "type": "roommind/rooms/save", "area_id": "office"})
    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {"office": {"occupancy_vacant": True}}
    mock_coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    await _list_rooms(ws_hass, connection, {"id": 3, "type": "roommind/rooms/list"})

    rooms = connection.send_result.call_args[0][1]["rooms"]
    assert rooms["office"]["live"]["occupancy_vacant"] is True


@pytest.mark.asyncio
async def test_save_room_preheat_switch_roundtrip(ws_hass, store, connection):
    """Per-room pre-heat switch (#478) is validated and stored."""
    import voluptuous as vol

    schema = websocket_save_room._ws_schema
    base = {"type": "roommind/rooms/save", "id": 1, "area_id": "bedroom"}
    assert schema({**base, "preheat_enabled": False})["preheat_enabled"] is False
    with pytest.raises(vol.Invalid):
        schema({**base, "preheat_enabled": "maybe"})

    await store.async_load()
    await _save_room(
        ws_hass, connection, {"id": 11, "type": "roommind/rooms/save", "area_id": "bedroom", "preheat_enabled": False}
    )
    assert store.get_room("bedroom")["preheat_enabled"] is False


@pytest.mark.asyncio
async def test_analytics_reports_the_price_unit(ws_hass, store, connection):
    """The price panel labels its axis with the source's unit (ct/kWh, EUR/kWh, ...)."""
    await store.async_load()
    await store.async_save_room("room_a", {"thermostats": ["climate.trv1"]})
    await store.async_save_settings({"price_entity": "sensor.strompreis"})
    price_state = MagicMock()
    price_state.attributes = {"unit_of_measurement": "ct/kWh"}
    ws_hass.states.get = MagicMock(side_effect=lambda eid: price_state if eid == "sensor.strompreis" else None)
    coordinator = _make_analytics_coordinator(history_rows=[])
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    msg = {"id": 64, "type": "roommind/analytics/get", "area_id": "room_a"}
    await _get_analytics(ws_hass, connection, msg)

    assert connection.send_result.call_args[0][1]["price_unit"] == "ct/kWh"


@pytest.mark.asyncio
async def test_list_rooms_includes_recent_temps(ws_hass, store, connection):
    """The dashboard cards draw the short series from rooms/list."""
    from custom_components.roommind.managers.recent_temps import RecentTemps

    await store.async_load()
    await _save_room(
        ws_hass,
        connection,
        {"id": 2, "type": "roommind/rooms/save", "area_id": "kitchen"},
    )
    connection.send_result.reset_mock()
    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {}
    mock_coordinator.async_request_refresh = AsyncMock()
    mock_coordinator.recent_temps = RecentTemps()
    mock_coordinator.recent_temps.add("kitchen", 1000.0, 20.5, 21.0)
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    await _list_rooms(ws_hass, connection, {"id": 3, "type": "roommind/rooms/list"})

    room = connection.send_result.call_args[0][1]["rooms"]["kitchen"]
    assert room["live"]["recent_temps"] == [[1000, 20.5, 21.0]]


async def _save_sched_room(ws_hass, store, connection):
    await store.async_load()
    await _save_room(
        ws_hass,
        connection,
        {"id": 2, "type": "roommind/rooms/save", "area_id": "sched", "comfort_temp": 22.0, "eco_temp": 17.0},
    )
    connection.send_result.reset_mock()


@pytest.mark.asyncio
async def test_override_until_schedule_change_uses_next_block(ws_hass, store, connection):
    """A tap on the dashboard card holds until the schedule's next block boundary."""
    await _save_sched_room(ws_hass, store, connection)
    nxt = time.time() + 5000
    with patch("custom_components.roommind.websocket_api.next_schedule_change", return_value=nxt):
        await _override_set(
            ws_hass,
            connection,
            {
                "id": 3,
                "type": "roommind/override/set",
                "area_id": "sched",
                "override_type": "eco",
                "duration": 4,
                "until_schedule_change": True,
            },
        )
    assert store.get_room("sched")["override_until"] == nxt


@pytest.mark.asyncio
async def test_override_until_schedule_change_falls_back_to_duration(ws_hass, store, connection):
    """Without a schedule the given duration applies, never an indefinite override."""
    await _save_sched_room(ws_hass, store, connection)
    before = time.time()
    with patch("custom_components.roommind.websocket_api.next_schedule_change", return_value=None):
        await _override_set(
            ws_hass,
            connection,
            {
                "id": 3,
                "type": "roommind/override/set",
                "area_id": "sched",
                "override_type": "boost",
                "duration": 4,
                "until_schedule_change": True,
            },
        )
    until = store.get_room("sched")["override_until"]
    assert before + 4 * 3600 - 1 <= until <= time.time() + 4 * 3600 + 1


@pytest.mark.asyncio
async def test_override_until_schedule_change_ignores_past_boundary(ws_hass, store, connection):
    await _save_sched_room(ws_hass, store, connection)
    with patch("custom_components.roommind.websocket_api.next_schedule_change", return_value=time.time() - 10):
        await _override_set(
            ws_hass,
            connection,
            {
                "id": 3,
                "type": "roommind/override/set",
                "area_id": "sched",
                "override_type": "boost",
                "duration": 1,
                "until_schedule_change": True,
            },
        )
    assert store.get_room("sched")["override_until"] > time.time() + 3000


@pytest.mark.asyncio
async def test_list_rooms_passes_price_shift_to_the_hero(ws_hass, store, connection):
    """The hero's price line reads the shift and level from rooms/list."""
    await store.async_load()
    await _save_room(
        ws_hass,
        connection,
        {"id": 2, "type": "roommind/rooms/save", "area_id": "kitchen"},
    )
    connection.send_result.reset_mock()
    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {
        "kitchen": {
            "price_level": "cheap",
            "price_shift_heat": 1.0,
            "price_shift_cool": 0.0,
            "preheat": {"mode": "heating", "target": 21.0, "at": 5},
        }
    }
    mock_coordinator.async_request_refresh = AsyncMock()
    mock_coordinator.recent_temps.get.return_value = []
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    await _list_rooms(ws_hass, connection, {"id": 3, "type": "roommind/rooms/list"})

    live = connection.send_result.call_args[0][1]["rooms"]["kitchen"]["live"]
    assert live["price_level"] == "cheap"
    assert live["price_shift_heat"] == 1.0
    assert live["price_shift_cool"] == 0.0
    assert live["preheat"] == {"mode": "heating", "target": 21.0, "at": 5}


@pytest.mark.asyncio
async def test_analytics_model_reports_which_gains_were_observed(ws_hass, store, connection):
    """A gain still at its start variance was never seen in the data: the tile must not show it as learned."""
    from custom_components.roommind.control.thermal_model import ThermalEKF

    await store.async_load()
    await store.async_save_room("room_a", {"temperature_sensor": "sensor.temp"})
    init = ThermalEKF._p_init_diag()
    P = [[0.0] * len(init) for _ in range(len(init))]
    for i, variance in enumerate(init):
        P[i][i] = variance
    P[6][6] = 0.1  # sun seen: its variance fell well below the start value
    est = _make_mock_estimator(model_dict={"Q_heat": 3.0, "Q_solar": 0.5, "Q_occupancy": 0.3}, P=P)
    coordinator = _make_analytics_coordinator(history_rows=[], estimator=est)
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    await _get_analytics(ws_hass, connection, {"id": 63, "type": "roommind/analytics/get", "area_id": "room_a"})

    model = connection.send_result.call_args[0][1]["model"]
    assert model["solar_learned"] is True
    assert model["occupancy_learned"] is False


def test_air_quality_schema():
    """CO2 threshold 800..5000 ppm, VOC threshold free in the sensor unit, ventilation fan/switch/input_boolean (#267)."""
    import voluptuous as vol

    schema = websocket_save_room._ws_schema
    base = {"type": "roommind/rooms/save", "id": 1, "area_id": "office"}
    result = schema(
        {
            **base,
            "co2_sensor": "sensor.office_co2",
            "co2_threshold": "1200",
            "voc_sensor": "sensor.office_voc",
            "voc_threshold": 250,
            "ventilation_entity": "fan.office",
        }
    )
    assert result["co2_threshold"] == 1200
    assert result["voc_threshold"] == 250.0
    assert schema({**base, "voc_threshold": None})["voc_threshold"] is None
    for eid in ("", "switch.kwl_boost", "input_boolean.luefter"):
        assert schema({**base, "ventilation_entity": eid})["ventilation_entity"] == eid
    for bad in (
        {"co2_threshold": 799},
        {"co2_threshold": 5001},
        {"voc_threshold": 0},
        {"ventilation_entity": "climate.ac"},
    ):
        with pytest.raises(vol.Invalid):
            schema({**base, **bad})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("co2_sensor", "sensor.roommind_living_room_target_temp"),
        ("ventilation_entity", "switch.roommind_living_room_climate_control"),
    ],
)
async def test_save_room_rejects_own_air_quality_entities(ws_hass, store, connection, field, value):
    await store.async_load()
    await _save_room(ws_hass, connection, {"id": 2, "type": "roommind/rooms/save", "area_id": "office", field: value})
    assert connection.send_error.call_args[0][1] == "invalid_entity"


@pytest.mark.asyncio
async def test_save_room_air_quality_defaults_and_roundtrip(ws_hass, store, connection):
    await store.async_load()
    await _save_room(ws_hass, connection, {"id": 2, "type": "roommind/rooms/save", "area_id": "office"})
    stored = store.get_room("office")
    assert (stored["co2_sensor"], stored["co2_threshold"], stored["voc_threshold"], stored["ventilation_entity"]) == (
        "",
        1000,
        None,
        "",
    )
    msg = {
        "id": 3,
        "type": "roommind/rooms/save",
        "area_id": "office",
        "co2_sensor": "sensor.co2",
        "co2_threshold": 900,
    }
    await _save_room(ws_hass, connection, msg)
    stored = store.get_room("office")
    assert stored["co2_sensor"] == "sensor.co2" and stored["co2_threshold"] == 900


@pytest.mark.asyncio
async def test_list_rooms_forwards_air_quality(ws_hass, store, connection):
    """Hint, sensor rows and the dashboard card read the air quality from rooms/list (#267)."""
    await store.async_load()
    await _save_room(ws_hass, connection, {"id": 2, "type": "roommind/rooms/save", "area_id": "office"})
    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {
        "office": {
            "co2": 1210.0,
            "voc": None,
            "air_quality_bad": True,
            "air_quality_reason": "co2",
            "ventilation_active": True,
        }
    }
    mock_coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    await _list_rooms(ws_hass, connection, {"id": 3, "type": "roommind/rooms/list"})

    live = connection.send_result.call_args[0][1]["rooms"]["office"]["live"]
    assert {k: live[k] for k in ("co2", "voc", "air_quality_bad", "air_quality_reason", "ventilation_active")} == {
        "co2": 1210.0,
        "voc": None,
        "air_quality_bad": True,
        "air_quality_reason": "co2",
        "ventilation_active": True,
    }


# ── Admin-only commands and non-admin room saves ───────────────────────


@pytest.mark.parametrize(
    "handler_name",
    [
        "websocket_delete_room",
        "websocket_save_settings",
        "websocket_thermal_reset",
        "websocket_thermal_reset_all",
        "websocket_boost_learning",
        "websocket_get_diagnostics",
    ],
)
def test_configuration_commands_need_an_admin(handler_name):
    """A non-admin user is turned away before the handler runs."""
    from homeassistant.exceptions import Unauthorized

    from custom_components.roommind import websocket_api as ws

    conn = MagicMock()
    conn.user.is_admin = False
    hass = MagicMock()
    with pytest.raises(Unauthorized):
        getattr(ws, handler_name)(hass, conn, {"id": 1, "type": "x", "area_id": "a"})
    hass.async_create_background_task.assert_not_called()


@pytest.fixture
def user_connection(connection):
    connection.user.is_admin = False
    return connection


@pytest.mark.asyncio
async def test_non_admin_saves_what_the_card_changes(ws_hass, store, user_connection):
    """Targets, climate mode and control of an existing room stay open for the dashboard card."""
    await store.async_load()
    await store.async_save_room("living", {"thermostats": ["climate.living"]})

    msg = {
        "id": 5,
        "type": "roommind/rooms/save",
        "area_id": "living",
        "comfort_heat": 22.0,
        "eco_cool": 28.0,
        "climate_mode": "heat_only",
        "climate_control_enabled": False,
    }
    await _save_room(ws_hass, user_connection, msg)

    user_connection.send_error.assert_not_called()
    room = store.get_room("living")
    assert room["comfort_heat"] == 22.0
    assert room["climate_mode"] == "heat_only"
    assert room["climate_control_enabled"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [
        {"ventilation_entity": "switch.anything"},
        {"covers": ["cover.blinds"]},
        {"comfort_heat": 22.0, "window_sensors": ["binary_sensor.w"]},
    ],
)
async def test_non_admin_cannot_change_the_room_setup(ws_hass, store, user_connection, fields):
    """Every other field stays with the admins, also when sent next to a card field."""
    await store.async_load()
    await store.async_save_room("living", {"thermostats": ["climate.living"]})
    before = dict(store.get_room("living"))

    await _save_room(ws_hass, user_connection, {"id": 6, "area_id": "living", **fields})

    assert user_connection.send_error.call_args.args[1] == "unauthorized"
    assert store.get_room("living") == before


@pytest.mark.asyncio
async def test_non_admin_cannot_create_a_room(ws_hass, store, user_connection):
    await store.async_load()

    await _save_room(ws_hass, user_connection, {"id": 7, "area_id": "new_room", "comfort_heat": 21.0})

    assert user_connection.send_error.call_args.args[1] == "unauthorized"
    assert store.get_room("new_room") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("area_id", ["nonexistent", "../../tmp/x"])
async def test_analytics_for_an_unknown_room_is_not_found(ws_hass, store, connection, area_id):
    """No history file is read for an id that is no stored room."""
    await store.async_load()

    with patch("custom_components.roommind.websocket_api.build_analytics_data") as build:
        await _get_analytics(ws_hass, connection, {"id": 8, "area_id": area_id})

    build.assert_not_called()
    assert connection.send_error.call_args.args[1] == "not_found"


@pytest.mark.parametrize("value", [4.9, 35.1, "nan", "inf", -40])
@pytest.mark.parametrize("field", ["comfort_temp", "eco_temp", "comfort_heat", "comfort_cool", "eco_heat", "eco_cool"])
def test_room_targets_stay_within_the_climate_entity_limits(field, value):
    """5-35 °C like the room climate entity; nan/inf never reach the controller."""
    import voluptuous as vol

    schema = websocket_save_room._ws_schema
    with pytest.raises(vol.Invalid):
        schema({"type": "roommind/rooms/save", "id": 1, "area_id": "a", field: value})
    assert schema({"type": "roommind/rooms/save", "id": 1, "area_id": "a", field: 35})[field] == 35.0


@pytest.mark.parametrize("side", ["heat", "cool"])
def test_custom_override_stays_within_the_climate_entity_limits(side):
    import voluptuous as vol

    schema = websocket_override_set._ws_schema
    base = {"type": "roommind/override/set", "id": 1, "area_id": "a", "override_type": "custom"}
    for value in (4.9, 35.1, "nan"):
        with pytest.raises(vol.Invalid):
            schema({**base, side: value})
    assert schema({**base, side: 5})[side] == 5.0


# ---------------------------------------------------------------------------
# Central HVAC: room schema and live payload
# ---------------------------------------------------------------------------


def _room_save_schema():
    return websocket_save_room._ws_schema


_ROOM_BASE = {"type": "roommind/rooms/save", "id": 1, "area_id": "office"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("devices", []),
        ("thermostats", []),
        ("acs", []),
        ("heating_system_type", "radiator"),
        ("follow_device_setpoint", True),
        ("heat_source_orchestration", True),
        ("valve_protection_exclude", []),
    ],
)
def test_room_schema_no_per_room_device_fields(field, value):
    """The Central Plant owns the hardware; the retired v1 device keys are rejected."""
    import voluptuous as vol

    with pytest.raises(vol.Invalid):
        _room_save_schema()({**_ROOM_BASE, field: value})


def test_room_schema_central_hvac_fields():
    """Occupancy, door and read-only cover fields of the v2 room model are accepted."""
    result = _room_save_schema()(
        {
            **_ROOM_BASE,
            "occupancy_mode": "guardrail_only",
            "occupancy_dwell_minutes": 5,
            "occupancy_eco_enabled": True,  # legacy alias next to the canonical mode
            "exterior_window_sensors": ["binary_sensor.window"],
            "interior_door_sensors": ["binary_sensor.door"],
            "interior_door_delay": 120,
            "door_closed_mix_factor": 0.25,
            "window_open_delay": 30,
            "window_close_delay": 45,
            "read_only_covers": ["sensor.cover_position", "cover.other_blind"],
        }
    )
    assert result["occupancy_mode"] == "guardrail_only"
    assert result["occupancy_dwell_minutes"] == 5.0
    assert result["occupancy_eco_enabled"] is True
    assert result["exterior_window_sensors"] == ["binary_sensor.window"]
    assert result["interior_door_sensors"] == ["binary_sensor.door"]
    assert result["interior_door_delay"] == 120
    assert result["door_closed_mix_factor"] == 0.25
    assert result["window_open_delay"] == 30
    assert result["window_close_delay"] == 45
    assert result["read_only_covers"] == ["sensor.cover_position", "cover.other_blind"]


@pytest.mark.parametrize(
    "changes",
    [
        {"occupancy_mode": "always"},
        {"occupancy_dwell_minutes": 15.5},
        {"occupancy_dwell_minutes": -1},
        {"window_open_delay": 3601},
        {"window_close_delay": -1},
        {"interior_door_delay": 601},
        {"door_closed_mix_factor": 0.05},
        {"door_closed_mix_factor": 1.1},
        {"read_only_covers": ["climate.air_conditioner"]},
        {"read_only_covers": ["cover.roommind_living_room_auto"]},
        {"read_only_covers": [5]},
    ],
)
def test_room_schema_central_hvac_field_validation(changes):
    import voluptuous as vol

    with pytest.raises(vol.Invalid):
        _room_save_schema()({**_ROOM_BASE, **changes})


def test_read_only_cover_validator_accepts_covers_and_position_sensors():
    from custom_components.roommind.websocket_api import _read_only_cover_entity

    assert _read_only_cover_entity("cover.blinds") == "cover.blinds"
    assert _read_only_cover_entity("sensor.blinds_position") == "sensor.blinds_position"


@pytest.mark.asyncio
async def test_save_room_central_hvac_fields_roundtrip(ws_hass, store, connection):
    """The new room fields persist, with the legacy window_sensors alias kept in sync."""
    await store.async_load()

    await _save_room(
        ws_hass,
        connection,
        {
            "id": 2,
            "type": "roommind/rooms/save",
            "area_id": "office",
            "occupancy_mode": "reactive_and_trend",
            "occupancy_dwell_minutes": 2.5,
            "exterior_window_sensors": ["binary_sensor.window"],
            "interior_door_sensors": ["binary_sensor.door"],
            "interior_door_delay": 90,
            "door_closed_mix_factor": 0.5,
            "window_open_delay": 60,
            "window_close_delay": 30,
            "read_only_covers": ["sensor.cover_position"],
        },
    )

    room = store.get_room("office")
    assert room["occupancy_mode"] == "reactive_and_trend"
    assert room["occupancy_eco_enabled"] is True
    assert room["occupancy_dwell_minutes"] == 2.5
    assert room["exterior_window_sensors"] == ["binary_sensor.window"]
    assert room["window_sensors"] == ["binary_sensor.window"]
    assert room["interior_door_sensors"] == ["binary_sensor.door"]
    assert room["interior_door_delay"] == 90
    assert room["door_closed_mix_factor"] == 0.5
    assert room["window_open_delay"] == 60
    assert room["window_close_delay"] == 30
    assert room["read_only_covers"] == ["sensor.cover_position"]


@pytest.mark.asyncio
async def test_save_room_window_sensors_legacy_alias_promotes(ws_hass, store, connection):
    """An old card sending window_sensors still fills the canonical exterior field."""
    await store.async_load()

    await _save_room(
        ws_hass,
        connection,
        {
            "id": 2,
            "type": "roommind/rooms/save",
            "area_id": "office",
            "window_sensors": ["binary_sensor.window"],
        },
    )

    room = store.get_room("office")
    assert room["exterior_window_sensors"] == ["binary_sensor.window"]
    assert room["window_sensors"] == ["binary_sensor.window"]


@pytest.mark.asyncio
async def test_list_rooms_forwards_the_central_hvac_room_fields(ws_hass, store, connection):
    """The dashboard reads the Room Priority and door/window state from rooms/list."""
    await store.async_load()
    await _save_room(
        ws_hass,
        connection,
        {"id": 2, "type": "roommind/rooms/save", "area_id": "office"},
    )
    connection.send_result.reset_mock()

    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {
        "office": {
            "effective_temp": 21.8,
            "priority_weight": 0.92,
            "occupancy_probability": 0.75,
            "occupancy_dwell": 1.0,
            "interior_door_open": True,
            "exterior_window_open": True,
            "fan_circulating": True,
            "beta_fan_rate": 1.4,
        }
    }
    mock_coordinator.data = {"rooms": {}}
    mock_coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    await _list_rooms(ws_hass, connection, {"id": 3, "type": "roommind/rooms/list"})

    live = connection.send_result.call_args[0][1]["rooms"]["office"]["live"]
    assert live["effective_temp"] == 21.8
    assert live["priority_weight"] == 0.92
    assert live["occupancy_probability"] == 0.75
    assert live["occupancy_dwell"] == 1.0
    assert live["interior_door_open"] is True
    assert live["exterior_window_open"] is True
    assert live["fan_circulating"] is True
    assert live["beta_fan_rate"] == 1.4


def test_build_room_live_defaults_without_a_coordinator():
    """Before the first coordinator cycle the live block reports the inactive defaults."""
    from custom_components.roommind.websocket_api import _build_room_live

    live = _build_room_live(None, "office", {}, {}, None)
    assert live["priority_weight"] == 0.0
    assert live["occupancy_probability"] == 0.0
    assert live["occupancy_dwell"] == 0.0
    assert live["interior_door_open"] is False
    assert live["exterior_window_open"] is False
    assert live["fan_circulating"] is False
    assert live["beta_fan_rate"] == 0.0
    assert live["effective_temp"] is None
    assert live["recent_temps"] == []


@pytest.mark.asyncio
async def test_list_rooms_forwards_the_live_central_plant(ws_hass, store, connection):
    """The plant block comes from the coordinator once it has run."""
    await store.async_load()
    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {}
    mock_coordinator.data = {
        "rooms": {},
        "central_plant": {
            "action": "heating",
            "house_ref_temp": 20.5,
            "guard_active": True,
            "guard_reason": "min_run",
        },
    }
    mock_coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    await _list_rooms(ws_hass, connection, {"id": 4, "type": "roommind/rooms/list"})

    plant = connection.send_result.call_args[0][1]["central_plant"]
    assert plant["action"] == "heating"
    assert plant["house_ref_temp"] == 20.5
    assert plant["guard_reason"] == "min_run"


def test_central_plant_live_handles_a_missing_block():
    from custom_components.roommind.websocket_api import _central_plant_live

    assert _central_plant_live(None) == {}
    coordinator = MagicMock()
    coordinator.data = {"rooms": {}}
    assert _central_plant_live(coordinator) == {}
    coordinator.data = {"central_plant": "not-a-dict"}
    assert _central_plant_live(coordinator) == {}


@pytest.mark.asyncio
async def test_delete_room_drops_it_from_the_settings_lists(ws_hass, store, connection):
    """A deleted room must not keep its slot in the dashboard lists."""
    await store.async_load()
    await store.async_save_room("old", {})
    await store.async_save_settings(
        {"hidden_rooms": ["old"], "room_order": ["old"], "learning_disabled_rooms": ["old"]}
    )

    await _delete_room(ws_hass, connection, {"id": 9, "type": "roommind/rooms/delete", "area_id": "old"})

    settings = store.get_settings()
    assert settings["hidden_rooms"] == []
    assert settings["room_order"] == []
    assert settings["learning_disabled_rooms"] == []


@pytest.mark.asyncio
async def test_pause_and_resume_a_resting_room(ws_hass, store, connection):
    """Deleting/creating an area pauses and resumes its stored room (#494)."""
    from custom_components.roommind.websocket_api import async_pause_room, async_resume_room

    await store.async_load()
    await store.async_save_room("old", {})
    coordinator = MagicMock()
    coordinator.async_room_paused = AsyncMock()
    coordinator.async_room_added = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    await async_pause_room(ws_hass, "old")
    coordinator.async_room_paused.assert_awaited_once_with("old")

    await async_resume_room(ws_hass, "old")
    coordinator.async_room_added.assert_awaited_once()

    # Unknown rooms touch nothing
    coordinator.async_room_paused.reset_mock()
    coordinator.async_room_added.reset_mock()
    await async_pause_room(ws_hass, "missing")
    await async_resume_room(ws_hass, "missing")
    coordinator.async_room_paused.assert_not_awaited()
    coordinator.async_room_added.assert_not_awaited()


# ---------------------------------------------------------------------------
# Central plant settings
# ---------------------------------------------------------------------------


def _settings_schema():
    return websocket_save_settings._ws_schema


def test_central_plant_schema_accepts_a_full_block():
    block = {
        "thermostat_entity": "climate.central",
        "thermostat_area_id": "hallway",
        "nudge_delta_c": 1.0,
        "max_nudge_headroom_c": 3.0,
        "fallback_heat_c": 15.0,
        "fallback_cool_c": 27.0,
        "min_deadband_c": 2.0,
        "circulate_fan_mode": "high",
        "idle_fan_mode": "auto",
        "min_run_minutes": 15,
        "min_off_minutes": 6,
        "min_fan_minutes": 12,
        "changeover_minutes": 45,
        "follow_thermostat_setpoint": False,
        "thermostat_override_hours": 3.5,
        "exterior_open_policy": "pause_always",
        "humidity_comfort_enabled": False,
        "dehumidify_rh_threshold": 65.0,
        "max_dehumidify_overcool_c": 1.2,
        "wet_coil_lockout_minutes": 30,
        "wet_coil_max_rh": 55.0,
        "price_shift_modes": "both",
    }
    result = _settings_schema()({"type": "roommind/settings/save", "id": 1, "central_plant": block})
    assert result["central_plant"]["thermostat_entity"] == "climate.central"
    assert result["central_plant"]["min_run_minutes"] == 15
    assert result["central_plant"]["price_shift_modes"] == "both"


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("thermostat_entity", "sensor.central"),
        ("thermostat_entity", "climate.roommind_hall"),
        ("thermostat_entity", 5),
        ("thermostat_area_id", 5),
        ("nudge_delta_c", 0.4),
        ("nudge_delta_c", 3.6),
        ("max_nudge_headroom_c", 6.1),
        ("fallback_heat_c", 4.9),
        ("fallback_heat_c", 25.1),
        ("fallback_cool_c", 17.9),
        ("fallback_cool_c", 35.1),
        ("min_deadband_c", 0.9),
        ("min_deadband_c", 5.1),
        ("min_run_minutes", 0),
        ("min_run_minutes", 61),
        ("min_off_minutes", 0),
        ("min_off_minutes", 31),
        ("min_fan_minutes", 0),
        ("min_fan_minutes", 61),
        ("changeover_minutes", 4),
        ("changeover_minutes", 181),
        ("thermostat_override_hours", -0.1),
        ("thermostat_override_hours", 24.1),
        ("exterior_open_policy", "open_always"),
        ("dehumidify_rh_threshold", 39.0),
        ("dehumidify_rh_threshold", 81.0),
        ("max_dehumidify_overcool_c", -0.1),
        ("max_dehumidify_overcool_c", 2.6),
        ("wet_coil_lockout_minutes", -1),
        ("wet_coil_lockout_minutes", 121),
        ("wet_coil_max_rh", 34.0),
        ("wet_coil_max_rh", 81.0),
        ("price_shift_modes", "heat_only"),
    ],
)
def test_central_plant_schema_rejects_out_of_range_values(key, bad):
    import voluptuous as vol

    with pytest.raises(vol.Invalid):
        _settings_schema()({"type": "roommind/settings/save", "id": 1, "central_plant": {key: bad}})


def test_central_plant_schema_allows_an_empty_thermostat_entity():
    result = _settings_schema()({"type": "roommind/settings/save", "id": 1, "central_plant": {"thermostat_entity": ""}})
    assert result["central_plant"]["thermostat_entity"] == ""


@pytest.mark.parametrize(
    "retired",
    [
        {"compressor_groups": []},
        {"valve_protection_enabled": True},
        {"valve_protection_interval_days": 7},
        {"coil_dry_enabled": True},
        {"coil_dry_mode": "dry"},
    ],
)
def test_settings_schema_no_longer_accepts_retired_plant_keys(retired):
    """Compressor groups, valve protection and coil drying moved into the Central Plant."""
    import voluptuous as vol

    with pytest.raises(vol.Invalid):
        _settings_schema()({"type": "roommind/settings/save", "id": 1, **retired})


@pytest.mark.asyncio
async def test_save_settings_merges_the_central_plant_block(ws_hass, store, connection):
    """A partial plant update keeps the keys it does not mention and refreshes the controller."""
    await store.async_load()
    await store.async_save_settings({"central_plant": {"thermostat_entity": "climate.central", "nudge_delta_c": 2.0}})
    coordinator = MagicMock()
    coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    await _save_settings(
        ws_hass,
        connection,
        {"id": 20, "type": "roommind/settings/save", "central_plant": {"min_run_minutes": 20}},
    )

    settings = connection.send_result.call_args[0][1]["settings"]
    assert settings["central_plant"]["thermostat_entity"] == "climate.central"
    assert settings["central_plant"]["nudge_delta_c"] == 2.0
    assert settings["central_plant"]["min_run_minutes"] == 20
    coordinator.async_request_refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_settings_rejects_an_inverted_fallback_band(ws_hass, store, connection):
    """fallback_cool_c must stay above fallback_heat_c, also against the stored side."""
    await store.async_load()
    await store.async_save_settings({"central_plant": {"fallback_cool_c": 20.0}})

    await _save_settings(
        ws_hass,
        connection,
        {"id": 21, "type": "roommind/settings/save", "central_plant": {"fallback_heat_c": 22.0}},
    )

    assert connection.send_error.call_args[0][1] == "invalid_central_plant"
    assert store.get_settings()["central_plant"]["fallback_heat_c"] == 16.5


@pytest.mark.asyncio
async def test_save_settings_accepts_an_ordered_fallback_band(ws_hass, store, connection):
    await store.async_load()

    await _save_settings(
        ws_hass,
        connection,
        {
            "id": 22,
            "type": "roommind/settings/save",
            "central_plant": {"fallback_heat_c": 15.0, "fallback_cool_c": 27.0},
        },
    )

    connection.send_error.assert_not_called()
    assert store.get_settings()["central_plant"]["fallback_cool_c"] == 27.0


def test_central_plant_band_validation_unit():
    from custom_components.roommind.websocket_api import _validate_central_plant_band

    assert _validate_central_plant_band({}, {"fallback_heat_c": 10.0, "fallback_cool_c": 20.0}) is True
    assert _validate_central_plant_band({}, {"fallback_heat_c": 25.0, "fallback_cool_c": 20.0}) is False
    # A malformed value is left to the schema; the band check must not crash on it
    assert _validate_central_plant_band({}, {"fallback_heat_c": "hot"}) is True


@pytest.mark.asyncio
async def test_get_settings_returns_the_resolved_central_plant(ws_hass, store, connection):
    await store.async_load()
    await store.async_save_settings({"central_plant": {"thermostat_entity": "climate.central"}})

    await _get_settings(ws_hass, connection, {"id": 23, "type": "roommind/settings/get"})

    settings = connection.send_result.call_args[0][1]["settings"]
    assert settings["central_plant"]["thermostat_entity"] == "climate.central"
    assert settings["central_plant"]["fallback_heat_c"] == 16.5


# ---------------------------------------------------------------------------
# Clear a Central Thermostat override
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plant_clear_override_calls_the_coordinator_hook(ws_hass, store, connection):
    await store.async_load()
    await store.async_save_settings(
        {"plant_override_heat": 22.0, "plant_override_cool": 24.0, "plant_override_until": 1234.0}
    )
    coordinator = MagicMock()
    coordinator.async_clear_plant_override = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    await websocket_plant_clear_override.__wrapped__(
        ws_hass, connection, {"id": 30, "type": "roommind/plant/clear_override"}
    )

    settings = store.get_settings()
    assert settings["plant_override_heat"] is None
    assert settings["plant_override_cool"] is None
    assert settings["plant_override_until"] is None
    coordinator.async_clear_plant_override.assert_awaited_once()
    connection.send_result.assert_called_once_with(30, {"cleared": True})


@pytest.mark.asyncio
async def test_plant_clear_override_falls_back_to_a_refresh(ws_hass, store, connection):
    """A coordinator without the hook still gets a refresh after the store was cleared."""
    await store.async_load()
    await store.async_save_settings({"plant_override_heat": 22.0})
    coordinator = MagicMock(spec=["async_request_refresh"])
    coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    await websocket_plant_clear_override.__wrapped__(
        ws_hass, connection, {"id": 31, "type": "roommind/plant/clear_override"}
    )

    assert store.get_settings()["plant_override_heat"] is None
    coordinator.async_request_refresh.assert_awaited_once()
    connection.send_result.assert_called_once_with(31, {"cleared": True})


@pytest.mark.asyncio
async def test_plant_clear_override_without_a_coordinator(ws_hass, store, connection):
    await store.async_load()
    await store.async_save_settings({"plant_override_cool": 24.0})

    await websocket_plant_clear_override.__wrapped__(
        ws_hass, connection, {"id": 32, "type": "roommind/plant/clear_override"}
    )

    assert store.get_settings()["plant_override_cool"] is None
    connection.send_result.assert_called_once_with(32, {"cleared": True})


# ---------------------------------------------------------------------------
# Whole-House settings block (WebSocket)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_settings_exposes_the_resolved_house_block(ws_hass, store, connection):
    """settings/get composes the house block alongside the central plant."""
    await store.async_load()
    await store.async_save_settings({"house": {"heat_c": 22.0, "cool_c": 25.0}})

    await _get_settings(ws_hass, connection, {"id": 40, "type": "roommind/settings/get"})

    settings = connection.send_result.call_args[0][1]["settings"]
    assert settings["house"]["heat_c"] == 22.0
    assert settings["house"]["cool_c"] == 25.0
    # Unspecified keys are still defaulted, not dropped by the partial save.
    assert settings["house"]["eco_heat_c"] == 17.0
    assert settings["house"]["climate_mode"] == "auto"


@pytest.mark.asyncio
async def test_save_settings_merges_a_partial_house_change(ws_hass, store, connection):
    """A partial house save keeps the other house keys."""
    await store.async_load()
    await store.async_save_settings({"house": {"heat_c": 20.0, "cool_c": 26.0, "climate_mode": "heat_only"}})

    await _save_settings(
        ws_hass, connection, {"id": 41, "type": "roommind/settings/save", "house": {"preheat_enabled": False}}
    )

    house = store.get_settings()["house"]
    assert house["preheat_enabled"] is False
    assert house["heat_c"] == 20.0
    assert house["cool_c"] == 26.0
    assert house["climate_mode"] == "heat_only"


@pytest.mark.asyncio
async def test_save_settings_refreshes_when_house_changes(ws_hass, store, connection):
    coordinator = MagicMock()
    coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = coordinator
    await store.async_load()

    await _save_settings(ws_hass, connection, {"id": 42, "type": "roommind/settings/save", "house": {"enabled": True}})

    coordinator.async_request_refresh.assert_awaited_once()


@pytest.mark.parametrize(
    "changes",
    [
        {"heat_c": 24.0, "cool_c": 22.0},  # inverted comfort band
        {"eco_heat_c": 20.0, "eco_cool_c": 18.0},  # inverted eco band
    ],
)
@pytest.mark.asyncio
async def test_save_settings_rejects_an_invalid_house_band(ws_hass, store, connection, changes):
    await store.async_load()
    before = store.get_settings()["house"]

    await _save_settings(ws_hass, connection, {"id": 43, "type": "roommind/settings/save", "house": changes})

    assert connection.send_error.call_args.args[1] == "invalid_house_band"
    assert store.get_settings()["house"] == before
    connection.send_result.assert_not_called()


@pytest.mark.asyncio
async def test_non_admin_settings_save_still_rejected(ws_hass, store, user_connection):
    """roommind/settings/save stays admin-only even for a house change."""
    from homeassistant.exceptions import Unauthorized

    await store.async_load()

    with pytest.raises(Unauthorized):
        websocket_save_settings(
            ws_hass,
            user_connection,
            {"id": 44, "type": "roommind/settings/save", "house": {"enabled": True}},
        )


@pytest.mark.asyncio
async def test_house_save_admin_writes_everything(ws_hass, store, connection):
    """An admin may write any house field, and the resolved block is echoed."""
    await store.async_load()
    coordinator = MagicMock()
    coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    await _house_save(
        ws_hass,
        connection,
        {
            "id": 50,
            "type": "roommind/house/save",
            "enabled": True,
            "heat_c": 20.0,
            "cool_c": 26.0,
            "climate_mode": "heat_only",
            "schedules": [{"entity_id": "schedule.house"}],
            "preheat_enabled": False,
        },
    )

    connection.send_error.assert_not_called()
    coordinator.async_request_refresh.assert_awaited_once()
    house = connection.send_result.call_args[0][1]["house"]
    assert house["enabled"] is True
    assert house["heat_c"] == 20.0
    assert house["cool_c"] == 26.0
    assert house["climate_mode"] == "heat_only"
    assert house["schedules"] == [{"entity_id": "schedule.house"}]
    assert house["preheat_enabled"] is False
    assert house["eco_heat_c"] == 17.0  # defaulted


@pytest.mark.asyncio
async def test_house_save_non_admin_targets_allowed(ws_hass, store, user_connection):
    user_connection.user.is_admin = False
    await store.async_load()

    await _house_save(
        ws_hass, user_connection, {"id": 51, "type": "roommind/house/save", "heat_c": 21.0, "cool_c": 25.0}
    )

    user_connection.send_error.assert_not_called()
    house = user_connection.send_result.call_args[0][1]["house"]
    assert house["heat_c"] == 21.0
    assert house["cool_c"] == 25.0


@pytest.mark.parametrize("field", ["climate_mode", "schedules", "enabled"])
@pytest.mark.asyncio
async def test_house_save_non_admin_structure_rejected(ws_hass, store, user_connection, field):
    await store.async_load()
    before = store.get_settings()["house"]
    values = {"climate_mode": "heat_only", "schedules": [{"entity_id": "schedule.house"}], "enabled": False}

    await _house_save(ws_hass, user_connection, {"id": 52, "type": "roommind/house/save", field: values[field]})

    assert user_connection.send_error.call_args.args[1] == "unauthorized"
    assert store.get_settings()["house"] == before
    user_connection.send_result.assert_not_called()


@pytest.mark.asyncio
async def test_house_save_target_alone_enables_the_layer(ws_hass, store, connection):
    await store.async_load()
    await store.async_save_settings({"house": {"enabled": False}})

    await _house_save(ws_hass, connection, {"id": 53, "type": "roommind/house/save", "heat_c": 20.0})

    house = store.get_settings()["house"]
    assert house["enabled"] is True
    assert house["heat_c"] == 20.0


@pytest.mark.asyncio
async def test_house_save_rejects_an_invalid_band_and_writes_nothing(ws_hass, store, connection):
    await store.async_load()
    before = store.get_settings()["house"]

    await _house_save(ws_hass, connection, {"id": 54, "type": "roommind/house/save", "heat_c": 25.0, "cool_c": 20.0})

    assert connection.send_error.call_args.args[1] == "invalid_house_band"
    assert store.get_settings()["house"] == before
    connection.send_result.assert_not_called()


@pytest.mark.asyncio
async def test_house_clear_override_clears_four_keys_and_leaves_the_rest(ws_hass, store, connection):
    """house/clear_override keeps the band and schedules intact; any user may call it."""
    await store.async_load()
    await store.async_save_settings(
        {
            "house": {
                "heat_c": 20.0,
                "cool_c": 26.0,
                "override_heat": 24.0,
                "override_cool": 28.0,
                "override_until": 1234.0,
                "override_type": "boost",
            }
        }
    )
    coordinator = MagicMock()
    coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    await _house_clear_override(ws_hass, connection, {"id": 55, "type": "roommind/house/clear_override"})

    house = store.get_settings()["house"]
    assert house["override_heat"] is None
    assert house["override_cool"] is None
    assert house["override_until"] is None
    assert house["override_type"] is None
    assert house["heat_c"] == 20.0
    assert house["cool_c"] == 26.0
    coordinator.async_request_refresh.assert_awaited_once()
    connection.send_result.assert_called_once_with(55, {"cleared": True})


@pytest.mark.asyncio
async def test_house_clear_override_without_a_coordinator(ws_hass, store, connection):
    await store.async_load()
    await store.async_save_settings({"house": {"override_heat": 24.0}})

    await _house_clear_override(ws_hass, connection, {"id": 56, "type": "roommind/house/clear_override"})

    assert store.get_settings()["house"]["override_heat"] is None
    connection.send_result.assert_called_once_with(56, {"cleared": True})


@pytest.mark.asyncio
async def test_list_rooms_includes_house_without_a_coordinator(ws_hass, store, connection):
    """With no coordinator the house block comes from the store settings."""
    await store.async_load()
    await store.async_save_settings({"house": {"heat_c": 19.0, "cool_c": 27.0}})

    await _list_rooms(ws_hass, connection, {"id": 60, "type": "roommind/rooms/list"})

    payload = connection.send_result.call_args[0][1]
    assert payload["house"]["heat_c"] == 19.0
    assert payload["house"]["cool_c"] == 27.0
    assert payload["house"]["eco_heat_c"] == 17.0
    # Existing keys untouched.
    assert "rooms" in payload
    assert "central_plant" in payload
    assert "server_time" in payload


@pytest.mark.asyncio
async def test_list_rooms_uses_the_live_house_block_from_the_coordinator(ws_hass, store, connection):
    """A published coordinator house block wins over the stored one."""
    await store.async_load()
    coordinator = MagicMock()
    coordinator.rooms = {}
    coordinator.data = {"rooms": {}, "house": {"enabled": True, "heat_c": 15.0, "cool_c": 30.0}}
    coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = coordinator

    await _list_rooms(ws_hass, connection, {"id": 61, "type": "roommind/rooms/list"})

    house = connection.send_result.call_args[0][1]["house"]
    assert house["heat_c"] == 15.0
    assert house["cool_c"] == 30.0


def test_house_live_handles_missing_and_non_dict_coordinator_data():
    from custom_components.roommind.websocket_api import _house_live

    store = MagicMock()
    store.get_settings.return_value = {"house": {"heat_c": 18.0}}
    assert _house_live(None, store)["heat_c"] == 18.0

    coordinator = MagicMock()
    coordinator.data = None
    assert _house_live(coordinator, store)["heat_c"] == 18.0

    coordinator.data = {"house": "not-a-dict"}
    assert _house_live(coordinator, store)["heat_c"] == 18.0


@pytest.mark.asyncio
async def test_save_room_accepts_follow_house_existing(ws_hass, store, connection):
    await store.async_load()
    await store.async_save_room("living", {"follow_house": True})

    await _save_room(
        ws_hass, connection, {"id": 70, "type": "roommind/rooms/save", "area_id": "living", "follow_house": False}
    )

    connection.send_error.assert_not_called()
    assert store.get_room("living")["follow_house"] is False


@pytest.mark.asyncio
async def test_save_room_accepts_follow_house_new(ws_hass, store, connection):
    await store.async_load()

    await _save_room(
        ws_hass, connection, {"id": 71, "type": "roommind/rooms/save", "area_id": "new", "follow_house": False}
    )

    connection.send_error.assert_not_called()
    assert store.get_room("new")["follow_house"] is False


@pytest.mark.asyncio
async def test_non_admin_may_toggle_follow_house(ws_hass, store, user_connection):
    await store.async_load()
    await store.async_save_room("living", {"follow_house": True})

    await _save_room(
        ws_hass, user_connection, {"id": 72, "type": "roommind/rooms/save", "area_id": "living", "follow_house": False}
    )

    user_connection.send_error.assert_not_called()
    assert store.get_room("living")["follow_house"] is False
