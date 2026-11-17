"""The coordinator surface the entities, the websocket API and diagnostics use."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.roommind.const import DOMAIN

from .conftest import Harness, make_room

pytestmark = pytest.mark.asyncio


async def test_the_price_signal_and_status_are_exposed(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    assert coordinator.price_signal is None
    assert coordinator.price_status() is None

    now = time.time()
    coordinator._price_manager._entity = "sensor.price"
    coordinator._price_manager.signal = MagicMock()
    coordinator._price_manager.status = MagicMock(return_value={"kind": "history"})

    assert coordinator.price_signal is coordinator._price_manager.signal
    assert coordinator.price_status() == {"kind": "history"}
    del now


async def test_the_history_store_is_exposed(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    assert coordinator.history_store is coordinator._history_store


async def test_the_override_seed_is_masked_for_the_grace_period(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    coordinator.note_override_seed("living_room")

    assert coordinator.is_override_seed_pending("living_room") is True
    masked = coordinator._without_pending_override_seed(
        {"area_id": "living_room", "override_heat": 23.0, "override_period": 1}
    )
    assert masked["override_heat"] is None

    coordinator.clear_override_seed("living_room")
    assert coordinator.is_override_seed_pending("living_room") is False
    plain = {"area_id": "living_room", "override_heat": 23.0}
    assert coordinator._without_pending_override_seed(plain) is plain


async def test_an_expired_seed_is_dropped(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    coordinator._override_seeded_at["living_room"] = time.monotonic() - 100.0

    plain = {"area_id": "living_room", "override_heat": 23.0}
    assert coordinator._without_pending_override_seed(plain) is plain
    assert "living_room" not in coordinator._override_seeded_at


async def test_clearing_the_plant_override_saves_and_refreshes(harness: Harness):
    coordinator = await harness.seed(
        {"living_room": make_room()},
        settings={"plant_override_heat": 23.0, "plant_override_until": time.time() + 3600},
    )
    coordinator.async_request_refresh = AsyncMock()

    await coordinator.async_clear_plant_override()

    settings = coordinator.hass.data[DOMAIN]["store"].get_settings()
    assert settings["plant_override_heat"] is None
    assert settings["plant_override_until"] is None
    coordinator.async_request_refresh.assert_awaited_once()


async def test_turning_a_room_off_disables_its_control(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    coordinator.async_request_refresh = AsyncMock()

    await coordinator.async_turn_room_off("living_room")

    assert coordinator.hass.data[DOMAIN]["store"].get_room("living_room")["climate_control_enabled"] is False
    coordinator.async_request_refresh.assert_awaited_once()


async def test_room_added_registers_every_entity_type(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    coordinator.async_request_refresh = AsyncMock()
    coordinator.async_add_entities = MagicMock()
    coordinator.async_add_climate_entities = MagicMock()
    coordinator.async_add_switch_entities = MagicMock()
    coordinator.async_add_binary_sensor_entities = MagicMock()
    room = make_room({"covers": ["cover.living"], "co2_sensor": "sensor.co2"})
    room["area_id"] = "new_room"

    await coordinator.async_room_added(room)

    coordinator.async_add_entities.assert_called_once()
    coordinator.async_add_climate_entities.assert_called_once()
    # Climate control switch + cover auto switch, then cover paused + air quality sensors.
    assert coordinator.async_add_switch_entities.call_count == 2
    assert coordinator.async_add_binary_sensor_entities.call_count == 2
    coordinator.async_request_refresh.assert_awaited_once()


async def test_room_added_without_callbacks_does_not_crash(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    coordinator.async_request_refresh = AsyncMock()

    await coordinator.async_room_added({"area_id": "new_room"})

    coordinator.async_request_refresh.assert_awaited_once()


async def test_room_added_twice_registers_its_entities_once(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    coordinator.async_request_refresh = AsyncMock()
    coordinator.async_add_entities = MagicMock()
    coordinator.async_add_climate_entities = MagicMock()
    coordinator.async_add_switch_entities = MagicMock()
    coordinator.async_add_binary_sensor_entities = MagicMock()

    await coordinator.async_room_added({"area_id": "new_room"})
    await coordinator.async_room_added({"area_id": "new_room"})

    coordinator.async_add_entities.assert_called_once()
    coordinator.async_add_climate_entities.assert_called_once()


async def test_room_removed_drops_model_occupancy_and_history(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    await harness.cycle()
    coordinator.async_request_refresh = AsyncMock()
    history = coordinator._history_store
    assert history is not None

    await coordinator.async_room_removed("living_room")

    assert coordinator._model_manager.get_room_ids() == []
    history.remove_room.assert_called_once_with("living_room")
    coordinator.async_request_refresh.assert_awaited_once()


async def test_a_paused_room_keeps_its_learned_state(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    await harness.cycle()
    model_ids = coordinator._model_manager.get_room_ids()

    await coordinator.async_room_paused("living_room")

    assert coordinator._model_manager.get_room_ids() == model_ids
    assert "living_room" not in coordinator._previous_modes


async def test_note_room_deleted_reports_whether_the_plant_served_it(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})

    assert coordinator.note_room_deleted("living_room", make_room()) is True
    assert coordinator.note_room_deleted("garden", make_room({"is_outdoor": True})) is False
    assert coordinator.note_room_deleted("off_room", make_room({"climate_control_enabled": False})) is False


async def test_orphaned_entities_are_removed(harness: Harness, fake_registry):
    await harness.seed({"living_room": make_room()})

    def entry(entity_id: str, platform: str, unique_id: str) -> Any:
        return SimpleNamespace(entity_id=entity_id, platform=platform, unique_id=unique_id)

    fake_registry.entities = {
        "sensor.own": entry("sensor.own", DOMAIN, f"{DOMAIN}_living_room_mode"),
        "sensor.foreign": entry("sensor.foreign", "other", "other_uid"),
        "sensor.unknown_uid": entry("sensor.unknown_uid", DOMAIN, f"{DOMAIN}_ghost_mode"),
        "switch.global": entry("switch.global", DOMAIN, f"{DOMAIN}_vacation"),
        "switch.cover": entry("switch.cover", DOMAIN, f"{DOMAIN}_living_room_cover_auto"),
        "binary_sensor.air": entry("binary_sensor.air", DOMAIN, f"{DOMAIN}_living_room_air_quality"),
    }
    coordinator = await harness.seed({"living_room": make_room()})

    coordinator.cleanup_orphaned_entities()

    assert set(fake_registry.removed) == {
        "sensor.unknown_uid",
        "switch.cover",
        "binary_sensor.air",
    }
    assert "sensor.own" in fake_registry.entities


async def test_orphaned_entities_are_kept_when_the_features_are_configured(harness: Harness, fake_registry):
    def entry(entity_id: str, unique_id: str) -> Any:
        return SimpleNamespace(entity_id=entity_id, platform=DOMAIN, unique_id=unique_id)

    fake_registry.entities = {
        "switch.cover": entry("switch.cover", f"{DOMAIN}_living_room_cover_auto"),
        "binary_sensor.air": entry("binary_sensor.air", f"{DOMAIN}_living_room_air_quality"),
    }
    coordinator = await harness.seed(
        {"living_room": make_room({"covers": ["cover.living"], "co2_sensor": "sensor.co2"})}
    )

    coordinator.cleanup_orphaned_entities()

    assert fake_registry.removed == []


async def test_the_plant_and_house_entities_survive_the_orphan_cleanup(harness: Harness, fake_registry):
    """Integration-owned entities belong to no room and must never be swept.

    The cleanup runs on every setup, right after the platforms are forwarded, so a
    global entity missing from the keep set vanishes from the state machine on each
    restart. The house thermostat did exactly that in 3.0.1.
    """
    global_uids = [
        f"{DOMAIN}_vacation",
        f"{DOMAIN}_climate_control",
        f"{DOMAIN}_central_action",
        f"{DOMAIN}_house_temperature",
        f"{DOMAIN}_wet_coil_lockout",
        f"{DOMAIN}_house_climate",
    ]
    fake_registry.entities = {
        f"sensor.global_{index}": SimpleNamespace(entity_id=f"sensor.global_{index}", platform=DOMAIN, unique_id=uid)
        for index, uid in enumerate(global_uids)
    }
    coordinator = await harness.seed({"living_room": make_room()})

    coordinator.cleanup_orphaned_entities()

    assert fake_registry.removed == []


async def test_every_global_entity_declares_a_protected_unique_id(harness: Harness):
    """The keep set must be the one place a new global entity is registered.

    A global entity whose unique_id is missing from ``GLOBAL_ENTITY_UNIQUE_IDS``
    is deleted by the orphan cleanup on the next restart, so the two cannot drift.
    """
    from custom_components.roommind.binary_sensor import RoomMindWetCoilLockoutBinarySensor
    from custom_components.roommind.climate import RoomMindHouseClimate
    from custom_components.roommind.const import GLOBAL_ENTITY_UNIQUE_IDS
    from custom_components.roommind.sensor import (
        RoomMindCentralActionSensor,
        RoomMindHouseTemperatureSensor,
    )
    from custom_components.roommind.switch import (
        RoomMindGlobalClimateControlSwitch,
        RoomMindVacationSwitch,
    )

    coordinator = await harness.seed({"living_room": make_room()})
    global_entities = [
        entity_class(coordinator)
        for entity_class in (
            RoomMindCentralActionSensor,
            RoomMindHouseTemperatureSensor,
            RoomMindVacationSwitch,
            RoomMindGlobalClimateControlSwitch,
            RoomMindWetCoilLockoutBinarySensor,
            RoomMindHouseClimate,
        )
    ]

    declared = {entity.unique_id for entity in global_entities}
    assert declared <= GLOBAL_ENTITY_UNIQUE_IDS
    # …and the keep set must not protect a unique_id no entity declares, or a
    # renamed entity would silently be swept because the old id stayed behind.
    assert declared == set(GLOBAL_ENTITY_UNIQUE_IDS)


async def test_forgetting_a_room_removes_its_registered_entities(harness: Harness, fake_registry):
    fake_registry.entities = {
        "sensor.mode": SimpleNamespace(
            entity_id="sensor.mode", platform=DOMAIN, unique_id=f"{DOMAIN}_living_room_mode"
        ),
        "sensor.other": SimpleNamespace(
            entity_id="sensor.other", platform=DOMAIN, unique_id=f"{DOMAIN}_other_room_mode"
        ),
    }
    coordinator = await harness.seed({"living_room": make_room()})

    coordinator._forget_room("living_room")

    assert fake_registry.removed == ["sensor.mode"]


async def test_reset_thermal_room_drops_model_and_charge(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    await harness.cycle()
    assert coordinator._model_manager.get_room_ids()

    coordinator.reset_thermal_room("living_room")

    assert coordinator._model_manager.get_room_ids() == []
    assert coordinator._rebound_tracker.charge("living_room") == 0.0


async def test_reset_thermal_all_rebuilds_the_model_manager(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    await harness.cycle()
    old_manager = coordinator._model_manager

    room_ids = coordinator.reset_thermal_all()

    assert room_ids == ["living_room"]
    assert coordinator._model_manager is not old_manager
    assert coordinator._ekf_training._model_manager is coordinator._model_manager
    assert coordinator._cover_orchestrator._model_manager is coordinator._model_manager
    assert coordinator._rebound_tracker.charge("living_room") == 0.0


async def test_reset_occupancy_room_saves_the_trends(harness: Harness):
    coordinator = await harness.seed(
        {"living_room": make_room()},
        occupancy_trends={"living_room": {"trend": {"weekday_bins": [1.0] * 96, "days_observed": 7.0}}},
    )
    coordinator.hass.async_create_task = MagicMock(side_effect=lambda coro: coro.close())

    coordinator.reset_occupancy_room("living_room")

    assert coordinator._occupancy_manager.trend_model("living_room") is not None
    coordinator.hass.async_create_task.assert_called_once()


async def test_reset_occupancy_room_of_a_deleted_room_still_resets(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})

    coordinator.reset_occupancy_room("gone_room")

    trend = coordinator._occupancy_manager.trend_model("gone_room")
    assert trend is not None
    assert trend.days_observed == 0.0
    coordinator.hass.async_create_task.assert_not_called()


async def test_boost_learning_returns_the_observation_count(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    await harness.cycle()

    assert coordinator.boost_learning("living_room") == 0
    assert coordinator.boost_learning("unknown_room") is None


async def test_clear_cover_override_resumes_automatic_control(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    coordinator._cover_orchestrator.clear_user_override = MagicMock()

    coordinator.clear_cover_override("living_room")

    coordinator._cover_orchestrator.clear_user_override.assert_called_once_with("living_room")
