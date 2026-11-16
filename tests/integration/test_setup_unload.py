"""Integration setup and unload of a config entry."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.roommind import (
    _async_check_version_mismatch,
    _async_migrate_storage,
    _migrate_storage_sync,
    async_setup,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.roommind.const import DOMAIN, VERSION

pytestmark = pytest.mark.asyncio


def _tmp_hass(plant_hass, tmp_path):
    plant_hass.data.setdefault(DOMAIN, {})
    plant_hass.config.path = MagicMock(side_effect=lambda *parts: str(tmp_path.joinpath(*parts)))
    plant_hass.http.async_register_static_paths = AsyncMock()
    plant_hass.config_entries.async_forward_entry_setups = AsyncMock()
    plant_hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)
    return plant_hass


async def test_setup_registers_the_websocket_commands(plant_hass):
    hass = _tmp_hass(plant_hass, Path("/tmp/roommind-setup"))
    with patch("custom_components.roommind.async_register_websocket_commands") as register:
        assert await async_setup(hass, {}) is True
    register.assert_called_once_with(hass)
    assert DOMAIN in hass.data


async def test_setup_entry_builds_the_store_and_the_coordinator(plant_hass, tmp_path, mock_config_entry):
    hass = _tmp_hass(plant_hass, tmp_path)
    coordinator = MagicMock()
    coordinator.async_config_entry_first_refresh = AsyncMock()
    coordinator.cleanup_orphaned_entities = MagicMock()
    with (
        patch("custom_components.roommind.RoomMindCoordinator", return_value=coordinator) as coordinator_cls,
        patch("custom_components.roommind.async_schedule_deprecation_check"),
        patch("custom_components.roommind._async_register_panel", AsyncMock()),
        patch("custom_components.roommind._async_register_card", AsyncMock()),
        patch("custom_components.roommind._async_check_version_mismatch", AsyncMock()),
    ):
        assert await async_setup_entry(hass, mock_config_entry) is True

    coordinator_cls.assert_called_once()
    coordinator.async_config_entry_first_refresh.assert_awaited_once()
    coordinator.cleanup_orphaned_entities.assert_called_once()
    hass.config_entries.async_forward_entry_setups.assert_awaited_once()
    assert hass.data[DOMAIN]["coordinator"] is coordinator
    assert hass.data[DOMAIN][mock_config_entry.entry_id] is coordinator
    assert "store" in hass.data[DOMAIN]


async def test_setup_entry_then_unload_cleans_the_data(plant_hass, tmp_path, mock_config_entry):
    hass = _tmp_hass(plant_hass, tmp_path)
    coordinator = MagicMock()
    coordinator.async_config_entry_first_refresh = AsyncMock()
    coordinator.cleanup_orphaned_entities = MagicMock()
    coordinator.async_release_ventilation = AsyncMock()
    with (
        patch("custom_components.roommind.RoomMindCoordinator", return_value=coordinator),
        patch("custom_components.roommind.async_schedule_deprecation_check"),
        patch("custom_components.roommind._async_register_panel", AsyncMock()),
        patch("custom_components.roommind._async_register_card", AsyncMock()),
        patch("custom_components.roommind._async_check_version_mismatch", AsyncMock()),
    ):
        await async_setup_entry(hass, mock_config_entry)
        with patch("custom_components.roommind.async_remove_panel") as remove_panel:
            assert await async_unload_entry(hass, mock_config_entry) is True

    coordinator.async_release_ventilation.assert_awaited_once()
    hass.config_entries.async_unload_platforms.assert_awaited_once()
    assert mock_config_entry.entry_id not in hass.data[DOMAIN]
    assert "coordinator" not in hass.data[DOMAIN]
    # The shared store outlives the entry, so the panel stays for the next setup.
    remove_panel.assert_not_called()


async def test_unload_that_fails_keeps_running(plant_hass, tmp_path, mock_config_entry):
    hass = _tmp_hass(plant_hass, tmp_path)
    hass.config_entries.async_unload_platforms = AsyncMock(return_value=False)
    hass.data[DOMAIN] = {"store": MagicMock(), "coordinator": MagicMock()}

    with patch("custom_components.roommind.async_remove_panel") as remove_panel:
        assert await async_unload_entry(hass, mock_config_entry) is False

    remove_panel.assert_not_called()
    assert "store" in hass.data[DOMAIN]


async def test_unload_releases_ventilation_and_tolerates_a_failure(plant_hass, mock_config_entry):
    hass = _tmp_hass(plant_hass, Path("/tmp/roommind-unload"))
    coordinator = MagicMock()
    coordinator.async_release_ventilation = AsyncMock(side_effect=RuntimeError("ventilation down"))
    hass.data[DOMAIN] = {
        "store": MagicMock(),
        "coordinator": coordinator,
        mock_config_entry.entry_id: coordinator,
    }
    hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)

    assert await async_unload_entry(hass, mock_config_entry) is True
    assert "coordinator" not in hass.data[DOMAIN]


async def test_unload_removes_the_panel_when_nothing_is_left(plant_hass, mock_config_entry):
    hass = _tmp_hass(plant_hass, Path("/tmp/roommind-unload"))
    hass.data[DOMAIN] = {mock_config_entry.entry_id: MagicMock()}
    hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)

    with patch("custom_components.roommind.async_remove_panel") as remove_panel:
        assert await async_unload_entry(hass, mock_config_entry) is True

    remove_panel.assert_called_once_with(hass, "roommind")


async def test_storage_migration_renames_the_old_files(plant_hass, tmp_path):
    hass = _tmp_hass(plant_hass, tmp_path)
    storage = tmp_path / ".storage"
    storage.mkdir()
    (storage / "roomsense").write_text(json.dumps({"key": "roomsense", "data": {}}))
    old_history = storage / "roomsense_history"
    old_history.mkdir()
    (old_history / "living_room_detail.csv").write_text("timestamp\n1\n")

    await _async_migrate_storage(hass)

    assert (storage / "roommind").exists()
    assert not (storage / "roomsense").exists()
    assert json.loads((storage / "roommind").read_text())["key"] == "roommind"
    assert (storage / "roommind_history" / "living_room_detail.csv").exists()
    assert not old_history.exists()


async def test_storage_migration_keeps_the_new_file_and_merges_history(plant_hass, tmp_path):
    hass = _tmp_hass(plant_hass, tmp_path)
    storage = tmp_path / ".storage"
    storage.mkdir()
    (storage / "roommind").write_text(json.dumps({"key": "roommind", "data": {}}))
    old_history = storage / "roomsense_history"
    old_history.mkdir()
    (old_history / "living_room_detail.csv").write_text("timestamp\n1\n")
    (old_history / "bedroom_detail.csv").write_text("timestamp\n3\n")
    new_history = storage / "roommind_history"
    new_history.mkdir()
    (new_history / "living_room_detail.csv").write_text("timestamp\n2\n")

    await _async_migrate_storage(hass)

    assert json.loads((storage / "roommind").read_text())["key"] == "roommind"
    rows = (new_history / "living_room_detail.csv").read_text().splitlines()
    assert rows == ["timestamp", "1", "2"]
    # A CSV only the old directory had is moved over as it is.
    assert (new_history / "bedroom_detail.csv").read_text() == "timestamp\n3\n"
    assert not old_history.exists()


async def test_storage_migration_sync_tolerates_a_corrupt_file(plant_hass, tmp_path):
    storage = tmp_path / ".storage"
    storage.mkdir()
    (storage / "roommind").write_text("not json")

    _migrate_storage_sync(storage)

    assert (storage / "roommind").read_text() == "not json"


async def test_a_version_mismatch_creates_the_restart_issue(plant_hass, monkeypatch):
    hass = _tmp_hass(plant_hass, Path("/tmp/roommind-version"))
    hasher = MagicMock()
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: json.dumps({"version": "9.9.9"}))
    with (
        patch("custom_components.roommind.ir.async_create_issue") as create,
        patch("custom_components.roommind.ir.async_delete_issue") as delete,
    ):
        await _async_check_version_mismatch(hass)

    create.assert_called_once()
    assert create.call_args.kwargs["translation_placeholders"] == {"version": "9.9.9"}
    delete.assert_not_called()
    del hasher


async def test_the_running_version_clears_the_restart_issue(plant_hass, monkeypatch):
    hass = _tmp_hass(plant_hass, Path("/tmp/roommind-version"))
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: json.dumps({"version": VERSION}))
    with (
        patch("custom_components.roommind.ir.async_create_issue") as create,
        patch("custom_components.roommind.ir.async_delete_issue") as delete,
    ):
        await _async_check_version_mismatch(hass)

    create.assert_not_called()
    delete.assert_called_once_with(hass, DOMAIN, "restart_required")


async def test_an_unreadable_manifest_is_ignored(plant_hass, monkeypatch):
    hass = _tmp_hass(plant_hass, Path("/tmp/roommind-version"))

    def _raise(self, *args, **kwargs):
        raise OSError("no manifest")

    monkeypatch.setattr(Path, "read_text", _raise)
    with patch("custom_components.roommind.ir.async_create_issue") as create:
        await _async_check_version_mismatch(hass)

    create.assert_not_called()
