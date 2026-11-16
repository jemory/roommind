"""Sidebar panel registration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.roommind import _async_register_panel
from custom_components.roommind.const import DOMAIN

pytestmark = pytest.mark.asyncio


async def test_the_panel_is_for_admins_only(hass):
    """The panel edits the whole setup; non-admins keep the dashboard card and the room entities."""
    hass.data[DOMAIN] = {}
    hass.http.async_register_static_paths = AsyncMock()
    with (
        patch.object(Path, "exists", return_value=True),
        patch("custom_components.roommind.async_register_built_in_panel") as register,
    ):
        await _async_register_panel(hass)

    register.assert_called_once()
    assert register.call_args.kwargs["require_admin"] is True
    assert register.call_args.kwargs["frontend_url_path"] == "roommind"


async def test_without_a_built_panel_nothing_is_registered(hass):
    hass.data[DOMAIN] = {}
    hass.http.async_register_static_paths = AsyncMock()
    with (
        patch.object(Path, "exists", return_value=False),
        patch("custom_components.roommind.async_register_built_in_panel") as register,
    ):
        await _async_register_panel(hass)

    register.assert_not_called()
    hass.http.async_register_static_paths.assert_not_awaited()
    assert "panel_registered" not in hass.data[DOMAIN]


async def test_an_already_served_panel_still_registers(hass):
    hass.data[DOMAIN] = {}
    hass.http.async_register_static_paths = AsyncMock(side_effect=RuntimeError("already registered"))
    with (
        patch.object(Path, "exists", return_value=True),
        patch(
            "custom_components.roommind.async_register_built_in_panel",
            side_effect=ValueError("already registered"),
        ) as register,
    ):
        await _async_register_panel(hass)

    register.assert_called_once()
    assert hass.data[DOMAIN]["panel_registered"] is True


async def test_the_panel_is_registered_only_once(hass):
    hass.data[DOMAIN] = {}
    hass.http.async_register_static_paths = AsyncMock()
    with (
        patch.object(Path, "exists", return_value=True),
        patch("custom_components.roommind.async_register_built_in_panel") as register,
    ):
        await _async_register_panel(hass)
        await _async_register_panel(hass)

    register.assert_called_once()
