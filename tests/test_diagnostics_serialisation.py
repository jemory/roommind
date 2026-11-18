"""The diagnostics payload has to survive the websocket's JSON encoder.

`roommind/diagnostics/get` sends the dict through HA's websocket connection, which
encodes with orjson plus `json_encoder_default`. That default only knows sets, tuples,
floats, `as_dict` objects, Paths and datetimes — anything else (a numpy scalar, an
ndarray, a Decimal, an enum) raises TypeError, HA turns it into `unknown_error`, and the
panel's export buttons appear to do nothing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from homeassistant.helpers.json import json_bytes

from custom_components.roommind.const import DOMAIN
from custom_components.roommind.diagnostics import async_get_config_entry_diagnostics

from .conftest import Harness, make_room

pytestmark = pytest.mark.asyncio


def _offending_paths(value: object, path: str = "") -> list[str]:
    """Every leaf the websocket encoder would reject, with its path."""
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _offending_paths(v, f"{path}.{k}")]
    if isinstance(value, (list, tuple)):
        return [p for i, v in enumerate(value) for p in _offending_paths(v, f"{path}[{i}]")]
    if value is None or isinstance(value, (str, bool, int, float)):
        return []
    return [f"{path}: {type(value).__module__}.{type(value).__name__}"]


async def test_the_diagnostics_payload_serialises(harness: Harness, fake_states):
    coordinator = await harness.seed({"living_room": make_room()})
    harness.hass.data[DOMAIN]["coordinator"] = coordinator
    await harness.cycle()

    result = await async_get_config_entry_diagnostics(harness.hass, MagicMock())

    offending = _offending_paths(result)
    assert offending == [], "the websocket encoder cannot serialise these values:\n" + "\n".join(offending)
    json_bytes(result)
