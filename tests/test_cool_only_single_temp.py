"""cool_only rooms configured with a single comfort_temp/eco_temp (API, old configs).

The single value landed on the heat side next to the default comfort_cool (24.0), e.g.
heat 24.5 / cool 24.0: the optimizer lifted the cool target to 24.5 and the cover
target took 24.5, while guard, hysteresis and the panel used 24.0.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.core import CoreState

from custom_components.roommind.const import DEFAULT_COMFORT_COOL, DEFAULT_ECO_COOL, TargetTemps
from custom_components.roommind.utils.schedule_utils import cap_cool_only_heat, make_target_resolver

from .coordinator.conftest import DEFAULT_THERMOSTAT_ATTRIBUTES, THERMOSTAT_ENTITY, FakeStates, Harness, make_room


@pytest.mark.parametrize(
    ("heat", "cool", "mode", "expected"),
    [
        (24.5, 24.0, "cool_only", (24.0, 24.0)),
        (21.0, 24.0, "cool_only", (21.0, 24.0)),
        (None, 24.0, "cool_only", (None, 24.0)),
        (24.5, None, "cool_only", (24.5, None)),
        (24.5, 24.0, "auto", (24.5, 24.0)),
        (24.5, 24.0, "heat_only", (24.5, 24.0)),
    ],
)
def test_cap_cool_only_heat(heat, cool, mode, expected):
    assert cap_cool_only_heat(heat, cool, mode) == expected


@pytest.mark.asyncio
async def test_api_create_with_single_temps_sets_the_cool_side(store):
    room = await store.async_save_room("buero", {"climate_mode": "cool_only", "comfort_temp": 24.5, "eco_temp": 28.0})
    assert room["comfort_cool"] == 24.5
    assert room["eco_cool"] == 28.0


@pytest.mark.asyncio
async def test_api_update_with_single_temps_sets_the_cool_side(store):
    await store.async_save_room("buero", {"climate_mode": "cool_only", "comfort_cool": 23.0})
    room = await store.async_save_room("buero", {"comfort_temp": 25.0})
    assert room["comfort_cool"] == 25.0


@pytest.mark.asyncio
async def test_panel_values_and_other_modes_are_untouched(store):
    panel = await store.async_save_room(
        "buero", {"climate_mode": "cool_only", "comfort_temp": 21.0, "comfort_cool": 24.5}
    )
    assert panel["comfort_cool"] == 24.5
    auto = await store.async_save_room("wohnzimmer", {"climate_mode": "auto", "comfort_temp": 22.0})
    assert auto["comfort_heat"] == 22.0
    assert auto["comfort_cool"] == DEFAULT_COMFORT_COOL
    heat = await store.async_save_room("bad", {"climate_mode": "heat_only", "eco_temp": 18.0})
    assert heat["eco_cool"] == DEFAULT_ECO_COOL


@pytest.mark.asyncio
async def test_old_cool_only_room_without_split_fields_migrates_to_the_cool_side(store):
    store._store.async_load = AsyncMock(
        return_value={
            "rooms": {
                "buero": {
                    "area_id": "buero",
                    "climate_mode": "cool_only",
                    "comfort_temp": 24.5,
                    "eco_temp": 28.0,
                    "thermostats": [],
                    "acs": [],
                    "schedules": [],
                }
            }
        }
    )
    await store.async_load()
    room = store.get_room("buero")
    assert room["comfort_cool"] == 24.5
    assert room["eco_cool"] == 28.0


def test_resolver_series_never_inverts_a_cool_only_band():
    room = {"climate_mode": "cool_only", "comfort_heat": 24.5, "comfort_cool": 24.0}
    assert make_target_resolver(None, room, {})(0.0) == TargetTemps(heat=24.0, cool=24.0)
    auto = {**room, "climate_mode": "auto"}
    assert make_target_resolver(None, auto, {})(0.0) == TargetTemps(heat=24.5, cool=24.0)


@pytest.fixture
def central_hass(hass):
    """A mocked Home Assistant that can run a full Central Plant cycle.

    Mirrors the ``plant_hass`` fixture of ``tests/coordinator/conftest.py``; that
    conftest's fixtures only apply inside its directory, so the cool_only case
    lives here with its own wiring.
    """
    states = FakeStates()
    hass.states = states
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
    states.set("sensor.outdoor_temp", "5.0", {"unit_of_measurement": "°C"})
    states.set("sensor.room_temp", "25.0", {"unit_of_measurement": "°C"})
    states.set("sensor.room_humidity", "55.0")
    states.set(THERMOSTAT_ENTITY, "cool", dict(DEFAULT_THERMOSTAT_ATTRIBUTES))
    return hass


@pytest.mark.asyncio
async def test_coordinator_resolves_a_consistent_cool_only_band(central_hass):
    """A cool_only room keeps one band: no lifted cool side next to the default heat side."""
    harness = Harness(central_hass)
    await harness.seed(
        rooms={
            "buero": make_room(
                climate_mode="cool_only",
                comfort_heat=24.5,
                comfort_temp=24.5,
                comfort_cool=24.0,
            )
        }
    )
    data = await harness.cycle()
    state = data["rooms"]["buero"]
    assert (state["heat_target"], state["cool_target"], state["target_temp"]) == (24.0, 24.0, 24.0)
