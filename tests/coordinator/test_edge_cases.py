"""Rare branches of the control loop: corrupt settings, forecast paths and failures."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.roommind.const import DOMAIN, MODE_HEATING
from custom_components.roommind.control.solar import (
    build_oriented_solar_series,
    build_solar_series,
    compute_q_solar_norm,
)
from custom_components.roommind.coordinator import _RoomCycle
from custom_components.roommind.utils.history_store import HistoryStore

from .conftest import Harness, make_room

pytestmark = pytest.mark.asyncio

_SOLAR_NOON_TS = datetime(2023, 6, 21, 11, 8, tzinfo=UTC).timestamp()


def weather_service(forecast: list[dict]) -> Any:
    async def _call(domain: str, service: str, data: dict, **kwargs: Any) -> Any:
        if domain == "weather" and service == "get_forecasts":
            return {data["entity_id"]: {"forecast": forecast}}
        return {}

    return _call


async def test_a_corrupt_window_delay_falls_back_to_zero(harness: Harness, fake_states):
    fake_states.set("binary_sensor.window", "on")
    await harness.seed(
        {
            "living_room": make_room(
                {
                    "exterior_window_sensors": ["binary_sensor.window"],
                    "window_open_delay": "junk",
                }
            )
        }
    )

    data = await harness.cycle()

    assert data["rooms"]["living_room"]["window_open"] is True


async def test_a_room_deleted_after_the_snapshot_is_skipped(harness: Harness, monkeypatch):
    coordinator = await harness.seed({"living_room": make_room()})
    store = coordinator.hass.data[DOMAIN]["store"]
    real_get_rooms = store.get_rooms
    state = {"first": True}

    def get_rooms() -> dict:
        rooms = real_get_rooms()
        if state["first"]:
            # The snapshot still carries the room, the store no longer does.
            state["first"] = False
            rooms["ghost"] = {"area_id": "ghost"}
        return rooms

    monkeypatch.setattr(store, "get_rooms", get_rooms)
    data = await harness.cycle()

    assert set(data["rooms"]) == {"living_room"}


async def test_a_room_without_an_area_id_gets_one(harness: Harness, monkeypatch):
    coordinator = await harness.seed({"living_room": make_room()})
    store = coordinator.hass.data[DOMAIN]["store"]
    room = store.get_room("living_room")
    assert room is not None
    room.pop("area_id")
    monkeypatch.setattr(store, "get_rooms", lambda: {"living_room": room})

    data = await harness.cycle()

    assert data["rooms"]["living_room"]["area_id"] == "living_room"


async def test_the_weather_entity_supplies_the_outdoor_temperature(harness: Harness, fake_states):
    fake_states.remove("sensor.outdoor_temp")
    fake_states.set("weather.home", "sunny", {"temperature": 7.5})
    await harness.seed({"living_room": make_room()}, settings={"weather_entity": "weather.home"})

    data = await harness.cycle()

    assert data["central_plant"]["outdoor_temp"] == pytest.approx(7.5)
    assert data["central_plant"]["outdoor_source"] == "weather"


async def test_an_unavailable_weather_entity_falls_back_to_none(harness: Harness, fake_states):
    fake_states.remove("sensor.outdoor_temp")
    fake_states.set("weather.home", "unavailable", {"temperature": 7.5})
    await harness.seed({"living_room": make_room()}, settings={"weather_entity": "weather.home"})

    data = await harness.cycle()

    assert data["central_plant"]["outdoor_temp"] is None
    assert data["central_plant"]["outdoor_source"] == "none"


async def test_the_cloud_cover_is_read_for_the_solar_estimate(harness: Harness, fake_states):
    fake_states.set("weather.home", "cloudy", {"cloud_coverage": 100})
    coordinator = await harness.seed({"living_room": make_room()}, settings={"weather_entity": "weather.home"})

    q_solar = coordinator._read_current_q_solar({"weather_entity": "weather.home"})

    expected = compute_q_solar_norm(
        harness.hass.config.latitude,
        harness.hass.config.longitude,
        time.time(),
        100.0,
    )
    assert q_solar == pytest.approx(expected, abs=1e-6)


async def test_a_second_cycle_does_not_notify_again(harness: Harness, fake_states):
    fake_states.remove("sensor.outdoor_temp")
    coordinator = await harness.seed({"living_room": make_room()})
    coordinator._outdoor_warning_sent = True

    with patch("custom_components.roommind.coordinator.async_create_notification") as create:
        await harness.cycle()

    create.assert_not_called()


async def test_the_outdoor_forecast_builds_the_plan_series(harness: Harness, fake_states):
    fake_states.set("weather.home", "sunny", {"temperature": 4.0, "cloud_coverage": 20})
    harness.hass.services.async_call = AsyncMock(
        side_effect=weather_service(
            [
                {"temperature": 4.0, "cloud_coverage": 20},
                {"temperature": 6.0, "cloud_coverage": 30},
                {"temperature": 8.0, "cloud_coverage": 40},
            ]
        )
    )
    await harness.seed({"living_room": make_room()}, settings={"weather_entity": "weather.home"})

    data = await harness.cycle()

    assert data["central_plant"]["action"] == MODE_HEATING
    assert data["central_plant"]["actions"]


async def test_mold_prevention_lifts_a_presence_off_band(harness: Harness, fake_states):
    fake_states.set("person.alex", "not_home")
    fake_states.set("sensor.room_humidity", "92.0")
    fake_states.set("sensor.room_temp", "16.0", {"unit_of_measurement": "°C"})
    await harness.seed(
        {"living_room": make_room({"presence_persons": ["person.alex"]})},
        settings={
            "presence_enabled": True,
            "presence_persons": ["person.alex"],
            "presence_away_action": "off",
            "mold_prevention_enabled": True,
            "mold_prevention_intensity": "medium",
        },
    )

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["force_off"] is False
    assert room["heat_target"] is not None
    assert room["mold_boost"] > 0


async def test_a_one_sided_cooling_override_cannot_take_a_mold_boost(harness: Harness, fake_states):
    fake_states.set("sensor.room_humidity", "92.0")
    fake_states.set("sensor.room_temp", "22.0", {"unit_of_measurement": "°C"})
    await harness.seed(
        {
            "living_room": make_room(
                {
                    "override_cool": 24.0,
                    "override_heat": None,
                    "override_until": time.time() + 3600,
                    "override_type": "custom",
                }
            )
        },
        settings={"mold_prevention_enabled": True, "mold_prevention_intensity": "medium"},
    )

    data = await harness.cycle()

    room = data["rooms"]["living_room"]
    assert room["mold_prevention_active"] is True
    # A cooling-only band has no heating side to lift; the boost is reported as zero.
    assert room["heat_target"] is None
    assert room["mold_boost"] == 0.0


async def test_a_history_rotation_failure_does_not_break_the_cycle(harness: Harness):
    from custom_components.roommind.const import HISTORY_ROTATE_CYCLES

    coordinator = await harness.seed({"living_room": make_room()})
    coordinator._history_rotate_count = HISTORY_ROTATE_CYCLES - 1
    history = coordinator._history_store
    assert history is not None
    history.rotate.side_effect = OSError("disk full")

    data = await harness.cycle()

    assert "living_room" in data["rooms"]


async def test_the_history_row_is_skipped_without_a_history_store(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    coordinator._history_store = None

    # No exception, no row.
    await coordinator._async_record_history_row(
        "living_room",
        MagicMock(current_temp_raw=18.0),
        {},
        None,
        time.time(),
    )


async def test_a_ventilation_failure_does_not_break_the_cycle(harness: Harness, fake_states):
    fake_states.set("sensor.co2", "1500")
    fake_states.set("fan.vent", "off")
    coordinator = await harness.seed(
        {"living_room": make_room({"co2_sensor": "sensor.co2", "ventilation_entity": "fan.vent"})}
    )
    coordinator._ventilation.async_reconcile = AsyncMock(side_effect=RuntimeError("ventilation down"))

    data = await harness.cycle()

    assert "living_room" in data["rooms"]


async def test_releasing_ventilation_tolerates_a_failing_reconcile(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    coordinator._ventilation.async_reconcile = AsyncMock(side_effect=RuntimeError("bus down"))

    await coordinator.async_release_ventilation()


async def test_a_failing_recent_history_read_is_ignored(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    history = coordinator._history_store
    assert history is not None
    history.read_detail.side_effect = OSError("gone")

    data = await harness.cycle()

    assert data["rooms"]["living_room"]["current_temp"] == 18.0


async def test_the_history_store_is_created_on_first_load(harness: Harness):
    coordinator = await harness.seed({"living_room": make_room()})
    coordinator._history_store = None
    store = coordinator.hass.data[DOMAIN]["store"]

    coordinator._load_learned_state(store, {})

    assert isinstance(coordinator._history_store, HistoryStore)


async def test_an_entity_with_a_foreign_uid_is_left_alone(harness: Harness, fake_registry):
    fake_registry.entities = {
        "sensor.foreign": SimpleNamespace(entity_id="sensor.foreign", platform=DOMAIN, unique_id="other_uid"),
    }
    coordinator = await harness.seed({"living_room": make_room()})

    coordinator.cleanup_orphaned_entities()

    assert fake_registry.removed == []
    assert "sensor.foreign" in fake_registry.entities


# ── Per-room solar / topology fallbacks and MPC memoisation ─────────────


async def test_mpc_builds_one_oriented_series_per_azimuth_tuple(harness: Harness):
    """Same-facing rooms share one POA series; unconfigured rooms reuse one GHI series."""
    coordinator = await harness.seed(
        {
            "south_east": make_room({"solar_azimuths": [180, 90]}),
            "east_south": make_room({"solar_azimuths": [90, 180]}),
            "west": make_room({"solar_azimuths": [270]}),
            "plain": make_room(),
        }
    )
    with (
        patch(
            "custom_components.roommind.coordinator.build_oriented_solar_series",
            wraps=build_oriented_solar_series,
        ) as oriented,
        patch("custom_components.roommind.coordinator.build_solar_series", wraps=build_solar_series) as ghi,
        # Pin solar noon so the facade series are non-zero and distinguishable.
        patch("custom_components.roommind.coordinator.time.time", return_value=_SOLAR_NOON_TS),
        patch.object(
            coordinator._mpc_controller,
            "evaluate",
            wraps=coordinator._mpc_controller.evaluate,
        ) as evaluate,
    ):
        await harness.cycle()

    assert oriented.call_count == 2
    assert ghi.call_count == 1
    assert {tuple(call.args[3]) for call in oriented.call_args_list} == {(90, 180), (270,)}
    rooms = {room.area_id: room for room in evaluate.call_args.kwargs["rooms"]}
    assert rooms["south_east"].solar_series == rooms["east_south"].solar_series
    assert rooms["west"].solar_series != rooms["south_east"].solar_series


async def test_room_reference_falls_back_for_an_isolated_room(harness: Harness):
    """A room with no neighbours keeps the house mean, exactly."""
    coordinator = await harness.seed({"solo": make_room()})
    cycles = [_RoomCycle(area_id="solo", room=make_room())]
    graph = coordinator._build_room_graph(cycles)
    temps = {"solo": 20.0}
    included = coordinator._house_ref_included_area_ids(cycles)

    assert not graph.configured
    assert coordinator._room_ref_temp(cycles[0], graph, temps, included, 21.0) == 21.0


async def test_only_neighbour_defines_the_local_reference(harness: Harness):
    """A room whose only neighbour is warm uses that neighbour's temperature."""
    coordinator = await harness.seed({"cool": make_room(), "warm": make_room({"neighbour_rooms": ["cool"]})})
    cycles = [
        _RoomCycle(area_id="cool", room=make_room(), current_temp=20.0),
        _RoomCycle(
            area_id="warm",
            room=make_room({"neighbour_rooms": ["cool"]}),
            current_temp=24.0,
        ),
    ]
    graph = coordinator._build_room_graph(cycles)
    temps = {"cool": 20.0, "warm": 24.0}
    included = coordinator._house_ref_included_area_ids(cycles)

    assert coordinator._room_ref_temp(cycles[0], graph, temps, included, 22.0) == 24.0
    assert coordinator._room_ref_temp(cycles[1], graph, temps, included, 22.0) == 20.0


async def test_a_window_open_neighbour_is_excluded_from_the_reference(harness: Harness):
    """An open window removes a neighbour from the include set, restoring the fallback."""
    coordinator = await harness.seed({"cool": make_room(), "warm": make_room()})
    cycles = [
        _RoomCycle(area_id="cool", room=make_room(), window_open=True, current_temp=20.0),
        _RoomCycle(area_id="warm", room=make_room({"neighbour_rooms": ["cool"]}), current_temp=24.0),
    ]
    graph = coordinator._build_room_graph(cycles)
    temps = {"cool": 20.0, "warm": 24.0}
    included = coordinator._house_ref_included_area_ids(cycles)

    assert "cool" not in included
    assert coordinator._room_ref_temp(cycles[1], graph, temps, included, 22.0) == 22.0


async def test_outdoor_cycles_are_never_topology_neighbours(harness: Harness):
    """An outdoor area, even one declaring neighbours, is never part of the graph."""
    coordinator = await harness.seed({"living": make_room()})
    cycles = [
        _RoomCycle(area_id="living", room=make_room()),
        _RoomCycle(
            area_id="porch",
            room=make_room({"is_outdoor": True, "neighbour_rooms": ["living"]}),
            is_outdoor=True,
        ),
    ]
    graph = coordinator._build_room_graph(cycles)

    assert set(graph.weights) == {"living"}
    assert "porch" not in graph.mixing_group
