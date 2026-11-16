"""Small helpers re-exported from the coordinator for the entity platforms."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from custom_components.roommind.coordinator import _get_area_name, _match_room_entity, has_air_quality_sensor


def test_has_air_quality_sensor_only_for_indoor_rooms_with_a_sensor():
    assert has_air_quality_sensor({"co2_sensor": "sensor.co2"}) is True
    assert has_air_quality_sensor({"voc_sensor": "sensor.voc"}) is True
    assert has_air_quality_sensor({"co2_sensor": ""}) is False
    assert has_air_quality_sensor({"co2_sensor": "sensor.co2", "is_outdoor": True}) is False


def test_room_entity_suffixes_match_exactly():
    rooms = {"bedroom": {}, "bedroom_2": {}}
    assert _match_room_entity("bedroom_2_override", rooms) == ("bedroom_2", "_override")
    assert _match_room_entity("bedroom_override", rooms) == ("bedroom", "_override")
    assert _match_room_entity("kitchen_override", rooms) is None


def test_the_area_name_comes_from_the_area_registry():
    area = SimpleNamespace(name="Living Room")
    registry = MagicMock()
    registry.async_get_area = MagicMock(return_value=area)
    with patch("homeassistant.helpers.area_registry.async_get", return_value=registry):
        assert _get_area_name(MagicMock(), "living_room") == "Living Room"


def test_an_unknown_area_falls_back_to_its_id():
    registry = MagicMock()
    registry.async_get_area = MagicMock(return_value=None)
    with patch("homeassistant.helpers.area_registry.async_get", return_value=registry):
        assert _get_area_name(MagicMock(), "living_room") == "living_room"


def test_a_missing_area_registry_falls_back_to_the_id():
    with patch("homeassistant.helpers.area_registry.async_get", side_effect=KeyError("area_registry")):
        assert _get_area_name(MagicMock(), "living_room") == "living_room"
