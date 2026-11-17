"""Tests for the room-configuration accessors."""

from __future__ import annotations

import pytest

from custom_components.roommind.utils.room_utils import room_solar_azimuths


def test_unconfigured_room_returns_an_empty_tuple():
    """Empty means "not configured": the caller keeps the unoriented estimate."""
    assert room_solar_azimuths({}) == ()
    assert room_solar_azimuths({"solar_azimuths": []}) == ()
    assert room_solar_azimuths({"solar_azimuths": None}) == ()


def test_azimuths_are_sorted_and_deduplicated():
    assert room_solar_azimuths({"solar_azimuths": [225, 180, 180, 45]}) == (45, 180, 225)


def test_azimuths_are_canonicalised_onto_the_compass():
    """A stored 360 or a float must not produce a second, equivalent signature."""
    assert room_solar_azimuths({"solar_azimuths": [360, 0]}) == (0,)
    assert room_solar_azimuths({"solar_azimuths": [225.4]}) == (225,)
    assert room_solar_azimuths({"solar_azimuths": [-45]}) == (315,)


@pytest.mark.parametrize("raw", [["north"], [None], [{}], [True], [180, "junk"]])
def test_junk_entries_are_skipped_not_raised(raw: list[object]) -> None:
    """The stored room dict is user data; reading it must never raise."""
    result = room_solar_azimuths({"solar_azimuths": raw})
    assert all(isinstance(value, int) for value in result)


def test_a_valid_entry_survives_alongside_junk():
    assert room_solar_azimuths({"solar_azimuths": [180, "junk", None]}) == (180,)
