"""Room-configuration accessors shared by the thermal model inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def room_solar_azimuths(room: Mapping[str, Any]) -> tuple[int, ...]:
    """Azimuths [deg, 0 = N] of the room's exterior glazing, sorted and de-duplicated.

    An empty tuple means "not configured": the caller keeps the unoriented
    horizontal-plane irradiance estimate. Values are coerced defensively because
    the stored room dict is user data that may have been written by an older API
    client or hand-edited.
    """
    raw = room.get("solar_azimuths") or ()
    azimuths: set[int] = set()
    for value in raw:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        azimuths.add(int(round(number)) % 360)
    return tuple(sorted(azimuths))
