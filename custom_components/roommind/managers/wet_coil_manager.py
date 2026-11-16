"""Central evaporator wet-coil tracking for RoomMind.

A cooling run wets the indoor coil with condensate.  Immediately afterwards,
Circulation would evaporate that moisture back into the rooms: the air gets
stickier instead of more comfortable, exactly when the compressor just stopped.
This tracker models the coil's remaining water charge and blocks Circulation
until either enough time has passed or the indoor air is dry enough that a
little evaporating coil moisture is harmless.

The model is deliberately simple: a first-order charge that fills during
cooling and drains with a different time constant depending on what the plant
does (idle, fan-only, heating, or another cooling run).  It is not a physical
psychrometric model; it only needs to answer "is the coil still wet enough to
matter?".
"""

from __future__ import annotations

import math
import time

WET_CHARGE_TAU_MINUTES = 8.0
IDLE_DRAIN_TAU_MINUTES = 25.0
FAN_EVAP_TAU_MINUTES = 12.0
HEAT_DRY_TAU_MINUTES = 4.0
LOCKOUT_CHARGE_THRESHOLD = 0.25

# Observed plant actions that drive the charge model.
ACTION_COOLING = "cooling"
ACTION_HEATING = "heating"
ACTION_CIRCULATING = "circulating"

# Below this the remaining charge is irrelevant and is snapped to zero.
_CHARGE_EPSILON = 0.01
# A change at least this large is worth persisting for the UI/diagnostics.
_DIRTY_CHARGE_DELTA = 0.05


class WetCoilTracker:
    """Track one central coil's wetness and decide when Circulation is unsafe."""

    def __init__(self) -> None:
        self.charge: float = 0.0
        self.last_cooling_ts: float | None = None
        self.state_dirty: bool = False
        self._last_action: str | None = None

    def step(self, observed_action: str | None, dt_minutes: float, now_ts: float | None = None) -> float:
        """Advance the charge model and return the new ``[0, 1]`` charge."""
        now = time.time() if now_ts is None else now_ts
        dt = max(0.0, float(dt_minutes))
        previous = self.charge

        if observed_action == ACTION_COOLING:
            self.charge = 1.0 + (self.charge - 1.0) * math.exp(-dt / WET_CHARGE_TAU_MINUTES)
            self.last_cooling_ts = now
        elif observed_action == ACTION_HEATING:
            self.charge = self.charge * math.exp(-dt / HEAT_DRY_TAU_MINUTES)
        elif observed_action == ACTION_CIRCULATING:
            self.charge = self.charge * math.exp(-dt / FAN_EVAP_TAU_MINUTES)
        else:
            self.charge = self.charge * math.exp(-dt / IDLE_DRAIN_TAU_MINUTES)

        self.charge = max(0.0, min(1.0, self.charge))
        if self.charge < _CHARGE_EPSILON:
            self.charge = 0.0

        cooling_edge = (observed_action == ACTION_COOLING) != (self._last_action == ACTION_COOLING)
        if abs(self.charge - previous) >= _DIRTY_CHARGE_DELTA or cooling_edge:
            self.state_dirty = True
        self._last_action = observed_action
        return self.charge

    def is_circulation_locked(
        self,
        weighted_indoor_rh: float | None,
        plant_settings: dict,
        now_ts: float | None = None,
    ) -> bool:
        """Return True when Circulation should be held back to let the coil dry.

        The lockout only applies while the coil carries a meaningful charge and
        while the configured post-cooling window has not expired.  With dry
        indoor air (``< wet_coil_max_rh``) evaporating the remaining moisture is
        harmless, so Circulation is allowed again even before the window ends.
        """
        now = time.time() if now_ts is None else now_ts
        lockout_min = int(plant_settings.get("wet_coil_lockout_minutes", 25))
        max_rh = float(plant_settings.get("wet_coil_max_rh", 52.0))

        if lockout_min <= 0 or self.charge < LOCKOUT_CHARGE_THRESHOLD:
            return False
        if self.last_cooling_ts is not None and (now - self.last_cooling_ts) >= lockout_min * 60.0:
            return False
        if weighted_indoor_rh is None:
            return True
        return bool(weighted_indoor_rh >= max_rh)

    def to_dict(self) -> dict:
        """Persistable coil state (wall-clock, so it survives a restart)."""
        return {
            "charge": self.charge,
            "last_cooling_ts": self.last_cooling_ts,
        }

    def load_state(self, data: dict) -> None:
        """Restore persisted coil state, clamping malformed values."""
        self.charge = 0.0
        self.last_cooling_ts = None
        self.state_dirty = False
        self._last_action = None
        if not isinstance(data, dict):
            return
        charge = data.get("charge")
        if isinstance(charge, int | float):
            self.charge = max(0.0, min(1.0, float(charge)))
        last_cooling = data.get("last_cooling_ts")
        if isinstance(last_cooling, int | float):
            self.last_cooling_ts = float(last_cooling)

    def reset(self) -> None:
        """Forget the coil state (e.g. after a manual plant reset)."""
        self.charge = 0.0
        self.last_cooling_ts = None
        self.state_dirty = False
        self._last_action = None
