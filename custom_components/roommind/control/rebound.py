"""Symmetric forced-air air-vs-mass transient relaxation for RoomMind.

A forced-air system (furnace, heat pump) conditions the **air node** faster than
the building mass (drywall, furniture, screed) follows. While the unit runs the
room sensor leads the mass; after it stops the air relaxes back towards the
mass temperature within minutes. The effect is symmetric:

* heating blows warm air in: the sensor rises above the mass (charge ``+``),
  then settles back down after the stop;
* cooling blows cold air in: the sensor drops below the mass (charge ``-``),
  then rebounds up after the stop.

The signed charge is tracked per room as a first-order lag of the mode's drive:
``+1`` while heating, ``-1`` while cooling, ``0`` when the fan stops
(``advance_rebound``). The mean charge over a step (``mean_rebound``) is what
the RC model and the EKF consume: with a non-negative ``beta_f`` (the air/mass
offset rate) the regressor ``drive - mean_charge`` gives the sign of the
observed fast term in every phase (see :func:`fast_regressor`).

The old AC-only rebound (``residual_heat.advance_rebound`` / ``mean_rebound``)
is the cooling-only predecessor of this module; RoomMind's central HVAC
adaptation replaces it with the symmetric charge here.
"""

from __future__ import annotations

import math

MODE_CIRCULATING = "circulating"

# Relaxation time of the air/mass offset: the air has exchanged with the mass
# (or the next cycle) within ~3 tau, so a stopped unit stops leading the sensor.
FORCED_AIR_REBOUND_TAU_MINUTES: float = 10.0

# Below this the charge is treated as zero so the state (and its persistence)
# does not keep a vanishing residue alive forever.
FORCED_AIR_REBOUND_CUTOFF: float = 0.01


def drive_for_mode(mode: str | None) -> float:
    """Signed drive of *mode*: ``+1`` heating, ``-1`` cooling, ``0`` otherwise.

    The drive is the charge a long run would reach: heating charges the air hot
    relative to the mass, cooling charges it cold.
    """
    if mode == "heating":
        return 1.0
    if mode == "cooling":
        return -1.0
    return 0.0


def advance_rebound(
    charge: float,
    drive: float,
    dt_minutes: float,
    tau_minutes: float = FORCED_AIR_REBOUND_TAU_MINUTES,
) -> float:
    """Charge after *dt_minutes* of a constant *drive*.

    Exact exponential lag towards *drive* (``charge`` lives in ``[-1, 1]``).
    A vanishing charge is snapped to ``0.0`` once the drive is gone, and the
    result is clamped to ``[-1, 1]`` so the caller's model term stays bounded.
    """
    elapsed = max(0.0, dt_minutes)
    result = drive + (charge - drive) * math.exp(-elapsed / tau_minutes)
    if drive == 0.0 and abs(result) < FORCED_AIR_REBOUND_CUTOFF:
        return 0.0
    return max(-1.0, min(1.0, result))


def mean_rebound(
    charge: float,
    drive: float,
    dt_minutes: float,
    tau_minutes: float = FORCED_AIR_REBOUND_TAU_MINUTES,
) -> float:
    """Time-average of the charge over ``[0, dt_minutes]`` (the step's ``q_rebound``).

    Integrating the exponential lag gives
    ``drive + (charge - drive) * (1 - exp(-ratio)) / ratio``; the average is
    used instead of the end value because the offset decays within minutes and
    the value at flush time would understate the step.
    """
    if dt_minutes <= 0:
        return charge
    ratio = dt_minutes / tau_minutes
    average = drive + (charge - drive) * (1.0 - math.exp(-ratio)) / ratio
    if drive == 0.0 and abs(average) < FORCED_AIR_REBOUND_CUTOFF:
        return 0.0
    return average


def fast_regressor(mode: str | None, mean_charge: float | None) -> float:
    """Regressor of the air/mass offset: ``drive - mean_charge``.

    With a non-negative ``beta_f`` the sign of ``beta_f * r_fast`` is correct in
    every phase:

    ============  ======================  ==============  ===================
    phase         drive   mean charge     r_fast          beta_f * r_fast
    ============  ======================  ==============  ===================
    heating on    +1      ~0              > 0             > 0 (rapid rise)
    heating off   0       > 0             < 0             < 0 (settles down)
    cooling on    -1      ~0              < 0             < 0 (rapid drop)
    cooling off   0       < 0             > 0             > 0 (rebounds up)
    ============  ======================  ==============  ===================

    ``None`` (no charge tracked) gives ``0.0``, so callers that do not track
    the charge keep the plain RC model.
    """
    if mean_charge is None:
        return 0.0
    return drive_for_mode(mode) - mean_charge


class ForcedAirReboundTracker:
    """Per-room signed air/mass charge of a forced-air system.

    The tracker is driven by the mode the EKF learns from, so the charge always
    matches the training labels. Rooms that are not tracked read as ``0.0``.
    """

    def __init__(self) -> None:
        self._charge: dict[str, float] = {}

    def charge(self, area_id: str) -> float:
        """Signed charge of *area_id* in ``[-1, 1]`` (``0.0`` when never charged)."""
        return self._charge.get(area_id, 0.0)

    def step(self, area_id: str, mode: str | None, dt_minutes: float) -> tuple[float, float]:
        """Advance one step of *mode*; return ``(mean charge, charge after)``.

        The mean is what the step's model term uses, the end charge is stored
        for the next step. A charge that decayed below the cutoff is dropped, so
        an idle room holds no residue.
        """
        drive = drive_for_mode(mode)
        previous = self.charge(area_id)
        average = mean_rebound(previous, drive, dt_minutes)
        following = advance_rebound(previous, drive, dt_minutes)
        if following == 0.0:
            self._charge.pop(area_id, None)
        else:
            self._charge[area_id] = following
        return average, following

    def remove_room(self, area_id: str) -> None:
        """Discard the tracked charge of a removed room."""
        self._charge.pop(area_id, None)

    def reset(self) -> None:
        """Discard every tracked charge."""
        self._charge.clear()
