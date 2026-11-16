"""Tests for the symmetric forced-air rebound (air/mass transient) helpers."""

from __future__ import annotations

import math

import pytest

from custom_components.roommind.control.rebound import (
    FORCED_AIR_REBOUND_CUTOFF,
    FORCED_AIR_REBOUND_TAU_MINUTES,
    MODE_CIRCULATING,
    ForcedAirReboundTracker,
    advance_rebound,
    drive_for_mode,
    fast_regressor,
    mean_rebound,
)

TAU = FORCED_AIR_REBOUND_TAU_MINUTES


def test_constants_match_the_cooling_only_predecessor():
    """The forced-air relaxation uses the same 10 min time constant as the AC rebound."""
    assert TAU == 10.0
    assert FORCED_AIR_REBOUND_CUTOFF == 0.01
    assert MODE_CIRCULATING == "circulating"


# ---------------------------------------------------------------------------
# drive_for_mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("heating", 1.0),
        ("cooling", -1.0),
        ("idle", 0.0),
        (MODE_CIRCULATING, 0.0),
        ("window_open", 0.0),
        (None, 0.0),
    ],
)
def test_drive_for_mode(mode: str | None, expected: float) -> None:
    """Heating charges positive, cooling negative, everything else not at all."""
    assert drive_for_mode(mode) == expected


# ---------------------------------------------------------------------------
# advance_rebound
# ---------------------------------------------------------------------------


def test_advance_rebound_charges_towards_plus_one_while_heating():
    charge = advance_rebound(0.0, 1.0, TAU)
    assert charge == pytest.approx(1.0 - math.exp(-1.0))
    assert advance_rebound(charge, 1.0, 10 * TAU) == pytest.approx(1.0, abs=1e-4)


def test_advance_rebound_charges_towards_minus_one_while_cooling():
    charge = advance_rebound(0.0, -1.0, TAU)
    assert charge == pytest.approx(-(1.0 - math.exp(-1.0)))
    assert advance_rebound(charge, -1.0, 10 * TAU) == pytest.approx(-1.0, abs=1e-4)


def test_advance_rebound_is_exact_exponential_lag():
    """A step of dt is the closed-form lag, not an Euler approximation."""
    charge, drive, dt = 0.4, 1.0, 7.0
    expected = drive + (charge - drive) * math.exp(-dt / TAU)
    assert advance_rebound(charge, drive, dt) == pytest.approx(expected)


def test_advance_rebound_decays_without_drive():
    assert advance_rebound(0.8, 0.0, TAU) == pytest.approx(0.8 * math.exp(-1.0))
    assert advance_rebound(-0.8, 0.0, TAU) == pytest.approx(-0.8 * math.exp(-1.0))


def test_advance_rebound_non_positive_dt_keeps_the_charge():
    assert advance_rebound(0.5, 0.0, 0.0) == 0.5
    assert advance_rebound(-0.5, 1.0, -3.0) == -0.5


def test_advance_rebound_cutoff_snaps_idle_residue_to_zero():
    """A nearly relaxed charge becomes exactly zero so it does not linger in the state."""
    assert advance_rebound(0.005, 0.0, 0.0) == 0.0
    assert advance_rebound(-0.005, 0.0, 0.0) == 0.0
    assert advance_rebound(0.5, 0.0, 20 * TAU) == 0.0
    # A running unit keeps its charge even while below the cutoff.
    assert advance_rebound(0.005, 1.0, 0.0) == pytest.approx(0.005)


def test_advance_rebound_clamps_to_the_signed_unit_interval():
    assert advance_rebound(1.0, 1.0, 0.0) == 1.0
    assert advance_rebound(5.0, 1.0, 0.0) == 1.0
    assert advance_rebound(-5.0, -1.0, 0.0) == -1.0


# ---------------------------------------------------------------------------
# mean_rebound
# ---------------------------------------------------------------------------


def test_mean_rebound_is_the_interval_mean_of_the_trajectory():
    """The mean is the integral of the lag, not a midpoint or end value."""
    dt = 5.0
    n = 1000
    for charge, drive in ((0.6, 0.0), (0.2, 1.0), (-0.3, -1.0)):
        samples = [advance_rebound(charge, drive, dt * (i + 0.5) / n) for i in range(n)]
        assert mean_rebound(charge, drive, dt) == pytest.approx(sum(samples) / n, rel=1e-3)


def test_mean_rebound_closed_form():
    charge, drive, dt = 0.4, -1.0, 4.0
    ratio = dt / TAU
    expected = drive + (charge - drive) * (1.0 - math.exp(-ratio)) / ratio
    assert mean_rebound(charge, drive, dt) == pytest.approx(expected)


def test_mean_rebound_non_positive_dt_returns_the_charge():
    assert mean_rebound(0.3, 1.0, 0.0) == 0.3
    assert mean_rebound(-0.3, -1.0, -2.0) == -0.3


def test_mean_rebound_cutoff_snaps_idle_residue_to_zero():
    assert mean_rebound(0.005, 0.0, 1.0) == 0.0
    assert mean_rebound(-0.005, 0.0, 1.0) == 0.0
    # With a drive the mean stays meaningful.
    assert mean_rebound(0.005, 1.0, 1.0) == pytest.approx(1.0 - 0.995 * (1.0 - math.exp(-0.1)) / 0.1)


# ---------------------------------------------------------------------------
# fast_regressor
# ---------------------------------------------------------------------------


def test_fast_regressor_without_a_tracked_charge_is_zero():
    assert fast_regressor("heating", None) == 0.0
    assert fast_regressor(None, None) == 0.0


@pytest.mark.parametrize(
    ("mode", "mean_charge", "expected"),
    [
        ("heating", 0.0, 1.0),
        ("heating", 0.9, 0.1),
        (None, 0.9, -0.9),
        ("cooling", 0.0, -1.0),
        ("cooling", -0.9, -0.1),
        (None, -0.9, 0.9),
        (MODE_CIRCULATING, 0.4, -0.4),
    ],
)
def test_fast_regressor_is_drive_minus_mean_charge(mode: str | None, mean_charge: float, expected: float) -> None:
    assert fast_regressor(mode, mean_charge) == pytest.approx(expected)


def test_fast_regressor_sign_gives_the_observed_air_behaviour():
    """With beta_f >= 0 the fast term is positive exactly when the air leads the mass."""
    cases = [
        # (mode, mean charge, expected sign of beta_f * r_fast)
        ("heating", 0.0, 1),  # run start: air rises faster than the mass
        (None, 0.5, -1),  # after a heating run: air settles down
        ("cooling", 0.0, -1),  # run start: air drops faster than the mass
        (None, -0.5, 1),  # after a cooling run: air rebounds up
    ]
    for mode, mean_charge, expected_sign in cases:
        regressor = fast_regressor(mode, mean_charge)
        assert math.copysign(1.0, regressor) == expected_sign


# ---------------------------------------------------------------------------
# ForcedAirReboundTracker
# ---------------------------------------------------------------------------


def test_tracker_starts_uncharged():
    tracker = ForcedAirReboundTracker()
    assert tracker.charge("living") == 0.0


def test_tracker_step_returns_mean_and_next_charge():
    tracker = ForcedAirReboundTracker()
    average, following = tracker.step("living", "heating", TAU)
    assert average == pytest.approx(mean_rebound(0.0, 1.0, TAU))
    assert following == pytest.approx(advance_rebound(0.0, 1.0, TAU))
    assert tracker.charge("living") == pytest.approx(following)


def test_tracker_charges_while_heating_and_relaxes_after_the_stop():
    tracker = ForcedAirReboundTracker()
    for _ in range(5):
        tracker.step("living", "heating", TAU)
    charged = tracker.charge("living")
    assert charged > 0.9

    averages = []
    previous = charged
    for _ in range(8):
        average, following = tracker.step("living", "idle", TAU)
        averages.append(average)
        assert 0.0 <= following <= previous
        previous = following
    assert averages[0] > 0.0  # the mean of a decaying positive charge is positive
    assert averages[-1] < averages[0]
    # After ~8 tau the residue is gone entirely.
    assert tracker.charge("living") == 0.0


def test_tracker_cooling_charge_is_negative_and_rebounds_up():
    tracker = ForcedAirReboundTracker()
    for _ in range(5):
        tracker.step("living", "cooling", TAU)
    charged = tracker.charge("living")
    assert charged < -0.9
    average, following = tracker.step("living", "idle", TAU)
    assert average < 0.0  # mean charge while still cold
    assert charged < following < 0.0  # relaxed upwards towards zero, still cold


def test_tracker_tracks_rooms_independently():
    tracker = ForcedAirReboundTracker()
    tracker.step("a", "heating", TAU)
    tracker.step("b", "cooling", TAU)
    assert tracker.charge("a") > 0.0
    assert tracker.charge("b") < 0.0
    assert tracker.charge("c") == 0.0
    tracker.remove_room("a")
    assert tracker.charge("a") == 0.0
    assert tracker.charge("b") < 0.0


def test_tracker_remove_unknown_room_is_a_noop():
    tracker = ForcedAirReboundTracker()
    tracker.remove_room("nope")
    assert tracker.charge("nope") == 0.0


def test_tracker_reset_clears_every_room():
    tracker = ForcedAirReboundTracker()
    tracker.step("a", "heating", TAU)
    tracker.step("b", "cooling", TAU)
    tracker.reset()
    assert tracker.charge("a") == 0.0
    assert tracker.charge("b") == 0.0
