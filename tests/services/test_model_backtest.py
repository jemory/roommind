"""One-hour backtest of the room model on recorded history."""

from __future__ import annotations

from custom_components.roommind.control.rebound import advance_rebound, drive_for_mode, mean_rebound
from custom_components.roommind.control.thermal_model import RCModel
from custom_components.roommind.services.model_backtest import (
    MIN_BACKTEST_WINDOWS,
    _drive_of,
    _factor,
    _step_mode,
    backtest,
)

# A central forced-air plant: the room air, the rest of the house behind a
# closed interior door, the blower and the air/mass offset all in play.
TRUTH = RCModel(
    U=0.05,
    Q_heat=1.5,
    Q_cool=1.0,
    Q_solar=0.8,
    Q_occupancy=0.2,
    Q_fast=0.6,
    U_house=0.3,
    Q_fan=0.9,
)
HOUSE_TEMP = 17.0
DOOR_MIX = 0.4
DOOR_SUPPLY = 0.75


def _history(
    hours: float = 6.0,
    step_s: float = 300.0,
    model: RCModel = TRUTH,
    power=lambda k: 1.0 if (k // 6) % 3 == 0 else 0.0,
    window=lambda k: False,
    fan=lambda k: 0.0,
    house_temp: float | None = HOUSE_TEMP,
    door_mix: float = DOOR_MIX,
    door_supply: float = DOOR_SUPPLY,
) -> list[dict]:
    """A room driven by *model* with the recorded inputs, one row every *step_s*.

    ``model_power`` is the signed delivered share; ``fan(k)`` runs the blower
    without conditioning (circulation). The signed forced-air charge is advanced
    exactly as the backtest reconstructs it from the recorded rows.
    """
    points = []
    T = 20.0
    charge = 0.0
    for k in range(int(hours * 3600 / step_s) + 1):
        share = power(k)
        u_fan = fan(k)
        p = {
            "ts": 1_000_000.0 + k * step_s,
            "room_temp": round(T, 3),
            "outdoor_temp": 5.0,
            "window_open": window(k),
            "model_power": share,
            "model_solar": 0.3,
            "model_occupancy": 1.0,
            "model_house_temp": house_temp,
            "model_fan": u_fan,
            "model_door_mix": door_mix,
            "model_door_supply": door_supply,
        }
        points.append(p)
        dt = step_s / 60.0
        mode = _step_mode(share, u_fan)
        if p["window_open"]:
            T = model.predict_window_open(T, 5.0, 3.0, dt)
        else:
            Q = share * (model.Q_heat if share > 0 else model.Q_cool)
            T = model.predict(
                T,
                5.0,
                Q,
                dt,
                T_house=house_temp,
                u_fan=u_fan,
                door_mix_factor=door_mix,
                door_supply_factor=door_supply,
                q_solar=0.3,
                q_occupancy=1.0,
                q_rebound=mean_rebound(charge, drive_for_mode(mode), dt),
                mode=mode,
            )
        # charge[k] belongs to the step that starts at row k
        charge = advance_rebound(charge, drive_for_mode(mode), dt)
    return points


def test_the_true_model_is_almost_exact():
    result = backtest(_history(), TRUTH, 3.0)
    figures = result.figures()
    assert figures["horizon_minutes"] == 60
    assert figures["active"]["mae"] < 0.01
    assert figures["idle"]["n"] + figures["active"]["n"] == len(result.predicted)
    # The first hour has no reading an hour before
    assert min(result.predicted) >= 1_000_000.0 + 3600


def test_a_wrong_model_shows_its_error_and_its_direction():
    wrong = RCModel(U=0.05, Q_heat=0.75, Q_cool=1.0, Q_solar=0.8, Q_occupancy=0.2, Q_fast=0.6, U_house=0.3, Q_fan=0.9)
    figures = backtest(_history(), wrong, 3.0).figures()
    # Half the heating rate: the room ends up warmer than the model says
    assert figures["active"]["mae"] > 0.1
    assert figures["active"]["bias"] < -0.1


def test_hours_without_heating_count_as_idle():
    figures = backtest(_history(power=lambda k: 0.0), TRUTH, 3.0).figures()
    assert figures["active"]["n"] == 0
    assert figures["idle"]["mae"] < 0.01
    assert figures["active"]["mae"] is None


def test_cooling_counts_as_active():
    figures = backtest(_history(power=lambda k: -1.0 if k % 4 == 0 else 0.0), TRUTH, 3.0).figures()
    assert figures["active"]["n"] > 0
    assert figures["active"]["mae"] < 0.01


def test_rows_without_model_inputs_break_the_chain():
    points = _history()
    for p in points[30:40]:
        p["model_power"] = None
    result = backtest(points, TRUTH, 3.0)
    # No hour may run across the unknown rows
    for ts in result.predicted:
        start = ts - 3600
        assert not any(start <= p["ts"] < ts and p["model_power"] is None for p in points)
    assert points[35]["ts"] not in result.predicted


def test_a_gap_breaks_the_chain():
    points = _history()
    for p in points[40:]:
        p["ts"] += 3600
    result = backtest(points, TRUTH, 3.0)
    assert points[40]["ts"] not in result.predicted
    assert points[40 + 12]["ts"] in result.predicted


def test_open_windows_are_drawn_but_not_counted():
    points = _history(window=lambda k: 20 <= k < 24)
    result = backtest(points, TRUTH, 3.0)
    assert points[24]["ts"] in result.predicted
    assert abs(result.predicted[points[24]["ts"]] - points[24]["room_temp"]) < 0.01
    counted = result.idle.n + result.active.n
    assert counted < len(result.predicted)


def test_too_few_hours_show_no_figure():
    points = _history(hours=1.0 + (MIN_BACKTEST_WINDOWS - 2) * 300 / 3600)
    figures = backtest(points, TRUTH, 3.0).figures()
    assert figures["active"]["n"] + figures["idle"]["n"] < MIN_BACKTEST_WINDOWS
    assert figures["active"]["mae"] is None and figures["idle"]["mae"] is None


def test_nothing_to_compute():
    assert backtest([], TRUTH, 3.0).predicted == {}
    assert backtest(_history(), RCModel(U=0.0), 3.0).predicted == {}


def test_missing_readings_are_skipped():
    points = _history()
    points[50]["room_temp"] = None
    result = backtest(points, TRUTH, 3.0)
    assert points[50]["ts"] not in result.predicted


# ── The forced-air replay: modes, house node, blower and closed doors ──

_CYCLE = lambda k: [1.0, -1.0, 0.0, 0.0][k % 4]  # noqa: E731 - heating, cooling, fan-only, idle
_FAN = lambda k: 1.0 if k % 4 == 2 else 0.0  # noqa: E731


def test_the_true_model_replays_heating_cooling_and_circulation():
    points = _history(power=_CYCLE, fan=_FAN)
    figures = backtest(points, TRUTH, 3.0).figures()

    assert figures["active"]["n"] > 0
    assert figures["active"]["mae"] < 0.01
    assert figures["idle"]["mae"] is None or figures["idle"]["mae"] < 0.01


def test_circulation_alone_counts_as_an_active_hour():
    figures = backtest(_history(power=lambda k: 0.0, fan=lambda k: 1.0), TRUTH, 3.0).figures()

    assert figures["active"]["n"] > 0
    assert figures["idle"]["n"] == 0
    assert figures["active"]["mae"] < 0.01


def test_a_closed_interior_door_is_replayed():
    points = _history()
    assert backtest(points, TRUTH, 3.0).figures()["active"]["mae"] < 0.01

    opened = [dict(p, model_door_mix=1.0, model_door_supply=1.0) for p in points]
    assert backtest(opened, TRUTH, 3.0).figures()["active"]["mae"] > 0.05


def test_the_house_node_is_replayed():
    points = _history()
    assert backtest(points, TRUTH, 3.0).figures()["active"]["mae"] < 0.01

    without_house = [dict(p, model_house_temp=None) for p in points]
    assert backtest(without_house, TRUTH, 3.0).figures()["active"]["mae"] > 0.05


def test_the_recorded_blower_is_replayed():
    # A warm house behind the blower: circulating is what keeps the room there
    points = _history(power=lambda k: 0.0, fan=lambda k: 1.0, house_temp=24.0)
    assert backtest(points, TRUTH, 3.0).figures()["active"]["mae"] < 0.01

    without_fan = [dict(p, model_fan=0.0) for p in points]
    # Dropping the blower also turns the hours idle, so compare the trajectory itself
    drifted = backtest(without_fan, TRUTH, 3.0).predicted
    last_ts = points[-1]["ts"]
    assert abs(drifted[last_ts] - points[-1]["room_temp"]) > 0.1


def test_the_signed_charge_is_reconstructed_from_the_recorded_sequence():
    """Heating then an off step: without the charge the settle-back is missing."""
    points = _history(power=lambda k: 1.0 if k < 12 else 0.0)
    assert backtest(points, TRUTH, 3.0).figures()["active"]["mae"] < 0.01

    no_fast = RCModel(U=0.05, Q_heat=1.5, Q_cool=1.0, Q_solar=0.8, Q_occupancy=0.2, U_house=0.3, Q_fan=0.9)
    assert backtest(points, no_fast, 3.0).figures()["active"]["mae"] > 0.02


def test_rows_from_before_the_columns_replay_as_a_standalone_room():
    """Rows without the new inputs must not break the replay (they are idle, open-door steps)."""
    plain = RCModel(U=0.05, Q_heat=1.5, Q_cool=1.0, Q_solar=0.8, Q_occupancy=0.2)
    points = _history(model=plain, house_temp=None, door_mix=1.0, door_supply=1.0)
    for p in points:
        del p["model_house_temp"], p["model_fan"], p["model_door_mix"], p["model_door_supply"]

    figures = backtest(points, plain, 3.0).figures()

    assert figures["active"]["mae"] < 0.01


# ── The recorded share / blower -> mode and drive mapping ─────────────


class TestStepMode:
    def test_delivered_power_wins_over_the_blower(self):
        assert _step_mode(1.0, 1.0) == "heating"
        assert _step_mode(-1.0, 1.0) == "cooling"

    def test_the_blower_alone_circulates(self):
        assert _step_mode(0.0, 1.0) == "circulating"

    def test_nothing_running_is_idle(self):
        assert _step_mode(0.0, 0.0) == "idle"

    def test_the_drive_carries_the_sign(self):
        assert _drive_of(1.0) == 1.0
        assert _drive_of(-1.0) == -1.0
        assert _drive_of(0.0) == 0.0
        assert _drive_of(None) == 0.0

    def test_a_missing_coupling_factor_means_open(self):
        assert _factor(None) == 1.0
        assert _factor(0.4) == 0.4
        assert _factor("0.4") == 1.0
