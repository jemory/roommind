"""How far the room model is off one hour ahead, measured on the recorded history.

For every recorded point the model starts from the reading one hour earlier and runs forward
with what really happened in between (delivered heating/cooling share, outdoor temperature,
sun behind the covers, occupancy, the rest-of-house node, the blower, the interior-door
coupling, open windows): the inputs the model learned from, recorded per row since 2.0
(``model_*`` columns). The difference to the reading at the end is the model's error over
that hour.

The replay tracks the same signed forced-air charge the EKF trains with: the air/mass offset
of a furnace or heat pump run (``+`` heating, ``-`` cooling) relaxes after the unit stops, and
a blower circulating the air mixes the room with the house without conditioning it. Only the
charge along the *recorded* sequence is reconstructed — the model is today's model applied to
the past, not a forecast made back then: over long ranges it mixes seasons and changes to the
room. Rows from before the columns have no inputs and are left out instead of guessing the
delivered power (the history only has the commanded one).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..const import MODE_CIRCULATING, MODE_COOLING, MODE_HEATING, MODE_IDLE
from ..control.rebound import advance_rebound, drive_for_mode, mean_rebound
from ..control.thermal_model import RCModel

BACKTEST_HORIZON_S = 3600.0
# A longer gap between two rows breaks the chain (restart, missing sensor)
MAX_STEP_S = 900.0
# Windows needed before a figure is shown
MIN_BACKTEST_WINDOWS = 12


@dataclass
class BacktestFigures:
    """Mean absolute and signed error after one hour, per kind of hour."""

    n: int = 0
    abs_sum: float = 0.0
    signed_sum: float = 0.0

    def add(self, error: float) -> None:
        self.n += 1
        self.abs_sum += abs(error)
        self.signed_sum += error

    def as_dict(self) -> dict:
        if self.n < MIN_BACKTEST_WINDOWS:
            return {"n": self.n, "mae": None, "bias": None}
        return {
            "n": self.n,
            "mae": round(self.abs_sum / self.n, 3),
            "bias": round(self.signed_sum / self.n, 3),
        }


@dataclass
class BacktestResult:
    predicted: dict[float, float] = field(default_factory=dict)
    idle: BacktestFigures = field(default_factory=BacktestFigures)
    active: BacktestFigures = field(default_factory=BacktestFigures)

    def figures(self) -> dict:
        return {
            "horizon_minutes": round(BACKTEST_HORIZON_S / 60),
            "idle": self.idle.as_dict(),
            "active": self.active.as_dict(),
        }


def _inputs_known(p: dict) -> bool:
    return p.get("room_temp") is not None and p.get("outdoor_temp") is not None and p.get("model_power") is not None


def _drive_of(share: float | None) -> float:
    """Signed forced-air drive of a recorded delivered-power *share* (+ heating, - cooling)."""
    if share is None:
        return 0.0
    if share > 0:
        return 1.0
    return -1.0 if share < 0 else 0.0


def _step_mode(share: float, u_fan: float) -> str:
    """Mode a recorded step ran in: its delivered power first, the blower only without it."""
    if share > 0:
        return MODE_HEATING
    if share < 0:
        return MODE_COOLING
    return MODE_CIRCULATING if u_fan > 0.0 else MODE_IDLE


def _factor(value: object) -> float:
    """A recorded coupling factor, 1.0 when the row has none (open door, undamped supply)."""
    return float(value) if isinstance(value, (int, float)) else 1.0


def backtest(points: list[dict], model: RCModel, k_window: float) -> BacktestResult:
    """Run the one-hour backtest over *points* (sorted by ``ts``, typed like _csv_to_points).

    ``predicted`` maps the end timestamp of each hour to the model's temperature; hours with an
    open window are drawn but not counted (the window model is the roughest part). An hour
    counts as active when the model saw heating, cooling or the blower running in it.
    """
    result = BacktestResult()
    n = len(points)
    if n < 2 or model.U <= 0:
        return result

    ts = [p["ts"] for p in points]
    known = [_inputs_known(p) for p in points]
    # breaks[k]: steps up to point k that cannot be simulated (unknown inputs or a gap)
    breaks = [0] * n
    for k in range(1, n):
        bad = not known[k - 1] or ts[k] - ts[k - 1] > MAX_STEP_S or ts[k] <= ts[k - 1]
        breaks[k] = breaks[k - 1] + int(bad)

    # Signed air/mass charge along the recorded sequence, from the delivered power the row
    # before each step recorded (the EKF's charge follows its own labels)
    charge = [0.0] * n
    for k in range(1, n):
        drive = _drive_of(points[k - 1].get("model_power"))
        charge[k] = advance_rebound(charge[k - 1], drive, max(0.0, ts[k] - ts[k - 1]) / 60.0)

    j = 0
    for i in range(1, n):
        if not known[i]:
            continue
        start = ts[i] - BACKTEST_HORIZON_S
        while j + 1 < i and ts[j + 1] <= start:
            j += 1
        if ts[j] > start or ts[i] - ts[j] > BACKTEST_HORIZON_S + MAX_STEP_S or not known[j]:
            continue
        if breaks[i] - breaks[j]:
            continue
        T = points[j]["room_temp"]
        active = False
        window = False
        for k in range(j, i):
            p = points[k]
            dt = (ts[k + 1] - ts[k]) / 60.0
            if p.get("window_open"):
                window = True
                T = model.predict_window_open(T, p["outdoor_temp"], k_window, dt)
                continue
            share: float = p.get("model_power") or 0.0
            u_fan = p.get("model_fan") or 0.0
            mode = _step_mode(share, u_fan)
            # share < 0 already carries the sign, so Q < 0 while cooling
            Q = share * model.Q_heat if share > 0 else (share * model.Q_cool if share < 0 else 0.0)
            active = active or Q != 0.0 or u_fan > 0.0
            T = model.predict(
                T,
                p["outdoor_temp"],
                Q,
                dt,
                T_house=p.get("model_house_temp"),
                u_fan=u_fan,
                door_mix_factor=_factor(p.get("model_door_mix")),
                door_supply_factor=_factor(p.get("model_door_supply")),
                q_solar=p.get("model_solar") or 0.0,
                q_occupancy=p.get("model_occupancy") or 0.0,
                q_rebound=mean_rebound(charge[k], drive_for_mode(mode), dt),
                mode=mode,
            )
        result.predicted[ts[i]] = round(T, 2)
        if not window:
            (result.active if active else result.idle).add(T - points[i]["room_temp"])
    return result
