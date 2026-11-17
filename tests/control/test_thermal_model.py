"""Tests for the coupled forced-air thermal model: RCModel, ThermalEKF, RoomModelManager."""

from __future__ import annotations

import math
import random
from unittest.mock import MagicMock

import pytest

from custom_components.roommind.control import thermal_model as tm
from custom_components.roommind.control.rebound import MODE_CIRCULATING, ForcedAirReboundTracker
from custom_components.roommind.control.thermal_model import (
    EKF_VERSION,
    RCModel,
    RoomModelManager,
    ThermalEKF,
    regressor_signature,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _physical_model() -> RCModel:
    """A plausible forced-air room: U=0.15, U_house=0.25, beta_h=6, beta_f=2.5."""
    return RCModel(C=1.0, U=0.15, Q_heat=6.0, Q_cool=4.5, Q_fan=1.2, U_house=0.25, Q_fast=2.5)


def _initialized_ekf(T_init: float = 21.0, T_house: float | None = 21.0) -> ThermalEKF:
    """An EKF past its first (initialization-only) update."""
    ekf = ThermalEKF()
    ekf.update(T_init, 10.0, "idle", 1.0, T_house=T_house)
    return ekf


def _feed(ekf: ThermalEKF, *, mode: str = "idle", temp: float = 21.4, steps: int = 20, **kwargs: float | None) -> None:
    """Feed *steps* one-minute updates with a persistent innovation."""
    for _ in range(steps):
        ekf.update(temp, 10.0, mode, 1.0, **kwargs)


# ---------------------------------------------------------------------------
# RCModel: parameters, physics, serialization
# ---------------------------------------------------------------------------


def test_rc_model_defaults_are_c_normalized():
    model = RCModel()
    assert model.C == 1.0
    assert model.U == pytest.approx(0.12)
    assert model.Q_heat == pytest.approx(4.0)
    assert model.Q_cool == pytest.approx(3.5)
    assert model.Q_solar == 0.0
    assert model.Q_occupancy == 0.0
    assert model.Q_fast == 0.0
    assert model.U_house == pytest.approx(0.20)
    assert model.Q_fan == pytest.approx(0.80)


def test_rc_model_keeps_positional_parameter_order():
    """Legacy positional callers must keep landing on the same parameters."""
    model = RCModel(2.0, 50.0, 800.0, 1200.0, 1.0, 2.0, 3.0)
    assert model.C == 2.0
    assert model.U == 50.0
    assert model.Q_heat == 800.0
    assert model.Q_cool == 1200.0
    assert model.Q_solar == 1.0
    assert model.Q_occupancy == 2.0
    assert model.Q_fast == 3.0
    assert model.U_house == RCModel.DEFAULT_U_HOUSE
    assert model.Q_fan == RCModel.DEFAULT_Q_FAN


def test_rc_model_predict_matches_the_closed_form():
    # No house node: the coupling (gamma) must be zero to compare with 1R1C.
    model = RCModel(C=1.0, U=0.12, U_house=0.0)
    result = model.predict(20.0, 5.0, 0.0, 30)
    expected = 5.0 + (20.0 - 5.0) * math.exp(-0.12 * 0.5)
    assert result == pytest.approx(expected)


def test_rc_model_predict_heating_and_cooling_directions():
    model = RCModel(C=1.0, U=0.12, Q_heat=4.0, Q_cool=3.5)
    assert model.predict(18.0, 5.0, 4.0, 30) > 18.0
    assert model.predict(28.0, 32.0, -3.5, 30) < 28.0


def test_rc_model_steady_state_is_t_out_plus_q_over_u():
    model = RCModel(C=1.0, U=0.12, Q_heat=4.0, U_house=0.0)
    result = model.predict(20.0, 5.0, 4.0, 10_000)
    assert result == pytest.approx(5.0 + 4.0 / 0.12, abs=0.01)


def test_rc_model_capacitance_scales_the_decay():
    fast = RCModel(C=1.0, U=0.12)
    slow = RCModel(C=2.0, U=0.12)
    assert fast.predict(20.0, 5.0, 0.0, 60) < slow.predict(20.0, 5.0, 0.0, 60)


@pytest.mark.parametrize("dt", [0.0, -1.0])
def test_rc_model_zero_or_negative_dt_returns_the_room_temperature(dt: float):
    model = RCModel()
    assert model.predict(21.0, 5.0, 4.0, dt) == 21.0


def test_rc_model_output_is_clamped():
    model = RCModel(C=1.0, U=1.0, Q_heat=1000.0)
    assert model.predict(60.0, 5.0, 0.0, 0.1) <= 50.0
    assert model.predict(-10.0, 5.0, 0.0, 0.1) >= 0.0


def test_rc_model_house_coupling_only_acts_with_a_house_temperature():
    model = RCModel(C=1.0, U=0.1, U_house=0.3, Q_heat=0.0, Q_cool=0.0)
    no_house = model.predict(20.0, 5.0, 0.0, 60)
    with_house = model.predict(20.0, 5.0, 0.0, 60, T_house=25.0)
    assert with_house > no_house
    # No house reading == the house node sits at the room temperature.
    assert model.predict(20.0, 5.0, 0.0, 60, T_house=None) == pytest.approx(no_house)


def test_rc_model_door_mix_factor_scales_the_house_coupling():
    model = RCModel(C=1.0, U=0.1, U_house=0.3, Q_heat=0.0, Q_cool=0.0)
    open_door = model.predict(20.0, 5.0, 0.0, 60, T_house=25.0)
    closed_door = model.predict(20.0, 5.0, 0.0, 60, T_house=25.0, door_mix_factor=0.0)
    no_house = model.predict(20.0, 5.0, 0.0, 60)
    assert closed_door < open_door
    assert closed_door < no_house  # no coupling at all -> cools to the outdoor equilibrium


def test_rc_model_door_supply_factor_scales_the_hvac_input():
    model = RCModel(C=1.0, U=0.12, Q_heat=4.0)
    full = model.predict(18.0, 5.0, 4.0, 30, mode="heating")
    half = model.predict(18.0, 5.0, 4.0, 30, mode="heating", door_supply_factor=0.5)
    assert 18.0 < half < full


def test_rc_model_circulating_blower_adds_house_mixing():
    model = RCModel(C=1.0, U=0.1, U_house=0.2, Q_fan=1.0)
    idle = model.predict(20.0, 5.0, 0.0, 60, mode="idle", T_house=25.0)
    circulating = model.predict(20.0, 5.0, 0.0, 60, mode=MODE_CIRCULATING, u_fan=1.0, T_house=25.0)
    assert circulating > idle
    # The blower term only applies while actually circulating.
    assert model.predict(20.0, 5.0, 0.0, 60, mode="idle", u_fan=1.0, T_house=25.0) == pytest.approx(idle)
    # A circulating mode with no commanded fan is plain idle.
    assert model.predict(20.0, 5.0, 0.0, 60, mode=MODE_CIRCULATING, u_fan=0.0, T_house=25.0) == pytest.approx(idle)


def test_rc_model_infers_the_mode_from_power_and_fan():
    model = RCModel(C=1.0, U=0.12, Q_heat=4.0, Q_cool=3.5, Q_fan=1.0, U_house=0.2)
    assert model.predict(18.0, 5.0, 4.0, 30) == pytest.approx(model.predict(18.0, 5.0, 4.0, 30, mode="heating"))
    assert model.predict(28.0, 32.0, -3.5, 30) == pytest.approx(model.predict(28.0, 32.0, -3.5, 30, mode="cooling"))
    assert model.predict(20.0, 5.0, 0.0, 30, u_fan=1.0, T_house=25.0) == pytest.approx(
        model.predict(20.0, 5.0, 0.0, 30, mode=MODE_CIRCULATING, u_fan=1.0, T_house=25.0)
    )
    assert model.predict(20.0, 5.0, 0.0, 30) == pytest.approx(model.predict(20.0, 5.0, 0.0, 30, mode="idle"))


def test_rc_model_rebound_term_is_signed():
    model = RCModel(C=1.0, U=0.12, Q_heat=4.0, Q_fast=2.0)
    # Heating run start: a fresh air node leads the mass -> the fast term is positive.
    fresh = model.predict(18.0, 5.0, 4.0, 30, mode="heating", q_rebound=0.0)
    charged = model.predict(18.0, 5.0, 4.0, 30, mode="heating", q_rebound=1.0)
    assert fresh > charged
    # After a heating run the positive charge cools the air back down.
    settling = model.predict(22.0, 5.0, 0.0, 30, mode="idle", q_rebound=0.8)
    plain = model.predict(22.0, 5.0, 0.0, 30, mode="idle", q_rebound=0.0)
    assert settling < plain
    # After a cooling run the negative charge rebounds the air up.
    rebounding = model.predict(24.0, 32.0, 0.0, 30, mode="idle", q_rebound=-0.8)
    assert rebounding > plain


def test_rc_model_untracked_charge_equals_a_neutral_charge():
    model = RCModel(C=1.0, U=0.12, Q_heat=4.0, Q_fast=2.0)
    # q_rebound=None means "not tracked" (no fast term); a fully charged air node
    # under heating also has a zero regressor, so the predictions match.
    assert model.predict(18.0, 5.0, 4.0, 30, mode="heating", q_rebound=None) == pytest.approx(
        model.predict(18.0, 5.0, 4.0, 30, mode="heating", q_rebound=1.0)
    )


def test_rc_model_legacy_residual_heat_only_while_idle():
    model = RCModel(C=1.0, U=0.12, Q_heat=4.0)
    idle = model.predict(20.0, 10.0, 0.0, 30)
    idle_residual = model.predict(20.0, 10.0, 0.0, 30, q_residual=0.5)
    assert idle_residual > idle
    heating = model.predict(20.0, 10.0, 4.0, 30)
    heating_residual = model.predict(20.0, 10.0, 4.0, 30, q_residual=0.5)
    assert heating_residual == pytest.approx(heating)


def test_rc_model_euler_branch_with_zero_loss():
    """U=0 and U_house=0 gives lam=0: the step must fall back to the Euler form."""
    model = RCModel(C=1.0, U=0.0, U_house=0.0, Q_heat=4.0)
    assert model.predict(20.0, 5.0, 4.0, 60) == pytest.approx(24.0)
    assert model.predict(20.0, 5.0, 0.0, 60) == pytest.approx(20.0)


def test_rc_model_solar_and_occupancy_gains():
    model = RCModel(C=1.0, U=0.12, Q_solar=2.0, Q_occupancy=1.0)
    plain = model.predict(20.0, 5.0, 0.0, 30)
    assert model.predict(20.0, 5.0, 0.0, 30, q_solar=1.0) > plain
    assert model.predict(20.0, 5.0, 0.0, 30, q_occupancy=1.0) > plain
    assert model.predict(20.0, 5.0, 0.0, 30, q_solar=0.0, q_occupancy=0.0) == pytest.approx(plain)


def test_rc_model_predict_window_open_is_bidirectional():
    model = RCModel(C=1.0, U=0.12)
    winter = model.predict_window_open(22.0, 5.0, 10.0, 5.0)
    summer = model.predict_window_open(22.0, 35.0, 10.0, 5.0)
    assert winter < 22.0
    assert summer > 22.0
    assert winter < model.predict(22.0, 5.0, 0.0, 5.0)
    assert summer > model.predict(22.0, 35.0, 0.0, 5.0)


@pytest.mark.parametrize("dt", [0.0, -1.0])
def test_rc_model_predict_window_open_non_positive_dt(dt: float):
    model = RCModel()
    assert model.predict_window_open(22.0, 5.0, 10.0, dt) == 22.0


def test_rc_model_predict_window_open_clamps():
    model = RCModel(C=1.0, U=100.0)
    assert model.predict_window_open(22.0, 60.0, 50.0, 60) <= 50.0
    assert model.predict_window_open(22.0, -20.0, 50.0, 60) >= 0.0


def test_rc_model_predict_trajectory_heats_then_cools():
    model = RCModel(C=1.0, U=0.12, Q_heat=4.0)
    trajectory = model.predict_trajectory(
        T_room=18.0,
        T_outdoor_series=[5.0] * 6,
        Q_active_series=[4.0, 4.0, 4.0, 0.0, 0.0, 0.0],
        dt_minutes=10,
    )
    assert len(trajectory) == 7
    assert trajectory[3] > trajectory[0]
    assert trajectory[6] < trajectory[3]


def test_rc_model_predict_trajectory_length_mismatch():
    model = RCModel()
    with pytest.raises(ValueError, match="same length"):
        model.predict_trajectory(20.0, [5.0, 5.0], [0.0], 5.0)


def test_rc_model_predict_trajectory_carries_the_new_inputs():
    """Solar, occupancy, rebound, mode, fan and house series are accepted per step."""
    model = RCModel(C=1.0, U=0.12, Q_heat=4.0, Q_solar=2.0, Q_occupancy=1.0, Q_fast=2.0)
    baseline = model.predict_trajectory(20.0, [5.0, 5.0], [0.0, 0.0], 30)
    enriched = model.predict_trajectory(
        20.0,
        [5.0, 5.0],
        [0.0, 0.0],
        30,
        q_solar_series=[1.0, 1.0],
        q_occupancy_series=[1.0, 1.0],
        q_rebound_series=[0.5, 0.5],
        mode_series=["idle", "idle"],
        u_fan_series=[0.0, 0.0],
        T_house=24.0,
        door_mix_factor=1.0,
        door_supply_factor=1.0,
    )
    for cold, warm in zip(baseline[1:], enriched[1:], strict=False):
        assert warm > cold
    # Explicit modes and a circulating fan series are honoured.
    fan = model.predict_trajectory(
        20.0,
        [5.0, 5.0],
        [0.0, 0.0],
        30,
        mode_series=[None, None],
        u_fan_series=[0.0, 1.0],
        T_house=24.0,
    )
    assert fan[1] > baseline[1]  # the fan mixes in the warmer house air


def test_rc_model_serialization_roundtrip_includes_the_coupling():
    model = RCModel(C=1.0, U=0.005, Q_heat=2.0, Q_cool=3.0, Q_solar=1.0, Q_occupancy=0.4, Q_fast=1.5)
    model.U_house = 0.05
    model.Q_fan = 0.3
    data = model.to_dict()
    assert data["U_house"] == pytest.approx(0.05)
    assert data["Q_fan"] == pytest.approx(0.3)
    restored = RCModel.from_dict(data)
    assert restored.to_dict() == pytest.approx(data)


def test_rc_model_from_dict_defaults_and_floors():
    old = {"C": 0.0, "U": 0.0, "Q_heat": -1.0, "Q_cool": -2.0}
    model = RCModel.from_dict(old)
    assert model.C == pytest.approx(0.1)  # floor
    assert model.U == pytest.approx(1e-6)  # learned models can have tiny U
    assert model.Q_heat == 0.0
    assert model.Q_cool == 0.0
    assert model.U_house == RCModel.DEFAULT_U_HOUSE
    assert model.Q_fan == RCModel.DEFAULT_Q_FAN


def test_rc_model_helpers_and_repr():
    assert RCModel.cooling_drive(0.0) == 0.0
    assert RCModel.cooling_drive(-1.0) == 1.0
    assert RCModel.cooling_drive(1.0) == 0.0
    assert RCModel.rebound_drive("heating") == 1.0
    assert RCModel.rebound_drive("cooling") == -1.0
    assert RCModel.rebound_drive(None) == 0.0
    text = repr(RCModel(C=1.0, U=0.12, Q_heat=4.0, Q_cool=3.5, U_house=0.2, Q_fan=0.8))
    assert "RCModel" in text
    assert "U_house=0.200" in text
    assert "Q_fan=0.80" in text


# ---------------------------------------------------------------------------
# ThermalEKF: exact 9x9 Jacobian verified by finite differences
# ---------------------------------------------------------------------------

_JACOBIAN_STATE = [21.0, 0.12, 0.20, 4.0, 3.5, 0.80, 0.50, 0.30, 0.30]
_MODES = ["idle", "heating", "cooling", MODE_CIRCULATING]
_REBOUND_BY_MODE = {
    "idle": 0.3,
    "heating": 0.3,
    "cooling": -0.5,
    MODE_CIRCULATING: 0.3,
}
_JACOBIAN_INPUTS: dict[str, float] = {
    "power_fraction": 0.8,
    "q_solar": 0.4,
    "q_occupancy": 1.0,
    "T_house": 25.0,
    "u_fan": 1.0,
    "door_mix_factor": 1.0,
    "door_supply_factor": 0.9,
}


def _jacobian_inputs(mode: str) -> dict[str, float]:
    return {**_JACOBIAN_INPUTS, "q_rebound": _REBOUND_BY_MODE[mode]}


def _fd_jacobian(
    ekf: ThermalEKF,
    x: list[float],
    T_outdoor: float,
    dt_h: float,
    mode: str,
    **kwargs: float | None,
) -> tuple[list[list[float]], list[float]]:
    """Analytic Jacobian and central finite differences of the same predict map."""
    F = ekf._compute_jacobian(x, T_outdoor, dt_h, mode, **kwargs)
    numeric: list[float] = []
    for i in range(ekf._N):
        eps = 1e-6 * max(1.0, abs(x[i]))
        plus = list(x)
        minus = list(x)
        plus[i] += eps
        minus[i] -= eps
        t_plus = ekf._step_terms(plus, T_outdoor, dt_h, mode, **kwargs).T_new
        t_minus = ekf._step_terms(minus, T_outdoor, dt_h, mode, **kwargs).T_new
        numeric.append((t_plus - t_minus) / (2.0 * eps))
    return F, numeric


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("index", range(9))
def test_ekf_jacobian_matches_finite_differences(mode: str, index: int) -> None:
    """Every F[0][i] is the exact derivative of the analytical predict map."""
    ekf = ThermalEKF()
    F, numeric = _fd_jacobian(ekf, _JACOBIAN_STATE, 5.0, 0.1, mode, **_jacobian_inputs(mode))
    assert F[0][index] == pytest.approx(numeric[index], rel=1e-5, abs=1e-9), (
        f"mode={mode} index={index}: analytic={F[0][index]!r} numeric={numeric[index]!r}"
    )


@pytest.mark.parametrize("mode", _MODES)
def test_ekf_jacobian_excitation_pattern_is_physical(mode: str) -> None:
    """Only the parameters that act in a mode have a non-zero temperature entry."""
    ekf = ThermalEKF()
    F, numeric = _fd_jacobian(ekf, _JACOBIAN_STATE, 5.0, 0.1, mode, **_jacobian_inputs(mode))
    always_on = {0, 1, 2, 6, 7, 8}
    expected_on = (
        always_on
        | {
            "heating": {3},
            "cooling": {4},
            MODE_CIRCULATING: {5},
            "idle": set(),
        }[mode]
    )
    for index in range(9):
        if index in expected_on:
            assert F[0][index] != 0.0, f"mode={mode}: F[0][{index}] should be non-zero"
        else:
            assert F[0][index] == 0.0, f"mode={mode}: F[0][{index}] should be zero"
        assert numeric[index] == pytest.approx(F[0][index], rel=1e-5, abs=1e-9)


@pytest.mark.parametrize("mode", _MODES)
def test_ekf_jacobian_is_identity_on_parameter_rows(mode: str) -> None:
    ekf = ThermalEKF()
    F = ekf._compute_jacobian(_JACOBIAN_STATE, 5.0, 0.1, mode, **_jacobian_inputs(mode))
    lam = 0.12 + 0.20 + (1.0 * 0.80 if mode == MODE_CIRCULATING else 0.0)
    assert F[0][0] == pytest.approx(math.exp(-lam * 0.1))
    for i in range(1, ekf._N):
        for j in range(ekf._N):
            assert F[i][j] == (1.0 if i == j else 0.0)


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("index", range(9))
def test_ekf_euler_jacobian_matches_finite_differences(monkeypatch: pytest.MonkeyPatch, mode: str, index: int) -> None:
    """The linearized lam -> 0 branch has its own exact Jacobian."""
    monkeypatch.setattr(tm, "_LAM_SMALL", 0.5)
    ekf = ThermalEKF()
    state = [21.0, 0.0, 0.0, 4.0, 3.5, 0.80, 0.50, 0.30, 0.30]
    F, numeric = _fd_jacobian(ekf, state, 5.0, 0.1, mode, **_jacobian_inputs(mode))
    assert F[0][index] == pytest.approx(numeric[index], rel=1e-5, abs=1e-9), (
        f"mode={mode} index={index}: analytic={F[0][index]!r} numeric={numeric[index]!r}"
    )


def test_ekf_euler_branch_transition_is_linear():
    """lam=0: the transition is T + dt*(u_src - alpha*(T-T_out) + gamma*(T_h-T))."""
    monkeypatch_state = [20.0, 0.0, 0.0, 4.0, 3.5, 0.8, 0.0, 0.0, 0.0]
    ekf = ThermalEKF()
    terms = ekf._step_terms(monkeypatch_state, 5.0, 0.5, "heating", power_fraction=1.0)
    assert terms.euler is True
    assert terms.T_new == pytest.approx(20.0 + 0.5 * 4.0)


def test_ekf_exact_branch_transition_equals_the_rc_model():
    """The filter and the MPC model must predict identically for the same parameters."""
    ekf = ThermalEKF()
    ekf._x = [21.0, 0.12, 0.20, 4.0, 3.5, 0.80, 0.50, 0.30, 0.30]
    model = ekf.get_model()
    for mode, q_active in (("heating", 4.0), ("cooling", -3.5), ("idle", 0.0), (MODE_CIRCULATING, 0.0)):
        for rebound in (None, 0.3, -0.6):
            kwargs: dict[str, float | None] = {
                "power_fraction": 0.7,
                "q_solar": 0.4,
                "q_occupancy": 1.0,
                "q_rebound": rebound,
                "T_house": 24.0,
                "u_fan": 1.0,
                "door_mix_factor": 0.9,
                "door_supply_factor": 0.95,
            }
            terms = ekf._step_terms(ekf._x, 5.0, 0.2, mode, **kwargs)
            model_prediction = model.predict(
                ekf._x[0],
                5.0,
                q_active * 0.7,
                12.0,
                q_solar=0.4,
                q_occupancy=1.0,
                q_rebound=rebound,
                mode=mode,
                u_fan=1.0,
                T_house=24.0,
                door_mix_factor=0.9,
                door_supply_factor=0.95,
            )
            assert model_prediction == pytest.approx(terms.T_new, abs=1e-9)


# ---------------------------------------------------------------------------
# ThermalEKF: state, update cycle, bounds
# ---------------------------------------------------------------------------


def test_ekf_version_and_initial_state():
    ekf = ThermalEKF()
    assert EKF_VERSION == 8
    assert ThermalEKF.EKF_VERSION == 8
    assert ekf._N == 9
    assert len(ekf._x) == 9
    assert len(ekf._P) == 9
    assert all(len(row) == 9 for row in ekf._P)
    assert ekf._x[1:] == pytest.approx([0.12, 0.20, 4.0, 3.5, 0.80, 0.5, 0.3, 0.3])
    assert [ekf._P[i][i] for i in range(9)] == pytest.approx([0.5, 0.04, 0.09, 9.0, 9.0, 1.0, 2.0, 0.5, 2.0])
    assert ekf._n_circulating == 0
    assert not ekf._initialized
    assert ekf.confidence == 0.0
    assert ekf._applicable_modes == {"idle", "heating", "cooling", MODE_CIRCULATING}


def test_ekf_first_update_only_initializes():
    ekf = ThermalEKF(T_init=15.0)
    ekf.update(22.0, 10.0, "idle", 5.0)
    assert ekf._x[0] == pytest.approx(22.0)
    assert ekf._initialized
    assert ekf._n_updates == 0
    assert ekf._n_idle == 0


@pytest.mark.parametrize("dt", [0.0, -1.0])
def test_ekf_update_with_non_positive_dt_is_a_noop(dt: float):
    ekf = _initialized_ekf()
    before = list(ekf._x)
    ekf.update(21.0, 10.0, "idle", dt)
    assert ekf._x == before
    assert ekf._n_updates == 0


def test_ekf_counts_every_mode_including_circulating():
    ekf = _initialized_ekf()
    for mode in ("heating", "cooling", MODE_CIRCULATING, "idle", "something_else"):
        ekf.update(21.0, 10.0, mode, 1.0, T_house=24.0)
    assert ekf._n_heating == 1
    assert ekf._n_cooling == 1
    assert ekf._n_circulating == 1
    assert ekf._n_idle == 2  # "idle" and the unknown mode both count as idle
    assert ekf._last_mode == "something_else"
    assert ekf._n_updates == 5


def test_ekf_sync_temperature_keeps_parameters():
    ekf = ThermalEKF()
    for _ in range(20):
        ekf.update(21.0, 10.0, "idle", 3.0)
    params = list(ekf._x[1:])
    ekf.sync_temperature(18.5)
    assert ekf._x[0] == 18.5
    assert ekf._x[1:] == params
    assert ekf._initialized


def test_ekf_parameter_bounds_hold_under_extreme_data():
    ekf = ThermalEKF()
    ekf.update(20.0, 10.0, "idle", 5.0)
    for i in range(80):
        temp = 20.0 + (10.0 if i % 2 == 0 else -10.0)
        mode = ("heating", "cooling", MODE_CIRCULATING, "idle")[i % 4]
        ekf.update(temp, 10.0, mode, 5.0, q_solar=1.0, q_occupancy=1.0, q_rebound=0.9, T_house=30.0)
    bounds = [
        (1, ThermalEKF._ALPHA_MIN, ThermalEKF._ALPHA_MAX),
        (2, ThermalEKF._ALPHA_HOUSE_MIN, ThermalEKF._ALPHA_HOUSE_MAX),
        (3, ThermalEKF._BETA_H_MIN, ThermalEKF._BETA_H_MAX),
        (4, ThermalEKF._BETA_C_MIN, ThermalEKF._BETA_C_MAX),
        (5, ThermalEKF._BETA_FAN_MIN, ThermalEKF._BETA_FAN_MAX),
        (6, ThermalEKF._BETA_S_MIN, ThermalEKF._BETA_S_MAX),
        (7, ThermalEKF._BETA_O_MIN, ThermalEKF._BETA_O_MAX),
        (8, ThermalEKF._BETA_F_MIN, ThermalEKF._BETA_F_MAX),
    ]
    for index, low, high in bounds:
        assert low <= ekf._x[index] <= high


def test_ekf_psd_and_symmetry_hold_after_mixed_training():
    ekf = ThermalEKF()
    ekf.update(20.0, 10.0, "idle", 3.0)
    for step in range(80):
        mode = ("idle", "heating", "cooling", MODE_CIRCULATING)[step % 4]
        ekf.update(21.0 + 0.02 * (step % 5), 10.0, mode, 3.0, q_solar=0.2, q_occupancy=0.5, q_rebound=0.2)
        for i in range(ekf._N):
            assert ekf._P[i][i] >= 0.0
            for j in range(ekf._N):
                assert ekf._P[i][j] == pytest.approx(ekf._P[j][i], abs=1e-12)


def test_ekf_anomaly_soft_reject_keeps_parameters_stable():
    true_model = _physical_model()
    ekf = ThermalEKF()
    T, T_out = 20.0, 10.0
    ekf.update(T, T_out, "idle", 5.0)
    for _ in range(10):
        T = true_model.predict(T, T_out, 0.0, 5.0)
        ekf.update(T, T_out, "idle", 5.0)
    alpha_before = ekf._x[1]
    beta_h_before = ekf._x[3]
    ekf.update(T + 10.0, T_out, "idle", 5.0)
    assert abs(ekf._x[1] - alpha_before) < 0.05
    assert abs(ekf._x[3] - beta_h_before) < 0.5


def test_ekf_update_step_skips_a_degenerate_innovation_covariance():
    ekf = ThermalEKF()
    ekf._initialized = True
    ekf._x[0] = 20.0
    ekf._P[0][0] = 0.0
    ekf._R = 0.0
    before = list(ekf._x)
    ekf._update_step(21.0)
    assert ekf._x == before


def test_ekf_enforce_psd_floors_negative_diagonals():
    ekf = ThermalEKF()
    ekf._P[2][2] = -0.5
    ekf._enforce_psd()
    assert ekf._P[2][2] == pytest.approx(1e-10)


def test_ekf_repr_mentions_the_new_parameters():
    text = repr(ThermalEKF())
    assert "ThermalEKF" in text
    assert "alpha_house=" in text
    assert "beta_fan=" in text
    assert "confidence=" in text


def test_ekf_set_applicable_modes():
    ekf = ThermalEKF()
    ekf.set_applicable_modes(can_heat=True, can_cool=False)
    assert ekf._applicable_modes == {"idle", "heating", MODE_CIRCULATING}
    ekf.set_applicable_modes(can_heat=False, can_cool=True)
    assert ekf._applicable_modes == {"idle", "cooling", MODE_CIRCULATING}


# ---------------------------------------------------------------------------
# ThermalEKF: excitation gating
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", _MODES)
def test_ekf_excitation_flags_follow_the_mode(mode: str) -> None:
    ekf = ThermalEKF()
    ekf._initialized = True
    ekf._x[0] = 20.0
    ekf._predict_step(
        5.0,
        mode,
        0.05,
        power_fraction=1.0,
        q_solar=0.5,
        q_occupancy=1.0,
        q_rebound=_REBOUND_BY_MODE[mode],
        T_house=25.0,
        u_fan=1.0,
    )
    assert ekf._excited[0] is True
    assert ekf._excited[1] is True  # alpha_out is always observable
    assert ekf._excited[2] is (mode == "idle")
    assert ekf._excited[3] is (mode == "heating")
    assert ekf._excited[4] is (mode == "cooling")
    assert ekf._excited[5] is (mode == MODE_CIRCULATING)
    assert ekf._excited[6] is (mode != "heating")  # sunny; never while heating
    assert ekf._excited[7] is True  # occupied
    assert ekf._excited[8] is True  # a charge is tracked


def test_ekf_house_parameters_need_a_measurable_house_difference():
    ekf = ThermalEKF()
    ekf._initialized = True
    ekf._x[0] = 20.0
    ekf._predict_step(5.0, "idle", 0.05, T_house=20.1)
    assert ekf._excited[2] is False
    ekf._predict_step(5.0, MODE_CIRCULATING, 0.05, T_house=ekf._x[0], u_fan=1.0)
    assert ekf._excited[5] is False
    ekf._predict_step(5.0, MODE_CIRCULATING, 0.05, T_house=None, u_fan=1.0)
    assert ekf._excited[2] is False
    assert ekf._excited[5] is False


def test_ekf_beta_fan_is_untouched_outside_circulating_excitation():
    ekf = _initialized_ekf(T_house=21.0)
    before = ekf._x[5]
    variance = ekf._P[5][5]

    # Circulating but the house is at the room temperature: nothing to learn from.
    for _ in range(20):
        ekf.update(21.4, 10.0, MODE_CIRCULATING, 1.0, T_house=ekf._x[0], u_fan=1.0)
    assert ekf._x[5] == before
    assert ekf._P[5][5] == variance

    # Circulating without any house reading at all.
    for _ in range(20):
        ekf.update(21.4, 10.0, MODE_CIRCULATING, 1.0, u_fan=1.0)
    assert ekf._x[5] == before

    # Heating with a large house difference.
    for _ in range(20):
        ekf.update(21.4, 10.0, "heating", 1.0, T_house=27.0)
    assert ekf._x[5] == before
    assert ekf._P[5][5] == variance

    # Cooling with a large house difference.
    for _ in range(20):
        ekf.update(21.4, 10.0, "cooling", 1.0, T_house=16.0)
    assert ekf._x[5] == before
    assert ekf._P[5][5] == variance


def test_ekf_beta_fan_learns_while_circulating_with_a_house_difference():
    ekf = _initialized_ekf(T_house=21.0)
    before = ekf._x[5]
    for i in range(60):
        ekf.update(21.0 + 0.4 * (i % 2), 10.0, MODE_CIRCULATING, 1.0, T_house=25.0, u_fan=1.0)
    assert ekf._x[5] != before
    assert ekf.is_param_learned(5)


def test_ekf_alpha_house_is_gated_to_idle_with_a_house_difference():
    ekf = _initialized_ekf(T_house=21.0)
    before = ekf._x[2]
    variance = ekf._P[2][2]
    for _ in range(20):
        ekf.update(21.4, 10.0, "heating", 1.0, T_house=27.0)
    for _ in range(20):
        ekf.update(21.4, 10.0, "cooling", 1.0, T_house=16.0)
    for _ in range(20):
        ekf.update(21.4, 10.0, MODE_CIRCULATING, 1.0, T_house=27.0, u_fan=1.0)
    assert ekf._x[2] == before
    assert ekf._P[2][2] == variance

    for i in range(60):
        ekf.update(21.0 + 0.4 * (i % 2), 10.0, "idle", 1.0, T_house=27.0)
    assert ekf._x[2] != before
    assert ekf.is_param_learned(2)


def test_ekf_beta_s_is_gated_to_sunlit_non_heating_steps():
    ekf = _initialized_ekf()
    before = ekf._x[6]
    variance = ekf._P[6][6]
    _feed(ekf, mode="idle", q_solar=0.0)
    assert ekf._x[6] == before
    assert ekf._P[6][6] == variance
    _feed(ekf, mode="heating", q_solar=0.8)
    assert ekf._x[6] == before
    assert ekf._P[6][6] == variance
    _feed(ekf, mode="idle", q_solar=0.8)
    assert ekf._x[6] != before
    assert ekf.is_param_learned(6)


def test_ekf_beta_o_is_gated_to_occupied_steps():
    ekf = _initialized_ekf()
    before = ekf._x[7]
    variance = ekf._P[7][7]
    _feed(ekf, mode="idle", q_occupancy=0.0)
    assert ekf._x[7] == before
    assert ekf._P[7][7] == variance
    _feed(ekf, mode="idle", q_occupancy=1.0)
    assert ekf._x[7] != before
    assert ekf.is_param_learned(7)


def test_ekf_beta_f_is_gated_to_a_charged_air_node():
    ekf = _initialized_ekf()
    before = ekf._x[8]
    variance = ekf._P[8][8]
    _feed(ekf, mode="idle", q_rebound=0.0)
    assert ekf._x[8] == before
    assert ekf._P[8][8] == variance
    _feed(ekf, mode="idle", q_rebound=None)
    assert ekf._x[8] == before
    _feed(ekf, mode="idle", q_rebound=0.8)
    assert ekf._x[8] != before
    assert ekf.is_param_learned(8)


def test_ekf_beta_h_and_beta_c_are_gated_to_their_modes():
    ekf = _initialized_ekf()
    beta_h, beta_c = ekf._x[3], ekf._x[4]
    _feed(ekf, mode="idle")
    assert ekf._x[3] == beta_h
    assert ekf._x[4] == beta_c
    _feed(ekf, mode="heating")
    assert ekf._x[3] != beta_h
    assert ekf._x[4] == beta_c
    _feed(ekf, mode="cooling")
    assert ekf._x[4] != beta_c


def test_ekf_zero_power_heating_step_is_not_excited():
    ekf = ThermalEKF()
    ekf._initialized = True
    ekf._x[0] = 20.0
    ekf._predict_step(5.0, "heating", 0.05, power_fraction=0.0)
    assert ekf._excited[3] is False
    assert ekf._P[3][3] == ThermalEKF._P_INIT_BETA


# ---------------------------------------------------------------------------
# ThermalEKF: online convergence simulations
# ---------------------------------------------------------------------------


def test_ekf_converges_beta_h_and_beta_f_under_forced_air_heating():
    """Furnace runs plus the post-run settle teach the heating rate and the air/mass offset."""
    rng = random.Random(7)
    true_model = _physical_model()  # beta_h=6, beta_f=2.5, U=0.15, U_house=0.25
    ekf = ThermalEKF()
    tracker = ForcedAirReboundTracker()
    T, T_out, T_house = 20.0, 5.0, 19.0
    ekf.update(T, T_out, "idle", 1.0, T_house=T_house)
    for step in range(2400):
        mode = "heating" if step % 50 < 20 else "idle"
        mean_charge, _ = tracker.step("room", mode, 1.0)
        T = true_model.predict(
            T,
            T_out,
            6.0 if mode == "heating" else 0.0,
            1.0,
            mode=mode,
            T_house=T_house,
            q_rebound=mean_charge,
        )
        T += rng.gauss(0.0, 0.02)
        ekf.update(T, T_out, mode, 1.0, q_rebound=mean_charge, T_house=T_house)

    assert ekf._n_heating > 0
    assert ekf._n_idle > 0
    assert ekf._x[3] == pytest.approx(6.0, abs=0.8)
    assert ekf._x[8] == pytest.approx(2.5, abs=0.8)
    assert ekf._x[1] == pytest.approx(0.15, abs=0.05)
    # The blower never ran: beta_fan must still be its prior.
    assert ekf._x[5] == pytest.approx(ThermalEKF._DEFAULT_BETA_FAN)


def test_ekf_converges_beta_c_and_beta_f_under_forced_air_cooling():
    """Heat-pump cooling plus the post-run rebound teach the cooling rate and the offset."""
    rng = random.Random(11)
    true_model = RCModel(C=1.0, U=0.15, Q_heat=5.0, Q_cool=5.0, Q_fan=1.2, U_house=0.25, Q_fast=2.0)
    ekf = ThermalEKF()
    tracker = ForcedAirReboundTracker()
    T, T_out, T_house = 26.0, 32.0, 27.0
    ekf.update(T, T_out, "idle", 1.0, T_house=T_house)
    for step in range(2400):
        mode = "cooling" if step % 50 < 20 else "idle"
        mean_charge, _ = tracker.step("room", mode, 1.0)
        T = true_model.predict(
            T,
            T_out,
            -5.0 if mode == "cooling" else 0.0,
            1.0,
            mode=mode,
            T_house=T_house,
            q_rebound=mean_charge,
        )
        T += rng.gauss(0.0, 0.02)
        ekf.update(T, T_out, mode, 1.0, q_rebound=mean_charge, T_house=T_house)

    assert ekf._n_cooling > 0
    assert ekf._x[4] == pytest.approx(5.0, abs=0.8)
    assert ekf._x[8] == pytest.approx(2.0, abs=0.7)
    assert ekf._x[1] == pytest.approx(0.15, abs=0.05)


def test_ekf_converges_beta_fan_during_circulating_runs():
    """A fan-only run with the house warmer than the room identifies the blower mixing rate."""
    rng = random.Random(13)
    true_model = RCModel(C=1.0, U=0.15, Q_heat=5.0, Q_cool=4.0, Q_fan=1.5, U_house=0.25)
    ekf = ThermalEKF()
    T, T_out, T_house = 21.0, 5.0, 25.0
    ekf.update(T, T_out, "idle", 1.0, T_house=T_house)
    for step in range(600):
        mode = MODE_CIRCULATING if step % 50 < 20 else "idle"
        T = true_model.predict(T, T_out, 0.0, 1.0, mode=mode, T_house=T_house, u_fan=1.0)
        T += rng.gauss(0.0, 0.02)
        ekf.update(T, T_out, mode, 1.0, T_house=T_house, u_fan=1.0)

    assert ekf._n_circulating > 0
    assert ekf._x[5] == pytest.approx(1.5, abs=0.15)
    # No forced-air charge is tracked, so beta_f keeps its prior.
    assert ekf._x[8] == pytest.approx(ThermalEKF._DEFAULT_BETA_F)


def test_ekf_learns_alpha_and_heating_rate_from_plain_rc_data():
    """The coupled model reduces to the classic RC model without a house node."""
    true_model = RCModel(C=1.0, U=0.2, Q_heat=6.0, Q_cool=4.0, U_house=0.0)
    ekf = ThermalEKF()
    T, T_out = 22.0, 10.0
    ekf.update(T, T_out, "idle", 3.0)
    ekf._x[2] = 0.0  # this data set has no house node
    for _ in range(300):
        T = true_model.predict(T, T_out, 0.0, 3.0)
        ekf.update(T, T_out, "idle", 3.0)
    assert ekf._x[1] == pytest.approx(0.2, abs=0.03)

    # Resume heating from a known temperature so the jump is not learned as a rate.
    T = 18.0
    ekf.sync_temperature(T)
    for _ in range(300):
        T = true_model.predict(T, T_out, 6.0, 3.0)
        if T > 28.0:
            T = 18.0
            ekf.sync_temperature(T)
        ekf.update(T, T_out, "heating", 3.0)
    assert ekf._x[3] == pytest.approx(6.0, abs=0.8)


def test_ekf_prediction_bias_converges():
    """After training, one-step model predictions are accurate (bias < 0.1 degC)."""
    true_model = RCModel(C=1.0, U=0.2, Q_heat=6.0, Q_cool=4.0)
    ekf = ThermalEKF()
    T, T_out = 20.0, 10.0
    ekf.update(T, T_out, "idle", 5.0)
    for _ in range(200):
        T = true_model.predict(T, T_out, 0.0, 5.0)
        ekf.update(T, T_out, "idle", 5.0)
    for _ in range(100):
        T = true_model.predict(T, T_out, 6.0, 5.0)
        ekf.update(T, T_out, "heating", 5.0)

    errors = []
    for _ in range(30):
        actual = true_model.predict(T, T_out, 0.0, 5.0)
        predicted = ekf.get_model().predict(T, T_out, 0.0, 5.0)
        errors.append(abs(predicted - actual))
        ekf.update(actual, T_out, "idle", 5.0)
        T = actual
    assert sum(errors) / len(errors) < 0.1


# ---------------------------------------------------------------------------
# ThermalEKF: get_model and is_param_learned
# ---------------------------------------------------------------------------


def test_ekf_get_model_maps_every_parameter():
    ekf = ThermalEKF()
    ekf._x = [21.0, 0.5, 0.3, 2.5, 3.2, 1.1, 2.0, 0.4, 1.7]
    model = ekf.get_model()
    assert model.C == pytest.approx(1.0)
    assert model.U == pytest.approx(0.5)
    assert model.U_house == pytest.approx(0.3)
    assert model.Q_heat == pytest.approx(2.5)
    assert model.Q_cool == pytest.approx(3.2)
    assert model.Q_fan == pytest.approx(1.1)
    assert model.Q_solar == pytest.approx(2.0)
    assert model.Q_occupancy == pytest.approx(0.4)
    assert model.Q_fast == pytest.approx(1.7)


def test_ekf_get_model_floors_negative_states():
    ekf = ThermalEKF()
    ekf._x = [21.0, 0.0, -0.1, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    model = ekf.get_model()
    assert model.U == pytest.approx(ThermalEKF._ALPHA_MIN)
    assert model.U_house == 0.0
    assert model.Q_heat == 0.0
    assert model.Q_fan == 0.0


def test_ekf_is_param_learned_tracks_information():
    ekf = ThermalEKF()
    assert [ekf.is_param_learned(i) for i in range(9)] == [False] * 9
    assert ekf.is_param_learned(-1) is False
    assert ekf.is_param_learned(9) is False
    ekf.update(21.0, 10.0, "idle", 1.0)
    assert ekf.is_param_learned(0) is False  # initialization only
    ekf.update(21.0, 10.0, "idle", 1.0)
    assert ekf.is_param_learned(0) is True
    # Unobserved parameters keep their prior variance exactly.
    assert ekf.is_param_learned(4) is False
    for i in range(40):
        ekf.update(21.0 + 0.1 * (i % 3), 10.0, "heating", 1.0, T_house=18.0)
    assert ekf.is_param_learned(3) is True
    assert ekf.is_param_learned(4) is False


# ---------------------------------------------------------------------------
# ThermalEKF: confidence and prediction_std
# ---------------------------------------------------------------------------


def test_ekf_confidence_needs_three_updates():
    ekf = ThermalEKF()
    ekf.update(20.0, 10.0, "idle", 3.0)  # initialization only
    ekf.update(20.0, 10.0, "idle", 3.0)
    ekf.update(20.0, 10.0, "idle", 3.0)
    assert ekf._n_updates == 2
    assert ekf.confidence == 0.0


def test_ekf_confidence_grows_with_idle_and_active_data():
    true_model = _physical_model()
    ekf = ThermalEKF()
    T, T_out = 20.0, 5.0
    ekf.update(T, T_out, "idle", 3.0)
    for _ in range(120):
        T = true_model.predict(T, T_out, 0.0, 3.0)
        ekf.update(T, T_out, "idle", 3.0)
    for _ in range(60):
        T = true_model.predict(T, T_out, 6.0, 3.0)
        ekf.update(T, T_out, "heating", 3.0)
    assert ekf._n_idle >= 60
    assert ekf._n_heating >= 2
    assert ekf.confidence > 0.6


def test_ekf_confidence_includes_cooling_data():
    true_model = _physical_model()
    ekf = ThermalEKF()
    T, T_out = 26.0, 32.0
    ekf.update(T, T_out, "idle", 3.0)
    for _ in range(80):
        T = true_model.predict(T, T_out, 0.0, 3.0)
        ekf.update(T, T_out, "idle", 3.0)
    for _ in range(40):
        T = true_model.predict(T, T_out, -4.5, 3.0)
        ekf.update(T, T_out, "cooling", 3.0)
    assert ekf._n_cooling >= 2
    assert ekf.confidence > 0.0
    assert ekf.confidence > 0.4


def test_ekf_confidence_includes_circulating_data_in_the_active_share():
    true_model = _physical_model()
    ekf = ThermalEKF()
    T, T_out = 21.0, 5.0
    ekf.update(T, T_out, "idle", 3.0)
    for _ in range(120):
        T = true_model.predict(T, T_out, 0.0, 3.0)
        ekf.update(T, T_out, "idle", 3.0)
    without_fan = ekf.confidence
    ekf._n_circulating = 20  # as if an hour of fan-only runs had been seen
    assert ekf.confidence > without_fan
    assert ekf.confidence > 0.6


@pytest.mark.parametrize(("predicted_std", "accuracy"), [(0.10, 1.0), (0.35, 0.5), (0.60, 0.0)])
def test_ekf_confidence_maps_prediction_std_onto_the_accuracy_factor(
    monkeypatch: pytest.MonkeyPatch, predicted_std: float, accuracy: float
) -> None:
    ekf = ThermalEKF()
    ekf._n_updates = 100
    ekf._n_idle = 60  # idle_frac = 1.0, no active data -> data_factor = 0.5
    monkeypatch.setattr(ekf, "prediction_std", lambda *args, **kwargs: predicted_std)
    assert ekf.confidence == pytest.approx(0.5 * (0.3 + 0.7 * accuracy))


def test_ekf_prediction_std_shrinks_with_training():
    true_model = _physical_model()
    ekf = ThermalEKF()
    initial = ekf.prediction_std(0.0, 20.0, 10.0, 5.0)
    T, T_out = 20.0, 10.0
    ekf.update(T, T_out, "idle", 5.0)
    for _ in range(100):
        T = true_model.predict(T, T_out, 0.0, 5.0)
        ekf.update(T, T_out, "idle", 5.0)
    assert ekf.prediction_std(0.0, 20.0, 10.0, 5.0) < initial


def test_ekf_prediction_std_is_mode_aware():
    true_model = _physical_model()
    ekf = ThermalEKF()
    T, T_out = 22.0, 10.0
    ekf.update(T, T_out, "idle", 5.0)
    for _ in range(100):
        T = true_model.predict(T, T_out, 0.0, 5.0)
        ekf.update(T, T_out, "idle", 5.0)
    model = ekf.get_model()
    std_idle = ekf.prediction_std(0.0, 20.0, 10.0, 5.0)
    std_heating = ekf.prediction_std(model.Q_heat, 20.0, 10.0, 5.0)
    assert std_heating > std_idle * 1.2


@pytest.mark.parametrize("dt", [0.0, -1.0])
def test_ekf_prediction_std_with_non_positive_dt(dt: float):
    ekf = ThermalEKF()
    assert ekf.prediction_std(0.0, 20.0, 10.0, dt) == pytest.approx(math.sqrt(ekf._P[0][0]))


def test_ekf_prediction_std_infers_cooling_from_a_negative_power():
    ekf = ThermalEKF()
    ekf.update(24.0, 30.0, "cooling", 3.0)
    explicit = ekf.prediction_std(-3.5, 20.0, 25.0, 5.0, mode="cooling")
    inferred = ekf.prediction_std(-3.5, 20.0, 25.0, 5.0)
    assert inferred == pytest.approx(explicit)


def test_ekf_prediction_std_accepts_the_coupled_inputs():
    ekf = ThermalEKF()
    ekf.update(21.0, 10.0, "idle", 3.0)
    std = ekf.prediction_std(
        0.0,
        20.0,
        10.0,
        5.0,
        q_solar=0.5,
        q_occupancy=1.0,
        q_rebound=-0.4,
        mode=MODE_CIRCULATING,
        T_house=25.0,
        u_fan=1.0,
        door_mix_factor=0.8,
        door_supply_factor=0.9,
    )
    assert std > 0.0
    assert math.isfinite(std)


# ---------------------------------------------------------------------------
# ThermalEKF: window-open learning
# ---------------------------------------------------------------------------


def test_ekf_window_open_learns_k_window():
    ekf = ThermalEKF(22.0)
    for _ in range(5):
        ekf.update(22.0, 10.0, "idle", 0.5)
    assert ekf._k_window == ThermalEKF._K_WINDOW_DEFAULT
    T = 22.0
    for _ in range(10):
        T = T + (5.0 - T) * 0.05
        ekf.update_window_open(T, 5.0, 0.5)
    assert ekf._k_window_n == 10
    assert ekf._k_window != ThermalEKF._K_WINDOW_DEFAULT
    assert ThermalEKF._K_WINDOW_MIN <= ekf._k_window <= ThermalEKF._K_WINDOW_MAX


def test_ekf_window_open_does_not_touch_parameters():
    ekf = ThermalEKF(22.0)
    for _ in range(5):
        ekf.update(22.0, 10.0, "idle", 0.5)
    params = list(ekf._x[1:])
    T = 22.0
    for _ in range(20):
        T -= 0.5
        ekf.update_window_open(T, 5.0, 0.5)
    assert ekf._x[1:] == params


def test_ekf_window_open_tracks_temperature():
    ekf = ThermalEKF(22.0)
    ekf.update(22.0, 10.0, "idle", 0.5)
    ekf.update_window_open(20.5, 5.0, 0.5)
    assert ekf._x[0] == 20.5
    ekf.update_window_open(19.0, 5.0, 0.5)
    assert ekf._x[0] == 19.0


def test_ekf_window_open_initializes():
    ekf = ThermalEKF(15.0)
    ekf.update_window_open(22.0, 5.0, 1.0)
    assert ekf._initialized
    assert ekf._x[0] == 22.0
    assert ekf._k_window_n == 0


def test_ekf_window_open_small_dt_is_skipped():
    ekf = ThermalEKF(22.0)
    ekf.update(22.0, 10.0, "idle", 0.5)
    ekf.update_window_open(21.0, 5.0, 0.1)
    assert ekf._k_window_n == 0
    assert ekf._x[0] == 22.0


def test_ekf_window_open_small_delta_t_tracks_without_learning():
    ekf = ThermalEKF(20.0)
    ekf.update(20.0, 10.0, "idle", 0.5)
    k_before = ekf._k_window
    ekf.update_window_open(20.05, 20.0, 1.0)
    assert ekf._k_window == k_before
    assert ekf._x[0] == 20.05


def test_ekf_window_open_invalid_ratio_tracks_without_learning():
    ekf = ThermalEKF(22.0)
    ekf.update(22.0, 10.0, "idle", 0.5)
    k_before = ekf._k_window
    ekf.update_window_open(23.0, 10.0, 0.5)  # moving away from the outdoor temperature
    assert ekf._k_window == k_before
    assert ekf._x[0] == 23.0
    ekf.update_window_open(5.0, 5.0, 0.5)  # ratio == 0
    assert ekf._k_window == k_before


def test_ekf_window_open_second_sample_uses_ema():
    ekf = ThermalEKF(22.0)
    ekf.update(22.0, 10.0, "idle", 0.5)
    ekf._k_window = 20.0
    ekf._k_window_n = 5
    ekf.update_window_open(21.0, 10.0, 0.5)  # ratio 0.9167 -> moderate k_obs
    assert ekf._k_window != 20.0
    assert ThermalEKF._K_WINDOW_MIN <= ekf._k_window <= ThermalEKF._K_WINDOW_MAX


# ---------------------------------------------------------------------------
# Serialization and legacy migration
# ---------------------------------------------------------------------------


def test_ekf_serialization_roundtrip_preserves_the_9_state_filter():
    true_model = _physical_model()
    ekf = ThermalEKF()
    T, T_out = 20.0, 5.0
    ekf.update(T, T_out, "idle", 3.0, T_house=19.0)
    for step in range(40):
        mode = ("idle", "heating", MODE_CIRCULATING)[step % 3]
        T = true_model.predict(T, T_out, 4.0 if mode == "heating" else 0.0, 3.0, mode=mode, T_house=19.0)
        ekf.update(T, T_out, mode, 3.0, q_rebound=0.2, T_house=19.0, u_fan=1.0)
    data = ekf.to_dict()
    restored = ThermalEKF.from_dict(data)
    assert data["ekf_version"] == 8
    assert len(data["x"]) == 9
    assert len(data["P"]) == 9
    assert restored._x == pytest.approx(ekf._x, rel=1e-9)
    for i in range(9):
        for j in range(9):
            assert restored._P[i][j] == pytest.approx(ekf._P[i][j], rel=1e-9)
    assert restored._n_updates == ekf._n_updates
    assert restored._n_heating == ekf._n_heating
    assert restored._n_circulating == ekf._n_circulating
    assert restored._n_idle == ekf._n_idle
    assert restored._initialized
    assert restored.confidence == pytest.approx(ekf.confidence)


def test_ekf_to_dict_writes_current_version_and_circulating_counter():
    ekf = _initialized_ekf()
    ekf.update(21.0, 10.0, MODE_CIRCULATING, 1.0, T_house=25.0, u_fan=1.0)
    data = ekf.to_dict()
    assert data["ekf_version"] == 8
    assert data["n_circulating"] == 1


def test_ekf_from_dict_migrates_the_v6_beta_f_block():
    data = {
        "ekf_version": 6,
        "x": [19.5, 0.09, 5.0, 3.0, 1.2, 0.4],
        "P": [[0.1 if i == j else 0.0 for j in range(6)] for i in range(6)],
        "beta_f": {"x": 1.7, "P": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25]},
        "n_updates": 420,
        "n_heating": 40,
        "n_cooling": 80,
        "n_idle": 300,
        "applicable_modes": ["heating", "cooling", "idle"],
        "last_mode": "cooling",
        "initialized": True,
    }
    ekf = ThermalEKF.from_dict(data)
    assert ekf._x == pytest.approx([19.5, 0.09, 0.20, 5.0, 3.0, 0.80, 1.2, 0.4, 1.7])
    # Learned values keep their own variance; the two new states start at the prior.
    assert ekf._P[0][0] == pytest.approx(0.1)
    assert ekf._P[1][1] == pytest.approx(0.1)
    assert ekf._P[3][3] == pytest.approx(0.1)
    assert ekf._P[4][4] == pytest.approx(0.1)
    assert ekf._P[6][6] == pytest.approx(0.1)
    assert ekf._P[7][7] == pytest.approx(0.1)
    assert ekf._P[8][8] == pytest.approx(0.25)
    assert ekf._P[2][2] == pytest.approx(ThermalEKF._P_INIT_ALPHA_HOUSE)
    assert ekf._P[5][5] == pytest.approx(ThermalEKF._P_INIT_BETA_FAN)
    assert ekf._n_updates == 420
    assert ekf._n_circulating == 0
    assert ekf._last_mode == "cooling"
    assert ekf._applicable_modes == {"heating", "cooling", "idle"}


def test_ekf_from_dict_migrates_the_5_state_layout():
    data = {
        "ekf_version": 3,
        "x": [20.0, 0.15, 3.0, 4.0, 0.5],
        "P": [[0.5 if i == j else 0.0 for j in range(5)] for i in range(5)],
        "n_updates": 100,
        "initialized": True,
    }
    ekf = ThermalEKF.from_dict(data)
    assert ekf._x == pytest.approx([20.0, 0.15, 0.20, 3.0, 4.0, 0.80, 0.5, 0.3, 0.3])
    assert ekf._P[5][5] == pytest.approx(ThermalEKF._P_INIT_BETA_FAN)
    assert ekf._P[7][7] == pytest.approx(ThermalEKF._P_INIT_BETA_O)
    assert ekf._P[8][8] == pytest.approx(ThermalEKF._P_INIT_BETA_F)
    assert ekf._n_updates == 100


def test_ekf_from_dict_migrates_a_full_7_state_list():
    data = {
        "ekf_version": 6,
        "x": [20.0, 0.1, 5.0, 3.0, 1.0, 0.4, 1.5],
        "P": [[0.2 if i == j else 0.0 for j in range(7)] for i in range(7)],
        "n_updates": 10,
        "initialized": True,
    }
    ekf = ThermalEKF.from_dict(data)
    assert ekf._x == pytest.approx([20.0, 0.1, 0.20, 5.0, 3.0, 0.80, 1.0, 0.4, 1.5])
    assert ekf._P[8][8] == pytest.approx(0.2)


def test_ekf_from_dict_ignores_a_malformed_beta_f_block():
    base = {
        "ekf_version": 6,
        "x": [20.0, 0.1, 5.0, 3.0, 1.0, 0.4],
        "P": [[0.2 if i == j else 0.0 for j in range(6)] for i in range(6)],
        "n_updates": 10,
    }
    for fast in ("bad", {"x": "bad", "P": []}):
        ekf = ThermalEKF.from_dict({**base, "beta_f": fast})
        assert ekf._x[8] == 0.3  # no usable value -> default
        assert ekf._P[8][8] == pytest.approx(ThermalEKF._P_INIT_BETA_F)
    # The value survives, but a too-short covariance row is ignored.
    ekf = ThermalEKF.from_dict({**base, "beta_f": {"x": 1.5, "P": [1.0, 2.0, 3.0]}})
    assert ekf._x[8] == pytest.approx(1.5)
    assert ekf._P[8][8] == pytest.approx(ThermalEKF._P_INIT_BETA_F)


def test_ekf_from_dict_handles_garbage_input():
    ekf = ThermalEKF.from_dict({})
    assert ekf._x == pytest.approx(ThermalEKF()._x)
    assert not ekf._initialized

    ekf = ThermalEKF.from_dict({"x": "bad", "P": "bad"})
    assert ekf._x == pytest.approx(ThermalEKF()._x)

    # Too few numeric entries -> treated as no saved state.
    ekf = ThermalEKF.from_dict({"x": [20.0, 0.1, 2.0, "bad"], "P": []})
    assert ekf._x == pytest.approx(ThermalEKF()._x)

    # Non-square P -> the prior covariance.
    ekf = ThermalEKF.from_dict({"x": [20.0, 0.1, 2.0, 2.0, 0.5], "P": [[1.0, 0.0], [0.0, 1.0]]})
    assert ekf._x[0] == pytest.approx(20.0)
    assert ekf._P[0][0] == pytest.approx(ThermalEKF._P_INIT_T)


def test_ekf_from_dict_keeps_metadata_and_ignores_bad_metadata():
    ekf = ThermalEKF()
    ekf._k_window = 8.5
    ekf._k_window_n = 42
    ekf.note_confidence_day("2026-01-01", True)
    data = ekf.to_dict()
    restored = ThermalEKF.from_dict(data)
    assert restored._k_window == 8.5
    assert restored._k_window_n == 42
    assert restored.confidence_history == data["confidence_history"]

    loaded = ThermalEKF.from_dict({**data, "k_window": "bad", "k_window_n": "bad"})
    assert loaded._k_window == ThermalEKF._K_WINDOW_DEFAULT
    assert loaded._k_window_n == 0
    loaded = ThermalEKF.from_dict({**data, "confidence_history": None, "last_update_at": "bad"})
    assert loaded.confidence_history == []
    assert loaded._last_update_at is None
    loaded = ThermalEKF.from_dict({**data, "n_updates": "bad", "n_heating": None, "last_mode": 3})
    assert loaded._n_updates == 0
    assert loaded._n_heating == 0
    assert loaded._last_mode is None

    broken = {**data, "applicable_modes": "bad"}
    assert ThermalEKF.from_dict(broken)._applicable_modes == ThermalEKF()._applicable_modes
    broken = {**data, "applicable_modes": [1, 2]}
    assert ThermalEKF.from_dict(broken)._applicable_modes == ThermalEKF()._applicable_modes
    broken = {**data, "confidence_history": [["2026-01-02", 0.5, 10, True, 0.2, 4], ["bad"], "junk"]}
    assert len(ThermalEKF.from_dict(broken).confidence_history) == 1
    broken = {**data, "forecast_error": {"day": 1, "sum": "bad", "n": "bad"}}
    assert ThermalEKF.from_dict(broken)._err_day is None


def test_ekf_from_dict_folds_legacy_forecast_error_and_history():
    data = {
        "x": [20.0, 0.12, 4.0, 3.5, 0.5, 0.3],
        "P": [[0.1 if i == j else 0.0 for j in range(6)] for i in range(6)],
        "ekf_version": 6,
        "forecast_error": {"day": "2026-02-01", "sum": 1.5, "n": 3},
        "confidence_history": [["2026-01-31", 0.4, 10, True, 0.2, 5]],
        "last_update_at": 1_700_000_000.0,
    }
    ekf = ThermalEKF.from_dict(data)
    assert ekf._err_day == "2026-02-01"
    assert ekf._err_sum == pytest.approx(1.5)
    assert ekf._err_n == 3
    assert ekf.confidence_history == [["2026-01-31", 0.4, 10, True, 0.2, 5]]
    assert ekf._last_update_at == pytest.approx(1_700_000_000.0)


def test_ekf_from_dict_resets_a_legacy_model_with_alpha_at_the_bound():
    data = {
        "ekf_version": 6,
        "x": [19.8, 2.0, 65.0, 20.0, 5.0, 2.0],
        "P": [[0.1 if i == j else 0.0 for j in range(6)] for i in range(6)],
        "n_updates": 10_000,
        "n_heating": 50,
        "n_idle": 9950,
        "applicable_modes": ["heating", "idle"],
        "last_mode": "idle",
        "initialized": True,
        "k_window": 8.5,
        "k_window_n": 42,
    }
    ekf = ThermalEKF.from_dict(data)
    assert ekf._x[1:] == pytest.approx([0.12, 0.20, 4.0, 3.5, 0.80, 0.5, 0.3, 0.3])
    assert ekf._x[0] == pytest.approx(19.8)  # temperature state preserved
    assert ekf._P[0][0] == pytest.approx(0.1)
    for i in range(9):
        for j in range(9):
            if i != j:
                assert ekf._P[i][j] == 0.0
    assert ekf._P[1][1] == pytest.approx(ThermalEKF._P_INIT_ALPHA)
    assert ekf._P[2][2] == pytest.approx(ThermalEKF._P_INIT_ALPHA_HOUSE)
    assert ekf._P[5][5] == pytest.approx(ThermalEKF._P_INIT_BETA_FAN)
    assert ekf._P[8][8] == pytest.approx(ThermalEKF._P_INIT_BETA_F)
    assert ekf._n_updates == 10_000
    assert ekf._n_heating == 50
    assert ekf._n_idle == 9950
    assert ekf._applicable_modes == {"heating", "idle"}
    assert ekf._last_mode == "idle"
    assert ekf._k_window == pytest.approx(8.5)
    assert ekf._k_window_n == 42


def test_ekf_from_dict_does_not_reset_a_current_model_at_the_bound():
    ekf = ThermalEKF()
    ekf._x[1] = ThermalEKF._ALPHA_MAX
    ekf._x[3] = 5.0
    data = ekf.to_dict()
    loaded = ThermalEKF.from_dict(data)
    assert loaded._x[1] == pytest.approx(ThermalEKF._ALPHA_MAX)
    assert loaded._x[3] == pytest.approx(5.0)


@pytest.mark.parametrize("version", [None, "7", 4.5])
def test_ekf_from_dict_non_int_version_is_treated_as_legacy(version: object) -> None:
    data = {
        "ekf_version": version,
        "x": [20.0, 2.0, 0.5, 1.6, 0.4, 0.1],
        "P": [[0.1 if i == j else 0.0 for j in range(6)] for i in range(6)],
        "n_updates": 100,
        "initialized": True,
    }
    ekf = ThermalEKF.from_dict(data)
    assert ekf._x[1] == pytest.approx(ThermalEKF._DEFAULT_ALPHA)


def test_ekf_from_dict_clamps_out_of_range_saved_parameters():
    data = {
        "ekf_version": 7,
        "x": [20.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
        "P": [[0.0] * 9 for _ in range(9)],
        "n_updates": 1,
    }
    ekf = ThermalEKF.from_dict(data)
    assert ekf._x[1] == pytest.approx(ThermalEKF._ALPHA_MAX)
    assert ekf._x[2] == pytest.approx(ThermalEKF._ALPHA_HOUSE_MAX)
    assert ekf._x[3] == pytest.approx(ThermalEKF._BETA_H_MAX)
    assert ekf._x[5] == pytest.approx(ThermalEKF._BETA_FAN_MAX)
    assert ekf._P[0][0] == pytest.approx(1e-10)  # PSD floor


def test_ekf_boost_covariance_widens_parameters_but_not_the_temperature():
    ekf = ThermalEKF()
    ekf.update(20.0, 10.0, "idle", 3.0)
    for _ in range(60):
        ekf.update(20.5, 10.0, "idle", 3.0)
    before = [ekf._P[i][i] for i in range(9)]
    ekf.boost_covariance(2.5)
    assert ekf._P[0][0] == pytest.approx(before[0])
    p_init = ThermalEKF._p_init_diag()
    for i in range(1, 9):
        assert before[i] <= ekf._P[i][i] <= p_init[i] + 1e-12
    for i in range(9):
        for j in range(9):
            assert ekf._P[i][j] == pytest.approx(ekf._P[j][i], abs=1e-12)
    # Repeated calls stop at the prior variance.
    for _ in range(5):
        ekf.boost_covariance(10.0)
    for i in range(1, 9):
        assert ekf._P[i][i] <= p_init[i] + 1e-12


def test_ekf_boost_covariance_handles_a_zero_variance():
    ekf = ThermalEKF()
    ekf._P[1][1] = 0.0
    ekf.boost_covariance()
    assert ekf._P[1][1] == pytest.approx(1e-10)


def test_ekf_confidence_history_records_and_folds_errors():
    ekf = ThermalEKF()
    ekf._n_updates = 10
    ekf._n_idle = 10
    ekf.note_confidence_day("2026-01-01", True)
    assert len(ekf.confidence_history) == 1
    assert ekf.confidence_history[0][0] == "2026-01-01"
    assert ekf.confidence_history[0][3] is True

    # Re-noting the same day replaces the entry (and folds the interval's error).
    ekf.update(20.0, 10.0, "idle", 3.0)
    ekf.update(21.0, 10.0, "idle", 3.0)
    ekf.note_confidence_day("2026-01-01", False)
    assert len(ekf.confidence_history) == 1
    assert ekf.confidence_history[0][3] is False
    assert ekf.confidence_history[0][4] is not None
    assert ekf.confidence_history[0][5] > 0

    # A new day closes the previous day's error columns; the new day starts empty.
    ekf.update(21.0, 10.0, "idle", 3.0)
    ekf.note_confidence_day("2026-01-02", True)
    assert len(ekf.confidence_history) == 2
    assert ekf.confidence_history[0][4] is not None
    ekf.update(21.0, 10.0, "idle", 3.0)
    ekf.note_confidence_day("2026-01-02", True)
    assert ekf.confidence_history[1][4] is not None


def test_ekf_confidence_history_is_trimmed():
    ekf = ThermalEKF()
    ekf._n_updates = 10
    ekf._n_idle = 10
    days = [f"2025-{month:02d}-{day:02d}" for month in (1, 2, 3) for day in range(1, 21)]
    ekf._confidence_history = [[day, 0.5, 10, True, None, 0] for day in days]
    ekf.note_confidence_day("2026-01-01", True)
    assert len(ekf.confidence_history) == tm.CONFIDENCE_HISTORY_DAYS
    assert ekf.confidence_history[-1][0] == "2026-01-01"


def test_ekf_day_error_without_samples():
    ekf = ThermalEKF()
    assert ekf._day_error() == (None, 0)


def test_ekf_last_update_time_survives_a_save(monkeypatch: pytest.MonkeyPatch):
    ekf = ThermalEKF()
    assert ekf._last_update_at is None
    monkeypatch.setattr(tm.time, "time", lambda: 1_700_000_000.0)
    ekf.update(20.0, 5.0, "idle", 3.0)
    ekf.update(19.98, 5.0, "idle", 3.0)
    assert ekf._last_update_at == 1_700_000_000.0
    ekf.sync_temperature(19.5)
    assert ekf._last_update_at == 1_700_000_000.0
    assert ThermalEKF.from_dict(ekf.to_dict())._last_update_at == 1_700_000_000.0


# ---------------------------------------------------------------------------
# Legacy residual-heat API compatibility (removed in a later wave)
# ---------------------------------------------------------------------------


def test_legacy_residual_properties_are_read_only_defaults():
    ekf = ThermalEKF()
    assert ekf.residual_scale == 1.0
    assert ekf.residual_scale_n == 0
    assert ekf.residual_tau is None
    # The deprecated setters exist so pre-Wave-3 callers do not crash.
    ekf.update_residual_scale(2.0)
    ekf.update_residual_tau(30.0, 20.0)
    assert ekf.residual_scale == 1.0
    assert ekf.residual_tau is None


def test_legacy_q_residual_is_accepted_but_ignored():
    plain = ThermalEKF()
    residual = ThermalEKF()
    for _ in range(10):
        plain.update(20.0, 10.0, "idle", 3.0)
        residual.update(20.0, 10.0, "idle", 3.0, q_residual=0.5)
    assert residual._x == pytest.approx(plain._x, abs=1e-12)
    assert residual.prediction_std(0.0, 20.0, 10.0, 5.0, q_residual=0.5) == pytest.approx(
        plain.prediction_std(0.0, 20.0, 10.0, 5.0)
    )


# ---------------------------------------------------------------------------
# RoomModelManager
# ---------------------------------------------------------------------------


def test_manager_get_or_create_and_room_ids():
    mgr = RoomModelManager()
    assert mgr.get_room_ids() == []
    estimator = mgr.get_estimator("living")
    assert isinstance(estimator, ThermalEKF)
    assert estimator.confidence == 0.0
    assert mgr.get_room_ids() == ["living"]


def test_manager_update_and_counts():
    mgr = RoomModelManager()
    mgr.update("living", 20.0, 5.0, "heating", 5.0)
    mgr.update("living", 20.5, 5.0, "heating", 5.0)
    mgr.update("living", 20.8, 5.0, MODE_CIRCULATING, 5.0, T_house=24.0, u_fan=1.0)
    mgr.update("living", 20.6, 5.0, "idle", 5.0, T_house=24.0)
    assert mgr.get_n_observations("living") == 3
    assert mgr.get_mode_counts("living") == (1, 1, 0)
    assert mgr.get_circulating_count("living") == 1
    assert mgr.get_mode_counts("unknown") == (0, 0, 0)
    assert mgr.get_circulating_count("unknown") == 0
    assert mgr.get_n_observations("unknown") == 0


def test_manager_passes_the_coupling_inputs_to_the_estimator():
    mgr = RoomModelManager()
    estimator = MagicMock()
    mgr._estimators["living"] = estimator
    mgr.update(
        "living",
        20.0,
        5.0,
        MODE_CIRCULATING,
        3.0,
        can_heat=True,
        can_cool=False,
        power_fraction=0.8,
        q_solar=0.2,
        q_residual=0.1,
        q_occupancy=1.0,
        q_rebound=-0.4,
        T_house=22.5,
        door_mix_factor=0.7,
        door_supply_factor=0.9,
        u_fan=0.5,
    )
    estimator.set_applicable_modes.assert_called_once_with(True, False)
    _, kwargs = estimator.update.call_args
    assert kwargs["T_house"] == 22.5
    assert kwargs["door_mix_factor"] == 0.7
    assert kwargs["door_supply_factor"] == 0.9
    assert kwargs["u_fan"] == 0.5
    assert kwargs["q_rebound"] == -0.4


def test_manager_house_data_changes_what_the_estimator_learns():
    with_house = RoomModelManager()
    without_house = RoomModelManager()
    for i in range(60):
        temp = 21.0 + 0.3 * (i % 2)
        with_house.update("r", temp, 10.0, "idle", 3.0, T_house=25.0)
        without_house.update("r", temp, 10.0, "idle", 3.0)
    assert with_house.get_estimator("r")._x[2] != without_house.get_estimator("r")._x[2]
    assert with_house.is_param_learned("r", 2) is True
    assert without_house.is_param_learned("r", 2) is False


def test_manager_is_param_learned_guards():
    mgr = RoomModelManager()
    assert mgr.is_param_learned("unknown", 1) is False
    mgr.update("r", 20.0, 10.0, "idle", 3.0)
    assert mgr.is_param_learned("r", 0) is False  # initialization only
    mgr.update("r", 20.0, 10.0, "idle", 3.0)
    assert mgr.is_param_learned("r", 0) is True
    assert mgr.is_param_learned("r", -1) is False
    assert mgr.is_param_learned("r", 9) is False


def test_manager_predict_and_model_accessors():
    mgr = RoomModelManager()
    model = mgr.get_model("living")
    result = mgr.predict("living", 20.0, 5.0, model.Q_heat, 10.0)
    assert isinstance(result, float)
    assert result > 20.0
    coupled = mgr.predict("living", 20.0, 5.0, 0.0, 60.0, T_house=25.0, mode="idle")
    assert coupled > mgr.predict("living", 20.0, 5.0, 0.0, 60.0)


def test_manager_prediction_std():
    mgr = RoomModelManager()
    assert mgr.get_prediction_std("unknown", 0.0, 20.0, 10.0, 5.0) == float("inf")
    mgr.update("r", 20.0, 10.0, "idle", 3.0)
    std = mgr.get_prediction_std(
        "r",
        0.0,
        20.0,
        10.0,
        5.0,
        q_solar=0.3,
        q_occupancy=1.0,
        q_rebound=0.2,
        mode=MODE_CIRCULATING,
        T_house=24.0,
        u_fan=1.0,
        door_mix_factor=0.8,
        door_supply_factor=0.9,
    )
    assert math.isfinite(std)
    assert std > 0.0


def test_manager_confidence_and_boost():
    mgr = RoomModelManager()
    assert mgr.get_confidence("unknown") == 0.0
    assert mgr.boost_learning("unknown") is None
    mgr.update("r", 20.0, 10.0, "idle", 3.0)
    mgr.update("r", 20.2, 10.0, "idle", 3.0)
    assert mgr.get_confidence("r") >= 0.0
    n = mgr.get_n_observations("r")
    assert mgr.boost_learning("r") == n


def test_manager_window_open_and_k_window():
    mgr = RoomModelManager()
    mgr.update("r", 22.0, 10.0, "idle", 0.5)
    mgr.update_window_open("r", 21.5, 5.0, 0.5)
    assert mgr.get_estimator("r")._initialized
    prediction = mgr.predict_window_open("r", 22.0, 5.0, 5.0)
    assert 0.0 < prediction < 22.0
    assert mgr.get_k_window("unknown") == ThermalEKF._K_WINDOW_DEFAULT
    assert mgr.get_k_window("r") == mgr.get_estimator("r").k_window


def test_manager_legacy_residual_api():
    mgr = RoomModelManager()
    assert mgr.get_residual_scale("unknown") == 1.0
    assert mgr.get_residual_tau("unknown") is None
    mgr.update("r", 20.0, 10.0, "idle", 3.0)
    mgr.update_residual_scale("r", 2.0)
    mgr.update_residual_tau("r", 30.0, 20.0)
    assert mgr.get_residual_scale("r") == 1.0
    assert mgr.get_residual_tau("r") is None


def test_manager_remove_room_and_sync():
    mgr = RoomModelManager()
    mgr.update("r", 20.0, 10.0, "idle", 3.0)
    mgr.sync_temperature("r", 18.0)
    assert mgr.get_estimator("r")._x[0] == 18.0
    mgr.remove_room("r")
    assert mgr.get_confidence("r") == 0.0
    assert mgr.get_room_ids() == []
    mgr.remove_room("r")  # idempotent


def test_manager_serialization_roundtrip():
    mgr = RoomModelManager()
    mgr.update("a", 20.0, 5.0, "heating", 5.0)
    mgr.update("a", 20.5, 5.0, "heating", 5.0)
    mgr.update("b", 24.0, 30.0, "cooling", 5.0)
    mgr.update("b", 23.6, 30.0, "cooling", 5.0)
    data = mgr.to_dict()
    restored = RoomModelManager.from_dict(data)
    assert set(restored.get_room_ids()) == {"a", "b"}
    assert restored.get_confidence("a") == pytest.approx(mgr.get_confidence("a"), abs=0.01)
    assert restored.get_confidence("b") == pytest.approx(mgr.get_confidence("b"), abs=0.01)
    assert restored.get_estimator("a")._x == pytest.approx(mgr.get_estimator("a")._x)


def test_manager_repr():
    mgr = RoomModelManager()
    mgr.update("room_a", 20.0, 5.0, "idle", 5.0)
    text = repr(mgr)
    assert "RoomModelManager" in text
    assert "room_a" in text


# ---------------------------------------------------------------------------
# Regressor signature and targeted relearning
# ---------------------------------------------------------------------------


def _ekf_with_parameters(**values: float) -> ThermalEKF:
    """An initialized EKF with the given parameters forced away from their defaults."""
    ekf = _initialized_ekf()
    for name, value in values.items():
        ekf._x[getattr(ThermalEKF, f"_STATE_{name.upper()}")] = value
    return ekf


def test_regressor_signature_is_legacy_without_azimuths():
    assert regressor_signature([], False) == {"solar": "legacy", "ref": "global"}
    assert regressor_signature([], True) == {"solar": "legacy", "ref": "local"}


def test_regressor_signature_canonicalises_the_azimuths():
    """Order, duplicates, floats and out-of-range values must not yield two signatures."""
    assert regressor_signature([225.4, 180, 180, 45], False)["solar"] == "45,180,225"
    assert regressor_signature([360, 0], False)["solar"] == "0"


def test_reset_parameter_restores_the_prior_and_drops_correlations():
    ekf = _initialized_ekf()
    idx = ThermalEKF._STATE_BETA_S
    ekf._x[idx] = 9.9
    ekf._P[idx][0] = 0.25
    ekf._P[0][idx] = 0.25
    untouched = list(ekf._x)

    ekf.reset_parameter(idx, ThermalEKF._DEFAULT_BETA_S)

    assert ekf._x[idx] == pytest.approx(ThermalEKF._DEFAULT_BETA_S)
    assert ekf._P[idx][idx] == pytest.approx(ThermalEKF._P_INIT_BETA_S)
    assert ekf._P[idx][0] == 0.0
    assert ekf._P[0][idx] == 0.0
    for other in range(ThermalEKF._N):
        if other != idx:
            assert ekf._x[other] == untouched[other]


def test_reset_parameter_refuses_the_temperature_state():
    """T is measured, not learned; resetting it would discard the current reading."""
    ekf = _initialized_ekf()
    with pytest.raises(ValueError, match="parameter index"):
        ekf.reset_parameter(ThermalEKF._STATE_T, 20.0)


def test_apply_regressor_signature_relearns_beta_s_when_solar_changes():
    ekf = _ekf_with_parameters(alpha_house=0.55, beta_s=7.5)

    reset = ekf.apply_regressor_signature(regressor_signature([180], False))

    assert reset == ("beta_s",)
    assert ekf._x[ThermalEKF._STATE_BETA_S] == pytest.approx(ThermalEKF._DEFAULT_BETA_S)
    assert ekf._x[ThermalEKF._STATE_ALPHA_HOUSE] == pytest.approx(0.55)
    assert ekf.regressor == {"solar": "180", "ref": "global"}


def test_apply_regressor_signature_relearns_the_house_pair_when_the_reference_changes():
    """alpha_house and beta_fan both couple to the reference, so both are meaningless after a change."""
    ekf = _ekf_with_parameters(beta_s=7.5)

    reset = ekf.apply_regressor_signature(regressor_signature([], True))

    assert reset == ("alpha_house", "beta_fan")
    assert ekf._x[ThermalEKF._STATE_ALPHA_HOUSE] == pytest.approx(ThermalEKF._DEFAULT_ALPHA_HOUSE)
    assert ekf._x[ThermalEKF._STATE_BETA_FAN] == pytest.approx(ThermalEKF._DEFAULT_BETA_FAN)
    assert ekf._x[ThermalEKF._STATE_BETA_S] == pytest.approx(7.5)


def test_apply_regressor_signature_is_a_noop_when_the_inputs_are_unchanged():
    """This is what keeps a room that configures nothing learning undisturbed."""
    ekf = _initialized_ekf()
    for _ in range(10):
        ekf.update(21.4, 5.0, "idle", 3.0, T_house=24.0, q_solar=0.6)
    before = list(ekf._x)

    reset = ekf.apply_regressor_signature(regressor_signature([], False))

    assert reset == ()
    assert ekf._x == pytest.approx(before)


def test_regressor_round_trips_through_persistence():
    ekf = _initialized_ekf()
    ekf.apply_regressor_signature(regressor_signature([225, 180], True))

    data = ekf.to_dict()

    assert data["regressor"] == {"solar": "180,225", "ref": "local"}
    assert ThermalEKF.from_dict(data).regressor == {"solar": "180,225", "ref": "local"}


def test_from_dict_without_a_regressor_falls_back_to_the_legacy_inputs():
    """Every model saved before v8 was trained on the unoriented GHI and the house mean."""
    ekf = ThermalEKF.from_dict({"ekf_version": 7, "x": [20.0] + [0.12] * 8, "n_updates": 5})

    assert ekf.regressor == {"solar": "legacy", "ref": "global"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, {"solar": "legacy", "ref": "global"}),
        ([], {"solar": "legacy", "ref": "global"}),
        ({"solar": 5, "ref": "sideways"}, {"solar": "legacy", "ref": "global"}),
        ({"ref": "local"}, {"solar": "legacy", "ref": "local"}),
    ],
)
def test_from_dict_ignores_a_malformed_regressor(raw: object, expected: dict[str, str]) -> None:
    data = {"ekf_version": 8, "x": [20.0] + [0.12] * 8, "n_updates": 5, "regressor": raw}

    assert ThermalEKF.from_dict(data).regressor == expected


def test_v7_model_with_pegged_alpha_is_not_reset_by_the_v8_migration():
    """The v8 bump is schema-only; the alpha-pegged recovery keeps its v<=6 bound."""
    data = {
        "ekf_version": 7,
        "x": [20.0, ThermalEKF._ALPHA_MAX, 0.2, 4.0, 3.5, 0.8, 0.5, 0.3, 0.3],
    }

    ekf = ThermalEKF.from_dict(data)

    assert ekf._x[ThermalEKF._STATE_ALPHA] == pytest.approx(ThermalEKF._ALPHA_MAX)
