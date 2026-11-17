"""Coupled forced-air RC thermal model with EKF parameter learning for RoomMind.

Physics (C normalized to 1, so every rate is a temperature rate [degC/h]):

    dT/dt = u_src - alpha_out * (T - T_outdoor) - gamma * (T - T_house)
    gamma  = door_mix_factor * (alpha_house + u_fan * beta_fan)
    u_src  = door_supply_factor * Q_active + beta_s * q_solar
             + beta_o * q_occupancy + beta_f * r_fast

Two nodes are coupled through the door: the room air (T) and the rest of the
house (T_house). A forced-air system conditions the room air, not the building
mass; the mass follows through ``alpha_house`` and, with the blower running,
through ``beta_fan``. The signed ``r_fast`` regressor (see
``control.rebound.fast_regressor``) captures the air leading the mass while a
furnace or heat pump runs and relaxing back after it stops, so the same
``beta_f`` learns both a heating rise/settle and a cooling drop/rebound.

Identifiability normalization (C = 1):
  alpha_out = U/C           [1/h]  outdoor heat loss rate
  alpha_house = U_house/C   [1/h]  house coupling rate
  beta_h = Q_heat/C         [degC/h]  heating rate
  beta_c = Q_cool/C         [degC/h]  cooling rate
  beta_fan = Q_fan/C        [degC/h]  blower mixing rate at full fan
  beta_s = Q_solar/C        [degC/h per kW/m²]  solar gain rate
  beta_o = Q_occupancy/C    [degC/h]  occupancy heat gain rate
  beta_f = Q_fast/C         [degC/h]  air/mass offset rate

RCModel solves the ODE analytically (exact for constant inputs over a step).
ThermalEKF learns the eight parameters online with an augmented 9-state
Extended Kalman Filter ``[T, alpha_out, alpha_house, beta_h, beta_c, beta_fan,
beta_s, beta_o, beta_f]``, gating each parameter to the excitation that makes
it observable. RoomModelManager provides per-room model access for the
coordinator.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from .rebound import (
    MODE_CIRCULATING,
    drive_for_mode,
    fast_regressor,
)

_LOGGER = logging.getLogger(__name__)

# Version stamp of the persisted EKF layout (9 states, coupled forced air).
# v8 adds the regressor signature, which records the solar input and the house
# reference the parameters were learned against, so a configuration change can
# relearn exactly the affected parameters instead of everything or nothing.
EKF_VERSION = 8

# Highest version whose alpha-pegged recovery still applies. It was written as
# ``version < EKF_VERSION`` when 7 was current, so it means "up to v6": bumping
# EKF_VERSION must not widen it, or a schema-only migration would reset the
# parameters of a perfectly healthy v7 model.
_ALPHA_RECOVERY_MAX_EKF_VERSION = 6

# The regressor a model is trained against, as persisted with it. "legacy" is
# the horizontal-plane GHI input against the whole-house mean: every model
# trained before orientation and room topology existed, and every room that
# configures neither.
LEGACY_SOLAR_REGRESSOR = "legacy"


def regressor_signature(solar_azimuths: Sequence[int], local_reference: bool) -> dict[str, str]:
    """Describe the inputs a room's parameters are fitted against.

    ``beta_s`` is only meaningful for one solar input, and ``alpha_house`` and
    ``beta_fan`` are only meaningful for one reference temperature. Recording
    which ones they were learned against lets a room relearn just those
    parameters when its configuration changes, and lets every room that
    configures nothing keep learning exactly as before.

    Args:
        solar_azimuths: exterior glazing azimuths in use, empty for the
            unoriented horizontal-plane estimate.
        local_reference: True when the room couples to a neighbour-weighted
            reference instead of the whole-house mean.

    Returns:
        ``{"solar": <signature>, "ref": "global"|"local"}``.
    """
    azimuths = sorted({int(round(float(azimuth))) % 360 for azimuth in solar_azimuths})
    solar = LEGACY_SOLAR_REGRESSOR if not azimuths else ",".join(str(azimuth) for azimuth in azimuths)
    return {"solar": solar, "ref": "local" if local_reference else "global"}


# Below this the exact solution's ``1 / lam`` gain is replaced by an Euler step.
_LAM_SMALL = 1e-6

# The house node only teaches the filter something once it is measurably off
# the room air; below this the two nodes are indistinguishable at sensor noise.
_HOUSE_EXCITE_MIN = 0.15

# The air/mass regressor only teaches beta_f when it carries a real fraction of
# a charged/discharged air node.
_FAST_EXCITE_MIN = 0.05


# ---------------------------------------------------------------------------
# RCModel -- deterministic coupled first-order RC thermal model
# ---------------------------------------------------------------------------


class RCModel:
    """Coupled air/house/mass RC thermal model with analytical solution.

    The room air exchanges heat with the outdoors (``U``), with the rest of the
    house through the door (``U_house``, scaled by ``door_mix_factor``), and --
    while the blower runs -- with the house air at ``Q_fan`` (scaled by
    ``u_fan``).  Active power ``Q_active`` enters through ``door_supply_factor``
    (duct losses, supply-air mixing), and the solar/occupancy/forced-air terms
    are added as temperature rates.

    The ODE has the closed-form solution over ``dt`` hours:

        lam   = U/C + door_mix_factor * (U_house/C + u_fan * Q_fan/C)
        A     = (U/C) * T_outdoor + gamma * T_house + u_src
        T(t)  = A/lam + (T_0 - A/lam) * exp(-lam * dt)

    ``T_house`` is optional: without it the house node sits at the room
    temperature, so the model reduces to the classic 1R1C RC model.
    """

    DEFAULT_C: float = 1.0  # normalized thermal capacitance
    DEFAULT_U: float = 0.12  # 1/h  -- outdoor heat loss rate (U/C)
    DEFAULT_Q_HEAT: float = 4.0  # degC/h -- heating rate (Q_heat/C)
    DEFAULT_Q_COOL: float = 3.5  # degC/h -- cooling rate (Q_cool/C)
    DEFAULT_Q_SOLAR: float = 0.0  # degC/h per kW/m² GHI -- solar gain
    DEFAULT_Q_OCCUPANCY: float = 0.0  # degC/h -- occupancy heat gain
    DEFAULT_Q_FAST: float = 0.0  # degC/h -- air/mass offset rate
    DEFAULT_U_HOUSE: float = 0.20  # 1/h  -- house coupling rate (U_house/C)
    DEFAULT_Q_FAN: float = 0.80  # degC/h -- blower mixing rate (Q_fan/C)

    def __init__(
        self,
        C: float = DEFAULT_C,
        U: float = DEFAULT_U,
        Q_heat: float = DEFAULT_Q_HEAT,
        Q_cool: float = DEFAULT_Q_COOL,
        Q_solar: float = DEFAULT_Q_SOLAR,
        Q_occupancy: float = DEFAULT_Q_OCCUPANCY,
        Q_fast: float = DEFAULT_Q_FAST,
        U_house: float = DEFAULT_U_HOUSE,
        Q_fan: float = DEFAULT_Q_FAN,
    ) -> None:
        self.C = C
        self.U = U
        self.Q_heat = Q_heat
        self.Q_cool = Q_cool
        self.Q_solar = Q_solar
        self.Q_occupancy = Q_occupancy
        self.Q_fast = Q_fast
        self.U_house = U_house
        self.Q_fan = Q_fan

    def predict(
        self,
        T_room: float,
        T_outdoor: float,
        Q_active: float,
        dt_minutes: float,
        *,
        q_solar: float = 0.0,
        q_residual: float = 0.0,
        q_occupancy: float = 0.0,
        q_rebound: float | None = None,
        T_house: float | None = None,
        mode: str | None = None,
        u_fan: float = 0.0,
        door_mix_factor: float = 1.0,
        door_supply_factor: float = 1.0,
    ) -> float:
        """Predict room temperature after *dt_minutes* using the analytical solution.

        Args:
            T_room: current room temperature [degC].
            T_outdoor: outdoor temperature [degC].
            Q_active: active power in C-normalized units [degC/h] (positive =
                heating, negative = cooling); with the default C = 1 this is the
                learned ``beta`` rate of the running system.
            dt_minutes: time step in minutes.
            q_solar: normalized solar irradiance (GHI/1000, 0-1).
            q_residual: legacy thermal-mass afterglow fraction (0-1). Only
                applied while ``Q_active == 0`` so the residual heat of a
                stopped radiator/UFH is not double-counted.
            q_occupancy: occupancy signal (0 = unoccupied, 1 = occupied).
            q_rebound: signed mean forced-air charge over the step
                (``rebound.mean_rebound``, -1..1). None leaves the air/mass
                term out (callers that do not track the charge).
            T_house: rest-of-house air temperature [degC]; None keeps the house
                node at the room temperature (classic 1R1C behaviour).
            mode: mode the step ran in ("heating", "cooling", "idle",
                "circulating"); None infers it from ``Q_active``/``u_fan``.
            u_fan: normalized blower command (0-1); only mixes in
                "circulating" mode.
            door_mix_factor: scaling of the door coupling (1.0 = open door).
            door_supply_factor: scaling of the HVAC supply into the air node.

        Returns:
            Predicted room temperature [degC], clamped to [0, 50].
        """
        if dt_minutes <= 0:
            return T_room
        dt_hours = dt_minutes / 60.0
        T_h = T_room if T_house is None else T_house
        eff_mode = mode
        if eff_mode is None:
            if Q_active > 0.0:
                eff_mode = "heating"
            elif Q_active < 0.0:
                eff_mode = "cooling"
            elif u_fan > 0.0:
                eff_mode = MODE_CIRCULATING
            else:
                eff_mode = "idle"
        u_fan_eff = u_fan if eff_mode == MODE_CIRCULATING else 0.0
        alpha_out = self.U / self.C
        alpha_house = self.U_house / self.C
        beta_fan = self.Q_fan / self.C
        gamma = door_mix_factor * (alpha_house + u_fan_eff * beta_fan)
        lam = alpha_out + gamma
        u_src = door_supply_factor * Q_active / self.C
        u_src += self.Q_solar * q_solar + self.Q_occupancy * q_occupancy
        u_src += self.Q_fast * fast_regressor(eff_mode, q_rebound)
        if Q_active == 0.0 and q_residual > 0.0:
            u_src += q_residual * (self.Q_heat / self.C)
        if abs(lam) < _LAM_SMALL:
            result = T_room + dt_hours * (u_src - alpha_out * (T_room - T_outdoor) + gamma * (T_h - T_room))
        else:
            T_eq = (alpha_out * T_outdoor + gamma * T_h + u_src) / lam
            result = T_eq + (T_room - T_eq) * math.exp(-lam * dt_hours)
        # Hard output clamp for safety
        return max(0.0, min(50.0, result))

    @staticmethod
    def cooling_drive(Q_active: float) -> float:
        """Legacy on/off rebound drive of *Q_active*: 1 while cooling, else 0."""
        return 1.0 if Q_active < 0.0 else 0.0

    @staticmethod
    def rebound_drive(mode: str | None) -> float:
        """Signed forced-air drive of *mode*: +1 heating, -1 cooling, 0 else."""
        return drive_for_mode(mode)

    def predict_window_open(
        self,
        T_room: float,
        T_outdoor: float,
        k_window: float,
        dt_minutes: float,
    ) -> float:
        """Predict temperature with window open (amplified heat exchange, Q=0).

        Uses U * k_window instead of U.  Works bidirectionally:
        winter (T_outdoor < T_room) gives cooling,
        summer (T_outdoor > T_room) gives warming.
        """
        if dt_minutes <= 0:
            return T_room
        dt_hours = dt_minutes / 60.0
        U_eff = self.U * k_window
        # Q=0 during window-open, so T_eq = T_outdoor
        T_eq = max(0.0, min(50.0, T_outdoor))
        decay = math.exp(-U_eff * dt_hours / self.C)
        result = T_eq + (T_room - T_eq) * decay
        return max(0.0, min(50.0, result))

    def predict_trajectory(
        self,
        T_room: float,
        T_outdoor_series: list[float],
        Q_active_series: list[float],
        dt_minutes: float,
        *,
        q_solar_series: list[float] | None = None,
        q_residual_series: list[float] | None = None,
        q_occupancy_series: list[float] | None = None,
        q_rebound_series: list[float] | None = None,
        mode_series: list[str | None] | None = None,
        u_fan_series: list[float] | None = None,
        T_house: float | None = None,
        door_mix_factor: float = 1.0,
        door_supply_factor: float = 1.0,
    ) -> list[float]:
        """Predict a temperature trajectory over multiple time steps.

        Each entry in *T_outdoor_series* / *Q_active_series* represents the
        constant value over that step (zero-order hold).

        Args:
            T_room: initial room temperature [degC].
            T_outdoor_series: outdoor temps per step [degC].
            Q_active_series: C-normalized active rate per step [degC/h].
            dt_minutes: duration of each step in minutes.
            q_solar_series: normalized solar irradiance per step (GHI/1000).
            q_residual_series: legacy residual heat fraction per step (0-1).
            q_occupancy_series: occupancy signal per step (0 or 1).
            q_rebound_series: signed mean forced-air charge per step (-1..1).
            mode_series: mode per step; None infers it per step.
            u_fan_series: normalized blower command per step (0-1).
            T_house: rest-of-house air temperature (constant over the horizon).
            door_mix_factor: scaling of the door coupling.
            door_supply_factor: scaling of the HVAC supply into the air node.

        Returns:
            List of len(series) + 1 temperatures (including the initial value).
        """
        if len(T_outdoor_series) != len(Q_active_series):
            raise ValueError("T_outdoor_series and Q_active_series must have the same length")
        n = len(T_outdoor_series)
        solar = q_solar_series or [0.0] * n
        residual = q_residual_series or [0.0] * n
        occupancy = q_occupancy_series or [0.0] * n
        modes = mode_series or [None] * n
        fans = u_fan_series or [0.0] * n
        trajectory = [T_room]
        T = T_room
        for i, (T_out, Q) in enumerate(zip(T_outdoor_series, Q_active_series, strict=False)):
            qs = solar[i] if i < len(solar) else 0.0
            qr = residual[i] if i < len(residual) else 0.0
            qo = occupancy[i] if i < len(occupancy) else 0.0
            qb = q_rebound_series[i] if q_rebound_series is not None and i < len(q_rebound_series) else None
            md = modes[i] if i < len(modes) else None
            fan = fans[i] if i < len(fans) else 0.0
            T = self.predict(
                T,
                T_out,
                Q,
                dt_minutes,
                q_solar=qs,
                q_residual=qr,
                q_occupancy=qo,
                q_rebound=qb,
                T_house=T_house,
                mode=md,
                u_fan=fan,
                door_mix_factor=door_mix_factor,
                door_supply_factor=door_supply_factor,
            )
            trajectory.append(T)
        return trajectory

    def to_dict(self) -> dict:
        """Serialize model parameters."""
        return {
            "C": self.C,
            "U": self.U,
            "Q_heat": self.Q_heat,
            "Q_cool": self.Q_cool,
            "Q_solar": self.Q_solar,
            "Q_occupancy": self.Q_occupancy,
            "Q_fast": self.Q_fast,
            "U_house": self.U_house,
            "Q_fan": self.Q_fan,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RCModel:
        """Restore model from a dict (older saves lack U_house/Q_fan)."""
        return cls(
            C=max(0.1, data.get("C", cls.DEFAULT_C)),
            U=max(1e-6, data.get("U", cls.DEFAULT_U)),
            Q_heat=max(0.0, data.get("Q_heat", cls.DEFAULT_Q_HEAT)),
            Q_cool=max(0.0, data.get("Q_cool", cls.DEFAULT_Q_COOL)),
            Q_solar=max(0.0, data.get("Q_solar", cls.DEFAULT_Q_SOLAR)),
            Q_occupancy=max(0.0, data.get("Q_occupancy", cls.DEFAULT_Q_OCCUPANCY)),
            Q_fast=max(0.0, data.get("Q_fast", cls.DEFAULT_Q_FAST)),
            U_house=max(0.0, data.get("U_house", cls.DEFAULT_U_HOUSE)),
            Q_fan=max(0.0, data.get("Q_fan", cls.DEFAULT_Q_FAN)),
        )

    def __repr__(self) -> str:
        return (
            f"RCModel(C={self.C:.2f}, U={self.U:.3f}, "
            f"Q_heat={self.Q_heat:.2f}, Q_cool={self.Q_cool:.2f}, "
            f"Q_solar={self.Q_solar:.2f}, Q_occupancy={self.Q_occupancy:.2f}, "
            f"Q_fast={self.Q_fast:.2f}, U_house={self.U_house:.3f}, Q_fan={self.Q_fan:.2f})"
        )


# ---------------------------------------------------------------------------
# ThermalEKF -- Extended Kalman Filter for online RC parameter learning
# ---------------------------------------------------------------------------


CONFIDENCE_HISTORY_DAYS = 60  # daily confidence snapshots kept per room


def _load_history_entry(e: list) -> list:
    """One stored day; entries from before the forecast error carry four fields."""
    error = e[4] if len(e) >= 6 and isinstance(e[4], (int, float)) else None
    steps = int(e[5]) if error is not None and isinstance(e[5], (int, float)) else 0
    return [e[0], float(e[1]), int(e[2]), bool(e[3]), None if error is None else float(error), steps]


class _StepTerms(NamedTuple):
    """Everything one predict step evaluates; shared by the Jacobian and tests."""

    dt_h: float
    T: float
    T_outdoor: float
    T_h: float
    gamma: float
    lam: float
    u_src: float
    r_fast: float
    u_fan_eff: float
    T_eq: float
    decay: float
    gain: float
    T_new: float
    euler: bool
    door_mix_factor: float
    door_supply_factor: float


class ThermalEKF:
    """Extended Kalman Filter for the coupled forced-air RC model.

    Augmented state vector x = [T, alpha_out, alpha_house, beta_h, beta_c,
    beta_fan, beta_s, beta_o, beta_f]:

      T            = room temperature [degC]  (directly measured)
      alpha_out    = U/C  outdoor heat loss rate [1/h]
      alpha_house  = U_house/C  house coupling rate [1/h]
      beta_h       = Q_heat/C  heating rate [degC/h]
      beta_c       = Q_cool/C  cooling rate [degC/h]
      beta_fan     = Q_fan/C  blower mixing rate at full fan [degC/h]
      beta_s       = Q_solar/C  solar gain rate [degC/h per kW/m²]
      beta_o       = Q_occupancy/C  occupancy gain rate [degC/h]
      beta_f       = Q_fast/C  air/mass offset rate [degC/h]

    The predict step uses the analytical coupled solution; parameters are
    random walks (constant + process noise) so they adapt slowly.  Each
    parameter is only updated (gain and process-noise inflation) while the
    step's excitation makes it observable:

      * alpha_house: idle step with ``|T_house - T_room| >= 0.15``
      * beta_fan: circulating step with the same house difference
      * beta_h / beta_c: a heating / cooling step
      * beta_s: sunlight, but not while heating (the emitter mismatch would
        be absorbed by the solar gain)
      * beta_o: occupied step
      * beta_f: a step whose forced-air charge is meaningfully off zero

    References:
      - Bacher & Madsen (2011): RC model selection for buildings
      - Radecki & Hencey (2016): Online model estimation (arXiv:1601.02947)
      - Zamani et al. (2025): UKF for RC model estimation (PMC 11798724)
    """

    EKF_VERSION: int = EKF_VERSION

    # State indices
    _STATE_T: int = 0
    _STATE_ALPHA: int = 1
    _STATE_ALPHA_HOUSE: int = 2
    _STATE_BETA_H: int = 3
    _STATE_BETA_C: int = 4
    _STATE_BETA_FAN: int = 5
    _STATE_BETA_S: int = 6
    _STATE_BETA_O: int = 7
    _STATE_BETA_F: int = 8

    # Innovation outlier threshold (in standard deviations)
    _ANOMALY_SIGMA: float = 4.0
    # Factor to inflate R when an outlier is detected (soft reject)
    _ANOMALY_R_INFLATE: float = 100.0

    # Parameter bounds (C=1 normalization: alpha=U/C, beta=Q/C)
    _ALPHA_MIN: float = 0.005  # time constant up to 200 h (very heavy building)
    _ALPHA_MAX: float = 1.5  # time constant down to 40 min -- defensive for residential rooms
    _ALPHA_HOUSE_MIN: float = 0.0  # sealed off from the rest of the house
    _ALPHA_HOUSE_MAX: float = 2.0  # open door, nearly one air volume
    _BETA_H_MIN: float = 0.2  # very weak heater relative to thermal mass
    _BETA_H_MAX: float = 30.0  # powerful heater in lightweight room
    _BETA_C_MIN: float = 0.2
    _BETA_C_MAX: float = 30.0
    _BETA_FAN_MIN: float = 0.05  # blower barely mixes
    _BETA_FAN_MAX: float = 6.0  # blower fully replaces the air volume each hour
    _BETA_S_MIN: float = 0.0  # interior rooms / north-facing -> 0
    _BETA_S_MAX: float = 50.0  # large south-facing sunroom
    _BETA_O_MIN: float = 0.0  # no occupancy gain
    _BETA_O_MAX: float = 10.0  # large occupancy gain (many people / small room)
    _BETA_F_MIN: float = 0.0  # no air/mass offset (forced air into a heavy, well-mixed room)
    _BETA_F_MAX: float = 20.0

    # Default initial parameter values (C=1 normalization)
    _DEFAULT_ALPHA: float = 0.12  # ~8 h time constant (moderate residential room)
    _DEFAULT_ALPHA_HOUSE: float = 0.20  # typical open-door coupling
    _DEFAULT_BETA_H: float = 4.0  # moderate furnace: T_eq = T_out + 33 degC
    _DEFAULT_BETA_C: float = 3.5  # moderate heat pump
    _DEFAULT_BETA_FAN: float = 0.80  # blower mixing at full fan
    _DEFAULT_BETA_S: float = 0.5  # small initial solar gain; learns from data
    _DEFAULT_BETA_O: float = 0.3  # ~100 W body heat, C=1 normalized
    _DEFAULT_BETA_F: float = 0.3  # forced air leads the mass by a few tenths of a K

    # Process noise (diagonal of Q_noise matrix); parameters are mode-gated:
    # only observable parameters receive drift noise.
    _Q_T: float = 0.01  # unmodeled disturbances (~0.1 degC/step)
    _Q_ALPHA: float = 1e-6  # building property drift (slow: seasonal, insulation)
    _Q_ALPHA_HOUSE: float = 1e-6  # door/house drift
    _Q_BETA: float = 1e-4  # heating and cooling power drift
    _Q_BETA_FAN: float = 5e-5  # blower mixing drift
    _Q_BETA_S: float = 1e-5  # solar gain drift
    _Q_BETA_O: float = 1e-6  # occupancy gain drift
    _Q_BETA_F: float = 1e-5  # air/mass offset drift

    # Measurement noise
    _R: float = 0.04  # sensor noise variance (0.2 degC std)

    # Initial covariance diagonal -- sized to prevent first-update overshoot.
    # Rule: K = F*P_init / (F^2*P_init + R) should stay below ~1.0.
    _P_INIT_T: float = 0.5
    _P_INIT_ALPHA: float = 0.04  # sigma 0.2 around default 0.12
    _P_INIT_ALPHA_HOUSE: float = 0.09  # sigma 0.3 around default 0.20
    _P_INIT_BETA: float = 9.0  # sigma 3 around defaults 4.0 / 3.5
    _P_INIT_BETA_FAN: float = 1.0  # sigma 1 around default 0.80
    _P_INIT_BETA_S: float = 2.0  # sigma 1.4 around default 0.5
    _P_INIT_BETA_O: float = 0.5  # sigma 0.7 around default 0.3
    _P_INIT_BETA_F: float = 2.0  # sigma 1.4 around default 0.3

    # Window-open heat exchange multiplier
    _K_WINDOW_DEFAULT: float = 5.0  # initial: 5x faster than closed
    _K_WINDOW_MIN: float = 1.5  # must be at least somewhat faster
    _K_WINDOW_MAX: float = 50.0  # physical upper bound
    _K_WINDOW_EMA_ALPHA: float = 0.05  # EMA smoothing factor
    _K_WINDOW_MIN_DT: float = 0.25  # min dt (minutes) for learning
    _K_WINDOW_MIN_DELTA_T: float = 0.1  # min |T_outdoor - T_room| for learning

    # Number of state dimensions
    _N: int = 9

    def __init__(self, T_init: float = 20.0) -> None:
        self._x: list[float] = [
            T_init,
            self._DEFAULT_ALPHA,
            self._DEFAULT_ALPHA_HOUSE,
            self._DEFAULT_BETA_H,
            self._DEFAULT_BETA_C,
            self._DEFAULT_BETA_FAN,
            self._DEFAULT_BETA_S,
            self._DEFAULT_BETA_O,
            self._DEFAULT_BETA_F,
        ]
        diag = self._p_init_diag()
        self._P: list[list[float]] = [[diag[i] if i == j else 0.0 for j in range(self._N)] for i in range(self._N)]
        self._n_updates: int = 0
        self._last_update_at: float | None = None  # Unix time of the last learning step
        self._n_heating: int = 0
        self._n_cooling: int = 0
        self._n_circulating: int = 0
        self._n_idle: int = 0
        self._applicable_modes: set[str] = {"heating", "cooling", "circulating", "idle"}
        self._last_mode: str | None = None
        self._initialized: bool = False
        self._k_window: float = self._K_WINDOW_DEFAULT
        self._k_window_n: int = 0
        # One entry per local day for the analytics chart:
        # [iso_date, confidence, n_updates, mpc_active, mean_abs_error_k | None, error_steps]
        self._confidence_history: list[list] = []
        # |innovation| of the day's regular updates (outliers left out), folded into the entry
        self._err_day: str | None = None
        self._err_sum: float = 0.0
        self._err_n: int = 0
        # Which states the last predict step could observe.
        self._excited: list[bool] = [True] * self._N
        # The inputs these parameters are fitted against (see regressor_signature).
        # A fresh model starts on the legacy ones, so a room that configures
        # neither orientation nor topology keeps learning exactly as it always did.
        self._regressor: dict[str, str] = {
            "solar": LEGACY_SOLAR_REGRESSOR,
            "ref": "global",
        }

    @classmethod
    def _p_init_diag(cls) -> list[float]:
        return [
            cls._P_INIT_T,
            cls._P_INIT_ALPHA,
            cls._P_INIT_ALPHA_HOUSE,
            cls._P_INIT_BETA,
            cls._P_INIT_BETA,
            cls._P_INIT_BETA_FAN,
            cls._P_INIT_BETA_S,
            cls._P_INIT_BETA_O,
            cls._P_INIT_BETA_F,
        ]

    # -- public API ----------------------------------------------------------

    @property
    def confidence(self) -> float:
        """Model reliability, 0.0 to 1.0.

        Multi-factor confidence that reflects actual model readiness:
        1. Data factor: how close to MPC-ready sample counts
        2. Accuracy factor: prediction std mapped relative to MPC threshold

        Combined as: 0.3 * data_factor + 0.7 * data_factor * accuracy_factor
        Data alone gives up to 30% (we have samples, model is learning).
        Accuracy multiplied by data adds the remaining 70%.

        This produces a meaningful 0-100% range:
        - 0%: no data
        - ~10-30%: learning (some data, covariance still wide)
        - ~40-60%: improving (enough idle data, active data growing)
        - ~80-100%: MPC-ready (enough data + accurate predictions)
        """
        if self._n_updates < 3:
            return 0.0

        # --- Data factor (0..1) ---
        # Mirrors MPC data gates: MIN_IDLE_UPDATES=60, MIN_ACTIVE_UPDATES=20.
        # Each update covers ~3 min (EKF_UPDATE_MIN_DT), so thresholds
        # correspond to ~3 h idle + ~1 h active real time.
        idle_frac = min(self._n_idle / 60.0, 1.0)

        # Active data: average across the active modes this room uses
        active_fracs: list[float] = []
        for count in (self._n_heating, self._n_cooling, self._n_circulating):
            if count >= 2:
                active_fracs.append(min(count / 20.0, 1.0))
        active_frac = sum(active_fracs) / len(active_fracs) if active_fracs else 0.0

        data_factor = 0.5 * idle_frac + 0.5 * active_frac

        # --- Accuracy factor (0..1) ---
        # Map weighted prediction std from [noise_floor, mpc_threshold] to [1.0, 0.0]
        noise_floor = 0.20
        mpc_threshold = 0.5  # MPC activation threshold

        # Frequency-weighted prediction std: each mode's contribution is
        # proportional to its sample count.  This prevents a rarely-used mode
        # from capping the entire accuracy metric via an unlearned parameter.
        std_weights: list[tuple[int, float]] = [
            (max(self._n_idle, 1), self.prediction_std(0.0, 20.0, 15.0, 5.0)),
        ]
        if self._n_heating >= 2:
            std_weights.append((self._n_heating, self.prediction_std(self._x[self._STATE_BETA_H], 20.0, 10.0, 5.0)))
        if self._n_cooling >= 2:
            std_weights.append((self._n_cooling, self.prediction_std(-self._x[self._STATE_BETA_C], 20.0, 25.0, 5.0)))
        total_w = sum(w for w, _ in std_weights)
        weighted_std = sum(w / total_w * s for w, s in std_weights)

        if weighted_std <= noise_floor:
            accuracy_factor = 1.0
        elif weighted_std >= mpc_threshold:
            accuracy_factor = 0.0
        else:
            accuracy_factor = 1.0 - (weighted_std - noise_floor) / (mpc_threshold - noise_floor)

        # Data alone contributes up to 30%, accuracy adds remaining 70%
        return data_factor * (0.3 + 0.7 * accuracy_factor)

    def prediction_std(
        self,
        Q_active: float,
        T_room: float,
        T_outdoor: float,
        dt_minutes: float,
        *,
        q_solar: float = 0.0,
        q_residual: float = 0.0,
        q_occupancy: float = 0.0,
        q_rebound: float | None = None,
        mode: str | None = None,
        T_house: float | None = None,
        u_fan: float = 1.0,
        door_mix_factor: float = 1.0,
        door_supply_factor: float = 1.0,
    ) -> float:
        """Prediction uncertainty in degC for a given operating point.

        Propagates the covariance matrix through the Jacobian to compute
        the temperature prediction variance at dt_minutes ahead.  ``mode``
        defaults to the one ``Q_active``'s sign implies; ``q_residual`` is
        accepted for backward compatibility and no longer enters the model.
        """
        if dt_minutes <= 0:
            return math.sqrt(max(self._P[0][0], 0.0))

        if mode is None:
            if Q_active > 0.0:
                mode = "heating"
            elif Q_active < 0.0:
                mode = "cooling"
            else:
                mode = "idle"

        dt_h = dt_minutes / 60.0
        F = self._compute_jacobian(
            self._x,
            T_outdoor,
            dt_h,
            mode,
            q_solar=q_solar,
            q_occupancy=q_occupancy,
            q_rebound=q_rebound,
            T_house=T_house,
            u_fan=u_fan,
            door_mix_factor=door_mix_factor,
            door_supply_factor=door_supply_factor,
        )

        # P_pred = F @ P @ F^T + Q_noise (only need element [0][0])
        # Compute F[0,:] @ P
        FP_row = [sum(F[0][k] * self._P[k][j] for k in range(self._N)) for j in range(self._N)]
        # Compute FP_row @ F[0,:]^T = F[0,:] @ P @ F[0,:]^T
        var = sum(FP_row[j] * F[0][j] for j in range(self._N))
        var += self._Q_T  # add process noise
        return math.sqrt(max(var, 0.0))

    @property
    def k_window(self) -> float:
        """Learned window heat-exchange multiplier."""
        return self._k_window

    @property
    def residual_scale(self) -> float:
        """Legacy read-only afterglow scale; the forced-air model no longer learns it."""
        return 1.0

    @property
    def residual_scale_n(self) -> int:
        """Legacy afterglow sample counter (always 0 now)."""
        return 0

    @property
    def residual_tau(self) -> float | None:
        """Legacy read-only afterglow decay time; the forced-air model no longer learns it."""
        return None

    def update_residual_scale(self, sample: float) -> None:
        """Deprecated no-op: kept so pre-Wave-3 callers do not crash on the API.

        The residual-heat (radiator/UFH) learner is being retired in favour of
        the forced-air model; the value stays at its read-only default.
        """

    def update_residual_tau(self, sample: float, profile_tau: float) -> None:
        """Deprecated no-op: kept so pre-Wave-3 callers do not crash on the API."""

    def update_window_open(
        self,
        T_measured: float,
        T_outdoor: float,
        dt_minutes: float,
        *,
        learn_k_window: bool = True,
    ) -> None:
        """Track temperature state and optionally learn k_window during window-open.

        Always updates _x[0] so the EKF resumes with current temperature.
        When *learn_k_window* is True, also computes the observed heat loss
        rate and updates the k_window multiplier.
        """
        if dt_minutes < self._K_WINDOW_MIN_DT:
            return
        if not self._initialized:
            self._x[0] = T_measured
            self._initialized = True
            return

        T_prev = self._x[0]
        delta_T = T_prev - T_outdoor

        if not learn_k_window or abs(delta_T) < self._K_WINDOW_MIN_DELTA_T:
            self._x[0] = T_measured
            return

        # From analytical 1R1C solution with Q=0:
        # T_new = T_outdoor + (T_prev - T_outdoor) * exp(-alpha_obs * dt_h)
        # => ratio = (T_measured - T_outdoor) / (T_prev - T_outdoor) = exp(-alpha_obs * dt_h)
        dt_h = dt_minutes / 60.0
        ratio = (T_measured - T_outdoor) / delta_T

        # ratio must be in (0, 1) for a valid observation
        if ratio <= 0.01 or ratio >= 1.0:
            self._x[0] = T_measured
            return

        alpha_obs = -math.log(ratio) / dt_h
        alpha_normal = max(self._x[self._STATE_ALPHA], _LAM_SMALL)

        k_obs = alpha_obs / alpha_normal
        k_obs = max(self._K_WINDOW_MIN, min(self._K_WINDOW_MAX, k_obs))

        if self._k_window_n == 0:
            self._k_window = k_obs
        else:
            ema = self._K_WINDOW_EMA_ALPHA
            self._k_window = ema * k_obs + (1.0 - ema) * self._k_window

        self._k_window_n += 1
        self._k_window = max(self._K_WINDOW_MIN, min(self._K_WINDOW_MAX, self._k_window))

        # Track temperature without updating EKF parameters
        self._x[0] = T_measured

    def set_applicable_modes(self, can_heat: bool, can_cool: bool) -> None:
        """Update which modes this room supports."""
        modes = {"idle", MODE_CIRCULATING}
        if can_heat:
            modes.add("heating")
        if can_cool:
            modes.add("cooling")
        self._applicable_modes = modes

    def update(
        self,
        T_measured: float,
        T_outdoor: float,
        mode: str,
        dt_minutes: float,
        *,
        power_fraction: float = 1.0,
        q_solar: float = 0.0,
        q_residual: float = 0.0,
        q_occupancy: float = 0.0,
        q_rebound: float | None = None,
        T_house: float | None = None,
        door_mix_factor: float = 1.0,
        door_supply_factor: float = 1.0,
        u_fan: float = 1.0,
    ) -> None:
        """Run one full EKF cycle: predict then update with measurement.

        Args:
            T_measured: current room temperature measurement [degC].
            T_outdoor: outdoor temperature [degC].
            mode: mode active during the interval ('heating', 'cooling',
                'circulating', 'idle').
            dt_minutes: time since last call [min].
            power_fraction: fraction of max heating/cooling power applied (0-1).
            q_solar: normalized solar irradiance (GHI/1000, 0-1).
            q_residual: legacy residual heat fraction; accepted for backward
                compatibility and no longer used by the forced-air model.
            q_occupancy: occupancy signal (0 = unoccupied, 1 = occupied).
            q_rebound: signed mean forced-air charge over the interval (-1..1);
                None leaves beta_f out (no charge tracked).
            T_house: rest-of-house air temperature [degC]; None keeps the house
                node at the room temperature.
            door_mix_factor: scaling of the door coupling (1.0 = open door).
            door_supply_factor: scaling of the HVAC supply into the air node.
            u_fan: normalized blower command (0-1), used in circulating mode.
        """
        if dt_minutes <= 0:
            return

        # On first call, initialize temperature state from measurement
        if not self._initialized:
            self._x[0] = T_measured
            self._initialized = True
            self._last_mode = mode
            return

        # `mode` represents the mode during the accumulated batch (set by the
        # EkfTrainingManager on flush).  Using self._last_mode would lag by one
        # call and propagate the next batch with the previous batch's dynamics,
        # which is wrong at mode transitions.
        predict_mode = mode
        dt_h = dt_minutes / 60.0

        # --- Predict step ---
        self._predict_step(
            T_outdoor,
            predict_mode,
            dt_h,
            power_fraction=power_fraction,
            q_solar=q_solar,
            q_occupancy=q_occupancy,
            q_rebound=q_rebound,
            T_house=T_house,
            door_mix_factor=door_mix_factor,
            door_supply_factor=door_supply_factor,
            u_fan=u_fan,
        )

        # --- Update step ---
        self._update_step(T_measured)

        # --- Enforce parameter bounds ---
        self._clamp_parameters()

        # --- Ensure P stays symmetric and PSD ---
        self._enforce_psd()

        # Update bookkeeping
        self._last_mode = mode
        self._n_updates += 1
        self._last_update_at = time.time()
        if predict_mode == "heating":
            self._n_heating += 1
        elif predict_mode == "cooling":
            self._n_cooling += 1
        elif predict_mode == MODE_CIRCULATING:
            self._n_circulating += 1
        else:
            self._n_idle += 1

    def sync_temperature(self, T_measured: float) -> None:
        """Set the temperature state to a measurement without learning from the jump."""
        self._x[0] = T_measured
        self._initialized = True

    def is_param_learned(self, state_idx: int) -> bool:
        """Whether *state_idx* has been learned from data (variance below its prior).

        Parameters the filter never observed keep their initial variance exactly
        (their gain and drift noise are gated off), so a shrunken variance is
        positive evidence that measurement data moved them.
        """
        if not 0 <= state_idx < self._N:
            return False
        if state_idx == self._STATE_T:
            return self._n_updates > 0
        return self._P[state_idx][state_idx] < self._p_init_diag()[state_idx] - 1e-12

    def get_model(self) -> RCModel:
        """Extract an RCModel (C=1 normalization) for the MPC optimizer.

        Since only ratios alpha=U/C, beta=Q/C are identifiable, we set C=1
        and map every learned rate onto its model parameter.
        """
        return RCModel(
            C=1.0,
            U=max(self._x[self._STATE_ALPHA], self._ALPHA_MIN),
            Q_heat=max(self._x[self._STATE_BETA_H], 0.0),
            Q_cool=max(self._x[self._STATE_BETA_C], 0.0),
            Q_solar=max(self._x[self._STATE_BETA_S], 0.0),
            Q_occupancy=max(self._x[self._STATE_BETA_O], 0.0),
            Q_fast=max(self._x[self._STATE_BETA_F], 0.0),
            U_house=max(self._x[self._STATE_ALPHA_HOUSE], 0.0),
            Q_fan=max(self._x[self._STATE_BETA_FAN], 0.0),
        )

    # -- EKF internals -------------------------------------------------------

    def _step_terms(
        self,
        x: list[float],
        T_outdoor: float,
        dt_h: float,
        mode: str,
        *,
        power_fraction: float = 1.0,
        q_solar: float = 0.0,
        q_occupancy: float = 0.0,
        q_rebound: float | None = None,
        T_house: float | None = None,
        door_mix_factor: float = 1.0,
        door_supply_factor: float = 1.0,
        u_fan: float = 1.0,
    ) -> _StepTerms:
        """Evaluate one analytical predict step for the state vector *x*.

        Pure function of the state and inputs, so the test suite can verify the
        analytical Jacobian against finite differences of exactly this map.
        """
        T = x[self._STATE_T]
        alpha_out = x[self._STATE_ALPHA]
        alpha_house = x[self._STATE_ALPHA_HOUSE]
        beta_fan = x[self._STATE_BETA_FAN]
        T_h = T if T_house is None else T_house
        u_fan_eff = u_fan if mode == MODE_CIRCULATING else 0.0
        gamma = door_mix_factor * (alpha_house + u_fan_eff * beta_fan)
        lam = alpha_out + gamma
        r_fast = fast_regressor(mode, q_rebound)

        q_active = 0.0
        if mode == "heating":
            q_active = x[self._STATE_BETA_H] * power_fraction
        elif mode == "cooling":
            q_active = -x[self._STATE_BETA_C] * power_fraction
        u_src = door_supply_factor * q_active
        u_src += x[self._STATE_BETA_S] * q_solar
        u_src += x[self._STATE_BETA_O] * q_occupancy
        u_src += x[self._STATE_BETA_F] * r_fast

        if abs(lam) < _LAM_SMALL:
            T_new = T + dt_h * (u_src - alpha_out * (T - T_outdoor) + gamma * (T_h - T))
            return _StepTerms(
                dt_h=dt_h,
                T=T,
                T_outdoor=T_outdoor,
                T_h=T_h,
                gamma=gamma,
                lam=lam,
                u_src=u_src,
                r_fast=r_fast,
                u_fan_eff=u_fan_eff,
                T_eq=0.0,
                decay=1.0,
                gain=dt_h,
                T_new=T_new,
                euler=True,
                door_mix_factor=door_mix_factor,
                door_supply_factor=door_supply_factor,
            )

        T_eq = (alpha_out * T_outdoor + gamma * T_h + u_src) / lam
        decay = math.exp(-lam * dt_h)
        gain = (1.0 - decay) / lam
        T_new = T_eq + (T - T_eq) * decay
        return _StepTerms(
            dt_h=dt_h,
            T=T,
            T_outdoor=T_outdoor,
            T_h=T_h,
            gamma=gamma,
            lam=lam,
            u_src=u_src,
            r_fast=r_fast,
            u_fan_eff=u_fan_eff,
            T_eq=T_eq,
            decay=decay,
            gain=gain,
            T_new=T_new,
            euler=False,
            door_mix_factor=door_mix_factor,
            door_supply_factor=door_supply_factor,
        )

    def _compute_jacobian(
        self,
        x: list[float],
        T_outdoor: float,
        dt_h: float,
        mode: str,
        *,
        power_fraction: float = 1.0,
        q_solar: float = 0.0,
        q_occupancy: float = 0.0,
        q_rebound: float | None = None,
        T_house: float | None = None,
        door_mix_factor: float = 1.0,
        door_supply_factor: float = 1.0,
        u_fan: float = 1.0,
    ) -> list[list[float]]:
        """Exact 9x9 Jacobian of the state transition at state *x*.

        Only the temperature row couples the parameters; parameter rows are the
        identity (random walks).  For a parameter that enters only ``u_src``::

            dT_new/dtheta = (1 - exp(-lam*dt_h)) / lam * du_src/dtheta

        For a parameter that enters both ``A`` and ``lambda`` with
        ``dA/dtheta = c*T_ref`` and ``dlambda/dtheta = c`` (alpha_out with
        ``c=1, T_ref=T_outdoor``; alpha_house with ``c=door_mix_factor,
        T_ref=T_house``; beta_fan with ``c=door_mix_factor*u_fan,
        T_ref=T_house``)::

            dT_new/dtheta = c * [ gain*(T_ref - T_eq) - dt_h*decay*(T - T_eq) ]

        which is verified against finite differences in the unit tests.
        """
        terms = self._step_terms(
            x,
            T_outdoor,
            dt_h,
            mode,
            power_fraction=power_fraction,
            q_solar=q_solar,
            q_occupancy=q_occupancy,
            q_rebound=q_rebound,
            T_house=T_house,
            door_mix_factor=door_mix_factor,
            door_supply_factor=door_supply_factor,
            u_fan=u_fan,
        )
        return self._jacobian_from_terms(terms, mode, power_fraction, q_solar, q_occupancy)

    def _jacobian_from_terms(
        self,
        terms: _StepTerms,
        mode: str,
        power_fraction: float,
        q_solar: float,
        q_occupancy: float,
    ) -> list[list[float]]:
        """Build the 9x9 Jacobian from an evaluated predict step."""
        N = self._N
        F: list[list[float]] = [[0.0] * N for _ in range(N)]
        for i in range(1, N):
            F[i][i] = 1.0

        supply = terms.door_supply_factor * power_fraction
        if terms.euler:
            F[0][self._STATE_T] = 1.0 - terms.lam * terms.dt_h
            F[0][self._STATE_ALPHA] = terms.dt_h * (terms.T_outdoor - terms.T)
            F[0][self._STATE_ALPHA_HOUSE] = terms.dt_h * terms.door_mix_factor * (terms.T_h - terms.T)
            F[0][self._STATE_BETA_H] = terms.dt_h * supply if mode == "heating" else 0.0
            F[0][self._STATE_BETA_C] = -terms.dt_h * supply if mode == "cooling" else 0.0
            F[0][self._STATE_BETA_FAN] = terms.dt_h * terms.door_mix_factor * terms.u_fan_eff * (terms.T_h - terms.T)
            F[0][self._STATE_BETA_S] = terms.dt_h * q_solar
            F[0][self._STATE_BETA_O] = terms.dt_h * q_occupancy
            F[0][self._STATE_BETA_F] = terms.dt_h * terms.r_fast
            return F

        excursion = terms.T - terms.T_eq
        coupling = terms.door_mix_factor * (
            terms.gain * (terms.T_h - terms.T_eq) - terms.dt_h * terms.decay * excursion
        )
        F[0][self._STATE_T] = terms.decay
        F[0][self._STATE_ALPHA] = terms.gain * (terms.T_outdoor - terms.T_eq) - terms.dt_h * terms.decay * excursion
        F[0][self._STATE_ALPHA_HOUSE] = coupling
        F[0][self._STATE_BETA_H] = terms.gain * supply if mode == "heating" else 0.0
        F[0][self._STATE_BETA_C] = -terms.gain * supply if mode == "cooling" else 0.0
        F[0][self._STATE_BETA_FAN] = terms.u_fan_eff * coupling
        F[0][self._STATE_BETA_S] = terms.gain * q_solar
        F[0][self._STATE_BETA_O] = terms.gain * q_occupancy
        F[0][self._STATE_BETA_F] = terms.gain * terms.r_fast
        return F

    def _excitation(
        self,
        terms: _StepTerms,
        mode: str,
        q_solar: float,
        q_occupancy: float,
        jacobian: list[list[float]],
    ) -> list[bool]:
        """Which parameters this step can observe (gain + drift noise)."""
        excited = [True] * self._N
        house_diff = abs(terms.T_h - terms.T)
        excited[self._STATE_ALPHA_HOUSE] = mode == "idle" and house_diff >= _HOUSE_EXCITE_MIN
        excited[self._STATE_BETA_H] = mode == "heating"
        excited[self._STATE_BETA_C] = mode == "cooling"
        excited[self._STATE_BETA_FAN] = mode == MODE_CIRCULATING and house_diff >= _HOUSE_EXCITE_MIN
        # Solar gain is learned from sunny steps without heating only: while
        # heating, the innovation is dominated by how far the emitter's real
        # output differs from beta_h * power_fraction, and beta_s would absorb it.
        excited[self._STATE_BETA_S] = q_solar > 0.0 and mode != "heating"
        excited[self._STATE_BETA_O] = q_occupancy > 0.0
        excited[self._STATE_BETA_F] = abs(terms.r_fast) > _FAST_EXCITE_MIN
        # A parameter the step cannot move gets no gain either, whatever its
        # cross-covariance says.
        for i in range(1, self._N):
            excited[i] = excited[i] and jacobian[0][i] != 0.0
        return excited

    def _predict_step(
        self,
        T_outdoor: float,
        mode: str,
        dt_h: float,
        *,
        power_fraction: float = 1.0,
        q_solar: float = 0.0,
        q_occupancy: float = 0.0,
        q_rebound: float | None = None,
        T_house: float | None = None,
        door_mix_factor: float = 1.0,
        door_supply_factor: float = 1.0,
        u_fan: float = 1.0,
    ) -> None:
        """EKF predict: propagate state and covariance forward."""
        terms = self._step_terms(
            self._x,
            T_outdoor,
            dt_h,
            mode,
            power_fraction=power_fraction,
            q_solar=q_solar,
            q_occupancy=q_occupancy,
            q_rebound=q_rebound,
            T_house=T_house,
            door_mix_factor=door_mix_factor,
            door_supply_factor=door_supply_factor,
            u_fan=u_fan,
        )
        self._x = [terms.T_new] + self._x[1:]

        F = self._jacobian_from_terms(terms, mode, power_fraction, q_solar, q_occupancy)
        self._excited = self._excitation(terms, mode, q_solar, q_occupancy, F)

        # Covariance prediction: P = F @ P @ F^T + Q_noise.  Mode-gated process
        # noise: only parameters that are currently observable receive drift.
        # Unobservable params keep their variance frozen -- inflating it without
        # any measurement to reduce it again just degrades confidence and
        # destabilises the correlated parameters (alpha <-> beta_h coupling ->
        # time-constant oscillation).
        alpha = self._x[self._STATE_ALPHA]
        alpha_ratio = max(alpha, self._ALPHA_MIN) / self._DEFAULT_ALPHA
        q_alpha = self._Q_ALPHA * min(1.0, alpha_ratio * alpha_ratio)
        Q = [
            self._Q_T,
            q_alpha,
            self._Q_ALPHA_HOUSE if self._excited[self._STATE_ALPHA_HOUSE] else 0.0,
            self._Q_BETA if self._excited[self._STATE_BETA_H] else 0.0,
            self._Q_BETA if self._excited[self._STATE_BETA_C] else 0.0,
            self._Q_BETA_FAN if self._excited[self._STATE_BETA_FAN] else 0.0,
            self._Q_BETA_S if self._excited[self._STATE_BETA_S] else 0.0,
            self._Q_BETA_O if self._excited[self._STATE_BETA_O] else 0.0,
            self._Q_BETA_F if self._excited[self._STATE_BETA_F] else 0.0,
        ]

        N = self._N
        P = self._P
        # FP = F @ P
        FP = [[sum(F[i][k] * P[k][j] for k in range(N)) for j in range(N)] for i in range(N)]

        # P_new = FP @ F^T + diag(Q)
        P_new = [[0.0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                P_new[i][j] = sum(FP[i][k] * F[j][k] for k in range(N))
                if i == j:
                    P_new[i][j] += Q[i]

        self._P = P_new

    def _update_step(self, T_measured: float) -> None:
        """EKF update: correct state with measurement.

        Measurement model: H = [1, 0, ...], so y = T_measured.
        """
        N = self._N
        P = self._P

        # Innovation
        y_pred = self._x[0]
        innovation = T_measured - y_pred

        # Innovation covariance: S = P[0][0] + R
        S = P[0][0] + self._R

        # Outlier detection via normalized innovation
        R_eff = self._R
        outlier = False
        if S > 0:
            norm_innov = abs(innovation) / math.sqrt(S)
            if norm_innov > self._ANOMALY_SIGMA:
                outlier = True
                _LOGGER.debug(
                    "EKF: outlier detected (innovation=%.2f, %.1f sigma), softening update",
                    innovation,
                    norm_innov,
                )
                R_eff = self._R * self._ANOMALY_R_INFLATE
                S = P[0][0] + R_eff
        if not outlier:
            self._err_sum += abs(innovation)
            self._err_n += 1

        if S < 1e-12:
            return  # Numerical safety

        # Kalman gain: K = P[:, 0] / S. Parameters the step could not observe
        # get no gain: otherwise their cross-covariance with T and alpha lets
        # every unmodelled disturbance leak into them.
        K = [P[i][0] / S if self._excited[i] else 0.0 for i in range(N)]

        # State update: x = x + K * innovation
        for i in range(N):
            self._x[i] += K[i] * innovation

        # Joseph form (I - K H) P (I - K H)^T + K R K^T, valid for the gated gain,
        # plus the extra K R K^T this filter has always added; with the full
        # gain this reduces to the previous P - K P[0] + K R K^T.
        P_new = [
            [P[i][j] - K[i] * P[0][j] - P[i][0] * K[j] + K[i] * (P[0][0] + 2.0 * R_eff) * K[j] for j in range(N)]
            for i in range(N)
        ]

        self._P = P_new

    def _clamp_parameters(self) -> None:
        """Enforce physical bounds on parameters."""
        bounds = (
            (self._STATE_ALPHA, self._ALPHA_MIN, self._ALPHA_MAX),
            (self._STATE_ALPHA_HOUSE, self._ALPHA_HOUSE_MIN, self._ALPHA_HOUSE_MAX),
            (self._STATE_BETA_H, self._BETA_H_MIN, self._BETA_H_MAX),
            (self._STATE_BETA_C, self._BETA_C_MIN, self._BETA_C_MAX),
            (self._STATE_BETA_FAN, self._BETA_FAN_MIN, self._BETA_FAN_MAX),
            (self._STATE_BETA_S, self._BETA_S_MIN, self._BETA_S_MAX),
            (self._STATE_BETA_O, self._BETA_O_MIN, self._BETA_O_MAX),
            (self._STATE_BETA_F, self._BETA_F_MIN, self._BETA_F_MAX),
        )
        for idx, low, high in bounds:
            self._x[idx] = max(low, min(high, self._x[idx]))

    def _enforce_psd(self) -> None:
        """Enforce symmetry and positive semi-definiteness of P."""
        N = self._N
        P = self._P
        # Enforce symmetry
        for i in range(N):
            for j in range(i + 1, N):
                avg = (P[i][j] + P[j][i]) / 2.0
                P[i][j] = avg
                P[j][i] = avg
        # Enforce positive diagonal
        for i in range(N):
            if P[i][i] < 1e-10:
                P[i][i] = 1e-10

    @property
    def confidence_history(self) -> list[list]:
        """Daily confidence snapshots, oldest first."""
        return [list(entry) for entry in self._confidence_history]

    def _day_error(self) -> tuple[float | None, int]:
        if self._err_n == 0:
            return None, 0
        return round(self._err_sum / self._err_n, 4), self._err_n

    def note_confidence_day(self, day: str, mpc_active: bool) -> None:
        """Record today's confidence and forecast error; a later call on the same day replaces the entry."""
        history = self._confidence_history
        if self._err_day is not None and self._err_day != day:
            # Steps since the last note belong to the day that just ended
            if history and history[-1][0] == self._err_day:
                history[-1][4:6] = self._day_error()
            self._err_sum = 0.0
            self._err_n = 0
        self._err_day = day
        entry = [day, round(self.confidence, 3), self._n_updates, bool(mpc_active), *self._day_error()]
        if history and history[-1][0] == day:
            history[-1] = entry
        elif not history or history[-1][0] < day:
            history.append(entry)
            del history[: max(0, len(history) - CONFIDENCE_HISTORY_DAYS)]

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the full 9-state EKF for persistence."""
        return {
            "ekf_version": self.EKF_VERSION,
            "regressor": dict(self._regressor),
            "x": list(self._x),
            "P": [list(row) for row in self._P],
            "n_updates": self._n_updates,
            "last_update_at": self._last_update_at,
            "n_heating": self._n_heating,
            "n_cooling": self._n_cooling,
            "n_circulating": self._n_circulating,
            "n_idle": self._n_idle,
            "applicable_modes": sorted(self._applicable_modes),
            "last_mode": self._last_mode,
            "initialized": self._initialized,
            "k_window": self._k_window,
            "k_window_n": self._k_window_n,
            "confidence_history": self.confidence_history,
            "forecast_error": {"day": self._err_day, "sum": self._err_sum, "n": self._err_n},
        }

    @property
    def regressor(self) -> dict[str, str]:
        """The inputs these parameters were learned against."""
        return dict(self._regressor)

    def reset_parameter(self, state_idx: int, default: float) -> None:
        """Relearn one parameter from its prior, leaving the rest of the filter alone.

        The parameter's correlations are dropped and its variance returned to the
        initial prior, so the filter re-estimates it on the evidence it would have
        had on a fresh model, while every other learned value, the temperature
        state and the sample counters survive.
        """
        if not 1 <= state_idx < self._N:
            raise ValueError(f"state_idx must be a parameter index 1..{self._N - 1}")
        self._x[state_idx] = default
        p_init = self._p_init_diag()
        for j in range(self._N):
            if j == state_idx:
                continue
            self._P[state_idx][j] = 0.0
            self._P[j][state_idx] = 0.0
        self._P[state_idx][state_idx] = p_init[state_idx]
        self._enforce_psd()

    def apply_regressor_signature(self, signature: Mapping[str, str]) -> tuple[str, ...]:
        """Adopt *signature*, relearning exactly the parameters its inputs changed.

        ``beta_s`` is only meaningful for the solar input it was fitted against, and
        ``alpha_house`` and ``beta_fan`` only for the reference temperature they
        couple the room to, so a changed input resets those parameters and nothing
        else. An unchanged signature is a no-op: that is what keeps a room which
        configures neither orientation nor topology learning undisturbed.

        Returns:
            The names of the parameters that were reset (empty when unchanged).
        """
        current = dict(self._regressor)
        target = dict(signature)
        reset: list[str] = []
        if target.get("solar", LEGACY_SOLAR_REGRESSOR) != current.get("solar", LEGACY_SOLAR_REGRESSOR):
            self.reset_parameter(self._STATE_BETA_S, self._DEFAULT_BETA_S)
            reset.append("beta_s")
        if target.get("ref", "global") != current.get("ref", "global"):
            self.reset_parameter(self._STATE_ALPHA_HOUSE, self._DEFAULT_ALPHA_HOUSE)
            self.reset_parameter(self._STATE_BETA_FAN, self._DEFAULT_BETA_FAN)
            reset.extend(("alpha_house", "beta_fan"))
        self._regressor = target
        if reset:
            _LOGGER.info(
                "ThermalEKF: relearning %s after its model inputs changed (%s -> %s)",
                ", ".join(reset),
                current,
                target,
            )
        return tuple(reset)

    def boost_covariance(self, factor: float = 2.5) -> None:
        """Widen the parameter variances by *factor* to re-learn faster after a change in the room.

        P becomes S P S with S = diag(1, s_1..s_n): every correlation stays as learned and the
        temperature variance is left alone. A variance never grows past its initial value, so
        repeated calls stop there.
        """
        p_init = self._p_init_diag()
        scale = [1.0]
        for i in range(1, self._N):
            var = self._P[i][i]
            target = min(var * factor, p_init[i])
            scale.append(math.sqrt(target / var) if var > 0 and target > var else 1.0)
        self._P = [[self._P[i][j] * scale[i] * scale[j] for j in range(self._N)] for i in range(self._N)]
        self._enforce_psd()

    @staticmethod
    def _float_list(values: object) -> list[float]:
        """Numeric entries of a persisted list (non-numeric entries dropped)."""
        if not isinstance(values, list):
            return []
        return [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]

    @staticmethod
    def _as_float(value: object, default: float) -> float:
        """A finite persisted number, else *default* (older/corrupt stores)."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return default
        number = float(value)
        return number if math.isfinite(number) else default

    @staticmethod
    def _as_int(value: object, default: int) -> int:
        """A persisted integer count, else *default*."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return default
        return int(value)

    @classmethod
    def from_dict(cls, data: dict) -> ThermalEKF:
        """Restore EKF from persisted data, migrating older layouts to 9 states.

        Layouts seen in the wild:
          * 5 states: [T, alpha, beta_h, beta_c, beta_s]
          * 6 states (+ beta_o), and 6 states plus a separate ``beta_f`` block (v6)
          * 7 states: the full pre-forced-air vector
          * 9 states: this version (``ekf_version`` 7)

        Legacy layouts keep every learned value in its semantic slot; the two new
        parameters (alpha_house, beta_fan) start at their defaults with the
        initial prior variance, so the filter starts neutral instead of guessing.
        """
        ekf = cls()
        p_init = cls._p_init_diag()
        old = cls._float_list(data.get("x"))
        new_x = list(ekf._x)
        new_layout = len(old) == cls._N
        if new_layout:
            new_x = list(old)
        elif 5 <= len(old) <= 7:
            # legacy layout [T, alpha, beta_h, beta_c, beta_s, beta_o?, beta_f?]
            new_x[cls._STATE_T] = old[0]
            new_x[cls._STATE_ALPHA] = old[1]
            new_x[cls._STATE_BETA_H] = old[2]
            new_x[cls._STATE_BETA_C] = old[3]
            new_x[cls._STATE_BETA_S] = old[4]
            if len(old) >= 6:
                new_x[cls._STATE_BETA_O] = old[5]
            if len(old) >= 7:
                new_x[cls._STATE_BETA_F] = old[6]

        new_P = [[p_init[i] if i == j else 0.0 for j in range(cls._N)] for i in range(cls._N)]
        P_raw = data.get("P")
        old_P: list[list[float]] = []
        if isinstance(P_raw, list) and 5 <= len(P_raw) <= cls._N:
            rows = [cls._float_list(row) for row in P_raw]
            if all(len(row) == len(rows) for row in rows):
                old_P = rows
        if new_layout and len(old_P) == cls._N:
            new_P = [list(row) for row in old_P]
        elif old_P:
            # [T, alpha_out, beta_h, beta_c, beta_s, beta_o, beta_f] -> 9-state
            index_map = {
                0: cls._STATE_T,
                1: cls._STATE_ALPHA,
                2: cls._STATE_BETA_H,
                3: cls._STATE_BETA_C,
                4: cls._STATE_BETA_S,
                5: cls._STATE_BETA_O,
                6: cls._STATE_BETA_F,
            }
            for i, row in enumerate(old_P):
                for j, value in enumerate(row):
                    if i in index_map and j in index_map:
                        new_P[index_map[i]][index_map[j]] = value

        # v6 kept beta_f and its covariance row out of the 6D x/P block.
        if not new_layout:
            fast = data.get("beta_f")
            if isinstance(fast, dict):
                fast_x = fast.get("x")
                if isinstance(fast_x, (int, float)):
                    new_x[cls._STATE_BETA_F] = float(fast_x)
                fast_values = cls._float_list(fast.get("P"))
                if len(fast_values) >= 7:
                    index_map_fast = {
                        0: cls._STATE_T,
                        1: cls._STATE_ALPHA,
                        2: cls._STATE_BETA_H,
                        3: cls._STATE_BETA_C,
                        4: cls._STATE_BETA_S,
                        5: cls._STATE_BETA_O,
                        6: cls._STATE_BETA_F,
                    }
                    for old_idx in range(7):
                        new_idx = index_map_fast[old_idx]
                        new_P[cls._STATE_BETA_F][new_idx] = fast_values[old_idx]
                        new_P[new_idx][cls._STATE_BETA_F] = fast_values[old_idx]

        ekf._x = new_x
        ekf._P = new_P

        ekf._n_updates = cls._as_int(data.get("n_updates"), 0)
        last_update_at = cls._as_float(data.get("last_update_at"), float("nan"))
        ekf._last_update_at = last_update_at if math.isfinite(last_update_at) else None
        ekf._n_heating = cls._as_int(data.get("n_heating"), 0)
        ekf._n_cooling = cls._as_int(data.get("n_cooling"), 0)
        ekf._n_circulating = cls._as_int(data.get("n_circulating"), 0)
        ekf._n_idle = cls._as_int(data.get("n_idle"), 0)
        modes = data.get("applicable_modes")
        if isinstance(modes, list) and all(isinstance(m, str) for m in modes):
            ekf._applicable_modes = set(modes)
        last_mode = data.get("last_mode")
        ekf._last_mode = last_mode if isinstance(last_mode, str) else None
        ekf._initialized = bool(data.get("initialized", ekf._n_updates > 0))
        k_window = cls._as_float(data.get("k_window"), cls._K_WINDOW_DEFAULT)
        ekf._k_window = max(cls._K_WINDOW_MIN, min(cls._K_WINDOW_MAX, k_window))
        ekf._k_window_n = cls._as_int(data.get("k_window_n"), 0)
        history = data.get("confidence_history")
        if isinstance(history, list):
            ekf._confidence_history = [
                _load_history_entry(e)
                for e in history
                if isinstance(e, list) and len(e) >= 4 and isinstance(e[0], str) and isinstance(e[1], (int, float))
            ][-CONFIDENCE_HISTORY_DAYS:]
        err = data.get("forecast_error")
        if (
            isinstance(err, dict)
            and isinstance(err.get("day"), str)
            and isinstance(err.get("sum"), (int, float))
            and isinstance(err.get("n"), int)
        ):
            ekf._err_day = err["day"]
            ekf._err_sum = float(err["sum"])
            ekf._err_n = err["n"]
        # v8 records which inputs the parameters were fitted against. Older
        # saves carry none, and every one of them was trained on the legacy
        # inputs against the whole-house mean.
        raw_regressor = data.get("regressor")
        regressor = {"solar": LEGACY_SOLAR_REGRESSOR, "ref": "global"}
        if isinstance(raw_regressor, dict):
            saved_solar = raw_regressor.get("solar")
            saved_ref = raw_regressor.get("ref")
            if isinstance(saved_solar, str):
                regressor["solar"] = saved_solar
            if saved_ref in ("global", "local"):
                regressor["ref"] = saved_ref
        ekf._regressor = regressor
        # Recovery migrations for models with alpha pegged at the upper bound.
        # Such models cannot escape on their own: 4-sigma anomaly detection
        # soft-rejects the large innovations and cross-covariance pulls alpha
        # back to the bound after every predict step.  Reset all RC parameters
        # and their P block to defaults; keep counters, T-state, P[0][0],
        # modes and k_window so MPC data gates stay satisfied while the
        # accuracy gate self-throttles until the filter relearns.
        #
        # v<5 -> v5: first recovery for pre-#150 ghost-heating.
        # v<6 -> v6 (#301): recovery for models that drifted back to the bound.
        # v<=6 -> v7: legacy single-node layouts are migrated above; a pegged
        #   alpha cannot be a healthy forced-air model, so it is reset once.
        # The bound is a fixed version rather than ``EKF_VERSION``: v8 only adds
        # the regressor signature, so it must not trigger a wholesale reset.
        version_raw = data.get("ekf_version", 0)
        version = version_raw if isinstance(version_raw, int) else 0
        if version <= _ALPHA_RECOVERY_MAX_EKF_VERSION and ekf._x[cls._STATE_ALPHA] >= cls._ALPHA_MAX * 0.99:
            cls._reset_rc_params_for_recovery(ekf, version)
        # Apply bounds in case stored data is out of range
        ekf._clamp_parameters()
        ekf._enforce_psd()
        return ekf

    @classmethod
    def _reset_rc_params_for_recovery(cls, ekf: ThermalEKF, ekf_version: int) -> None:
        """Reset RC parameters and their P block to defaults.

        Used by load-time recovery migrations when alpha is pegged at the
        upper bound.  Preserves T-state, P[0][0], counters, modes and
        k_window.
        """
        legacy_alpha = ekf._x[cls._STATE_ALPHA]
        defaults = (
            cls._DEFAULT_ALPHA,
            cls._DEFAULT_ALPHA_HOUSE,
            cls._DEFAULT_BETA_H,
            cls._DEFAULT_BETA_C,
            cls._DEFAULT_BETA_FAN,
            cls._DEFAULT_BETA_S,
            cls._DEFAULT_BETA_O,
            cls._DEFAULT_BETA_F,
        )
        for offset, value in enumerate(defaults):
            ekf._x[offset + 1] = value
        p_init_diag = cls._p_init_diag()
        for i in range(cls._N):
            for j in range(cls._N):
                if i != j:
                    ekf._P[i][j] = 0.0
            if i >= 1:
                ekf._P[i][i] = p_init_diag[i]
        _LOGGER.warning(
            "ThermalEKF: recovered from alpha pegged at upper bound "
            "(legacy ekf_version=%d, alpha=%.3f -> reset to defaults). "
            "Predictions will improve over the next few hours as the filter relearns.",
            ekf_version,
            legacy_alpha,
        )

    def __repr__(self) -> str:
        return (
            f"ThermalEKF(T={self._x[self._STATE_T]:.1f}, alpha={self._x[self._STATE_ALPHA]:.3f}, "
            f"alpha_house={self._x[self._STATE_ALPHA_HOUSE]:.2f}, beta_h={self._x[self._STATE_BETA_H]:.2f}, "
            f"beta_c={self._x[self._STATE_BETA_C]:.2f}, beta_fan={self._x[self._STATE_BETA_FAN]:.2f}, "
            f"beta_s={self._x[self._STATE_BETA_S]:.2f}, beta_o={self._x[self._STATE_BETA_O]:.2f}, "
            f"beta_f={self._x[self._STATE_BETA_F]:.2f}, "
            f"confidence={self.confidence:.2f}, n={self._n_updates})"
        )


# ---------------------------------------------------------------------------
# RoomModelManager -- per-room model management
# ---------------------------------------------------------------------------


class RoomModelManager:
    """Manages per-room ThermalEKF instances.

    Provides a simple facade so the coordinator can update observations and
    request predictions without worrying about per-room bookkeeping.
    """

    def __init__(self) -> None:
        self._estimators: dict[str, ThermalEKF] = {}

    def get_room_ids(self) -> list[str]:
        """Return list of room IDs with learned models."""
        return list(self._estimators.keys())

    def get_estimator(self, area_id: str) -> ThermalEKF:
        """Return the estimator for *area_id*, creating one if needed."""
        if area_id not in self._estimators:
            self._estimators[area_id] = ThermalEKF()
        return self._estimators[area_id]

    def update(
        self,
        area_id: str,
        T_new: float,
        T_outdoor: float,
        mode: str,
        dt_minutes: float,
        *,
        can_heat: bool = True,
        can_cool: bool = True,
        power_fraction: float = 1.0,
        q_solar: float = 0.0,
        q_residual: float = 0.0,
        q_occupancy: float = 0.0,
        q_rebound: float | None = None,
        T_house: float | None = None,
        door_mix_factor: float = 1.0,
        door_supply_factor: float = 1.0,
        u_fan: float = 1.0,
    ) -> None:
        """Feed an observed transition to the room's estimator."""
        est = self.get_estimator(area_id)
        est.set_applicable_modes(can_heat, can_cool)
        est.update(
            T_new,
            T_outdoor,
            mode,
            dt_minutes,
            power_fraction=power_fraction,
            q_solar=q_solar,
            q_residual=q_residual,
            q_occupancy=q_occupancy,
            q_rebound=q_rebound,
            T_house=T_house,
            door_mix_factor=door_mix_factor,
            door_supply_factor=door_supply_factor,
            u_fan=u_fan,
        )

    def predict(
        self,
        area_id: str,
        T_room: float,
        T_outdoor: float,
        Q_active: float,
        dt_minutes: float,
        *,
        T_house: float | None = None,
        mode: str | None = None,
        u_fan: float = 0.0,
        door_mix_factor: float = 1.0,
        door_supply_factor: float = 1.0,
    ) -> float:
        """Predict future temperature for *area_id*."""
        return (
            self.get_estimator(area_id)
            .get_model()
            .predict(
                T_room,
                T_outdoor,
                Q_active,
                dt_minutes,
                T_house=T_house,
                mode=mode,
                u_fan=u_fan,
                door_mix_factor=door_mix_factor,
                door_supply_factor=door_supply_factor,
            )
        )

    def get_model(self, area_id: str) -> RCModel:
        """Return the learned RCModel for *area_id* (C=1 normalization)."""
        return self.get_estimator(area_id).get_model()

    def get_confidence(self, area_id: str) -> float:
        """Return model confidence for *area_id* (0.0 if never seen)."""
        if area_id not in self._estimators:
            return 0.0
        return self._estimators[area_id].confidence

    def get_prediction_std(
        self,
        area_id: str,
        Q_active: float,
        T_room: float,
        T_outdoor: float,
        dt_minutes: float,
        *,
        q_solar: float = 0.0,
        q_residual: float = 0.0,
        q_occupancy: float = 0.0,
        q_rebound: float | None = None,
        mode: str | None = None,
        T_house: float | None = None,
        u_fan: float = 1.0,
        door_mix_factor: float = 1.0,
        door_supply_factor: float = 1.0,
    ) -> float:
        """Return prediction uncertainty in degC for *area_id* at given conditions."""
        if area_id not in self._estimators:
            return float("inf")
        return self._estimators[area_id].prediction_std(
            Q_active,
            T_room,
            T_outdoor,
            dt_minutes,
            q_solar=q_solar,
            q_residual=q_residual,
            q_occupancy=q_occupancy,
            q_rebound=q_rebound,
            mode=mode,
            T_house=T_house,
            u_fan=u_fan,
            door_mix_factor=door_mix_factor,
            door_supply_factor=door_supply_factor,
        )

    def get_mode_counts(self, area_id: str) -> tuple[int, int, int]:
        """Return (n_idle, n_heating, n_cooling) for *area_id*."""
        if area_id not in self._estimators:
            return (0, 0, 0)
        est = self._estimators[area_id]
        return (est._n_idle, est._n_heating, est._n_cooling)

    def get_circulating_count(self, area_id: str) -> int:
        """Return the number of circulating (fan-only) training samples for *area_id*."""
        est = self._estimators.get(area_id)
        return est._n_circulating if est else 0

    def get_n_observations(self, area_id: str) -> int:
        """Return total number of EKF updates for *area_id*."""
        est = self._estimators.get(area_id)
        return est._n_updates if est else 0

    def is_param_learned(self, area_id: str, state_idx: int) -> bool:
        """Whether *area_id*'s EKF state *state_idx* has been learned from data."""
        est = self._estimators.get(area_id)
        if est is None:
            return False
        return est.is_param_learned(state_idx)

    def sync_temperature(self, area_id: str, T_measured: float) -> None:
        """Set *area_id*'s temperature state to a measurement, keeping its parameters."""
        self.get_estimator(area_id).sync_temperature(T_measured)

    def update_window_open(
        self,
        area_id: str,
        T_new: float,
        T_outdoor: float,
        dt_minutes: float,
        *,
        learn_k_window: bool = True,
    ) -> None:
        """Track temperature state and optionally learn k_window during window-open."""
        est = self.get_estimator(area_id)
        est.update_window_open(T_new, T_outdoor, dt_minutes, learn_k_window=learn_k_window)

    def predict_window_open(
        self,
        area_id: str,
        T_room: float,
        T_outdoor: float,
        dt_minutes: float,
    ) -> float:
        """Predict temperature during window-open using learned k_window."""
        est = self.get_estimator(area_id)
        return est.get_model().predict_window_open(T_room, T_outdoor, est.k_window, dt_minutes)

    def get_k_window(self, area_id: str) -> float:
        """Return the learned window multiplier for *area_id*."""
        if area_id not in self._estimators:
            return ThermalEKF._K_WINDOW_DEFAULT
        return self._estimators[area_id].k_window

    def get_residual_scale(self, area_id: str) -> float:
        """Legacy read-only afterglow scale (1.0 = profile) for *area_id*."""
        return 1.0

    def update_residual_scale(self, area_id: str, sample: float) -> None:
        """Deprecated no-op: kept so pre-Wave-3 callers do not crash on the API."""

    def get_residual_tau(self, area_id: str) -> float | None:
        """Legacy read-only afterglow decay time; the forced-air model no longer learns it."""
        return None

    def update_residual_tau(self, area_id: str, sample: float, profile_tau: float) -> None:
        """Deprecated no-op: kept so pre-Wave-3 callers do not crash on the API."""

    def boost_learning(self, area_id: str) -> int | None:
        """Boost covariance for *area_id* to accelerate re-learning.

        Returns the current ``n_updates`` (the anchor the frontend counts the fade from),
        None when the room has no model yet.
        """
        est = self._estimators.get(area_id)
        if est is None:
            return None
        est.boost_covariance()
        return est._n_updates

    def remove_room(self, area_id: str) -> None:
        """Discard learned data for *area_id*."""
        self._estimators.pop(area_id, None)

    def to_dict(self) -> dict:
        """Serialize all room estimators."""
        return {area_id: est.to_dict() for area_id, est in self._estimators.items()}

    @classmethod
    def from_dict(cls, data: dict) -> RoomModelManager:
        """Restore all room estimators from persisted data."""
        mgr = cls()
        for area_id, est_data in data.items():
            mgr._estimators[area_id] = ThermalEKF.from_dict(est_data)
        return mgr

    def __repr__(self) -> str:
        rooms = ", ".join(f"{aid}({e.confidence:.0%})" for aid, e in self._estimators.items())
        return f"RoomModelManager([{rooms}])"
