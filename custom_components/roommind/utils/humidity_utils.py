"""Psychrometric and comfort humidity utilities (pure math, no HA dependencies)."""

from __future__ import annotations

import math


def saturation_vapor_pressure_hpa(temp_c: float) -> float:
    """Return the saturation vapor pressure in hPa at a temperature in Celsius."""
    return 6.105 * math.exp((17.27 * temp_c) / (237.7 + temp_c))


def vapor_pressure_hpa(temp_c: float, rh_pct: float) -> float:
    """Return the actual vapor pressure in hPa at a relative humidity in percent."""
    rh = max(0.0, min(100.0, rh_pct))
    return (rh / 100.0) * saturation_vapor_pressure_hpa(temp_c)


def effective_temperature_c(
    temp_c: float,
    rh_pct: float | None,
    *,
    enabled: bool = True,
    ref_rh_pct: float = 45.0,
) -> float:
    """Return the temperature adjusted for the actual moisture load."""
    if not enabled or rh_pct is None or math.isnan(rh_pct):
        return temp_c
    e_actual = vapor_pressure_hpa(temp_c, rh_pct)
    e_ref = vapor_pressure_hpa(temp_c, ref_rh_pct)
    ramp = max(0.0, min(1.0, (temp_c - 16.0) / 4.0))
    delta = max(-1.2, min(2.5, 0.18 * (e_actual - e_ref) * ramp))
    return round(temp_c + delta, 3)


def predict_rh_at_temp(
    temp_now_c: float,
    rh_now_pct: float,
    temp_future_c: float,
    *,
    cooling_active: bool = False,
    dt_minutes: float = 5.0,
) -> float:
    """Predict the relative humidity after a cooling run."""
    e_now = vapor_pressure_hpa(temp_now_c, rh_now_pct)
    if cooling_active and dt_minutes > 0:
        e_now *= math.exp(-0.004 * dt_minutes)
    rh_future = 100.0 * e_now / max(saturation_vapor_pressure_hpa(temp_future_c), 0.01)
    return min(100.0, max(5.0, rh_future))


def dehumidify_cool_offset_c(
    weighted_rh: float | None,
    threshold_rh: float = 60.0,
    max_overcool_c: float = 0.8,
) -> float:
    """Return the overcooling offset needed to reach the RH threshold."""
    if weighted_rh is None or max_overcool_c <= 0.0 or weighted_rh <= threshold_rh:
        return 0.0
    frac = min(1.0, (weighted_rh - threshold_rh) / 15.0)
    return round(frac * max_overcool_c, 3)
