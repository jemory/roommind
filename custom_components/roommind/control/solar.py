"""Solar irradiance estimation for thermal model solar gain.

Computes Global Horizontal Irradiance (GHI) from:
  1. Solar position (NOAA simplified equations) using latitude/longitude
  2. Clear-sky GHI via Meinel atmospheric transmittance model
  3. Cloud attenuation via Kasten-Czeplak (1980) model

Also transposes GHI onto a vertical (or tilted) plane using an isotropic-sky
diffuse model plus ground reflection (Erbs, Klein & Duffie 1982 diffuse split).
Computation uses only ``math``, ``time`` and ``datetime``; the three tuning
constants are shared from ``const`` rather than redefined here.
"""

from __future__ import annotations

import math
import time
from datetime import UTC

from ..const import SOLAR_GROUND_ALBEDO, SOLAR_MIN_BEAM_SIN_ELEVATION, SOLAR_WINDOW_TILT_DEG

# Solar constant (W/m²) — TSI at 1 AU
_SOLAR_CONSTANT: float = 1361.0


def _solar_position(latitude: float, longitude: float, timestamp: float) -> tuple[float, float]:
    """Return (elevation_deg, azimuth_deg) for a given location and time.

    Azimuth uses meteorological convention: 0=N, 90=E, 180=S, 270=W.
    Uses the NOAA simplified solar position equations.
    Negative elevation means the sun is below the horizon (night).
    """
    from datetime import datetime

    dt = datetime.fromtimestamp(timestamp, tz=UTC)
    day_of_year = dt.timetuple().tm_yday
    hour_utc = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

    # Fractional year (radians)
    gamma = 2.0 * math.pi * (day_of_year - 1 + (hour_utc - 12.0) / 24.0) / 365.0

    # Equation of time (minutes)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )

    # Solar declination (radians)
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.00148 * math.sin(3.0 * gamma)
    )

    # True solar time and hour angle
    time_offset = eqtime + 4.0 * longitude  # minutes (UTC, no timezone offset)
    tst = hour_utc * 60.0 + time_offset
    ha = math.radians((tst / 4.0) - 180.0)

    # Solar elevation
    lat_rad = math.radians(latitude)
    sin_elev = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(ha)
    sin_elev = max(-1.0, min(1.0, sin_elev))
    elevation_deg = math.degrees(math.asin(sin_elev))

    # Solar azimuth (meteorological: 0=N, 90=E, 180=S, 270=W)
    # NOAA convention: morning (ha<0) → 180+acos(…); afternoon (ha≥0) → (540-acos(…)) mod 360
    cos_elev = math.cos(math.radians(elevation_deg))
    if abs(cos_elev) < 1e-10:
        azimuth_deg = 180.0  # Sun directly overhead — arbitrary
    else:
        cos_az = (math.sin(lat_rad) * sin_elev - math.sin(decl)) / (cos_elev * math.cos(lat_rad))
        cos_az = max(-1.0, min(1.0, cos_az))
        arc = math.degrees(math.acos(cos_az))
        # ha < 0 = morning (sun in the east): azimuth ∈ [0°,180°]
        # ha ≥ 0 = afternoon/noon (sun in the south/west): azimuth ∈ [180°,360°]
        azimuth_deg = (540.0 - arc) % 360.0 if ha < 0 else 180.0 + arc

    return elevation_deg, azimuth_deg


def _solar_elevation(latitude: float, longitude: float, timestamp: float) -> float:
    """Return solar elevation angle in degrees (negative = below horizon)."""
    return _solar_position(latitude, longitude, timestamp)[0]


def solar_elevation(latitude: float, longitude: float, timestamp: float) -> float:
    """Public API: solar elevation angle in degrees (negative = below horizon)."""
    return _solar_position(latitude, longitude, timestamp)[0]


def solar_azimuth(latitude: float, longitude: float, timestamp: float) -> float:
    """Solar azimuth in degrees (0=N, 90=E, 180=S, 270=W)."""
    return _solar_position(latitude, longitude, timestamp)[1]


def surface_irradiance_factor(solar_az_deg: float, solar_el_deg: float, surface_az_deg: float) -> float:
    """Fraction of GHI that hits a vertical surface facing *surface_az_deg*.

    Returns 0–1. Night (elevation ≤ 0) or sun behind the surface → 0.
    """
    if solar_el_deg <= 0:
        return 0.0
    return max(
        0.0,
        math.cos(math.radians(solar_el_deg)) * math.cos(math.radians(solar_az_deg - surface_az_deg)),
    )


def clear_sky_index(cloud_coverage: float | None) -> float:
    """Return the Kasten-Czeplak (1980) clear-sky index from cloud cover.

    The index is the ratio of actual global irradiance to its clear-sky value:

    ``kt = 1 - 0.75 * n**3.4``

    where ``n`` is fractional cloud cover clamped to 0..1. ``None`` means no
    cloud information, which is treated as a clear sky (index 1.0).

    Args:
        cloud_coverage: 0-100 percent, or ``None`` for clear sky.

    Returns:
        Multiplicative clearness index in (0, 1].
    """
    if cloud_coverage is None:
        return 1.0
    n = max(0.0, min(1.0, cloud_coverage / 100.0))
    return float(1.0 - 0.75 * (n**3.4))


def _cloud_attenuation(cloud_coverage: float) -> float:
    """Kasten-Czeplak (1980) cloud attenuation factor.

    Backwards-compatible wrapper: delegates to :func:`clear_sky_index`.

    Args:
        cloud_coverage: 0-100 (percent).

    Returns:
        Multiplicative factor in (0, 1].
    """
    return clear_sky_index(cloud_coverage)


def _clear_sky_dni(elevation_deg: float) -> float:
    """Estimate clear-sky direct-normal irradiance (DNI) in W/m².

    Uses the Meinel atmospheric transmittance model with Kasten & Young
    (1989) air mass: ``dni = _SOLAR_CONSTANT * 0.7 ** (am ** 0.678)``.
    Returns 0 when the sun is at or below the horizon.
    """
    if elevation_deg <= 0.0:
        return 0.0

    sin_elev = math.sin(math.radians(elevation_deg))

    # Air mass (Kasten & Young, 1989)
    am = 1.0 / (sin_elev + 0.50572 * (elevation_deg + 6.07995) ** (-1.6364))

    # Direct normal irradiance — Meinel model
    return float(_SOLAR_CONSTANT * 0.7 ** (am**0.678))


def _clear_sky_ghi(elevation_deg: float) -> float:
    """Estimate clear-sky GHI in W/m² using the Meinel model.

    ``GHI = DNI × sin(elevation)``, preserving the exact current behaviour.
    Returns 0 when the sun is at or below the horizon.
    """
    if elevation_deg <= 0.0:
        return 0.0

    sin_elev = math.sin(math.radians(elevation_deg))
    dni = _clear_sky_dni(elevation_deg)

    # GHI = DNI × sin(elevation)
    return float(max(0.0, dni * sin_elev))


def diffuse_fraction(clearness_index: float) -> float:
    """Return the diffuse fraction of global irradiance per Erbs et al. (1982).

    Erbs, Klein & Duffie, "Estimation of the diffuse radiation fraction for
    hourly, daily and monthly-average global radiation", Solar Energy 28(4),
    1982. The piecewise fit maps the clearness index ``kt`` to the ratio
    DHI/GHI:

      * ``kt <= 0.22``: ``1 - 0.09 * kt``
      * ``0.22 < kt <= 0.80``: ``0.9511 - 0.1604*kt + 4.388*kt²
        - 16.638*kt³ + 12.336*kt⁴``
      * ``kt > 0.80``: ``0.165``

    The result is clamped to [0.0, 1.0].
    """
    if clearness_index <= 0.22:
        fraction = 1.0 - 0.09 * clearness_index
    elif clearness_index <= 0.80:
        fraction = (
            0.9511
            - 0.1604 * clearness_index
            + 4.388 * clearness_index**2
            - 16.638 * clearness_index**3
            + 12.336 * clearness_index**4
        )
    else:
        fraction = 0.165
    return max(0.0, min(1.0, float(fraction)))


def plane_of_array_irradiance(
    latitude: float,
    longitude: float,
    timestamp: float,
    surface_azimuth_deg: float,
    *,
    cloud_coverage: float | None = None,
    tilt_deg: float = SOLAR_WINDOW_TILT_DEG,
    albedo: float = SOLAR_GROUND_ALBEDO,
) -> float:
    """Return total plane-of-array irradiance (W/m²) on a tilted surface.

    Transposes modelled GHI onto a plane facing *surface_azimuth_deg* using an
    isotropic diffuse sky and ground reflection:

      ``POA = DNI*cos(incidence) + DHI*(1+cos(tilt))/2
              + GHI*albedo*(1-cos(tilt))/2``

    Diffuse and beam are split from GHI with the Erbs (1982) clearness-index
    correlation. At grazing sun the beam split divides by ``sin(elevation)``,
    which is singular as elevation approaches 0 and would amplify any rounding
    error into a spurious huge beam spike; below
    :data:`SOLAR_MIN_BEAM_SIN_ELEVATION` we therefore treat all irradiance as
    diffuse and drop the beam term entirely. Returns 0 at night.
    """
    elev, az = _solar_position(latitude, longitude, timestamp)
    if elev <= 0.0:
        return 0.0

    kt = clear_sky_index(cloud_coverage)
    ghi = _clear_sky_ghi(elev) * kt

    df = diffuse_fraction(kt)
    dhi = df * ghi
    sin_elev = math.sin(math.radians(elev))
    if sin_elev < SOLAR_MIN_BEAM_SIN_ELEVATION:
        # At grazing sun the beam split divides by a vanishing sine: the
        # division is singular and would produce a huge/erroneous beam term.
        # Treat the remaining irradiance as all diffuse instead.
        dni = 0.0
        dhi = ghi
    else:
        dni = (ghi - dhi) / sin_elev

    cos_incidence = max(
        0.0,
        math.cos(math.radians(elev)) * math.cos(math.radians(az - surface_azimuth_deg)),
    )
    cos_tilt = math.cos(math.radians(tilt_deg))

    beam = dni * cos_incidence
    sky = dhi * (1.0 + cos_tilt) / 2.0
    ground = ghi * albedo * (1.0 - cos_tilt) / 2.0
    return max(0.0, beam + sky + ground)


# -- Public API --


def estimate_solar_ghi(
    latitude: float,
    longitude: float,
    timestamp: float,
    cloud_coverage: float | None = None,
) -> float:
    """Estimate Global Horizontal Irradiance in W/m².

    Args:
        latitude: degrees (from ``hass.config.latitude``).
        longitude: degrees (from ``hass.config.longitude``).
        timestamp: Unix timestamp (UTC).
        cloud_coverage: 0-100 percent (from weather entity).
            ``None`` → clear-sky (no attenuation).

    Returns:
        GHI in W/m² (0 at night, peaks ~900 W/m² at solar noon on clear day).
    """
    elev = _solar_elevation(latitude, longitude, timestamp)
    ghi = _clear_sky_ghi(elev)
    if cloud_coverage is not None and ghi > 0.0:
        ghi *= _cloud_attenuation(cloud_coverage)
    return ghi


def compute_q_solar_norm(
    latitude: float,
    longitude: float,
    timestamp: float,
    cloud_coverage: float | None = None,
) -> float:
    """Normalized solar irradiance for the thermal model.

    Returns ``GHI / 1000`` so that the value is in [0, ~1.0], making the
    learned parameter ``beta_s`` comparable in magnitude to ``beta_h`` and
    ``beta_c``.
    """
    return estimate_solar_ghi(latitude, longitude, timestamp, cloud_coverage) / 1000.0


def build_solar_series(
    latitude: float,
    longitude: float,
    n_blocks: int,
    dt_minutes: float = 5.0,
    *,
    start_ts: float | None = None,
    cloud_series: list[float | None] | None = None,
) -> list[float]:
    """Build a series of Q_solar_norm values for MPC/analytics lookahead.

    Args:
        latitude / longitude: site coordinates.
        n_blocks: number of future time blocks.
        dt_minutes: duration of each block.
        start_ts: start Unix timestamp (default: now).
        cloud_series: per-block cloud coverage (0-100 or None).
            If shorter than *n_blocks*, the last value is repeated.
            If ``None``, clear-sky is used for all blocks.
    """
    ts = start_ts if start_ts is not None else time.time()
    dt_sec = dt_minutes * 60.0

    # Prepare cloud values
    clouds: list[float | None]
    if cloud_series:
        clouds = list(cloud_series)
        while len(clouds) < n_blocks:
            clouds.append(clouds[-1])
    else:
        clouds = [None] * n_blocks

    series: list[float] = []
    for i in range(n_blocks):
        block_ts = ts + i * dt_sec
        cc = clouds[i] if i < len(clouds) else None
        series.append(compute_q_solar_norm(latitude, longitude, block_ts, cc))
    return series


def build_oriented_solar_series(
    latitude: float,
    longitude: float,
    n_blocks: int,
    surface_azimuths: list[float],
    dt_minutes: float = 5.0,
    *,
    start_ts: float | None = None,
    cloud_series: list[float | None] | None = None,
) -> list[float]:
    """Return normalised plane-of-array irradiance averaged over orientations.

    For each time step this computes the true plane-of-array irradiance with
    :func:`plane_of_array_irradiance` for every azimuth in *surface_azimuths*,
    takes the mean of those values, then divides by 1000 (same normalisation as
    :func:`compute_q_solar_norm`). It is no longer GHI times an averaged beam
    projection factor: each facade receives its own diffuse sky and ground
    reflection, which is why a north-facing room gets a small positive value
    instead of exactly zero. Averaging the orientations still handles corner
    rooms with windows on several walls — each wall contributes equally.

    Args:
        surface_azimuths: list of surface azimuths in degrees (0=N, 90=E …).
            Must not be empty.
    """
    if not surface_azimuths:
        raise ValueError("surface_azimuths must not be empty")

    ts = start_ts if start_ts is not None else time.time()
    dt_sec = dt_minutes * 60.0

    # Pad the cloud series to n_blocks by repeating its last value (same
    # behaviour as build_solar_series); None means clear sky for all blocks.
    clouds: list[float | None]
    if cloud_series:
        clouds = list(cloud_series)
        while len(clouds) < n_blocks:
            clouds.append(clouds[-1])
    else:
        clouds = [None] * n_blocks

    result: list[float] = []
    for i in range(n_blocks):
        step_ts = ts + i * dt_sec
        cc = clouds[i] if i < len(clouds) else None
        total_poa = sum(
            plane_of_array_irradiance(latitude, longitude, step_ts, s, cloud_coverage=cc) for s in surface_azimuths
        )
        result.append(total_poa / len(surface_azimuths) / 1000.0)
    return result
