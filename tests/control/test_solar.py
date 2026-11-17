"""Tests for the solar irradiance estimation module."""

from __future__ import annotations

from datetime import UTC

import pytest

from custom_components.roommind.const import SOLAR_GROUND_ALBEDO
from custom_components.roommind.control.solar import (
    _clear_sky_ghi,
    _cloud_attenuation,
    _solar_elevation,
    _solar_position,
    build_oriented_solar_series,
    build_solar_series,
    clear_sky_index,
    compute_q_solar_norm,
    diffuse_fraction,
    estimate_solar_ghi,
    plane_of_array_irradiance,
    solar_azimuth,
    surface_irradiance_factor,
)

# ---------------------------------------------------------------------------
# Solar elevation
# ---------------------------------------------------------------------------


def test_solar_elevation_night():
    """Sun is below horizon at midnight in Berlin (winter)."""
    # 2024-01-15 00:00 UTC, Berlin (52.52°N, 13.41°E)
    from datetime import datetime

    dt = datetime(2024, 1, 15, 0, 0, tzinfo=UTC)
    elev = _solar_elevation(52.52, 13.41, dt.timestamp())
    assert elev < 0, "Sun should be below horizon at midnight"


def test_solar_elevation_noon_summer():
    """Sun is high at solar noon in summer."""
    from datetime import datetime

    # 2024-06-21 ~12:00 UTC, Equator (0°N, 0°E)
    dt = datetime(2024, 6, 21, 12, 0, tzinfo=UTC)
    elev = _solar_elevation(0.0, 0.0, dt.timestamp())
    assert elev > 50, f"Sun should be high at equator noon in summer, got {elev}"


def test_solar_elevation_noon_winter_northern():
    """Sun is lower at noon in winter at northern latitude."""
    from datetime import datetime

    dt = datetime(2024, 12, 21, 12, 0, tzinfo=UTC)
    elev = _solar_elevation(50.0, 10.0, dt.timestamp())
    assert 10 < elev < 30, f"Expected moderate elevation in winter noon, got {elev}"


# ---------------------------------------------------------------------------
# Clear-sky GHI
# ---------------------------------------------------------------------------


def test_clear_sky_ghi_night():
    """GHI is 0 when sun is below horizon."""
    assert _clear_sky_ghi(-5.0) == 0.0


def test_clear_sky_ghi_noon():
    """GHI is high with sun near zenith."""
    ghi = _clear_sky_ghi(70.0)
    assert 800 < ghi < 1100, f"Expected ~900 W/m² at 70° elevation, got {ghi}"


def test_clear_sky_ghi_low_elevation():
    """GHI is lower at low elevation angles."""
    ghi_low = _clear_sky_ghi(10.0)
    ghi_high = _clear_sky_ghi(60.0)
    assert ghi_low < ghi_high, "GHI should increase with elevation"


# ---------------------------------------------------------------------------
# Cloud attenuation
# ---------------------------------------------------------------------------


def test_cloud_attenuation_clear():
    """No clouds = no attenuation."""
    assert _cloud_attenuation(0.0) == pytest.approx(1.0)


def test_cloud_attenuation_overcast():
    """Full cloud cover strongly attenuates."""
    factor = _cloud_attenuation(100.0)
    assert 0.1 < factor < 0.4, f"Expected strong attenuation at 100% clouds, got {factor}"


def test_cloud_attenuation_partial():
    """Partial cloud cover gives intermediate attenuation."""
    factor = _cloud_attenuation(50.0)
    assert 0.5 < factor < 1.0, f"Expected moderate attenuation at 50% clouds, got {factor}"


def test_cloud_attenuation_clamp():
    """Values outside 0-100 are clamped."""
    assert _cloud_attenuation(-10.0) == pytest.approx(1.0)
    assert _cloud_attenuation(150.0) == _cloud_attenuation(100.0)


# ---------------------------------------------------------------------------
# estimate_solar_ghi
# ---------------------------------------------------------------------------


def test_ghi_night_returns_zero():
    """GHI is 0 at night."""
    from datetime import datetime

    dt = datetime(2024, 1, 15, 2, 0, tzinfo=UTC)
    ghi = estimate_solar_ghi(50.0, 10.0, dt.timestamp())
    assert ghi == 0.0


def test_ghi_day_positive():
    """GHI is positive during daytime."""
    from datetime import datetime

    dt = datetime(2024, 6, 21, 12, 0, tzinfo=UTC)
    ghi = estimate_solar_ghi(50.0, 10.0, dt.timestamp())
    assert ghi > 500, f"Expected high GHI at noon in summer, got {ghi}"


def test_ghi_clouds_reduce():
    """Cloud coverage reduces GHI."""
    from datetime import datetime

    dt = datetime(2024, 6, 21, 12, 0, tzinfo=UTC)
    ts = dt.timestamp()
    ghi_clear = estimate_solar_ghi(50.0, 10.0, ts)
    ghi_cloudy = estimate_solar_ghi(50.0, 10.0, ts, cloud_coverage=80.0)
    assert ghi_cloudy < ghi_clear, "Clouds should reduce GHI"
    assert ghi_cloudy > 0, "Even cloudy should have some GHI"


def test_ghi_clouds_none_means_clear():
    """cloud_coverage=None gives same result as 0."""
    from datetime import datetime

    dt = datetime(2024, 6, 21, 12, 0, tzinfo=UTC)
    ts = dt.timestamp()
    ghi_none = estimate_solar_ghi(50.0, 10.0, ts, cloud_coverage=None)
    ghi_clear = estimate_solar_ghi(50.0, 10.0, ts, cloud_coverage=0.0)
    # cloud_coverage=0 still passes through cloud attenuation (factor=1.0)
    assert ghi_none == pytest.approx(ghi_clear, rel=0.01)


# ---------------------------------------------------------------------------
# compute_q_solar_norm
# ---------------------------------------------------------------------------


def test_q_solar_norm_range():
    """Normalized solar is GHI/1000, in [0, ~1.0]."""
    from datetime import datetime

    dt = datetime(2024, 6, 21, 12, 0, tzinfo=UTC)
    qs = compute_q_solar_norm(50.0, 10.0, dt.timestamp())
    assert 0.0 < qs < 1.2, f"Expected q_solar_norm in [0, ~1.0], got {qs}"


def test_q_solar_norm_night():
    """Normalized solar is 0 at night."""
    from datetime import datetime

    dt = datetime(2024, 1, 15, 2, 0, tzinfo=UTC)
    qs = compute_q_solar_norm(50.0, 10.0, dt.timestamp())
    assert qs == 0.0


# ---------------------------------------------------------------------------
# build_solar_series
# ---------------------------------------------------------------------------


def test_build_solar_series_length():
    """Series has the requested number of blocks."""
    series = build_solar_series(50.0, 10.0, 12, 5.0)
    assert len(series) == 12


def test_build_solar_series_night():
    """All values are 0 at night."""
    from datetime import datetime

    dt = datetime(2024, 1, 15, 2, 0, tzinfo=UTC)
    series = build_solar_series(50.0, 10.0, 12, 5.0, start_ts=dt.timestamp())
    assert all(v == 0.0 for v in series)


def test_build_solar_series_with_clouds():
    """Cloud series reduces values compared to clear sky."""
    from datetime import datetime

    dt = datetime(2024, 6, 21, 12, 0, tzinfo=UTC)
    ts = dt.timestamp()
    clear = build_solar_series(50.0, 10.0, 6, 5.0, start_ts=ts)
    cloudy = build_solar_series(50.0, 10.0, 6, 5.0, start_ts=ts, cloud_series=[80.0])
    for c, cl in zip(clear, cloudy, strict=False):
        if c > 0:
            assert cl < c, "Cloudy values should be lower"


def test_build_solar_series_cloud_padding():
    """Short cloud_series is padded by repeating last value."""
    from datetime import datetime

    dt = datetime(2024, 6, 21, 12, 0, tzinfo=UTC)
    ts = dt.timestamp()
    # cloud_series with 1 entry should be padded to 6
    series = build_solar_series(50.0, 10.0, 6, 5.0, start_ts=ts, cloud_series=[50.0])
    assert len(series) == 6


# ---------------------------------------------------------------------------
# _solar_position / solar_azimuth
# ---------------------------------------------------------------------------


def test_solar_position_returns_elevation_and_azimuth():
    """_solar_position returns a 2-tuple (elevation, azimuth)."""
    from datetime import datetime

    dt = datetime(2024, 6, 21, 12, 0, tzinfo=UTC)
    el, az = _solar_position(50.0, 10.0, dt.timestamp())
    assert -90 <= el <= 90
    assert 0 <= az < 360


def test_solar_azimuth_noon_approx_south():
    """At solar noon in Northern Hemisphere, azimuth is near 180° (south)."""
    from datetime import datetime

    # True solar noon at longitude 0 is close to 12:00 UTC
    dt = datetime(2024, 6, 21, 12, 0, tzinfo=UTC)
    az = solar_azimuth(50.0, 0.0, dt.timestamp())
    # Should be between 150° and 210° (roughly south)
    assert 150 < az < 210, f"Expected south azimuth near noon, got {az}"


def test_solar_azimuth_morning_eastern():
    """In the morning (before noon), sun is in the eastern half."""
    from datetime import datetime

    dt = datetime(2024, 6, 21, 6, 0, tzinfo=UTC)
    az = solar_azimuth(50.0, 0.0, dt.timestamp())
    assert az < 180, f"Morning sun should be in eastern half, got {az}"


# ---------------------------------------------------------------------------
# surface_irradiance_factor
# ---------------------------------------------------------------------------


def test_surface_irradiance_factor_night():
    """Night (elevation ≤ 0) → factor = 0."""
    assert surface_irradiance_factor(180.0, -5.0, 180.0) == 0.0
    assert surface_irradiance_factor(180.0, 0.0, 180.0) == 0.0


def test_surface_irradiance_factor_facing_sun():
    """Sun due south, surface due south → positive factor."""
    factor = surface_irradiance_factor(180.0, 30.0, 180.0)
    assert 0 < factor <= 1.0


def test_surface_irradiance_factor_facing_away():
    """Sun due south, surface due north → factor = 0."""
    factor = surface_irradiance_factor(180.0, 60.0, 0.0)
    assert factor == 0.0, f"North surface, south sun should give 0, got {factor}"


def test_surface_irradiance_factor_perpendicular():
    """Sun due south, surface due east → factor = 0 (90° off)."""
    factor = surface_irradiance_factor(180.0, 45.0, 90.0)
    assert factor == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# build_oriented_solar_series
# ---------------------------------------------------------------------------


def test_build_oriented_solar_series_length():
    """Oriented series has the same length as requested."""
    from datetime import datetime

    dt = datetime(2024, 6, 21, 12, 0, tzinfo=UTC)
    series = build_oriented_solar_series(50.0, 10.0, 12, [180.0], 5.0, start_ts=dt.timestamp())
    assert len(series) == 12


def test_build_oriented_solar_series_north_facing_near_zero():
    """North facade gets diffuse sky + ground reflection, not exactly zero.

    The old beam-only projection returned exactly 0.0 for a surface facing
    away from the sun. A vertical north plane still sees half the isotropic
    diffuse sky plus ground-reflected irradiance, so the value is now small
    but strictly positive and well below the south-facing value.
    """
    from datetime import datetime

    dt = datetime(2024, 6, 21, 12, 0, tzinfo=UTC)
    ts = dt.timestamp()
    north = build_oriented_solar_series(50.0, 0.0, 6, [0.0], 5.0, start_ts=ts)
    south = build_oriented_solar_series(50.0, 0.0, 6, [180.0], 5.0, start_ts=ts)
    # North receives diffuse + ground: positive, but far below the sunlit south facade.
    assert all(v > 0.0 for v in north), f"North facade should get diffuse+ground > 0, got {north}"
    assert all(n < s for n, s in zip(north, south, strict=False)), "North must stay well below south"


def test_build_oriented_solar_series_south_facing_positive():
    """South-facing vertical plane in winter exceeds horizontal GHI.

    With a low winter sun a vertical south facade sees the beam much closer
    to normal incidence than the horizontal plane does, so its POA irradiance
    is larger than GHI — the opposite of the old beam-only model's assumption
    that an oriented value could never exceed GHI.
    """
    from datetime import datetime

    # 41.9 deg N, winter noon: sun low in the south (~25 deg elevation).
    dt = datetime(2024, 12, 21, 12, 0, tzinfo=UTC)
    ts = dt.timestamp()
    oriented = build_oriented_solar_series(41.9, 0.0, 1, [180.0], 5.0, start_ts=ts)
    base = build_solar_series(41.9, 0.0, 1, 5.0, start_ts=ts)
    # Winter south vertical plane receives more than horizontal GHI.
    assert oriented[0] > 0
    assert oriented[0] > base[0], "Winter south POA should exceed horizontal GHI"


def test_build_oriented_solar_series_corner_room_average():
    """Corner room (south + east windows) → average of the two POA values.

    The series is the mean of per-orientation plane-of-array irradiance, so
    the two-window average is still exactly the midpoint of the two single
    orientations.
    """
    from datetime import datetime

    dt = datetime(2024, 6, 21, 12, 0, tzinfo=UTC)  # Noon, sun roughly south
    ts = dt.timestamp()
    south = build_oriented_solar_series(50.0, 0.0, 1, [180.0], 5.0, start_ts=ts)
    east = build_oriented_solar_series(50.0, 0.0, 1, [90.0], 5.0, start_ts=ts)
    corner = build_oriented_solar_series(50.0, 0.0, 1, [180.0, 90.0], 5.0, start_ts=ts)
    expected_avg = (south[0] + east[0]) / 2
    assert corner[0] == pytest.approx(expected_avg, rel=1e-6)


# ---------------------------------------------------------------------------
# clear_sky_index (Kasten-Czeplak)
# ---------------------------------------------------------------------------


def test_clear_sky_index_clear():
    """0% cloud → index 1.0 (no attenuation)."""
    assert clear_sky_index(0.0) == pytest.approx(1.0)


def test_clear_sky_index_partial():
    """50% cloud → 1 - 0.75 * 0.5**3.4 (hand-computed Kasten-Czeplak)."""
    expected = 1.0 - 0.75 * (0.5**3.4)
    assert clear_sky_index(50.0) == pytest.approx(expected)


def test_clear_sky_index_overcast():
    """100% cloud → index 0.25 (1 - 0.75)."""
    assert clear_sky_index(100.0) == pytest.approx(0.25)


def test_clear_sky_index_none_is_clear():
    """Missing cloud data means clear sky (index 1.0)."""
    assert clear_sky_index(None) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# diffuse_fraction (Erbs, Klein & Duffie 1982)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kt", "expected"),
    [
        # kt <= 0.22 branch: 1 - 0.09*kt
        (0.15, 1.0 - 0.09 * 0.15),  # 0.9865
        (0.22, 1.0 - 0.09 * 0.22),  # 0.9802
        # 0.22 < kt <= 0.80 branch: quartic fit
        (
            0.5,
            0.9511 - 0.1604 * 0.5 + 4.388 * 0.5**2 - 16.638 * 0.5**3 + 12.336 * 0.5**4,
        ),  # 0.65915
        (
            0.8,
            0.9511 - 0.1604 * 0.8 + 4.388 * 0.8**2 - 16.638 * 0.8**3 + 12.336 * 0.8**4,
        ),  # 0.1652696
        # kt > 0.80 branch: constant
        (0.9, 0.165),
    ],
)
def test_diffuse_fraction_erbs(kt: float, expected: float):
    """Erbs (1982) piecewise diffuse fraction matches hand-computed values."""
    assert diffuse_fraction(kt) == pytest.approx(expected)


def test_diffuse_fraction_clamped():
    """Result is clamped to [0, 1] even for out-of-range clearness indices."""
    assert 0.0 <= diffuse_fraction(-1.0) <= 1.0
    assert 0.0 <= diffuse_fraction(2.0) <= 1.0


# ---------------------------------------------------------------------------
# plane_of_array_irradiance
# ---------------------------------------------------------------------------


def test_plane_of_array_irradiance_night():
    """POA is 0 at night (sun below horizon)."""
    from datetime import datetime

    dt = datetime(2024, 1, 15, 2, 0, tzinfo=UTC)
    assert plane_of_array_irradiance(50.0, 10.0, dt.timestamp(), 180.0) == 0.0


def test_plane_of_array_irradiance_north_noon_positive():
    """North-facing vertical plane at noon gets small positive diffuse + ground.

    The sun is in the south, so the beam term is zero for a north facade;
    what remains is half the diffuse sky plus ground reflection (~150 W/m²
    at 50 deg N on a clear June noon).
    """
    from datetime import datetime

    ts = datetime(2024, 6, 21, 12, 0, tzinfo=UTC).timestamp()
    north = plane_of_array_irradiance(50.0, 0.0, ts, 0.0)
    south = plane_of_array_irradiance(50.0, 0.0, ts, 180.0)
    assert 0.0 < north < south
    # North sees only diffuse sky + ground on a vertical plane (cos_tilt ~ 0):
    # POA = 0.5 * df * ghi + 0.5 * albedo * ghi.
    ghi = estimate_solar_ghi(50.0, 0.0, ts)
    expected = 0.5 * diffuse_fraction(clear_sky_index(None)) * ghi + 0.5 * SOLAR_GROUND_ALBEDO * ghi
    assert north == pytest.approx(expected, rel=1e-6)


def test_plane_of_array_overcast_azimuth_converges():
    """Under uniform overcast, south and north facades converge.

    At 100% cloud cover kt = 0.25 and the Erbs diffuse fraction is ~0.973,
    so irradiance is almost entirely diffuse; the beam contribution that
    distinguishes orientations becomes negligible and both facades approach
    the same diffuse+ground value.
    """
    from datetime import datetime

    ts = datetime(2024, 6, 21, 12, 0, tzinfo=UTC).timestamp()
    south = plane_of_array_irradiance(50.0, 0.0, ts, 180.0, cloud_coverage=100.0)
    north = plane_of_array_irradiance(50.0, 0.0, ts, 0.0, cloud_coverage=100.0)
    # Both are close to the pure diffuse+ground value for a vertical plane
    # (cos_tilt ~ 0): POA = 0.5 * df * ghi + 0.5 * albedo * ghi.
    ghi = estimate_solar_ghi(50.0, 0.0, ts, cloud_coverage=100.0)
    diffuse_poa = 0.5 * diffuse_fraction(clear_sky_index(100.0)) * ghi + 0.5 * SOLAR_GROUND_ALBEDO * ghi
    assert south == pytest.approx(diffuse_poa, rel=0.05)
    assert north == pytest.approx(diffuse_poa, rel=0.05)
    # Azimuth barely matters under overcast: facades within a few percent.
    assert abs(south - north) / south < 0.05


def test_plane_of_array_grazing_sun_finite():
    """Grazing sun just above the horizon gives a finite, small value.

    Below SOLAR_MIN_BEAM_SIN_ELEVATION the beam split divides by a vanishing
    sine; the model drops the beam term entirely rather than producing a huge
    or NaN spike.
    """
    import math
    from datetime import datetime

    # Equator, a few minutes after sunrise: elevation ~0.8 deg (below clamp).
    ts = datetime(2024, 6, 21, 6, 5, tzinfo=UTC).timestamp()
    el, _ = _solar_position(0.0, 0.0, ts)
    assert 0.0 < el < 2.0
    value = plane_of_array_irradiance(0.0, 0.0, ts, 180.0)
    assert math.isfinite(value)
    assert 0.0 <= value < 10.0, f"Grazing-sun POA should be small and finite, got {value}"


def test_plane_of_array_south_winter_exceeds_ghi():
    """Winter south vertical plane exceeds horizontal GHI at 41.9 deg N.

    A low winter sun strikes a south-facing wall almost head-on while the
    horizontal plane receives it at a shallow angle, so POA > GHI (~598 vs
    ~299 W/m² in this model — roughly double).
    """
    from datetime import datetime

    ts = datetime(2024, 12, 21, 12, 0, tzinfo=UTC).timestamp()
    poa = plane_of_array_irradiance(41.9, 0.0, ts, 180.0)
    ghi = estimate_solar_ghi(41.9, 0.0, ts)
    assert poa > ghi > 0.0
