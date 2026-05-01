import numpy as np
from geographiclib.geodesic import Geodesic

_GEOD = Geodesic.WGS84  # type: ignore[attr-defined]
_METERS_PER_KM = 1000.0


def get_range_and_bearing(ref_lat: float, ref_lon: float, lat: float, lon: float) -> tuple[float, float]:
    """Compute geodesic range (km) and bearing (rad) from a reference to a target point.

    Args:
        ref_lat: Reference latitude in decimal degrees.
        ref_lon: Reference longitude in decimal degrees.
        lat: Target latitude in decimal degrees.
        lon: Target longitude in decimal degrees.

    Returns:
        Tuple of (range_km, bearing_rad) where bearing_rad is the azimuth clockwise from North.
    """
    result = _GEOD.Inverse(ref_lat, ref_lon, lat, lon)
    range_km: float = result["s12"] / _METERS_PER_KM
    bearing_rad: float = np.deg2rad(result["azi1"])
    return range_km, bearing_rad
