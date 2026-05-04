import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class TimeMetadata(BaseModel):
    """Metadata for time information.

    Attributes:
        timezone: Timezone string.
        unix_epoch: UNIX epoch time.
        utc_iso: UTC time in ISO format.
        local_iso: Local time in ISO format, if available.
    """

    timezone: str
    unix_epoch: int
    utc_iso: str
    local_iso: str | None


class BaseStats(BaseModel):
    """Statistical summary for a data field.

    Attributes:
        min: Minimum value.
        max: Maximum value.
        mean: Mean value.
        std: Standard deviation.
    """

    min: float
    max: float
    mean: float
    std: float


class ReferenceSystem(BaseModel):
    """Geographic bounding box.

    Attributes:
        north: Northern boundary.
        south: Southern boundary.
        east: Eastern boundary.
        west: Western boundary.
    """

    north: float
    south: float
    east: float
    west: float


class DataFieldsStats(BaseModel):
    """Statistical summaries for all raw data fields.

    Attributes:
        Altitude: Altitude statistics.
        Speed: Speed statistics.
        Heading: Heading statistics.
        Lat: Latitude statistics.
        Lon: Longitude statistics.
        Range: Range statistics.
        Bearing: Bearing statistics.
        x: x-coordinate statistics.
        y: y-coordinate statistics.
    """

    Altitude: BaseStats
    Speed: BaseStats
    Heading: BaseStats
    Lat: BaseStats
    Lon: BaseStats
    Range: BaseStats
    Bearing: BaseStats
    x: BaseStats
    y: BaseStats


class ReferenceMetadata(BaseModel):
    """Reference metadata for an airport.

    Attributes:
        airport_name: Airport name.
        airport_id: Airport identifier.
        ref_lat: Reference latitude.
        ref_lon: Reference longitude.
        range_scale: Scale for range measurements.
        ll_offset: Latitude/longitude offset.
        espg_4326: Reference system bounds in EPSG:4326.
        espg_3857: Reference system bounds in EPSG:3857.
        limits: Data field statistics limits.
    """

    airport_name: str
    airport_id: str
    ref_lat: float
    ref_lon: float
    range_scale: float
    ll_offset: float
    espg_4326: ReferenceSystem
    espg_3857: ReferenceSystem
    limits: DataFieldsStats

    _reference_system: ReferenceSystem | None = None

    def set_reference(self, espg: str) -> None:
        """Set the active reference system.

        Args:
            espg: EPSG code, either ``"ESPG:4326"`` or ``"ESPG:3857"``.
        """
        if espg == "ESPG:4326":
            self._reference_system = self.espg_4326
        elif espg == "ESPG:3857":
            self._reference_system = self.espg_3857
        else:
            error_message = f"Unsupported EPSG code: {espg}"
            raise ValueError(error_message)

    @property
    def reference_system(self) -> ReferenceSystem | None:
        """Active reference system, or None if not set."""
        return self._reference_system

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceMetadata":
        """Create an instance from a dictionary.

        Args:
            data: Metadata dictionary.

        Returns:
            Constructed instance.
        """
        return cls(**data)

    @classmethod
    def from_json_file(cls, filepath: Path) -> "ReferenceMetadata":
        """Create an instance from a JSON file.

        Args:
            filepath: Path to the JSON file.

        Returns:
            Constructed instance.
        """
        with open(filepath) as f:
            data = json.load(f)
        return cls(**data)
