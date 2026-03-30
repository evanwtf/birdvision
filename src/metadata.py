"""
Location and time-based prior probabilities over species, backed by eBird
bar chart data stored in a local SQLite database.

The bar chart has 48 periods per year (4 per month). Given a date, we map
to the nearest period and look up the observed frequency for each species in
the matched region. If the media GPS does not fall inside one of the supported
bounding boxes, eBird weighting is skipped and the visual probabilities are
left alone.

Species with zero eBird frequency are floored at `zero_floor` so the model
can still surface genuinely rare birds that the observer might be lucky enough
to see.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 48 periods/year: period = (month-1)*4 + (day-1)//(days_in_month//4)
# Simpler approximation: period = day_of_year / (365/48)
_PERIODS_PER_YEAR = 48

# Rough bounding boxes are sufficient here. They intentionally trade geometric
# precision for a simple "inside supported area or not" check.
_REGION_BOUNDS = {
    "long_island": {
        "name": "Long Island",
        "fips": ["US-NY-047", "US-NY-081", "US-NY-059", "US-NY-103"],
        "lat_min": 40.54,
        "lat_max": 41.12,
        "lon_min": -74.05,
        "lon_max": -71.80,
    },
}


def _date_to_period(dt: datetime) -> int:
    day_of_year = dt.timetuple().tm_yday  # 1-365
    period = int((day_of_year - 1) / (365.0 / _PERIODS_PER_YEAR))
    return min(period, _PERIODS_PER_YEAR - 1)


class MetadataPrior:
    def __init__(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        db_path: Optional[str] = None,
        fips: Optional[str] = None,
        zero_floor: float = 0.01,
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.db_path = db_path
        self.default_fips = fips
        self.zero_floor = zero_floor
        self._con: Optional[sqlite3.Connection] = None
        self._county_fips: set[str] = set()
        self._county_names: Dict[str, str] = {}

        if db_path and Path(db_path).exists():
            self._con = sqlite3.connect(db_path, check_same_thread=False)
            counties = self._con.execute("SELECT fips, name FROM counties").fetchall()
            self._county_fips = {row[0] for row in counties}
            self._county_names = {row[0]: row[1] for row in counties}
            county_names = [row[1] for row in counties]
            logger.info(
                f"eBird priors loaded from {db_path} "
                f"({len(counties)} counties: {', '.join(county_names)})"
            )
        else:
            if db_path:
                logger.warning(f"eBird DB not found at {db_path} — using uniform priors")
            else:
                logger.info("No eBird DB configured — using uniform priors")

    def resolve_region(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Optional[dict]:
        lat = self.latitude if latitude is None else latitude
        lon = self.longitude if longitude is None else longitude

        if self.default_fips and self.default_fips in self._county_fips:
            return {
                "name": self._county_names.get(self.default_fips, self.default_fips),
                "fips": [self.default_fips],
            }
        if lat is None or lon is None:
            return None

        for region in _REGION_BOUNDS.values():
            region_fips = [fips for fips in region["fips"] if fips in self._county_fips]
            if not region_fips:
                continue
            if (
                region["lat_min"] <= lat <= region["lat_max"]
                and region["lon_min"] <= lon <= region["lon_max"]
            ):
                return {
                    "name": region["name"],
                    "fips": region_fips,
                }
        return None

    def resolve_county_name(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Optional[str]:
        region = self.resolve_region(latitude=latitude, longitude=longitude)
        if not region:
            return None
        return region["name"]

    def get_priors(
        self,
        species_names: List[str],
        dt: Optional[datetime] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Dict[str, float]:
        if self._con is None or dt is None:
            return {s: 1.0 for s in species_names}

        region = self.resolve_region(latitude=latitude, longitude=longitude)
        if region is None:
            return {s: 1.0 for s in species_names}

        period = _date_to_period(dt)
        placeholders = ",".join("?" * len(species_names))
        fips_placeholders = ",".join("?" * len(region["fips"]))
        rows = self._con.execute(
            f"""
            SELECT species, AVG(freq) FROM barchart
            WHERE fips IN ({fips_placeholders}) AND period = ?
              AND species IN ({placeholders})
            GROUP BY species
            """,
            [*region["fips"], period, *species_names],
        ).fetchall()

        freq_map = {row[0]: row[1] for row in rows}
        return {
            s: max(freq_map.get(s, 0.0), self.zero_floor)
            for s in species_names
        }

    def apply(
        self,
        predictions: List[Tuple[str, float]],
        dt: Optional[datetime] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """Re-weight and re-normalize predictions using eBird location/time priors."""
        species = [s for s, _ in predictions]
        priors = self.get_priors(species, dt, latitude=latitude, longitude=longitude)
        adjusted = [(s, p * priors.get(s, 1.0)) for s, p in predictions]
        total = sum(p for _, p in adjusted)
        if total > 0:
            adjusted = [(s, p / total) for s, p in adjusted]
        return sorted(adjusted, key=lambda x: -x[1])
