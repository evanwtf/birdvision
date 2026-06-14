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
import math
import sqlite3
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


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
        latitude: float | None = None,
        longitude: float | None = None,
        db_path: str | None = None,
        fips: str | None = None,
        zero_floor: float = 0.01,
        prior_mode: str = "seasonal",  # "seasonal" | "location_only"
        local_priors_file: str | None = None,
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.db_path = db_path
        self.default_fips = fips
        self.zero_floor = zero_floor
        self.prior_mode = prior_mode
        self._con: sqlite3.Connection | None = None
        self._county_fips: set[str] = set()
        self._county_names: dict[str, str] = {}
        self._local_locations: list[dict] = []

        if local_priors_file and Path(local_priors_file).exists():
            with open(local_priors_file) as f:
                local_cfg = yaml.safe_load(f)
            self._local_locations = local_cfg.get("locations", [])
            for loc in self._local_locations:
                n_sp = len(loc.get("species", {}))
                logger.info(
                    "Local priors loaded: %r (%.4f, %.4f) radius=%.1fkm  %d species overridden",
                    loc.get("name", "unnamed"),
                    loc.get("lat"),
                    loc.get("lon"),
                    loc.get("radius_km", 1.0),
                    n_sp,
                )
        elif local_priors_file:
            logger.warning("local_priors_file not found: %s", local_priors_file)

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
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> dict | None:
        has_explicit_location = latitude is not None or longitude is not None
        lat = self.latitude if latitude is None else latitude
        lon = self.longitude if longitude is None else longitude

        if (
            not has_explicit_location
            and self.default_fips
            and self.default_fips in self._county_fips
        ):
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
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> str | None:
        region = self.resolve_region(latitude=latitude, longitude=longitude)
        if not region:
            return None
        return region["name"]

    def _query_frequencies_annual(
        self,
        fips_list: list[str],
        species_names: list[str],
    ) -> dict[str, float]:
        """Return average frequency across all 48 periods (year-round)."""
        placeholders = ",".join("?" * len(species_names))
        fips_placeholders = ",".join("?" * len(fips_list))
        rows = self._con.execute(
            f"""
            SELECT species, AVG(freq) FROM barchart
            WHERE fips IN ({fips_placeholders})
              AND species IN ({placeholders})
            GROUP BY species
            """,
            [*fips_list, *species_names],
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def _resolve_local_overrides(
        self,
        latitude: float | None,
        longitude: float | None,
    ) -> dict[str, float]:
        """Return per-species override dict for the first matching location, or {} if none match."""
        if latitude is None or longitude is None or not self._local_locations:
            return {}
        for loc in self._local_locations:
            dist = _haversine_km(latitude, longitude, loc["lat"], loc["lon"])
            if dist <= loc.get("radius_km", 1.0):
                logger.debug("Local override active: %r (%.2fkm away)", loc.get("name"), dist)
                return dict(loc.get("species", {}))
        return {}

    def get_priors(
        self,
        species_names: list[str],
        dt: datetime | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> dict[str, float]:
        if self._con is None:
            return {s: 1.0 for s in species_names}
        if self.prior_mode == "seasonal" and dt is None:
            return {s: 1.0 for s in species_names}

        lat = latitude if latitude is not None else self.latitude
        lon = longitude if longitude is not None else self.longitude

        region = self.resolve_region(latitude=latitude, longitude=longitude)
        if region is None:
            return {s: 1.0 for s in species_names}

        fips_placeholders = ",".join("?" * len(region["fips"]))

        if self.prior_mode == "location_only":
            freq_map = self._query_frequencies_annual(region["fips"], species_names)
        else:
            period = _date_to_period(dt)
            placeholders = ",".join("?" * len(species_names))
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

        # Local overrides bypass zero_floor — the explicit value is used as-is.
        # eBird-sourced species that are not overridden still get zero_floor applied.
        local = self._resolve_local_overrides(lat, lon)
        return {
            s: (local[s] if s in local else max(freq_map.get(s, 0.0), self.zero_floor))
            for s in species_names
        }

    def apply(
        self,
        predictions: list[tuple[str, float]],
        dt: datetime | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> list[tuple[str, float]]:
        """Re-weight and re-normalize predictions using eBird location/time priors."""
        species = [s for s, _ in predictions]
        priors = self.get_priors(species, dt, latitude=latitude, longitude=longitude)
        adjusted = [(s, p * priors.get(s, 1.0)) for s, p in predictions]
        total = sum(p for _, p in adjusted)
        if total > 0:
            adjusted = [(s, p / total) for s, p in adjusted]
        return sorted(adjusted, key=lambda x: -x[1])
