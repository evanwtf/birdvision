"""
Location and time-based prior probabilities over species.

Currently a stub returning uniform weights. Designed for drop-in replacement
with eBird range map data: given lat/lon + date, eBird can return the probability
of each species being present, which we multiply against the classifier output.

eBird API docs: https://documenter.getpostman.com/view/664302/S1ENwy59
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SEASON_MONTHS = {
    "spring": {3, 4, 5},
    "summer": {6, 7, 8},
    "fall":   {9, 10, 11},
    "winter": {12, 1, 2},
}


def get_season(dt: datetime) -> str:
    for season, months in SEASON_MONTHS.items():
        if dt.month in months:
            return season
    return "unknown"


class MetadataPrior:
    def __init__(self, latitude: Optional[float] = None, longitude: Optional[float] = None):
        self.latitude = latitude
        self.longitude = longitude
        if latitude and longitude:
            logger.info(f"Location prior: {latitude:.4f}, {longitude:.4f}")
        else:
            logger.info("No location set — using uniform species priors.")

    def get_priors(self, species_names: List[str], dt: Optional[datetime] = None) -> Dict[str, float]:
        """
        Returns species_name -> probability multiplier (1.0 = neutral).
        TODO: replace with eBird API call using self.latitude/longitude + dt.
        """
        return {s: 1.0 for s in species_names}

    def apply(
        self,
        predictions: List[Tuple[str, float]],
        dt: Optional[datetime] = None,
    ) -> List[Tuple[str, float]]:
        """Re-weight and re-normalize predictions using location/time priors."""
        species = [s for s, _ in predictions]
        priors = self.get_priors(species, dt)
        adjusted = [(s, p * priors.get(s, 1.0)) for s, p in predictions]
        total = sum(p for _, p in adjusted)
        if total > 0:
            adjusted = [(s, p / total) for s, p in adjusted]
        return sorted(adjusted, key=lambda x: -x[1])
