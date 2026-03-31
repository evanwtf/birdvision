"""Unit tests for src/metadata.py — date-to-period mapping, region resolution,
eBird priors, and apply reweighting."""

from datetime import datetime

import pytest

from src.metadata import MetadataPrior, _date_to_period


# ---------------------------------------------------------------------------
# _date_to_period
# ---------------------------------------------------------------------------

class TestDateToPeriod:
    def test_january_1(self):
        assert _date_to_period(datetime(2024, 1, 1)) == 0

    def test_december_31(self):
        assert _date_to_period(datetime(2024, 12, 31)) == 47

    def test_mid_year(self):
        # Day 183 (July 2 in non-leap year) -> period = int(182 / (365/48)) ~ 23
        period = _date_to_period(datetime(2023, 7, 2))
        assert 22 <= period <= 24

    def test_period_never_exceeds_47(self):
        # Leap year Dec 31 is day 366
        assert _date_to_period(datetime(2024, 12, 31)) <= 47

    def test_early_february(self):
        # Day ~35 -> period ~ 4
        period = _date_to_period(datetime(2024, 2, 4))
        assert 3 <= period <= 5


# ---------------------------------------------------------------------------
# MetadataPrior — no database
# ---------------------------------------------------------------------------

class TestMetadataPriorNoDb:
    def test_no_db_returns_uniform(self):
        mp = MetadataPrior()
        priors = mp.get_priors(["Robin", "Sparrow"], dt=datetime(2024, 1, 1))
        assert priors == {"Robin": 1.0, "Sparrow": 1.0}

    def test_no_datetime_returns_uniform(self):
        mp = MetadataPrior()
        priors = mp.get_priors(["Robin"], dt=None)
        assert priors == {"Robin": 1.0}

    def test_resolve_region_no_db(self):
        mp = MetadataPrior()
        assert mp.resolve_region(latitude=40.7, longitude=-73.5) is None

    def test_resolve_county_name_no_db(self):
        mp = MetadataPrior()
        assert mp.resolve_county_name(latitude=40.7, longitude=-73.5) is None


# ---------------------------------------------------------------------------
# MetadataPrior — with database fixture
# ---------------------------------------------------------------------------

class TestMetadataPriorWithDb:
    def test_resolve_region_inside_long_island(self, ebird_db):
        mp = MetadataPrior(db_path=ebird_db)
        region = mp.resolve_region(latitude=40.7, longitude=-73.5)
        assert region is not None
        assert region["name"] == "Long Island"
        assert "US-NY-059" in region["fips"]
        assert "US-NY-103" in region["fips"]

    def test_resolve_region_outside_long_island(self, ebird_db):
        mp = MetadataPrior(db_path=ebird_db)
        # Chicago
        region = mp.resolve_region(latitude=41.9, longitude=-87.6)
        assert region is None

    def test_resolve_region_no_coords(self, ebird_db):
        mp = MetadataPrior(db_path=ebird_db)
        assert mp.resolve_region() is None

    def test_resolve_region_default_fips(self, ebird_db):
        mp = MetadataPrior(db_path=ebird_db, fips="US-NY-059")
        region = mp.resolve_region()
        assert region is not None
        assert region["name"] == "Nassau"
        assert region["fips"] == ["US-NY-059"]

    def test_resolve_county_name(self, ebird_db):
        mp = MetadataPrior(db_path=ebird_db)
        name = mp.resolve_county_name(latitude=40.7, longitude=-73.5)
        assert name == "Long Island"

    def test_get_priors_with_region(self, ebird_db):
        mp = MetadataPrior(db_path=ebird_db)
        priors = mp.get_priors(
            ["Mourning Dove", "Blue Jay", "Bald Eagle"],
            dt=datetime(2024, 1, 1),  # period 0
            latitude=40.7,
            longitude=-73.5,
        )
        # Mourning Dove: avg(0.45, 0.50) = 0.475
        assert priors["Mourning Dove"] == pytest.approx(0.475)
        # Blue Jay: avg(0.30, 0.25) = 0.275
        assert priors["Blue Jay"] == pytest.approx(0.275)
        # Bald Eagle: avg(0.02, 0.01) = 0.015
        assert priors["Bald Eagle"] == pytest.approx(0.015)

    def test_zero_floor_for_missing_species(self, ebird_db):
        mp = MetadataPrior(db_path=ebird_db, zero_floor=0.01)
        priors = mp.get_priors(
            ["Nonexistent Bird"],
            dt=datetime(2024, 1, 1),
            latitude=40.7,
            longitude=-73.5,
        )
        assert priors["Nonexistent Bird"] == pytest.approx(0.01)

    def test_custom_zero_floor(self, ebird_db):
        mp = MetadataPrior(db_path=ebird_db, zero_floor=0.05)
        priors = mp.get_priors(
            ["Nonexistent Bird"],
            dt=datetime(2024, 1, 1),
            latitude=40.7,
            longitude=-73.5,
        )
        assert priors["Nonexistent Bird"] == pytest.approx(0.05)

    def test_no_region_returns_uniform(self, ebird_db):
        mp = MetadataPrior(db_path=ebird_db)
        # Outside Long Island
        priors = mp.get_priors(
            ["Robin"],
            dt=datetime(2024, 1, 1),
            latitude=35.0,
            longitude=-80.0,
        )
        assert priors == {"Robin": 1.0}


# ---------------------------------------------------------------------------
# MetadataPrior.apply
# ---------------------------------------------------------------------------

class TestMetadataPriorApply:
    def test_apply_reweights_and_normalizes(self, ebird_db):
        mp = MetadataPrior(db_path=ebird_db)
        predictions = [("Mourning Dove", 0.5), ("Blue Jay", 0.5)]
        result = mp.apply(
            predictions,
            dt=datetime(2024, 1, 1),
            latitude=40.7,
            longitude=-73.5,
        )
        result_dict = dict(result)
        total = sum(result_dict.values())
        assert total == pytest.approx(1.0)
        # Mourning Dove has higher frequency, should be boosted
        assert result_dict["Mourning Dove"] > result_dict["Blue Jay"]

    def test_apply_sorted_descending(self, ebird_db):
        mp = MetadataPrior(db_path=ebird_db)
        predictions = [("Blue Jay", 0.5), ("Mourning Dove", 0.5)]
        result = mp.apply(
            predictions,
            dt=datetime(2024, 1, 1),
            latitude=40.7,
            longitude=-73.5,
        )
        assert result[0][1] >= result[1][1]

    def test_apply_no_db_passthrough(self):
        mp = MetadataPrior()
        predictions = [("Robin", 0.7), ("Sparrow", 0.3)]
        result = mp.apply(predictions, dt=datetime(2024, 1, 1))
        result_dict = dict(result)
        assert result_dict["Robin"] == pytest.approx(0.7)
        assert result_dict["Sparrow"] == pytest.approx(0.3)

    def test_apply_no_datetime_passthrough(self, ebird_db):
        mp = MetadataPrior(db_path=ebird_db)
        predictions = [("Robin", 0.6), ("Sparrow", 0.4)]
        result = mp.apply(predictions, dt=None)
        result_dict = dict(result)
        assert result_dict["Robin"] == pytest.approx(0.6)
