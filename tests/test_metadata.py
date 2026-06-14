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

    def test_explicit_gps_overrides_default_fips(self, ebird_db):
        mp = MetadataPrior(db_path=ebird_db, fips="US-NY-059")
        region = mp.resolve_region(latitude=40.7, longitude=-73.5)
        assert region is not None
        assert region["name"] == "Long Island"
        assert region["fips"] == ["US-NY-059", "US-NY-103"]

    def test_explicit_gps_outside_supported_region_ignores_default_fips(self, ebird_db):
        mp = MetadataPrior(db_path=ebird_db, fips="US-NY-059")
        assert mp.resolve_region(latitude=41.9, longitude=-87.6) is None

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


# ---------------------------------------------------------------------------
# MetadataPrior — location_only mode
# ---------------------------------------------------------------------------


class TestMetadataPriorLocationOnly:
    def test_prior_mode_stored(self, ebird_db):
        mp = MetadataPrior(db_path=ebird_db, prior_mode="location_only")
        assert mp.prior_mode == "location_only"

    def test_default_prior_mode_is_seasonal(self, ebird_db):
        mp = MetadataPrior(db_path=ebird_db)
        assert mp.prior_mode == "seasonal"

    def test_location_only_applies_without_date(self, ebird_db):
        """location_only mode should apply priors even when dt=None."""
        mp = MetadataPrior(db_path=ebird_db, prior_mode="location_only")
        priors = mp.get_priors(
            ["Mourning Dove", "Nonexistent Bird"],
            dt=None,
            latitude=40.7,
            longitude=-73.5,
        )
        # Mourning Dove: AVG of (0.45, 0.50, 0.60, 0.55) = 0.525
        assert priors["Mourning Dove"] == pytest.approx(0.525)
        assert priors["Nonexistent Bird"] == pytest.approx(0.01)

    def test_location_only_applies_with_date(self, ebird_db):
        """location_only mode ignores date for period selection, uses year-round avg."""
        mp = MetadataPrior(db_path=ebird_db, prior_mode="location_only")
        priors_no_date = mp.get_priors(
            ["Mourning Dove"],
            dt=None,
            latitude=40.7,
            longitude=-73.5,
        )
        priors_with_date = mp.get_priors(
            ["Mourning Dove"],
            dt=datetime(2024, 1, 1),
            latitude=40.7,
            longitude=-73.5,
        )
        # Both should give the same year-round average
        assert priors_no_date["Mourning Dove"] == pytest.approx(priors_with_date["Mourning Dove"])

    def test_location_only_annual_average(self, ebird_db):
        """Verify year-round average is different from period-specific seasonal value."""
        mp_location = MetadataPrior(db_path=ebird_db, prior_mode="location_only")
        mp_seasonal = MetadataPrior(db_path=ebird_db, prior_mode="seasonal")
        # Period 0 (Jan 1): Mourning Dove avg(Nassau, Suffolk) = avg(0.45, 0.50) = 0.475
        seasonal_priors = mp_seasonal.get_priors(
            ["Mourning Dove"],
            dt=datetime(2024, 1, 1),
            latitude=40.7,
            longitude=-73.5,
        )
        location_priors = mp_location.get_priors(
            ["Mourning Dove"],
            dt=None,
            latitude=40.7,
            longitude=-73.5,
        )
        # Seasonal at period 0 = 0.475; year-round avg = 0.525 (includes period 24)
        assert seasonal_priors["Mourning Dove"] == pytest.approx(0.475)
        assert location_priors["Mourning Dove"] == pytest.approx(0.525)

    def test_location_only_outside_region_returns_uniform(self, ebird_db):
        """Geographic gating still applies in location_only mode."""
        mp = MetadataPrior(db_path=ebird_db, prior_mode="location_only")
        priors = mp.get_priors(
            ["Mourning Dove"],
            dt=None,
            latitude=41.9,
            longitude=-87.6,  # Chicago
        )
        assert priors == {"Mourning Dove": 1.0}

    def test_location_only_zero_floor(self, ebird_db):
        """Zero-floor still applies in location_only mode."""
        mp = MetadataPrior(db_path=ebird_db, prior_mode="location_only", zero_floor=0.05)
        priors = mp.get_priors(
            ["Nonexistent Bird"],
            dt=None,
            latitude=40.7,
            longitude=-73.5,
        )
        assert priors["Nonexistent Bird"] == pytest.approx(0.05)

    def test_location_only_apply_reweights(self, ebird_db):
        """apply() in location_only mode reweights and normalizes predictions."""
        mp = MetadataPrior(db_path=ebird_db, prior_mode="location_only")
        predictions = [("Mourning Dove", 0.5), ("Blue Jay", 0.5)]
        result = mp.apply(
            predictions,
            dt=None,
            latitude=40.7,
            longitude=-73.5,
        )
        result_dict = dict(result)
        total = sum(result_dict.values())
        assert total == pytest.approx(1.0)
        # Mourning Dove has higher year-round frequency than Blue Jay
        assert result_dict["Mourning Dove"] > result_dict["Blue Jay"]

    def test_location_only_apply_sorted_descending(self, ebird_db):
        """apply() returns results sorted by probability descending."""
        mp = MetadataPrior(db_path=ebird_db, prior_mode="location_only")
        predictions = [("Blue Jay", 0.5), ("Mourning Dove", 0.5)]
        result = mp.apply(
            predictions,
            dt=None,
            latitude=40.7,
            longitude=-73.5,
        )
        assert result[0][1] >= result[1][1]

    def test_location_only_no_db_returns_uniform(self):
        """Without a DB, location_only still returns uniform priors."""
        mp = MetadataPrior(prior_mode="location_only")
        priors = mp.get_priors(["Robin", "Sparrow"], dt=None)
        assert priors == {"Robin": 1.0, "Sparrow": 1.0}


# ---------------------------------------------------------------------------
# Local priors overrides
# ---------------------------------------------------------------------------


class TestLocalPriorOverrides:
    """Tests for user-defined local prior overrides via local_priors.yaml."""

    def _make_local_priors_file(self, tmp_path, content: str) -> str:
        p = tmp_path / "local_priors.yaml"
        p.write_text(content)
        return str(p)

    def test_override_bypasses_zero_floor(self, tmp_path, ebird_db):
        """A suppressed species should use the override value, not zero_floor."""
        yaml_content = """
locations:
  - name: Test Feeder
    lat: 40.7
    lon: -73.5
    radius_km: 1.0
    species:
      Atlantic Puffin: 0.001
"""
        lp = self._make_local_priors_file(tmp_path, yaml_content)
        mp = MetadataPrior(
            db_path=ebird_db,
            zero_floor=0.01,
            local_priors_file=lp,
        )
        priors = mp.get_priors(
            ["Atlantic Puffin", "Mourning Dove"],
            dt=datetime(2024, 1, 1),  # period 0 — present in fixture
            latitude=40.7,
            longitude=-73.5,
        )
        # Override wins — 0.001, not floored to 0.01
        assert priors["Atlantic Puffin"] == pytest.approx(0.001)
        # Non-overridden species still gets eBird value (with zero_floor)
        assert priors["Mourning Dove"] > 0.01

    def test_non_overridden_species_gets_ebird_value(self, tmp_path, ebird_db):
        """Species absent from the local override still use eBird + zero_floor."""
        yaml_content = """
locations:
  - name: Test Feeder
    lat: 40.7
    lon: -73.5
    radius_km: 1.0
    species:
      House Sparrow: 0.9
"""
        lp = self._make_local_priors_file(tmp_path, yaml_content)
        mp = MetadataPrior(db_path=ebird_db, zero_floor=0.01, local_priors_file=lp)
        priors = mp.get_priors(
            ["House Sparrow", "Mourning Dove"],
            dt=datetime(2024, 1, 1),
            latitude=40.7,
            longitude=-73.5,
        )
        assert priors["House Sparrow"] == pytest.approx(0.9)
        # Mourning Dove not overridden — comes from eBird
        assert priors["Mourning Dove"] == pytest.approx(0.475)

    def test_outside_radius_no_override(self, tmp_path, ebird_db):
        """GPS outside radius_km does not activate the local override."""
        yaml_content = """
locations:
  - name: Test Feeder
    lat: 40.7
    lon: -73.5
    radius_km: 0.1
    species:
      Atlantic Puffin: 0.001
"""
        lp = self._make_local_priors_file(tmp_path, yaml_content)
        mp = MetadataPrior(db_path=ebird_db, zero_floor=0.01, local_priors_file=lp)
        priors = mp.get_priors(
            ["Atlantic Puffin"],
            dt=datetime(2024, 1, 1),
            latitude=40.8,  # ~15 km away
            longitude=-73.5,
        )
        # No override — zero_floor applies
        assert priors["Atlantic Puffin"] == pytest.approx(0.01)

    def test_no_gps_no_override(self, tmp_path, ebird_db):
        """When GPS is unavailable, local overrides are never applied."""
        yaml_content = """
locations:
  - name: Test Feeder
    lat: 40.7
    lon: -73.5
    radius_km: 100.0
    species:
      Atlantic Puffin: 0.001
"""
        lp = self._make_local_priors_file(tmp_path, yaml_content)
        mp = MetadataPrior(
            db_path=ebird_db, zero_floor=0.01, local_priors_file=lp, fips="US-NY-059"
        )
        priors = mp.get_priors(
            ["Atlantic Puffin"],
            dt=datetime(2024, 1, 1),
            # no lat/lon passed, no default lat/lon set
        )
        # Override needs GPS to activate; without GPS no override is applied
        assert priors["Atlantic Puffin"] == pytest.approx(0.01)
