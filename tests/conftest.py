"""Shared fixtures for BirdVision unit tests."""

import sqlite3
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_dir(tmp_path):
    """Provide a temporary directory (alias for tmp_path)."""
    return tmp_path


@pytest.fixture()
def ebird_db(tmp_path):
    """Create a tiny eBird SQLite database for metadata prior tests.

    Contains two counties (Nassau and Suffolk on Long Island) with a few
    species and frequency entries.
    """
    db_path = tmp_path / "ebird_priors.db"
    con = sqlite3.connect(str(db_path))
    con.execute("CREATE TABLE counties (fips TEXT PRIMARY KEY, name TEXT)")
    con.execute("INSERT INTO counties VALUES ('US-NY-059', 'Nassau')")
    con.execute("INSERT INTO counties VALUES ('US-NY-103', 'Suffolk')")
    con.execute(
        "CREATE TABLE barchart ("
        "  fips TEXT, species TEXT, period INTEGER, freq REAL,"
        "  PRIMARY KEY (fips, species, period)"
        ")"
    )
    # Period 0 = early January
    rows = [
        ("US-NY-059", "Mourning Dove", 0, 0.45),
        ("US-NY-103", "Mourning Dove", 0, 0.50),
        ("US-NY-059", "Blue Jay", 0, 0.30),
        ("US-NY-103", "Blue Jay", 0, 0.25),
        ("US-NY-059", "Bald Eagle", 0, 0.02),
        ("US-NY-103", "Bald Eagle", 0, 0.01),
        # Period 24 = mid-year (roughly July)
        ("US-NY-059", "Mourning Dove", 24, 0.60),
        ("US-NY-103", "Mourning Dove", 24, 0.55),
    ]
    con.executemany("INSERT INTO barchart VALUES (?, ?, ?, ?)", rows)
    con.commit()
    con.close()
    return str(db_path)
