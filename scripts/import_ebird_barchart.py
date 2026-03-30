#!/usr/bin/env python3
"""
Import eBird bar chart TSV files into a SQLite database.

Usage:
    uv run scripts/import_ebird_barchart.py ebird_data/ --db data/ebird_priors.db
"""
import argparse
import logging
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("import_ebird")

# eBird sometimes uses regional prefixes not in common field-guide names.
# Map eBird name -> canonical name used in our species list.
NAME_OVERRIDES = {
    "American Herring Gull": "Herring Gull",
    "American Barn Owl":     "Barn Owl",
    "Mew Gull":              "Common Gull",
}

FIPS_COUNTY = {
    "US-NY-047": "Kings (Brooklyn)",
    "US-NY-059": "Nassau",
    "US-NY-061": "New York (Manhattan)",
    "US-NY-081": "Queens",
    "US-NY-103": "Suffolk",
}


def normalize_name(name: str) -> str:
    return NAME_OVERRIDES.get(name, name)


def parse_fips(filename: str) -> tuple[str, str]:
    m = re.search(r"(US-[A-Z]{2}-\d+)", filename)
    fips = m.group(1) if m else "unknown"
    return fips, FIPS_COUNTY.get(fips, fips)


def parse_barchart(path: Path) -> list[tuple[str, int, float]]:
    """
    Returns list of (species, period_index, frequency).
    period_index: 0-47 (4 periods per month, 0=early Jan … 47=late Dec)
    frequency: 0.0-1.0 fraction of checklists reporting this species
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            species_raw = parts[0].strip()
            if not species_raw or species_raw in ("", "Sample Size:") or "Frequency" in species_raw:
                continue
            # Skip hybrid/slash species
            if "/" in species_raw or "(" in species_raw:
                continue
            try:
                freqs = [float(v) if v else 0.0 for v in parts[1:49]]
            except ValueError:
                continue
            if len(freqs) < 48:
                freqs += [0.0] * (48 - len(freqs))
            species = normalize_name(species_raw)
            for period, freq in enumerate(freqs):
                rows.append((species, period, freq))
    return rows


def build_db(barchart_dir: Path, db_path: Path):
    files = sorted(barchart_dir.glob("*barchart*.txt"))
    if not files:
        logger.error(f"No barchart files found in {barchart_dir}")
        sys.exit(1)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS barchart;
        DROP TABLE IF EXISTS counties;
        CREATE TABLE counties (
            fips    TEXT PRIMARY KEY,
            name    TEXT
        );
        CREATE TABLE barchart (
            fips    TEXT,
            species TEXT,
            period  INTEGER,   -- 0-47 (4 per month)
            freq    REAL,
            PRIMARY KEY (fips, species, period)
        );
        CREATE INDEX idx_barchart_species_period ON barchart (species, period);
    """)

    for f in files:
        fips, county_name = parse_fips(f.name)
        cur.execute("INSERT OR REPLACE INTO counties VALUES (?, ?)", (fips, county_name))
        rows = parse_barchart(f)
        cur.executemany(
            "INSERT OR REPLACE INTO barchart (fips, species, period, freq) VALUES (?, ?, ?, ?)",
            [(fips, species, period, freq) for species, period, freq in rows],
        )
        logger.info(f"  {county_name} ({fips}): {len(rows)} rows")

    con.commit()
    con.close()
    logger.info(f"Database written to {db_path}")


def main():
    parser = argparse.ArgumentParser(description="Import eBird bar chart data into SQLite")
    parser.add_argument("barchart_dir", help="Directory containing barchart .txt files")
    parser.add_argument("--db", default="data/ebird_priors.db", help="Output SQLite database path")
    args = parser.parse_args()

    build_db(Path(args.barchart_dir), Path(args.db))


if __name__ == "__main__":
    main()
