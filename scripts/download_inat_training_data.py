"""Download iNaturalist research-grade bird photos for EfficientNet-S fine-tuning.

Creates an ImageFolder-compatible directory structure matching train_efficientnet.py:
    <output_dir>/
        american_robin/
            123456789.jpg
            ...
        dark_eyed_junco/
            ...

Uses the iNaturalist v1 API (no account required).  Rate-limited to stay
within iNaturalist's guidelines (~100 requests/min unauthenticated).

Typical usage:
    uv run scripts/download_inat_training_data.py \\
        --output-dir ./train_data \\
        --place-id 48 \\
        --max-per-species 500

The script is fully resumable — already-downloaded files are skipped.
"""

import argparse
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

INAT_API = "https://api.inaturalist.org/v1"
API_PER_PAGE = 200
API_DELAY = 0.9        # seconds between API calls → ~67 req/min, safely under limit
PHOTO_SIZE = "medium"  # 500px square — plenty for 224×224 training
DOWNLOAD_WORKERS = 8
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 30.0


def normalize_species_name(name: str) -> str:
    """'American Robin' → 'american_robin'  (matches train_efficientnet.py)."""
    name = name.strip().lower()
    name = re.sub(r"['\u2019]", "", name)
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name


def load_species_list(path: Path) -> list[str]:
    species = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            species.append(line)
    return species


def photo_url(raw_url: str, size: str = PHOTO_SIZE) -> str:
    """Replace iNaturalist thumbnail size token in URL.
    e.g. .../photos/123/square.jpg → .../photos/123/medium.jpg
    """
    return re.sub(r"/(square|thumb|small|medium|large|original)\.", f"/{size}.", raw_url)


def fetch_observation_photos(
    client: httpx.Client,
    taxon_name: str,
    place_id: int,
    max_photos: int,
) -> list[tuple[str, str]]:
    """Return list of (observation_id, photo_url) for a species.

    Paginates until max_photos reached or results exhausted.
    """
    results: list[tuple[str, str]] = []
    page = 1

    while len(results) < max_photos:
        params = {
            "taxon_name": taxon_name,
            "quality_grade": "research",
            "captive": "false",
            "place_id": str(place_id),
            "has[]": "photos",
            "iconic_taxon_name": "Aves",
            "per_page": str(min(API_PER_PAGE, max_photos - len(results))),
            "page": str(page),
            "order": "desc",
            "order_by": "created_at",
        }
        try:
            resp = client.get(f"{INAT_API}/observations", params=params, timeout=CONNECT_TIMEOUT)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("API error for %s page %d: %s", taxon_name, page, exc)
            break

        data = resp.json()
        observations = data.get("results", [])
        if not observations:
            break

        for obs in observations:
            photos = obs.get("photos", [])
            if photos:
                obs_id = str(obs["id"])
                url = photo_url(photos[0]["url"])
                results.append((obs_id, url))

        total = data.get("total_results", 0)
        logger.debug("  %s page %d: %d/%d observations fetched so far (total=%d)",
                     taxon_name, page, len(results), max_photos, total)

        if len(observations) < API_PER_PAGE or len(results) >= total:
            break

        page += 1
        time.sleep(API_DELAY)

    return results


def download_photo(client: httpx.Client, url: str, dest: Path) -> bool:
    """Download a single photo to dest. Returns True on success."""
    try:
        resp = client.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), follow_redirects=True)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except httpx.HTTPError as exc:
        logger.warning("Failed to download %s: %s", url, exc)
        return False


def download_species(
    species_name: str,
    output_dir: Path,
    place_id: int,
    max_photos: int,
    api_client: httpx.Client,
    dl_client: httpx.Client,
) -> tuple[int, int]:
    """Download photos for one species. Returns (downloaded, skipped)."""
    folder = output_dir / normalize_species_name(species_name)
    folder.mkdir(parents=True, exist_ok=True)

    photo_list = fetch_observation_photos(api_client, species_name, place_id, max_photos)
    time.sleep(API_DELAY)  # rate limit after last API call for this species

    to_download: list[tuple[Path, str]] = []
    skipped = 0
    for obs_id, url in photo_list:
        dest = folder / f"{obs_id}.jpg"
        if dest.exists():
            skipped += 1
        else:
            to_download.append((dest, url))

    if not to_download:
        return 0, skipped

    downloaded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(download_photo, dl_client, url, dest): dest
                   for dest, url in to_download}
        for fut in as_completed(futures):
            if fut.result():
                downloaded += 1
            else:
                failed += 1

    if failed:
        logger.warning("  %s: %d download(s) failed", species_name, failed)

    return downloaded, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Download iNaturalist training images for BirdVision")
    parser.add_argument("--output-dir", type=Path, default=Path("train_data"),
                        help="Root directory for ImageFolder output (default: ./train_data)")
    parser.add_argument("--species-list", type=Path,
                        default=Path("data/species_lists/north_america_common.txt"))
    parser.add_argument("--place-id", type=int, default=48,
                        help="iNaturalist place ID (default: 48 = New York state)")
    parser.add_argument("--max-per-species", type=int, default=500,
                        help="Max photos to download per species (default: 500)")
    parser.add_argument("--species", nargs="+",
                        help="Download only these species (by common name); default: all")
    args = parser.parse_args()

    all_species = load_species_list(args.species_list)
    if args.species:
        species_set = {s.lower() for s in args.species}
        target_species = [s for s in all_species if s.lower() in species_set]
        unknown = species_set - {s.lower() for s in target_species}
        if unknown:
            logger.warning("Species not found in list: %s", unknown)
    else:
        target_species = all_species

    logger.info("place_id=%d  max_per_species=%d  species=%d  output=%s",
                args.place_id, args.max_per_species, len(target_species), args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "BirdVision-training-data-downloader/1.0"}
    total_dl = 0
    total_skip = 0

    with httpx.Client(headers=headers) as api_client, \
         httpx.Client(headers=headers) as dl_client:

        for i, species in enumerate(target_species, 1):
            logger.info("[%d/%d] %s", i, len(target_species), species)
            dl, skip = download_species(
                species, args.output_dir, args.place_id,
                args.max_per_species, api_client, dl_client,
            )
            total_dl += dl
            total_skip += skip
            logger.info("  → downloaded=%d  skipped=%d", dl, skip)

    logger.info("Done. Total downloaded=%d  skipped=%d", total_dl, total_skip)


if __name__ == "__main__":
    main()
