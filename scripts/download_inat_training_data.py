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
from log_utils import add_logging_args, configure_logging, estimate_remaining, format_duration

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


def existing_observation_ids(folder: Path) -> set[str]:
    """Return existing observation ids already present in a species folder."""
    return {path.stem for path in folder.iterdir() if path.is_file()}


def resolve_taxon(
    client: httpx.Client,
    species_name: str,
) -> tuple[int, str] | None:
    """Resolve a species name to a stable iNaturalist bird taxon id."""
    try:
        resp = client.get(
            f"{INAT_API}/taxa/autocomplete",
            params={"q": species_name, "locale": "en"},
            timeout=CONNECT_TIMEOUT,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Taxon lookup failed for %s: %s", species_name, exc)
        return None

    target = normalize_species_name(species_name)
    candidates: list[tuple[int, dict]] = []
    for taxon in resp.json().get("results", []):
        if taxon.get("iconic_taxon_name") != "Aves":
            continue
        if taxon.get("rank") not in {"species", "subspecies"}:
            continue

        score = 0
        for field, bonus in (
            ("preferred_common_name", 100),
            ("matched_term", 80),
            ("name", 60),
        ):
            value = taxon.get(field)
            if value and normalize_species_name(value) == target:
                score = max(score, bonus)
        if taxon.get("rank") == "species":
            score += 10
        if taxon.get("is_active"):
            score += 5
        if score:
            candidates.append((score, taxon))

    if not candidates:
        logger.warning("No matching bird taxon found for %s", species_name)
        return None

    candidates.sort(
        key=lambda item: (item[0], item[1].get("observations_count", 0)),
        reverse=True,
    )
    best = candidates[0][1]
    display_name = best.get("preferred_common_name") or best.get("name") or species_name
    return int(best["id"]), display_name


def fetch_observation_photos(
    client: httpx.Client,
    taxon_id: int,
    taxon_name: str,
    place_id: int,
    target_total: int,
    existing_ids: set[str],
) -> list[tuple[str, str]]:
    """Return unseen (observation_id, photo_url) pairs up to target_total."""
    results: list[tuple[str, str]] = []
    seen_ids = set(existing_ids)
    page = 1
    needed = max(0, target_total - len(existing_ids))

    if needed == 0:
        return results

    while len(results) < needed:
        params = {
            "taxon_id": str(taxon_id),
            "quality_grade": "research",
            "captive": "false",
            "place_id": str(place_id),
            "has[]": "photos",
            "iconic_taxon_name": "Aves",
            "per_page": str(API_PER_PAGE),
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
                if obs_id in seen_ids:
                    continue
                url = photo_url(photos[0]["url"])
                results.append((obs_id, url))
                seen_ids.add(obs_id)
                if len(results) >= needed:
                    break

        total = data.get("total_results", 0)
        logger.debug("  %s page %d: %d/%d unseen observations fetched so far (total=%d)",
                     taxon_name, page, len(results), needed, total)

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
    existing_ids = existing_observation_ids(folder)
    existing_count = len(existing_ids)

    if existing_count >= max_photos:
        logger.info("  already at target: %d/%d", existing_count, max_photos)
        return 0, existing_count

    resolved = resolve_taxon(api_client, species_name)
    time.sleep(API_DELAY)
    if resolved is None:
        return 0, existing_count

    taxon_id, resolved_name = resolved
    logger.info("  resolved to iNat taxon %s (%s)", taxon_id, resolved_name)
    photo_list = fetch_observation_photos(
        api_client,
        taxon_id,
        species_name,
        place_id,
        max_photos,
        existing_ids,
    )
    time.sleep(API_DELAY)  # rate limit after last API call for this species

    to_download: list[tuple[Path, str]] = []
    for obs_id, url in photo_list:
        dest = folder / f"{obs_id}.jpg"
        to_download.append((dest, url))

    if not to_download:
        logger.info("  no new observations available; current=%d target=%d", existing_count, max_photos)
        return 0, existing_count

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

    return downloaded, existing_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Download iNaturalist training images for BirdVision")
    parser.add_argument("--output-dir", type=Path, default=Path("train_data"),
                        help="Root directory for ImageFolder output (default: ./train_data)")
    parser.add_argument("--species-list", type=Path,
                        default=Path("data/species_lists/north_america_common.txt"))
    parser.add_argument("--place-id", type=int, default=48,
                        help="iNaturalist place ID (default: 48 = New York state)")
    parser.add_argument("--max-per-species", type=int, default=500,
                        help="Target final file count per species (default: 500)")
    parser.add_argument("--species", nargs="+",
                        help="Download only these species (by common name); default: all")
    add_logging_args(parser)
    args = parser.parse_args()

    log_path = configure_logging(
        "download_inat_training_data",
        log_file=args.log_file,
        log_dir=args.log_dir,
    )
    if log_path:
        logger.info("File logging enabled: %s", log_path)

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
    overall_start = time.time()

    with httpx.Client(headers=headers) as api_client, \
         httpx.Client(headers=headers) as dl_client:

        for i, species in enumerate(target_species, 1):
            logger.info("[%d/%d] %s", i, len(target_species), species)
            dl, existing = download_species(
                species, args.output_dir, args.place_id,
                args.max_per_species, api_client, dl_client,
            )
            total_dl += dl
            total_skip += existing
            elapsed = time.time() - overall_start
            eta = estimate_remaining(elapsed, i, len(target_species))
            logger.info(
                "  → existing=%d  downloaded=%d  final=%d  elapsed=%s  eta=%s",
                existing,
                dl,
                existing + dl,
                format_duration(elapsed),
                format_duration(eta),
            )

    logger.info("Done. Total downloaded=%d  existing_before_run=%d", total_dl, total_skip)


if __name__ == "__main__":
    main()
