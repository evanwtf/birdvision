"""
Extract date/time and GPS coordinates from video file metadata using ExifTool.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import exiftool

logger = logging.getLogger(__name__)

DATETIME_FORMATS = [
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
]


@dataclass
class VideoMetadata:
    recorded_at: Optional[datetime] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    @property
    def has_gps(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def osm_url(self) -> Optional[str]:
        if not self.has_gps:
            return None
        return (
            f"https://www.openstreetmap.org/?mlat={self.latitude:.6f}"
            f"&mlon={self.longitude:.6f}#map=15/{self.latitude:.6f}/{self.longitude:.6f}"
        )


def extract(video_path: str) -> VideoMetadata:
    meta = VideoMetadata()
    try:
        with exiftool.ExifToolHelper() as et:
            tags = et.get_metadata(video_path)[0]

        # Date — prefer QuickTime CreateDate, fall back to other fields
        for key in ("QuickTime:CreateDate", "EXIF:DateTimeOriginal", "File:FileModifyDate"):
            raw = tags.get(key)
            if raw:
                for fmt in DATETIME_FORMATS:
                    try:
                        meta.recorded_at = datetime.strptime(raw[:19], fmt)
                        break
                    except ValueError:
                        continue
            if meta.recorded_at:
                break

        # GPS — ExifTool exposes composite Latitude/Longitude in decimal degrees
        lat = tags.get("Composite:GPSLatitude") or tags.get("EXIF:GPSLatitude")
        lon = tags.get("Composite:GPSLongitude") or tags.get("EXIF:GPSLongitude")
        if lat is not None and lon is not None:
            meta.latitude = float(lat)
            meta.longitude = float(lon)

    except Exception as e:
        logger.warning(f"Could not extract metadata from {Path(video_path).name}: {e}")

    return meta
