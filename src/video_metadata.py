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
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    focal_length: Optional[str] = None

    @property
    def has_gps(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def camera_info(self) -> Optional[str]:
        """Human-readable camera description, e.g. 'Apple iPhone 15 Pro, 77mm'."""
        parts = []
        if self.camera_make and self.camera_model:
            # Avoid doubling up "Apple Apple iPhone" etc.
            if self.camera_model.lower().startswith(self.camera_make.lower()):
                parts.append(self.camera_model)
            else:
                parts.append(f"{self.camera_make} {self.camera_model}")
        elif self.camera_model:
            parts.append(self.camera_model)
        if self.focal_length:
            parts.append(self.focal_length)
        return ", ".join(parts) if parts else None

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

        # Camera info
        meta.camera_make = tags.get("EXIF:Make") or tags.get("QuickTime:Make")
        meta.camera_model = tags.get("EXIF:Model") or tags.get("QuickTime:Model")
        fl = tags.get("EXIF:FocalLengthIn35mmFormat") or tags.get("EXIF:FocalLength")
        if fl is not None:
            # ExifTool may return a float (mm) or a string like "77 mm"
            try:
                meta.focal_length = f"{float(fl):.0f}mm"
            except (ValueError, TypeError):
                meta.focal_length = str(fl)

    except Exception as e:
        logger.warning(f"Could not extract metadata from {Path(video_path).name}: {e}")

    return meta
